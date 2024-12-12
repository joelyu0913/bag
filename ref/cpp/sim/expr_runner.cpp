#include "yang/sim/expr_runner.h"

#include "yang/expr/expr.h"
#include "yang/expr/mat_data_source.h"
#include "yang/math/mat_ops.h"
#include "yang/util/unordered_map.h"
#include "yang/util/unordered_set.h"

namespace yang {

ExprRunner::ExprRunner(const Env *env) : env_(env) {
  AddBaseData(default_base_data());
  AddGroups(default_groups());
}

ExprRunner::ExprRunner(const Env *env, const std::vector<std::string> &base_data,
                       const std::vector<std::string> &groups,
                       const std::vector<DataInfo> &extra_data)
    : env_(env) {
  AddBaseData(base_data);
  AddGroups(groups);
  AddExtraData(extra_data);
}

void ExprRunner::AddGroups(const std::vector<std::string> &groups) {
  for (auto &g : groups) groups_.insert(g);
}

void ExprRunner::AddBaseData(const std::vector<std::string> &base_data) {
  for (auto &name : base_data) {
    if (name == "sharesout" || name == "cap" || name == "cumadj" || name == "sharesfloat") {
      data_[name] = {"base/" + name, "base/" + name};
    } else {
      data_[name] = {"ibase/i_" + name, "base/" + name};
    }
  }
}

void ExprRunner::AddExtraData(const std::vector<DataInfo> &extra_data) {
  for (auto &[name, live, eod] : extra_data) {
    data_[name] = {live, eod};
  }
}

void ExprRunner::Run(std::string_view expr_str, expr::OutFloatMat output, Mode mode) {
  Run(expr_str, "", output, mode);
}

void ExprRunner::Run(std::string_view expr_str, std::string_view univ, expr::OutFloatMat output,
                     Mode mode) {
  Run(expr_str, univ, output, mode, env_->dates_size(), env_->start_di(), env_->end_di());
}

void ExprRunner::Run(std::string_view expr_str, std::string_view univ, expr::OutFloatMat output,
                     Mode mode, int dates_size, int start_di, int end_di) {
  int univ_size = env_->univ_size();

  expr::MatDataSource data_src(dates_size, univ_size);
  expr::Expr expr(expr_str, &data_src);

  // fast path
  if (expr.dag().depth == 0) {
    RunSimple(expr.dag().value, univ, output, mode, start_di, end_di);
    return;
  }

  // same as intraday mode
  if (mode == Mode::MIXED && expr.full_hist_len() == 1) mode = Mode::INTRADAY;

  auto add_group = [&](auto &name, auto &array) {
    ENSURE2(array.shape(0) >= dates_size && array.shape(1) == env_->max_univ_size());
    data_src.AddGroup(name, array.mat_view().block(0, 0, dates_size, univ_size));
  };
  for (auto &group : expr.CollectGroups()) {
    if (groups_.count(group)) {
      add_group(group, *env_->ReadData<Array<int>>("base", group));
    } else {
      add_group(group, *env_->ReadData<Array<int>>(group));
    }
  }

  auto add_data = [&](auto &name, auto &array, auto &eod_array) {
    ENSURE2(array.shape(0) >= dates_size && array.shape(1) == env_->max_univ_size());
    data_src.AddData(name, array.mat_view().block(0, 0, dates_size, univ_size),
                     eod_array.mat_view().block(0, 0, dates_size, univ_size));
  };
  for (auto &name : expr.CollectRawData()) {
    auto it = data_.find(name);
    std::string_view live_path;
    std::string_view eod_path;
    if (it == data_.end()) {
      live_path = name;
      eod_path = name;
    } else {
      live_path = it->second.first;
      eod_path = it->second.second;
    }
    if (mode == Mode::EOD) {
      live_path = eod_path;
    }
    auto item_type = ArrayBase::GetItemType(env_->cache_dir().GetPath(live_path));
    if (item_type == "double") {
      add_data(name, *env_->ReadData<Array<double>>(live_path),
               *env_->ReadData<Array<double>>(eod_path));
    } else if (item_type == "float") {
      add_data(name, *env_->ReadData<Array<float>>(live_path),
               *env_->ReadData<Array<float>>(eod_path));
    } else if (item_type == "int32") {
      add_data(name, *env_->ReadData<Array<int32_t>>(live_path),
               *env_->ReadData<Array<int32_t>>(eod_path));
    } else {
      LOG_FATAL("unsupported extra_data type {}", item_type);
    }
  }

  if (!univ.empty()) {
    auto &univ_array = *env_->ReadData<Array<bool>>(univ);
    data_src.set_mask(univ_array.mat_view().block(0, 0, dates_size, univ_size));
  }

  int hist_start_di = std::max(start_di - expr.full_hist_len() + 1, 0);
  LOG_INFO("Running expr: {} ({} - {}) (univ: {})", expr_str, hist_start_di, end_di, univ);

  data_src.set_live(mode == Mode::INTRADAY);
  for (int di = hist_start_di; di < start_di; ++di) {
    data_src.set_cur_row(di);
    expr.EvalPreload();
  }

  data_src.set_live(mode != Mode::EOD);
  for (int di = start_di; di < end_di; ++di) {
    if (mode == Mode::MIXED && !expr.results().empty()) {
      data_src.set_live(false);
      expr.Eval(true);
      data_src.set_live(true);
    }
    data_src.set_cur_row(di);
    auto result = expr.Eval();
    output.SaveRow(result, di);
  }
}

void ExprRunner::RunSimple(std::string_view name, std::string_view univ, expr::OutFloatMat output,
                           Mode mode, int start_di, int end_di) {
  std::string_view data_path;
  auto data_it = data_.find(name);
  if (data_it == data_.end()) {
    data_path = name;
  } else {
    data_path = mode == Mode::EOD ? data_it->second.second : data_it->second.first;
  }

  auto item_type = ArrayBase::GetItemType(env_->cache_dir().GetPath(data_path));
  expr::FloatMat input(yang::math::MatView<const float>{});
  if (item_type == "double") {
    input = expr::FloatMat(env_->ReadData<Array<double>>(data_path)->mat_view());
  } else if (item_type == "float") {
    input = expr::FloatMat(env_->ReadData<Array<float>>(data_path)->mat_view());
  } else if (item_type == "int32") {
    input = expr::FloatMat(env_->ReadData<Array<int32_t>>(data_path)->mat_view());
  } else {
    LOG_FATAL("unsupported data type {}", item_type);
  }

  int univ_size = env_->univ_size();
  LOG_INFO("Running expr: {} ({} - {}) (univ: {}) (fast path)", name, start_di, end_di, univ);
  auto copy_mat = [&](auto in_mat, auto out_mat) {
    out_mat = out_mat.slice(start_di, end_di, 0, univ_size);
    in_mat = in_mat.slice(start_di, end_di, 0, univ_size);
    out_mat.copy_from(in_mat);
    if (!univ.empty()) {
      auto &univ_array = *env_->ReadData<Array<bool>>(univ);
      yang::math::ops::filter(out_mat, univ_array.mat_view().slice(start_di, end_di, 0, univ_size));
    }
  };
  if (input.type_index() == output.type_index()) {
    if (input.type_index() == 0) {
      copy_mat(input.raw<float>(), output.raw<float>());
    } else {
      copy_mat(input.raw<double>(), output.raw<double>());
    }
  } else if (input.type_index() == 0) {
    copy_mat(input.raw<float>(), output.raw<double>());
  } else if (input.type_index() == 1) {
    copy_mat(input.raw<double>(), output.raw<float>());
  } else {
    if (output.type_index() == 0) {
      copy_mat(input.raw<int32_t>(), output.raw<float>());
    } else {
      copy_mat(input.raw<int32_t>(), output.raw<double>());
    }
  }
}

}  // namespace yang
