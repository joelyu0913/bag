#include "base_daily.h"

#include "yang/data/block_mat.h"
#include "yang/math/eigen.h"
#include "yang/util/strings.h"
#include "yao/operation_manager.h"

namespace yao {

void BaseDaily::RunImpl() {
  max_date_sz = env().max_dates_size();
  max_univ_sz = env().max_univ_size();
  date_sz = env().dates_size();
  univ_sz = env().univ_size();

  start_di = env().start_di();
  end_di = env().end_di();

  user_proc();
  ApplyOps();
}

void BaseDaily::LoadBaseData() {
  std::vector<std::pair<std::string, std::string>> vb_daily = {
      {"univ_all", "bool"},  {"cty", "int"},          {"sector", "int"},
      {"industry", "int"},   {"subindustry", "int"},  {"st", "int"},
      {"open", "float"},     {"close", "float"},      {"high", "float"},
      {"low", "float"},      {"adj", "float"},        {"cumadj", "float"},
      {"ret", "float"},      {"vol", "float"},        {"dvol", "float"},
      {"vwap", "float"},     {"sharesout", "float"},  {"cap", "float"},
      {"limit_up", "float"}, {"limit_down", "float"}, {"halt", "bool"},
      {"adj_open", "float"}, {"adj_close", "float"},  {"adj_high", "float"},
      {"adj_low", "float"},  {"adj_vol", "float"},    {"adj_vwap", "float"},
  };
  for (auto &it : vb_daily) {
    auto &data = it.first;
    auto &type = it.second;
    if (type == "bool") {
      b_b[data] = ReadArray<bool>("base", data);
    } else if (type == "int") {
      b_i[data] = ReadArray<int>("base", data);
    } else if (type == "float") {
      b_f[data] = ReadArray<float>("base", data);
    }
  }
}

void BaseDaily::LoadIBaseData() {
  std::vector<std::pair<std::string, std::string>> vb_i = {
      {"i_open", "float"},      {"i_close", "float"},    {"i_high", "float"},
      {"i_low", "float"},       {"i_ret", "float"},      {"i_vol", "float"},
      {"i_dvol", "float"},      {"i_vwap", "float"},     {"i_adj_open", "float"},
      {"i_adj_close", "float"}, {"i_adj_high", "float"}, {"i_adj_low", "float"},
      {"i_adj_vol", "float"},   {"i_adj_vwap", "float"},
  };
  for (auto &it : vb_i) {
    auto &data = it.first;
    b_f[data] = ReadArray<float>("ibase", data);
  }
}

void BaseDaily::base_load(const std::vector<std::string> &vv) {
  for (const std::string &load_str : vv) {
    std::vector<std::string_view> vec_load = yang::StrSplit(load_str, '|');
    std::string_view name_load = vec_load[0];

    // already loaded
    if (!set_loaded.emplace(name_load).second) return;

    if (name_load == "univ") {
      std::string univ_choice = config<std::string>("__univ");
      if (univ_choice == "all") {
        b_univ = *ReadArray<bool>("base", "univ_all");
      } else {
        b_univ = *ReadArray<bool>("sup_univ", univ_choice);
      }
      ENSURE2(b_univ.shape(0) >= env().dates_size() && b_univ.shape(1) == env().max_univ_size());
    } else if (name_load == "sig") {
      b_sig = WriteArray<float>("b_sig");
    } else {
      LOG_FATAL("Not valid loading item: {}", name_load);
    }
  }
}

void BaseDaily::ApplyOps() {
  auto ops = config()["ops"];
  if (!ops) return;

  ENSURE(b_sig.ndim() > 0, "b_sig missing, cannot apply operations");
  auto b_sig_op = WriteArray<float>("b_sig_op");
  OperationManager::Apply(&env(), b_sig.mat_view(), b_sig_op.mat_view(), start_di, end_di, ops);
}

}  // namespace yao
