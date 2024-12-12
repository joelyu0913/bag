#include "yang/expr/ts_functions.h"

#include "yang/expr/ops.h"
#include "yang/math/mat_view.h"
#include "yang/math/regression.h"

namespace yang::expr {

namespace detail {

static int FillInvalid(math::VecView<Float> range, int ts_len) {
  if (TS_FILL_LEN > 0) {
    ops::linear_fill(range.begin(), range.end(), TS_FILL_LEN);
  }
  auto len = std::min<int>(ts_len, range.size());
  for (int i = 0; i < len; ++i) {
    if (!IsValid(range[range.size() - 1 - i])) {
      return i;
    }
  }
  return len;
}

template <class Func>
void ApplyTsRange(bool fill_invalid, const FunctionArgs &args, Func &&func, int ts_len = 0) {
  if (ts_len == 0) ts_len = args.scalar_args[0];
  auto hist = args.inputs[0];
  ENSURE2(!hist.empty());

  auto &output = args.output;
  int size = args.output.size();
  std::vector<Float> range;
  range.reserve(hist.size());
  for (int i = 0; i < size; ++i) {
    if (!IsValid(hist.back()[i])) {
      output[i] = NAN;
      continue;
    }

    range.clear();
    for (int j = 0; j < hist.size(); ++j) {
      range.push_back(hist[j][i]);
    }
    int len;
    if (fill_invalid) {
      len = FillInvalid(range, ts_len);
    } else {
      len = std::min<int>(ts_len, range.size());
    }
    OutVecView<Float> view(range.data() + range.size() - len, len);
    output[i] = func(view);
  }
}

template <class Func>
void ApplyTsBinary(bool fill_invalid, const FunctionArgs &args, Func &&func) {
  int ts_len = args.scalar_args[0];
  auto x_hist = args.inputs[0];
  auto y_hist = args.inputs[1];
  ENSURE2(!x_hist.empty());
  ENSURE2(x_hist.size() == y_hist.size());

  auto &output = args.output;
  int size = args.output.size();
  std::vector<Float> x;
  std::vector<Float> y;
  x.reserve(x_hist.size());
  y.reserve(x_hist.size());
  for (int i = 0; i < size; ++i) {
    if (!IsValid(x_hist.back()[i]) || !IsValid(y_hist.back()[i])) {
      output[i] = NAN;
      continue;
    }
    x.clear();
    y.clear();
    for (int j = 0; j < x_hist.size(); ++j) {
      x.push_back(x_hist[j][i]);
      y.push_back(y_hist[j][i]);
    }
    int len;
    if (fill_invalid) {
      len = std::min(FillInvalid(x, ts_len), FillInvalid(y, ts_len));
    } else {
      len = std::min<int>(ts_len, x.size());
    }
    OutVecView<Float> x_view(x.data() + x.size() - len, len);
    OutVecView<Float> y_view(y.data() + y.size() - len, len);
    output[i] = func(x_view, y_view);
  }
}

}  // namespace detail

int TsFunction::ComputeTsLenInternal(VecView<Float> scalar_args) const {
  // float can store integer values up to 2 ^ 24 - 1 without precision loss
  int len = 1;
  for (int i = 0; i < num_ts_args(); ++i) {
    len = std::max(len, static_cast<int>(scalar_args[i]));
  }
  return len;
}

void TsMin::ApplyImpl(const FunctionArgs &args) const {
  detail::ApplyTsRange(fill_invalid(), args,
                       [](auto &range) { return ops::min(range.begin(), range.end()); });
}

void TsMax::ApplyImpl(const FunctionArgs &args) const {
  detail::ApplyTsRange(fill_invalid(), args,
                       [](auto &range) { return ops::max(range.begin(), range.end()); });
}

void TsSum::ApplyImpl(const FunctionArgs &args) const {
  detail::ApplyTsRange(fill_invalid(), args,
                       [](auto &range) { return ops::sum(range.begin(), range.end()); });
}

void TsCountValid::ApplyImpl(const FunctionArgs &args) const {
  detail::ApplyTsRange(fill_invalid(), args, [](auto &range) {
    return ops::count<ops::CheckFinite>(range.begin(), range.end());
  });
}

void TsCountBool::ApplyImpl(const FunctionArgs &args) const {
  detail::ApplyTsRange(fill_invalid(), args, [](auto &range) {
    return ops::count<ops::CheckBool>(range.begin(), range.end());
  });
}

void TsProd::ApplyImpl(const FunctionArgs &args) const {
  detail::ApplyTsRange(fill_invalid(), args,
                       [](auto &range) { return ops::prod(range.begin(), range.end()); });
}

void TsMean::ApplyImpl(const FunctionArgs &args) const {
  detail::ApplyTsRange(fill_invalid(), args,
                       [](auto &range) { return ops::mean(range.begin(), range.end()); });
}

void TsRMean::ApplyImpl(const FunctionArgs &args) const {
  detail::ApplyTsRange(fill_invalid(), args,
                       [&](auto &range) { return ops::rmean(range.begin(), range.end()); });
}

void TsStd::ApplyImpl(const FunctionArgs &args) const {
  detail::ApplyTsRange(fill_invalid(), args,
                       [](auto &range) { return ops::stdev(range.begin(), range.end()); });
}

void TsMeanDev::ApplyImpl(const FunctionArgs &args) const {
  detail::ApplyTsRange(fill_invalid(), args,
                       [](auto &range) { return ops::mean_dev(range.begin(), range.end()); });
}

void TsZScore::ApplyImpl(const FunctionArgs &args) const {
  detail::ApplyTsRange(fill_invalid(), args, [](auto &range) {
    ops::zscore(range.begin(), range.end());
    return range[range.size() - 1];
  });
}

void TsDemean::ApplyImpl(const FunctionArgs &args) const {
  detail::ApplyTsRange(fill_invalid(), args, [](auto &range) {
    return range[range.size() - 1] - ops::mean(range.begin(), range.end());
  });
}

void TsCenterExp::ApplyImpl(const FunctionArgs &args) const {
  auto exp = args.scalar_args[1];
  detail::ApplyTsRange(fill_invalid(), args, [exp](auto &range) {
    Float sum = 0;
    ops::demean(range.begin(), range.end());
    for (auto &v : range) sum += ops::detail::pow(v, exp);
    return sum;
  });
}

void TsCorrStep::ApplyImpl(const FunctionArgs &args) const {
  detail::ApplyTsRange(fill_invalid(), args, [](auto &range) {
    return ops::corr_step<double>(range.begin(), range.end());
  });
}

void TsCorr::ApplyImpl(const FunctionArgs &args) const {
  detail::ApplyTsBinary(fill_invalid(), args, [](auto &x, auto &y) {
    return ops::corr<double>(x.begin(), x.end(), y.begin());
  });
}

void TsSumIf::ApplyImpl(const FunctionArgs &args) const {
  detail::ApplyTsBinary(fill_invalid(), args, [](auto &x, auto &y) {
    return ops::sum_masked(x.begin(), x.end(), y.begin());
  });
}

void TsAutoCorr::ApplyImpl(const FunctionArgs &args) const {
  int lag = args.scalar_args[1];
  detail::ApplyTsRange(
      fill_invalid(), args,
      [&](auto &range) { return ops::auto_corr<double>(range.begin(), range.end(), lag); },
      args.scalar_args[0] + lag);
}

void TsDecayLinear::ApplyImpl(const FunctionArgs &args) const {
  detail::ApplyTsRange(fill_invalid(), args,
                       [](auto &range) -> Float { return ops::decay_linear(range); });
}

void TsArgmax::ApplyImpl(const FunctionArgs &args) const {
  detail::ApplyTsRange(fill_invalid(), args, [](auto &range) -> Float {
    auto ret = ops::argmax(range);
    return ret == -1 ? NAN : ret;
  });
}

void TsArgmin::ApplyImpl(const FunctionArgs &args) const {
  detail::ApplyTsRange(fill_invalid(), args, [](auto &range) -> Float {
    auto ret = ops::argmin(range);
    return ret == -1 ? NAN : ret;
  });
}

void TsRank::ApplyImpl(const FunctionArgs &args) const {
  detail::ApplyTsRange(fill_invalid(), args, [](auto &range) {
    ops::rank(range.begin(), range.end());
    return range.back();
  });
}

void TsCov::ApplyImpl(const FunctionArgs &args) const {
  detail::ApplyTsBinary(fill_invalid(), args, [](auto &x, auto &y) {
    return ops::cov<double>(x.begin(), x.end(), y.begin());
  });
}

void TsEma::ApplyImpl(const FunctionArgs &args) const {
  auto ratio = args.scalar_args[1];
  ApplyRatio(args, ratio);
}

void TsEma::ApplyRatio(const FunctionArgs &args, Float ratio) const {
  detail::ApplyTsRange(fill_invalid(), args, [ratio](auto &range) {
    return ops::ema<Float>(range.begin(), range.end(), ratio);
  });
}

void TsSma::ApplyImpl(const FunctionArgs &args) const {
  auto n = args.scalar_args[1];
  auto m = args.scalar_args[2];
  this->ApplyRatio(args, m / n);
}

void TsWma::ApplyImpl(const FunctionArgs &args) const {
  detail::ApplyTsRange(fill_invalid(), args, [](auto &range) {
    Float acc = NAN;
    for (auto &x : range) {
      if (IsValid(acc) && IsValid(x)) {
        acc = x + acc * 0.9;
      } else {
        acc = x;
      }
    }
    return acc;
  });
}

void TsDelay::ApplyImpl(const FunctionArgs &args) const {
  int delay = args.scalar_args[0];
  detail::ApplyTsRange(
      fill_invalid(), args, [](auto &range) { return range.front(); }, delay + 1);
}

void TsDelta::ApplyImpl(const FunctionArgs &args) const {
  int lag = args.scalar_args[0];
  detail::ApplyTsRange(
      fill_invalid(), args, [](auto &range) { return range.back() - range.front(); }, lag + 1);
}

void TsDelayRange::ApplyImpl(const FunctionArgs &args) const {
  detail::ApplyTsRange(fill_invalid(), args, [](auto &range) { return range.front(); });
}

void TsDeltaRange::ApplyImpl(const FunctionArgs &args) const {
  detail::ApplyTsRange(fill_invalid(), args,
                       [](auto &range) { return range.back() - range.front(); });
}

void TsSumMaxPct::ApplyImpl(const FunctionArgs &args) const {
  auto pct = args.scalar_args[1];
  std::vector<int> indexes;
  indexes.reserve(args.scalar_args[0]);
  detail::ApplyTsBinary(fill_invalid(), args, [&](auto &xs, auto &ys) {
    indexes.clear();
    for (int i = 0; i < xs.size(); ++i) indexes.push_back(i);
    int n = std::ceil(xs.size() * pct);
    std::nth_element(indexes.begin(), indexes.begin() + n, indexes.end(),
                     [&](auto i, auto j) { return xs[i] > xs[j]; });
    Float sum = 0;
    for (int i = 0; i < n; ++i) sum += ys[indexes[i]];
    return sum;
  });
}

void TsReg::ApplyImpl(const FunctionArgs &args) const {
  int ts_len = args.scalar_args[0];
  int n = args.scalar_args[1];
  auto y_hist = args.inputs[0];
  ENSURE(args.inputs.size() >= 3, "TsReg should have at least 3 inputs");

  int num_xs = args.inputs.size() - 1;
  ENSURE2(n >= 0 && n <= num_xs);
  if (y_hist.size() < num_xs) return;

  ENSURE2(!y_hist.empty());
  for (int k = 0; k < num_xs; ++k) ENSURE2(args.inputs[k + 1].size() == y_hist.size());

  auto &output = args.output;
  int size = args.output.size();
  std::vector<Float> y(y_hist.size());
  std::vector<Float> xs(y_hist.size() * num_xs);
  math::MatView<Float> xs_mat(xs.data(), num_xs, y_hist.size());

  auto validate_row = [&](int row, int i) {
    if (!IsValid(y_hist[row][i])) return false;
    for (int k = 0; k < num_xs; ++k) {
      if (!IsValid(args.inputs[k + 1][row][i])) return false;
    }
    return true;
  };

  for (int i = 0; i < size; ++i) {
    if (!validate_row(y_hist.size() - 1, i)) continue;

    int len = 0;
    for (int ti = 0; ti < ts_len; ++ti) {
      if (ti > static_cast<int>(y_hist.size()) - 1) break;
      int j = y_hist.size() - 1 - ti;
      if (!validate_row(j, i)) continue;
      y[len] = y_hist[j][i];
      for (int k = 0; k < num_xs; ++k) {
        xs_mat(k, len) = args.inputs[k + 1][j][i];
      }
      ++len;
    }
    if (len < num_xs) continue;
    arma::mat Y(len, 1);
    arma::mat X(len, num_xs);
    for (int j = 0; j < len; ++j) Y(j, 0) = y[j];
    for (int k = 0; k < num_xs; ++k) {
      for (int j = 0; j < len; ++j) {
        X(j, k) = xs_mat(k, j);
      }
    }
    auto reg = math::LinearRegression(X, Y);
    if (n > 0) {
      output[i] = reg(n - 1, 0);
    } else {
      output[i] = Y(0, 0) - arma::dot(reg.col(0), X.row(0));
    }
  }
}

void CsReg::ApplyImpl(const FunctionArgs &args) const {
  int ts_len = args.scalar_args[0];
  int n = args.scalar_args[1];
  auto y_hist = args.inputs[0];
  ENSURE(args.inputs.size() >= 3, "CsReg should have at least 3 inputs");

  int num_xs = args.inputs.size() - 1;
  ENSURE2(n >= 0 && n <= num_xs);

  ENSURE2(!y_hist.empty());
  for (int k = 0; k < num_xs; ++k) ENSURE2(args.inputs[k + 1].size() == y_hist.size());

  auto &output = args.output;
  int size = args.output.size();
  std::vector<Float> y;
  y.reserve(y_hist.size() * size);
  std::vector<Float> xs;
  xs.reserve(y_hist.size() * size * num_xs);

  std::vector<bool> output_filter(size);

  std::vector<Float> tmp_y(y_hist.size());
  std::vector<Float> tmp_xs(y_hist.size() * num_xs);
  math::MatView<Float> tmp_xs_mat(xs.data(), num_xs, y_hist.size());
  std::vector<int> head_idx(size);

  auto validate_row = [&](int row, int i) {
    if (!IsValid(y_hist[row][i])) return false;
    for (int k = 0; k < num_xs; ++k) {
      if (!IsValid(args.inputs[k + 1][row][i])) return false;
    }
    return true;
  };

  auto mask = args.mask;
  for (int i = 0; i < size; ++i) {
    if (!mask.empty() && !mask[i]) {
      continue;
    }
    if (!validate_row(y_hist.size() - 1, i)) continue;

    output_filter[i] = true;

    head_idx[i] = y.size();
    for (int ti = 0; ti < ts_len; ++ti) {
      if (ti > static_cast<int>(y_hist.size()) - 1) break;
      int j = y_hist.size() - 1 - ti;
      if (!validate_row(j, i)) continue;

      y.push_back(y_hist[j][i]);
      for (int k = 0; k < num_xs; ++k) {
        xs.push_back(args.inputs[k + 1][j][i]);
      }
    }
  }
  int num_samples = y.size();
  if (num_samples < num_xs) return;
  arma::mat Y(num_samples, 1);
  arma::mat X(num_samples, num_xs);
  for (int i = 0; i < num_samples; ++i) {
    Y(i, 0) = y[i];
  }
  for (int i = 0; i < num_samples; ++i) {
    for (int j = 0; j < num_xs; ++j) {
      X(i, j) = xs[i * num_xs + j];
    }
  }
  auto reg = math::LinearRegression(X, Y);
  if (n > 0) {
    for (int i = 0; i < size; ++i) {
      if (output_filter[i]) output[i] = reg(n - 1, 0);
    }
  } else {
    for (int i = 0; i < size; ++i) {
      if (output_filter[i]) {
        output[i] = Y(head_idx[i], 0) - arma::dot(reg.col(0), X.row(head_idx[i]));
      }
    }
  }
}

void TsIndex::ApplyImpl(const FunctionArgs &args) const {
  detail::ApplyTsBinary(fill_invalid(), args, [](auto &x, auto &y) -> Float {
    int idx = y.back();
    if (idx > x.size() - 1) return NAN;
    return x[x.size() - 1 - idx];
  });
}

}  // namespace yang::expr
