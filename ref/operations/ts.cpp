#include <vector>

#include "yang/data/array.h"
#include "yang/math/mat_view.h"
#include "yang/math/ops.h"
#include "yang/sim/operation.h"
#include "yang/util/strings.h"

namespace {

namespace ops = yang::math::ops;

static void FillInvalid(yang::math::VecView<float> range, int ts_len) {
  constexpr int TS_FILL_LEN = 60;
  ops::linear_fill(range.begin(), range.end(), TS_FILL_LEN);
}

template <bool COPY_RANGE = false>
void ApplyTsRange(yang::math::MatView<float> sig, int ts_len, auto &&func) {
  std::vector<float> col;
  std::vector<int> next_valid;
  std::vector<float> range_copy;
  for (int ii = 0; ii < sig.cols(); ++ii) {
    col.resize(sig.rows());
    for (int di = 0; di < sig.rows(); ++di) col[di] = sig(di, ii);
    FillInvalid(col, ts_len);

    next_valid.resize(col.size());
    for (int di = col.size() - 1; di >= 0; --di) {
      if (std::isfinite(col[di])) {
        next_valid[di] = di;
      } else if (di < static_cast<int>(col.size()) - 1) {
        next_valid[di] = next_valid[di + 1];
      } else {
        next_valid[di] = di + 1;
      }
    }

    for (int di = 0; di < sig.rows(); ++di) {
      if (!std::isfinite(sig(di, ii))) continue;
      int range_di_start = next_valid[std::max(0, di - ts_len + 1)];
      yang::math::VecView<float> range(col.data() + range_di_start, di - range_di_start + 1);
      if constexpr (COPY_RANGE) {
        range_copy.assign(range.begin(), range.end());
        sig(di, ii) = func(range_copy);
      } else {
        sig(di, ii) = func(range);
      }
    }
  }
}

struct OpTs : yang::Operation {
  bool lookback() const final {
    return true;
  }

  using yang::Operation::Apply;

  void Apply(MatView<float> sig, const yang::Env &env, int start_di, int end_di,
             const std::vector<std::string> &args) final {
    Apply(sig.block(0, 0, end_di, env.univ_size()), yang::CheckAtoi(args.at(0)));
  }

  virtual void Apply(MatView<float> sig, int ts_len) = 0;
};

struct OpTsMin : OpTs {
  void Apply(MatView<float> sig, int ts_len) final {
    ApplyTsRange(sig, ts_len, [](auto &range) { return ops::min(range.begin(), range.end()); });
  }
};
REGISTER_OPERATION("tsmin", OpTsMin);

struct OpTsMax : OpTs {
  void Apply(MatView<float> sig, int ts_len) final {
    ApplyTsRange(sig, ts_len, [](auto &range) { return ops::max(range.begin(), range.end()); });
  }
};
REGISTER_OPERATION("tsmax", OpTsMax);

struct OpTsSum : OpTs {
  void Apply(MatView<float> sig, int ts_len) final {
    ApplyTsRange(sig, ts_len, [](auto &range) { return ops::sum(range.begin(), range.end()); });
  }
};
REGISTER_OPERATION("tssum", OpTsSum);

struct OpTsCount : OpTs {
  void Apply(MatView<float> sig, int ts_len) final {
    ApplyTsRange(sig, ts_len, [](auto &range) {
      return ops::count<ops::CheckBool>(range.begin(), range.end());
    });
  }
};
REGISTER_OPERATION("tscount", OpTsCount);

struct OpTsProd : OpTs {
  void Apply(MatView<float> sig, int ts_len) final {
    ApplyTsRange(sig, ts_len, [](auto &range) { return ops::prod(range.begin(), range.end()); });
  }
};
REGISTER_OPERATION("tsprod", OpTsProd);

struct OpTsMean : OpTs {
  void Apply(MatView<float> sig, int ts_len) final {
    ApplyTsRange(sig, ts_len, [](auto &range) { return ops::mean(range.begin(), range.end()); });
  }
};
REGISTER_OPERATION("tsmean", OpTsMean);

struct OpTsDemean : OpTs {
  void Apply(MatView<float> sig, int ts_len) final {
    ApplyTsRange(sig, ts_len, [](auto &range) {
      return range[range.size() - 1] - ops::mean(range.begin(), range.end());
    });
  }
};
REGISTER_OPERATION("tsdemean", OpTsDemean);

struct OpTsDemean2 : OpTs {
  void Apply(MatView<float> sig, int ts_len) final {
    ApplyTsRange(sig, ts_len, [](auto &range) {
      return range[range.size() - 1] - ops::mean(range.begin(), range.end() - 1);
    });
  }
};
REGISTER_OPERATION("tsdemean2", OpTsDemean2);

struct OpTsRMean : OpTs {
  void Apply(MatView<float> sig, int ts_len) final {
    ApplyTsRange(sig, ts_len, [](auto &range) { return ops::rmean(range.begin(), range.end()); });
  }
};
REGISTER_OPERATION("tsrmean", OpTsRMean);

struct OpTsStd : OpTs {
  void Apply(MatView<float> sig, int ts_len) final {
    ApplyTsRange(sig, ts_len, [](auto &range) { return ops::stdev(range.begin(), range.end()); });
  }
};
REGISTER_OPERATION("tsstd", OpTsStd);

struct OpTsMeanDev : OpTs {
  void Apply(MatView<float> sig, int ts_len) final {
    ApplyTsRange(sig, ts_len,
                 [](auto &range) { return ops::mean_dev(range.begin(), range.end()); });
  }
};
REGISTER_OPERATION("tsmeandev", OpTsMeanDev);

struct OpTsZScore : OpTs {
  void Apply(MatView<float> sig, int ts_len) final {
    ApplyTsRange<true>(sig, ts_len, [](auto &range) {
      ops::zscore(range.begin(), range.end());
      return range[range.size() - 1];
    });
  }
};
REGISTER_OPERATION("tszscore", OpTsZScore);

struct OpTsCorrStep : OpTs {
  void Apply(MatView<float> sig, int ts_len) final {
    ApplyTsRange(sig, ts_len,
                 [](auto &range) { return ops::corr_step(range.begin(), range.end()); });
  }
};
REGISTER_OPERATION("tscorrstep", OpTsCorrStep);

struct OpTsArgmin : OpTs {
  void Apply(MatView<float> sig, int ts_len) final {
    ApplyTsRange(sig, ts_len, [](auto &range) { return ops::argmin(range.begin(), range.end()); });
  }
};
REGISTER_OPERATION("tsargmin", OpTsArgmin);

struct OpTsArgmax : OpTs {
  void Apply(MatView<float> sig, int ts_len) final {
    ApplyTsRange(sig, ts_len, [](auto &range) { return ops::argmax(range.begin(), range.end()); });
  }
};
REGISTER_OPERATION("tsargmax", OpTsArgmax);

struct OpTsRank : OpTs {
  void Apply(MatView<float> sig, int ts_len) final {
    ApplyTsRange<true>(sig, ts_len, [](auto &range) {
      ops::rank(range.begin(), range.end());
      return range.back();
    });
  }
};
REGISTER_OPERATION("tsrank", OpTsRank);

struct OpTsDelta : OpTs {
  void Apply(MatView<float> sig, int ts_len) final {
    ApplyTsRange(sig, ts_len + 1, [](auto &range) { return range.back() - range.front(); });
  }
};
REGISTER_OPERATION("tsdelta", OpTsDelta);

struct OpTsDecayLinear : OpTs {
  void Apply(MatView<float> sig, int ts_len) final {
    ApplyTsRange(sig, ts_len,
                 [](auto &range) { return ops::decay_linear(range.begin(), range.end()); });
  }
};
REGISTER_OPERATION("tsdecaylinear", OpTsDecayLinear);

struct OpTsEma : yang::Operation {
  bool lookback() const final {
    return true;
  }

  using yang::Operation::Apply;

  void Apply(MatView<float> sig, const yang::Env &env, int start_di, int end_di,
             const std::vector<std::string> &args) final {
    Apply(sig.block(0, 0, end_di, env.univ_size()), yang::CheckAtoi(args.at(0)),
          yang::CheckAtof(args.at(1)));
  }

  void Apply(MatView<float> sig, int ts_len, float ratio) {
    ApplyTsRange(sig, ts_len, [ratio](auto &range) {
      return ops::ema<float>(range.begin(), range.end(), ratio);
    });
  }
};
REGISTER_OPERATION("tsema", OpTsEma);

}  // namespace
