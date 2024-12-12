#include <vector>

#include "yang/sim/operation.h"
#include "yang/util/strings.h"

namespace {
// algo assigns cap proptional of the alpha to index, to ensure similarity level.
template <bool USE_LIMIT>
struct OpAdjWtBase : yang::Operation {
  bool lookback() const final {
    return true;
  }

  using yang::Operation::Apply;

  void Apply(MatView<float> sig, const yang::Env &env, int start_di, int end_di,
             const std::vector<std::string> &args) final {
    Apply(sig, env, start_di, end_di, args.at(0), yang::CheckAtod(args.at(1)),
          yang::CheckAtod(args.at(2)), yang::CheckAtod(args.at(3)), yang::CheckAtod(args.at(4)));
  }

  void Apply(MatView<float> sig, const yang::Env &env, int start_di, int end_di,
             const std::string &index, double cap_lower, double cap_upper, double similarity,
             double step) {
    auto close = env.ReadData<yang::Array<float>>("base", "close")->mat_view();
    auto limit_up = env.ReadData<yang::Array<float>>("base", "limit_up")->mat_view();
    auto limit_down = env.ReadData<yang::Array<float>>("base", "limit_down")->mat_view();
    auto halt = env.ReadData<yang::Array<bool>>("base", "halt")->mat_view();
    auto quota = env.ReadData<yang::Array<float>>("sup_univ", index + "_member")->mat_view();

    std::vector<bool> fixed(env.univ_size());
    for (int di = start_di; di < end_di; ++di) {
      for (int ii = 0; ii < env.univ_size(); ++ii) {
        fixed[ii] = halt(di, ii);
        if (!fixed[ii] && di > 0 && USE_LIMIT) {
          fixed[ii] = (close(di - 1, ii) > limit_up(di - 1, ii) * 0.999) ||
                      (close(di - 1, ii) < limit_down(di - 1, ii) * 1.001);
        }
      }
      double target_cap = 0;
      double cap_max = cap_upper;
      double cap_min = cap_lower;
      while (cap_max - cap_min >= step) {
        double cap = (cap_max + cap_min) / 2;
        double sum = 0;
        for (int ii = 0; ii < env.univ_size(); ++ii) {
          if (sig(di, ii) < 0 && quota(di, ii) > 0) {
            sum += std::min<double>(cap * -sig(di, ii), quota(di, ii));
          }
        }
        // assuming the sum of neg weight equals -1
        if (sum > similarity * cap * 1.0) {
          cap_min = cap;
          target_cap = cap;
        } else {
          cap_max = cap;
        }
      }
      if (target_cap == 0) {
        double usable_quota = 0;
        for (int ii = 0; ii < env.univ_size(); ++ii) {
          if (sig(di, ii) < 0 && quota(di, ii) > 0) {
            usable_quota += quota(di, ii);
          }
        }
        target_cap = std::min(usable_quota, cap_lower);
      }
      double sum_short = 0;
      double sum_fixed = 0;
      for (int ii = 0; ii < env.univ_size(); ++ii) {
        if (fixed[ii]) {
          sig(di, ii) = di == 0 ? 0 : sig(di - 1, ii);
          if (!std::isfinite(sig(di, ii))) sig(di, ii) = 0;
          sum_fixed += sig(di, ii);
        } else if (sig(di, ii) < 0) {
          if (quota(di, ii) > 0) {
            sig(di, ii) = std::max<double>(sig(di, ii) * target_cap, -quota(di, ii));
          } else {
            sig(di, ii) = 0;
          }
          sum_short += sig(di, ii);
        }
      }
      double pos_sum = -(sum_short + sum_fixed);
      if (pos_sum < 0) continue;
      double tmp_sum = 0;
      for (int ii = 0; ii < env.univ_size(); ++ii) {
        if (sig(di, ii) > 0 && !fixed[ii]) {
          tmp_sum += sig(di, ii);
        }
      }
      for (int ii = 0; ii < env.univ_size(); ++ii) {
        if (sig(di, ii) > 0 && !fixed[ii]) {
          sig(di, ii) *= pos_sum / tmp_sum;
        }
      }
    }
  }
};

using OpAdjWt = OpAdjWtBase<false>;
REGISTER_OPERATION("adjwt", OpAdjWt);

using OpAdjWtLimit = OpAdjWtBase<true>;
REGISTER_OPERATION("adjwtlimit", OpAdjWtLimit);

}  // namespace
