#include "yang/math/mat_ops.h"
#include "yang/math/ops.h"
#include "yang/sim/operation.h"
#include "yang/util/strings.h"
namespace {
namespace ops = yang::math::ops;
struct OpFillna : yang::Operation {
  using yang::Operation::Apply;

  void Apply(MatView<float> sig, const yang::Env &env, int start_di, int end_di,
             const std::vector<std::string> &args) final {
    for (int di = start_di; di < end_di; di++) {
      for (int ii = 0; ii < env.univ_size(); ii++) {
        if (!std::isfinite(sig(di, ii))) {
          sig(di, ii) = 0.;
        }
      }
    }
  }
};

struct OpFfillna : yang::Operation {
  using yang::Operation::Apply;

  void Apply(MatView<float> sig, const yang::Env &env, int start_di, int end_di,
             const std::vector<std::string> &args) final {
    for (int di = std::max(1, start_di); di < end_di; di++) {
      for (int ii = 0; ii < env.univ_size(); ii++) {
        if (std::isfinite(sig(di - 1, ii)) && !std::isfinite(sig(di, ii))) {
          sig(di, ii) = 0.;
        }
      }
    }
  }
};

REGISTER_OPERATION("fillna", OpFillna);
REGISTER_OPERATION("ffillna", OpFfillna);
}  // namespace
