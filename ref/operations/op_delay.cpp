#include "yang/math/mat_ops.h"
#include "yang/sim/operation.h"
#include "yang/util/strings.h"

namespace {

struct OpDelay : yang::Operation {
  bool lookback() const final {
    return true;
  }

  using yang::Operation::Apply;

  void Apply(MatView<float> sig, const yang::Env &env, int start_di, int end_di,
             const std::vector<std::string> &args) final {
    Apply(sig, env, start_di, end_di, yang::CheckAtoi<int>(args.at(0)));
  }

  void Apply(MatView<float> sig, const yang::Env &env, int start_di, int end_di, int periods) {
    auto mat = sig.block(0, 0, end_di, sig.cols());
    yang::math::ops::shift_rows(mat, periods);
  }
};

REGISTER_OPERATION("delay", OpDelay);

}  // namespace
