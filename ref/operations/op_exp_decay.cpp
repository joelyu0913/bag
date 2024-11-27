#include "yang/math/mat_ops.h"
#include "yang/sim/operation.h"
#include "yang/util/strings.h"

namespace {

struct OpExpDecay : yang::Operation {
  bool lookback() const final {
    return true;
  }

  using yang::Operation::Apply;

  void Apply(MatView<float> sig, const yang::Env &env, int start_di, int end_di,
             const std::vector<std::string> &args) final {
    Apply(sig, env, start_di, end_di, yang::CheckAtof(args.at(0)));
  }

  void Apply(MatView<float> sig, const yang::Env &env, int start_di, int end_di, float ratio) {
    auto mat = sig.block(0, 0, end_di, env.univ_size());
    yang::math::ops::ewa(mat, ratio);
  }
};

REGISTER_OPERATION("expdecay", OpExpDecay);

}  // namespace
