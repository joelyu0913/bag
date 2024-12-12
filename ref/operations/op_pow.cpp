#include "yang/math/mat_ops.h"
#include "yang/sim/operation.h"
#include "yang/util/strings.h"

namespace {

struct OpPow : yang::Operation {
  using yang::Operation::Apply;

  void Apply(MatView<float> sig, const yang::Env &env, int start_di, int end_di,
             const std::vector<std::string> &args) final {
    Apply(sig, env, start_di, end_di, yang::CheckAtof(args.at(0)));
  }

  void Apply(MatView<float> sig, const yang::Env &env, int start_di, int end_di, float exp) {
    auto mat = sig.block(start_di, 0, end_di - start_di, env.univ_size());
    yang::math::ops::spow(mat, exp);
  }
};

REGISTER_OPERATION("pow", OpPow);

}  // namespace
