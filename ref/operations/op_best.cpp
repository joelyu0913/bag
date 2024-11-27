#include "yang/math/mat_ops.h"
#include "yang/sim/operation.h"
#include "yang/util/strings.h"

namespace {

struct OpBest : yang::Operation {
  void Apply(MatView<float> sig, const yang::Env &env, int start_di, int end_di,
             const std::vector<std::string> &args) final {
    Apply(sig, env, start_di, end_di, yang::CheckAtoi(args.at(0)));
  }

  void Apply(MatView<float> sig, const yang::Env &env, int start_di, int end_di, int k) {
    auto mat = sig.block(start_di, 0, end_di - start_di, env.univ_size());
    yang::math::ops::select_n_largest(mat, k);
  }
};

REGISTER_OPERATION("best", OpBest);

}  // namespace
