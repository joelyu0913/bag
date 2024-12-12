#include "yang/math/mat_ops.h"
#include "yang/sim/operation.h"

namespace {

struct OpRank : yang::Operation {
  void Apply(MatView<float> sig, const yang::Env &env, int start_di, int end_di) final {
    auto mat = sig.block(start_di, 0, end_di - start_di, env.univ_size());
    auto listing = yang::Array<bool>::MMap(env.cache_dir().GetPath("env", "listing"));
    yang::math::ops::filter(
        mat, listing.mat_view().block(start_di, 0, end_di - start_di, env.univ_size()));
    yang::math::ops::rank(mat, 1e-6);
  }
};

REGISTER_OPERATION("rank", OpRank);

}  // namespace
