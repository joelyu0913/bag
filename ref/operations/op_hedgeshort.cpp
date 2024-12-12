#include "yang/math/mat_ops.h"
#include "yang/sim/operation.h"
#include "yang/util/strings.h"

namespace {

struct OpHedgeShort : yang::Operation {
  void Apply(MatView<float> sig, const yang::Env &env, int start_di, int end_di,
             const std::vector<std::string> &args) final {
    Apply(sig, env, start_di, end_di, yang::CheckAtoi<int>(args.at(0)));
  }

  using yang::Operation::Apply;

  void Apply(MatView<float> sig, const yang::Env &env, int start_di, int end_di, int idx) {
    auto mat = sig.block(start_di, 0, end_di - start_di, sig.cols());
    yang::math::ops::hedgeshort(mat, env.univ_size(), idx + env.univ().index_id_start());
  }
};

REGISTER_OPERATION("hedgeshort", OpHedgeShort);

}  // namespace
