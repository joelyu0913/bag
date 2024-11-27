#include "yang/math/mat_ops.h"
#include "yang/sim/operation.h"
#include "yang/util/strings.h"

namespace {

struct OpHedge : yang::Operation {
  void Apply(MatView<float> sig, const yang::Env &env, int start_di, int end_di,
             const std::vector<std::string> &args) final {
    auto sid_str = args.at(0);
    if (sid_str == "csi300") {
      Apply(sig, env, start_di, end_di, 1);
    } else if (sid_str == "csi500") {
      Apply(sig, env, start_di, end_di, 0);
    } else if (sid_str == "csi1000") {
      Apply(sig, env, start_di, end_di, 4);
    } else if (sid_str == "csi2000") {
      Apply(sig, env, start_di, end_di, 5);
    } else {
      Apply(sig, env, start_di, end_di, yang::CheckAtoi<int>(args.at(0)));
    }
  }

  using yang::Operation::Apply;

  void Apply(MatView<float> sig, const yang::Env &env, int start_di, int end_di, int idx) {
    auto mat = sig.block(start_di, 0, end_di - start_di, sig.cols());
    yang::math::ops::hedge(mat, env.univ_size(), idx + env.univ().index_id_start());
  }
};

REGISTER_OPERATION("hedge", OpHedge);

}  // namespace
