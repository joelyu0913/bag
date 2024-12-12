#include "yang/math/mat_ops.h"
#include "yang/sim/operation.h"
#include "yang/util/strings.h"

namespace {

struct OpScaleRaw : yang::Operation {
  using yang::Operation::Apply;

  void Apply(MatView<float> sig, const yang::Env &env, int start_di, int end_di,
             const std::vector<std::string> &args) final {
    float scale_size = yang::DISPLAY_BOOK_SIZE * 2;
    if (args.size() > 0) scale_size = yang::CheckAtof(args.at(0));
    Apply(sig, env, start_di, end_di, scale_size);
  }

  void Apply(MatView<float> sig, const yang::Env &env, int start_di, int end_di, float scale_size) {
    auto mat = sig.block(start_di, 0, end_di - start_di, sig.cols());
    yang::math::ops::scale(mat, scale_size, 1e-10);
  }
};

REGISTER_OPERATION("scaleraw", OpScaleRaw);

}  // namespace
