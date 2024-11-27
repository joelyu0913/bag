#include "yang/math/mat_ops.h"
#include "yang/sim/operation.h"
#include "yang/util/strings.h"

namespace {

struct OpScale : yang::Operation {
  using yang::Operation::Apply;

  void Apply(MatView<float> sig, const yang::Env &env, int start_di, int end_di,
             const std::vector<std::string> &args) final {
    float scale_size = yang::DISPLAY_BOOK_SIZE * 2;
    if (args.size() > 0) scale_size = yang::CheckAtof(args.at(0));
    Apply(sig, env, start_di, end_di, scale_size);
  }

  void Apply(MatView<float> sig, const yang::Env &env, int start_di, int end_di, float scale_size) {
    auto mat = sig.block(start_di, 0, end_di - start_di, sig.cols());
    auto *univ_arr = env.ReadData<yang::Array<bool>>("base", "univ_all");
    auto u_mat = univ_arr->mat_view().block(start_di, 0, end_di - start_di, sig.cols());
    yang::math::ops::filter(mat, u_mat);
    yang::math::ops::scale(mat, scale_size, 1e-10);
  }
};

struct OpScaleBound : yang::Operation {
  using yang::Operation::Apply;

  void Apply(MatView<float> sig, const yang::Env &env, int start_di, int end_di,
             const std::vector<std::string> &args) final {
    Apply(sig, env, start_di, end_di, yang::CheckAtof(args.at(0)));
  }

  void Apply(MatView<float> sig, const yang::Env &env, int start_di, int end_di, float bound) {
    for (int di = start_di; di < end_di; di++) {
      float sum = 0.;
      for (int ii = 0; ii < env.univ_size(); ii++) {
        if (std::isfinite(sig(di, ii))) {
          sum += fabs(sig(di, ii));
        }
      }
      if (sum > bound) {
        float ratio = bound / sum;
        for (int ii = 0; ii < env.univ_size(); ii++) {
          sig(di, ii) *= ratio;
        }
      }
    }
  }
};

REGISTER_OPERATION("scale", OpScale);
REGISTER_OPERATION("scalebound", OpScaleBound);

}  // namespace
