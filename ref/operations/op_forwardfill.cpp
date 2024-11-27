#include "yang/math/mat_ops.h"
#include "yang/math/ops.h"
#include "yang/sim/operation.h"
#include "yang/util/strings.h"
namespace {
namespace ops = yang::math::ops;
struct OpForwardfill : yang::Operation {
  bool lookback() const final {
    return true;
  }

  using yang::Operation::Apply;

  void Apply(MatView<float> sig, const yang::Env &env, int start_di, int end_di,
             const std::vector<std::string> &args) final {
    if (args.size() > 0) {
      Apply(sig, env, start_di, end_di, yang::CheckAtoi<int>(args.at(0)));
    } else {
      Apply(sig, env, start_di, end_di, 100);
    }
  }

  void Apply(MatView<float> sig, const yang::Env &env, int start_di, int end_di, int back_len) {
    for (int ii = 0; ii < (int)env.univ_size(); ii++) {
      auto it = sig.col(ii).to_vec().begin();
      yang::math::ops::ffill(it + start_di, it + end_di, it + start_di, back_len);
    }
  }
};

REGISTER_OPERATION("forwardfill", OpForwardfill);

}  // namespace
