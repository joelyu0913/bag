#include "yang/math/mat_ops.h"
#include "yang/sim/operation.h"
#include "yang/util/strings.h"

namespace {

struct OpTozero : yang::Operation {
  using yang::Operation::Apply;

  void Apply(MatView<float> sig, const yang::Env &env, int start_di, int end_di,
             const std::vector<std::string> &args) final {
    if (args.size() == 1) {
      Apply(sig, env, start_di, end_di, -yang::CheckAtof(args.at(0)), yang::CheckAtof(args.at(0)));
    } else {
      Apply(sig, env, start_di, end_di, yang::CheckAtof(args.at(0)), yang::CheckAtof(args.at(1)));
    }
  }

  void Apply(MatView<float> sig, const yang::Env &env, int start_di, int end_di, float lower,
             float upper) {
    for (int di = start_di; di < end_di; di++) {
      for (int ii = 0; ii < env.univ_size(); ii++) {
        if (sig(di, ii) > lower && sig(di, ii) < upper) {
          sig(di, ii) = 0.;
        }
      }
    }
  }
};

REGISTER_OPERATION("tozero", OpTozero);

}  // namespace
