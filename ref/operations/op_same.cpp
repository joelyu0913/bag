#include "yang/math/mat_ops.h"
#include "yang/sim/operation.h"
#include "yang/util/strings.h"

namespace {

struct OpSame : yang::Operation {
  void Apply(MatView<float> sig, const yang::Env &env, int start_di, int end_di,
             const std::vector<std::string> &args) final {
    Apply(sig, env, start_di, end_di, yang::CheckAtof(args.at(0)));
  }

  void Apply(MatView<float> sig, const yang::Env &env, int start_di, int end_di, float thd) {
    for (int di = start_di; di < end_di; di++) {
      for (int ii = 0; ii < env.univ_size(); ii++) {
        if (sig(di, ii) > thd) {
          sig(di, ii) = 1.;
        } else {
          sig(di, ii) = NAN;
        }
      }
    }
  }
};

REGISTER_OPERATION("same", OpSame);

}  // namespace
