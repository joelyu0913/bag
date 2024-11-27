#include "yang/sim/operation.h"
#include "yang/util/strings.h"

namespace {

struct OpReject : yang::Operation {
  using yang::Operation::Apply;

  void Apply(MatView<float> sig, const yang::Env &env, int start_di, int end_di,
             const std::vector<std::string> &args) final {
    Apply(sig, env, start_di, end_di, args.at(0));
  }

  void Apply(MatView<float> sig, const yang::Env &env, int start_di, int end_di,
             const std::string &cond) {
    std::string cond_data;
    if (cond == "state") {
      cond_data = "data_state/state";
    } else {
      LOG_FATAL("Unknown condition: {}", cond);
    }
    auto &cond_arr = *env.ReadData<yang::Array<int>>(cond_data);
    for (int di = start_di; di < end_di; ++di) {
      for (int ii = 0; ii < env.univ_size(); ++ii) {
        if (cond_arr(di, ii) != 0) sig(di, ii) = NAN;
      }
    }
  }
};

REGISTER_OPERATION("reject", OpReject);

}  // namespace
