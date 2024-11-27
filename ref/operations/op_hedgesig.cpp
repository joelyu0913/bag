#include "yang/math/mat_ops.h"
#include "yang/sim/operation.h"
#include "yang/util/strings.h"

namespace {

struct OpHedgeSig : yang::Operation {
  void Apply(MatView<float> sig, const yang::Env &env, int start_di, int end_di,
             const std::vector<std::string> &args) final {
    Apply(sig, env, start_di, end_di, args.at(0));
  }

  using yang::Operation::Apply;

  void Apply(MatView<float> sig, const yang::Env &env, int start_di, int end_di,
             const std::string &sid) {
    // auto mat = sig.block(start_di, 0, end_di - start_di, sig.cols());
    auto sid_ = sid;
    for (auto &c : sid_) c = std::toupper(c);
    int ii_hedge = env.univ().Find(sid_);
    if (ii_hedge < 0) {
      LOG_FATAL("Invalid sid = {}, ii = {}", sid_, ii_hedge);
    }
    for (int di = start_di; di < end_di; di++) {
      auto hedge_val = sig(di, ii_hedge);

      if (std::isfinite(hedge_val)) {
        for (int ii = 0; ii < sig.cols(); ii++) {
          sig(di, ii) -= hedge_val;
        }
      }
    }
  }
};

REGISTER_OPERATION("hedgesig", OpHedgeSig);

}  // namespace
