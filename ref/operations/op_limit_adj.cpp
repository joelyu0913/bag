#include "yao/B/basic/operations/op_limit_rounding.h"

namespace {

struct OpLimitAdj : yao::OpLimitRounding {
  void Apply(MatView<float> sig, const yang::Env &env, int start_di, int end_di,
             const std::vector<std::string> &args) override {
    float booksize = 1e9;
    ApplyImpl(sig, env, start_di, end_di, booksize);
  }
};

REGISTER_OPERATION("limitadj", OpLimitAdj);

}  // namespace
