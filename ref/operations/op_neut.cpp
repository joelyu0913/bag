#include "yang/data/array.h"
#include "yang/math/mat_ops.h"
#include "yang/sim/operation.h"
#include "yang/util/strings.h"

namespace {

struct OpNeut : yang::Operation {
  using yang::Operation::Apply;

  void Apply(MatView<float> sig, const yang::Env &env, int start_di, int end_di,
             const std::vector<std::string> &args) final {
    Apply(sig, env, start_di, end_di, std::string(args.at(0)));
  }

  void Apply(MatView<float> sig, const yang::Env &env, int start_di, int end_di,
             const std::string &group) {
    if (group != "cty" && group != "sector" && group != "industry" && group != "subindustry") {
      throw yang::InvalidArgument("Invalid OpNeut group " + group);
    }
    auto *group_arr = env.ReadData<yang::Array<int>>("base", group);
    auto mat = sig.block(start_di, 0, end_di - start_di, env.univ_size());
    auto g_mat = group_arr->mat_view().block(start_di, 0, end_di - start_di, env.univ_size());
    yang::math::ops::group_demean(mat, g_mat);
  }
};

REGISTER_OPERATION("neut", OpNeut);

}  // namespace
