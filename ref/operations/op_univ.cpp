#include "yang/math/mat_ops.h"
#include "yang/sim/operation.h"
#include "yang/util/strings.h"

namespace {

struct OpUniv : yang::Operation {
  using yang::Operation::Apply;

  void Apply(MatView<float> sig, const yang::Env &env, int start_di, int end_di,
             const std::vector<std::string> &args) final {
    std::string str_univ = args.at(0);
    for (int i = 1; i < (int)args.size(); i++) {
      str_univ += "_" + args.at(i);
    }
    Apply(sig, env, start_di, end_di, str_univ);
  }

  void Apply(MatView<float> sig, const yang::Env &env, int start_di, int end_di,
             const std::string &univ) {
    auto mat = sig.block(start_di, 0, end_di - start_di, env.max_univ_size());
    std::string univ_data;
    if (univ == "all") {
      univ_data = "base/univ_all";
    } else if (univ == "listing") {
      univ_data = "env/listing";
    } else {
      univ_data = "sup_univ/" + univ;
    }
    auto *univ_arr = env.ReadData<yang::Array<bool>>(univ_data);
    auto u_mat = univ_arr->mat_view().block(start_di, 0, end_di - start_di, env.max_univ_size());
    yang::math::ops::filter(mat, u_mat);
  }
};

REGISTER_OPERATION("univ", OpUniv);

}  // namespace
