#include <cmath>

#include "yang/math/mat_ops.h"
#include "yang/sim/operation.h"
#include "yang/util/strings.h"
#include "yang/util/unordered_map.h"

namespace {

struct OpNeutIdx : yang::Operation {
  void Apply(MatView<float> sig, const yang::Env &env, int start_di, int end_di,
             const std::vector<std::string> &args) final {
    ENSURE2(args.size() == 2);
    auto member = env.ReadData<yang::Array<float>>("sup_univ", args[0] + "_member")->mat_view();
    auto b_group = env.ReadData<yang::Array<int32_t>>("base", args[1])->mat_view();

    yang::unordered_map<int, float> mem_ind = {};
    yang::unordered_map<int, float> stk_ind = {};
    for (int di = start_di; di < end_di; di++) {
      mem_ind.clear();
      stk_ind.clear();

      float mem_ind_sum = 0.;
      float stk_ind_sum = 0.;
      for (int ii = 0; ii < env.univ_size(); ii++) {
        int group_id = b_group(di, ii);
        if (group_id > -1) {
          auto mem_ = member(di, ii);
          auto stk_ = sig(di, ii);
          if (std::isfinite(mem_)) {
            if (mem_ind.count(group_id) == 0) {
              mem_ind[group_id] = 0.;
            }
            mem_ind[group_id] += mem_;
            mem_ind_sum += mem_;
          }
          if (std::isfinite(stk_)) {
            if (stk_ind.count(group_id) == 0) {
              stk_ind[group_id] = 0.;
            }
            stk_ind[group_id] += stk_;
            stk_ind_sum += stk_;
          }
        }
      }

      for (auto &pr : stk_ind) {
        pr.second /= stk_ind_sum;
      }
      for (auto &pr : mem_ind) {
        pr.second /= mem_ind_sum;
        pr.second /= stk_ind[pr.first];
      }

      for (int ii = 0; ii < env.univ_size(); ii++) {
        int group_id = b_group(di, ii);
        if (group_id > -1) {
          sig(di, ii) *= mem_ind[group_id];
        } else {
          sig(di, ii) = NAN;
        }
      }
    }
  }
};

REGISTER_OPERATION("neutidx", OpNeutIdx);

}  // namespace
