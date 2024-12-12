#include <cmath>

#include "yang/math/mat_ops.h"
#include "yang/sim/operation.h"
#include "yang/util/strings.h"

namespace {

struct OpRisk : yang::Operation {
  using yang::Operation::Apply;

  void Apply(MatView<float> sig, const yang::Env &env, int start_di, int end_di,
             const std::vector<std::string> &args) final {
    std::map<std::string, std::string> mp = {{"carra", "B_risk_carra_cne5"},
                                             {"carra1", "B_risk_carra_cne5v1"},
                                             {"barra", "R_barra_cne5"},
                                             {"rq", "R_rq_fexp"},
                                             {"tl", "B_risk_tl"}};

    auto rdata_name = args[0];
    std::vector<std::string> names(args.begin() + 1, args.end());
    auto member = env.ReadData<yang::Array<float>>("sup_univ/csi500_member")->mat_view();

    int rsize = names.size();

    std::vector<yang::math::MatView<const float>> rf = {};

    for (const std::string &rfactor : names) {
      rf.push_back(
          env.ReadData<yang::Array<float>>(fmt::format("{0}/{1}", mp.at(rdata_name), rfactor))
              ->mat_view());
    }

    auto univ_size = env.univ_size();
    for (int di = start_di; di < end_di; di++) {
      std::vector<std::vector<float>> val_norm =
          std::vector<std::vector<float>>(rsize, std::vector<float>(univ_size, NAN));

      float sum_sig = 0.;
      for (int ii = 0; ii < univ_size; ii++) {
        if (sig(di, ii) > 0.) sum_sig += sig(di, ii);
      }
      for (int ii = 0; ii < univ_size; ii++) {
        if (sig(di, ii) > 0.) {
          sig(di, ii) /= sum_sig;
        } else {
          sig(di, ii) = 0.;
        }
      }
      for (int ri = 0; ri < rsize; ri++) {
        auto rdata = rf[ri];
        int cnt = 0;
        float mean = 0.;
        float std = 0.;
        for (int ii = 0; ii < univ_size; ii++) {
          auto val = rdata(di, ii);
          if (std::isfinite(val)) {
            cnt++;
            mean += val;
            std += val * val;
          }
        }
        mean /= cnt;
        std = pow(std / cnt, 0.5);
        if (!std::isfinite(std) && di > 300) {
          LOG_ERROR("Invalid std: {} {}", di, std);
        }
        for (int ii = 0; ii < univ_size; ii++) {
          auto val = rdata(di, ii);
          if (std::isfinite(val)) {
            val_norm[ri][ii] = (val - mean) / std;
          }
        }
      }

      std::vector<float> target = std::vector<float>(names.size(), 0.);
      for (int ri = 0; ri < (int)names.size(); ri++) {
        float idx_expo = 0.;
        float idx_wt = 0.;
        for (int ii = 0; ii < univ_size; ii++) {
          if (std::isfinite(member(di, ii)) && std::isfinite(val_norm[ri][ii])) {
            idx_wt += member(di, ii);
            idx_expo += member(di, ii) * val_norm[ri][ii];
          }
        }
        target[ri] = idx_expo / idx_wt;
      }

      int iters = 200;
      // int iters = 1;
      int iter = 0;
      while (iter < iters) {
        iter++;
        std::vector<float> delta = std::vector<float>(univ_size, 0.);
        for (int ri = 0; ri < (int)names.size(); ri++) {
          std::vector<int> index;
          std::vector<float> alpha_value;
          float alphas_sum = 0.0;
          float size_sum = 0.0;
          float size_size_sum = 0.0;
          float size_alpha_sum = 0.0;
          for (int ii = 0; ii < univ_size; ++ii) {
            if (sig(di, ii) > 0. && std::isfinite(val_norm[ri][ii])) {
              index.push_back(ii);
              alpha_value.push_back(sig(di, ii));
              alphas_sum += sig(di, ii);
              size_sum += val_norm[ri][ii];
              size_size_sum += val_norm[ri][ii] * val_norm[ri][ii];
              size_alpha_sum += sig(di, ii) * val_norm[ri][ii];
            }
          }

          float lambda = (size_alpha_sum / alphas_sum - target[ri]) /
                         (target[ri] * size_sum - size_size_sum) / 2.0 / 20.;
          float times = 1 / (1 + lambda * size_sum);
          for (int l = 0; l < (int)index.size(); l++) {
            delta[index[l]] += -sig(di, index[l]) + times * (alpha_value[l] / alphas_sum +
                                                             lambda * val_norm[ri][index[l]]);
          }
        }
        for (int ii = 0; ii < univ_size; ++ii) {
          if (sig(di, ii) > 0.) sig(di, ii) += delta[ii] / names.size();
          if (sig(di, ii) <= 0) {
            sig(di, ii) = NAN;
          }
        }
      }
    }
  }
};

REGISTER_OPERATION("risk", OpRisk);

}  // namespace
