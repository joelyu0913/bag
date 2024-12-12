#include <cmath>

#include "yang/math/mat_ops.h"
#include "yang/sim/operation.h"
#include "yang/util/strings.h"

namespace {
// bound size based on interval dvol to avoid impact
struct OpBoundSize : yang::Operation {
  using yang::Operation::Apply;

  void Apply(MatView<float> sig, const yang::Env &env, int start_di, int end_di,
             const std::vector<std::string> &args) final {
    Apply(sig, env, start_di, end_di, std::stof(args[0]));
  }

  void Apply(MatView<float> sig, const yang::Env &env, int start_di, int end_di, float bound_) {
    for (int di = start_di; di < end_di; di++) {
      float sum_tt = 0.;
      std::vector<int> iis = {};
      for (int ii = 0; ii < env.univ_size(); ii++) {
        if (sig(di, ii) > 0.) {
          sum_tt += sig(di, ii);
          iis.push_back(ii);
        }
      }
      float bound = std::max(1 / float(iis.size()), bound_);
      float sum_ = sum_tt;
      float bound_x = bound * sum_tt;
      bool flag_run = true;
      while (flag_run) {
        flag_run = false;
        float sum_var = 0.;
        std::vector<int> iis_tmp = {};
        for (int ii : iis) {
          if (sig(di, ii) > bound_x) {
            sum_ -= bound_x;
            sig(di, ii) = bound_x;
          } else if (sig(di, ii) > 0.) {
            sum_var += sig(di, ii);
            iis_tmp.push_back(ii);
          }
        }
        iis = iis_tmp;
        float r = sum_ / sum_var;
        for (auto ii : iis) {
          sig(di, ii) *= r;
          if (sig(di, ii) > bound_x) {
            flag_run = true;
          }
        }
      }
      for (int ii = 0; ii < env.univ_size(); ii++) {
        if (sig(di, ii) > bound_x)
          LOG_INFO("({}, {}), {} :{}, {} : {}", di, ii, bound, sum_tt, sum_, sig(di, ii));
      }
    }
  }
};

struct OpBoundDvol : yang::Operation {
  using yang::Operation::Apply;

  void Apply(MatView<float> sig, const yang::Env &env, int start_di, int end_di,
             const std::vector<std::string> &args) final {
    Apply(sig, env, start_di, end_di, args[0], std::stof(args[1]), std::stof(args[2]));
  }

  void Apply(MatView<float> sig, const yang::Env &env, int start_di, int end_di, std::string type,
             float booksize, float thd) {
    std::string dvol_cache = "";
    if (type == "30m") {
      dvol_cache = "sup_dvol_itd__30min/b_sig";
    } else if (type == "30mrolling") {
      dvol_cache = "sup_dvol_itd__30m_rolling/b_sig_op";

    } else {
      LOG_INFO("error!");
    }

    auto dvol = env.ReadData<yang::Array<float>>(dvol_cache)->mat_view();
    for (int di = start_di; di < end_di; di++) {
      float sum_tt = 0.;
      std::vector<std::pair<int, float>> wts = {};
      std::map<int, float> bounds = {};
      for (int ii = 0; ii < env.univ_size(); ii++) {
        if (sig(di, ii) > 1e-6) {
          sum_tt += sig(di, ii);
          wts.push_back(std::make_pair(ii, sig(di, ii)));
          bounds[ii] = dvol(di, ii) * thd / booksize;
        }
        if (std::isfinite(sig(di, ii))) {
          sig(di, ii) = 0;
        }
      }

      float remaining_wt = 0.;
      std::vector<std::pair<int, float>> wts_new = {};
      for (int i = 0; i < (int)wts.size(); i++) {
        int ii = wts[i].first;
        wts[i].second /= sum_tt;
        if (wts[i].second > bounds[ii]) {
          sig(di, ii) = bounds[ii];
          remaining_wt += (wts[i].second - bounds[ii]);
        } else {
          sig(di, ii) = wts[i].second;
          wts_new.push_back(std::make_pair(ii, bounds[ii] - wts[i].second));
        }
      }

      std::stable_sort(wts_new.begin(), wts_new.end(),
                       [&](std::pair<int, float> s1, std::pair<int, float> s2) {
                         return s1.second < s2.second;
                       });

      int wts_sz = wts_new.size();
      for (int i = 0; i < wts_sz; i++) {
        float quota = wts_new[i].second;
        float inc_wt = (wts_sz - i) * quota;
        if (inc_wt < remaining_wt) {
          for (int j = i; j < wts_sz; j++) {
            sig(di, wts_new[j].first) += quota;
          }
          remaining_wt -= inc_wt;
        } else {
          float avg_wt = remaining_wt / (wts_sz - i);
          for (int j = i; j < wts_sz; j++) {
            sig(di, wts_new[j].first) += avg_wt;
          }
          remaining_wt = 0.;
          break;
        }
      }
    }
  }
};

REGISTER_OPERATION("boundsize", OpBoundSize);
REGISTER_OPERATION("bounddvol", OpBoundDvol);

}  // namespace
