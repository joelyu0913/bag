#include "yang/util/unordered_set.h"
#include "yao/base_daily.h"
using namespace yao;

namespace {
using namespace std;
class SupUnivCn : public yang::Module {
 public:
  void RunImpl() final {
    using hash_int_set = yang::unordered_set<int>;

    auto univ = config<std::string>("par_univ");
    auto b_univ_custom = WriteArray<bool>(univ);
    auto b_sup_passed = WriteArray<bool>(univ + "_passed");
    int univ_size_ = config<int>("par_univ_size");
    int lookback_ = config<int>("par_univ_lookback");
    int sticky_ = config<int>("par_univ_sticky_days");
    int vol_window_ = config<int>("par_univ_vol_window");
    float pass_rate_ = config<float>("par_univ_pass_rate");
    float min_cap_ = config<float>("par_univ_min_cap");
    float min_dvol_ = config<float>("par_univ_min_dvol");
    float min_prc_ = config<float>("par_univ_min_prc");
    float max_prc_ = config<float>("par_univ_max_prc");
    string vol_mode_ = "mean";

    auto &b_univ_parent = *ReadArray<bool>(config<string>("par_univ_parent", "base/univ_all"));

    std::map<int, hash_int_set> candidates;  // map of date => stocks in candidate universe
    auto &b_close = *ReadArray<float>("base/close");
    auto &b_vol = *ReadArray<float>("base/vol");
    auto &b_cap = *ReadArray<float>("base/cap");
    auto &b_st = *ReadArray<int>("base/st");

    auto get_dollar_vol = [&](int di, int ii) {
      // int currency = (*currency_)(di, ii);
      // if (IsNull(currency)) return Null<float>();

      // float fxrate = (*fxrate_)(di, currency);
      float fxrate = 1;
      float price = b_close(di, ii);
      float volume = b_vol(di, ii);

      return price * volume / fxrate;
    };

    for (int di = (start_di() > lookback_ ? start_di() - lookback_ : 1); di < end_di(); ++di) {
      std::multimap<float, int> sorted_vol;  // sorted map of avg_vol => instrument
      for (int ii = 0; ii < univ_size(); ++ii) {
        // filter by exchange

        float fx = 1;

        if (b_univ_parent(di, ii) && b_st(di, ii) == 0 && my_valid(b_cap(di - 1, ii)) &&
            (b_cap(di - 1, ii) / fx >= min_cap_) && (b_close(di - 1, ii) / fx >= min_prc_) &&
            (b_close(di - 1, ii) / fx <= max_prc_)) {
          // compute trailing average notional volume for current
          // instrument

          vector<float> dvols;
          for (int w = 0; w < vol_window_; ++w) {
            if (w > di - 1) break;
            float dvol = get_dollar_vol(di - 1 - w, ii);
            if (dvol >= 0 && my_valid(dvol)) {
              dvols.push_back(dvol);
            }
          }

          if (dvols.size() > 0) {
            float est_dvol = CalcEstimatedVol(dvols, vol_mode_);

            if (est_dvol >= min_dvol_) sorted_vol.insert(std::make_pair(est_dvol, ii));
          }
        }
      }

      // select the top (univ_size) stocks as candidates
      // traversing sorted_vol backwards gives us the avg volumes in
      // decreasing order
      candidates.insert(std::make_pair(di, hash_int_set(univ_size_)));
      hash_int_set &result = candidates[di];
      for (std::multimap<float, int>::reverse_iterator rcurr = sorted_vol.rbegin();
           rcurr != sorted_vol.rend() && static_cast<int>(result.size()) < univ_size_; ++rcurr) {
        result.insert(rcurr->second);
      }
      if (result.empty()) {
        LOG_ERROR("[{}] no stocks matched universe volume criteria on {}", name(), dates()[di]);
      } else {
        LOG_INFO("[{}] Selected {} candidates on {} in the first pass", name(), result.size(),
                 dates()[di]);
      }
    }

    // 2nd pass
    std::vector<int> pick_cnt(univ_size());
    std::vector<int> last_good(univ_size(), -1);
    for (int ii = 0; ii < univ_size(); ++ii) {
      const int di = start_di() - 1;
      if (di < 0) continue;
      const int lb1 = (di > lookback_) ? lookback_ : di;
      const int lb2 = (di > sticky_) ? sticky_ : di;
      for (int dl = 0; dl < lb1; ++dl) {
        pick_cnt[ii] += candidates[di - dl].find(ii) != candidates[di - dl].end();
      }
      for (int dl = 0; dl < lb2; ++dl) {
        if (b_sup_passed(di - dl, ii)) {
          last_good[ii] = di - dl;
          break;
        }
      }
    }
    for (int di = start_di(); di < end_di(); ++di) {
      if (di < 1) continue;  // this selection has to always be delay 1

      const int lb1 = (di > lookback_) ? lookback_ : di;
      const int lb2 = (di > sticky_) ? sticky_ : di;

      // error check: make sure all necessary lookback dates were covered
      // in 1st pass
      int dl;
      for (dl = 0; dl < lb1; ++dl) {
        if (candidates.find(di - dl) == candidates.end()) {
          LOG_WARN("[{}] universe: candidate universe was not computed for {}", name(),
                   di - 1 - dl);
        }
      }

      int activated = 0, added = 0, removed = 0;
      for (int ii = 0; ii < univ_size(); ++ii) {
        // master universe gets priority

        bool good = b_univ_parent(di, ii);
        if (di > lookback_ && pick_cnt[ii] > 0) {
          pick_cnt[ii] -= candidates[di - lookback_].find(ii) != candidates[di - lookback_].end();
        }

        if (good) {
          // check the last 'lookback' days
          // candidates was computed with delay 1 info, so we can
          // start with today
          if (candidates[di].find(ii) != candidates[di].end()) ++pick_cnt[ii];
          ENSURE2(pick_cnt[ii] <= lb1);

          if (lb1 > 0 && static_cast<double>(pick_cnt[ii]) / lb1 * 100. < pass_rate_) good = false;

          // record whether it passed_ today
          b_sup_passed(di, ii) = good;

          if (good) {
            last_good[ii] = di;
          } else {
            // must be bad for 'sticky' days in order to deactivate
            good = last_good[ii] >= di - lb2;
          }
        }

        if (good) {
          ++activated;
          b_univ_custom(di, ii) = true;
          if (0 == di || !b_univ_custom(di - 1, ii)) ++added;
        } else {
          b_univ_custom(di, ii) = false;
          if (0 == di || b_univ_custom(di - 1, ii)) ++removed;
        }
      }

      LOG_INFO("[{}] Activated {} instruments on {} (+{} -{})", name(), activated,
               env().dates()[di], added, removed);
    }
  }

  float CalcEstimatedVol(vector<float> &vols, string vol_mode_) {
    float est_vol = 0.0;
    auto find_median = [&]() {
      int n = vols.size() / 2;
      std::nth_element(vols.begin(), vols.begin() + n, vols.end());
      if (vols.size() % 2 == 0) {
        return (vols[n] + *std::max_element(vols.begin(), vols.begin() + n)) / 2;
      } else {
        return vols[n];
      }
    };
    auto find_mean = [&]() {
      float sum = 0;
      for (auto &v : vols) {
        sum += v;
      }
      return sum / vols.size();
    };

    if (vol_mode_ == "median") {
      est_vol = find_median();
    } else if (vol_mode_ == "min_median_mean") {
      est_vol = std::min(find_median(), find_mean());
    } else {
      est_vol = find_mean();
    }

    return est_vol;
  }
};

REGISTER_MODULE(SupUnivCn);
}  // namespace
