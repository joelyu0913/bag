#include "yao/base_daily.h"
#include "yao/taq.h"

using namespace yao;

namespace {

class SupCut : public BaseDaily {
 public:
  TaqLoader taq_;
  int32_t ti_cut_;
  std::string tm_cut_;
  yang::Array<float> ret_vwap_tail_;
  yang::Array<float> ret_trd_cut_;
  yang::Array<float> ret_mid_cut_;
  yang::Array<float> vwap_tail_;
  yang::Array<float> high_;
  yang::Array<float> low_;
  yang::Array<float> close_;
  yang::Array<float> cumdvol_;
  yang::Array<float> cumvol_;
  yang::Array<float> cumvwap_;

  void user_proc() {
    if (config<std::string>("par_cut", "") != "") {
      LOG_FATAL("sup_cut.par_cut is deprecated, please use env.cut_time");
    }

    taq_.Initialize(env());
    int cut_time = env().config<int>("cut_time", 1445);
    LOG_INFO("cut_time: {:04d}", cut_time);
    ti_cut_ = std::min(taq_.GetTi(cut_time), taq_.taq_size() - 1);
    tm_cut_ = std::to_string(taq_[ti_cut_]);

    auto taq = config<std::string>("taq");
    ret_vwap_tail_ = WriteArray<float>(taq, "b_ret_vwap_tail." + tm_cut_);
    ret_trd_cut_ = WriteArray<float>(taq, "b_ret_trd_cut." + tm_cut_);
    ret_mid_cut_ = WriteArray<float>(taq, "b_ret_mid_cut." + tm_cut_);
    vwap_tail_ = WriteArray<float>(taq, "b_vwap_tail." + tm_cut_);

    high_ = WriteArray<float>(taq, "b_high." + tm_cut_);
    low_ = WriteArray<float>(taq, "b_low." + tm_cut_);
    close_ = WriteArray<float>(taq, "b_close." + tm_cut_);
    cumdvol_ = WriteArray<float>(taq, "b_dvol." + tm_cut_);
    cumvol_ = WriteArray<float>(taq, "b_vol." + tm_cut_);
    cumvwap_ = WriteArray<float>(taq, "b_vwap." + tm_cut_);

    cal();
  }

 private:
  void cal() {
    LOG_INFO("first stage cal ret, cut: {}/{}", tm_cut_, ti_cut_);
    int start_di_ = std::max(0, start_di - 1);
    std::vector<float> pvwap_tail = {};
    std::vector<float> vwap_tail = {};
    auto &i_trd_cumdvol = taq_["i_trd_cumdvol"];
    auto &i_trd_cumvol = taq_["i_trd_cumvol"];
    auto &i_trd_cumvwap = taq_["i_trd_cumvwap"];
    auto &i_trd_last = taq_["i_trd_last"];
    auto &i_mid_last = taq_["i_mid_last"];
    auto &i_trd_high = taq_["i_trd_high"];
    auto &i_trd_low = taq_["i_trd_low"];
    auto adj = ReadArray<float>("base/adj");

    auto get_tail_vwap = [&](int di, int ii, int start_ti) {
      int end_ti = taq_.taq_size() - 1;
      float dvol_c = i_trd_cumdvol(di, end_ti, ii) - i_trd_cumdvol(di, start_ti + 1, ii);
      float vol_c = i_trd_cumvol(di, end_ti, ii) - i_trd_cumvol(di, start_ti + 1, ii);

      return dvol_c / vol_c;
    };

    for (int ii = 0; ii < max_univ_sz; ++ii) {
      pvwap_tail.push_back(get_tail_vwap(start_di_, ii, ti_cut_));
    }
    for (int di = start_di_ + 1; di < end_di; di++) {
      vwap_tail = {};
      for (int ii = 0; ii < max_univ_sz; ++ii) {
        vwap_tail.push_back(get_tail_vwap(di, ii, ti_cut_));
      }
      for (int ii = 0; ii < max_univ_sz; ++ii) {
        ret_vwap_tail_(di, ii) = vwap_tail[ii] * adj->Get(di, ii) / pvwap_tail[ii] - 1;
        ret_trd_cut_(di, ii) =
            i_trd_last(di, ti_cut_, ii) * adj->Get(di, ii) / i_trd_last(di - 1, ti_cut_, ii) - 1;
        ret_mid_cut_(di, ii) =
            i_mid_last(di, ti_cut_, ii) * adj->Get(di, ii) / i_mid_last(di - 1, ti_cut_, ii) - 1;
        vwap_tail_(di, ii) = vwap_tail[ii];
      }
      std::swap(pvwap_tail, vwap_tail);
    }

    LOG_INFO("second stage, cut: {}/{}", tm_cut_, ti_cut_);

    for (int di = start_di; di < end_di; di++) {
      LOG_INFO("[{}] Update information on date {}. (finished {}/{})", name(), dates()[di], di,
               date_sz - 1);
      for (int ii = 0; ii < max_univ_sz; ii++) {
        float vol_d = 0, dvol_d = 0, vwap_d = NAN;
        for (int ti = 0; ti <= ti_cut_; ti++) {
          float cumvol = i_trd_cumvol(di, ti, ii);
          float cumvwap = i_trd_cumvwap(di, ti, ii);
          float cumdvol = i_trd_cumdvol(di, ti, ii);

          if (my_valid(cumvol)) {
            vol_d = cumvol;
          }
          if (my_valid(cumdvol)) {
            dvol_d = cumdvol;
          }
          if (my_valid(cumvwap)) {
            vwap_d = cumvwap;
          }
        }
        cumdvol_(di, ii) = dvol_d;
        cumvol_(di, ii) = vol_d;
        cumvwap_(di, ii) = vwap_d;
      }

      for (int ii = 0; ii < max_univ_sz; ii++) {
        float high_d = NAN, low_d = NAN, last_d = NAN;
        for (int ti = 0; ti <= ti_cut_; ti++) {
          float high = i_trd_high(di, ti, ii);
          float low = i_trd_low(di, ti, ii);
          float last = i_trd_last(di, ti, ii);

          if (!my_valid(high_d) && my_valid(high)) {
            high_d = high;
          }
          if (!my_valid(low_d) && my_valid(low)) {
            low_d = low;
          }
          if (high_d < high) {
            high_d = high;
          }
          if (low < low_d) {
            low_d = low;
          }
          if (my_valid(last)) {
            last_d = last;
          }
        }
        high_(di, ii) = high_d;
        low_(di, ii) = low_d;
        close_(di, ii) = last_d;
      }
    }
  }
};

REGISTER_MODULE(SupCut);

}  // namespace
