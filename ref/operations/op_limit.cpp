#include "yao/B/basic/operations/op_limit.h"

#include <cmath>

#include "yang/math/mat_ops.h"
#include "yang/math/ops.h"
#include "yang/util/strings.h"

namespace yao {
void OpLimit::Apply(MatView<float> sig, const yang::Env &env, int start_di, int end_di,
                    const std::vector<std::string> &args) {
  float booksize = 1e9;
  std::string mod_name = "";

  if (args[0] == "ori") {
    mod_name = "sup_limit_onprc__i_close";
  } else if (args[0] == "taq") {
    mod_name = "sup_limit_onprc__" + args[1];
  } else if (args[0] == "open") {
    mod_name = "sup_limit_onprc__open";
  }
  auto limit_up_flag = env.ReadData<yang::Array<bool>>(mod_name, "limit_up_flag")->mat_view();
  auto limit_down_flag = env.ReadData<yang::Array<bool>>(mod_name, "limit_down_flag")->mat_view();

  auto halt = env.ReadData<yang::Array<bool>>("base", "halt")->mat_view();

  auto i_close_ = env.ReadData<yang::Array<float>>("ibase", "i_close");

  yang::Array<float> i_close(i_close_->shape());
  int fill_lookback = 1000;
  int offset = std::max(0, start_di - fill_lookback);
  for (int ii = 0; ii < (int)env.univ_size(); ii++) {
    yang::math::ops::ffill(i_close_->col_vec(ii).begin() + offset, i_close_->col_vec(ii).end(),
                           i_close.col_vec(ii).begin() + offset, fill_lookback);
  }

  for (int di = start_di; di < end_di; di++) {
    std::vector<int64_t> iis_pos = {};
    std::vector<int64_t> iis_neg = {};
    float sum_var_pos = 0.;
    float sum_fix_pos = 0.;
    float sum_var_neg = 0.;
    float sum_fix_neg = 0.;
    for (int ii = 0; ii < env.univ_size(); ii++) {
      float val_p = (di == 0) ? 0 : sig(di - 1, ii);
      float shares_p = (di == 0) ? 0 : sig(di - 1, ii) / i_close(di - 1, ii);
      float val_c = sig(di, ii);
      float shares_c = sig(di, ii) / i_close(di, ii);

      if (!std::isfinite(val_p)) {
        val_p = 0;
        shares_p = 0;
      }
      if (!std::isfinite(val_c)) {
        val_c = 0;
        shares_c = 0;
      }

      if (halt(di, ii) || limit_up_flag(di, ii) || limit_down_flag(di, ii)) {
        shares_c = shares_p;
        val_c = shares_c * i_close(di, ii);
        sig(di, ii) = val_c;
        if (val_c > 0.) {
          sum_fix_pos += val_c;
        } else if (val_c < 0.) {
          sum_fix_neg += val_c;
        }
      } else {
        if (val_c > 0.) {
          sum_var_pos += val_c;
          iis_pos.push_back(ii);
        } else if (val_c < 0.) {
          sum_var_neg += val_c;
          iis_neg.push_back(ii);
        }
      }
    }
    float ratio_pos = (yang::DISPLAY_BOOK_SIZE - sum_fix_pos) / sum_var_pos;
    if (ratio_pos > 0.) {
      for (auto ii : iis_pos) {
        if (sig(di, ii) > 0.) {
          sig(di, ii) *= ratio_pos;
        }
      }
    }
    float ratio_neg = (-yang::DISPLAY_BOOK_SIZE - sum_fix_neg) / sum_var_neg;
    if (ratio_neg) {
      for (auto ii : iis_neg) {
        if (sig(di, ii) < 0.) {
          sig(di, ii) *= ratio_neg;
        }
      }
    }
  }
  double scale_ratio = booksize / yang::DISPLAY_BOOK_SIZE;
  for (int di = start_di; di < end_di; di++) {
    for (int ii = 0; ii < env.univ_size(); ii++) {
      sig(di, ii) = double(std::round(sig(di, ii) * scale_ratio / i_close(di, ii) / 100) * 100) *
                    i_close(di, ii) / scale_ratio;
    }
  }
}

REGISTER_OPERATION("limit", OpLimit);

}  // namespace yao
