#include "yao/base_daily.h"

namespace {
using namespace yao;
using namespace std;
class D_kdr : public BaseDaily {
 public:
  void user_proc() {
    base_load({"sig"});
    init();
    cal();
  }

 private:
  int par_k;
  void init() {
    par_k = config<int>("par_k");
  }

  void cal() {
    LoadBaseData();
    LoadIBaseData();
    for (int di = start_di; di < end_di; di++) {
      if (di > par_k) {
        REP(ii, univ_sz) {
          if (b_b["univ_all"]->Get(di, ii)) {
            if (!IsValid(b_f["cumadj"]->Get(di, ii))) {
              LOG_INFO("{} {} {}", dates()[di], univ()[ii], b_f["cumadj"]->Get(di, ii));
            }
            b_sig(di, ii) = -(b_f["close"]->Get(di, ii) / b_f["close"]->Get(di - par_k, ii) *
                                  b_f["cumadj"]->Get(di, ii) / b_f["cumadj"]->Get(di - par_k, ii) -
                              1);
          } else {
            b_sig(di, ii) = NAN;
          }
        }
      }
    }

    if (false) {
      for (int di = start_di; di < end_di; di++) {
        REP(ii, univ_sz) {
          if (b_univ(di, ii)) {
            if (my_valid(b_f["close"]->Get(di, ii)) && my_valid(b_f["i_close"]->Get(di, ii)) &&
                abs(b_f["close"]->Get(di, ii) - b_f["i_close"]->Get(di, ii)) > 1e-10) {
              LOG_INFO("{},{},{},{}", di, ii, b_f["close"]->Get(di, ii),
                       b_f["i_close"]->Get(di, ii));
            }
          }
        }
      }
    }
  }
};
REGISTER_MODULE(D_kdr);
}  // namespace
