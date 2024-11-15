#include "yao/base_daily.h"

namespace {
using namespace yao;
using namespace std;
class D_kdr_c : public BaseDaily {
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
    auto close = *ReadArray<float>("base", "close");
    auto univ_all = *ReadArray<bool>("base", "univ_all");

    for (int di = start_di; di < end_di; di++) {
      if (di > par_k) {
        REP(ii, univ_sz) {
          if (univ_all(di, ii)) {
            b_sig(di, ii) = -(close(di, ii) / close(di - par_k, ii) - 1);
          } else {
            b_sig(di, ii) = NAN;
          }
        }
      }
    }
  }
};
REGISTER_MODULE(D_kdr_c);
}  // namespace
