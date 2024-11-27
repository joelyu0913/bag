#include <cmath>

#include "yang/math/mat_ops.h"
#include "yang/sim/operation.h"
#include "yang/util/strings.h"

namespace {

struct OpRzrq : yang::Operation {
  bool lookback() const final {
    return true;
  }

  using yang::Operation::Apply;

  void Apply(MatView<float> sig, const yang::Env &env, int start_di, int end_di,
             const std::vector<std::string> &args) final {
    float booksize = env.trade_book_size();
    std::string par_index = "";
    int index_type = yang::CheckAtoi<int>(args.at(0));
    if (index_type == 0) {
      par_index = "csi500_member";
    }

    double short_ratio = 1.;
    if (args.size() > 1) short_ratio = yang::CheckAtof(args.at(1));
    double short_size = short_ratio * booksize;

    auto i_close = env.ReadData<yang::Array<float>>("ibase", "i_close")->mat_view();
    auto close = env.ReadData<yang::Array<float>>("base", "close")->mat_view();
    auto member = env.ReadData<yang::Array<float>>("sup_univ", par_index)->mat_view();

    double scale_ratio = booksize / yang::DISPLAY_BOOK_SIZE;
    for (int di = std::max(1, start_di); di < end_di; di++) {
      double sum = 0.;
      for (int ii = 0; ii < env.univ_size(); ii++) {
        if (sig(di, ii) < 0) {
          if (std::isfinite(member(di, ii))) {
            int shares = sig(di, ii) * scale_ratio / i_close(di, ii);
            int quota =
                (int)(short_size * (member(di, ii) / 100.) / close(di - 1, ii) / 100) * 100.;
            if (shares < 0 && shares < -quota) {
              sig(di, ii) = -quota / scale_ratio * i_close(di, ii);
            }
          } else {
            sig(di, ii) = 0.;
          }
        }
        if (std::isfinite(sig(di, ii))) sum += sig(di, ii);
      }
      sig(di, env.univ_size() + index_type) = -sum;
    }
  }
};

REGISTER_OPERATION("rzrq", OpRzrq);

}  // namespace
