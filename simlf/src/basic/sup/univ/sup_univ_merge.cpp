#include "yang/io/open.h"
#include "yang/util/dates.h"
#include "yao/base_daily.h"

namespace {
using namespace yao;
using namespace std;
class SupUnivMerge : public BaseDaily {
 public:
  void user_proc() {
    cal();
  }

 private:
  void include(yang::Array<bool> &a, const yang::Array<bool> &b) {
    for (int di = start_di; di < end_di; di++) {
      for (int ii = 0; ii < max_univ_sz; ii++) {
        if (b(di, ii)) {
          a(di, ii) = true;
        }
      }
    }
  }

  void intersect(yang::Array<bool> &a, const yang::Array<bool> &b) {
    for (int di = start_di; di < end_di; di++) {
      for (int ii = 0; ii < max_univ_sz; ii++) {
        if (!b(di, ii)) {
          a(di, ii) = false;
        }
      }
    }
  }

  void exclude(yang::Array<bool> &a, const yang::Array<bool> &b) {
    for (int di = start_di; di < end_di; di++) {
      for (int ii = 0; ii < max_univ_sz; ii++) {
        if (b(di, ii)) {
          a(di, ii) = false;
        }
      }
    }
  }
  void cal() {
    auto univ = config<std::string>("par_univ");
    auto b_univ_custom = WriteArray<bool>(univ);
    auto vs_include = config<vector<string>>("par_include", {});
    auto vs_intersect = config<vector<string>>("par_intersect", {});
    auto vs_exclude = config<vector<string>>("par_exclude", {});
    // auto vs_intersect = my_split(config<string>("par_intersect", ""), '|');
    // auto vs_exclude = my_split(config<string>("par_exclude", ""), '|');
    for (int di = start_di; di < end_di; di++) {
      for (int ii = 0; ii < max_univ_sz; ii++) {
        b_univ_custom(di, ii) = false;
      }
    }

    for (int i = 0; i < (int)vs_include.size(); i++) {
      auto str_univ = vs_include[i];
      LOG_INFO("Include: {}", str_univ);
      if (str_univ != "") {
        auto &b_univ_tmp = *ReadArray<bool>(str_univ);
        include(b_univ_custom, b_univ_tmp);
      }
    }
    for (int i = 0; i < (int)vs_intersect.size(); i++) {
      auto str_univ = vs_intersect[i];
      LOG_INFO("Intersect: {}", str_univ);
      if (str_univ != "") {
        auto &b_univ_tmp = *ReadArray<bool>(str_univ);
        intersect(b_univ_custom, b_univ_tmp);
      }
    }
    for (int i = 0; i < (int)vs_exclude.size(); i++) {
      auto str_univ = vs_exclude[i];
      LOG_INFO("Exclude: {}", str_univ);
      if (str_univ != "") {
        auto &b_univ_tmp = *ReadArray<bool>(str_univ);
        exclude(b_univ_custom, b_univ_tmp);
      }
    }
  }
};
REGISTER_MODULE(SupUnivMerge);
}  // namespace
