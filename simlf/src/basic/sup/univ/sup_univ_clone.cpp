#include "yao/base_daily.h"
using namespace yao;

namespace {
using namespace std;
class SupUnivClone : public yang::Module {
 public:
  void RunImpl() final {
    auto univ_to = WriteArray<bool>(config<string>("par_to"));
    auto &univ_from = *ReadArray<bool>(config<string>("par_from"));
    univ_to.CopyFrom(univ_from, start_di(), end_di());
  }
};

REGISTER_MODULE(SupUnivClone);
}  // namespace
