#include "yang/sim/data_directory.h"

namespace yang {

void DataDirectory::Initialize(const std::string &user_dir, const std::string &sys_dir) {
  user_dir_ = user_dir;
  sys_dir_ = sys_dir;
  if (!fs::exists(user_dir_)) fs::create_directories(user_dir_);
}

}  // namespace yang
