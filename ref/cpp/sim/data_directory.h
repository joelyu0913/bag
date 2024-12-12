#pragma once

#include <string>
#include <string_view>

#include "yang/util/fs.h"

namespace yang {

class DataDirectory {
 public:
  void Initialize(const std::string &user_dir) {
    Initialize(user_dir, user_dir);
  }

  void Initialize(const std::string &user_dir, const std::string &sys_dir);

  std::string_view user_dir() const {
    return user_dir_.native();
  }

  std::string_view sys_dir() const {
    return sys_dir_.native();
  }

  std::string GetPath(std::string_view mod) const {
    return GetReadPath(mod);
  }

  std::string GetPath(std::string_view mod, std::string_view data) const {
    return GetReadPath(mod, data);
  }

  std::string GetReadPath(std::string_view mod) const {
    auto user_path = user_dir_ / mod;
    if (fs::exists(user_path)) return user_path;
    auto sys_path = sys_dir_ / mod;
    if (TestSysPath(sys_path)) return sys_path;
    return user_path;
  }

  std::string GetReadPath(std::string_view mod, std::string_view data) const {
    auto user_path = user_dir_ / mod / data;
    if (fs::exists(user_path)) return user_path;
    auto sys_path = sys_dir_ / mod / data;
    if (TestSysPath(sys_path)) return sys_path;
    return user_path;
  }

  std::string GetWritePath(std::string_view mod) const {
    return user_dir_ / mod;
  }

  std::string GetWritePath(std::string_view mod, std::string_view data) const {
    return user_dir_ / mod / data;
  }

 private:
  fs::path user_dir_;
  fs::path sys_dir_;

  static bool TestSysPath(const std::string &p) {
    return fs::exists(p) || fs::exists(p + ".meta") || fs::exists(p + ".id");
  }
};

}  // namespace yang
