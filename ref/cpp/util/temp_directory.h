#pragma once

#include <unistd.h>

#include <cstdio>
#include <string>

#include "yang/util/fs.h"
#include "yang/util/logging.h"

namespace yang {

class TempDirectory {
 public:
  TempDirectory(const std::string &prefix, bool system_temp = true) {
    path_ = prefix + ".XXXXXX";
    if (system_temp) {
      path_ = (fs::temp_directory_path() / path_).string();
    }
    ENSURE2(mkdtemp(path_.data()) != nullptr);
  }

  TempDirectory(TempDirectory &&other) {
    this->operator=(std::move(other));
  }

  TempDirectory &operator=(TempDirectory &&other) {
    Destroy();
    path_ = std::move(other.path_);
    other.path_ = "";
    return *this;
  }

  ~TempDirectory() {
    Destroy();
  }

  const std::string &path() const {
    return path_;
  }

  std::string subpath(const std::string &name) const {
    auto p = fs::path(path_) / name;
    return p.string();
  }

 private:
  std::string path_;

  TempDirectory() {}

  void Destroy() {
    if (!path_.empty()) {
      fs::remove_all(path_);
    }
  }
};

}  // namespace yang
