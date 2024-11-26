#pragma once

#include <string>

namespace yang {

struct ModuleLoader {
  static void *Load(const std::string &path);
  static void Unload(void *handle);
};

}  // namespace yang
