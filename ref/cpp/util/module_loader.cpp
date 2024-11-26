#include "yang/util/module_loader.h"

#include <dlfcn.h>

#include <filesystem>
#include <mutex>
#include <unordered_map>

#include "yang/util/logging.h"

namespace yang {

static std::mutex loaded_mutex;
static std::unordered_map<std::string, void *> loaded_paths;

void *ModuleLoader::Load(const std::string &path) {
  {
    std::lock_guard guard(loaded_mutex);
    auto it = loaded_paths.find(path);
    if (it != loaded_paths.end()) return it->second;
  }
  void *handle = nullptr;
  if (std::filesystem::exists(path)) {
    handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (handle) {
      LOG_INFO("Loaded {}", path);
    } else {
      LOG_ERROR("Failed to load {}: {}", path, dlerror());
    }
  } else {
    LOG_ERROR("Missing module: {}", path);
  }
  {
    std::lock_guard guard(loaded_mutex);
    loaded_paths[path] = handle;
  }
  return handle;
}

void ModuleLoader::Unload(void *handle) {
  auto ret = dlclose(handle);
  if (ret != 0) {
    LOG_ERROR("Failed to unload: {}", dlerror());
  }
}

}  // namespace yang
