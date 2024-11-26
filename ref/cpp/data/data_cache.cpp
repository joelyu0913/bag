#include "yang/data/data_cache.h"

namespace yang {

void DataCache::Clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  data_dict_.clear();
}

}  // namespace yang
