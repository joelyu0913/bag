#pragma once

#include <functional>
#include <string>
#include <vector>

#include "yang/util/unordered_map.h"

namespace yang {

class Scheduler {
 public:
  using Id = std::string;
  using Func = std::function<void()>;

  void Run(int num_threads, const unordered_map<Id, std::vector<Id>> &dep_map,
           const std::vector<std::pair<Id, Func>> &tasks);
};

}  // namespace yang
