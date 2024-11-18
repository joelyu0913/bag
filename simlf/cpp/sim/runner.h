#pragma once

#include <memory>
#include <set>

#include "yang/sim/env.h"

namespace yang {

struct RunnerOptions {
  std::set<RunStage> stages;
  bool live = false;
  bool prod = false;
};

class Runner {
 public:
  void Run(RunnerOptions options, const Config &config, int num_threads = 1);
};

}  // namespace yang
