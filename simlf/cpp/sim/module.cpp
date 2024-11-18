#include "yang/sim/module.h"

namespace yang {

void Module::Initialize(const std::string &name, const Config &config, const Env *env) {
  name_ = name;
  env_ = env;
  config_ = config;
}

void Module::Run() {
  BeforeRun();
  RunImpl();
  AfterRun();
}

}  // namespace yang
