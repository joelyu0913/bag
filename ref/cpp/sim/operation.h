#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "yang/math/mat_view.h"
#include "yang/sim/env.h"
#include "yang/util/config.h"
#include "yang/util/factory_registry.h"

namespace yang {

struct Operation {
  template <class T>
  using MatView = math::MatView<T>;

  virtual ~Operation() {}

  virtual bool lookback() const {
    return false;
  }

  virtual void Apply(MatView<float> sig, const Env &env, int start_di, int end_di,
                     const std::vector<std::string> &args,
                     const std::unordered_map<std::string, std::string> &kwargs) {
    Apply(sig, env, start_di, end_di, args);
  }

  virtual void Apply(MatView<float> sig, const Env &env, int start_di, int end_di,
                     const std::vector<std::string> &args) {
    Apply(sig, env, start_di, end_di);
  }

  virtual void Apply(MatView<float> sig, const Env &env, int start_di, int end_di) {
    throw FatalError("Unimplemented");
  }
};

#define REGISTER_OPERATION(name, cls) REGISTER_FACTORY("operation", cls, cls, name)

}  // namespace yang
