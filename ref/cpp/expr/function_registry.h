#pragma once

#include <memory>
#include <string>
#include <string_view>

#include "yang/expr/function.h"
#include "yang/util/unordered_map.h"

namespace yang::expr {

class FunctionRegistry {
 public:
  static const FunctionRegistry &instance() {
    static FunctionRegistry inst;
    return inst;
  }

  static Function *Get(const std::string_view name) {
    auto it = instance().functions_.find(name);
    if (it != instance().functions_.end()) {
      return it->second.get();
    }
    std::string lower_name(name);
    std::for_each(lower_name.begin(), lower_name.end(), [](auto &c) { c = std::tolower(c); });
    it = instance().functions_.find(lower_name);
    if (it != instance().functions_.end()) {
      return it->second.get();
    }
    return nullptr;
  }

 private:
  FunctionRegistry();
  FunctionRegistry(const FunctionRegistry &other) = delete;
  FunctionRegistry &operator=(const FunctionRegistry &other) = delete;

  unordered_map<std::string, std::unique_ptr<Function>> functions_;
};

}  // namespace yang::expr
