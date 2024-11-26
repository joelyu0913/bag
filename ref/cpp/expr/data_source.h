#pragma once

#include <string_view>

#include "yang/expr/vec.h"

namespace yang::expr {

struct DataSource {
  virtual int vec_size() const = 0;
  virtual VecView<Float> GetData(std::string_view name) const = 0;
  virtual VecView<int> GetGroup(std::string_view name) const = 0;
  virtual VecView<bool> GetMask() const = 0;
};

}  // namespace yang::expr
