#pragma once

#include "yang/base/valid.h"

namespace yang::expr {

using Float = double;

inline bool IsValid(Float f) {
  return ::yang::IsValid(f);
}

}  // namespace yang::expr
