#pragma once

#include <cmath>
#include <type_traits>

namespace yang {

template <class T, std::enable_if_t<std::is_floating_point_v<T>, int> = 0>
bool IsValid(T v) {
  return std::isfinite(v);
}

}  // namespace yang
