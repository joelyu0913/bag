#pragma once

#include <utility>

namespace yang::math::ops {

template <size_t N, size_t... Is>
auto add_to_index_sequence(std::index_sequence<Is...>) {
  return std::index_sequence<(Is + N)...>();
}

}  // namespace yang::math::ops
