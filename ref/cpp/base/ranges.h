#pragma once

#ifdef __APPLE__
#include <range/v3/view.hpp>
#else
#include <ranges>
#endif
#include <type_traits>

namespace yang {
#ifdef __APPLE__
namespace ranges = ::ranges;
#else
namespace ranges = std::ranges;
#endif

template <class T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
auto range(T end) {
  return ranges::views::iota(static_cast<T>(0), end);
}

template <class T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
auto range(T start, T end) {
  return ranges::views::iota(start, std::max(start, end));
}

}  // namespace yang
