#pragma once

#include <array>
#include <cmath>
#include <limits>
#include <type_traits>

namespace yang {
namespace detail {

template <class T>
struct NullValue {
  static constexpr T Get() {
    if constexpr (std::is_floating_point_v<T>) {
      return NAN;
    }
    if constexpr (std::is_same_v<T, bool>) {
      return false;
    }
    if constexpr (std::is_integral_v<T>) {
      if constexpr (std::is_signed_v<T>) {
        return std::numeric_limits<T>::min();
      } else {
        return std::numeric_limits<T>::max();
      }
    }
    return T{};
  }
};

template <class T, size_t N>
struct NullValue<std::array<T, N>> {
  static constexpr std::array<T, N> Get() {
    std::array<T, N> arr;
    for (size_t i = 0; i < N; ++i) arr[i] = NullValue<T>::Get();
    return arr;
  }
};

}  // namespace detail

template <class T>
constexpr T GetNullValue() {
  return detail::NullValue<T>::Get();
}

template <class T>
constexpr T null_v = GetNullValue<T>();

template <class T>
constexpr bool IsNull(const T &v) {
  if constexpr (std::is_floating_point_v<T>) {
    return std::isnan(v);
  }
  return v == GetNullValue<T>();
}

}  // namespace yang
