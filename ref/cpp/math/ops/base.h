#pragma once

#include <cmath>
#include <type_traits>

#include "yang/base/inline.h"
#include "yang/base/type_traits.h"

namespace yang::math::ops {

namespace detail {

template <class T>
constexpr T default_value() {
  if constexpr (std::is_floating_point_v<T>) return NAN;
  return T{};
}

template <class T>
constexpr iter_value_t<T> default_iter_value() {
  return default_value<iter_value_t<T>>();
}

}  // namespace detail

constexpr double EPSILON = 1e-6;

template <class T>
constexpr bool is_valid_check_v = std::is_invocable_v<T, float>;

struct CheckNothing {
  template <class T>
  ALWAYS_INLINE constexpr bool operator()(const T &v) const {
    return true;
  }
};

struct CheckFinite {
  template <class T>
  ALWAYS_INLINE bool operator()(const T &v) const {
    if constexpr (std::is_floating_point_v<T>) {
      return std::isfinite(v);
    } else {
      return true;
    }
  }
};

struct CheckBool {
  template <class T>
  ALWAYS_INLINE bool operator()(const T &v) const {
    return static_cast<bool>(v);
  }
};

using DefaultCheck = CheckFinite;

template <bool NanEq = false, class T1, class T2>
bool float_eq(T1 x, T2 y, decltype(T1{} - T2{}) eps = 0) {
  if constexpr (NanEq) {
    if (std::isnan(x) && std::isnan(y)) return true;
  }
  return std::abs(x - y) <= eps;
}

template <bool NanEq = false, class T1, class T2>
bool float_eq_pct(T1 x, T2 y, decltype(T1{} - T2{}) eps = 0) {
  if constexpr (NanEq) {
    if (std::isnan(x) && std::isnan(y)) return true;
  }
  return x == y || std::abs(x - y) / (std::abs(x) + std::abs(y)) <= eps;
}

template <bool CheckNan = true, class T, std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
T sign(T v) {
  if constexpr (CheckNan && std::is_floating_point_v<T>) {
    if (std::isnan(v)) return NAN;
  }
  return v > 0 ? 1 : (v < 0 ? -1 : 0);
}

namespace detail {

template <class T, class E, std::enable_if_t<std::is_integral_v<E>, int> = 0>
auto pow(T base, E exp) -> decltype(std::pow(base, exp)) {
  using R = decltype(std::pow(base, exp));

  if (exp < 0) return 1 / pow(static_cast<R>(base), -exp);

  int sign = (base < 0 && (exp & 1)) ? -1 : 1;
  R ret = 1;
  R mul = std::abs(base);
  while (exp > 0) {
    if (exp & 1) ret *= mul;
    if (exp > 1) mul *= mul;
    exp /= 2;
  }
  return ret * sign;
}

template <class T, class E, std::enable_if_t<std::is_floating_point_v<E>, int> = 0>
auto pow(T base, E exp) {
  return std::pow(base, exp);
}

};  // namespace detail

}  // namespace yang::math::ops
