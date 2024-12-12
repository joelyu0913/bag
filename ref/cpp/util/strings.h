#pragma once

#include <cmath>
#include <string_view>
#include <type_traits>

#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "yang/base/exception.h"
#include "yang/base/likely.h"

namespace yang {

template <class... Args>
auto StrSplit(Args &&...args) {
  return absl::StrSplit(std::forward<Args>(args)...);
}

template <class Int>
bool SafeAtoi(std::string_view str, Int &out) {
  return absl::SimpleAtoi(str, &out);
}

inline bool SafeAtof(std::string_view str, float &out) {
  return absl::SimpleAtof(str, &out);
}

inline bool SafeAtod(std::string_view str, double &out) {
  return absl::SimpleAtod(str, &out);
}

inline bool SafeAtob(std::string_view str, bool &out) {
  return absl::SimpleAtob(str, &out);
}

template <class Int = int>
Int Atoi(std::string_view str, Int fallback = Int{}) {
  Int out;
  return SafeAtoi(str, out) ? out : fallback;
}

inline float Atof(std::string_view str, float fallback = NAN) {
  float out;
  return SafeAtof(str, out) ? out : fallback;
}

inline double Atod(std::string_view str, double fallback = NAN) {
  double out;
  return SafeAtod(str, out) ? out : fallback;
}

inline bool Atob(std::string_view str, bool fallback = false) {
  bool out;
  return SafeAtob(str, out) ? out : fallback;
}

// Convert str to arithmetic type T. Returns a fallback value if the conversion
// failed.
template <class T, class... Args>
T StrConv(std::string_view str, Args &&...args) {
  static_assert(std::is_integral_v<T> || std::is_floating_point_v<T> || std::is_same_v<T, bool>);
  if constexpr (std::is_integral_v<T>)
    return Atoi<T>(str, std::forward<Args>(args)...);
  else if constexpr (std::is_same_v<T, float>)
    return Atof(str, std::forward<Args>(args)...);
  else if constexpr (std::is_same_v<T, double>)
    return Atod(str, std::forward<Args>(args)...);
  else if constexpr (std::is_same_v<T, bool>)
    return Atob(str, std::forward<Args>(args)...);
}

struct ConvError : Exception {
  using Exception::Exception;
};

template <class Int = int>
Int CheckAtoi(std::string_view str) {
  Int out;
  if UNLIKELY (!SafeAtoi(str, out)) throw MakeExcept<ConvError>("Atoi failed: {}", str);
  return out;
}

inline float CheckAtof(std::string_view str) {
  float out;
  if UNLIKELY (!SafeAtof(str, out)) throw MakeExcept<ConvError>("Atof failed: {}", str);
  return out;
}

inline double CheckAtod(std::string_view str) {
  double out;
  if UNLIKELY (!SafeAtod(str, out)) throw MakeExcept<ConvError>("Atod failed: {}", str);
  return out;
}

inline double CheckAtob(std::string_view str) {
  bool out;
  if UNLIKELY (!SafeAtob(str, out)) throw MakeExcept<ConvError>("Atob failed: {}", str);
  return out;
}

}  // namespace yang
