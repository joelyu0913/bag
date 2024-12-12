#pragma once

#include <cstdint>
#include <string_view>
#include <type_traits>

namespace yang {

template <class T>
consteval std::string_view GetTypeName() {
  if constexpr (std::is_same_v<T, int8_t>) return "int8";
  if constexpr (std::is_same_v<T, int16_t>) return "int16";
  if constexpr (std::is_same_v<T, int32_t>) return "int32";
  if constexpr (std::is_same_v<T, int64_t>) return "int64";
  if constexpr (std::is_same_v<T, uint8_t>) return "uint8";
  if constexpr (std::is_same_v<T, uint16_t>) return "uint16";
  if constexpr (std::is_same_v<T, uint32_t>) return "uint32";
  if constexpr (std::is_same_v<T, uint64_t>) return "uint64";
  if constexpr (std::is_same_v<T, float>) return "float";
  if constexpr (std::is_same_v<T, double>) return "double";

  std::string_view full = __PRETTY_FUNCTION__;
#ifdef __clang__
  auto start = full.find("[T = ") + 5;
  auto end = full.find(']', start);
#else
  auto start = full.find("with T = ") + 9;
  auto end = full.find(';', start);
#endif
  return full.substr(start, end - start);
}

}  // namespace yang
