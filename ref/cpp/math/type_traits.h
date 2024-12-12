#pragma once

#include <type_traits>

#include "yang/base/type_traits.h"

namespace yang::math {

template <class T, class = void>
struct is_allocator : std::false_type {};

template <class T>
struct is_allocator<T, void_t<decltype(std::declval<T>().allocate(0))>> : std::true_type {};

template <class T>
constexpr bool is_allocator_v = is_allocator<T>::value;

template <class T, class = void>
struct is_mat_view : std::false_type {};

template <class T>
struct is_mat_view<T, void_t<decltype(std::declval<T>().row(0)), decltype(std::declval<T>()(0, 0))>>
    : std::true_type {};

template <class T>
constexpr bool is_mat_view_v = is_mat_view<T>::value;

}  // namespace yang::math
