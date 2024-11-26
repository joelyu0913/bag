#pragma once

#ifdef __APPLE__
#include <range/v3/iterator/access.hpp>
#endif

#include <iterator>
#include <type_traits>

namespace yang {

template <class...>
using void_t = void;

template <class T, class = void>
struct is_iterator : std::false_type {};

template <class T>
struct is_iterator<T, void_t<decltype(++std::declval<T>()), decltype(*std::declval<T>())>>
    : std::true_type {};

template <class T>
constexpr bool is_iterator_v = is_iterator<T>::value;

template <class T, class = void>
struct is_iterable : std::false_type {};

template <class T>
struct is_iterable<
    T, void_t<decltype(std::begin(std::declval<T>())), decltype(std::end(std::declval<T>()))>>
    : std::true_type {};

template <class T>
constexpr bool is_iterable_v = is_iterable<T>::value;

template <class T, class = void>
struct is_random_access_iterator : std::false_type {};

template <class T>
struct is_random_access_iterator<T, void_t<typename std::iterator_traits<T>::iterator_category>>
    : std::is_same<typename std::iterator_traits<T>::iterator_category,
                   std::random_access_iterator_tag> {};

template <class T>
constexpr bool is_random_access_iterator_v = is_random_access_iterator<T>::value;

template <class T>
#ifdef __APPLE__
using iter_value_t = ranges::iter_value_t<T>;
#else
using iter_value_t = std::iter_value_t<T>;
#endif

}  // namespace yang
