#pragma once

#include <string>
#include <type_traits>

#include "yang/base/type_traits.h"

namespace yang {

namespace data::detail {
template <class T, class RT = T, class = void>
struct has_load : std::false_type {};

template <class T, class RT>
struct has_load<T, RT, void_t<decltype(T::Load(""))>> : std::is_same<decltype(T::Load("")), RT> {};

template <class T, class RT = T, class = void>
struct has_load_ptr : std::false_type {};

template <class T, class RT>
struct has_load_ptr<T, RT, void_t<decltype(T::LoadPtr(""))>>
    : std::is_same<decltype(T::LoadPtr("")), RT *> {};

template <class T, class = void>
struct has_instance_load : std::false_type {};

template <class T>
struct has_instance_load<T, void_t<decltype(std::declval<T>().Load(""))>> : std::true_type {};
}  // namespace data::detail

template <class T>
struct DataLoader {
  template <class T1 = T, std::enable_if_t<data::detail::has_load<T1>::value, int> = 0>
  static T Load(const std::string &path) {
    return T::Load(path);
  }

  template <class T1 = T, std::enable_if_t<data::detail::has_instance_load<T1>::value, int> = 0>
  static T *LoadPtr(const std::string &path) {
    auto val = new T;
    val.Load(path);
    return val;
  }
};

template <class T>
class DataLoaderTraits {
 public:
  static constexpr bool HAS_LOAD = data::detail::has_load<DataLoader<T>, T>::value;
  static constexpr bool HAS_LOAD_PTR = data::detail::has_load_ptr<DataLoader<T>, T>::value;
};

}  // namespace yang
