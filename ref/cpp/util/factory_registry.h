#pragma once

#include <functional>
#include <iostream>
#include <string>
#include <unordered_map>

#include "yang/util/logging.h"

namespace yang {

using FactoryStorage =
    std::unordered_map<std::string, std::unordered_map<std::string, std::function<void *()>>>;

namespace detail {

// https://isocpp.org/wiki/faq/ctors#static-init-order-on-first-use
FactoryStorage *GetFactoryStorage();

}  // namespace detail

class FactoryRegistry {
 public:
  template <class T>
  static T *TryMake(const std::string &type, const std::string &name) {
    auto &registry = (*detail::GetFactoryStorage())[type];
    auto it = registry.find(name);
    if (it == registry.end()) {
      return nullptr;
    }
    return reinterpret_cast<T *>(it->second());
  }

  template <class T>
  static T *Make(const std::string &type, const std::string &name) {
    auto *ptr = TryMake<T>(type, name);
    ENSURE(ptr != nullptr, "Unknown factory type: {} name: {}", type, name);
    return ptr;
  }

  template <class T>
  static void Register(const std::string &type, const std::string &name) {
    auto &registry = (*detail::GetFactoryStorage())[type];
    ENSURE(!registry.contains(name), "Duplicate factory type: {} name: {}", type, name);
    registry.emplace(name, []() { return new T(); });
  }
};

namespace detail {
template <class T, T /*unnamed*/>
struct AutoRegisterForceInit {};

template <class T>
class AutoRegister {
 public:
  struct AutoRegisterProxy {
    AutoRegisterProxy() {
      T::Register();
    }
    void f() {}
  };

  static AutoRegisterProxy __proxy;
  typedef AutoRegisterForceInit<AutoRegisterProxy &, __proxy> __proxy_dummy;
};

template <class T>
typename AutoRegister<T>::AutoRegisterProxy AutoRegister<T>::__proxy;
}  // namespace detail

}  // namespace yang

#define REGISTER_FACTORY(type, cls_name, cls, name)                                        \
  namespace detail {                                                                       \
  struct AutoRegister_##cls_name : ::yang::detail::AutoRegister<AutoRegister_##cls_name> { \
    static void Register() {                                                               \
      ::yang::FactoryRegistry::Register<cls>(type, name);                                  \
    }                                                                                      \
    static void Foo() {                                                                    \
      (void)__proxy;                                                                       \
    }                                                                                      \
  };                                                                                       \
  }
