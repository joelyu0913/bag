#pragma once

#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>

#include "yang/data/array.h"
#include "yang/data/data_loader.h"
#include "yang/util/logging.h"
#include "yang/util/type_name.h"
#include "yang/util/unordered_map.h"

namespace yang {

template <class T>
struct DataLoader<Array<T>> {
  static Array<T> Load(const std::string &path) {
    return Array<T>::MMap(path);
  }
};

class DataCache {
 public:
  DataCache(int max_size = 1024) : max_size_(max_size) {}

  int size() const {
    return data_list_.size();
  }

  int max_size() const {
    return max_size_;
  }

  template <class T>
  const T *GetOrLoad(std::string_view name, const std::string &path) {
    return GetOrLoad<T>(name, [&path]() { return path; });
  }

  template <class T, class Func, std::enable_if_t<std::is_invocable_v<Func>, bool> = true>
  const T *GetOrLoad(std::string_view name, Func &&f) {
    return GetOrMake<T>(name, [&f]() {
      static_assert(DataLoaderTraits<T>::HAS_LOAD || DataLoaderTraits<T>::HAS_LOAD_PTR);
      if constexpr (DataLoaderTraits<T>::HAS_LOAD) {
        return DataLoader<T>::Load(f());
      } else {
        return DataLoader<T>::LoadPtr(f());
      }
    });
  }

  template <class T, class Func>
  const T *GetOrMake(std::string_view name, Func &&f) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto *ptr = GetInternal<T>(name);
    if (ptr == nullptr) {
      auto holder = std::make_unique<DataHolderT<T>>(f());
      ptr = holder->ptr;
      data_list_.emplace_front(std::string(name), std::move(holder));
      data_dict_.emplace(name, data_list_.begin());
      // delete old entries
      while (static_cast<int>(data_list_.size()) > max_size_) {
        data_dict_.erase(data_list_.back().first);
        data_list_.pop_back();
      }
    }
    return ptr;
  }

  template <class T>
  const T *Get(std::string_view name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return GetInternal<T>(name);
  }

  void Clear();

 private:
  struct DataHolder {
    virtual ~DataHolder() {}
    virtual std::string_view data_type() const = 0;
  };

  template <class T>
  struct DataHolderT : DataHolder {
    T *ptr;

    DataHolderT() : ptr(new T) {}
    DataHolderT(T &&data) : ptr(new T(std::move(data))) {}
    DataHolderT(T *ptr) : ptr(ptr) {}

    DataHolderT(const DataHolderT &) = delete;
    DataHolderT &operator=(const DataHolderT &) = delete;

    std::string_view data_type() const override {
      return GetTypeName<T>();
    }

    ~DataHolderT() {
      if (ptr) delete ptr;
    }
  };

  using List = std::list<std::pair<std::string, std::unique_ptr<DataHolder>>>;

  int max_size_;
  mutable unordered_map<std::string, List::iterator> data_dict_;
  mutable List data_list_;
  mutable std::mutex mutex_;

  template <class T>
  const T *GetInternal(std::string_view name) const {
    auto it = data_dict_.find(name);
    if (it != data_dict_.end()) {
      auto &holder = it->second->second;
      ENSURE(holder->data_type() == GetTypeName<T>(),
             "Data type mismatched for {}, expected {}, got {}", name, GetTypeName<T>(),
             holder->data_type());
      auto ptr = static_cast<DataHolderT<T> *>(holder.get())->ptr;

      // move to front
      auto &list_it = it->second;
      if (list_it != data_list_.begin()) {
        auto name_v = std::move(list_it->first);
        auto holder_v = std::move(list_it->second);
        data_list_.erase(list_it);
        data_list_.emplace_front(std::string(name_v), std::move(holder_v));
        list_it = data_list_.begin();
      }

      return ptr;
    } else {
      return nullptr;
    }
  }
};

}  // namespace yang
