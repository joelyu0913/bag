#pragma once

#include <algorithm>
#include <fstream>
#include <string>
#include <type_traits>
#include <vector>

#include "yang/base/size.h"
#include "yang/util/fs.h"
#include "yang/util/logging.h"
#include "yang/util/unordered_map.h"

namespace yang {

template <class T>
class IndexBase {
 public:
  IndexBase() {}

  IndexBase(IndexBase &&other) {
    this->operator=(std::move(other));
  }

  IndexBase &operator=(IndexBase &&other) {
    path_ = std::move(other.path_);
    items_ = std::move(other.items_);
    other.path_ = "";
    return *this;
  }

  const std::string &path() const {
    return path_;
  }

  void set_path(const std::string &path) {
    path_ = path;
  }

  bool empty() const {
    return items_.empty();
  }

  SizeType size() const {
    return items_.size();
  }

  const T &operator()(SizeType i) const {
    return Get(i);
  }

  const T &operator[](SizeType i) const {
    return Get(i);
  }

  const T &Get(SizeType i) const {
    return items_[i];
  }

  const std::vector<T> &items() const {
    return items_;
  }

  virtual void Save() {
    fs::create_directories(fs::path(path_).parent_path());
    std::ofstream ofs(path_);
    for (auto &x : items_) ofs << x << '\n';
    ofs.flush();
    ENSURE(ofs.good(), "Failed to save {}", path_);
  }

  auto begin() {
    return items_.begin();
  }

  auto end() {
    return items_.end();
  }

  auto begin() const {
    return items_.begin();
  }

  auto end() const {
    return items_.end();
  }

 protected:
  std::string path_;
  std::vector<T> items_;

  void InternalLoad(const std::string &path) {
    path_ = path;
    LoadImpl();
  }

  virtual void LoadImpl() {
    if (fs::exists(path_)) {
      std::ifstream ifs(path_);
      T v;
      if constexpr (std::is_same_v<T, std::string>) {
        while (std::getline(ifs, v)) {
          items_.emplace_back(std::move(v));
        }
      } else {
        while (ifs >> v) {
          items_.emplace_back(std::move(v));
        }
      }
    }
  }
};

template <class T>
class Index : public IndexBase<T> {
 public:
  Index() {}

  Index(Index &&other) {
    this->operator=(std::move(other));
  }

  Index &operator=(Index &&other) {
    index_ = std::move(other.index_);
    IndexBase<T>::operator=(std::move(other));
    return *this;
  }

  template <class K>
  int Find(const K &k) const {
    auto it = index_.find(k);
    if (it != index_.end()) {
      return it->second;
    } else {
      return -1;
    }
  }

  int Insert(const T &v) {
    return TryInsert(v).first;
  }

  std::pair<int, bool> TryInsert(const T &v) {
    auto [it, inserted] = index_.try_emplace(v, index_.size());
    if (inserted) {
      this->items_.push_back(v);
    }
    return {it->second, inserted};
  }

  void Shrink(SizeType size) {
    if (size >= this->size()) return;
    for (SizeType i = size; i < this->size(); ++i) {
      index_.erase(this->items_[i]);
    }
    this->items_.resize(size);
  }

  [[nodiscard]] static Index Load(const std::string &path, bool ensure_exists = false) {
    if (ensure_exists) {
      ENSURE(fs::exists(path), "Missing {}", path);
    }
    Index index;
    index.InternalLoad(path);
    return index;
  }

 protected:
  unordered_map<T, int> index_;

  void LoadImpl() override {
    IndexBase<T>::LoadImpl();
    for (int i = 0; i < this->size(); ++i) {
      index_.emplace(this->items_[i], i);
    }
  }
};

template <class T>
class OrderedIndex : public IndexBase<T> {
 public:
  static_assert(std::is_fundamental_v<T>);

  OrderedIndex() {}

  OrderedIndex(const std::vector<T> &items) {
    this->items_ = items;
  }

  int Find(const T &v) const {
    int ret = LowerBound(v);
    if (ret == this->size() || this->items_[ret] != v) {
      return -1;
    }
    return ret;
  }

  // Returns the index to the first element that is not less than v, or size() if not
  // found
  int LowerBound(const T &v, bool inclusive = true) const {
    auto it = std::lower_bound(this->items_.begin(), this->items_.end(), v);
    if (!inclusive && it != this->items_.end() && *it == v) --it;
    return it - this->items_.begin();
  }

  // Returns the index to the first element that is greater than v, or size() if not
  // found
  int UpperBound(const T &v, bool inclusive = false) const {
    auto it = std::upper_bound(this->items_.begin(), this->items_.end(), v);
    if (inclusive && it != this->items_.begin() && *(it - 1) == v) --it;
    return it - this->items_.begin();
  }

  int GreaterThan(const T &v) const {
    return UpperBound(v);
  }

  int GreaterEqualThan(const T &v) const {
    return LowerBound(v);
  }

  int LessThan(const T &v) const {
    int idx = LowerBound(v);
    if (idx == static_cast<int>(this->items_.size()) || this->items_[idx] >= v) {
      --idx;
    }
    return idx;
  }

  int LessEqualThan(const T &v) const {
    int idx = LowerBound(v);
    if (idx == static_cast<int>(this->items_.size()) || this->items_[idx] > v) {
      --idx;
    }
    return idx;
  }

  void PushBack(const T &v) {
    ENSURE(this->items_.empty() || v > this->items_.back(),
           "new value must be bigger, new: {}, last: {}", v, this->items_.back());
    this->items_.push_back(v);
  }

  void Shrink(SizeType size) {
    if (size < this->items_.size()) {
      this->items_.resize(size);
    }
  }

  [[nodiscard]] static OrderedIndex Load(const std::string &path, bool ensure_exists = false) {
    if (ensure_exists) {
      ENSURE(fs::exists(path), "Missing {}", path);
    }
    OrderedIndex index;
    index.InternalLoad(path);
    return index;
  }
};

}  // namespace yang
