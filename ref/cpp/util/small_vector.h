#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <type_traits>

#include "yang/util/fmt.h"

namespace yang {

template <class T, int MaxSize = 8>
class small_vector {
  static_assert(std::is_trivially_copyable_v<T>);

 public:
  small_vector() : size_(0) {}

  small_vector(uint8_t size) : size_(size) {
    std::fill_n(begin(), size_, T{});
  }

  template <class U>
  small_vector(std::initializer_list<U> l) : size_(0) {
    for (auto &v : l) push_back(v);
  }

  small_vector(const small_vector &other) {
    this->operator=(other);
  }

  small_vector &operator=(const small_vector &other) {
    size_ = other.size_;
    std::copy(other.begin(), other.begin() + size(), begin());
    return *this;
  }

  bool operator==(const small_vector &other) const {
    if (size() != other.size()) return false;
    for (int i = 0; i < static_cast<int>(size()); ++i) {
      if ((*this)[i] != other[i]) return false;
    }
    return true;
  }

  int size() const {
    return size_;
  }

  bool empty() const {
    return size_ == 0;
  }

  int capacity() const {
    return MaxSize;
  }

  T &operator[](int i) {
    return begin()[i];
  }

  const T &operator[](int i) const {
    return begin()[i];
  }

  T *begin() {
    return reinterpret_cast<T *>(buf_);
  }

  T *end() {
    return begin() + size_;
  }

  const T *begin() const {
    return reinterpret_cast<const T *>(buf_);
  }

  const T *end() const {
    return begin() + size_;
  }

  T *data() {
    return begin();
  }

  const T *data() const {
    return begin();
  }

  void push_back(const T &v) {
    assert(size_ < capacity());
    begin()[size_] = v;
    ++size_;
  }

  void pop_back() {
    --size_;
  }

  void clear() {
    size_ = 0;
  }

  void resize(int new_size) {
    if (new_size > size_) {
      std::fill_n(begin() + size_, new_size - size_, T{});
    }
    size_ = new_size;
  }

  friend std::ostream &operator<<(std::ostream &os, const small_vector &v) {
    for (int i = 0; i < static_cast<int>(v.size()); ++i) {
      if (i) os << ' ';
      os << v[i];
    }
    return os;
  }

 private:
  uint8_t buf_[sizeof(T) * MaxSize];
  uint8_t size_;
};

}  // namespace yang

#ifdef FMT_OSTREAM_FORMATTER
template <class T, int MaxSize>
struct fmt::formatter<yang::small_vector<T, MaxSize>> : ostream_formatter {};
#endif
