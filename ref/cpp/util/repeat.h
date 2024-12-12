#pragma once

#include <array>

namespace yang {

template <class T>
class RepeatIterator {
 public:
  using ValueType = const T;
  using PointerType = const T *;

  RepeatIterator(PointerType data, int size, int i) : data_(data), size_(size), idx_(i) {}

  template <size_t N>
  RepeatIterator(const std::array<T, N> &array) : data_(array.data()), size_(N), idx_(0) {}

  PointerType operator->() const {
    return &data_[idx_];
  }

  ValueType &operator*() const {
    return data_[idx_];
  }

  ValueType &operator[](int idx) const {
    auto it = *this + idx;
    return *it;
  }

  RepeatIterator &operator++() {
    ++idx_;
    if (idx_ >= size_) idx_ -= size_;
    return *this;
  }

  RepeatIterator operator++(int) {
    RepeatIterator tmp(this);
    ++*this;
    return tmp;
  }

  RepeatIterator &operator--() {
    --idx_;
    if (idx_ < 0) idx_ += size_;
    return *this;
  }

  RepeatIterator operator--(int) {
    RepeatIterator tmp(this);
    --*this;
    return tmp;
  }

  bool operator==(const RepeatIterator &other) const {
    return data_ == other.data_ && size_ = other.size_ && idx_ == other.idx_;
  }

  bool operator!=(const RepeatIterator &other) const {
    return !(*this == other);
  }

  bool operator>(const RepeatIterator &other) const {
    return idx_ > other.idx_;
  }

  bool operator<(const RepeatIterator &other) const {
    return idx_ < other.idx_;
  }

  bool operator>=(const RepeatIterator &other) const {
    return idx_ >= other.idx_;
  }

  bool operator<=(const RepeatIterator &other) const {
    return idx_ <= other.idx_;
  }

  RepeatIterator operator+(int offset) const {
    RepeatIterator it(data_, size_, idx_);
    it += offset;
    return it;
  }

  RepeatIterator operator-(int offset) const {
    return *this + (-offset);
  }

  int operator-(const RepeatIterator &other) const {
    return idx_ - other.idx_;
  }

  RepeatIterator &operator+=(int offset) {
    idx_ += offset;
    idx_ %= size_;
    if (idx_ < 0) idx_ += size_;
    return *this;
  }

  RepeatIterator &operator-=(int offset) {
    return *this += -offset;
  }

 private:
  PointerType data_;
  int size_;
  int idx_;
};

}  // namespace yang
