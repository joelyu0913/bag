#pragma once

#include <iterator>

#include "yang/math/size.h"

namespace yang::math {

template <class T, StaticSize StrideSize = DYNAMIC_STRIDE>
class VecIterator {
 public:
  static constexpr StaticSize Stride = StrideSize;
  using value_type = T;
  using pointer = T *;
  using reference = T &;
  using difference_type = IndexType;
  using iterator_category = std::random_access_iterator_tag;

  VecIterator() : ptr_(nullptr), stride_(0) {}
  VecIterator(T *ptr, SizeType stride) : ptr_(ptr), stride_(stride) {}
  VecIterator(const VecIterator &other) = default;
  VecIterator &operator=(const VecIterator &other) = default;

  SizeType stride() const {
    if constexpr (StrideSize == DYNAMIC_STRIDE) return stride_;
    return StrideSize;
  }

  T *operator->() const {
    return ptr_;
  }

  T &operator*() const {
    return *ptr_;
  }

  T &operator[](IndexType idx) const {
    return ptr_[idx * stride_];
  }

  VecIterator &operator++() {
    ptr_ += stride();
    return *this;
  }

  VecIterator operator++(int) {
    VecIterator tmp(this);
    ++*this;
    return tmp;
  }

  VecIterator &operator--() {
    ptr_ -= stride();
    return *this;
  }

  VecIterator operator--(int) {
    VecIterator tmp(this);
    --*this;
    return tmp;
  }

  bool operator==(const VecIterator &other) const {
    return ptr_ == other.ptr_ && stride_ == other.stride_;
  }

  bool operator!=(const VecIterator &other) const {
    return !(*this == other);
  }

  bool operator>(const VecIterator &other) const {
    return ptr_ > other.ptr_;
  }

  bool operator<(const VecIterator &other) const {
    return ptr_ < other.ptr_;
  }

  bool operator>=(const VecIterator &other) const {
    return ptr_ >= other.ptr_;
  }

  bool operator<=(const VecIterator &other) const {
    return ptr_ <= other.ptr_;
  }

  VecIterator operator+(IndexType offset) const {
    return VecIterator(ptr_ + offset * stride_, stride_);
  }

  VecIterator operator-(IndexType offset) const {
    return *this + (-offset);
  }

  IndexType operator-(const VecIterator &other) const {
    return (ptr_ - other.ptr_) / stride_;
  }

  VecIterator &operator+=(IndexType offset) {
    ptr_ += offset * stride_;
    return *this;
  }

  VecIterator &operator-=(IndexType offset) {
    ptr_ -= offset * stride_;
    return *this;
  }

 private:
  T *ptr_;
  SizeType stride_;
};

template <class T, StaticSize StrideSize>
inline VecIterator<T, StrideSize> operator+(IndexType offset, VecIterator<T, StrideSize> it) {
  return it + offset;
}

}  // namespace yang::math
