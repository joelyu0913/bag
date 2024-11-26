#pragma once

#include <span>

#include "yang/expr/base.h"
#include "yang/math/vec_view.h"
#include "yang/util/logging.h"

namespace yang::expr {

template <class T = Float>
using VecView = yang::math::VecView<const T>;

template <class T = Float>
using OutVecView = yang::math::VecView<T>;

template <class T>
class VecBuffer {
 public:
  VecBuffer() : capacity_(0), head_(0), size_(0) {}

  VecBuffer(int capacity, int vec_size)
      : capacity_(capacity), head_(0), size_(0), vec_size_(vec_size), data_(capacity * vec_size) {}

  VecBuffer(VecBuffer &&other) {
    this->operator=(std::move(other));
  }

  VecBuffer &operator=(VecBuffer &&other) {
    capacity_ = other.capacity_;
    head_ = other.head_;
    size_ = other.size_;
    vec_size_ = other.vec_size_;
    data_ = std::move(other.data_);
    return *this;
  }

  int capacity() const {
    return capacity_;
  }

  int size() const {
    return size_;
  }

  bool empty() const {
    return size() == 0;
  }

  int vec_size() const {
    return vec_size_;
  }

  VecView<T> operator[](int i) const {
    auto ptr = Get(i);
    return VecView<T>(ptr, vec_size_);
  }

  void PushBack(VecView<T> data) {
    ENSURE(data.size() == vec_size_, "Invalid data vec size: {}, expected: {}", data.size(),
           vec_size_);

    if (size_ == capacity_) PopFront();
    auto ptr = Get(size_);
    for (int i = 0; i < vec_size_; ++i) ptr[i] = data[i];
    ++size_;
  }

  void PushBack(T value) {
    if (size_ == capacity_) PopFront();
    auto ptr = Get(size_);
    for (int i = 0; i < vec_size_; ++i) ptr[i] = value;
    ++size_;
  }

  void PopFront() {
    if (size_ == 0) return;
    ++head_;
    if (head_ == capacity_) head_ = 0;
    --size_;
  }

  void PopBack() {
    if (size_ == 0) return;
    --size_;
  }

  VecView<T> front() const {
    return (*this)[0];
  }

  VecView<T> back() const {
    return (*this)[size_ - 1];
  }

  OutVecView<T> writable_back() {
    auto ptr = Get(size_ - 1);
    return OutVecView<T>(ptr, vec_size_);
  }

 private:
  int capacity_;
  int head_;
  int size_;
  int vec_size_;
  std::vector<T> data_;

  const T *Get(int i) const {
    return &data_[GetOffset(i) * vec_size_];
  }

  T *Get(int i) {
    return &data_[GetOffset(i) * vec_size_];
  }

  int GetOffset(int i) const {
    int off = head_ + i;
    return off >= capacity_ ? off - capacity_ : off;
  }
};

template <class T>
class VecBufferSpan {
 public:
  VecBufferSpan() {}

  VecBufferSpan(const VecBuffer<T> &buf, int start, int size)
      : buf_(&buf), start_(start), size_(size) {}

  int size() const {
    return size_;
  }

  bool empty() const {
    return size() == 0;
  }

  int vec_size() const {
    return buf_->vec_size();
  }

  VecView<T> operator[](int i) const {
    return (*buf_)[start_ + i];
  }

  VecView<T> front() const {
    return (*buf_)[start_];
  }

  VecView<T> back() const {
    return (*buf_)[start_ + size_ - 1];
  }

 private:
  const VecBuffer<T> *buf_ = nullptr;
  int start_;
  int size_;
};

}  // namespace yang::expr
