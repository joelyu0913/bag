#pragma once

#include <array>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "yang/base/exception.h"
#include "yang/base/likely.h"
#include "yang/base/type_traits.h"
#include "yang/math/size.h"
#include "yang/math/vec_iterator.h"
#include "yang/util/fmt.h"
#include "yang/util/logging.h"

namespace yang::math {

template <class T, StaticSize StrideSize = 1>
class VecView {
 public:
  static constexpr StaticSize Stride = StrideSize;
  using value_type = T;
  using iterator = VecIterator<T, StrideSize>;
  using const_iterator = VecIterator<const T, StrideSize>;

  VecView() : ptr_(nullptr), size_(0), stride_(0) {}
  VecView(T *ptr, SizeType size, SizeType stride = 1) : ptr_(ptr), size_(size), stride_(stride) {}
  VecView(const VecView &other) = default;
  VecView &operator=(const VecView &other) = default;

  // implicit conversion from std::vector
  template <class U = T, std::enable_if_t<std::is_const_v<U>, int> = 0>
  VecView(const std::vector<std::remove_const_t<U>> &v) : VecView(v.data(), v.size()) {}

  // implicit conversion from std::vector
  template <class U = T, std::enable_if_t<!std::is_const_v<U>, int> = 0>
  VecView(std::vector<U> &v) : VecView(v.data(), v.size()) {}

  // implicit conversion from std::array
  template <size_t N, class U = T, std::enable_if_t<std::is_const_v<U>, int> = 0>
  VecView(const std::array<std::remove_const_t<U>, N> &v) : VecView(v.data(), v.size()) {}

  // implicit conversion from std::array
  template <size_t N, class U = T, std::enable_if_t<!std::is_const_v<U>, int> = 0>
  VecView(std::array<U, N> &v) : VecView(v.data(), v.size()) {}

  SizeType stride() const {
    if constexpr (StrideSize == DYNAMIC_STRIDE) return stride_;
    return StrideSize;
  }

  SizeType size() const {
    return size_;
  }

  bool empty() const {
    return size_ == 0;
  }

  T &operator[](IndexType idx) const {
    return *Get(idx);
  }

  T &operator()(IndexType idx) const {
    return *Get(idx);
  }

  T &at(IndexType idx) const {
    return *GetChecked(idx);
  }

  T *data() const {
    return ptr_;
  }

  T front() const {
    return (*this)[0];
  }

  T back() const {
    return (*this)[size_ - 1];
  }

  iterator begin() const {
    return iterator(ptr_, stride());
  }

  iterator end() const {
    return iterator(GetUnchecked(size_), stride());
  }

  const_iterator cbegin() const {
    return const_iterator(ptr_, stride());
  }

  const_iterator cend() const {
    return const_iterator(GetUnchecked(size_), stride());
  }

  template <class U>
  bool operator==(const U &other) const {
    if (size() != static_cast<SizeType>(other.size())) return false;
    for (SizeType i = 0; i < size(); ++i) {
      if ((*this)[i] != other[i]) return false;
    }
    return true;
  }

  operator VecView<const T, Stride>() const {
    return VecView<const T, Stride>(ptr_, size_, stride());
  }

  VecView slice(IndexType start) const {
    if UNLIKELY (start > size_ || start < 0)
      throw MakeExcept<OutOfRange>("Invalid start:{}, size:{}", start, size_);
    return VecView(GetUnchecked(start), size_ - start);
  }

  VecView slice(IndexType start, IndexType end) const {
    if UNLIKELY (start > size_ || start < 0)
      throw MakeExcept<OutOfRange>("Invalid start:{}, size:{}", start, size_);
    return VecView(GetUnchecked(start), std::max(0, std::min(size_, end) - start));
  }

  template <class U>
  void copy_from(const U &other) const {
    ENSURE(size() == static_cast<int>(other.size()),
           "copy_from size mismatch, expected: {}, got: {}", size(), other.size());
    for (SizeType i = 0; i < size(); ++i) {
      (*this)[i] = other[i];
    }
  }

  void fill(const T &v) const {
    for (SizeType i = 0; i < size(); ++i) {
      (*this)[i] = v;
    }
  }

  template <class F>
  void for_each(F &&f) const {
    for (SizeType i = 0; i < size(); ++i) {
      f((*this)[i]);
    }
  }

 private:
  T *ptr_;
  SizeType size_;
  SizeType stride_;

  T *Get(SizeType i) const {
#ifndef YANG_FORCE_BOUNDS_CHECK
    return GetUnchecked(i);
#else
    return GetChecked(i);
#endif
  }

  T *GetChecked(SizeType i) const {
    if UNLIKELY (i >= size_ || i < 0)
      throw MakeExcept<OutOfRange>("Invalid index:{}, size:{}", i, size_);
    return GetUnchecked(i);
  }

  T *GetUnchecked(SizeType i) const {
    return ptr_ + i * stride();
  }
};

template <class T, StaticSize StrideSize>
std::ostream &operator<<(std::ostream &os, const VecView<T, StrideSize> &vec) {
  for (SizeType i = 0; i < vec.size(); ++i) {
    if (i) os << ' ';
    os << vec[i];
  }
  return os;
}

}  // namespace yang::math

#ifdef FMT_OSTREAM_FORMATTER
template <class T, yang::math::StaticSize StrideSize>
struct fmt::formatter<yang::math::VecView<T, StrideSize>> : ostream_formatter {};
#endif
