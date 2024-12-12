#pragma once

#include <type_traits>

#include "yang/base/likely.h"
#include "yang/math/mat_shape.h"
#include "yang/math/mat_stride.h"
#include "yang/math/vec_iterator.h"
#include "yang/math/vec_view.h"
#include "yang/util/fmt.h"

namespace yang::math {

// TODO: alignment for SIMD load/store
template <class T, class ShapeType = MatShape<>, class StrideType = MatStride<UNSPECIFIED, 1>>
class MatView {
 public:
  using Shape = ShapeType;
  using Stride =
      MatStride<StrideType::STATIC_ROW_STRIDE != UNSPECIFIED
                    ? StrideType::STATIC_ROW_STRIDE
                    : (ShapeType::STATIC_COL_SIZE != DYNAMIC_SIZE ? ShapeType::STATIC_COL_SIZE
                                                                  : DYNAMIC_STRIDE),
                StrideType::STATIC_COL_STRIDE>;
  using Row = MatView<T, MatShape<1, Shape::STATIC_COL_SIZE>, Stride>;
  using Col = MatView<T, MatShape<Shape::STATIC_ROW_SIZE, 1>, Stride>;
  using ConstRow = MatView<const T, MatShape<1, Shape::STATIC_COL_SIZE>, Stride>;
  using ConstCol = MatView<const T, MatShape<Shape::STATIC_ROW_SIZE, 1>, Stride>;

  using value_type = T;
  using iterator = VecIterator<T, Shape::STATIC_ROW_SIZE == 1 ? Stride::STATIC_COL_STRIDE
                                                              : Stride::STATIC_ROW_STRIDE>;
  using const_iterator =
      VecIterator<const T, Shape::STATIC_ROW_SIZE == 1 ? Stride::STATIC_COL_STRIDE
                                                       : Stride::STATIC_ROW_STRIDE>;

  MatView() : ptr_(nullptr), stride_(0, 0), shape_(0, 0) {}

  MatView(T *ptr, Shape shape, StrideType stride)
      : ptr_(ptr), stride_(stride.row(), stride.col()), shape_(shape) {}

  template <class S = Stride, std::enable_if_t<!std::is_same_v<S, StrideType>, int> = 0>
  MatView(T *ptr, Shape shape, Stride stride) : ptr_(ptr), stride_(stride), shape_(shape) {}

  MatView(T *ptr, Shape shape) : ptr_(ptr), stride_(shape.col(), 1), shape_(shape) {}

  MatView(T *ptr, SizeType rows, SizeType cols) : ptr_(ptr), stride_(cols, 1), shape_(rows, cols) {}

  MatView(T *ptr, SizeType rows, SizeType cols, SizeType row_stride, SizeType col_stride)
      : ptr_(ptr), stride_(row_stride, col_stride), shape_(rows, cols) {}

  T *data() const {
    return ptr_;
  }

  T &operator()(SizeType row_i, SizeType col_i) const {
#ifndef YANG_FORCE_BOUNDS_CHECK
    return GetUnchecked(row_i, col_i);
#else
    return GetChecked(row_i, col_i);
#endif
  }

  T &at(SizeType row_i, SizeType col_i) const {
    return GetChecked(row_i, col_i);
  }

  Shape shape() const {
    return shape_;
  }

  Stride stride() const {
    return stride_;
  }

  SizeType rows() const {
    return shape_.row();
  }

  SizeType cols() const {
    return shape_.col();
  }

  SizeType row_stride() const {
    return stride_.row();
  }

  SizeType col_stride() const {
    return stride_.col();
  }

  Row row(SizeType i) const {
    if UNLIKELY (i >= shape_.row() || i < 0)
      throw MakeExcept<OutOfRange>("Invalid row:{}, row size:{}", i, shape_.row());
    return Row(ptr_ + i * stride_.row(), {1, shape_.col()}, {stride_.row(), stride_.col()});
  }

  Col col(SizeType i) const {
    if UNLIKELY (i >= shape_.col() || i < 0)
      throw MakeExcept<OutOfRange>("Invalid col:{}, col size:{}", i, shape_.col());
    return Col(ptr_ + i * stride_.col(), {shape_.row(), 1}, {stride_.row(), stride_.col()});
  }

  static constexpr bool is_vec() {
    return Shape::STATIC_ROW_SIZE == 1 || Shape::STATIC_COL_SIZE == 1;
  }

  template <class M = MatView, typename std::enable_if_t<M::is_vec(), int> = 0>
  iterator begin() const {
    if constexpr (Shape::STATIC_ROW_SIZE == 1) {
      return iterator(ptr_, stride_.col());
    } else {
      return iterator(ptr_, stride_.row());
    }
  }

  template <class M = MatView, typename std::enable_if_t<M::is_vec(), int> = 0>
  iterator end() const {
    if constexpr (Shape::STATIC_ROW_SIZE == 1) {
      return iterator(ptr_ + shape_.col() * stride_.col(), stride_.col());
    } else {
      return iterator(ptr_ + shape_.row() * stride_.row(), stride_.row());
    }
  }

  template <class M = MatView, typename std::enable_if_t<M::is_vec(), int> = 0>
  const_iterator cbegin() const {
    if constexpr (Shape::STATIC_ROW_SIZE == 1) {
      return const_iterator(ptr_, stride_.col());
    } else {
      return const_iterator(ptr_, stride_.row());
    }
  }

  template <class M = MatView, typename std::enable_if_t<M::is_vec(), int> = 0>
  const_iterator cend() const {
    if constexpr (Shape::STATIC_ROW_SIZE == 1) {
      return const_iterator(ptr_ + shape_.col() * stride_.col(), stride_.col());
    } else {
      return const_iterator(ptr_ + shape_.row() * stride_.row(), stride_.row());
    }
  }

  MatView block(SizeType row_i, SizeType col_i, SizeType rows, SizeType cols) const {
    if UNLIKELY (row_i > shape_.row() || row_i < 0 || row_i + rows > shape_.row())
      throw MakeExcept<OutOfRange>("Invalid row_i:{}, rows:{}, row size:{}", row_i, rows,
                                   shape_.row());
    if UNLIKELY (col_i > shape_.col() || col_i < 0 || col_i + cols > shape_.col())
      throw MakeExcept<OutOfRange>("Invalid col_i:{}, cols:{}, col size:{}", col_i, cols,
                                   shape_.col());
    auto ptr = ptr_ + stride_.row() * row_i + stride_.col() * col_i;
    return MatView(ptr, Shape(rows, cols), stride_);
  }

  MatView slice(SizeType row_begin, SizeType row_end, SizeType col_begin, SizeType col_end) const {
    if UNLIKELY (row_begin > shape_.row() || row_begin < 0)
      throw MakeExcept<OutOfRange>("Invalid row_begin:{}, row size:{}", row_begin, shape_.row());
    if UNLIKELY (col_begin > shape_.col() || col_begin < 0)
      throw MakeExcept<OutOfRange>("Invalid col_begin:{}, col size:{}", col_begin, shape_.col());
    auto ptr = ptr_ + stride_.row() * row_begin + stride_.col() * col_begin;
    row_end = std::max(row_begin, std::min(shape_.row(), row_end));
    col_end = std::max(col_begin, std::min(shape_.col(), col_end));
    return MatView(ptr, Shape(row_end - row_begin, col_end - col_begin), stride_);
  }

  MatView slice(SizeType row_begin, SizeType row_end) const {
    return slice(row_begin, row_end, 0, cols());
  }

  bool operator==(const MatView &other) const {
    return ptr_ == other.ptr_ && stride_ == other.stride_ && shape_ == other.shape_;
  }

  bool operator!=(const MatView &other) const {
    return !this->operator==(other);
  }

  operator MatView<const T, ShapeType, StrideType>() const {
    return MatView<const T, ShapeType, StrideType>(ptr_, shape_.row(), shape_.col(), stride_.row(),
                                                   stride_.col());
  }

  bool empty() const {
    return ptr_ == nullptr;
  }

  template <class M = MatView, typename std::enable_if_t<M::is_vec(), int> = 0>
  auto to_vec() const {
    if constexpr (Shape::STATIC_ROW_SIZE == 1) {
      // row vector
      return VecView<T, Stride::STATIC_COL_STRIDE>(ptr_, cols(), stride_.col());
    } else {
      // col vector
      return VecView<T, Stride::STATIC_ROW_STRIDE>(ptr_, rows(), stride_.row());
    }
  }

  auto transpose() const {
    return MatView<T, MatShape<Shape::STATIC_COL_SIZE, Shape::STATIC_ROW_SIZE>,
                   MatStride<Stride::STATIC_COL_STRIDE, Stride::STATIC_ROW_STRIDE>>(
        ptr_, cols(), rows(), col_stride(), row_stride());
  }

  template <class U>
  void copy_from(const U &other) const {
    ENSURE2(other.rows() == rows() && other.cols() == cols());
    if (stride_.col() == 1 || stride_.col() <= stride_.row()) {
      // Row major
      for (int i = 0; i < rows(); ++i) {
        row(i).to_vec().copy_from(other.row(i).to_vec());
      }
    } else {
      for (int i = 0; i < cols(); ++i) {
        col(i).to_vec().copy_from(other.col(i).to_vec());
      }
    }
  }

  void fill(const T &v) const {
    iterate_axis([&](auto &&a) { a.fill(v); });
  }

  template <class F>
  void for_each(F &&f) const {
    iterate_axis([&](auto &&a) { a.for_each(f); });
  }

 private:
  T *ptr_;
  Stride stride_;
  Shape shape_;

  template <class F>
  void iterate_axis(F &&f) const {
    if (stride_.col() == 1 || stride_.col() <= stride_.row()) {
      // Row major
      for (int i = 0; i < rows(); ++i) {
        f(row(i).to_vec());
      }
    } else {
      for (int i = 0; i < cols(); ++i) {
        f(col(i).to_vec());
      }
    }
  }

  T &GetUnchecked(SizeType row_i, SizeType col_i) const {
    return ptr_[stride_.row() * row_i + stride_.col() * col_i];
  }

  T &GetChecked(SizeType row_i, SizeType col_i) const {
    if UNLIKELY (row_i >= shape_.row() || row_i < 0)
      throw MakeExcept<OutOfRange>("Invalid row:{}, row size:{}", row_i, shape_.row());
    if UNLIKELY (col_i >= shape_.col() || col_i < 0)
      throw MakeExcept<OutOfRange>("Invalid col:{}, col size:{}", col_i, shape_.col());
    return GetUnchecked(row_i, col_i);
  }
};

template <class T, class ShapeType, class StrideType>
std::ostream &operator<<(std::ostream &os, const MatView<T, ShapeType, StrideType> &mat) {
  for (SizeType i = 0; i < mat.rows(); ++i) {
    if (i) os << '\n';
    for (SizeType j = 0; j < mat.cols(); ++j) {
      if (j) os << ' ';
      os << mat(i, j);
    }
  }
  return os;
}
}  // namespace yang::math

#ifdef FMT_OSTREAM_FORMATTER
template <class T, class ShapeType, class StrideType>
struct fmt::formatter<yang::math::MatView<T, ShapeType, StrideType>> : ostream_formatter {};
#endif
