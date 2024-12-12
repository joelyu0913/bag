#pragma once

#include "yang/math/size.h"
#include "yang/util/logging.h"

namespace yang::math {

template <StaticSize RowStride = DYNAMIC_STRIDE, StaticSize ColStride = DYNAMIC_STRIDE>
class MatStride {
 public:
  static constexpr StaticSize STATIC_ROW_STRIDE = RowStride;
  static constexpr StaticSize STATIC_COL_STRIDE = ColStride;

  MatStride() {}

  MatStride(SizeType row, SizeType col) {}

  template <class U>
  MatStride(std::initializer_list<U> s) {}

  SizeType row() const {
    return RowStride;
  }

  SizeType col() const {
    return ColStride;
  }

  bool operator==(const MatStride &other) const {
    return true;
  }

  bool operator!=(const MatStride &other) const {
    return false;
  }
};

template <StaticSize RowStride>
class MatStride<RowStride, DYNAMIC_STRIDE> {
 public:
  static constexpr StaticSize STATIC_ROW_STRIDE = RowStride;
  static constexpr StaticSize STATIC_COL_STRIDE = DYNAMIC_STRIDE;

  MatStride(SizeType row, SizeType col) : col_(col) {}

  template <class U>
  MatStride(std::initializer_list<U> s) {
    ENSURE2(s.size() == 2);
    col_ = s.begin()[1];
  }

  SizeType row() const {
    return RowStride;
  }

  SizeType col() const {
    return col_;
  }

  bool operator==(const MatStride &other) const {
    return col_ == other.col_;
  }

  bool operator!=(const MatStride &other) const {
    return !this->operator==(other);
  }

 private:
  SizeType col_;
};

template <StaticSize ColStride>
class MatStride<DYNAMIC_STRIDE, ColStride> {
 public:
  static constexpr StaticSize STATIC_ROW_STRIDE = DYNAMIC_STRIDE;
  static constexpr StaticSize STATIC_COL_STRIDE = ColStride;

  MatStride(SizeType row, SizeType col) : row_(row) {}

  template <class U>
  MatStride(std::initializer_list<U> s) {
    ENSURE2(s.size() == 2);
    row_ = s.begin()[0];
  }

  SizeType row() const {
    return row_;
  }

  SizeType col() const {
    return ColStride;
  }

  bool operator==(const MatStride &other) const {
    return row_ == other.row_;
  }

  bool operator!=(const MatStride &other) const {
    return !this->operator==(other);
  }

 private:
  SizeType row_;
};

template <>
class MatStride<DYNAMIC_STRIDE, DYNAMIC_STRIDE> {
 public:
  static constexpr StaticSize STATIC_ROW_STRIDE = DYNAMIC_STRIDE;
  static constexpr StaticSize STATIC_COL_STRIDE = DYNAMIC_STRIDE;

  MatStride(SizeType row, SizeType col) : row_(row), col_(col) {}

  template <class U>
  MatStride(std::initializer_list<U> s) {
    ENSURE2(s.size() == 2);
    row_ = s.begin()[0];
    col_ = s.begin()[1];
  }

  SizeType row() const {
    return row_;
  }

  SizeType col() const {
    return col_;
  }

  bool operator==(const MatStride &other) const {
    return row_ == other.row_ && col_ == other.col_;
  }

  bool operator!=(const MatStride &other) const {
    return !this->operator==(other);
  }

 private:
  SizeType row_;
  SizeType col_;
};

}  // namespace yang::math
