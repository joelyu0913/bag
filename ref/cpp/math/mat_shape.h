#pragma once

#include "yang/math/size.h"
#include "yang/util/logging.h"

namespace yang::math {

template <StaticSize RowSize = DYNAMIC_SIZE, StaticSize ColSize = DYNAMIC_SIZE>
class MatShape {
 public:
  static constexpr StaticSize STATIC_ROW_SIZE = RowSize;
  static constexpr StaticSize STATIC_COL_SIZE = ColSize;

  MatShape() {}

  MatShape(SizeType row, SizeType col) {}

  template <class U>
  MatShape(std::initializer_list<U> s) {}

  SizeType row() const {
    return RowSize;
  }

  SizeType col() const {
    return ColSize;
  }

  bool operator==(const MatShape &other) const {
    return true;
  }

  bool operator!=(const MatShape &other) const {
    return false;
  }
};

template <StaticSize RowSize>
class MatShape<RowSize, DYNAMIC_SIZE> {
 public:
  static constexpr StaticSize STATIC_ROW_SIZE = RowSize;
  static constexpr StaticSize STATIC_COL_SIZE = DYNAMIC_SIZE;

  MatShape(SizeType row, SizeType col) : col_(col) {}

  template <class U>
  MatShape(std::initializer_list<U> s) {
    ENSURE2(s.size() == 2);
    col_ = s.begin()[1];
  }

  SizeType row() const {
    return RowSize;
  }

  SizeType col() const {
    return col_;
  }

  bool operator==(const MatShape &other) const {
    return col_ == other.col_;
  }

  bool operator!=(const MatShape &other) const {
    return !this->operator==(other);
  }

 private:
  SizeType col_;
};

template <StaticSize ColSize>
class MatShape<DYNAMIC_SIZE, ColSize> {
 public:
  static constexpr StaticSize STATIC_ROW_SIZE = DYNAMIC_SIZE;
  static constexpr StaticSize STATIC_COL_SIZE = ColSize;

  MatShape(SizeType row, SizeType col) : row_(row) {}

  template <class U>
  MatShape(std::initializer_list<U> s) {
    ENSURE2(s.size() == 2);
    row_ = s.begin()[0];
  }

  SizeType row() const {
    return row_;
  }

  SizeType col() const {
    return ColSize;
  }

  bool operator==(const MatShape &other) const {
    return row_ == other.row_;
  }

  bool operator!=(const MatShape &other) const {
    return !this->operator==(other);
  }

 private:
  SizeType row_;
};

template <>
class MatShape<DYNAMIC_SIZE, DYNAMIC_SIZE> {
 public:
  static constexpr StaticSize STATIC_ROW_SIZE = DYNAMIC_SIZE;
  static constexpr StaticSize STATIC_COL_SIZE = DYNAMIC_SIZE;

  MatShape(SizeType row, SizeType col) : row_(row), col_(col) {}

  template <class U>
  MatShape(std::initializer_list<U> s) {
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

  bool operator==(const MatShape &other) const {
    return row_ == other.row_ && col_ == other.col_;
  }

  bool operator!=(const MatShape &other) const {
    return !this->operator==(other);
  }

 private:
  SizeType row_;
  SizeType col_;
};

}  // namespace yang::math
