#pragma once

#include <Eigen/Dense>
#include <type_traits>

#include "yang/math/mat_view.h"

namespace yang::math {

template <class Mat, int Options = Eigen::RowMajor>
auto mat_to_eigen(const Mat &mat) {
  using T = std::remove_const_t<typename Mat::value_type>;
  using EigenMat = Eigen::Array<
      T, Mat::Shape::STATIC_ROW_SIZE == DYNAMIC_SIZE ? Eigen::Dynamic : Mat::Shape::STATIC_ROW_SIZE,
      Mat::Shape::STATIC_COL_SIZE == DYNAMIC_SIZE ? Eigen::Dynamic : Mat::Shape::STATIC_COL_SIZE,
      Options>;
  constexpr auto ROW_STRIDE = Mat::Stride::STATIC_ROW_STRIDE == DYNAMIC_STRIDE
                                  ? Eigen::Dynamic
                                  : Mat::Stride::STATIC_ROW_STRIDE;
  constexpr auto COL_STRIDE = Mat::Stride::STATIC_COL_STRIDE == DYNAMIC_STRIDE
                                  ? Eigen::Dynamic
                                  : Mat::Stride::STATIC_COL_STRIDE;
  constexpr auto OUTER_STRIDE = (Options & Eigen::RowMajor) ? ROW_STRIDE : COL_STRIDE;
  constexpr auto INNER_STRIDE = (Options & Eigen::RowMajor) ? COL_STRIDE : ROW_STRIDE;
  using EigenStride = Eigen::Stride<OUTER_STRIDE, INNER_STRIDE>;
  EigenStride stride((Options & Eigen::RowMajor) ? mat.row_stride() : mat.col_stride(),
                     (Options & Eigen::RowMajor) ? mat.col_stride() : mat.row_stride());
  if constexpr (std::is_const_v<typename Mat::value_type>) {
    return Eigen::Map<const EigenMat, Eigen::Unaligned, EigenStride>(mat.data(), mat.rows(),
                                                                     mat.cols(), stride);
  } else {
    return Eigen::Map<EigenMat, Eigen::Unaligned, EigenStride>(mat.data(), mat.rows(), mat.cols(),
                                                               stride);
  }
}

template <class Vec>
auto vec_to_eigen(const Vec &vec) {
  using Mat = MatView<typename Vec::value_type, MatShape<1, DYNAMIC_SIZE>,
                      MatStride<Eigen::Dynamic, Vec::Stride>>;
  return mat_to_eigen(Mat(vec.data(), 1, vec.size(), vec.size(), vec.stride()));
}

template <class Map>
auto eigen_to_mat(Map &map) {
  constexpr auto EIGEN_ROWS = Map::RowsAtCompileTime;
  constexpr auto EIGEN_COLS = Map::ColsAtCompileTime;
  constexpr auto EIGEN_ROW_STRIDE =
      Map::IsRowMajor ? int{Map::OuterStrideAtCompileTime} : int{Map::InnerStrideAtCompileTime};
  constexpr auto EIGEN_COL_STRIDE =
      Map::IsRowMajor ? int{Map::InnerStrideAtCompileTime} : int{Map::OuterStrideAtCompileTime};

  constexpr StaticSize ROWS = EIGEN_ROWS == Eigen::Dynamic ? DYNAMIC_SIZE : EIGEN_ROWS;
  constexpr StaticSize COLS = EIGEN_COLS == Eigen::Dynamic ? DYNAMIC_SIZE : EIGEN_COLS;
  constexpr StaticSize ROW_STRIDE =
      EIGEN_ROW_STRIDE == Eigen::Dynamic ? DYNAMIC_STRIDE : EIGEN_ROW_STRIDE;
  constexpr StaticSize COL_STRIDE =
      EIGEN_COL_STRIDE == Eigen::Dynamic ? DYNAMIC_STRIDE : EIGEN_COL_STRIDE;
  SizeType row_stride = Map::IsRowMajor ? map.outerStride() : map.innerStride();
  SizeType col_stride = Map::IsRowMajor ? map.innerStride() : map.outerStride();
  using T = std::remove_reference_t<decltype(*map.data())>;
  return MatView<T, MatShape<ROWS, COLS>, MatStride<ROW_STRIDE, COL_STRIDE>>(
      map.data(), {map.rows(), map.cols()}, {row_stride, col_stride});
}

}  // namespace yang::math
