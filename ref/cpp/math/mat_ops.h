#pragma once

#include <cmath>

#include "yang/math/eigen.h"
#include "yang/math/ops.h"
#include "yang/math/type_traits.h"

// Common matrix operators that can handle invalid values
namespace yang::math::ops {

namespace detail {
template <class Func, class Mat, class... Mats>
void iterate_rows(Func &&f, const Mat &mat, Mats &&...mats) {
  for (int row = 0; row < mat.rows(); ++row) {
    auto begin_row = [&](auto &m) { return m.row(row).begin(); };
    auto in_row = mat.row(row);
    f(in_row.begin(), in_row.end(), begin_row(mats)...);
  }
}
}  // namespace detail

// Exponentially weighted average
template <class ValidCheck = DefaultCheck, class Mat, class OutMat,
          std::enable_if_t<is_mat_view_v<Mat>, int> = 0>
void ewa(const Mat &mat, const OutMat &out_mat, typename OutMat::value_type ratio) {
  ValidCheck is_valid;
  for (int col = 0; col < mat.cols(); ++col) {
    out_mat(0, col) = mat(0, col);
  }
  for (int row = 1; row < mat.rows(); ++row) {
    for (int col = 0; col < mat.cols(); ++col) {
      auto pv = out_mat(row - 1, col);
      auto v = mat(row, col);
      if (is_valid(pv) && is_valid(v)) {
        v = v * ratio + pv * (1 - ratio);
        out_mat(row, col) = v;
      } else {
        out_mat(row, col) = v;
      }
    }
  }
}

// Exponentially weighted average, in place version
template <class ValidCheck = DefaultCheck, class Mat, std::enable_if_t<is_mat_view_v<Mat>, int> = 0>
void ewa(const Mat &mat, typename Mat::value_type ratio) {
  ewa<ValidCheck>(mat, mat, ratio);
}

// Smooth Exponentially weighted average
template <class ValidCheck = DefaultCheck, class Mat, class OutMat,
          class Allocator = std::allocator<int>, std::enable_if_t<is_mat_view_v<Mat>, int> = 0>
void smooth_ewa(const Mat &mat, const OutMat &out_mat, typename OutMat::value_type ratio,
                const Allocator &alloc = Allocator()) {
  using T = typename Mat::value_type;
  using TAllocator = typename std::allocator_traits<Allocator>::template rebind_alloc<T>;

  std::vector<T, TAllocator> f(mat.rows());
  f[0] = 0;
  T fv = 1;
  for (int i = 1; i < mat.rows(); ++i) {
    fv *= 1 - ratio;
    f[i] = f[i - 1] + i * fv;
  }
  auto g = [](T x, int k) { return (1 - std::pow(x, k + 1)) / (1 - x); };
  std::vector<int, Allocator> last_idx(mat.rows(), -1);

  ValidCheck is_valid;
  for (int col = 0; col < mat.cols(); ++col) {
    out_mat(0, col) = mat(0, col);
    if (is_valid(mat(0, col))) last_idx[col] = 0;
  }
  for (int row = 1; row < mat.rows(); ++row) {
    for (int col = 0; col < mat.cols(); ++col) {
      auto v = mat(row, col);
      if (is_valid(v)) {
        if (last_idx[col] >= 0) {
          int p = row - last_idx[col];
          out_mat(row, col) = ratio / p * mat(last_idx[col], col) * f[p - 1] +
                              ratio / p * v * (p * g(1 - ratio, p - 1) - f[p - 1]) +
                              std::pow(1 - ratio, p) * out_mat(last_idx[col], col);
        } else {
          out_mat(row, col) = v;
        }
        last_idx[col] = row;
      } else {
        out_mat(row, col) = v;
      }
    }
  }
}

// Smooth Exponentially weighted average
template <class ValidCheck = DefaultCheck, class Mat, class Allocator = std::allocator<int>,
          std::enable_if_t<is_mat_view_v<Mat>, int> = 0>
void smooth_ewa(const Mat &mat, typename Mat::value_type ratio,
                const Allocator &alloc = Allocator()) {
  smooth_ewa(mat, mat, ratio, alloc);
}

// Rank by rows
template <class Mat, class OutMat, class Allocator = std::allocator<int>,
          std::enable_if_t<is_mat_view_v<Mat>, int> = 0,
          std::enable_if_t<is_mat_view_v<OutMat>, int> = 0>
void rank(const Mat &mat, const OutMat &out_mat, typename OutMat::value_type eps = EPSILON,
          const Allocator &alloc = Allocator()) {
  detail::iterate_rows([&](auto first, auto last, auto out) { rank(first, last, out, eps, alloc); },
                       mat, out_mat);
}

// Rank by rows, in place version
template <class Mat, class Allocator = std::allocator<int>,
          std::enable_if_t<is_mat_view_v<Mat>, int> = 0>
void rank(const Mat &mat, typename Mat::value_type eps = EPSILON,
          const Allocator &alloc = Allocator()) {
  rank(mat, mat, eps, alloc);
}

// rank_pow by rows
template <class Mat, class OutMat, class Allocator = std::allocator<int>,
          std::enable_if_t<is_mat_view_v<Mat>, int> = 0,
          std::enable_if_t<is_mat_view_v<OutMat>, int> = 0>
void rank_pow(const Mat &mat, const OutMat &out_mat, typename OutMat::value_type exp,
              typename OutMat::value_type eps = EPSILON, const Allocator &alloc = Allocator()) {
  detail::iterate_rows(
      [&](auto first, auto last, auto out) { rank_pow(first, last, out, exp, eps, alloc); }, mat,
      out_mat);
}

// rank_pow, in place version
template <class Mat, class Allocator = std::allocator<int>,
          std::enable_if_t<is_mat_view_v<Mat>, int> = 0>
void rank_pow(const Mat &mat, typename Mat::value_type exp, typename Mat::value_type eps = EPSILON,
              const Allocator &alloc = Allocator()) {
  rank_pow(mat, mat, exp, eps, alloc);
}

// Demean by rows
template <class ValidCheck = DefaultCheck, class Mat, class OutMat,
          std::enable_if_t<is_mat_view_v<Mat>, int> = 0>
void demean(const Mat &mat, const OutMat &out_mat) {
  detail::iterate_rows(
      [&](auto first, auto last, auto out) { demean<ValidCheck>(first, last, out); }, mat, out_mat);
}

// Demean by rows, in place version
template <class ValidCheck = DefaultCheck, class Mat, std::enable_if_t<is_mat_view_v<Mat>, int> = 0>
void demean(const Mat &mat) {
  demean<ValidCheck>(mat, mat);
}
// Momentum by group, by rows
template <class ValidCheck = DefaultCheck, class Mat, class GMat, class OutMat,
          class Allocator = std::allocator<int>, std::enable_if_t<is_mat_view_v<Mat>, int> = 0,
          std::enable_if_t<is_mat_view_v<GMat>, int> = 0,
          std::enable_if_t<is_mat_view_v<OutMat>, int> = 0>
void group_mom(const Mat &mat, const GMat &g_mat, const OutMat &out_mat,
               const Allocator &alloc = Allocator()) {
  detail::iterate_rows([&](auto first, auto last, auto g_first,
                           auto out) { group_mom<ValidCheck>(first, last, g_first, out, alloc); },
                       mat, g_mat, out_mat);
}

// Demean by rows, in place version
template <class ValidCheck = DefaultCheck, class Mat, class GMat,
          class Allocator = std::allocator<int>, std::enable_if_t<is_mat_view_v<Mat>, int> = 0,
          std::enable_if_t<is_mat_view_v<GMat>, int> = 0,
          std::enable_if_t<is_allocator_v<Allocator>, int> = 0>
void group_mom(const Mat &mat, const GMat &g_mat, const Allocator &alloc = Allocator()) {
  group_mom<ValidCheck>(mat, g_mat, mat, alloc);
}

// Demean by group, by rows
template <class ValidCheck = DefaultCheck, class Mat, class GMat, class OutMat,
          class Allocator = std::allocator<int>, std::enable_if_t<is_mat_view_v<Mat>, int> = 0,
          std::enable_if_t<is_mat_view_v<GMat>, int> = 0,
          std::enable_if_t<is_mat_view_v<OutMat>, int> = 0>
void group_demean(const Mat &mat, const GMat &g_mat, const OutMat &out_mat,
                  const Allocator &alloc = Allocator()) {
  detail::iterate_rows(
      [&](auto first, auto last, auto g_first, auto out) {
        group_demean<ValidCheck>(first, last, g_first, out, alloc);
      },
      mat, g_mat, out_mat);
}

// Demean by rows, in place version
template <class ValidCheck = DefaultCheck, class Mat, class GMat,
          class Allocator = std::allocator<int>, std::enable_if_t<is_mat_view_v<Mat>, int> = 0,
          std::enable_if_t<is_mat_view_v<GMat>, int> = 0,
          std::enable_if_t<is_allocator_v<Allocator>, int> = 0>
void group_demean(const Mat &mat, const GMat &g_mat, const Allocator &alloc = Allocator()) {
  group_demean<ValidCheck>(mat, g_mat, mat, alloc);
}

// Filter by rows
template <class Mat, class CondMat, class OutMat, std::enable_if_t<is_mat_view_v<Mat>, int> = 0>
void filter(const Mat &mat, const CondMat &cond_mat, const OutMat &out_mat,
            typename OutMat::value_type default_value =
                detail::default_value<typename OutMat::value_type>()) {
  detail::iterate_rows(
      [&](auto first, auto last, auto c_first, auto out) { filter(first, last, c_first, out); },
      mat, cond_mat, out_mat);
}

// Filter by rows, in place version
template <class Mat, class CondMat, std::enable_if_t<is_mat_view_v<Mat>, int> = 0>
void filter(
    const Mat &mat, const CondMat &cond_mat,
    typename Mat::value_type default_value = detail::default_value<typename Mat::value_type>()) {
  filter(mat, cond_mat, mat, default_value);
}

// spow by rows
template <class Mat, class OutMat, std::enable_if_t<is_mat_view_v<Mat>, int> = 0>
void spow(const Mat &mat, const OutMat &out_mat, typename OutMat::value_type exp) {
  detail::iterate_rows([&](auto first, auto last, auto out) { spow(first, last, out, exp); }, mat,
                       out_mat);
}

// spow by rows, in place version
template <class Mat, std::enable_if_t<is_mat_view_v<Mat>, int> = 0>
void spow(const Mat &mat, typename Mat::value_type exp) {
  spow(mat, mat, exp);
}

// pow by rows
template <class Mat, class OutMat, std::enable_if_t<is_mat_view_v<Mat>, int> = 0>
void pow(const Mat &mat, const OutMat &out_mat, typename OutMat::value_type exp) {
  detail::iterate_rows([&](auto first, auto last, auto out) { pow(first, last, out, exp); }, mat,
                       out_mat);
}

// pow by rows, in place version
template <class Mat, std::enable_if_t<is_mat_view_v<Mat>, int> = 0>
void pow(const Mat &mat, typename Mat::value_type exp) {
  pow(mat, mat, exp);
}

// shift by rows
// row'(i + periods) = row(i)
template <class Mat, class OutMat, std::enable_if_t<is_mat_view_v<Mat>, int> = 0>
void shift_rows(
    const Mat &mat, const OutMat &out_mat, int periods,
    typename OutMat::value_type fill_value = detail::default_value<typename OutMat::value_type>()) {
  int n = mat.rows();
  auto e_mat = mat_to_eigen(mat);
  auto e_out_mat = mat_to_eigen(out_mat);
  if (periods > 0) {
    for (int i = n - 1; i >= periods; --i) {
      e_out_mat.row(i) = e_mat.row(i - periods);
    }
    for (int i = 0; i < periods; ++i) e_out_mat.row(i).setConstant(fill_value);
  } else if (periods < 0) {
    periods = -periods;
    for (int i = 0; i < n - periods; ++i) {
      e_out_mat.row(i) = e_mat.row(i + periods);
    }
    for (int i = n - periods; i < n; ++i) e_out_mat.row(i).setConstant(fill_value);
  }
}

// shift by rows, in place version
template <class Mat, std::enable_if_t<is_mat_view_v<Mat>, int> = 0>
void shift_rows(
    const Mat &mat, int periods,
    typename Mat::value_type fill_value = detail::default_value<typename Mat::value_type>()) {
  shift_rows(mat, mat, periods);
}

// Hedge by rows
template <class Mat, class OutMat, std::enable_if_t<is_mat_view_v<Mat>, int> = 0>
void hedge(const Mat &mat, const OutMat &out_mat, int univ_size, int hedge_idx) {
  detail::iterate_rows(
      [&](auto first, auto last, auto out) { hedge(first, last, out, univ_size, hedge_idx); }, mat,
      out_mat);
}

// Hedge by rows, in place version
template <class Mat, std::enable_if_t<is_mat_view_v<Mat>, int> = 0>
void hedge(const Mat &mat, int univ_size, int hedge_idx) {
  hedge(mat, mat, univ_size, hedge_idx);
}

// Hedge by rows
template <class Mat, class OutMat, std::enable_if_t<is_mat_view_v<Mat>, int> = 0>
void hedgeshort(const Mat &mat, const OutMat &out_mat, int univ_size, int hedge_idx) {
  detail::iterate_rows(
      [&](auto first, auto last, auto out) { hedgeshort(first, last, out, univ_size, hedge_idx); },
      mat, out_mat);
}

// Hedge by rows, in place version
template <class Mat, std::enable_if_t<is_mat_view_v<Mat>, int> = 0>
void hedgeshort(const Mat &mat, int univ_size, int hedge_idx) {
  hedgeshort(mat, mat, univ_size, hedge_idx);
}

// Scale by rows
template <class Mat, class OutMat, std::enable_if_t<is_mat_view_v<Mat>, int> = 0,
          std::enable_if_t<is_mat_view_v<OutMat>, int> = 0>
void scale(const Mat &mat, const OutMat &out_mat, typename OutMat::value_type scale_size,
           typename OutMat::value_type eps = 0) {
  detail::iterate_rows(
      [&](auto first, auto last, auto out) { scale(first, last, out, scale_size, eps); }, mat,
      out_mat);
}

// Scale by rows, in place version
template <class Mat, std::enable_if_t<is_mat_view_v<Mat>, int> = 0>
void scale(const Mat &mat, typename Mat::value_type scale_size, typename Mat::value_type eps = 0) {
  scale(mat, mat, scale_size, eps);
}

// Select top n by rows
template <class Mat, class OutMat, class Compare, class Allocator = std::allocator<int>,
          std::enable_if_t<is_mat_view_v<Mat>, int> = 0,
          std::enable_if_t<is_mat_view_v<OutMat>, int> = 0,
          std::enable_if_t<!std::is_same_v<Compare, typename Mat::value_type>, int> = 0>
void select_n_largest(
    const Mat &mat, const OutMat &out_mat, int n, Compare cmp,
    typename OutMat::value_type null_value = detail::default_value<typename OutMat::value_type>(),
    const Allocator &alloc = Allocator()) {
  detail::iterate_rows(
      [&](auto first, auto last, auto out) {
        select_n_largest(first, last, out, n, cmp, null_value, alloc);
      },
      mat, out_mat);
}

// Select top n by rows
template <class Mat, class OutMat, class Allocator = std::allocator<int>,
          std::enable_if_t<is_mat_view_v<Mat>, int> = 0,
          std::enable_if_t<is_mat_view_v<OutMat>, int> = 0>
void select_n_largest(
    const Mat &mat, const OutMat &out_mat, int n,
    typename OutMat::value_type null_value = detail::default_value<typename OutMat::value_type>(),
    const Allocator &alloc = Allocator()) {
  detail::iterate_rows([&](auto first, auto last,
                           auto out) { select_n_largest(first, last, out, n, null_value, alloc); },
                       mat, out_mat);
}

// Select top n by rows, in-place version
template <class Mat, class Compare, class Allocator = std::allocator<int>,
          std::enable_if_t<is_mat_view_v<Mat>, int> = 0,
          std::enable_if_t<!std::is_same_v<Compare, typename Mat::value_type>, int> = 0>
void select_n_largest(
    const Mat &mat, int n, Compare cmp,
    typename Mat::value_type null_value = detail::default_value<typename Mat::value_type>(),
    const Allocator &alloc = Allocator()) {
  select_n_largest(mat, mat, cmp, null_value, alloc);
}

// Select top n by rows, in-place version
template <class Mat, class Allocator = std::allocator<int>,
          std::enable_if_t<is_mat_view_v<Mat>, int> = 0>
void select_n_largest(
    const Mat &mat, int n,
    typename Mat::value_type null_value = detail::default_value<typename Mat::value_type>(),
    const Allocator &alloc = Allocator()) {
  select_n_largest(mat, mat, n, null_value, alloc);
}

// Upbound by rows
template <class Mat, class OutMat, class Allocator = std::allocator<int>,
          std::enable_if_t<is_mat_view_v<Mat>, int> = 0,
          std::enable_if_t<is_mat_view_v<OutMat>, int> = 0>
void upbound(const Mat &mat, const OutMat &out_mat, int k,
             typename OutMat::value_type eps = EPSILON, const Allocator &alloc = Allocator()) {
  detail::iterate_rows(
      [&](auto first, auto last, auto out) { upbound(first, last, out, k, eps, alloc); }, mat,
      out_mat);
}

// Upbound by rows, in place version
template <class Mat, class Allocator = std::allocator<int>,
          std::enable_if_t<is_mat_view_v<Mat>, int> = 0>
void upbound(const Mat &mat, int k, typename Mat::value_type eps = EPSILON,
             const Allocator &alloc = Allocator()) {
  upbound(mat, mat, k, eps, alloc);
}

template <class ValidCheck = DefaultCheck, class Mat, class Vec,
          std::enable_if_t<is_mat_view_v<Mat>, int> = 0>
void mean(const Mat &mat, const Vec &out) {
  for (int i = 0; i < mat.rows(); ++i) {
    auto row = mat.row(i);
    out[i] = mean<typename Vec::value_type, ValidCheck>(row.begin(), row.end());
  }
}

template <class ValidCheck = DefaultCheck, class Mat, class Vec,
          std::enable_if_t<is_mat_view_v<Mat>, int> = 0>
void sum(const Mat &mat, const Vec &out) {
  for (int i = 0; i < mat.rows(); ++i) {
    auto row = mat.row(i);
    out[i] = sum<typename Vec::value_type, ValidCheck>(row.begin(), row.end());
  }
}

template <class ValidCheck = DefaultCheck, class Mat, class Vec,
          std::enable_if_t<is_mat_view_v<Mat>, int> = 0>
void variance(const Mat &mat, const Vec &out) {
  for (int i = 0; i < mat.rows(); ++i) {
    auto row = mat.row(i);
    out[i] = variance<typename Vec::value_type, ValidCheck>(row.begin(), row.end());
  }
}

template <class ValidCheck = DefaultCheck, class Mat, class Vec,
          std::enable_if_t<is_mat_view_v<Mat>, int> = 0>
void stdev(const Mat &mat, const Vec &out) {
  for (int i = 0; i < mat.rows(); ++i) {
    auto row = mat.row(i);
    out[i] = stdev<typename Vec::value_type, ValidCheck>(row.begin(), row.end());
  }
}

}  // namespace yang::math::ops
