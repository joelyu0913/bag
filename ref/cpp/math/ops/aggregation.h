#pragma once

#include <cmath>
#include <type_traits>

#include "yang/base/ranges.h"
#include "yang/math/ops/base.h"
#include "yang/math/type_traits.h"

namespace yang::math::ops {

template <class T, class ValidCheck = DefaultCheck, class U, class Iter, class Func>
T reduce(Iter first, Iter last, U init_value, Func &&f) {
  T acc = init_value;
  ValidCheck is_valid;
  for (auto it = first; it != last; ++it) {
    if (is_valid(*it)) acc = f(acc, *it);
  }
  return acc;
}

// Sum
template <class T, class ValidCheck = DefaultCheck, class Iter,
          std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
T sum(Iter first, Iter last) {
  return reduce<T, ValidCheck>(first, last, 0, [](auto s, auto v) { return s + v; });
}

// Sum
template <class ValidCheck = DefaultCheck, class Iter,
          std::enable_if_t<is_valid_check_v<ValidCheck>, int> = 0>
auto sum(Iter first, Iter last) {
  return sum<iter_value_t<Iter>, ValidCheck, Iter>(first, last);
}

// Sum
template <class T, class ValidCheck = DefaultCheck, class C,
          std::enable_if_t<std::is_arithmetic_v<T>, int> = 0,
          std::enable_if_t<is_iterable_v<C>, int> = 0>
T sum(const C &container) {
  return sum<T, ValidCheck>(std::begin(container), std::end(container));
}

// Sum
template <class ValidCheck = DefaultCheck, class C,
          std::enable_if_t<is_valid_check_v<ValidCheck>, int> = 0,
          std::enable_if_t<is_iterable_v<C>, int> = 0>
auto sum(const C &container) {
  return sum<ValidCheck>(std::begin(container), std::end(container));
}

// Weighted Sum
template <class T, class ValidCheck = DefaultCheck, class Iter, class WIter,
          std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
T wsum(Iter first, Iter last, WIter w_first) {
  T sum = 0;
  ValidCheck is_valid;
  auto it = first;
  auto w_it = w_first;
  for (; it != last; ++it, ++w_it) {
    if (is_valid(*it) && is_valid(*w_it)) sum += *it * *w_it;
  }
  return sum;
}

// Weighted Sum
template <class ValidCheck = DefaultCheck, class Iter, class WIter,
          std::enable_if_t<is_valid_check_v<ValidCheck>, int> = 0>
auto wsum(Iter first, Iter last, WIter w_first) {
  return wsum<iter_value_t<Iter>, ValidCheck, Iter, WIter>(first, last, w_first);
}

// Sum with boolean mask
template <class T, class ValidCheck = DefaultCheck, class MaskCheck = CheckBool, class Iter,
          class MIter, std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
T sum_masked(Iter first, Iter last, MIter mask_first) {
  T sum = 0;
  ValidCheck is_valid;
  MaskCheck is_masked;
  auto it = first;
  auto m_it = mask_first;
  for (; it != last; ++it, ++m_it) {
    if (is_valid(*it) && is_masked(*m_it)) sum += *it;
  }
  return sum;
}

// Sum with boolean mask
template <class ValidCheck = DefaultCheck, class MaskCheck = CheckBool, class Iter, class MIter,
          std::enable_if_t<is_valid_check_v<ValidCheck>, int> = 0>
auto sum_masked(Iter first, Iter last, MIter mask_first) {
  return sum_masked<iter_value_t<Iter>, ValidCheck, MaskCheck, Iter, MIter>(first, last,
                                                                            mask_first);
}

// Count
template <class ValidCheck = DefaultCheck, class Iter>
int count(Iter first, Iter last) {
  return reduce<int, ValidCheck>(first, last, 0, [](auto s, auto v) { return ++s; });
}

// Count
template <class T, class ValidCheck = DefaultCheck, class C,
          std::enable_if_t<is_iterable_v<C>, int> = 0>
T count(const C &container) {
  return count<T, ValidCheck>(std::begin(container), std::end(container));
}

// Count
template <class ValidCheck = DefaultCheck, class C,
          std::enable_if_t<is_valid_check_v<ValidCheck>, int> = 0,
          std::enable_if_t<is_iterable_v<C>, int> = 0>
auto count(const C &container) {
  return count<ValidCheck>(std::begin(container), std::end(container));
}

// Mean
template <class T, class ValidCheck = DefaultCheck, class Iter,
          std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
T mean(Iter first, Iter last, T default_value = detail::default_value<T>()) {
  T sum = 0;
  int n = 0;
  ValidCheck is_valid;
  for (auto it = first; it != last; ++it) {
    if (is_valid(*it)) {
      ++n;
      sum += *it;
    }
  }
  if (n == 0) return default_value;
  return sum / n;
}

// Mean
template <class ValidCheck = DefaultCheck, class Iter,
          std::enable_if_t<is_valid_check_v<ValidCheck>, int> = 0>
auto mean(Iter first, Iter last,
          iter_value_t<Iter> default_value = detail::default_iter_value<Iter>()) {
  return mean<iter_value_t<Iter>, ValidCheck, Iter>(first, last, default_value);
}

// Mean
template <class T, class ValidCheck = DefaultCheck, class C,
          std::enable_if_t<std::is_arithmetic_v<T>, int> = 0,
          std::enable_if_t<is_iterable_v<C>, int> = 0>
T mean(const C &container) {
  return mean<T, ValidCheck>(std::begin(container), std::end(container));
}

// Mean
template <class ValidCheck = DefaultCheck, class C,
          std::enable_if_t<is_valid_check_v<ValidCheck>, int> = 0,
          std::enable_if_t<is_iterable_v<C>, int> = 0>
auto mean(const C &container) {
  return mean<ValidCheck>(std::begin(container), std::end(container));
}

// Weighted Mean
template <class T, class ValidCheck = DefaultCheck, class Iter, class WIter,
          std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
T wmean(Iter first, Iter last, WIter w_first, T default_value = detail::default_value<T>()) {
  T sum = 0;
  T w_sum = 0;
  int n = 0;
  ValidCheck is_valid;
  auto it = first;
  auto w_it = w_first;
  for (; it != last; ++it, ++w_it) {
    if (is_valid(*it) && is_valid(*w_it)) {
      ++n;
      sum += *it * *w_it;
      w_sum += *w_it;
    }
  }
  if (n == 0) return default_value;
  return sum / w_sum;
}

// Weighted Mean
template <class ValidCheck = DefaultCheck, class Iter, class WIter,
          std::enable_if_t<is_valid_check_v<ValidCheck>, int> = 0>
auto wmean(Iter first, Iter last, WIter w_first,
           iter_value_t<Iter> default_value = detail::default_iter_value<Iter>()) {
  return wmean<iter_value_t<Iter>, ValidCheck, Iter, WIter>(first, last, w_first, default_value);
}

template <class T, class ValidCheck = DefaultCheck, class Iter,
          std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
T rmean(Iter first, Iter last, T default_value = detail::default_value<T>()) {
  T sum = 0;
  int n = 0;
  ValidCheck is_valid;
  T w = 1;
  for (auto it = first; it != last; ++it) {
    if (is_valid(*it)) {
      ++n;
      sum += *it * w;
    }
    w = -w;
  }
  if (n == 0) return default_value;
  return sum / n;
}

template <class ValidCheck = DefaultCheck, class Iter,
          std::enable_if_t<is_valid_check_v<ValidCheck>, int> = 0>
auto rmean(Iter first, Iter last,
           iter_value_t<Iter> default_value = detail::default_iter_value<Iter>()) {
  return rmean<iter_value_t<Iter>, ValidCheck, Iter>(first, last, default_value);
}

// Product
template <class T, class ValidCheck = DefaultCheck, class Iter,
          std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
T prod(Iter first, Iter last) {
  T prod = 1;
  ValidCheck is_valid;
  for (auto it = first; it != last; ++it) {
    if (is_valid(*it)) prod *= *it;
  }
  return prod;
}

// Product
template <class ValidCheck = DefaultCheck, class Iter,
          std::enable_if_t<is_valid_check_v<ValidCheck>, int> = 0>
auto prod(Iter first, Iter last) {
  return prod<iter_value_t<Iter>, ValidCheck, Iter>(first, last);
}

// Variance
template <class T, class ValidCheck = DefaultCheck, class Iter,
          std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
T variance(Iter first, Iter last, int ddof = 0, T default_value = detail::default_value<T>()) {
  int n = 0;
  T mean = ops::mean<T, ValidCheck, Iter>(first, last, default_value);
  T var = 0;
  ValidCheck is_valid;
  for (auto it = first; it != last; ++it) {
    if (is_valid(*it)) {
      auto diff = static_cast<T>(*it) - mean;
      var += diff * diff;
      ++n;
    }
  }
  if (n < 1 + ddof) return default_value;
  return var / (n - ddof);
}

// Variance
template <class ValidCheck = DefaultCheck, class Iter,
          std::enable_if_t<is_valid_check_v<ValidCheck>, int> = 0>
auto variance(Iter first, Iter last, int ddof = 0,
              iter_value_t<Iter> default_value = detail::default_iter_value<Iter>()) {
  return variance<iter_value_t<Iter>, ValidCheck, Iter>(first, last, ddof, default_value);
}

// Variance
template <class T, class ValidCheck = DefaultCheck, class C,
          std::enable_if_t<std::is_arithmetic_v<T>, int> = 0,
          std::enable_if_t<is_iterable_v<C>, int> = 0>
T variance(const C &container, int ddof = 0) {
  return variance<T, ValidCheck>(std::begin(container), std::end(container), ddof);
}

// Variance
template <class ValidCheck = DefaultCheck, class C,
          std::enable_if_t<is_valid_check_v<ValidCheck>, int> = 0,
          std::enable_if_t<is_iterable_v<C>, int> = 0>
auto variance(const C &container, int ddof = 0) {
  return variance<ValidCheck>(std::begin(container), std::end(container), ddof);
}

// Standard Deviation
template <class T, class ValidCheck = DefaultCheck, class Iter,
          std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
T stdev(Iter first, Iter last, int ddof = 0, T default_value = detail::default_value<T>()) {
  return std::sqrt(variance<T, ValidCheck, Iter>(first, last, ddof, default_value));
}

// Standard Deviation
template <class ValidCheck = DefaultCheck, class Iter,
          std::enable_if_t<is_valid_check_v<ValidCheck>, int> = 0>
auto stdev(Iter first, Iter last, int ddof = 0,
           iter_value_t<Iter> default_value = detail::default_iter_value<Iter>()) {
  return stdev<iter_value_t<Iter>, ValidCheck, Iter>(first, last, ddof, default_value);
}

// Standard Deviation
template <class T, class ValidCheck = DefaultCheck, class C,
          std::enable_if_t<std::is_arithmetic_v<T>, int> = 0,
          std::enable_if_t<is_iterable_v<C>, int> = 0>
T stdev(const C &container, int ddof = 0) {
  return stdev<T, ValidCheck>(std::begin(container), std::end(container), ddof);
}

// Standard Deviation
template <class ValidCheck = DefaultCheck, class C,
          std::enable_if_t<is_valid_check_v<ValidCheck>, int> = 0,
          std::enable_if_t<is_iterable_v<C>, int> = 0>
auto stdev(const C &container, int ddof = 0) {
  return stdev<ValidCheck>(std::begin(container), std::end(container), ddof);
}

// Covariance
template <class T, class ValidCheck = DefaultCheck, class XIter, class YIter,
          std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
T cov(XIter x_first, XIter x_last, YIter y_first, int ddof = 0,
      T default_value = detail::default_value<T>()) {
  T x = 0;
  T y = 0;
  T xy = 0;
  int n = 0;
  ValidCheck is_valid;
  auto x_it = x_first;
  auto y_it = y_first;
  for (; x_it != x_last; ++x_it, ++y_it) {
    if (is_valid(*x_it) && is_valid(*y_it)) {
      T x_i = *x_it;
      T y_i = *y_it;
      x += x_i;
      y += y_i;
      xy += x_i * y_i;
      ++n;
    }
  }
  if (n < 1 + ddof) return default_value;
  return (xy - x * y / (n - ddof)) / (n - ddof);
}

// Covariance
template <class ValidCheck = DefaultCheck, class XIter, class YIter,
          std::enable_if_t<is_valid_check_v<ValidCheck>, int> = 0>
auto cov(XIter x_first, XIter x_last, YIter y_first, int ddof = 0,
         iter_value_t<XIter> default_value = detail::default_iter_value<XIter>()) {
  return cov<iter_value_t<XIter>>(x_first, x_last, y_first, ddof, default_value);
}

// Correlation Coefficient
template <class T, class ValidCheck = DefaultCheck, class XIter, class YIter,
          std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
T corr(XIter x_first, XIter x_last, YIter y_first, T default_value = detail::default_value<T>()) {
  T x = 0;
  T y = 0;
  T xy = 0;
  T x2 = 0;
  T y2 = 0;
  int n = 0;
  ValidCheck is_valid;
  auto x_it = x_first;
  auto y_it = y_first;
  for (; x_it != x_last; ++x_it, ++y_it) {
    if (is_valid(*x_it) && is_valid(*y_it)) {
      T x_i = *x_it;
      T y_i = *y_it;
      x += x_i;
      y += y_i;
      xy += x_i * y_i;
      x2 += x_i * x_i;
      y2 += y_i * y_i;
      ++n;
    }
  }
  if (n <= 1) return default_value;
  auto var_x = x2 - x * x / n;
  auto var_y = y2 - y * y / n;
  if (var_x == 0 || var_y == 0) return default_value;
  return (xy - x * y / n) / std::sqrt(var_x) / std::sqrt(var_y);
}

// Correlation Coefficient
template <class ValidCheck = DefaultCheck, class XIter, class YIter,
          std::enable_if_t<is_valid_check_v<ValidCheck>, int> = 0>
auto corr(XIter x_first, XIter x_last, YIter y_first,
          iter_value_t<XIter> default_value = detail::default_iter_value<XIter>()) {
  return corr<iter_value_t<XIter>>(x_first, x_last, y_first, default_value);
}

// Corr with (0, 1, 2, ...)
template <class T, class ValidCheck = DefaultCheck, class XIter,
          std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
T corr_step(XIter x_first, XIter x_last, T default_value = detail::default_value<T>()) {
  auto steps = ranges::views::iota(0);
  return corr<T>(x_first, x_last, steps.begin(), default_value);
}

// Corr with (0, 1, 2, ...)
template <class ValidCheck = DefaultCheck, class XIter,
          std::enable_if_t<is_valid_check_v<ValidCheck>, int> = 0>
auto corr_step(XIter x_first, XIter x_last,
               iter_value_t<XIter> default_value = detail::default_iter_value<XIter>()) {
  return corr_step<iter_value_t<XIter>>(x_first, x_last, default_value);
}

// Auto correlation
template <class T, class ValidCheck = DefaultCheck, class XIter,
          std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
T auto_corr(XIter x_first, XIter x_last, int lag, T default_value = detail::default_value<T>()) {
  return corr<T>(x_first, x_last - lag, x_first + lag, default_value);
}

// Auto correlation
template <class ValidCheck = DefaultCheck, class XIter,
          std::enable_if_t<is_valid_check_v<ValidCheck>, int> = 0>
auto auto_corr(XIter x_first, XIter x_last, int lag,
               iter_value_t<XIter> default_value = detail::default_iter_value<XIter>()) {
  return auto_corr<iter_value_t<XIter>>(x_first, x_last, lag, default_value);
}

// Weighted Corr
template <class T, class ValidCheck = DefaultCheck, class XIter, class YIter, class WIter,
          std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
T wcorr(XIter x_first, XIter x_last, YIter y_first, WIter w_first,
        T default_value = detail::default_value<T>()) {
  T x_sum = 0;
  T y_sum = 0;
  int n = 0;
  ValidCheck is_valid;
  auto x_it = x_first;
  auto y_it = y_first;
  auto w_it = w_first;
  for (; x_it != x_last; ++x_it, ++y_it, ++w_it) {
    if (is_valid(*x_it) && is_valid(*y_it) && is_valid(*w_it)) {
      x_sum += *x_it;
      y_sum += *y_it;
      ++n;
    }
  }
  if (n <= 1) return default_value;
  T x_mean = x_sum / n;
  T y_mean = y_sum / n;
  T x2 = 0;
  T y2 = 0;
  T xy = 0;
  x_it = x_first;
  y_it = y_first;
  w_it = w_first;
  for (; x_it != x_last; ++x_it, ++y_it, ++w_it) {
    if (is_valid(*x_it) && is_valid(*y_it) && is_valid(*w_it)) {
      T x = *x_it - x_mean;
      T y = *y_it - y_mean;
      x2 = x * x * *w_it;
      y2 = y * y * *w_it;
      xy = x * y * *w_it;
    }
  }
  return xy / std::sqrt(x2 * y2);
}

// Weighted Corr
template <class ValidCheck = DefaultCheck, class XIter, class YIter, class WIter,
          std::enable_if_t<is_valid_check_v<ValidCheck>, int> = 0>
auto wcorr(XIter x_first, XIter x_last, YIter y_first, WIter w_first,
           iter_value_t<XIter> default_value = detail::default_iter_value<XIter>()) {
  return wcorr<iter_value_t<XIter>, ValidCheck, XIter, YIter, WIter>(x_first, x_last, y_first,
                                                                     w_first, default_value);
}

// Dot product
template <class T, class ValidCheck = DefaultCheck, class XIter, class YIter,
          std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
T dot(XIter x_first, XIter x_last, YIter y_first) {
  T sum = 0;
  ValidCheck is_valid;
  auto x_it = x_first;
  auto y_it = y_first;
  for (; x_it != x_last; ++x_it, ++y_it) {
    if (is_valid(*x_it) && is_valid(*y_it)) sum += static_cast<T>(*x_it) * static_cast<T>(*y_it);
  }
  return sum;
}

// Dot product
template <class ValidCheck = DefaultCheck, class XIter, class YIter,
          std::enable_if_t<is_valid_check_v<ValidCheck>, int> = 0>
auto dot(XIter x_first, XIter x_last, YIter y_first) {
  return dot<iter_value_t<XIter>, ValidCheck, XIter, YIter>(x_first, x_last, y_first);
}

// Squared Norm
template <class T, class ValidCheck = DefaultCheck, class Iter,
          std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
T squared_norm(Iter first, Iter last) {
  return reduce<T, ValidCheck>(first, last, 0, [](auto s, auto v) { return s + v * v; });
}

// Squared Norm
template <class ValidCheck = DefaultCheck, class Iter,
          std::enable_if_t<is_valid_check_v<ValidCheck>, int> = 0>
auto squared_norm(Iter first, Iter last) {
  return squared_norm<iter_value_t<Iter>, ValidCheck, Iter>(first, last);
}

// Squared Norm
template <class T, class ValidCheck = DefaultCheck, class C,
          std::enable_if_t<std::is_arithmetic_v<T>, int> = 0,
          std::enable_if_t<is_iterable_v<C>, int> = 0>
T squared_norm(const C &container) {
  return squared_norm<T, ValidCheck>(std::begin(container), std::end(container));
}

// Squared Norm
template <class ValidCheck = DefaultCheck, class C,
          std::enable_if_t<is_valid_check_v<ValidCheck>, int> = 0,
          std::enable_if_t<is_iterable_v<C>, int> = 0>
auto squared_norm(const C &container) {
  return squared_norm<ValidCheck>(std::begin(container), std::end(container));
}

// Norm
template <class T, class ValidCheck = DefaultCheck, class Iter,
          std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
T norm(Iter first, Iter last) {
  return std::sqrt(squared_norm<T, ValidCheck, Iter>(first, last));
}

// Norm
template <class ValidCheck = DefaultCheck, class Iter,
          std::enable_if_t<is_valid_check_v<ValidCheck>, int> = 0>
auto norm(Iter first, Iter last) {
  return norm<iter_value_t<Iter>, ValidCheck, Iter>(first, last);
}

// Norm
template <class T, class ValidCheck = DefaultCheck, class C,
          std::enable_if_t<std::is_arithmetic_v<T>, int> = 0,
          std::enable_if_t<is_iterable_v<C>, int> = 0>
T norm(const C &container) {
  return norm<T, ValidCheck>(std::begin(container), std::end(container));
}

// Norm
template <class ValidCheck = DefaultCheck, class C,
          std::enable_if_t<is_valid_check_v<ValidCheck>, int> = 0,
          std::enable_if_t<is_iterable_v<C>, int> = 0>
auto norm(const C &container) {
  return norm<ValidCheck>(std::begin(container), std::end(container));
}

// IC
template <class T, class ValidCheck = DefaultCheck, class XIter, class YIter,
          std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
T ic(XIter x_first, XIter x_last, YIter y_first) {
  T dot_sum = 0;
  T x_norm2 = 0;
  T y_norm2 = 0;
  ValidCheck is_valid;
  auto x_it = x_first;
  auto y_it = y_first;
  for (; x_it != x_last; ++x_it, ++y_it) {
    if (is_valid(*x_it) && is_valid(*y_it)) {
      auto x = *x_it;
      auto y = *y_it;
      dot_sum += x * y;
      x_norm2 += x * x;
      y_norm2 += y * y;
    }
  }
  return dot_sum / std::sqrt(x_norm2 * y_norm2);
}

// IC
template <class ValidCheck = DefaultCheck, class XIter, class YIter,
          std::enable_if_t<is_valid_check_v<ValidCheck>, int> = 0>
auto ic(XIter x_first, XIter x_last, YIter y_first) {
  return ic<iter_value_t<XIter>, ValidCheck, XIter, YIter>(x_first, x_last, y_first);
}

// Min
template <class ValidCheck = DefaultCheck, class Iter>
auto min(Iter first, Iter last) -> iter_value_t<Iter> {
  using T = iter_value_t<Iter>;
  bool val_set = false;
  return reduce<T, ValidCheck>(first, last, detail::default_value<T>(), [&](auto s, auto v) {
    auto ret = val_set ? std::min(s, v) : v;
    val_set = true;
    return ret;
  });
}

// Min
template <class ValidCheck = DefaultCheck, class C,
          std::enable_if_t<is_valid_check_v<ValidCheck>, int> = 0,
          std::enable_if_t<is_iterable_v<C>, int> = 0>
auto min(const C &container) {
  return min<ValidCheck>(std::begin(container), std::end(container));
}

// Max
template <class ValidCheck = DefaultCheck, class Iter>
auto max(Iter first, Iter last) -> iter_value_t<Iter> {
  using T = iter_value_t<Iter>;
  bool val_set = false;
  return reduce<T, ValidCheck>(first, last, detail::default_value<T>(), [&](auto s, auto v) {
    auto ret = val_set ? std::max(s, v) : v;
    val_set = true;
    return ret;
  });
}

// Max
template <class ValidCheck = DefaultCheck, class C,
          std::enable_if_t<is_valid_check_v<ValidCheck>, int> = 0,
          std::enable_if_t<is_iterable_v<C>, int> = 0>
auto max(const C &container) {
  return max<ValidCheck>(std::begin(container), std::end(container));
}

// Mean deviation
template <class ValidCheck = DefaultCheck, class Iter>
auto mean_dev(Iter first, Iter last) -> iter_value_t<Iter> {
  using T = iter_value_t<Iter>;
  auto mean = ops::mean(first, last);
  return reduce<T, ValidCheck>(first, last, 0,
                               [&](auto s, auto v) { return s + std::abs(v - mean); });
}

// Mean deviation
template <class ValidCheck = DefaultCheck, class C,
          std::enable_if_t<is_valid_check_v<ValidCheck>, int> = 0,
          std::enable_if_t<is_iterable_v<C>, int> = 0>
auto mean_dev(const C &container) {
  return mean_dev<ValidCheck>(std::begin(container), std::end(container));
}

// Argmax
template <class ValidCheck = DefaultCheck, template <class> class Compare = std::greater,
          class Iter>
auto argmax(Iter first, Iter last) -> iter_value_t<Iter> {
  using T = iter_value_t<Iter>;
  Compare<T> cmp;
  ValidCheck is_valid;
  T high_val{};
  int high_idx = -1;
  auto it = first;
  for (int i = 0; it != last; ++i, ++it) {
    if (!is_valid(*it)) continue;
    if (high_idx == -1 || cmp(*it, high_val)) {
      high_val = *it;
      high_idx = i;
    }
  }
  if (high_idx == -1) return -1;
  return last - first - 1 - high_idx;
}

// Argmax
template <class ValidCheck = DefaultCheck, class C,
          std::enable_if_t<is_valid_check_v<ValidCheck>, int> = 0,
          std::enable_if_t<is_iterable_v<C>, int> = 0>
auto argmax(const C &container) {
  return argmax<ValidCheck>(std::begin(container), std::end(container));
}

// Argmin
template <class ValidCheck = DefaultCheck, class Iter>
auto argmin(Iter first, Iter last) -> iter_value_t<Iter> {
  return argmax<ValidCheck, std::less>(first, last);
}

// Argmin
template <class ValidCheck = DefaultCheck, class C,
          std::enable_if_t<is_valid_check_v<ValidCheck>, int> = 0,
          std::enable_if_t<is_iterable_v<C>, int> = 0>
auto argmin(const C &container) {
  return argmin<ValidCheck>(std::begin(container), std::end(container));
}

// DecayLinear
template <class ValidCheck = DefaultCheck, class Iter>
auto decay_linear(Iter first, Iter last) -> iter_value_t<Iter> {
  if (first == last) return NAN;

  auto weights = ranges::views::iota(1);
  auto sum = ops::wsum(first, last, weights.begin());
  int size = last - first;
  auto w_sum = ops::sum_masked<ops::DefaultCheck, ops::CheckFinite>(weights.begin(),
                                                                    weights.begin() + size, first);
  return sum / w_sum;
}

// DecayLinear
template <class ValidCheck = DefaultCheck, class C,
          std::enable_if_t<is_valid_check_v<ValidCheck>, int> = 0,
          std::enable_if_t<is_iterable_v<C>, int> = 0>
auto decay_linear(const C &container) {
  return decay_linear<ValidCheck>(std::begin(container), std::end(container));
}

// EMA
template <class T, class ValidCheck = DefaultCheck, class Iter,
          std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
T ema(Iter first, Iter last, T ratio) {
  ValidCheck is_valid;
  T acc = NAN;
  for (auto it = first; it != last; ++it) {
    if (is_valid(acc) && is_valid(*it)) {
      acc = *it * ratio + acc * (1 - ratio);
    } else {
      acc = *it;
    }
  }
  return acc;
}

// EMA
template <class T, class ValidCheck = DefaultCheck, class C,
          std::enable_if_t<is_valid_check_v<ValidCheck>, int> = 0,
          std::enable_if_t<is_iterable_v<C>, int> = 0>
T ema(const C &container, T ratio) {
  return ema<T, ValidCheck>(std::begin(container), std::end(container), ratio);
}

}  // namespace yang::math::ops
