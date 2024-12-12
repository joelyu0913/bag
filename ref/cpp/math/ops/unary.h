#pragma once

#include <algorithm>
#include <cmath>
#include <type_traits>
#include <vector>

#include "yang/base/inline.h"
#include "yang/math/ops/aggregation.h"
#include "yang/math/ops/base.h"
#include "yang/math/type_traits.h"

namespace yang::math::ops {

template <class ValidCheck = DefaultCheck, class Iter, class OutIter, class Func>
ALWAYS_INLINE void simple_unary(
    Iter first, Iter last, OutIter out, Func &&f,
    iter_value_t<OutIter> default_value = detail::default_iter_value<OutIter>()) {
  ValidCheck is_valid;
  for (auto it = first; it != last; ++it, ++out) {
    auto x = *it;
    if (is_valid(x))
      *out = f(x);
    else
      *out = default_value;
  }
}

// abs(x)
template <class Iter, class OutIter, std::enable_if_t<is_iterator_v<Iter>, int> = 0,
          std::enable_if_t<is_iterator_v<OutIter>, int> = 0>
void abs(Iter first, Iter last, OutIter out) {
  simple_unary(first, last, out, [](auto x) { return std::abs(x); });
}

// abs(x), in-place version
template <class Iter, std::enable_if_t<is_iterator_v<Iter>, int> = 0>
void abs(Iter first, Iter last) {
  abs(first, last, first);
}

// 1 / x
template <class Iter, class OutIter, std::enable_if_t<is_iterator_v<Iter>, int> = 0,
          std::enable_if_t<is_iterator_v<OutIter>, int> = 0>
void inverse(Iter first, Iter last, OutIter out) {
  simple_unary(first, last, out, [](auto x) { return 1 / x; });
}

// 1 / x, in-place version
template <class Iter, std::enable_if_t<is_iterator_v<Iter>, int> = 0>
void inverse(Iter first, Iter last) {
  inverse(first, last, first);
}

// -x
template <class Iter, class OutIter, std::enable_if_t<is_iterator_v<Iter>, int> = 0,
          std::enable_if_t<is_iterator_v<OutIter>, int> = 0>
void negate(Iter first, Iter last, OutIter out) {
  simple_unary(first, last, out, [](auto x) { return -x; });
}

// -x, in-place version
template <class Iter, std::enable_if_t<is_iterator_v<Iter>, int> = 0>
void negate(Iter first, Iter last) {
  negate(first, last, first);
}

// log(x + 1)
template <class Iter, class OutIter, std::enable_if_t<is_iterator_v<Iter>, int> = 0,
          std::enable_if_t<is_iterator_v<OutIter>, int> = 0>
void log1(Iter first, Iter last, OutIter out) {
  simple_unary(first, last, out, [](auto x) { return std::log(x + 1); });
}

// log(x + 1), in-place version
template <class Iter, std::enable_if_t<is_iterator_v<Iter>, int> = 0>
void log1(Iter first, Iter last) {
  log1(first, last, first);
}

// 1 / (1 + exp(-x))
template <class Iter, class OutIter, std::enable_if_t<is_iterator_v<Iter>, int> = 0,
          std::enable_if_t<is_iterator_v<OutIter>, int> = 0>
void sigmoid(Iter first, Iter last, OutIter out) {
  simple_unary(first, last, out, [](auto x) { return 1 / (1 + std::exp(-x)); });
}

// 1 / (1 + exp(-x)), in-place version
template <class Iter, std::enable_if_t<is_iterator_v<Iter>, int> = 0>
void sigmoid(Iter first, Iter last) {
  sigmoid(first, last, first);
}

// sign
template <class Iter, class OutIter, std::enable_if_t<is_iterator_v<Iter>, int> = 0,
          std::enable_if_t<is_iterator_v<OutIter>, int> = 0>
void sign(Iter first, Iter last, OutIter out) {
  simple_unary(first, last, out, [](auto x) { return sign(x); });
}

// sign, in-place version
template <class Iter, std::enable_if_t<is_iterator_v<Iter>, int> = 0>
void sign(Iter first, Iter last) {
  sign(first, last, first);
}

// sinh
template <class Iter, class OutIter, std::enable_if_t<is_iterator_v<Iter>, int> = 0,
          std::enable_if_t<is_iterator_v<OutIter>, int> = 0>
void sinh(Iter first, Iter last, OutIter out) {
  simple_unary(first, last, out, [](auto x) { return std::sinh(x); });
}

// sinh, in-place version
template <class Iter, std::enable_if_t<is_iterator_v<Iter>, int> = 0>
void sinh(Iter first, Iter last) {
  sinh(first, last, first);
}

// tanh
template <class Iter, class OutIter, std::enable_if_t<is_iterator_v<Iter>, int> = 0,
          std::enable_if_t<is_iterator_v<OutIter>, int> = 0>
void tanh(Iter first, Iter last, OutIter out) {
  simple_unary(first, last, out, [](auto x) { return std::tanh(x); });
}

// tanh, in-place version
template <class Iter, std::enable_if_t<is_iterator_v<Iter>, int> = 0>
void tanh(Iter first, Iter last) {
  tanh(first, last, first);
}

// Sign-preserving Power
template <class Iter, class OutIter, class E, std::enable_if_t<is_iterator_v<Iter>, int> = 0,
          std::enable_if_t<is_iterator_v<OutIter>, int> = 0>
void spow(Iter first, Iter last, OutIter out, E exp) {
  simple_unary(first, last, out, [&](auto x) { return sign(x) * detail::pow(std::abs(x), exp); });
}

// Sign-preserving Power, in-place version
template <class Iter, class E, std::enable_if_t<is_iterator_v<Iter>, int> = 0>
void spow(Iter first, Iter last, E exp) {
  spow(first, last, first, exp);
}

// Power
template <class Iter, class OutIter, class E, std::enable_if_t<is_iterator_v<Iter>, int> = 0,
          std::enable_if_t<is_iterator_v<OutIter>, int> = 0>
void pow(Iter first, Iter last, OutIter out, E exp) {
  simple_unary(first, last, out, [&](auto x) { return detail::pow(x, exp); });
}

// Power, in-place version
template <class Iter, class E, std::enable_if_t<is_iterator_v<Iter>, int> = 0>
void pow(Iter first, Iter last, E exp) {
  pow(first, last, first, exp);
}

// Rank
template <class Iter, class OutIter, class Allocator = std::allocator<int>,
          std::enable_if_t<is_random_access_iterator_v<Iter>, int> = 0,
          std::enable_if_t<is_random_access_iterator_v<OutIter>, int> = 0>
void rank(Iter first, Iter last, OutIter out, iter_value_t<OutIter> eps = 0,
          const Allocator &alloc = Allocator()) {
  DefaultCheck is_valid;
  std::vector<int, Allocator> idx(alloc);
  idx.reserve(last - first);
  for (int i = 0; first + i != last; ++i) {
    if (is_valid(*(first + i)))
      idx.push_back(i);
    else
      *(out + i) = *(first + i);
  }
  if (idx.size() == 0) return;
  if (idx.size() == 1) {
    *(out + idx[0]) = 0.5;
    return;
  }
  std::sort(idx.begin(), idx.end(), [&](auto x, auto y) { return *(first + x) < *(first + y); });

  using Value = iter_value_t<Iter>;
  int n = idx.size();
  Value scale = 1 / 2.0 / (n - 1);
  int i = 0;
  while (i < n) {
    int j = i + 1;
    while (j < n && float_eq_pct(*(first + idx[i]), *(first + idx[j]), eps)) ++j;
    Value rk = (i + j - 1) * scale;
    while (i < j) {
      *(out + idx[i]) = rk;
      ++i;
    }
  }
}

// Rank, in-place version
template <class Iter, class Allocator = std::allocator<int>,
          std::enable_if_t<is_random_access_iterator_v<Iter>, int> = 0>
void rank(Iter first, Iter last, iter_value_t<Iter> eps = 0, const Allocator &alloc = Allocator()) {
  rank(first, last, first, eps);
}

// rank then spow
template <class Iter, class OutIter, class Allocator = std::allocator<int>,
          std::enable_if_t<is_random_access_iterator_v<Iter>, int> = 0,
          std::enable_if_t<is_random_access_iterator_v<OutIter>, int> = 0>
void rank_pow(Iter first, Iter last, OutIter out, iter_value_t<OutIter> exp,
              iter_value_t<OutIter> eps = 0, const Allocator &alloc = Allocator()) {
  rank(first, last, out, eps, alloc);
  int n = last - first;
  for (int i = 0; i < n; ++i) *(out + i) -= 0.5;
  spow(out, out + n, exp);
}

// rank then spow, in-place version
template <class Iter, class Allocator = std::allocator<int>,
          std::enable_if_t<is_random_access_iterator_v<Iter>, int> = 0>
void rank_pow(Iter first, Iter last, iter_value_t<Iter> exp, iter_value_t<Iter> eps = 0,
              const Allocator &alloc = Allocator()) {
  rank_pow(first, last, first, exp, eps, alloc);
}

// Demean
template <class ValidCheck = DefaultCheck, class Iter, class OutIter,
          std::enable_if_t<is_iterator_v<Iter>, int> = 0,
          std::enable_if_t<is_valid_check_v<ValidCheck>, int> = 0>
void demean(Iter first, Iter last, OutIter out) {
  if (first == last) return;
  auto mean = ops::mean<ValidCheck>(first, last);
  ValidCheck is_valid;
  for (auto it = first; it != last; ++it, ++out) {
    if (is_valid(*it)) {
      auto v = *it - mean;
      *out = v;
    } else {
      *out = *it;
    }
  }
}

// Demean, in-place version
template <class ValidCheck = DefaultCheck, class Iter,
          std::enable_if_t<is_iterator_v<Iter>, int> = 0,
          std::enable_if_t<is_valid_check_v<ValidCheck>, int> = 0>
void demean(Iter first, Iter last) {
  demean<ValidCheck>(first, last, first);
}

// Shift elements to the right (or left if periods is negative).
// A'(i + periods) = A(i)
template <class Iter, class OutIter, std::enable_if_t<is_random_access_iterator_v<Iter>, int> = 0,
          std::enable_if_t<is_random_access_iterator_v<OutIter>, int> = 0>
void shift(Iter first, Iter last, OutIter out, int periods,
           iter_value_t<OutIter> fill_value = detail::default_iter_value<OutIter>()) {
  int n = last - first;
  if (periods > 0) {
    for (int i = n - 1; i >= periods; --i) {
      out[i] = first[i - periods];
    }
    for (int i = 0; i < periods; ++i) out[i] = fill_value;
  } else if (periods < 0) {
    periods = -periods;
    for (int i = 0; i < n - periods; ++i) {
      out[i] = first[i + periods];
    }
    for (int i = n - periods; i < n; ++i) out[i] = fill_value;
  }
}

// Shift elements to the right (or left if periods is negative), in-place
// version
// A'(i + periods) = A(i)
template <class Iter, std::enable_if_t<is_iterator_v<Iter>, int> = 0>
void shift(Iter first, Iter last, int periods,
           iter_value_t<Iter> fill_value = detail::default_iter_value<Iter>()) {
  shift(first, last, first, periods, fill_value);
}

// Z-Score: (x - mean) / std
template <class ValidCheck = DefaultCheck, class Iter, class OutIter,
          std::enable_if_t<is_iterator_v<Iter>, int> = 0,
          std::enable_if_t<is_valid_check_v<ValidCheck>, int> = 0>
void zscore(Iter first, Iter last, OutIter out) {
  auto mean = ops::mean<ValidCheck>(first, last);
  auto std = ops::stdev<ValidCheck>(first, last);
  if (std == 0) std = NAN;
  for (auto it = first; it != last; ++it, ++out) {
    *out = (*it - mean) / std;
  }
}

// Z-Score, in-place version
template <class ValidCheck = DefaultCheck, class Iter,
          std::enable_if_t<is_iterator_v<Iter>, int> = 0,
          std::enable_if_t<is_valid_check_v<ValidCheck>, int> = 0>
void zscore(Iter first, Iter last) {
  zscore<ValidCheck>(first, last, first);
}

// Truncate to `(mean - std * cap, mean + std *cap)`
template <class ValidCheck = DefaultCheck, class Iter, class OutIter,
          std::enable_if_t<is_iterator_v<Iter>, int> = 0,
          std::enable_if_t<is_valid_check_v<ValidCheck>, int> = 0>
void truncate(Iter first, Iter last, OutIter out, iter_value_t<OutIter> cap) {
  auto mean = ops::mean<ValidCheck>(first, last);
  auto std = ops::stdev<ValidCheck>(first, last);
  auto upper_th = mean + std * cap;
  auto lower_th = mean - std * cap;
  for (auto it = first; it != last; ++it, ++out) {
    if (*it > upper_th) {
      *out = upper_th;
    } else if (*it < lower_th) {
      *out = lower_th;
    } else {
      *out = *it;
    }
  }
}

// Truncate to `(mean - std * cap, mean + std *cap)`, in-place version
template <class ValidCheck = DefaultCheck, class Iter,
          std::enable_if_t<is_iterator_v<Iter>, int> = 0,
          std::enable_if_t<is_valid_check_v<ValidCheck>, int> = 0>
void truncate(Iter first, Iter last, iter_value_t<Iter> cap) {
  truncate(first, last, first, cap);
}

// Truncate up to `mean + std * cap`
template <class ValidCheck = DefaultCheck, class Iter, class OutIter,
          std::enable_if_t<is_iterator_v<Iter>, int> = 0,
          std::enable_if_t<is_valid_check_v<ValidCheck>, int> = 0>
void truncate_upper(Iter first, Iter last, OutIter out, iter_value_t<OutIter> cap) {
  auto mean = ops::mean<ValidCheck>(first, last);
  auto std = ops::stdev<ValidCheck>(first, last);
  auto threshold = mean + std * cap;
  for (auto it = first; it != last; ++it, ++out) {
    if (*it > threshold)
      *out = threshold;
    else
      *out = *it;
  }
}

// Truncate up to `mean + std * cap`, in-place version
template <class ValidCheck = DefaultCheck, class Iter,
          std::enable_if_t<is_iterator_v<Iter>, int> = 0,
          std::enable_if_t<is_valid_check_v<ValidCheck>, int> = 0>
void truncate_upper(Iter first, Iter last, iter_value_t<Iter> cap) {
  truncate_upper(first, last, first, cap);
}

// Sig-window: apply zscore and set all absolute values exceeding cap1 to cap1, and
// all absolute values exceeding cap2 to NAN
template <class ValidCheck = DefaultCheck, class Iter, class OutIter,
          std::enable_if_t<is_iterator_v<Iter>, int> = 0,
          std::enable_if_t<is_valid_check_v<ValidCheck>, int> = 0>
void sigwin(Iter first, Iter last, OutIter out, iter_value_t<OutIter> cap1,
            iter_value_t<OutIter> cap2) {
  zscore<ValidCheck>(first, last, out);
  ValidCheck is_valid;
  for (auto it = first; it != last; ++it, ++out) {
    auto v = *out;
    if (!is_valid(v)) {
      *out = v;
    } else if (v > cap2 || v < -cap2) {
      *out = NAN;
    } else if (v > cap1) {
      *out = cap1;
    } else if (v < -cap1) {
      *out = -cap1;
    } else {
      *out = v;
    }
  }
}

// Sig-window, in-place version
template <class ValidCheck = DefaultCheck, class Iter,
          std::enable_if_t<is_iterator_v<Iter>, int> = 0,
          std::enable_if_t<is_valid_check_v<ValidCheck>, int> = 0>
void sigwin(Iter first, Iter last, iter_value_t<Iter> cap1, iter_value_t<Iter> cap2) {
  sigwin<ValidCheck>(first, last, first, cap1, cap2);
}

// Sig-window upper: apply zscore and set all values exceeding cap1 to cap1, and
// all values exceeding cap2 to NAN
template <class ValidCheck = DefaultCheck, class Iter, class OutIter,
          std::enable_if_t<is_iterator_v<Iter>, int> = 0,
          std::enable_if_t<is_valid_check_v<ValidCheck>, int> = 0>
void sigwin_upper(Iter first, Iter last, OutIter out, iter_value_t<OutIter> cap1,
                  iter_value_t<OutIter> cap2) {
  zscore<ValidCheck>(first, last, out);
  ValidCheck is_valid;
  for (auto it = first; it != last; ++it, ++out) {
    auto v = *out;
    if (!is_valid(v)) {
      *out = v;
    } else if (v > cap2) {
      *out = NAN;
    } else if (v > cap1) {
      *out = cap1;
    } else {
      *out = v;
    }
  }
}

// Sig-window upper, in-place version
template <class ValidCheck = DefaultCheck, class Iter,
          std::enable_if_t<is_iterator_v<Iter>, int> = 0,
          std::enable_if_t<is_valid_check_v<ValidCheck>, int> = 0>
void sigwin_upper(Iter first, Iter last, iter_value_t<Iter> cap1, iter_value_t<Iter> cap2) {
  sigwin_upper<ValidCheck>(first, last, first, cap1, cap2);
}

// Scale the sum of abs to scale_size
template <class Iter, class OutIter, std::enable_if_t<is_iterator_v<Iter>, int> = 0,
          std::enable_if_t<is_iterator_v<OutIter>, int> = 0>
void scale(Iter first, Iter last, OutIter out, iter_value_t<OutIter> scale_size,
           iter_value_t<OutIter> eps = 0) {
  DefaultCheck is_valid;

  using T = iter_value_t<OutIter>;
  T abs_sum = 0;
  for (auto it = first; it != last; ++it) {
    if (is_valid(*it)) abs_sum += std::abs(*it);
  }
  T coef = abs_sum >= eps ? scale_size / abs_sum : 0;
  for (auto it = first; it != last; ++it, ++out) {
    if (is_valid(*it)) {
      *out = *it * coef;
    } else {
      *out = *it;
    }
  }
}

// Scale the sum of abs to scale_size, in-place version
template <class Iter, std::enable_if_t<is_iterator_v<Iter>, int> = 0>
void scale(Iter first, Iter last, iter_value_t<Iter> scale_size, iter_value_t<Iter> eps = 0) {
  scale(first, last, first, scale_size, eps);
}

// Keep the largest n values (starting from 1st) and set others to null_value
// cmp: returns 1, 0, -1 for (>, =, <)
template <class Iter, class OutIter, class Compare, class Allocator = std::allocator<int>,
          std::enable_if_t<is_random_access_iterator_v<Iter>, int> = 0,
          std::enable_if_t<is_random_access_iterator_v<OutIter>, int> = 0,
          std::enable_if_t<!std::is_same_v<Compare, iter_value_t<Iter>>, int> = 0>
void select_n_largest(Iter first, Iter last, OutIter out, int n, Compare cmp,
                      iter_value_t<OutIter> null_value = detail::default_iter_value<OutIter>(),
                      const Allocator &alloc = Allocator()) {
  DefaultCheck is_valid;
  std::vector<int, Allocator> idx(alloc);
  int size = last - first;
  idx.reserve(size);
  for (int i = 0; i < size; ++i) {
    if (is_valid(first[i])) {
      idx.push_back(i);
    } else {
      out[i] = null_value;
    }
  }
  if (static_cast<int>(idx.size()) > n) {
    std::nth_element(idx.begin(), idx.begin() + n, idx.end(), [&](const auto &i, const auto &j) {
      int c = cmp(first[i], first[j]);
      if (c == 0) {
        return i < j;
      } else {
        return c > 0 ? true : false;
      }
    });
    for (int i = 0; i < n; ++i) {
      out[idx[i]] = first[idx[i]];
    }
    for (int i = n; i < static_cast<int>(idx.size()); ++i) {
      out[idx[i]] = null_value;
    }
  } else {
    for (auto i : idx) out[i] = first[i];
  }
}

// Keep the largest n values (starting from 1st) and set others to null_value
template <class Iter, class OutIter, class Allocator = std::allocator<int>,
          std::enable_if_t<is_random_access_iterator_v<Iter>, int> = 0,
          std::enable_if_t<is_random_access_iterator_v<OutIter>, int> = 0>
void select_n_largest(Iter first, Iter last, OutIter out, int n,
                      iter_value_t<OutIter> null_value = detail::default_iter_value<OutIter>(),
                      const Allocator &alloc = Allocator()) {
  using T = iter_value_t<Iter>;
  auto cmp = [](const T &x, const T &y) -> int {
    if constexpr (std::is_integral_v<T>) {
      return x - y;
    } else {
      return x > y ? 1 : (x == y ? 0 : -1);
    }
  };
  select_n_largest(first, last, out, n, cmp, null_value, alloc);
}

// Keep the largest n values (starting from 1st) and set others to null_value, in-place version
template <class Iter, class Compare, class Allocator = std::allocator<int>,
          std::enable_if_t<is_random_access_iterator_v<Iter>, int> = 0,
          std::enable_if_t<!std::is_same_v<Compare, iter_value_t<Iter>>, int> = 0>
void select_n_largest(Iter first, Iter last, int n, Compare cmp,
                      iter_value_t<Iter> null_value = detail::default_iter_value<Iter>(),
                      const Allocator &alloc = Allocator()) {
  select_n_largest(first, last, first, n, cmp, null_value, alloc);
}

// Keep the largest n values (starting from 1st) and set others to null_value, in-place version
template <class Iter, class Allocator = std::allocator<int>,
          std::enable_if_t<is_random_access_iterator_v<Iter>, int> = 0>
void select_n_largest(Iter first, Iter last, int n,
                      iter_value_t<Iter> null_value = detail::default_iter_value<Iter>(),
                      const Allocator &alloc = Allocator()) {
  select_n_largest(first, last, first, n, null_value, alloc);
}

}  // namespace yang::math::ops
