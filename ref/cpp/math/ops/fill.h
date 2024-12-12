#pragma once

#include <cmath>
#include <type_traits>

#include "yang/math/ops/base.h"
#include "yang/math/type_traits.h"

namespace yang::math::ops {

// Fixed horizon forward filling for invalid values
template <class ValidCheck = DefaultCheck, class Iter, class OutIter,
          std::enable_if_t<is_iterator_v<Iter>, int> = 0,
          std::enable_if_t<is_iterator_v<OutIter>, int> = 0>
void ffill(Iter first, Iter last, OutIter out, int horizon) {
  iter_value_t<OutIter> last_val = detail::default_iter_value<OutIter>();
  int last_idx = -1;
  int i = 0;
  ValidCheck is_valid;
  for (auto it = first; it != last; ++it, ++i, ++out) {
    auto v = *it;
    if (is_valid(v)) {
      *out = v;
      last_val = v;
      last_idx = i;
    } else if (i - last_idx <= horizon) {
      *out = last_val;
    } else {
      *out = v;
    }
  }
}

// Fixed horizon forward filling for invalid values, in-place version
template <class ValidCheck = DefaultCheck, class Iter,
          std::enable_if_t<is_iterator_v<Iter>, int> = 0>
void ffill(Iter first, Iter last, int horizon) {
  ffill<ValidCheck>(first, last, first, horizon);
}

// Fixed horizon linear interpolation for invalid values
template <class ValidCheck = DefaultCheck, class Iter, class OutIter,
          std::enable_if_t<is_random_access_iterator_v<Iter>, int> = 0,
          std::enable_if_t<is_random_access_iterator_v<OutIter>, int> = 0>
void linear_fill(Iter first, Iter last, OutIter out, int horizon) {
  ValidCheck is_valid;
  int n = last - first;
  int i = 0;
  // skip invalid values at the start
  while (i < n && !is_valid(first[i])) {
    out[i] = first[i];
    ++i;
  }
  while (true) {
    // copy valid values
    while (i < n && is_valid(first[i])) {
      out[i] = first[i];
      ++i;
    }
    if (i >= n) break;
    // i: first invalid value
    int j = i + 1;
    while (j < n && !is_valid(first[j])) ++j;
    // j: first valid value after i

    if (j < n && j - i <= horizon) {
      // fill invalid values
      auto step = (first[j] - first[i - 1]) / (j - i + 1);
      for (int k = i; k < j; ++k) {
        out[k] = out[i - 1] + step * (k - i + 1);
      }
    } else {
      for (int k = i; k < j; ++k) {
        out[k] = first[k];
      }
    }
    i = j;
  }
}

template <class ValidCheck = DefaultCheck, class Iter,
          std::enable_if_t<is_random_access_iterator_v<Iter>, int> = 0>
void linear_fill(Iter first, Iter last, int horizon) {
  linear_fill<ValidCheck>(first, last, first, horizon);
}

}  // namespace yang::math::ops
