#pragma once

#include <algorithm>
#include <type_traits>
#include <vector>

#include "yang/math/ops/base.h"
#include "yang/math/ops/unary.h"
#include "yang/math/type_traits.h"

namespace yang::math::ops {

// Hedge long positions on hedge_idx
template <class Iter, class OutIter, std::enable_if_t<is_random_access_iterator_v<Iter>, int> = 0,
          std::enable_if_t<is_random_access_iterator_v<OutIter>, int> = 0>
void hedge(Iter first, Iter last, OutIter out, int univ_size, int hedge_idx) {
  DefaultCheck is_valid;
  iter_value_t<Iter> long_sum = 0;
  int n = last - first;
  for (int i = 0; i < univ_size; ++i) {
    auto v = *(first + i);
    if (is_valid(v)) {
      if (v < 0) {
        *(out + i) = 0;
      } else {
        long_sum += v;
        *(out + i) = v;
      }
    } else {
      *(out + i) = v;
    }
  }
  for (int i = univ_size; i < n; ++i) {
    *(out + i) = *(first + i);
  }
  if (hedge_idx >= 0) {
    *(out + hedge_idx) = -long_sum;
  }
}

// Hedge long positions on hedge_idx, in-place version
template <class Iter, std::enable_if_t<is_random_access_iterator_v<Iter>, int> = 0>
void hedge(Iter first, Iter last, int univ_size, int hedge_idx) {
  hedge(first, last, first, univ_size, hedge_idx);
}

// Hedge short positions on hedge_idx
template <class Iter, class OutIter, std::enable_if_t<is_random_access_iterator_v<Iter>, int> = 0,
          std::enable_if_t<is_random_access_iterator_v<OutIter>, int> = 0>
void hedgeshort(Iter first, Iter last, OutIter out, int univ_size, int hedge_idx) {
  DefaultCheck is_valid;
  iter_value_t<Iter> short_sum = 0;
  int n = last - first;
  for (int i = 0; i < univ_size; ++i) {
    auto v = *(first + i);
    if (is_valid(v)) {
      if (v > 0) {
        *(out + i) = 0;
      } else {
        short_sum += v;
        *(out + i) = v;
      }
    } else {
      *(out + i) = v;
    }
  }
  for (int i = univ_size; i < n; ++i) {
    *(out + i) = *(first + i);
  }
  if (hedge_idx >= 0) {
    *(out + hedge_idx) = -short_sum;
  }
}

// Hedge long positions on hedge_idx, in-place version
template <class Iter, std::enable_if_t<is_random_access_iterator_v<Iter>, int> = 0>
void hedgeshort(Iter first, Iter last, int univ_size, int hedge_idx) {
  hedgeshort(first, last, first, univ_size, hedge_idx);
}

// Upbound
template <class Iter, class OutIter, class Allocator = std::allocator<int>,
          std::enable_if_t<is_random_access_iterator_v<Iter>, int> = 0,
          std::enable_if_t<is_random_access_iterator_v<OutIter>, int> = 0>
void upbound(Iter first, Iter last, OutIter out, int k, iter_value_t<OutIter> eps = EPSILON,
             const Allocator &alloc = Allocator()) {
  DefaultCheck is_valid;
  using T = iter_value_t<OutIter>;
  T sum = 0;
  int n = last - first;
  std::vector<int, Allocator> idx(alloc);
  idx.reserve(n);
  for (int i = 0; i < n; ++i) {
    if (is_valid(first[i])) {
      idx.push_back(i);
      sum += first[i];
    } else {
      out[i] = first[i];
    }
  }
  int idx_size = idx.size();
  if (idx_size >= k * 2) {
    std::sort(idx.begin(), idx.end(), [&](int i, int j) { return first[i] < first[j]; });

    auto lower = first[idx[k - 1]];
    auto upper = first[idx[idx_size - k]];
    for (int i = 0; i < k; ++i) {
      sum += lower - first[idx[i]];
    }
    for (int i = idx_size - k; i < idx_size; ++i) {
      sum += upper - first[idx[i]];
    }
    sum -= lower * idx_size;
    T ratio = idx_size * 0.5 / sum;
    for (int i = 0; i < idx_size; ++i) {
      T val;
      if (i < k) {
        val = lower;
      } else if (i < idx_size - k) {
        val = first[idx[i]];
      } else {
        val = upper;
      }
      out[idx[i]] = (val - lower) * ratio;
    }
  } else {
    rank(first, last, out, eps, alloc);
  }
}

// Upbound, in-place version
template <class Iter, class Allocator = std::allocator<int>,
          std::enable_if_t<is_random_access_iterator_v<Iter>, int> = 0>
void upbound(Iter first, Iter last, int k, iter_value_t<Iter> eps = EPSILON,
             const Allocator &alloc = Allocator()) {
  upbound(first, last, first, k, eps, alloc);
}

}  // namespace yang::math::ops
