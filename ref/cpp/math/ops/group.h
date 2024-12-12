#pragma once

#include <type_traits>
#include <vector>

#include "yang/base/inline.h"
#include "yang/math/ops/base.h"
#include "yang/math/ops/unary.h"
#include "yang/math/type_traits.h"

namespace yang::math::ops {

template <class ValidCheck = DefaultCheck, class Iter, class GIter, class OutIter, class Func,
          class Allocator = std::allocator<int>,
          std::enable_if_t<is_random_access_iterator_v<Iter>, int> = 0,
          std::enable_if_t<is_valid_check_v<ValidCheck>, int> = 0,
          std::enable_if_t<std::is_integral_v<iter_value_t<GIter>>, int> = 0>
ALWAYS_INLINE void simple_group(Iter first, Iter last, GIter g_first, OutIter out, Func &&f,
                                const Allocator &alloc = Allocator()) {
  int n = last - first;
  int g_max = -1;
  for (int i = 0; i < n; ++i) {
    auto g = *(g_first + i);
    if (g > g_max) g_max = g;
  }
  ++g_max;
  if (g_max <= 0) return;

  using T = iter_value_t<OutIter>;
  using PAllocator =
      typename std::allocator_traits<Allocator>::template rebind_alloc<std::vector<int, Allocator>>;
  ValidCheck is_valid;
  std::vector<std::vector<int, Allocator>, PAllocator> group_members(g_max + 1);
  for (int i = 0; i < n; ++i) {
    auto g = *(g_first + i);
    auto v = *(first + i);
    if (g >= 0 && is_valid(v)) {
      group_members[g].push_back(i);
    } else {
      *(out + i) = detail::default_value<T>();
    }
  }
  std::vector<T, typename std::allocator_traits<Allocator>::template rebind_alloc<T>> values;
  for (auto &group : group_members) {
    values.reserve(group.size());
    for (auto &i : group) values.push_back(*(first + i));
    f(values);
    for (int i = 0; i < static_cast<int>(group.size()); ++i) {
      *(out + group[i]) = values[i];
    }
    values.clear();
  }
}

// Momentum by group
template <class ValidCheck = DefaultCheck, class Iter, class GIter, class OutIter,
          class Allocator = std::allocator<int>,
          std::enable_if_t<is_random_access_iterator_v<Iter>, int> = 0,
          std::enable_if_t<is_valid_check_v<ValidCheck>, int> = 0,
          std::enable_if_t<std::is_integral_v<iter_value_t<GIter>>, int> = 0>
void group_mom(Iter first, Iter last, GIter g_first, OutIter out,
               const Allocator &alloc = Allocator()) {
  int n = last - first;
  int g_max = -1;
  for (int i = 0; i < n; ++i) {
    auto g = *(g_first + i);
    if (g > g_max) g_max = g;
  }
  ++g_max;
  if (g_max <= 0) return;

  using Value = iter_value_t<OutIter>;
  using PAllocator =
      typename std::allocator_traits<Allocator>::template rebind_alloc<std::pair<Value, int>>;
  std::vector<std::pair<Value, int>, PAllocator> g_mean(g_max, {0, 0}, PAllocator(alloc));
  ValidCheck is_valid;
  for (int i = 0; i < n; ++i) {
    auto g = *(g_first + i);
    auto v = *(first + i);
    if (g >= 0 && is_valid(v)) {
      g_mean[g].first += v;
      ++g_mean[g].second;
    }
  }
  for (auto &[sum, count] : g_mean) {
    if (count > 0) sum /= count;
  }
  for (int i = 0; i < n; ++i) {
    auto g = *(g_first + i);
    auto v = *(first + i);
    if (g >= 0 && is_valid(v)) {
      *(out + i) = g_mean[g].first;
    } else {
      *(out + i) = detail::default_value<Value>();
    }
  }
}

// Momentum by group, in-place version
template <class ValidCheck = DefaultCheck, class Iter, class GIter,
          class Allocator = std::allocator<int>,
          std::enable_if_t<is_random_access_iterator_v<Iter>, int> = 0,
          std::enable_if_t<is_valid_check_v<ValidCheck>, int> = 0,
          std::enable_if_t<is_allocator_v<Allocator>, int> = 0>
void group_mom(Iter first, Iter last, GIter g_first, const Allocator &alloc = Allocator()) {
  group_mom<ValidCheck>(first, last, g_first, first, alloc);
}

// Demean by group
template <class ValidCheck = DefaultCheck, class Iter, class GIter, class OutIter,
          class Allocator = std::allocator<int>,
          std::enable_if_t<is_random_access_iterator_v<Iter>, int> = 0,
          std::enable_if_t<is_valid_check_v<ValidCheck>, int> = 0,
          std::enable_if_t<std::is_integral_v<iter_value_t<GIter>>, int> = 0>
void group_demean(Iter first, Iter last, GIter g_first, OutIter out,
                  const Allocator &alloc = Allocator()) {
  int n = last - first;
  int g_max = -1;
  for (int i = 0; i < n; ++i) {
    auto g = *(g_first + i);
    if (g > g_max) g_max = g;
  }
  ++g_max;
  if (g_max <= 0) return;

  using Value = iter_value_t<OutIter>;
  using PAllocator =
      typename std::allocator_traits<Allocator>::template rebind_alloc<std::pair<Value, int>>;
  std::vector<std::pair<Value, int>, PAllocator> g_mean(g_max, {0, 0}, PAllocator(alloc));
  ValidCheck is_valid;
  for (int i = 0; i < n; ++i) {
    auto g = *(g_first + i);
    auto v = *(first + i);
    if (g >= 0 && is_valid(v)) {
      g_mean[g].first += v;
      ++g_mean[g].second;
    }
  }
  for (auto &[sum, count] : g_mean) {
    if (count > 0) sum /= count;
  }
  for (int i = 0; i < n; ++i) {
    auto g = *(g_first + i);
    auto v = *(first + i);
    if (g >= 0 && is_valid(v)) {
      v -= g_mean[g].first;
      *(out + i) = v;
    } else {
      *(out + i) = detail::default_value<Value>();
    }
  }
}

// Demean by group, in-place version
template <class ValidCheck = DefaultCheck, class Iter, class GIter,
          class Allocator = std::allocator<int>,
          std::enable_if_t<is_random_access_iterator_v<Iter>, int> = 0,
          std::enable_if_t<is_valid_check_v<ValidCheck>, int> = 0,
          std::enable_if_t<is_allocator_v<Allocator>, int> = 0>
void group_demean(Iter first, Iter last, GIter g_first, const Allocator &alloc = Allocator()) {
  group_demean<ValidCheck>(first, last, g_first, first, alloc);
}

// Rank by group
template <class ValidCheck = DefaultCheck, class Iter, class GIter, class OutIter,
          class Allocator = std::allocator<int>,
          std::enable_if_t<is_random_access_iterator_v<Iter>, int> = 0,
          std::enable_if_t<is_valid_check_v<ValidCheck>, int> = 0,
          std::enable_if_t<std::is_integral_v<iter_value_t<GIter>>, int> = 0>
void group_rank(Iter first, Iter last, GIter g_first, OutIter out,
                iter_value_t<OutIter> eps = EPSILON, const Allocator &alloc = Allocator()) {
  simple_group<ValidCheck>(
      first, last, g_first, out, [&](auto &g) { rank(g.begin(), g.end(), eps, alloc); }, alloc);
}

// Rank by group, in-place version
template <class ValidCheck = DefaultCheck, class Iter, class GIter,
          class Allocator = std::allocator<int>,
          std::enable_if_t<is_random_access_iterator_v<Iter>, int> = 0,
          std::enable_if_t<is_valid_check_v<ValidCheck>, int> = 0,
          std::enable_if_t<is_allocator_v<Allocator>, int> = 0>
void group_rank(Iter first, Iter last, GIter g_first, iter_value_t<Iter> eps = EPSILON,
                const Allocator &alloc = Allocator()) {
  group_rank<ValidCheck>(first, last, g_first, first, eps, alloc);
}

// rank_pow by group
template <class ValidCheck = DefaultCheck, class Iter, class GIter, class OutIter,
          class Allocator = std::allocator<int>,
          std::enable_if_t<is_random_access_iterator_v<Iter>, int> = 0,
          std::enable_if_t<is_valid_check_v<ValidCheck>, int> = 0,
          std::enable_if_t<std::is_integral_v<iter_value_t<GIter>>, int> = 0>
void group_rank_pow(Iter first, Iter last, GIter g_first, OutIter out, iter_value_t<OutIter> exp,
                    iter_value_t<OutIter> eps = EPSILON, const Allocator &alloc = Allocator()) {
  simple_group<ValidCheck>(
      first, last, g_first, out, [&](auto &g) { rank_pow(g.begin(), g.end(), exp, eps, alloc); },
      alloc);
}

// rank_pow by group, in-place version
template <class ValidCheck = DefaultCheck, class Iter, class GIter,
          class Allocator = std::allocator<int>,
          std::enable_if_t<is_random_access_iterator_v<Iter>, int> = 0,
          std::enable_if_t<is_valid_check_v<ValidCheck>, int> = 0,
          std::enable_if_t<is_allocator_v<Allocator>, int> = 0>
void group_rank_pow(Iter first, Iter last, GIter g_first, iter_value_t<Iter> exp,
                    iter_value_t<Iter> eps = EPSILON, const Allocator &alloc = Allocator()) {
  group_rank_pow<ValidCheck>(first, last, g_first, first, exp, eps, alloc);
}

// zscore by group
template <class ValidCheck = DefaultCheck, class Iter, class GIter, class OutIter,
          class Allocator = std::allocator<int>,
          std::enable_if_t<is_random_access_iterator_v<Iter>, int> = 0,
          std::enable_if_t<is_valid_check_v<ValidCheck>, int> = 0,
          std::enable_if_t<std::is_integral_v<iter_value_t<GIter>>, int> = 0>
void group_zscore(Iter first, Iter last, GIter g_first, OutIter out,
                  const Allocator &alloc = Allocator()) {
  simple_group<ValidCheck>(
      first, last, g_first, out, [&](auto &g) { zscore<ValidCheck>(g.begin(), g.end()); }, alloc);
}

// zscore by group, in-place version
template <class ValidCheck = DefaultCheck, class Iter, class GIter,
          class Allocator = std::allocator<int>,
          std::enable_if_t<is_random_access_iterator_v<Iter>, int> = 0,
          std::enable_if_t<is_valid_check_v<ValidCheck>, int> = 0,
          std::enable_if_t<is_allocator_v<Allocator>, int> = 0>
void group_zscore(Iter first, Iter last, GIter g_first, const Allocator &alloc = Allocator()) {
  group_zscore<ValidCheck>(first, last, g_first, first, alloc);
}

// truncate by group
template <class ValidCheck = DefaultCheck, class Iter, class GIter, class OutIter,
          class Allocator = std::allocator<int>,
          std::enable_if_t<is_random_access_iterator_v<Iter>, int> = 0,
          std::enable_if_t<is_valid_check_v<ValidCheck>, int> = 0,
          std::enable_if_t<std::is_integral_v<iter_value_t<GIter>>, int> = 0>
void group_truncate(Iter first, Iter last, GIter g_first, OutIter out, iter_value_t<OutIter> cap,
                    const Allocator &alloc = Allocator()) {
  simple_group<ValidCheck>(
      first, last, g_first, out, [&](auto &g) { truncate<ValidCheck>(g.begin(), g.end(), cap); },
      alloc);
}

// truncate by group, in-place version
template <class ValidCheck = DefaultCheck, class Iter, class GIter,
          class Allocator = std::allocator<int>,
          std::enable_if_t<is_random_access_iterator_v<Iter>, int> = 0,
          std::enable_if_t<is_valid_check_v<ValidCheck>, int> = 0,
          std::enable_if_t<is_allocator_v<Allocator>, int> = 0>
void group_truncate(Iter first, Iter last, GIter g_first, iter_value_t<Iter> cap,
                    const Allocator &alloc = Allocator()) {
  group_truncate<ValidCheck>(first, last, g_first, first, cap, alloc);
}

// truncate_upper by group
template <class ValidCheck = DefaultCheck, class Iter, class GIter, class OutIter,
          class Allocator = std::allocator<int>,
          std::enable_if_t<is_random_access_iterator_v<Iter>, int> = 0,
          std::enable_if_t<is_valid_check_v<ValidCheck>, int> = 0,
          std::enable_if_t<std::is_integral_v<iter_value_t<GIter>>, int> = 0>
void group_truncate_upper(Iter first, Iter last, GIter g_first, OutIter out,
                          iter_value_t<OutIter> cap, const Allocator &alloc = Allocator()) {
  simple_group<ValidCheck>(
      first, last, g_first, out,
      [&](auto &g) { truncate_upper<ValidCheck>(g.begin(), g.end(), cap); }, alloc);
}

// truncate_upper by group, in-place version
template <class ValidCheck = DefaultCheck, class Iter, class GIter,
          class Allocator = std::allocator<int>,
          std::enable_if_t<is_random_access_iterator_v<Iter>, int> = 0,
          std::enable_if_t<is_valid_check_v<ValidCheck>, int> = 0,
          std::enable_if_t<is_allocator_v<Allocator>, int> = 0>
void group_truncate_upper(Iter first, Iter last, GIter g_first, iter_value_t<Iter> cap,
                          const Allocator &alloc = Allocator()) {
  group_truncate_upper<ValidCheck>(first, last, g_first, first, cap, alloc);
}

// sigwin by group
template <class ValidCheck = DefaultCheck, class Iter, class GIter, class OutIter,
          class Allocator = std::allocator<int>,
          std::enable_if_t<is_random_access_iterator_v<Iter>, int> = 0,
          std::enable_if_t<is_valid_check_v<ValidCheck>, int> = 0,
          std::enable_if_t<std::is_integral_v<iter_value_t<GIter>>, int> = 0>
void group_sigwin(Iter first, Iter last, GIter g_first, OutIter out, iter_value_t<OutIter> cap1,
                  iter_value_t<OutIter> cap2, const Allocator &alloc = Allocator()) {
  simple_group<ValidCheck>(
      first, last, g_first, out,
      [&](auto &g) { sigwin<ValidCheck>(g.begin(), g.end(), cap1, cap2); }, alloc);
}

// sigwin by group, in-place version
template <class ValidCheck = DefaultCheck, class Iter, class GIter,
          class Allocator = std::allocator<int>,
          std::enable_if_t<is_random_access_iterator_v<Iter>, int> = 0,
          std::enable_if_t<is_valid_check_v<ValidCheck>, int> = 0,
          std::enable_if_t<is_allocator_v<Allocator>, int> = 0>
void group_sigwin(Iter first, Iter last, GIter g_first, iter_value_t<Iter> cap1,
                  iter_value_t<Iter> cap2, const Allocator &alloc = Allocator()) {
  group_sigwin<ValidCheck>(first, last, g_first, first, cap1, cap2, alloc);
}

// sigwin_upper by group
template <class ValidCheck = DefaultCheck, class Iter, class GIter, class OutIter,
          class Allocator = std::allocator<int>,
          std::enable_if_t<is_random_access_iterator_v<Iter>, int> = 0,
          std::enable_if_t<is_valid_check_v<ValidCheck>, int> = 0,
          std::enable_if_t<std::is_integral_v<iter_value_t<GIter>>, int> = 0>
void group_sigwin_upper(Iter first, Iter last, GIter g_first, OutIter out,
                        iter_value_t<OutIter> cap1, iter_value_t<OutIter> cap2,
                        const Allocator &alloc = Allocator()) {
  simple_group<ValidCheck>(
      first, last, g_first, out,
      [&](auto &g) { sigwin_upper<ValidCheck>(g.begin(), g.end(), cap1, cap2); }, alloc);
}

// sigwin_upper by group, in-place version
template <class ValidCheck = DefaultCheck, class Iter, class GIter,
          class Allocator = std::allocator<int>,
          std::enable_if_t<is_random_access_iterator_v<Iter>, int> = 0,
          std::enable_if_t<is_valid_check_v<ValidCheck>, int> = 0,
          std::enable_if_t<is_allocator_v<Allocator>, int> = 0>
void group_sigwin_upper(Iter first, Iter last, GIter g_first, iter_value_t<Iter> cap1,
                        iter_value_t<Iter> cap2, const Allocator &alloc = Allocator()) {
  group_sigwin_upper<ValidCheck>(first, last, g_first, first, cap1, cap2, alloc);
}

}  // namespace yang::math::ops
