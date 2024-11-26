#pragma once

#include <type_traits>

#include "yang/math/ops/base.h"
#include "yang/math/type_traits.h"

namespace yang::math::ops {

// Filter values and replace them with a default value
template <class Iter, class CondIter, class OutIter, std::enable_if_t<is_iterator_v<Iter>, int> = 0,
          std::enable_if_t<is_iterator_v<OutIter>, int> = 0>
void filter(Iter first, Iter last, CondIter cond, OutIter out,
            iter_value_t<OutIter> default_value = detail::default_iter_value<OutIter>()) {
  DefaultCheck is_valid;
  for (auto it = first; it != last; ++it, ++cond, ++out) {
    bool cond_v;
    if constexpr (std::is_floating_point_v<iter_value_t<CondIter>>) {
      cond_v = is_valid(*cond);
    } else {
      cond_v = *cond;
    }
    if (!cond_v) {
      *out = default_value;
    } else {
      *out = *it;
    }
  }
}

// Filter values and replace them with a default value, in-place version
template <class Iter, class CondIter, std::enable_if_t<is_iterator_v<Iter>, int> = 0>
void filter(Iter first, Iter last, CondIter cond,
            iter_value_t<Iter> default_value = detail::default_iter_value<Iter>()) {
  filter(first, last, cond, first, default_value);
}

}  // namespace yang::math::ops
