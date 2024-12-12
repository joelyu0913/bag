#pragma once

#include "yang/util/deque.h"

namespace yang {

// A deque with fixed capacity unless explicitly reset/reserve.
// When calling push_back/push_front, it is the caller's responsibility to ensure
// there is still empty space.
template <class T, class Allocator = std::allocator<T>>
using fixed_deque = deque<T, false, Allocator, true>;

}  // namespace yang
