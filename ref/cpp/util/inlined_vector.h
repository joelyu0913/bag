#pragma once

#include "absl/container/inlined_vector.h"

namespace yang {

template <class T, size_t N>
using InlinedVector = absl::InlinedVector<T, N>;

}  // namespace yang
