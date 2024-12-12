#pragma once

#include <cstdint>
#include <limits>

namespace yang::math {

using SizeType = int;
using IndexType = int;

using StaticSize = int;
static constexpr StaticSize DYNAMIC_SIZE = -1;
static constexpr StaticSize DYNAMIC_STRIDE = 0;
static constexpr StaticSize UNSPECIFIED = std::numeric_limits<StaticSize>::min();

}  // namespace yang::math
