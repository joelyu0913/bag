#pragma once

#include <utility>

#include "absl/container/btree_map.h"

namespace yang {

template <class K, class V, class Cmp = std::less<K>>
using ordered_map = absl::btree_map<K, V, Cmp>;

}
