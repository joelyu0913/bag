#pragma once

#include "absl/container/flat_hash_map.h"

namespace yang {

// NOTE: abseil flat_hash_map's iteration order is undeterministic
template <class K, class V, class Hash = absl::container_internal::hash_default_hash<K>>
using unordered_map = absl::flat_hash_map<K, V, Hash>;

}  // namespace yang
