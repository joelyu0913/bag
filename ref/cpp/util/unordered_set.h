#pragma once

#include <vector>

#include "absl/container/flat_hash_set.h"

namespace yang {

// NOTE: abseil flat_hash_set's iteration order is undeterministic
template <class K, class Hash = absl::container_internal::hash_default_hash<K>>
class unordered_set : public absl::flat_hash_set<K, Hash> {
 public:
  using absl::flat_hash_set<K, Hash>::flat_hash_set;

  unordered_set(const std::vector<K> &vec) : unordered_set(vec.begin(), vec.end()) {}
};

}  // namespace yang
