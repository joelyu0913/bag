#pragma once

#include <iostream>
#include <vector>

#include "yang/util/fmt.h"

namespace std {

template <class T>
ostream &operator<<(ostream &os, const vector<T> &v) {
  for (int i = 0; i < static_cast<int>(v.size()); ++i) {
    if (i) os << ' ';
    os << v[i];
  }
  return os;
}

}  // namespace std

#ifdef FMT_OSTREAM_FORMATTER
template <class T>
struct fmt::formatter<std::vector<T>> : ostream_formatter {};
#endif
