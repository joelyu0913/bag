#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "yang/base/exception.h"
#include "yang/util/strings.h"
#include "yang/util/unordered_map.h"

namespace yang {

class SimpleCsvReader {
 public:
  static unordered_map<std::string, int> ParseHeader(const std::string &line, char delim);

  SimpleCsvReader(const std::string &line);

  SimpleCsvReader(const std::string &line, char delim);

  void ReadNext(const std::string &line);

  std::string_view operator[](int i) const {
    return row_[i];
  }

  std::string_view operator[](std::string_view name) const {
    auto it = header_.find(name);
    if (it == header_.end()) {
      throw MakeExcept<OutOfRange>("Missing csv column {}", name);
    }
    return row_[it->second];
  }

  template <class T, class K>
  T Get(const K &key) const {
    auto col = (*this)[key];
    return StrConv<T>(col);
  }

  template <class T, class K>
  T Get(const K &key, T fallback) const {
    auto col = (*this)[key];
    return StrConv<T>(col, fallback);
  }

  const unordered_map<std::string, int> &header() const {
    return header_;
  }

  const std::vector<std::string_view> &row() const {
    return row_;
  }

 private:
  char delim_;
  unordered_map<std::string, int> header_;
  std::vector<std::string_view> row_;
};

}  // namespace yang
