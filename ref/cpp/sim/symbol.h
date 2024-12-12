#pragma once

#include <string>
#include <string_view>

#include "yang/base/exception.h"

namespace yang {

// Convert 000000.SH to sh000000
inline std::string ConvertCodeDotExch(std::string_view symbol) {
  if (symbol.size() != 9) throw MakeExcept<InvalidArgument>("Invalid symbol: {}", symbol);
  auto exch = std::string{symbol.substr(7, 2)};
  for (auto &c : exch) c = std::tolower(c);
  return exch + std::string(symbol.substr(0, 6));
}

}  // namespace yang
