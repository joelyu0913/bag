#pragma once

#include <memory>
#include <string_view>
#include <vector>

#include "yang/util/fmt.h"

namespace yang::expr {

struct AstNode {
  std::string_view value;
  std::vector<AstNode> children;
  std::string_view repr;
  int depth = 0;
  bool is_let = false;
};

AstNode ParseExpr(std::string_view expr);

std::ostream &operator<<(std::ostream &os, const AstNode &node);

}  // namespace yang::expr

#ifdef FMT_OSTREAM_FORMATTER
template <>
struct fmt::formatter<yang::expr::AstNode> : ostream_formatter {};
#endif
