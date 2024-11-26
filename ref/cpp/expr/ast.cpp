#include "yang/expr/ast.h"

#include "yang/util/logging.h"

namespace yang::expr {

std::pair<AstNode, int> ParseExprImpl(std::string_view expr) {
  AstNode node;
  auto start = expr.find_first_not_of(' ');
  ENSURE(start != std::string_view::npos, "Empty expr: {}", expr);

  expr = expr.substr(start);
  int len = 0;
  if (expr[0] == '(') {
    auto pos = expr.find_first_not_of(' ', 1);
    ENSURE(pos != std::string_view::npos, "Incomplete expr: {}", expr);
    auto func_end = expr.find_first_of(" )", pos);
    ENSURE(func_end != std::string_view::npos, "Incomplete expr: {}", expr);
    node.value = expr.substr(pos, func_end - pos);

    pos = func_end;
    while (true) {
      pos = expr.find_first_not_of(' ', pos);
      ENSURE(pos != std::string_view::npos, "Incomplete expr: {}", expr);
      if (expr[pos] == ')') {
        ++pos;
        break;
      }

      auto [child, len] = ParseExprImpl(expr.substr(pos));
      node.depth = std::max(node.depth, child.depth + 1);
      node.children.emplace_back(std::move(child));
      pos += len;
    }
    len = pos;

    if (node.value == "let" || node.value == "LET") {
      node.is_let = true;
      ENSURE(node.children.size() == 2, "Invalid let expr: {}", expr);
      ENSURE(node.children[0].children.empty(), "Invalid let expr: {}", expr);
      node.value = node.children[0].value;
      node.children.erase(node.children.begin());
    }
  } else {
    auto data_end = expr.find_first_of(" )");
    if (data_end == std::string_view::npos) {
      len = expr.size();
    } else {
      len = data_end;
    }
    node.value = expr.substr(0, len);
  }
  ENSURE(len > 0, "Invalid expr: {}", expr);
  node.repr = expr.substr(0, len);
  return {node, len + start};
}

AstNode ParseExpr(const std::string_view expr) {
  auto [node, len] = ParseExprImpl(expr);
  ENSURE(expr.find_first_not_of(' ', len) == std::string_view::npos, "Invalid expr: {}", expr);
  return node;
}

std::ostream &operator<<(std::ostream &os, const AstNode &node) {
  if (node.children.empty()) {
    os << node.value;
  } else {
    os << '(';
    if (node.is_let) os << "let ";
    os << node.value << ' ';
    for (int i = 0; i < static_cast<int>(node.children.size()); ++i) {
      if (i > 0) os << ' ';
      os << node.children[i];
    }
    os << ')';
  }
  return os;
}

}  // namespace yang::expr
