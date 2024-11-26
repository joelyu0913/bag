#include "yang/expr/dag.h"

#include "yang/expr/function_registry.h"
#include "yang/util/logging.h"
#include "yang/util/strings.h"
#include "yang/util/unordered_map.h"

namespace yang::expr {

static std::shared_ptr<ExprNode> ToDag(
    unordered_map<std::string_view, std::shared_ptr<ExprNode>> &nodes, const AstNode &ast) {
  auto node_it = nodes.find(ast.repr);
  if (node_it != nodes.end()) return node_it->second;

  auto expr = std::make_shared<ExprNode>();
  expr->value = ast.value;
  expr->repr = ast.repr;
  expr->depth = ast.depth;
  // expr: (f input1 input2 ... scalar1 scalar2 ... group)
  if (!ast.children.empty()) {
    expr->expr_type = ExprType::FUNCTION;
    auto func_name = ast.value;
    expr->func = FunctionRegistry::Get(func_name);
    ENSURE(expr->func != nullptr, "Unknown function: {}", func_name);

    int num_inputs;
    int num_scalar_args = expr->func->num_scalar_args();
    if (expr->func->variable_inputs()) {
      num_inputs = ast.children.size() - num_scalar_args - expr->func->use_group();
      ENSURE2(num_inputs >= 0);
    } else {
      num_inputs = expr->func->num_inputs();
      int num_args = num_inputs + num_scalar_args + expr->func->use_group();
      ENSURE(num_args == static_cast<int>(ast.children.size()),
             "Input arguments mismatched for function {}, expected {}, got {}", func_name, num_args,
             static_cast<int>(ast.children.size()));
    }

    int scalar_arg_offset = num_inputs;
    ENSURE2(num_scalar_args <= expr->scalar_args.capacity());
    for (int i = 0; i < num_scalar_args; ++i) {
      expr->scalar_args.push_back(CheckAtod(ast.children[i + scalar_arg_offset].value));
    }
    if (expr->func->use_group()) {
      expr->group = ast.children.back().value;
    }
    expr->ts_len = expr->func->ComputeTsLen(
        VecView<Float>(expr->scalar_args.data(), expr->scalar_args.size()));
    for (int i = 0; i < num_inputs; ++i) {
      auto input = ToDag(nodes, ast.children[i]);
      input->hist_len = std::max(expr->ts_len, input->hist_len);
      expr->inputs.emplace_back(std::move(input));
    }
  } else {
    double value;
    if (SafeAtod(expr->repr, value)) {
      expr->expr_type = ExprType::SCALAR;
      expr->scalar_value = value;
    } else {
      expr->expr_type = ExprType::DATA;
    }
  }
  nodes[ast.repr] = expr;
  return expr;
}

std::string NormalizeExpr(std::string_view expr) {
  std::vector<std::string_view> exprs = StrSplit(expr, absl::ByAnyChar(";\n"));
  std::vector<std::string> norm_exprs;
  for (auto e : exprs) {
    if (e.empty()) continue;
    norm_exprs.push_back(fmt::format("{}", ParseExpr(e)));
  }
  return fmt::format("{}", fmt::join(norm_exprs, ";"));
}

std::shared_ptr<ExprNode> ParseExprDag(std::string_view expr) {
  std::vector<std::string_view> exprs = StrSplit(expr, absl::ByAnyChar(";\n"));
  ENSURE(exprs.size() > 0, "Empty expr: {}", expr);
  std::vector<AstNode> ast_nodes;
  for (auto e : exprs) {
    if (e.empty()) continue;
    ast_nodes.emplace_back(ParseExpr(e));
  }
  ENSURE(!ast_nodes.back().is_let, "Last expr cannot be a let: {}", expr);
  unordered_map<std::string_view, std::shared_ptr<ExprNode>> nodes;
  for (int i = 0; i < static_cast<int>(ast_nodes.size()) - 1; ++i) {
    if (ast_nodes[i].is_let) {
      auto dag_node = ToDag(nodes, ast_nodes[i].children[0]);
      nodes[ast_nodes[i].value] = dag_node;
    }
  }
  return ToDag(nodes, ast_nodes.back());
}

}  // namespace yang::expr
