#pragma once

#include <memory>
#include <vector>

#include "yang/expr/ast.h"
#include "yang/expr/function.h"
#include "yang/expr/vec.h"
#include "yang/util/small_vector.h"

namespace yang::expr {

enum class ExprType {
  INVALID = -1,
  FUNCTION = 0,
  DATA = 1,
  SCALAR = 2,
};

struct ExprNode {
  std::string_view value;
  std::string_view repr;
  int depth = 0;

  ExprType expr_type = ExprType::INVALID;
  union {
    Function *func;
    Float scalar_value;
  };
  std::vector<std::shared_ptr<ExprNode>> inputs;
  small_vector<Float, 16> scalar_args;
  std::string_view group;
  int ts_len = 0;

  int hist_len = 1;
  VecBuffer<Float> results;
  bool preload_hist = false;
};

std::string NormalizeExpr(std::string_view expr);

std::shared_ptr<ExprNode> ParseExprDag(std::string_view expr);

}  // namespace yang::expr
