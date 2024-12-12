#pragma once

#include <memory>
#include <string_view>
#include <vector>

#include "yang/expr/ast.h"
#include "yang/expr/dag.h"
#include "yang/expr/data_source.h"
#include "yang/expr/function.h"
#include "yang/expr/vec.h"
#include "yang/util/unordered_set.h"

namespace yang::expr {

class Expr {
 public:
  Expr() {}

  Expr(std::string_view expr, DataSource *data_src) {
    Initialize(expr, data_src);
  }

  Expr(const Expr &other) = delete;
  Expr &operator=(const Expr &other) = delete;

  void Initialize(std::string_view repr, DataSource *data_src);

  VecView<Float> Eval(bool rerun = false);

  void EvalPreload() {
    EvalNodes(false, true);
  }

  const VecBuffer<Float> &results() const {
    return dag_->results;
  }

  int full_hist_len() const {
    return full_hist_len_;
  }

  const ExprNode &dag() const {
    return *dag_;
  }

  unordered_set<std::string> CollectGroups() const {
    unordered_set<std::string> ret;
    CollectGroups(ret);
    return ret;
  }

  unordered_set<std::string> CollectRawData() const {
    unordered_set<std::string> ret;
    CollectRawData(ret);
    return ret;
  }

 private:
  DataSource *data_src_ = nullptr;
  std::shared_ptr<ExprNode> dag_;
  std::vector<ExprNode *> dfs_order_;

  int full_hist_len_ = 0;  // the length of raw data history we use

  // store the expr string in the root Expr object
  std::string root_expr_;

  void ComputeFullHistLen();

  void CollectGroups(unordered_set<std::string> &ret) const;

  void CollectRawData(unordered_set<std::string> &ret) const;

  void EvalNodes(bool rerun, bool preload_hist);
};

}  // namespace yang::expr
