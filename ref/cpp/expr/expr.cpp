#include "yang/expr/expr.h"

#include "yang/expr/ops.h"
#include "yang/util/logging.h"
#include "yang/util/unordered_map.h"

namespace yang::expr {

static void DoDFS(ExprNode *node, unordered_set<ExprNode *> &visited,
                  std::vector<ExprNode *> &order) {
  if (!node->inputs.empty()) {
    std::vector<ExprNode *> inputs;
    inputs.reserve(node->inputs.size());
    for (auto &input : node->inputs) {
      inputs.push_back(input.get());
    }
    std::sort(inputs.begin(), inputs.end(), [](auto *x, auto *y) { return x->depth > y->depth; });
    for (auto input : inputs) {
      if (visited.count(input)) continue;
      DoDFS(input, visited, order);
    }
  }

  ENSURE2(visited.insert(node).second);
  order.push_back(node);
}

void Expr::Initialize(std::string_view repr, DataSource *data_src) {
  data_src_ = data_src;
  root_expr_ = repr;
  dag_ = ParseExprDag(root_expr_);

  unordered_set<ExprNode *> visited;
  DoDFS(dag_.get(), visited, dfs_order_);

  ComputeFullHistLen();

  for (auto it = dfs_order_.rbegin(); it != dfs_order_.rend(); ++it) {
    auto &node = *it;
    if (node->preload_hist || node->hist_len > 1) {
      node->preload_hist = true;
      for (auto &input : node->inputs) input->preload_hist = true;
    }
  }
}

void Expr::ComputeFullHistLen() {
  unordered_map<ExprNode *, int> lens;
  for (auto node : dfs_order_) {
    int max_child_len = 1;
    for (auto input : node->inputs) {
      max_child_len = std::max(max_child_len, lens[input.get()]);
    }

    // if this needs N-day history and children need M-day history,
    // the total required history is N + M - 1
    lens[node] = node->ts_len + max_child_len - 1;
  }
  full_hist_len_ = lens[dag_.get()];
}

void Expr::EvalNodes(bool rerun, bool preload_hist) {
  auto mask = data_src_->GetMask();
  for (auto node : dfs_order_) {
    if (preload_hist && !node->preload_hist) continue;
    if (node->results.capacity() == 0) {
      node->results = VecBuffer<Float>(node->hist_len, data_src_->vec_size());
    }
    if (rerun) node->results.PopBack();
    if (node->expr_type == ExprType::DATA) {
      node->results.PushBack(data_src_->GetData(node->repr));
    } else if (node->expr_type == ExprType::SCALAR) {
      node->results.PushBack(node->scalar_value);
    } else {
      node->results.PushBack(NAN);
      FunctionArgs args;
      args.output = node->results.writable_back();
      ENSURE2(static_cast<int>(node->inputs.size()) <= args.inputs.capacity());
      for (auto &input : node->inputs) {
        auto &results = input->results;
        int input_start = std::max(0, results.size() - node->ts_len);
        int input_size = results.size() - input_start;
        args.inputs.push_back(VecBufferSpan(results, input_start, input_size));
      }
      args.scalar_args = VecView<Float>(node->scalar_args.data(), node->scalar_args.size());
      args.mask = mask;
      if (!node->group.empty()) args.group = data_src_->GetGroup(node->group);
      node->func->Apply(args);
    }
  }
}

VecView<Float> Expr::Eval(bool rerun) {
  EvalNodes(rerun, false);
  auto mask = data_src_->GetMask();
  // apply mask on final output
  if (!mask.empty()) {
    auto result = dag_->results.writable_back();
    ops::filter(result.begin(), result.end(), mask.begin());
  }
  return dag_->results.back();
}

void Expr::CollectGroups(unordered_set<std::string> &ret) const {
  for (auto node : dfs_order_) {
    if (!node->group.empty()) ret.emplace(node->group);
  }
}

void Expr::CollectRawData(unordered_set<std::string> &ret) const {
  for (auto node : dfs_order_) {
    if (node->expr_type == ExprType::DATA) ret.emplace(node->repr);
  }
}

}  // namespace yang::expr
