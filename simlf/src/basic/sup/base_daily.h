#pragma once
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "yang/data/graph.h"
#include "yang/sim/module.h"
#include "yang/util/unordered_map.h"
#include "yang/util/unordered_set.h"
#include "yao/include/func_utils.h"

#define FOR(i, a, b) for (int i = (int)(a); i < (int)(b); ++i)
#define REP(i, a) FOR(i, 0, a)

namespace yao {

class BaseDaily : public yang::Module {
 public:
  int max_date_sz;
  int max_univ_sz;

  int date_sz;  // copy
  int univ_sz;  // copy
  int start_di;
  int end_di;

  yang::Array<bool> b_univ;
  yang::Array<float> b_sig;

  yang::unordered_map<std::string, const yang::Array<int> *> b_i = {};
  yang::unordered_map<std::string, const yang::Array<float> *> b_f = {};
  yang::unordered_map<std::string, const yang::Array<bool> *> b_b = {};

  yang::unordered_set<std::string> set_loaded = {};

  void RunImpl() override;

  virtual void user_proc() {}

  void base_load(const std::vector<std::string> &vv);

  void ApplyOps();

  void LoadBaseData();
  void LoadIBaseData();
};
}  // namespace yao
