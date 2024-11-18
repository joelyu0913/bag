#pragma once

#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include "yang/expr/mat.h"
#include "yang/sim/env.h"
#include "yang/util/unordered_map.h"
#include "yang/util/unordered_set.h"

namespace yang {

class ExprRunner {
 public:
  enum class Mode {
    INTRADAY = 0,
    EOD = 1,
    MIXED = 2,
  };

  using DataInfo = std::tuple<std::string, std::string, std::string>;

  static std::vector<std::string> default_base_data() {
    return {
        "open",     "close",   "high",      "low",      "ret",      "vol",
        "dvol",     "vwap",    "sharesout", "cap",      "adj_open", "adj_close",
        "adj_high", "adj_low", "adj_vol",   "adj_vwap", "cumadj",   "sharesfloat",
    };
  }

  static std::vector<std::string> default_groups() {
    return {"cty", "sector", "industry", "subindustry"};
  }

  ExprRunner(const Env *env);

  ExprRunner(const Env *env, const std::vector<std::string> &base_data,
             const std::vector<std::string> &groups, const std::vector<DataInfo> &extra_data);

  void AddGroups(const std::vector<std::string> &groups);

  void AddBaseData(const std::vector<std::string> &base_data);

  void AddExtraData(const std::vector<DataInfo> &extra_data);

  void Run(std::string_view expr_str, expr::OutFloatMat output, Mode mode = Mode::INTRADAY);

  void Run(std::string_view expr_str, std::string_view univ, expr::OutFloatMat output,
           Mode mode = Mode::INTRADAY);

  void Run(std::string_view expr_str, std::string_view univ, expr::OutFloatMat output, Mode mode,
           int dates_size, int start_di, int end_di);

  static Mode ParseMode(std::string_view str) {
    if (str == "intraday" || str == "INTRADAY") return Mode::INTRADAY;
    if (str == "eod" || str == "EOD") return Mode::EOD;
    if (str == "mixed" || str == "MIXED") return Mode::MIXED;
    LOG_FATAL("Invalid mode: {}", str);
  }

 protected:
  const Env *env_ = nullptr;
  unordered_set<std::string> groups_;
  unordered_map<std::string, std::pair<std::string, std::string>> data_;

  void RunSimple(std::string_view name, std::string_view univ, expr::OutFloatMat output, Mode mode,
                 int start_di, int end_di);
};

}  // namespace yang
