#pragma once
#include <string>
#include <vector>

#include "yang/sim/env.h"

namespace yang {

class CnEnv : public Env {
 public:
  std::vector<std::string> default_univ_indices() const final {
    // 000905.SH CSI 500 Index
    // 000300.SH CSI 300 Index
    // 000016.SH sh50,
    // 000842.SH 800 equal
    // 000852.SH CSI 1000 Index
    return {"000905.SH", "000300.SH", "000016.SH", "000842.SH", "000852.SH", "932000.SH"};
  }
};

}  // namespace yang
