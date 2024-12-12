#pragma once

#include "yang/sim/operation.h"

namespace yao {

struct OpLimit : yang::Operation {
  bool lookback() const final {
    return true;
  }

  void Apply(MatView<float> sig, const yang::Env &env, int start_di, int end_di,
             const std::vector<std::string> &args) override;
};

}  // namespace yao
