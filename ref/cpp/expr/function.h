#pragma once

#include <vector>

#include "yang/expr/base.h"
#include "yang/expr/vec.h"
#include "yang/util/small_vector.h"

namespace yang::expr {

struct FunctionArgs {
  OutVecView<Float> output;
  small_vector<VecBufferSpan<Float>, 16> inputs;

  VecView<bool> mask;
  VecView<int> group;

  VecView<Float> scalar_args;  // scalar args

  VecView<Float> input(int i) const {
    return inputs[i].back();
  }
};

class Function {
 public:
  virtual ~Function() {}

  void Apply(const FunctionArgs &args) const;

  virtual int num_inputs() const = 0;

  virtual int num_scalar_args() const {
    return 0;
  }

  virtual bool use_group() const {
    return false;
  }

  virtual bool variable_inputs() const {
    return false;
  }

  virtual int ComputeTsLen(VecView<Float> scalar_args) const;

 protected:
  virtual void ApplyImpl(const FunctionArgs &args) const = 0;

  static void ApplyMask(VecView<Float> input, VecView<bool> mask, OutVecView<Float> output);
};

}  // namespace yang::expr
