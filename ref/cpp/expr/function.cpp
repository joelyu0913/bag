#include "yang/expr/function.h"

#include "yang/expr/ops.h"
#include "yang/util/logging.h"

namespace yang::expr {

void Function::Apply(const FunctionArgs &args) const {
  ENSURE2(static_cast<int>(args.scalar_args.size()) == num_scalar_args());

  ENSURE(variable_inputs() || static_cast<int>(args.inputs.size()) == num_inputs(),
         "Input size mismatch, required {}, got {}", num_inputs(), args.inputs.size());

  for (int i = 0; i < static_cast<int>(args.inputs.size()); ++i) {
    ENSURE(args.output.size() == args.inputs[i].back().size(),
           "Input {} ({}) and output ({}) has different size", i, args.inputs[i].back().size(),
           args.output.size());
  }
  ApplyImpl(args);
}

int Function::ComputeTsLen(VecView<Float> scalar_args) const {
  return 1;
}

void Function::ApplyMask(VecView<Float> input, VecView<bool> mask, OutVecView<Float> output) {
  if (!mask.empty()) {
    ops::filter(input.begin(), input.end(), mask.begin(), output.begin());
  } else {
    output.copy_from(input);
  }
}

}  // namespace yang::expr
