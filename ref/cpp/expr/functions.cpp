#include "yang/expr/functions.h"

#include <cmath>

#include "yang/expr/ops.h"
#include "yang/util/dates.h"

namespace yang::expr {

template <class Func>
static void ApplyBinary(const FunctionArgs &args, Func &&func) {
  for (int i = 0; i < args.output.size(); ++i) {
    auto x = args.input(0)[i];
    auto y = args.input(1)[i];
    args.output[i] = func(x, y);
  }
}

void Abs::ApplyImpl(const FunctionArgs &args) const {
  ops::abs(args.input(0).begin(), args.input(0).end(), args.output.begin());
}

void Inverse::ApplyImpl(const FunctionArgs &args) const {
  ops::inverse(args.input(0).begin(), args.input(0).end(), args.output.begin());
}

void Negate::ApplyImpl(const FunctionArgs &args) const {
  ops::negate(args.input(0).begin(), args.input(0).end(), args.output.begin());
}

void Log::ApplyImpl(const FunctionArgs &args) const {
  ops::log1(args.input(0).begin(), args.input(0).end(), args.output.begin());
}

void Sigmoid::ApplyImpl(const FunctionArgs &args) const {
  ops::sigmoid(args.input(0).begin(), args.input(0).end(), args.output.begin());
}

void Sign::ApplyImpl(const FunctionArgs &args) const {
  ops::sign(args.input(0).begin(), args.input(0).end(), args.output.begin());
}

void Sinh::ApplyImpl(const FunctionArgs &args) const {
  ops::sinh(args.input(0).begin(), args.input(0).end(), args.output.begin());
}

void Tanh::ApplyImpl(const FunctionArgs &args) const {
  ops::tanh(args.input(0).begin(), args.input(0).end(), args.output.begin());
}

void Sin::ApplyImpl(const FunctionArgs &args) const {
  for (int i = 0; i < args.output.size(); ++i) {
    args.output[i] = std::sin(args.input(0)[i]);
  }
}

void Cos::ApplyImpl(const FunctionArgs &args) const {
  for (int i = 0; i < args.output.size(); ++i) {
    args.output[i] = std::cos(args.input(0)[i]);
  }
}

void Tan::ApplyImpl(const FunctionArgs &args) const {
  for (int i = 0; i < args.output.size(); ++i) {
    args.output[i] = std::tan(args.input(0)[i]);
  }
}

void Cot::ApplyImpl(const FunctionArgs &args) const {
  for (int i = 0; i < args.output.size(); ++i) {
    args.output[i] = 1.0 / std::tan(args.input(0)[i]);
  }
}

void FillNA::ApplyImpl(const FunctionArgs &args) const {
  for (int i = 0; i < args.output.size(); ++i) {
    auto x = args.input(0)[i];
    args.output[i] = IsValid(x) ? x : 0;
  }
}

void NotNA::ApplyImpl(const FunctionArgs &args) const {
  for (int i = 0; i < args.output.size(); ++i) {
    args.output[i] = IsValid(args.input(0)[i]);
  }
}

void Pow::ApplyImpl(const FunctionArgs &args) const {
  Float e = args.scalar_args[0];
  ops::pow(args.input(0).begin(), args.input(0).end(), args.output.begin(), e);
}

void SPow::ApplyImpl(const FunctionArgs &args) const {
  Float e = args.scalar_args[0];
  ops::spow(args.input(0).begin(), args.input(0).end(), args.output.begin(), e);
}

void PowInt::ApplyImpl(const FunctionArgs &args) const {
  int e = args.scalar_args[0];
  ops::pow(args.input(0).begin(), args.input(0).end(), args.output.begin(), e);
}

void SPowInt::ApplyImpl(const FunctionArgs &args) const {
  int e = args.scalar_args[0];
  ops::spow(args.input(0).begin(), args.input(0).end(), args.output.begin(), e);
}

void Add::ApplyImpl(const FunctionArgs &args) const {
  for (int i = 0; i < args.output.size(); ++i) {
    Float sum = 0;
    for (auto &input : args.inputs) {
      sum += input.back()[i];
    }
    args.output[i] = sum;
  }
}

void Subtract::ApplyImpl(const FunctionArgs &args) const {
  for (int i = 0; i < args.output.size(); ++i) {
    Float result = args.inputs[0].back()[i];
    for (int k = 1; k < static_cast<int>(args.inputs.size()); ++k) {
      result -= args.inputs[k].back()[i];
    }
    args.output[i] = result;
  }
}

void Multiply::ApplyImpl(const FunctionArgs &args) const {
  ENSURE2(!args.inputs.empty());
  for (int i = 0; i < args.output.size(); ++i) {
    Float prod = 1;
    for (auto &input : args.inputs) {
      prod *= input.back()[i];
    }
    args.output[i] = prod;
  }
}

void Divide::ApplyImpl(const FunctionArgs &args) const {
  ENSURE2(!args.inputs.empty());
  for (int i = 0; i < args.output.size(); ++i) {
    Float result = args.inputs[0].back()[i];
    for (int k = 1; k < static_cast<int>(args.inputs.size()); ++k) {
      result /= args.inputs[k].back()[i];
    }
    args.output[i] = result;
  }
}

void FilterInvalid::ApplyImpl(const FunctionArgs &args) const {
  ops::filter(args.input(0).begin(), args.input(0).end(), args.input(1).begin(),
              args.output.begin());
}

void Filter::ApplyImpl(const FunctionArgs &args) const {
  ApplyBinary(args, [](auto x, auto y) { return y != 0 ? x : NAN; });
}

void Greater::ApplyImpl(const FunctionArgs &args) const {
  ApplyBinary(args, [](auto x, auto y) { return x > y; });
}

void GreaterEqual::ApplyImpl(const FunctionArgs &args) const {
  ApplyBinary(args, [](auto x, auto y) { return x >= y; });
}

void Less::ApplyImpl(const FunctionArgs &args) const {
  ApplyBinary(args, [](auto x, auto y) { return x < y; });
}

void LessEqual::ApplyImpl(const FunctionArgs &args) const {
  ApplyBinary(args, [](auto x, auto y) { return x <= y; });
}

void Equal::ApplyImpl(const FunctionArgs &args) const {
  ApplyBinary(args, [](auto x, auto y) { return x == y; });
}

void Unequal::ApplyImpl(const FunctionArgs &args) const {
  ApplyBinary(args, [](auto x, auto y) { return x != y; });
}

void And::ApplyImpl(const FunctionArgs &args) const {
  for (int i = 0; i < args.output.size(); ++i) {
    if (args.inputs.size() == 0) {
      args.output[i] = false;
      return;
    }
    bool ret = true;
    for (auto &input : args.inputs) {
      if (!static_cast<bool>(input.back()[i])) {
        ret = false;
        break;
      }
    }
    args.output[i] = ret;
  }
}

void Or::ApplyImpl(const FunctionArgs &args) const {
  for (int i = 0; i < args.output.size(); ++i) {
    bool ret = false;
    for (auto &input : args.inputs) {
      if (static_cast<bool>(input.back()[i])) {
        ret = true;
        break;
      }
    }
    args.output[i] = ret;
  }
}

void Max::ApplyImpl(const FunctionArgs &args) const {
  for (int i = 0; i < args.output.size(); ++i) {
    Float ret = NAN;
    for (auto &input : args.inputs) {
      if (std::isnan(ret)) {
        ret = input.back()[i];
      } else {
        ret = std::max(ret, input.back()[i]);
      }
    }
    args.output[i] = ret;
  }
}

void Min::ApplyImpl(const FunctionArgs &args) const {
  for (int i = 0; i < args.output.size(); ++i) {
    Float ret = NAN;
    for (auto &input : args.inputs) {
      if (std::isnan(ret)) {
        ret = input.back()[i];
      } else {
        ret = std::min(ret, input.back()[i]);
      }
    }
    args.output[i] = ret;
  }
}

void DiffDate::ApplyImpl(const FunctionArgs &args) const {
  ApplyBinary(args, [](auto x, auto y) -> double {
    if (!IsValid(y) || !IsValid(x)) return NAN;
    return yang::SubDate(x, y);
  });
}

void Bound::ApplyImpl(const FunctionArgs &args) const {
  auto margin = args.scalar_args[0];
  ApplyBinary(args, [margin](auto x, auto y) -> double {
    if (!IsValid(y) || !IsValid(x)) return 0;
    return std::abs(x - y) <= margin;
  });
}

void If::ApplyImpl(const FunctionArgs &args) const {
  for (int i = 0; i < args.output.size(); ++i) {
    auto cond = args.input(0)[i];
    auto x = args.input(1)[i];
    auto y = args.input(2)[i];
    args.output[i] = cond != 0 ? x : y;
  }
}

void Const::ApplyImpl(const FunctionArgs &args) const {
  auto value = args.scalar_args[0];
  for (auto &v : args.output) v = value;
}

}  // namespace yang::expr
