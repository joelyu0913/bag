#pragma once

#include "yang/expr/function.h"

namespace yang::expr {

class UnaryFunction : public Function {
 public:
  int num_inputs() const final {
    return 1;
  }
};

class Abs : public UnaryFunction {
  void ApplyImpl(const FunctionArgs &args) const final;
};

class Inverse : public UnaryFunction {
  void ApplyImpl(const FunctionArgs &args) const final;
};

class Negate : public UnaryFunction {
  void ApplyImpl(const FunctionArgs &args) const final;
};

class Log : public UnaryFunction {
  void ApplyImpl(const FunctionArgs &args) const final;
};

class Sigmoid : public UnaryFunction {
  void ApplyImpl(const FunctionArgs &args) const final;
};

class Sign : public UnaryFunction {
  void ApplyImpl(const FunctionArgs &args) const final;
};

class Sinh : public UnaryFunction {
  void ApplyImpl(const FunctionArgs &args) const final;
};

class Tanh : public UnaryFunction {
  void ApplyImpl(const FunctionArgs &args) const final;
};

class Sin : public UnaryFunction {
  void ApplyImpl(const FunctionArgs &args) const final;
};

class Cos : public UnaryFunction {
  void ApplyImpl(const FunctionArgs &args) const final;
};

class Tan : public UnaryFunction {
  void ApplyImpl(const FunctionArgs &args) const final;
};

class Cot : public UnaryFunction {
  void ApplyImpl(const FunctionArgs &args) const final;
};

class FillNA : public UnaryFunction {
  void ApplyImpl(const FunctionArgs &args) const final;
};

class NotNA : public UnaryFunction {
  void ApplyImpl(const FunctionArgs &args) const final;
};

class Pow : public UnaryFunction {
  int num_scalar_args() const override {
    return 1;
  }

  void ApplyImpl(const FunctionArgs &args) const final;
};

class SPow : public UnaryFunction {
  int num_scalar_args() const override {
    return 1;
  }

  void ApplyImpl(const FunctionArgs &args) const final;
};

class PowInt : public UnaryFunction {
  int num_scalar_args() const override {
    return 1;
  }

  void ApplyImpl(const FunctionArgs &args) const final;
};

class SPowInt : public UnaryFunction {
  int num_scalar_args() const override {
    return 1;
  }

  void ApplyImpl(const FunctionArgs &args) const final;
};

class BinaryFunction : public Function {
 public:
  int num_inputs() const final {
    return 2;
  }
};

class VarArgsFunction : public Function {
 public:
  int num_inputs() const final {
    return 1;
  }

  bool variable_inputs() const final {
    return true;
  }
};

class Add : public VarArgsFunction {
  void ApplyImpl(const FunctionArgs &args) const final;
};

class Subtract : public VarArgsFunction {
  void ApplyImpl(const FunctionArgs &args) const final;
};

class Multiply : public VarArgsFunction {
  void ApplyImpl(const FunctionArgs &args) const final;
};

class Divide : public VarArgsFunction {
  void ApplyImpl(const FunctionArgs &args) const final;
};

class FilterInvalid : public BinaryFunction {
  void ApplyImpl(const FunctionArgs &args) const final;
};

class Filter : public BinaryFunction {
  void ApplyImpl(const FunctionArgs &args) const final;
};

class Greater : public BinaryFunction {
  void ApplyImpl(const FunctionArgs &args) const final;
};

class GreaterEqual : public BinaryFunction {
  void ApplyImpl(const FunctionArgs &args) const final;
};

class Less : public BinaryFunction {
  void ApplyImpl(const FunctionArgs &args) const final;
};

class LessEqual : public BinaryFunction {
  void ApplyImpl(const FunctionArgs &args) const final;
};

class Equal : public BinaryFunction {
  void ApplyImpl(const FunctionArgs &args) const final;
};

class Unequal : public BinaryFunction {
  void ApplyImpl(const FunctionArgs &args) const final;
};

class And : public VarArgsFunction {
  void ApplyImpl(const FunctionArgs &args) const final;
};

class Or : public VarArgsFunction {
  void ApplyImpl(const FunctionArgs &args) const final;
};

class Max : public VarArgsFunction {
  void ApplyImpl(const FunctionArgs &args) const final;
};

class Min : public VarArgsFunction {
  void ApplyImpl(const FunctionArgs &args) const final;
};

class DiffDate : public BinaryFunction {
  void ApplyImpl(const FunctionArgs &args) const final;
};

class Bound : public BinaryFunction {
  int num_scalar_args() const override {
    return 1;
  }

  void ApplyImpl(const FunctionArgs &args) const final;
};

class If : public Function {
 public:
  int num_inputs() const final {
    return 3;
  }

 protected:
  void ApplyImpl(const FunctionArgs &args) const final;
};

class Const : public Function {
 public:
  int num_inputs() const final {
    return 0;
  }

  int num_scalar_args() const override {
    return 1;
  }

 protected:
  void ApplyImpl(const FunctionArgs &args) const final;
};

}  // namespace yang::expr
