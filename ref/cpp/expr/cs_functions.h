#pragma once

#include "yang/expr/functions.h"

namespace yang::expr {

class CsFunction : public UnaryFunction {
 public:
  void ApplyImpl(const FunctionArgs &args) const final;

  virtual void ApplyCs(const FunctionArgs &args) const = 0;
};

class CsZScore : public CsFunction {
  void ApplyCs(const FunctionArgs &args) const final;
};

class CsTruncate : public CsFunction {
 public:
  int num_scalar_args() const {
    return 1;
  }

 protected:
  void ApplyCs(const FunctionArgs &args) const final;
};

class CsTruncateUpper : public CsFunction {
 public:
  int num_scalar_args() const {
    return 1;
  }

 protected:
  void ApplyCs(const FunctionArgs &args) const final;
};

class CsSigwin : public CsFunction {
 public:
  int num_scalar_args() const {
    return 2;
  }

 protected:
  void ApplyCs(const FunctionArgs &args) const final;
};

class CsSigwinUpper : public CsFunction {
 public:
  int num_scalar_args() const {
    return 2;
  }

 protected:
  void ApplyCs(const FunctionArgs &args) const final;
};

class CsRank : public CsFunction {
  void ApplyCs(const FunctionArgs &args) const final;
};

class CsDemean : public CsFunction {
  void ApplyCs(const FunctionArgs &args) const final;
};

class CsScale : public CsFunction {
 public:
  int num_scalar_args() const {
    return 1;
  }

 protected:
  void ApplyCs(const FunctionArgs &args) const final;
};

}  // namespace yang::expr
