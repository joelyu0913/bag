#pragma once

#include "yang/expr/function.h"

namespace yang::expr {

class GroupFunction : public Function {
 public:
  int num_inputs() const final {
    return 1;
  }

  bool use_group() const final {
    return true;
  }

 protected:
  void ApplyImpl(const FunctionArgs &args) const final;

  virtual void ApplyGroup(const FunctionArgs &args) const = 0;
};

class GroupDemean : public GroupFunction {
  void ApplyGroup(const FunctionArgs &args) const final;
};

class GroupRank : public GroupFunction {
  void ApplyGroup(const FunctionArgs &args) const final;
};

class GroupRankPow : public GroupFunction {
 public:
  int num_scalar_args() const final {
    return 1;
  }

 protected:
  void ApplyGroup(const FunctionArgs &args) const final;
};

class GroupZScore : public GroupFunction {
  void ApplyGroup(const FunctionArgs &args) const final;
};

class GroupTruncate : public GroupFunction {
 public:
  int num_scalar_args() const final {
    return 1;
  }

 protected:
  void ApplyGroup(const FunctionArgs &args) const final;
};

class GroupTruncateUpper : public GroupFunction {
 public:
  int num_scalar_args() const final {
    return 1;
  }

 protected:
  void ApplyGroup(const FunctionArgs &args) const final;
};

class GroupSigwin : public GroupFunction {
 public:
  int num_scalar_args() const final {
    return 2;
  }

 protected:
  void ApplyGroup(const FunctionArgs &args) const final;
};

class GroupSigwinUpper : public GroupFunction {
 public:
  int num_scalar_args() const final {
    return 2;
  }

 protected:
  void ApplyGroup(const FunctionArgs &args) const final;
};

}  // namespace yang::expr
