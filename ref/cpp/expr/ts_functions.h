#pragma once

#include <vector>

#include "yang/base/ranges.h"
#include "yang/expr/function.h"

namespace yang::expr {

constexpr int TS_FILL_LEN = 60;

class TsFunction : public Function {
 public:
  virtual int num_ts_args() const {
    return 0;
  }

  virtual bool fill_invalid() const {
    return true;
  }

  int ComputeTsLen(VecView<Float> scalar_args) const override {
    auto ts_len = ComputeTsLenInternal(scalar_args);
    if (fill_invalid()) ts_len += TS_FILL_LEN;
    return ts_len;
  }

 protected:
  virtual int ComputeTsLenInternal(VecView<Float> scalar_args) const;
};

class UnaryTsFunction : public TsFunction {
 public:
  int num_ts_args() const final {
    return 1;
  }

  int num_scalar_args() const override {
    return 1;
  }

  int num_inputs() const final {
    return 1;
  }
};

class BinaryTsFunction : public TsFunction {
 public:
  int num_ts_args() const final {
    return 1;
  }

  int num_scalar_args() const override {
    return 1;
  }

  int num_inputs() const final {
    return 2;
  }
};

class TsMin : public UnaryTsFunction {
  void ApplyImpl(const FunctionArgs &args) const final;
};

class TsMinRaw : public TsMin {
  bool fill_invalid() const final {
    return false;
  }
};

class TsMax : public UnaryTsFunction {
  void ApplyImpl(const FunctionArgs &args) const final;
};

class TsMaxRaw : public TsMax {
  bool fill_invalid() const final {
    return false;
  }
};

class TsSum : public UnaryTsFunction {
  void ApplyImpl(const FunctionArgs &args) const final;
};

class TsSumRaw : public TsSum {
  bool fill_invalid() const final {
    return false;
  }
};

class TsCountValid : public UnaryTsFunction {
  bool fill_invalid() const final {
    return false;
  }

  void ApplyImpl(const FunctionArgs &args) const final;
};

class TsCountBool : public UnaryTsFunction {
  bool fill_invalid() const final {
    return false;
  }

  void ApplyImpl(const FunctionArgs &args) const final;
};

class TsProd : public UnaryTsFunction {
  void ApplyImpl(const FunctionArgs &args) const final;
};

class TsProdRaw : public TsProd {
  bool fill_invalid() const final {
    return false;
  }
};

class TsMean : public UnaryTsFunction {
  void ApplyImpl(const FunctionArgs &args) const final;
};

class TsMeanRaw : public TsMean {
  bool fill_invalid() const final {
    return false;
  }
};

class TsRMean : public UnaryTsFunction {
  void ApplyImpl(const FunctionArgs &args) const final;
};

class TsRMeanRaw : public TsRMean {
  bool fill_invalid() const final {
    return false;
  }
};

class TsStd : public UnaryTsFunction {
  void ApplyImpl(const FunctionArgs &args) const final;
};

class TsStdRaw : public TsStd {
  bool fill_invalid() const final {
    return false;
  }
};

class TsMeanDev : public UnaryTsFunction {
  void ApplyImpl(const FunctionArgs &args) const final;
};

class TsMeanDevRaw : public TsMeanDev {
  bool fill_invalid() const final {
    return false;
  }
};

class TsZScore : public UnaryTsFunction {
  void ApplyImpl(const FunctionArgs &args) const final;
};

class TsZScoreRaw : public TsStd {
  bool fill_invalid() const final {
    return false;
  }
};

class TsDemean : public UnaryTsFunction {
  void ApplyImpl(const FunctionArgs &args) const final;
};

class TsDemeanRaw : public TsStd {
  bool fill_invalid() const final {
    return false;
  }
};

class TsCenterExp : public UnaryTsFunction {
  int num_scalar_args() const final {
    return 2;
  }

  void ApplyImpl(const FunctionArgs &args) const final;
};

class TsCenterExpRaw : public TsCenterExp {
  bool fill_invalid() const final {
    return false;
  }
};

class TsCorrStep : public UnaryTsFunction {
  void ApplyImpl(const FunctionArgs &args) const final;
};

class TsCorrStepRaw : public TsCorrStep {
  bool fill_invalid() const final {
    return false;
  }
};

class TsCorr : public BinaryTsFunction {
  void ApplyImpl(const FunctionArgs &args) const final;
};

class TsCorrRaw : public TsCorr {
  bool fill_invalid() const final {
    return false;
  }
};

class TsAutoCorr : public UnaryTsFunction {
 public:
  int num_scalar_args() const final {
    return 2;
  }

  int ComputeTsLenInternal(VecView<Float> scalar_args) const final {
    return scalar_args[0] + scalar_args[1];
  }

 protected:
  void ApplyImpl(const FunctionArgs &args) const final;
};

class TsAutoCorrRaw : public TsAutoCorr {
  bool fill_invalid() const final {
    return false;
  }
};

class TsDecayLinear : public UnaryTsFunction {
  void ApplyImpl(const FunctionArgs &args) const final;
};

class TsDecayLinearRaw : public TsDecayLinear {
  bool fill_invalid() const final {
    return false;
  }
};

class TsCov : public BinaryTsFunction {
  void ApplyImpl(const FunctionArgs &args) const final;
};

class TsCovRaw : public TsCov {
  bool fill_invalid() const final {
    return false;
  }
};

class TsSumIf : public BinaryTsFunction {
  void ApplyImpl(const FunctionArgs &args) const final;
};

class TsSumIfRaw : public TsSumIf {
  bool fill_invalid() const final {
    return false;
  }
};

class TsArgmax : public UnaryTsFunction {
  void ApplyImpl(const FunctionArgs &args) const final;
};

class TsArgmaxRaw : public TsArgmax {
  bool fill_invalid() const final {
    return false;
  }
};

class TsArgmin : public UnaryTsFunction {
  void ApplyImpl(const FunctionArgs &args) const final;
};

class TsArgminRaw : public TsArgmin {
  bool fill_invalid() const final {
    return false;
  }
};

class TsRank : public UnaryTsFunction {
  void ApplyImpl(const FunctionArgs &args) const final;
};

class TsRankRaw : public TsRank {
  bool fill_invalid() const final {
    return false;
  }
};

class TsEma : public UnaryTsFunction {
 public:
  int num_scalar_args() const override {
    return 2;
  }

 protected:
  void ApplyImpl(const FunctionArgs &args) const override;

  void ApplyRatio(const FunctionArgs &args, Float ratio) const;
};

class TsEmaRaw : public TsEma {
  bool fill_invalid() const final {
    return false;
  }
};

class TsSma : public TsEma {
 public:
  int num_scalar_args() const final {
    return 3;
  }

 private:
  void ApplyImpl(const FunctionArgs &args) const final;
};

class TsSmaRaw : public TsSma {
  bool fill_invalid() const final {
    return false;
  }
};

class TsWma : public UnaryTsFunction {
 public:
  int num_scalar_args() const final {
    return 1;
  }

 private:
  void ApplyImpl(const FunctionArgs &args) const final;
};

class TsWmaRaw : public TsWma {
  bool fill_invalid() const final {
    return false;
  }
};

class TsDelay : public UnaryTsFunction {
 public:
  int ComputeTsLenInternal(VecView<Float> scalar_args) const final {
    return scalar_args[0] + 1;
  }

 private:
  void ApplyImpl(const FunctionArgs &args) const final;
};

class TsDelayRaw : public TsDelay {
  bool fill_invalid() const final {
    return false;
  }
};

class TsDelta : public UnaryTsFunction {
 public:
  int ComputeTsLenInternal(VecView<Float> scalar_args) const final {
    return scalar_args[0] + 1;
  }

 private:
  void ApplyImpl(const FunctionArgs &args) const final;
};

class TsDeltaRaw : public TsDelta {
  bool fill_invalid() const final {
    return false;
  }
};

class TsDelayRange : public UnaryTsFunction {
  void ApplyImpl(const FunctionArgs &args) const final;
};

class TsDelayRangeRaw : public TsDelayRange {
  bool fill_invalid() const final {
    return false;
  }
};

class TsDeltaRange : public UnaryTsFunction {
  void ApplyImpl(const FunctionArgs &args) const final;
};

class TsDeltaRangeRaw : public TsDeltaRange {
  bool fill_invalid() const final {
    return false;
  }
};

class TsSumMaxPct : public BinaryTsFunction {
  int num_scalar_args() const final {
    return 2;
  }

  void ApplyImpl(const FunctionArgs &args) const final;
};

class TsSumMaxPctRaw : public TsSumMaxPct {
  bool fill_invalid() const final {
    return false;
  }
};

class TsReg : public TsFunction {
 public:
  bool fill_invalid() const final {
    return false;
  }

  int num_ts_args() const final {
    return 1;
  }

  int num_scalar_args() const override {
    return 2;
  }

  int num_inputs() const final {
    return 1;
  }

  bool variable_inputs() const final {
    return true;
  }

  void ApplyImpl(const FunctionArgs &args) const final;
};

class CsReg : public TsFunction {
 public:
  bool fill_invalid() const final {
    return false;
  }

  int num_ts_args() const final {
    return 1;
  }

  int num_scalar_args() const override {
    return 2;
  }

  int num_inputs() const final {
    return 1;
  }

  bool variable_inputs() const final {
    return true;
  }

  void ApplyImpl(const FunctionArgs &args) const final;
};

class TsIndex : public BinaryTsFunction {
 public:
  bool fill_invalid() const final {
    return false;
  }

  void ApplyImpl(const FunctionArgs &args) const final;
};

}  // namespace yang::expr
