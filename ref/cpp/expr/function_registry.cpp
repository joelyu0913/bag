#include "yang/expr/function_registry.h"

#include "yang/expr/cs_functions.h"
#include "yang/expr/functions.h"
#include "yang/expr/group_functions.h"
#include "yang/expr/ts_functions.h"

namespace yang::expr {

FunctionRegistry::FunctionRegistry() {
  functions_["abs"] = std::make_unique<Abs>();
  functions_["flip"] = std::make_unique<Negate>();
  functions_["inverse"] = std::make_unique<Inverse>();
  functions_["log"] = std::make_unique<Log>();
  functions_["neg"] = std::make_unique<Negate>();
  functions_["sigmoid"] = std::make_unique<Sigmoid>();
  functions_["sign"] = std::make_unique<Sign>();
  functions_["sinh"] = std::make_unique<Sinh>();
  functions_["tanh"] = std::make_unique<Tanh>();
  functions_["sin"] = std::make_unique<Sin>();
  functions_["cos"] = std::make_unique<Cos>();
  functions_["tan"] = std::make_unique<Tan>();
  functions_["cot"] = std::make_unique<Cot>();
  functions_["fill_na"] = std::make_unique<FillNA>();
  functions_["is_valid"] = std::make_unique<NotNA>();
  functions_["const"] = std::make_unique<Const>();
  functions_["bound"] = std::make_unique<Bound>();
  functions_["pow"] = std::make_unique<Pow>();
  functions_["spow"] = std::make_unique<SPow>();
  functions_["pow_int"] = std::make_unique<PowInt>();
  functions_["spow_int"] = std::make_unique<SPowInt>();

  functions_["zscore"] = std::make_unique<CsZScore>();
  functions_["c_zscore"] = std::make_unique<CsZScore>();
  functions_["truncate"] = std::make_unique<CsTruncate>();
  functions_["c_truncate"] = std::make_unique<CsTruncate>();
  functions_["truncate_u"] = std::make_unique<CsTruncateUpper>();
  functions_["c_truncate_u"] = std::make_unique<CsTruncateUpper>();
  functions_["sigwin"] = std::make_unique<CsSigwin>();
  functions_["c_sigwin"] = std::make_unique<CsSigwin>();
  functions_["sigwin_u"] = std::make_unique<CsSigwinUpper>();
  functions_["c_sigwin_u"] = std::make_unique<CsSigwinUpper>();
  functions_["rank"] = std::make_unique<CsRank>();
  functions_["c_rank"] = std::make_unique<CsRank>();
  functions_["demean"] = std::make_unique<CsDemean>();
  functions_["c_demean"] = std::make_unique<CsDemean>();
  functions_["scale"] = std::make_unique<CsScale>();
  functions_["c_scale"] = std::make_unique<CsScale>();
  functions_["c_reg"] = std::make_unique<CsReg>();

  functions_["add"] = std::make_unique<Add>();
  functions_["sub"] = std::make_unique<Subtract>();
  functions_["mul"] = std::make_unique<Multiply>();
  functions_["div"] = std::make_unique<Divide>();
  functions_["+"] = std::make_unique<Add>();
  functions_["-"] = std::make_unique<Subtract>();
  functions_["*"] = std::make_unique<Multiply>();
  functions_["/"] = std::make_unique<Divide>();

  functions_[">"] = std::make_unique<Greater>();
  functions_[">="] = std::make_unique<GreaterEqual>();
  functions_["<"] = std::make_unique<Less>();
  functions_["<="] = std::make_unique<LessEqual>();
  functions_["=="] = std::make_unique<Equal>();
  functions_["!="] = std::make_unique<Unequal>();
  functions_["&"] = std::make_unique<And>();
  functions_["|"] = std::make_unique<Or>();

  functions_["filter_v"] = std::make_unique<FilterInvalid>();
  functions_["filter"] = std::make_unique<Filter>();
  functions_["max"] = std::make_unique<Max>();
  functions_["min"] = std::make_unique<Min>();
  functions_["diff_date"] = std::make_unique<DiffDate>();
  functions_["if"] = std::make_unique<If>();

  functions_["ts_min"] = std::make_unique<TsMin>();
  functions_["ts_min_raw"] = std::make_unique<TsMinRaw>();
  functions_["ts_max"] = std::make_unique<TsMax>();
  functions_["ts_max_raw"] = std::make_unique<TsMaxRaw>();
  functions_["ts_mean"] = std::make_unique<TsMean>();
  functions_["ts_mean_raw"] = std::make_unique<TsMeanRaw>();
  functions_["ts_rmean"] = std::make_unique<TsRMean>();
  functions_["ts_rmean_raw"] = std::make_unique<TsRMeanRaw>();
  functions_["ts_std"] = std::make_unique<TsStd>();
  functions_["ts_std_raw"] = std::make_unique<TsStdRaw>();
  functions_["ts_mean_dev"] = std::make_unique<TsMeanDev>();
  functions_["ts_mean_dev_raw"] = std::make_unique<TsMeanDevRaw>();
  functions_["ts_zscore"] = std::make_unique<TsZScore>();
  functions_["ts_zscore_raw"] = std::make_unique<TsZScoreRaw>();
  functions_["ts_demean"] = std::make_unique<TsDemean>();
  functions_["ts_demean_raw"] = std::make_unique<TsDemeanRaw>();
  functions_["ts_central_moment"] = std::make_unique<TsCenterExp>();
  functions_["ts_central_moment_raw"] = std::make_unique<TsCenterExpRaw>();
  // functions_["ts_delay2"] = std::make_unique<TsDelay>();
  // functions_["ts_delay2_raw"] = std::make_unique<TsDelayRaw>();
  functions_["ts_delay"] = std::make_unique<TsDelay>();
  functions_["ts_delay_raw"] = std::make_unique<TsDelayRaw>();
  functions_["ts_sum"] = std::make_unique<TsSum>();
  functions_["ts_sum_raw"] = std::make_unique<TsSumRaw>();
  functions_["ts_count_v"] = std::make_unique<TsCountValid>();
  functions_["ts_count"] = std::make_unique<TsCountBool>();
  functions_["ts_prod"] = std::make_unique<TsProd>();
  functions_["ts_prod_raw"] = std::make_unique<TsProdRaw>();
  functions_["ts_delta2"] = std::make_unique<TsDelta>();
  functions_["ts_delta2_raw"] = std::make_unique<TsDeltaRaw>();
  functions_["ts_delta"] = std::make_unique<TsDeltaRange>();
  functions_["ts_delta_raw"] = std::make_unique<TsDeltaRangeRaw>();
  functions_["ts_corr_step"] = std::make_unique<TsCorrStep>();
  functions_["ts_corr_step_raw"] = std::make_unique<TsCorrStepRaw>();
  functions_["ts_corr"] = std::make_unique<TsCorr>();
  functions_["ts_corr_raw"] = std::make_unique<TsCorrRaw>();
  functions_["ts_auto_corr"] = std::make_unique<TsAutoCorr>();
  functions_["ts_auto_corr_raw"] = std::make_unique<TsAutoCorrRaw>();
  functions_["ts_ema"] = std::make_unique<TsEma>();
  functions_["ts_ema_raw"] = std::make_unique<TsEmaRaw>();
  functions_["ts_sma"] = std::make_unique<TsSma>();
  functions_["ts_sma_raw"] = std::make_unique<TsSmaRaw>();
  functions_["ts_wma"] = std::make_unique<TsWma>();
  functions_["ts_wma_raw"] = std::make_unique<TsWmaRaw>();
  functions_["ts_decay_linear"] = std::make_unique<TsDecayLinear>();
  functions_["ts_decay_linear_raw"] = std::make_unique<TsDecayLinearRaw>();
  functions_["ts_cov"] = std::make_unique<TsCov>();
  functions_["ts_cov_raw"] = std::make_unique<TsCovRaw>();
  functions_["ts_sum_if"] = std::make_unique<TsSumIf>();
  functions_["ts_sum_if_raw"] = std::make_unique<TsSumIfRaw>();
  functions_["ts_argmax"] = std::make_unique<TsArgmax>();
  functions_["ts_argmax_raw"] = std::make_unique<TsArgmaxRaw>();
  functions_["ts_argmin"] = std::make_unique<TsArgmin>();
  functions_["ts_argmin_raw"] = std::make_unique<TsArgminRaw>();
  functions_["ts_high_day"] = std::make_unique<TsArgmax>();
  functions_["ts_high_day_raw"] = std::make_unique<TsArgmaxRaw>();
  functions_["ts_low_day"] = std::make_unique<TsArgmin>();
  functions_["ts_low_day_raw"] = std::make_unique<TsArgminRaw>();
  functions_["ts_rank"] = std::make_unique<TsRank>();
  functions_["ts_rank_raw"] = std::make_unique<TsRankRaw>();
  functions_["ts_sum_max_pct"] = std::make_unique<TsSumMaxPct>();
  functions_["ts_sum_max_pct_raw"] = std::make_unique<TsSumMaxPctRaw>();
  functions_["ts_reg"] = std::make_unique<TsReg>();
  functions_["ts_idx"] = std::make_unique<TsIndex>();

  functions_["g_demean"] = std::make_unique<GroupDemean>();
  functions_["g_rank"] = std::make_unique<GroupRank>();
  functions_["g_rankpow"] = std::make_unique<GroupRankPow>();
  functions_["g_zscore"] = std::make_unique<GroupZScore>();
  functions_["g_truncate"] = std::make_unique<GroupTruncate>();
  functions_["g_truncate_u"] = std::make_unique<GroupTruncateUpper>();
  functions_["g_sigwin"] = std::make_unique<GroupSigwin>();
  functions_["g_sigwin_u"] = std::make_unique<GroupSigwinUpper>();
}

}  // namespace yang::expr
