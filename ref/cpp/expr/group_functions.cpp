#include "yang/expr/group_functions.h"

#include "yang/expr/ops.h"

namespace yang::expr {

void GroupFunction::ApplyImpl(const FunctionArgs &args) const {
  ApplyMask(args.input(0), args.mask, args.output);
  ApplyGroup(args);
}

void GroupDemean::ApplyGroup(const FunctionArgs &args) const {
  ops::group_demean<ops::DefaultCheck>(args.output.begin(), args.output.end(), args.group.begin());
}

void GroupRank::ApplyGroup(const FunctionArgs &args) const {
  ops::group_rank(args.output.begin(), args.output.end(), args.group.begin());
}

void GroupRankPow::ApplyGroup(const FunctionArgs &args) const {
  auto exp = args.scalar_args[0];
  ops::group_rank_pow(args.output.begin(), args.output.end(), args.group.begin(), exp);
}

void GroupZScore::ApplyGroup(const FunctionArgs &args) const {
  ops::group_zscore(args.output.begin(), args.output.end(), args.group.begin(),
                    args.output.begin());
}

void GroupTruncate::ApplyGroup(const FunctionArgs &args) const {
  auto cap = args.scalar_args[0];
  ops::group_truncate(args.output.begin(), args.output.end(), args.group.begin(), cap);
}

void GroupTruncateUpper::ApplyGroup(const FunctionArgs &args) const {
  auto cap = args.scalar_args[0];
  ops::group_truncate_upper(args.output.begin(), args.output.end(), args.group.begin(), cap);
}

void GroupSigwin::ApplyGroup(const FunctionArgs &args) const {
  auto cap1 = args.scalar_args[0];
  auto cap2 = args.scalar_args[1];
  ops::group_sigwin(args.output.begin(), args.output.end(), args.group.begin(), cap1, cap2);
}

void GroupSigwinUpper::ApplyGroup(const FunctionArgs &args) const {
  auto cap1 = args.scalar_args[0];
  auto cap2 = args.scalar_args[1];
  ops::group_sigwin_upper(args.output.begin(), args.output.end(), args.group.begin(), cap1, cap2);
}

}  // namespace yang::expr
