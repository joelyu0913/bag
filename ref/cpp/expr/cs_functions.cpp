#include "yang/expr/cs_functions.h"

#include "yang/expr/ops.h"

namespace yang::expr {

void CsFunction::ApplyImpl(const FunctionArgs &args) const {
  ApplyMask(args.input(0), args.mask, args.output);
  ApplyCs(args);
}

void CsZScore::ApplyCs(const FunctionArgs &args) const {
  ops::zscore(args.output.begin(), args.output.end());
}

void CsTruncate::ApplyCs(const FunctionArgs &args) const {
  auto cap = args.scalar_args[0];
  ops::truncate(args.output.begin(), args.output.end(), cap);
}

void CsTruncateUpper::ApplyCs(const FunctionArgs &args) const {
  auto cap = args.scalar_args[0];
  ops::truncate_upper(args.output.begin(), args.output.end(), cap);
}

void CsSigwin::ApplyCs(const FunctionArgs &args) const {
  auto cap1 = args.scalar_args[0];
  auto cap2 = args.scalar_args[1];
  ops::sigwin(args.output.begin(), args.output.end(), cap1, cap2);
}

void CsSigwinUpper::ApplyCs(const FunctionArgs &args) const {
  auto cap1 = args.scalar_args[0];
  auto cap2 = args.scalar_args[1];
  ops::sigwin_upper(args.output.begin(), args.output.end(), cap1, cap2);
}

void CsRank::ApplyCs(const FunctionArgs &args) const {
  ops::rank(args.output.begin(), args.output.end());
}

void CsDemean::ApplyCs(const FunctionArgs &args) const {
  ops::demean(args.output.begin(), args.output.end());
}

void CsScale::ApplyCs(const FunctionArgs &args) const {
  auto size = args.scalar_args[0];
  ops::scale(args.output.begin(), args.output.end(), size, 1e-10);
}

}  // namespace yang::expr
