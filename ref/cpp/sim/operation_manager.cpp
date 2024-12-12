#include "yang/sim/operation_manager.h"

#include <cmath>

namespace yang {

OperationFactory *py_op_factory = nullptr;

void SetPyOperationFactory(OperationFactory *factory) {
  py_op_factory = factory;
}

void OperationManager::Apply(math::MatView<float> sig, int start_di, int end_di,
                             const std::vector<OperationDesc> &ops) {
  Apply(sig, sig, start_di, end_di, ops);
}

void OperationManager::Apply(math::MatView<const float> sig_in, math::MatView<float> sig_out,
                             int start_di, int end_di, const std::vector<OperationDesc> &ops) {
  ApplyOperations(sig_in, sig_out, start_di, end_di, ops);
}

OperationManager::OperationWrapper OperationManager::MakeOperation(const std::string &name) {
  OperationWrapper wrapper;
  wrapper.underlying = FactoryRegistry::TryMake<Operation>("operation", name);
  if (!wrapper.underlying && py_op_factory) {
    auto ret = py_op_factory->Make(name);
    wrapper.id = ret.first;
    wrapper.underlying = ret.second;
    wrapper.py = true;
  }
  ENSURE(wrapper.underlying != nullptr, "Unknown operation {}", name);
  return wrapper;
}

template <class Ops>
void OperationManager::ApplyOperations(math::MatView<const float> sig_in,
                                       math::MatView<float> sig_out, int start_di, int end_di,
                                       const Ops &ops) {
  bool lookback = false;
  std::vector<OperationWrapper> op_wrappers;
  for (auto &[op_name, args, kwargs] : ops) {
    op_wrappers.emplace_back(MakeOperation(op_name));
    if (op_wrappers.back()->lookback()) lookback = true;
  }
  if (lookback) start_di = 0;

  if (sig_in != sig_out) {
    for (int di = start_di; di < end_di; ++di) {
      for (int ii = 0; ii < sig_in.cols(); ++ii) {
        if (std::isfinite(sig_in(di, ii))) {
          sig_out(di, ii) = sig_in(di, ii);
        } else {
          sig_out(di, ii) = NAN;
        }
      }
    }
  }
  for (int i = 0; i < static_cast<int>(ops.size()); ++i) {
    auto &[op_name, args, kwargs] = ops[i];
    Operation *op = op_wrappers[i].underlying;
    try {
      op->Apply(sig_out, *env_, start_di, end_di, args, kwargs);
    } catch (const std::exception &e) {
      LOG_ERROR("Failed to apply operation {}({}): {}", op_name, args, e.what());
      throw;
    }
  }
}

OperationManager::OperationWrapper::~OperationWrapper() {
  if (underlying == nullptr) return;
  if (py) {
    if (underlying) py_op_factory->Free(id, underlying);
  } else {
    delete underlying;
  }
}

}  // namespace yang
