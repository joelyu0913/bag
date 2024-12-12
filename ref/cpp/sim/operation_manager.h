#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

#include "yang/math/mat_view.h"
#include "yang/sim/operation.h"
#include "yang/util/config.h"
#include "yang/util/unordered_map.h"

namespace yang {

struct OperationFactory {
  virtual std::pair<int, Operation *> Make(std::string_view name) = 0;
  virtual void Free(int id, Operation *op) = 0;
};

void SetPyOperationFactory(OperationFactory *factory);

using OperationDesc =
    std::tuple<std::string, std::vector<std::string>, std::unordered_map<std::string, std::string>>;

class OperationManager {
 public:
  void Initialize(const Env *env) {
    env_ = env;
  }

  void Apply(math::MatView<float> sig, int start_di, int end_di,
             const std::vector<OperationDesc> &ops);

  void Apply(math::MatView<const float> sig_in, math::MatView<float> sig_out, int start_di,
             int end_di, const std::vector<OperationDesc> &ops);

 private:
  struct OperationWrapper {
    Operation *underlying = nullptr;
    int id = 0;
    bool py = false;

    OperationWrapper() {}
    ~OperationWrapper();

    OperationWrapper(OperationWrapper &&other) {
      this->operator=(std::move(other));
    }

    OperationWrapper &operator=(OperationWrapper &&other) {
      underlying = other.underlying;
      id = other.id;
      py = other.py;
      other.underlying = nullptr;
      return *this;
    }

    Operation &operator*() const {
      return *underlying;
    }

    Operation *operator->() const noexcept {
      return underlying;
    }
  };

  const Env *env_ = nullptr;

  template <class Ops>
  void ApplyOperations(math::MatView<const float> sig_in, math::MatView<float> sig_out,
                       int start_di, int end_di, const Ops &ops);

  static OperationWrapper MakeOperation(const std::string &name);
};

}  // namespace yang
