#include "yang/util/factory_registry.h"

namespace yang {
namespace detail {

FactoryStorage *GetFactoryStorage() {
  static FactoryStorage *instance = new FactoryStorage;
  return instance;
}

}  // namespace detail

}  // namespace yang
