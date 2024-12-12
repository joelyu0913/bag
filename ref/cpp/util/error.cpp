#include "yang/util/error.h"

#include <string.h>

namespace yang {

std::string GetErrorString(int errnum) {
  char buf[255];
  if (strerror_r(errnum, buf, sizeof(buf))) return "";
  return std::string(buf);
}

}  // namespace yang
