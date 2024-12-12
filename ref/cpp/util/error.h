#pragma once

#include <errno.h>

#include <string>

namespace yang {

// Thread-safe version of strerror
std::string GetErrorString(int errnum);

inline std::string GetErrorString() {
  return GetErrorString(errno);
}

}  // namespace yang
