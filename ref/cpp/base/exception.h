#pragma once

#include <fmt/core.h>
#include <fmt/ostream.h>

#include <stdexcept>

namespace yang {

class Exception : public std::runtime_error {
 public:
  Exception(const std::string &what) : std::runtime_error(what) {}
};

struct FatalError : Exception {
  using Exception::Exception;
};

struct IoError : Exception {
  using Exception::Exception;
};

struct OutOfRange : Exception {
  using Exception::Exception;
};

struct InvalidArgument : Exception {
  using Exception::Exception;
};

template <class T, class... Args>
T MakeExcept(fmt::format_string<Args...> f, Args &&...args) {
  return T(fmt::format(f, std::forward<Args>(args)...));
}

}  // namespace yang
