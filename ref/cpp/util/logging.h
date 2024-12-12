#pragma once

#include <spdlog/fmt/ostr.h>
#include <spdlog/spdlog.h>

#include <string>
#include <string_view>

#include "yang/base/exception.h"
#include "yang/base/likely.h"
#include "yang/util/error.h"
#include "yang/util/std_logging.h"

namespace yang {

namespace detail {
void PrintBacktrace();

inline void logger_call(spdlog::logger *logger, spdlog::source_loc source,
                        spdlog::level::level_enum lvl, std::string_view msg) {
  logger->log(source, lvl, msg);
}

template <typename... Args>
void logger_call(spdlog::logger *logger, spdlog::source_loc source, spdlog::level::level_enum lvl,
                 fmt::format_string<Args...> format, Args &&...args) {
  auto msg = fmt::vformat(format, fmt::make_format_args(std::forward<Args>(args)...));
  logger->log(source, lvl, msg);
}
}  // namespace detail

void ConfigureLogging(const std::string &pattern, const std::string &level = "info",
                      bool log_stderr = false, const std::string &log_file = "");

}  // namespace yang

#define LOG_TRACE(...) SPDLOG_TRACE(__VA_ARGS__)
#define LOG_DEBUG(...) SPDLOG_DEBUG(__VA_ARGS__)
#define LOG_INFO(...) SPDLOG_INFO(__VA_ARGS__)
#define LOG_WARN(...) SPDLOG_WARN(__VA_ARGS__)
#define LOG_ERROR(...) SPDLOG_ERROR(__VA_ARGS__)
#define LOG_FATAL(...)                                  \
  {                                                     \
    SPDLOG_CRITICAL(__VA_ARGS__);                       \
    ::yang::detail::PrintBacktrace();                   \
    throw ::yang::FatalError(fmt::format(__VA_ARGS__)); \
  }

#define ENSURE(cond, ...) \
  if UNLIKELY (!(cond)) LOG_FATAL(__VA_ARGS__);

#define ENSURE2(cond) \
  if UNLIKELY (!(cond)) LOG_FATAL("Condition check failed: " #cond);
