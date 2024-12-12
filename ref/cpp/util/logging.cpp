#include "yang/util/logging.h"

#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <iostream>

#include "yang/util/backtrace.h"

namespace yang {
namespace detail {

void PrintBacktrace() {
  for (auto &s : Backtrace(64)) {
    std::cout << s << std::endl;
  }
}

}  // namespace detail

void ConfigureLogging(const std::string &pattern, const std::string &level, bool log_stderr,
                      const std::string &log_file) {
  std::vector<spdlog::sink_ptr> sinks;
  if (log_stderr) {
    sinks.push_back(std::make_shared<spdlog::sinks::stderr_color_sink_mt>());
  } else {
    sinks.push_back(std::make_shared<spdlog::sinks::stdout_color_sink_mt>());
  }
  if (log_file != "") {
    sinks.push_back(std::make_shared<spdlog::sinks::basic_file_sink_mt>(log_file, false));
  }
  auto logger = std::make_shared<spdlog::logger>("yang", sinks.begin(), sinks.end());
  spdlog::register_logger(logger);
  spdlog::set_default_logger(logger);

  spdlog::set_pattern(pattern);
  spdlog::default_logger()->flush_on(spdlog::level::err);
  if (level == "error") {
    spdlog::set_level(spdlog::level::err);
  } else if (level == "warn") {
    spdlog::set_level(spdlog::level::warn);
  } else if (level == "debug") {
    spdlog::set_level(spdlog::level::debug);
  } else if (level == "trace") {
    spdlog::set_level(spdlog::level::trace);
  } else {
    spdlog::set_level(spdlog::level::info);
  }
  spdlog::flush_every(std::chrono::seconds(2));
}

}  // namespace yang
