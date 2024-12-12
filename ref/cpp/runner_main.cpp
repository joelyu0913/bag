#include <memory>
#include <string>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "yang/sim/runner.h"
#include "yang/util/logging.h"
#include "yang/util/term_handler.h"

ABSL_FLAG(std::string, config, "", "Config file path");
ABSL_FLAG(std::string, log_level, "info", "Logging level: trace,debug,info,warn,error");
ABSL_FLAG(std::string, log_file, "", "Output log file");
ABSL_FLAG(std::string, stage, "all", "Stage to run: all,prepare,intraday");
ABSL_FLAG(int, num_threads, 1, "Number of worker threads");
ABSL_FLAG(bool, live, false, "Enable live mode");
ABSL_FLAG(bool, prod, false, "Enable production mode");

int main(int argc, char **argv) {
  using namespace yang;

  SetTerminateHandler();

  absl::ParseCommandLine(argc, argv);

  ConfigureLogging("[%C-%m-%d %T.%e] [%t] [%s:%#] %^[%l]%$ %v", absl::GetFlag(FLAGS_log_level),
                   false, absl::GetFlag(FLAGS_log_file));

  auto config_file = absl::GetFlag(FLAGS_config);
  ENSURE(!config_file.empty(), "Missing config file");
  LOG_INFO("Using config file {}", config_file);
  auto config = Config::LoadFile(config_file);
  LOG_INFO("Loaded config");

  auto stage = absl::GetFlag(FLAGS_stage);
  RunnerOptions options;
  if (stage == "all") {
    options.stages = {RunStage::PREPARE, RunStage::OPEN, RunStage::INTRADAY, RunStage::EOD};
  } else {
    auto stage_val = ParseRunStage(stage);
    if (stage_val == RunStage::ERROR) {
      LOG_FATAL("Invalid stage {}", stage);
    }
    options.stages.insert(stage_val);
  }
  options.live = absl::GetFlag(FLAGS_live);
  options.prod = absl::GetFlag(FLAGS_prod);
  Runner runner;
  runner.Run(options, config, absl::GetFlag(FLAGS_num_threads));
  return 0;
}
