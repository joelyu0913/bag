#include "yang/sim/runner.h"

#include <string>

#include "yang/sim/module.h"
#include "yang/sim/scheduler.h"
#include "yang/util/factory_registry.h"
#include "yang/util/logging.h"
#include "yang/util/module_loader.h"
#include "yang/util/unordered_map.h"
#include "yang/util/unordered_set.h"

namespace yang {

void Runner::Run(RunnerOptions options, const Config &config, int num_threads) {
  std::vector<void *> libs;
  for (auto &lib : config.Get<std::vector<std::string>>("libs", {})) {
    auto handle = ModuleLoader::Load(lib);
    ENSURE2(handle != nullptr);
    libs.push_back(handle);
  }

  LOG_INFO("Running ({} threads)", num_threads);
  std::unique_ptr<Env> env(FactoryRegistry::Make<Env>("env", config.Get<std::string>("env")));
  LOG_INFO("daily: {}", config.Get<bool>("daily", true));
  LOG_INFO("live: {}", options.live);
  env->set_live(options.live);
  LOG_INFO("prod: {}", options.prod);
  env->set_prod(options.prod);

  env->Initialize(config);
  if (!env->user_mode() && options.stages.count(RunStage::PREPARE)) {
    env->Build();
  } else {
    env->Load();
  }

  LOG_INFO("univ_start_datetime: {}", env->univ_start_datetime());
  LOG_INFO("univ_end_datetime: {}", env->univ_end_datetime());
  LOG_INFO("sim_start_datetime: {}", env->sim_start_datetime());
  LOG_INFO("sim_end_datetime: {}", env->sim_end_datetime());

  auto load_mod = [&](const auto &name, const Config &config) {
    std::unique_ptr<Module> mod(
        FactoryRegistry::Make<Module>("module", config.Get<std::string>("class")));
    mod->Initialize(name, config, env.get());
    return mod;
  };

  std::unique_ptr<Module> base_data;
  std::vector<std::unique_ptr<Module>> mods;
  unordered_map<std::string, std::vector<std::string>> mod_deps;
  unordered_set<std::string> run_modules(config.Get<std::vector<std::string>>("run_modules", {}));
  unordered_set<std::string> always_run_mods(
      config.Get<std::vector<std::string>>("always_run_modules", {}));

  auto run_mod = [&](auto &mod) {
    // check stage
    auto stages = mod->config("stages", std::vector<std::string>{"intraday"});
    std::set<RunStage> stages_to_run;
    for (auto &s : stages) {
      auto stage_val = ParseRunStage(s);
      if (options.stages.count(stage_val)) stages_to_run.insert(stage_val);
    }
    if (stages_to_run.empty()) return;

    if (env->rerun_manager()) {
      if (!always_run_mods.count(mod->name()) &&
          env->rerun_manager()->CanSkipRun(mod->name(), mod_deps[mod->name()])) {
        LOG_DEBUG("Skip module {}: already built", mod->name());
        return;
      }
      env->rerun_manager()->RecordBeforeRun(mod->name());
    }

    for (auto stage : stages_to_run) {
      LOG_INFO("Running module {} - {}", mod->name(), ToString(stage));
      mod->set_stage(stage);
      mod->Run();
    }
    LOG_INFO("Finished module {}", mod->name());

    if (env->rerun_manager()) {
      env->rerun_manager()->RecordRun(mod->name());
    }
  };

  for (auto mod_config : config["modules"]) {
    // check lang
    if (mod_config.Get<std::string>("lang") != "cpp") continue;

    auto name = mod_config.Get<std::string>("name");
    if (!run_modules.empty() && !run_modules.count(name)) continue;
    auto mod = load_mod(name, mod_config);
    if (name == "base") {
      base_data = std::move(mod);
    } else {
      mods.emplace_back(std::move(mod));
      mod_deps[name] = mod_config.Get<std::vector<std::string>>("deps", {});
    }
  }

  if (base_data) {
    for (auto &[_, deps] : mod_deps) {
      deps.push_back("base");
    }
    mods.emplace_back(std::move(base_data));
  }

  Scheduler scheduler;
  std::vector<std::pair<std::string, Scheduler::Func>> tasks;
  tasks.reserve(mods.size());
  for (auto &mod : mods) {
    tasks.emplace_back(mod->name(), [&mod, &run_mod]() { run_mod(mod); });
  }
  scheduler.Run(num_threads, mod_deps, tasks);

  for (auto lib : libs) {
    ModuleLoader::Unload(lib);
  }
}

}  // namespace yang
