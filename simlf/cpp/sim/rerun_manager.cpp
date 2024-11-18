#include "yang/sim/rerun_manager.h"

#include <fstream>

#include "yang/util/config.h"
#include "yang/util/datetime.h"
#include "yang/util/logging.h"

namespace yang {

void RerunManager::Initialize(std::string_view workdir) {
  workdir_ = workdir;
  if (!fs::exists(workdir_)) fs::create_directories(workdir_);
}

void RerunManager::SetDates(int64_t start_date, int64_t end_date) {
  start_date_ = start_date;
  end_date_ = end_date;
}

bool RerunManager::CanSkipRun(std::string_view mod, const std::vector<std::string> &deps) {
  auto meta = GetMeta(mod);
  // rerun if different start_date or extend
  if (meta.start_date != start_date_ || meta.end_date < end_date_) return false;

  for (auto &dep : deps) {
    // rerun if any dep changes
    auto dep_meta = GetMeta(dep);
    if (dep_meta.timestamp > meta.timestamp) return false;
  }
  return true;
}

void RerunManager::RecordBeforeRun(std::string_view mod) {
  auto meta_path = GetMetaPath(mod);
  if (fs::exists(meta_path)) {
    fs::remove(meta_path);
  }
}

void RerunManager::RecordRun(std::string_view mod) {
  SaveMeta(mod);
}

void RerunManager::SaveMeta(std::string_view mod) {
  ModMeta meta;
  meta.timestamp = GetTimestamp();
  meta.start_date = start_date_;
  meta.end_date = end_date_;

  Config yaml;
  yaml.Set("timestamp", meta.timestamp);
  yaml.Set("start_date", meta.start_date);
  yaml.Set("end_date", meta.end_date);
  std::ofstream ofs(GetMetaPath(mod));
  ofs << yaml.ToYamlString();
  ENSURE2(ofs.good());

  {
    std::lock_guard guard(mutex_);
    mods_[mod] = meta;
  }
}

RerunManager::ModMeta RerunManager::GetMeta(std::string_view mod) {
  {
    std::lock_guard guard(mutex_);
    auto it = mods_.find(mod);
    if (it != mods_.end()) return it->second;
  }

  ModMeta meta;
  auto meta_path = GetMetaPath(mod);
  if (fs::exists(meta_path)) {
    try {
      auto yaml = Config::LoadFile(meta_path);
      meta.timestamp = yaml.Get<int64_t>("timestamp");
      meta.start_date = yaml.Get<int64_t>("start_date");
      meta.end_date = yaml.Get<int64_t>("end_date");
    } catch (const std::exception &ex) {
      LOG_ERROR("Failed to load meta: {}", meta_path);
      meta = ModMeta();
    }
  }
  {
    std::lock_guard guard(mutex_);
    auto ret = mods_.emplace(mod, meta);
    return ret.first->second;
  }
}

}  // namespace yang
