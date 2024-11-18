#pragma once

#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include "yang/util/fs.h"
#include "yang/util/unordered_map.h"

namespace yang {

// TODO: the rules are very simple now and do not work in all cases, e.g. live
class RerunManager {
 public:
  void Initialize(std::string_view workdir);

  void SetDates(int64_t start_date, int64_t end_date);

  bool CanSkipRun(std::string_view mod, const std::vector<std::string> &deps);

  void RecordBeforeRun(std::string_view mod);

  void RecordRun(std::string_view mod);

 private:
  struct ModMeta {
    int64_t timestamp = 0;
    int64_t start_date = 0;
    int64_t end_date = 0;
  };

  fs::path workdir_;
  unordered_map<std::string, ModMeta> mods_;
  std::mutex mutex_;
  int64_t start_date_ = 0;
  int64_t end_date_ = 0;

  void SaveMeta(std::string_view mod);

  ModMeta GetMeta(std::string_view mod);

  std::string GetMetaPath(std::string_view mod) const {
    return (workdir_ / mod).native() + ".yml";
  }
};

}  // namespace yang
