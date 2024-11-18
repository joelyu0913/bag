#pragma once

#include <memory>
#include <string>

#include "yang/data/array.h"
#include "yang/data/data_cache.h"
#include "yang/data/index.h"
#include "yang/data/univ_index.h"
#include "yang/sim/data_directory.h"
#include "yang/sim/rerun_manager.h"
#include "yang/util/config.h"
#include "yang/util/factory_registry.h"

namespace yang {

enum class RunStage {
  ERROR = 0,
  PREPARE,
  OPEN,
  INTRADAY,
  EOD,
};

constexpr RunStage ParseRunStage(std::string_view s) {
  if (s == "prepare" || s == "PREPARE") return RunStage::PREPARE;
  if (s == "open" || s == "OPEN") return RunStage::OPEN;
  if (s == "intraday" || s == "INTRADAY") return RunStage::INTRADAY;
  if (s == "eod" || s == "EOD") return RunStage::EOD;
  return RunStage::ERROR;
}

constexpr std::string_view ToString(RunStage stage) {
  switch (stage) {
    case RunStage::PREPARE:
      return "prepare";
    case RunStage::OPEN:
      return "open";
    case RunStage::INTRADAY:
      return "intraday";
    case RunStage::EOD:
      return "eod";
    default:
      return "error";
  }
}

using DateTimeIndex = OrderedIndex<int64_t>;
using TimeIndex = OrderedIndex<int>;

constexpr float DISPLAY_BOOK_SIZE = 100000;

class Env {
 public:
  virtual ~Env() {}

  const std::string name() const {
    return "env";
  }

  const DataDirectory &cache_dir() const {
    return cache_dir_;
  }

  DataCache &data_cache() const {
    return data_cache_;
  }

  RerunManager *rerun_manager() const {
    return rerun_manager_.get();
  }

  const Config &config() const {
    return config_;
  }

  template <class T>
  T config(const std::string &key) const {
    return config_.Get<T>(key);
  }

  template <class T>
  T config(const std::string &key, const T &default_value) const {
    return config_.Get<T>(key, default_value);
  }

  bool user_mode() const {
    return user_mode_;
  }

  int start_di() const {
    return start_dti_;
  }

  int end_di() const {
    return end_dti_;
  }

  int dates_size() const {
    return datetimes_.size();
  }

  int64_t univ_start_datetime() const {
    return univ_start_datetime_;
  }

  int64_t univ_end_datetime() const {
    return univ_end_datetime_;
  }

  int64_t sim_start_datetime() const {
    return sim_start_datetime_;
  }

  int64_t sim_end_datetime() const {
    return sim_end_datetime_;
  }

  int start_dti() const {
    return start_dti_;
  }

  int end_dti() const {
    return end_dti_;
  }

  int datetimes_size() const {
    return datetimes_.size();
  }

  int univ_size() const {
    return univ_.size();
  }

  int max_univ_size() const {
    return max_univ_size_;
  }

  // datetimes size round to the next multiple of ROUND
  int max_datetimes_size() const {
    constexpr uint32_t ROUND = 64;
    constexpr uint32_t ROUND_MASK = ~(ROUND - 1);
    return (datetimes_.size() + ROUND - 1) & ROUND_MASK;
  }

  const DateTimeIndex &datetimes() const {
    return datetimes_;
  }

  int max_dates_size() const {
    return max_datetimes_size();
  }

  const DateTimeIndex &dates() const {
    return datetimes_;
  }

  const UnivIndex &univ() const {
    return univ_;
  }

  const std::vector<std::string> &univ_indices() const {
    return univ_.indices();
  }

  const std::string &benchmark_index() const {
    return benchmark_index_;
  }

  // Find ii for univ_indices_[index]
  int FindIndexId(int index) const {
    return univ_.FindIndexId(index);
  }

  bool IsStock(int ii) const {
    return ii < univ_.index_id_start();
  }

  bool live() const {
    return live_;
  }

  void set_live(bool v) {
    live_ = v;
  }

  bool hist() const {
    return !live();
  }

  bool prod() const {
    return prod_;
  }

  void set_prod(bool v) {
    prod_ = v;
  }

  bool daily() const {
    return daily_;
  }

  // intraday time points in HHMM format
  const TimeIndex &intraday_times() const {
    return intraday_times_;
  }

  const std::string &taq() const {
    return taq_;
  }

  // taq time points in HHMM format
  const TimeIndex &taq_times() const {
    return taq_times_;
  }

  float trade_book_size() const {
    return config<float>("trade_book_size", DISPLAY_BOOK_SIZE);
  }

  template <class T>
  const T *ReadData(std::string_view mod, std::string_view data) const {
    auto key = fmt::format("{}.{}", mod, data);
    return data_cache().GetOrLoad<T>(key, cache_dir().GetReadPath(mod, data));
  }

  template <class T>
  const T *ReadData(std::string_view data) const {
    auto pos = data.find('/');
    ENSURE(pos != std::string_view::npos, "Missing module name in data spec: {}", data);
    return ReadData<T>(data.substr(0, pos), data.substr(pos + 1));
  }

  template <class T, class... Args>
  const auto *ReadArray(Args &&...args) {
    return ReadData<yang::Array<T>>(std::forward<Args>(args)...);
  }

  void Initialize(const Config &config_arg);

  void Build();

  void Load();

  virtual int default_max_univ_size() const {
    return 6000;
  }

  virtual int default_univ_indices_id_start() const {
    return 5800;
  }

  virtual std::vector<std::string> default_univ_indices() const {
    return {};
  }

  int days_per_year() const {
    return days_per_year_;
  }

  int intervals_per_day() const {
    return daily_ ? 1 : intraday_times_.size();
  }

 protected:
  DataDirectory cache_dir_;
  mutable DataCache data_cache_;
  std::unique_ptr<RerunManager> rerun_manager_;
  Config config_;

  bool live_ = false;
  bool prod_ = false;
  bool daily_ = true;
  bool user_mode_ = false;

  int64_t univ_start_datetime_ = 0;
  int64_t univ_end_datetime_ = 0;
  int64_t sim_start_datetime_ = 0;
  int64_t sim_end_datetime_ = 0;
  int start_dti_ = -1;
  int end_dti_ = -1;

  int max_univ_size_ = 0;

  DateTimeIndex datetimes_;
  TimeIndex intraday_times_;
  UnivIndex univ_;

  TimeIndex taq_times_;
  std::string taq_;

  int univ_indices_id_start_;
  std::vector<std::string> univ_indices_;

  int days_per_year_ = 0;
  bool short_book_size_ = false;
  std::string benchmark_index_;

  std::string GetMetaPath() const {
    return cache_dir_.GetPath(name(), "meta.yml");
  }

  void FillMeta(Config &meta);

  void CheckMetaChanges(const Config &meta) const;

  void SaveMeta();

  virtual void BuildImpl(int new_start_dti);

  void PostLoad();
};

#define REGISTER_ENV(cls) REGISTER_FACTORY("env", cls, cls, #cls)

}  // namespace yang
