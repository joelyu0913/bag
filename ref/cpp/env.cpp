#include "yang/sim/env.h"

#include <fstream>
#include <string_view>
#include <vector>

#include "yang/data/array.h"
#include "yang/io/open.h"
#include "yang/util/datetime.h"
#include "yang/util/fs.h"
#include "yang/util/logging.h"
#include "yang/util/simple_csv.h"
#include "yang/util/strings.h"

namespace yang {

void Env::Initialize(const Config &config_arg) {
  config_ = config_arg;
  if (config_["user_cache"]) {
    ENSURE(config_["sys_cache"], "sys_cache missing");
    cache_dir_.Initialize(config<std::string>("user_cache"), config<std::string>("sys_cache"));
    user_mode_ = true;
  } else if (config_["cache"]) {
    cache_dir_.Initialize(config<std::string>("cache"));
    user_mode_ = false;
  } else if (config_["sys_cache"]) {
    cache_dir_.Initialize(config<std::string>("sys_cache"));
    user_mode_ = false;
  } else {
    LOG_FATAL("cache config missing");
  }

  if (config_.Get("use_rerun_manager", false)) {
    ENSURE(!live(), "use_rerun_manager is not supported in live mode");
    rerun_manager_ = std::make_unique<RerunManager>();
    rerun_manager_->Initialize(cache_dir_.GetWritePath("_rerun"));
    LOG_DEBUG("Enabled RerunManager");
  }
}

void Env::Build() {
  ENSURE2(config_);
  ENSURE2(!user_mode());
  LOG_INFO("user_mode: {}", user_mode());
  LOG_INFO("user_cache: {}", cache_dir_.user_dir());
  LOG_INFO("sys_cache: {}", cache_dir_.sys_dir());

  daily_ = config<bool>("daily", true);
  max_univ_size_ = config<int>("max_univ_size", default_max_univ_size());
  if (daily_) {
    univ_start_datetime_ = config<int>("univ_start_date");
    univ_end_datetime_ = config<int>("univ_end_date");
    sim_start_datetime_ = config<int>("sim_start_date", univ_start_datetime_);
    sim_end_datetime_ = config<int>("sim_end_date", univ_end_datetime_);
  } else {
    univ_start_datetime_ = config<int64_t>("univ_start_datetime");
    univ_end_datetime_ = config<int64_t>("univ_end_datetime");
    sim_start_datetime_ = config<int64_t>("sim_start_datetime", univ_start_datetime_);
    sim_end_datetime_ = config<int64_t>("sim_end_datetime", univ_end_datetime_);

    for (auto &t : config<std::vector<std::string>>("intraday_times")) {
      intraday_times_.PushBack(std::stoi(t));
    }
  }

  univ_indices_id_start_ = config<int>("univ_indices_id_start", default_univ_indices_id_start());
  univ_indices_ = config<std::vector<std::string>>("univ_indices", default_univ_indices());

  if (auto taq_config = config_["taq"]) {
    taq_ = taq_config.Get<std::string>("module");
    for (auto &t : taq_config.Get<std::vector<std::string>>("times")) {
      taq_times_.PushBack(std::stoi(t));
    }
  }

  fs::create_directories(cache_dir_.GetPath(name()));

  // check existing meta
  auto meta_path = GetMetaPath();
  Config loaded_meta;
  DateTimeIndex prev_datetimes;
  if (fs::exists(meta_path)) {
    loaded_meta = Config::LoadFile(meta_path);
    CheckMetaChanges(loaded_meta);
    prev_datetimes = DateTimeIndex::Load(cache_dir_.GetPath(name(), "datetimes"));
  }

  // build datetimes
  auto trade_dates_file = config<std::string>("trade_dates");
  std::ifstream ifs(trade_dates_file);
  ENSURE(ifs.good(), "Failed to open {}", trade_dates_file);
  std::string line;
  while (std::getline(ifs, line)) {
    std::vector<std::string_view> tokens = StrSplit(line, ',');
    int64_t date;
    ENSURE(SafeAtoi(tokens[0], date), "Invalid date: {}", tokens[0]);
    if (daily_) {
      if (date >= univ_start_datetime_ && date <= univ_end_datetime_) {
        datetimes_.PushBack(date);
      }
    } else {
      for (auto time : intraday_times_) {
        auto dt = CombineDateTime(date, time);
        if (dt >= univ_start_datetime_ && dt <= univ_end_datetime_) {
          datetimes_.PushBack(dt);
        }
      }
    }
  }
  // check datetimes changes
  for (int i = 0; i < prev_datetimes.size(); ++i) {
    ENSURE(datetimes_[i] == prev_datetimes[i],
           "trade_datetimes changed, please delete cache and rebuild");
  }
  datetimes_.set_path(cache_dir_.GetPath(name(), "datetimes"));
  datetimes_.Save();
  LOG_INFO("Loaded {} datetimes", datetimes_.size());

  univ_ = UnivIndex::Load(cache_dir().GetPath(name(), "univ"));
  if (univ_.empty()) {
    univ_.SetIndices(univ_indices_, univ_indices_id_start_);
    ENSURE(univ_.max_id() < max_univ_size_, "Max instrument id exceeds max_univ_size");
  }

  days_per_year_ = config_.Get<int>("days_per_year", 250);
  short_book_size_ = config_.Get<bool>("short_book_size", false);
  benchmark_index_ = config_.Get<std::string>("benchmark_index", "");

  PostLoad();

  BuildImpl(prev_datetimes.size());

  ENSURE(max_univ_size_ >= univ_.size(), "max_univ_size_ ({}) is smaller than univ size ({})",
         max_univ_size_, univ_.size());

  SaveMeta();
}

void Env::BuildImpl(int new_start_dti) {
  auto listing = Array<bool>::MMap(cache_dir().GetPath("env", "listing"),
                                   {max_datetimes_size(), max_univ_size()});
  listing.FillNull(start_di(), end_di());

  auto sec_master = config<std::string>("sec_master");
  LOG_INFO("Using sec_master {}", sec_master);
  auto file = io::OpenBufferedFile(sec_master);
  ENSURE2(!file->error());
  std::string line;
  file->ReadLine(line);
  SimpleCsvReader reader(line, '|');
  auto last_univ_date = SplitDateTime(datetimes_.items().back()).first;
  int64_t datetime_multiplier = daily_ ? 1 : 10000;
  while (file->ReadLine(line)) {
    reader.ReadNext(line);
    auto symbol = reader["sid"];
    int list_date = reader.Get<int>("list");
    int delist_date = 0;
    if (!reader["delist"].empty()) {
      delist_date = reader.Get<int>("delist");
    }

    if (reader.header().count("list_entry") > 0 && !reader["list_entry"].empty()) {
      // use entry date if it comes later
      int list_entry = reader.Get<int>("list_entry");
      if (list_entry > last_univ_date) continue;
      if (list_entry > list_date) list_date = list_entry;
    }
    if (reader.header().count("delist_entry") > 0 && !reader["delist_entry"].empty()) {
      // use entry date if it comes later
      int delist_entry = reader.Get<int>("delist_entry");
      if (delist_entry > delist_date) delist_date = delist_entry;
    }

    // all dates >= list_date should be included
    int list_dti = datetimes_.LowerBound(list_date * datetime_multiplier);
    if (list_dti == end_dti()) list_dti = UnivIndex::UNKNOWN_LIST_DI;
    // newly added stock should not slip into historical dates, but live data
    // changes should also be permitted.
    int list_start_dti = new_start_dti;
    if (live())
      list_start_dti = std::min(list_start_dti, datetimes_.LowerBound(sim_start_datetime_));
    ENSURE(univ_.Find(symbol) >= 0 || list_dti >= list_start_dti,
           "sec_master history changed, please delete cache and rebuild: new symbol {} {}", symbol,
           list_date);
    int delist_dti = end_dti();
    if (delist_date > 0) {
      // all dates >= delist_date should be excluded
      delist_dti = std::min(end_dti(), datetimes_.LowerBound(delist_date * datetime_multiplier));
    }
    int ii = univ_.GetOrInsert(list_dti, std::string(symbol));
    for (int dti = std::max(start_dti(), list_dti); dti < std::min(end_dti(), delist_dti); ++dti) {
      listing(dti, ii) = true;
    }
  }
  for (auto &symbol : univ_indices_) {
    int ii = univ_.Find(symbol);
    for (int dti = start_dti(); dti < end_dti(); ++dti) {
      listing(dti, ii) = true;
    }
  }
  univ_.Save();
}

void Env::SaveMeta() {
  Config meta;
  FillMeta(meta);

  auto meta_path = GetMetaPath();
  std::ofstream ofs(meta_path);
  ofs << meta.ToYamlString() << std::endl;
  LOG_DEBUG("Saved {}", meta_path);
}

void Env::FillMeta(Config &meta) {
  meta.Set("daily", daily_);
  meta.Set("max_univ_size", max_univ_size_);
  meta.Set("univ_start_datetime", univ_start_datetime_);
  meta.Set("univ_end_datetime", univ_end_datetime_);
  meta.Set("univ_indices", univ_indices_);
  meta.Set("univ_indices_id_start", univ_indices_id_start_);
  meta.Set("intraday_times", intraday_times_.items());
  meta.Set("taq_times", taq_times_.items());
  meta.Set("days_per_year", days_per_year_);
  meta.Set("short_book_size", short_book_size_);
  meta.Set("benchmark_index", benchmark_index_);
}

void Env::CheckMetaChanges(const Config &meta) const {
  ENSURE(meta.Get<bool>("daily", true) == daily_, "daily changed, please delete cache and rebuild");
  ENSURE(meta.Get<int64_t>("univ_start_datetime", 0) == univ_start_datetime_,
         "univ_start_datetime changed, please delete cache and rebuild");
  ENSURE(meta.Get<int64_t>("univ_end_datetime", 0) <= univ_end_datetime_,
         "univ_end_datetime becomes smaller, please delete cache and rebuild");
  ENSURE(meta.Get<int>("univ_indices_id_start", 0) == univ_indices_id_start_,
         "univ_indices_id_start changed, please delete cache and rebuild");
  ENSURE(meta.Get<std::vector<std::string>>("univ_indices", {}) == univ_indices_,
         "univ_indices changed, please delete cache and rebuild");
  ENSURE(meta.Get<std::vector<int>>("intraday_times", {}) == intraday_times_.items(),
         "intraday_times changed, please delete cache and rebuild");
}

void Env::Load() {
  ENSURE2(config_);

  // check existing meta
  auto meta_path = GetMetaPath();
  auto meta = Config::LoadFile(meta_path);

  daily_ = meta.Get<bool>("daily");
  max_univ_size_ = meta.Get<int>("max_univ_size");
  univ_start_datetime_ = meta.Get<int64_t>("univ_start_datetime");
  univ_end_datetime_ = meta.Get<int64_t>("univ_end_datetime");
  if (daily_) {
    sim_start_datetime_ = config<int64_t>("sim_start_date", univ_start_datetime_);
    sim_end_datetime_ = config<int64_t>("sim_end_date", univ_end_datetime_);
  } else {
    sim_start_datetime_ = config<int64_t>("sim_start_datetime", univ_start_datetime_);
    sim_end_datetime_ = config<int64_t>("sim_end_datetime", univ_end_datetime_);
  }
  days_per_year_ = meta.Get<int>("days_per_year");
  short_book_size_ = meta.Get<bool>("short_book_size");
  benchmark_index_ = meta.Get<std::string>("benchmark_index");

  univ_indices_id_start_ = meta.Get<int>("univ_indices_id_start");
  univ_indices_ = meta.Get<std::vector<std::string>>("univ_indices");

  for (auto t : meta.Get<std::vector<int>>("intraday_times")) {
    intraday_times_.PushBack(t);
  }

  taq_ = config_["taq"].Get<std::string>("module", "");
  for (auto t : meta.Get<std::vector<int>>("taq_times")) {
    taq_times_.PushBack(t);
  }

  datetimes_ = DateTimeIndex::Load(cache_dir_.GetPath(name(), "datetimes"), true);
  univ_ = UnivIndex::Load(cache_dir().GetPath(name(), "univ"), true);
  PostLoad();
}

void Env::PostLoad() {
  start_dti_ = datetimes_.LowerBound(sim_start_datetime_);
  end_dti_ = datetimes_.UpperBound(sim_end_datetime_);
  if (rerun_manager_) {
    rerun_manager_->SetDates(datetimes_[0], datetimes_[end_dti_ - 1]);
  }
}

REGISTER_ENV(Env);

}  // namespace yang
