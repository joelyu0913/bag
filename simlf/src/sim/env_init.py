import logging
import sys, os

def env_init(config_): 
  user_mode_ = True
  if config_["cache"]:
    sys_cache_ = f'{config_["cache"]}/sys'
    os.makedirs(sys_cahce_, exist_ok=True)
  elif config_["sys_cache"]:
    sys_cache_ = config_["sys_cache"]
    os.makedirs(sys_cache_, exist_ok=True)
  else:
    logging.fatal("cache config missing")

  logging.info(f"sys_cache: {sys_cache_}")

  daily_ = config.get("daily", True)
  max_univ_size_ = int(config["max_univ_size"])
  if daily_:
      univ_start_datetime_ = int(config["univ_start_date"])
      univ_end_datetime_ = int(config["univ_end_date"])
      sim_start_datetime_ = int(config.get("sim_start_date", univ_start_datetime_))
      sim_end_datetime_ = int(config.get("sim_end_date", univ_end_datetime_))
  else:
      univ_start_datetime_ = int(config["univ_start_datetime"])
      univ_end_datetime_ = int(config["univ_end_datetime"])
      sim_start_datetime_ = int(config.get("sim_start_datetime", univ_start_datetime_))
      sim_end_datetime_ = int(config.get("sim_end_datetime", univ_end_datetime_))

    # for (auto &t : config<std::vector<std::string>>("intraday_times")) {
    #   intraday_times_.PushBack(std::stoi(t))
    # }

  univ_indices_id_start_ = int(config["univ_indices_id_start"])
  univ_indices_ = config.get("univ_indices", {}) #????


  # fs::create_directories(cache_dir_.GetPath(name()))

  dir_env = f"{config['sys_cache']}/env/"
  os.makedirs(dir_env, exist_ok=True)
  # yaml.safe_dump(meta, open(, 'w'), sort_keys=False)
  # check existing meta
  meta_path = f"{dir_env}/meta.yml"
  # DateTimeIndex prev_datetimes
  if os.path.exists(meta_path)
    loaded_meta = yaml.safe_load(open(meta_path))
    # CheckMetaChanges(loaded_meta)
    # prev_datetimes = DateTimeIndex::Load(cache_dir_.GetPath(name(), "datetimes"))
  return

  // build datetimes
  auto trade_dates_file = config<std::string>("trade_dates")
  std::ifstream ifs(trade_dates_file)
  ENSURE(ifs.good(), "Failed to open {}", trade_dates_file)
  std::string line
  while (std::getline(ifs, line)) {
    std::vector<std::string_view> tokens = StrSplit(line, ',')
    int64_t date
    ENSURE(SafeAtoi(tokens[0], date), "Invalid date: {}", tokens[0])
    if (daily_) {
      if (date >= univ_start_datetime_ && date <= univ_end_datetime_) {
        datetimes_.PushBack(date)
      }
    } else {
      for (auto time : intraday_times_) {
        auto dt = CombineDateTime(date, time)
        if (dt >= univ_start_datetime_ && dt <= univ_end_datetime_) {
          datetimes_.PushBack(dt)
        }
      }
    }
  }
  // check datetimes changes
  for (int i = 0 i < prev_datetimes.size() ++i) {
    ENSURE(datetimes_[i] == prev_datetimes[i],
           "trade_datetimes changed, please delete cache and rebuild")
  }
  datetimes_.set_path(cache_dir_.GetPath(name(), "datetimes"))
  datetimes_.Save()
  LOG_INFO("Loaded {} datetimes", datetimes_.size())

  univ_ = UnivIndex::Load(cache_dir().GetPath(name(), "univ"))
  if (univ_.empty()) {
    univ_.SetIndices(univ_indices_, univ_indices_id_start_)
    ENSURE(univ_.max_id() < max_univ_size_, "Max instrument id exceeds max_univ_size")
  }

  days_per_year_ = config_.Get<int>("days_per_year", 250)
  short_book_size_ = config_.Get<bool>("short_book_size", false)
  benchmark_index_ = config_.Get<std::string>("benchmark_index", "")


  BuildImpl(prev_datetimes.size())


void Env::BuildImpl(int new_start_dti) {
  auto listing = Array<bool>::MMap(cache_dir().GetPath("env", "listing"),
                                   {max_datetimes_size(), max_univ_size()})
  listing.FillNull(start_di(), end_di())

  auto sec_master = config<std::string>("sec_master")
  LOG_INFO("Using sec_master {}", sec_master)
  auto file = io::OpenBufferedFile(sec_master)
  ENSURE2(!file->error())
  std::string line
  file->ReadLine(line)
  SimpleCsvReader reader(line, '|')
  auto last_univ_date = SplitDateTime(datetimes_.items().back()).first
  int64_t datetime_multiplier = daily_ ? 1 : 10000
  while (file->ReadLine(line)) {
    reader.ReadNext(line)
    auto symbol = reader["sid"]
    int list_date = reader.Get<int>("list")
    int delist_date = 0
    if (!reader["delist"].empty()) {
      delist_date = reader.Get<int>("delist")
    }

    if (reader.header().count("list_entry") > 0 && !reader["list_entry"].empty()) {
      // use entry date if it comes later
      int list_entry = reader.Get<int>("list_entry")
      if (list_entry > last_univ_date) continue
      if (list_entry > list_date) list_date = list_entry
    }
    if (reader.header().count("delist_entry") > 0 && !reader["delist_entry"].empty()) {
      // use entry date if it comes later
      int delist_entry = reader.Get<int>("delist_entry")
      if (delist_entry > delist_date) delist_date = delist_entry
    }

    // all dates >= list_date should be included
    int list_dti = datetimes_.LowerBound(list_date * datetime_multiplier)
    if (list_dti == end_dti()) list_dti = UnivIndex::UNKNOWN_LIST_DI
    // newly added stock should not slip into historical dates, but live data
    // changes should also be permitted.
    int list_start_dti = new_start_dti
    if (live())
      list_start_dti = std::min(list_start_dti, datetimes_.LowerBound(sim_start_datetime_))
    ENSURE(univ_.Find(symbol) >= 0 || list_dti >= list_start_dti,
           "sec_master history changed, please delete cache and rebuild: new symbol {} {}", symbol,
           list_date)
    int delist_dti = end_dti()
    if (delist_date > 0) {
      // all dates >= delist_date should be excluded
      delist_dti = std::min(end_dti(), datetimes_.LowerBound(delist_date * datetime_multiplier))
    }
    int ii = univ_.GetOrInsert(list_dti, std::string(symbol))
    for (int dti = std::max(start_dti(), list_dti) dti < std::min(end_dti(), delist_dti) ++dti) {
      listing(dti, ii) = true
    }
  }
  for (auto &symbol : univ_indices_) {
    int ii = univ_.Find(symbol)
    for (int dti = start_dti() dti < end_dti() ++dti) {
      listing(dti, ii) = true
    }
  }
  univ_.Save()
  ENSURE(max_univ_size_ >= univ_.size(), "max_univ_size_ ({}) is smaller than univ size ({})",
         max_univ_size_, univ_.size())


  Config meta
  meta.Set("daily", daily_)
  meta.Set("max_univ_size", max_univ_size_)
  meta.Set("univ_start_datetime", univ_start_datetime_)
  meta.Set("univ_end_datetime", univ_end_datetime_)
  meta.Set("univ_indices", univ_indices_)
  meta.Set("univ_indices_id_start", univ_indices_id_start_)
  meta.Set("intraday_times", intraday_times_.items())
  meta.Set("taq_times", .items())
  meta.Set("days_per_year", days_per_year_)
  meta.Set("short_book_size", short_book_size_)
  meta.Set("benchmark_index", benchmark_index_)

    meta["daily"] = daily_
    max_univ_size = config['max_univ_size']
    meta["max_univ_size"] = max_univ_size
    meta["univ_start_datetime"] = config['univ_start_date'] if 'univ_start_date' in config else config['univ_start_datetime']
    meta["univ_end_datetime"] = config['univ_end_date'] if 'univ_end_date' in config else config['univ_end_datetime']
    meta["univ_indices"] = config.get('univ_indices', {})
    meta["univ_indices_id_start"] = config['univ_indices_id_start']
    meta["intraday_times"] = config.get('intraday_times', {})
    meta["taq_times"] =  config['taq']['times']
    meta["days_per_year"] = config['days_per_year']
    meta["short_book_size"] = config['short_book_size']
    meta["benchmark_index"] = config['benchmark_index']

  auto meta_path = GetMetaPath()
  std::ofstream ofs(meta_path)
  ofs << meta.ToYamlString() << std::endl
  LOG_DEBUG("Saved {}", meta_path)
