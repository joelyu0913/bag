import logging
import sys, os
import yaml
from data import DateTimeIndex, UnivIndex, Array
import pandas as pd
import numpy as np

def env_init(config): 
  user_mode_ = True
  if config.get("cache", ''):
    sys_cache = f'{config["cache"]}/sys'
    os.makedirs(sys_cahce_, exist_ok=True)
  elif config.get("sys_cache", ''):
    sys_cache_ = config["sys_cache"]
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

    # for auto &t : config<std::vector<std::string>>("intraday_times"):
    #   intraday_times_.PushBack(std::stoi(t))
    # }

  univ_indices_id_start_ = int(config["univ_indices_id_start"])
  univ_indices_ = config.get("univ_indices", {}) 

  dir_env = f"{config['sys_cache']}/env/"
  os.makedirs(dir_env, exist_ok=True)
  # check existing meta
  meta_path = f"{dir_env}/meta.yml"
  # DateTimeIndex prev_datetimes
  prev_datetimes = None
  if os.path.exists(meta_path):
    loaded_meta = yaml.safe_load(open(meta_path))
    # CheckMetaChanges(loaded_meta)
    # todo
    prev_datetimes = DateTimeIndex.load(f'{dir_env}/datetimes')
    print(prev_datetimes)

  # build datetimes
  df = pd.read_csv(config['trade_dates'], header = None)
  df_date = df[(int(config['univ_start_date']) <= df[0]) & (df[0] <= int(config['univ_end_date']))]
  df_date.reset_index(drop=True, inplace=True)
  print(df_date.head())
  print('xxxxxxxx')
  datetimes_ = DateTimeIndex(np.array(df_date[0]))


  #   } else {
  #     for auto time : intraday_times_:
  #       auto dt = CombineDateTime(date, time)
  #       if dt >= univ_start_datetime_ && dt <= univ_end_datetime_:
  #         datetimes_.PushBack(dt)
  #       }
  #     }
  #   }
  # }
  # check datetimes changes

  if prev_datetimes:
    for i in range(len(prev_datetimes)):
      assert datetimes_[i] == prev_datetimes[i], "trade_datetimes changed, please delete cache and rebuild"
  pd.DataFrame(datetimes_.items).to_csv(f'{dir_env}/datetimes', index=False, header=False)
  logging.info(f"Loaded {len(datetimes_)} datetimes")

  univ_ = UnivIndex.load(f'{dir_env}/univ') 
  if len(univ_):
    univ_.set_indices(univ_indices_, univ_indices_id_start_)
    print(univ_.max_id(), max_univ_size_)
    assert univ_.max_id() < max_univ_size_, "Max instrument id exceeds max_univ_size"

  days_per_year_ = int(config.get("days_per_year", 250))
  short_book_size_ = bool(config.get("short_book_size", False))
  benchmark_index_ = config.get("benchmark_index", "")

  start_dti_ = datetimes_.lower_bound(sim_start_datetime_)
  end_dti_ = datetimes_.upper_bound(sim_end_datetime_)

  new_start_dti = len(prev_datetimes) if prev_datetimes else 0
  listing = Array.mmap(f"{dir_env}/listing", True, (len(datetimes_), max_univ_size_), bool, False)

  sec_master = config["sec_master"]
  logging.info(f"Using sec_master {sec_master}")
  df_sec = pd.read_csv(sec_master, sep='|', dtype=str)
  last_univ_date = datetimes_.items[-1]
  datetime_multiplier = 1 if daily_ else 10000
  for idx, line in df_sec.iterrows():
    print(idx)
    symbol = line['sid']
    list_date = int(line['list'])
    delist_date = 0

    if str(line['delist']) != 'nan':
        delist_date = int(line['delist'])

    if str(line['list_entry']) != 'nan':
        list_entry_date = int(line['list_entry'])
        if list_entry_date > last_univ_date:
            continue
        if list_entry_date > list_date:
            list_date = list_entry_date
    if str(line['delist_entry']) != 'nan':
        delist_entry_date = int(line['delist_entry'])
        if delist_entry_date > delist_date:
            delist_date = delist_entry_date

    # all dates >= list_date should be included
    list_dti = datetimes_.lower_bound(list_date * datetime_multiplier)
    # if list_dti == end_dti_: 
    #   list_dti = UnivIndex::UNKNOWN_LIST_DI
    # newly added stock should not slip into historical dates, but live data
    # changes should also be permitted.
    # list_start_dti = new_start_dti
#     if live())
#       list_start_dti = std::min(list_start_dti, datetimes_.LowerBound(sim_start_datetime_))
#     ENSURE(univ_.Find(symbol) >= 0 || list_dti >= list_start_dti,
#            "sec_master history changed, please delete cache and rebuild: new symbol {} {}", symbol,
#            list_date)
    delist_dti = end_dti_
    if delist_date > 0:
      # all dates >= delist_date should be excluded
      delist_dti = min(delist_dti, datetimes_.lower_bound(delist_date * datetime_multiplier))
      ii = univ_.get_or_insert(list_dti, symbol)
      for dti in range(max(start_dti_, list_dti), min(end_dti_, delist_dti)):
        listing[dti, ii] = True

  for symbol in univ_indices_:
    ii = univ_.find(symbol)
    for dti in range(start_dti_, end_dti_):
      listing[dti, ii] = True
  univ_.save(f'{dir_env}/univ')
  assert max_univ_size_ >= len(univ_), f"max_univ_size_ ({max_univ_size_}) is smaller than univ size ({len(univ_)})"


#   meta.Set("univ_start_datetime", univ_start_datetime_)
#   meta.Set("univ_end_datetime", univ_end_datetime_)
#   meta.Set("univ_indices", univ_indices_)
#   meta.Set("univ_indices_id_start", univ_indices_id_start_)
#   meta.Set("intraday_times", intraday_times_.items())
#   meta.Set("taq_times", .items())
#   meta.Set("days_per_year", days_per_year_)
#   meta.Set("short_book_size", short_book_size_)
#   meta.Set("benchmark_index", benchmark_index_)

  meta["daily"] = daily_
  meta["max_univ_size"] = max_univ_size_
  meta["univ_start_datetime"] = univ_start_datetime_
  meta["univ_end_datetime"] = univ_end_datetime_
  print(meta)
  return
  # meta["univ_start_datetime"] = config['univ_start_date'] if 'univ_start_date' in config else config['univ_start_datetime']
  # meta["univ_end_datetime"] = config['univ_end_date'] if 'univ_end_date' in config else config['univ_end_datetime']
  # meta["univ_indices"] = config.get('univ_indices', {})
  # meta["univ_indices_id_start"] = config['univ_indices_id_start']
  # meta["intraday_times"] = config.get('intraday_times', {})
  # meta["taq_times"] =  config['taq']['times']
  # meta["days_per_year"] = config['days_per_year']
  # meta["short_book_size"] = config['short_book_size']
  # meta["benchmark_index"] = config['benchmark_index']

#   auto meta_path = GetMetaPath()
#   std::ofstream ofs(meta_path)
#   ofs << meta.ToYamlString() << std::endl
#   LOG_DEBUG("Saved {}", meta_path)
