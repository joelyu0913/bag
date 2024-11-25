#include "yang/sim/module.h"
#include "yang/util/dates.h"
#include "yang/util/simple_csv.h"
#include "yang/util/strings.h"
#include "yang/util/unordered_map.h"

# transform code from C++ to Python
import sys, os
import logging
import numpy as np
import pandas as pd
import datetime

from data import Array, Index
from sim import Module
class CnBaseStd(Module):
  '''
  0 | SSE 主板    | SSE Main
  1 | SZSE 主板   | SZSE Main
  2 | SZSE 创业板 | SZSE ChiNext
  3 | SZSE 中小板 | SZSE SME
  4 | SSE 科创板  | SSE STAR
  5 | SSE CDR     | SSE CDR
  6 | BSE         | BSE Main
  '''
  #  exchanges = {
  #     {"SSE Main", 0}, {"SZSE Main", 1}, {"SZSE ChiNext", 2}, {"SZSE SME", 3},
  #     {"SSE STAR", 4}, {"SSE CDR", 5},   {"BSE Main", 6}}
  def run_impl(self):
    exchanges = {"SSE Main": 0, "SZSE Main": 1, "SZSE ChiNext": 2, "SZSE SME": 3, "SSE STAR": 4, "SSE CDR": 5, "BSE Main": 6}

    env = self.env

    indices = {}
    for idx in env.univ:
      indices[idx] = env.univ.find(idx)

    open_arr = self.write_array("base/open")
    close_arr = self.write_array("base/close")
    high_arr = self.write_array("base/high")
    low_arr = self.write_array("base/low")
    vol_arr = self.write_array("base/vol", null_value=0)
    dvol_arr = self.write_array("base/dvol", null_value=0)
    vwap_arr = self.write_array("base/vwap")

    sector_idx = Index.load(self.cache_dir.get_read_path("base/sector_idx"))
    industry_idx = Index.load(self.cache_dir.get_read_path("base/industry_idx"))
    subindustry_idx = Index.load(self.cache_dir.get_read_path("base/subindustry_idx"))
    if len(sector_idx) == 0:
      sector_idx.insert("")
    if len(industry_idx) == 0:
      industry_idx.insert("")
    if len(subindustry_idx) == 0:
      subindustry_idx.insert("")

    cty_arr = self.write_array("base/cty", dtype=np.int32, null_value=-1)
    sector_arr = self.write_array("base/sector", dtype=np.int32, null_value=-1)
    industry_arr = self.write_array("base/industry", dtype=np.int32,null_value=-1)
    subindustry_arr = self.write_array("base/subindustry", dtype=np.int32, null_value=-1)
    sharesout_arr = self.write_array("base/sharesout", null_value=0)
    sharesfloat_arr = self.write_array("base/sharesfloat", null_value=0)
    cap_arr = self.write_array("base/cap", null_value=0)
    cumadj_arr = self.write_array("base/cumadj")
    adj_arr = self.write_array("base/adj", null_value=1.0)
    st_arr = self.write_array("base/st", dtype=np.int32, null_value=-1)
    exch_arr = self.write_array("base/exch", dtype=np.int32, null_value=-1)
    halt_arr = self.write_array("base/halt", dtype=bool, null_value=False)
    limit_up_arr = self.write_array("base/limit_up")
    limit_down_arr = self.write_array("base/limit_down")

    univ_all = self.write_array("base/univ_all", dtype=bool)
    listing = Array.mmap(self.cache_dir.get_read_path("env", "listing"))

    raw_prc_file = env.config["raw_prc_file"]
    index_file = env.config["index_file"]
    for di in range(self.start_di, self.end_di):
      date = env.dates[di]

      raw_prc_path = datetime.datetime.strptime(str(date), '%Y%m%d').strftime(raw_prc_file)
      index_path = datetime.datetime.strptime(str(date), '%Y%m%d').strftime(index_file) 
      logging.info(f"Loading {raw_prc_path}")
      logging.info(f"Loading {index_path}")

      updated_stock = 0
      updated_idx = 0

      univ_all[di] = listing[di]

      try:
        df = pd.read_csv(raw_prc_path, sep='|')
        for _, line in df.iterrows():
          sid = line["sid"]
          if len(sid) != 9:
            logging.fatal(f"sid {sid} error")
          
          ii = env.univ.find(sid)

          if ii < 0:
            continue
          open_arr[di, ii] = line["open"]
          close_arr[di, ii] = line["close"]
          high_arr[di, ii] = line["high"]
          low_arr[di, ii] = line["low"]
          vol_arr[di, ii] = line["vol"]
          dvol_arr[di, ii] = line["dvol"]
          vwap_arr[di, ii] = 0 if vol_arr[di, ii] == 0 else dvol_arr[di, ii] / vol_arr[di, ii]

          sector = line["sector"]
          industry = line["ind"]
          subindustry = line["subind"]

          cty_arr[di, ii] = 1
          sector_arr[di, ii] = sector_idx.insert(sector)
          industry_arr[di, ii] = industry_idx.insert(industry)
          subindustry_arr[di, ii] = subindustry_idx.insert(subindustry)

          sharesout_arr[di, ii] = line["sho"]
          sharesfloat_arr[di, ii] = line["flo"]
          cap_arr[di, ii] = sharesout_arr[di, ii] * line["pclose"]

  #         adj_arr(di, ii) = reader.Get<float>("adj");
  #         if (!IsValid(adj_arr(di, ii))) {
  #           adj_arr(di, ii) = 1.;
  #         }

  #         if (di > 0) {
  #           float adj_ = close_arr(di - 1, ii) / reader.Get<float>("pclose");
  #           if (std::abs(adj_arr(di, ii) - adj_) > 1e-5) {
  #             LOG_ERROR("Invalid adj: {} - {}, {} - {}: {}, {}", di, dates()[di], ii, univ()[ii],
  #                       adj_arr(di, ii), adj_);
  #           }
  #         }
  #         auto sid_name = reader["name"];
  #         auto st_pos = sid_name.find("ST");

  #         if (st_pos == std::string::npos) {
  #           st_arr(di, ii) = 0;
  #         } else {
  #           if (st_pos > 2) LOG_WARN("{},{},{}", sid_name, st_pos, sid_name.size());
  #           st_arr(di, ii) = 1;
  #         }
          adj_arr[di, ii] = line["adj"] if np.isfinite(line["adj"]) else 1.0
          if di > 0:
            if "pclose" in line:
              adj_ = close_arr[di - 1, ii] / line["pclose"]
              if np.abs(adj_arr[di, ii] - adj_) > 1e-5:
                logging.error(f"Invalid adj: {di} - {env.dates[di]}, {ii} - {env.univ[ii]}, {adj_arr[di, ii]}, {adj_}")
            else:
              pass
          sid_name = line["name"]
          st_pos = sid_name.find("ST")
          if st_pos == -1:
            st_arr[di, ii] = 0
          else:
            if st_pos > 2:
              logging.warn(f"{sid_name}, {st_pos}, {len(sid_name)}")
            st_arr[di, ii] = 1
  #         if (reader.header().count("halt")) {
  #           halt_arr(di, ii) = reader.Get<int>("halt", 0) != 0;
  #         } else {
  #           halt_arr(di, ii) = 1 - reader.Get<int>("active", 1);
  #         }
  #         if (halt_arr(di, ii)) univ_all(di, ii) = false;

  #         limit_up_arr(di, ii) = reader.Get<float>("up");
  #         limit_down_arr(di, ii) = reader.Get<float>("down");
  #         auto exch = reader["exch"];
  #         if (exchanges.count(exch)) {
  #           exch_arr(di, ii) = exchanges.at(exch);
  #         } else {
  #           LOG_WARN("Unknown exchange: {}", exch);
  #         }

  #         if (indices.contains(std::string_view(sid))) {
  #           updated_idx++;
  #         } else {
  #           updated_stock++;
  #         }
          if 'halt' in line:
            halt_arr[di, ii] = line.get("halt", 0) != 0
          else:
            halt_arr[di, ii] = 1 - line.get("active", 1)
          if halt_arr[di, ii]:
            univ_all[di, ii] = False
          limit_up_arr[di, ii] = line["up"] 
          limit_down_arr[di, ii] = line["down"]
          exch = line["exch"]
          if exch in exchanges:
            exch_arr[di, ii] = exchanges[exch]
          else:
            logging.warn(f"Unknown exchange: {exch}")
          
          if indices.get(sid):
            updated_idx += 1
          else:
            updated_stock += 1
      except Exception as e:
        logging.error(f"Failed to load {raw_prc_path}: {e}")

  #         auto iid = std::string(reader["sid"]);
  #         if (iid.size() != 9) LOG_FATAL("sid {} error", iid);
  #         int ii = env.univ().Find(di, iid);

  #         if (ii < 0) continue;

  #         open_arr(di, ii) = reader.Get<float>("open");
  #         close_arr(di, ii) = reader.Get<float>("close");
  #         high_arr(di, ii) = reader.Get<float>("high");
  #         low_arr(di, ii) = reader.Get<float>("low");
  #         vol_arr(di, ii) = reader.Get<float>("vol");
  #         dvol_arr(di, ii) = reader.Get<float>("dvol");
  #         vwap_arr(di, ii) = vol_arr(di, ii) == 0 ? 0 : dvol_arr(di, ii) / vol_arr(di, ii);

  #         cumadj_arr(di, ii) = 1;
  #         adj_arr(di, ii) = 1;
  #         univ_all(di, ii) = true;

  #         limit_up_arr(di, ii) = high_arr(di, ii) * 2;
  #         limit_down_arr(di, ii) = 0.;

  #         if (indices.contains(iid)) {
  #           updated_idx++;
  #         }
  #       }
  #     } catch (const std::exception &ex) {
  #       LOG_FATAL("Failed to load {}: {}", index_path, ex.what());
  #     }
  #     LOG_INFO("[{}] [{}] Loaded {} stocks, {} indices", name(), date, updated_stock, updated_idx);
  #   }
  #   sector_idx.Save();
  #   industry_idx.Save();
  #   subindustry_idx.Save();

      try:
        df = pd.read_csv(index_path, sep='|')
        for _, line in df.iterrows():
          iid = line["sid"]
          if len(iid) != 9:
            logging.fatal(f"sid {iid} error")
          ii = env.univ.find(iid)
          if ii < 0:
            continue
          open_arr[di, ii] = line["open"]
          close_arr[di, ii] = line["close"]
          high_arr[di, ii] = line["high"]
          low_arr[di, ii] = line["low"]
          vol_arr[di, ii] = line["vol"]
          dvol_arr[di, ii] = line["dvol"]
          vwap_arr[di, ii] = 0 if vol_arr[di, ii] == 0 else dvol_arr[di, ii] / vol_arr[di, ii]

          cumadj_arr[di, ii] = 1
          adj_arr[di, ii] = 1
          univ_all[di, ii] = True

          limit_up_arr[di, ii] = 99999999
          limit_down_arr[di, ii] = 0

          if indices.get(iid):
            updated_idx += 1
      except Exception as e:
        logging.fatal(f"Failed to load {index_path}: {e}")
      logging.info(f"[{self.name}] [{date}] Loaded {updated_stock} stocks, {updated_idx} indices")
    sector_idx.save()
    industry_idx.save()
    subindustry_idx.save()

    ret_arr = self.write_array("base/ret")
    adj_open_arr = self.write_array("base/adj_open")
    adj_close_arr = self.write_array("base/adj_close")
    adj_high_arr = self.write_array("base/adj_high")
    adj_low_arr = self.write_array("base/adj_low")
    adj_vol_arr = self.write_array("base/adj_vol", null_value=0)
    adj_vwap_arr = self.write_array("base/adj_vwap")

    for ii in range(env.max_univ_size):
      cumadj_arr[0, ii] = 1.0
    for di in range(self.start_di, self.end_di):
      for ii in range(env.max_univ_size):
        if di == 0:
          cumadj_arr[di, ii] = 1.0
        else:
          cumadj_arr[di, ii] = cumadj_arr[di - 1, ii] * adj_arr[di, ii]
        adj_open_arr[di, ii] = open_arr[di, ii] * cumadj_arr[di, ii]
        adj_close_arr[di, ii] = close_arr[di, ii] * cumadj_arr[di, ii]
        adj_high_arr[di, ii] = high_arr[di, ii] * cumadj_arr[di, ii]
        adj_low_arr[di, ii] = low_arr[di, ii] * cumadj_arr[di, ii]
        adj_vwap_arr[di, ii] = vwap_arr[di, ii] * cumadj_arr[di, ii]

        adj_vol_arr[di, ii] = vol_arr[di, ii] / cumadj_arr[di, ii]

        if di > 0:
          ret_arr[di, ii] = close_arr[di, ii] * adj_arr[di, ii] / close_arr[di - 1, ii] - 1
        else:
          ret_arr[di, ii] = np.nan
      