### transform cpp to python

#include "yang/io/open.h"
#include "yang/sim/module.h"
#include "yang/util/dates.h"
#include "yang/util/datetime.h"
#include "yang/util/fs.h"
#include "yang/util/logging.h"
#include "yang/util/simple_csv.h"

import os, sys
import numpy as np
import logging
import datetime
import gzip

from data import Array, Index
from sim import Module
from basic.lib.pycommon.data import read_header, read_line

class SupTaq5minCn(Module):
  def run_impl(self):

    self.taq_times_ = self.env.config["taq"].get("times")
    self.ignore_missing_ = self.config.get("ignore_missing", [])
    self.b_cumadj = self.read_array("base/cumadj")
    taq_sz = len(self.taq_times_) 
    self.i_trd_last_ = self.write_array("sup_taq/i_trd_last", shape=(self.max_dates_size, taq_sz, self.max_univ_size), null_value=np.nan)
    self.i_mid_last_ = self.write_array("sup_taq/i_mid_last", shape=(self.max_dates_size, taq_sz, self.max_univ_size), null_value=np.nan)
    self.i_trd_high_ = self.write_array("sup_taq/i_trd_high", shape=(self.max_dates_size, taq_sz, self.max_univ_size), null_value=np.nan)
    self.i_trd_low_ = self.write_array("sup_taq/i_trd_low", shape=(self.max_dates_size, taq_sz, self.max_univ_size), null_value=np.nan)
    self.i_trd_cumdvol_ = self.write_array("sup_taq/i_trd_cumdvol", shape=(self.max_dates_size, taq_sz, self.max_univ_size), null_value=np.nan)
    self.i_trd_cumvol_ = self.write_array("sup_taq/i_trd_cumvol", shape=(self.max_dates_size, taq_sz, self.max_univ_size), null_value=np.nan)
    self.i_trd_cumvwap_ = self.write_array("sup_taq/i_trd_cumvwap", shape=(self.max_dates_size, taq_sz, self.max_univ_size), null_value=np.nan)
    self.i_trd_dvol_ = self.write_array("sup_taq/i_trd_dvol", shape=(self.max_dates_size, taq_sz, self.max_univ_size), null_value=np.nan)
    self.i_trd_vol_ = self.write_array("sup_taq/i_trd_vol", shape=(self.max_dates_size, taq_sz, self.max_univ_size), null_value=np.nan)
    self.i_trd_vwap_ = self.write_array("sup_taq/i_trd_vwap", shape=(self.max_dates_size, taq_sz, self.max_univ_size), null_value=np.nan)

    self.i_adj_trd_last_ = self.write_array("sup_taq/i_adj_trd_last", shape=(self.max_dates_size, taq_sz, self.max_univ_size), null_value=np.nan)
    self.i_adj_mid_last_ = self.write_array("sup_taq/i_adj_mid_last", shape=(self.max_dates_size, taq_sz, self.max_univ_size), null_value=np.nan)
    self.i_adj_trd_high_ = self.write_array("sup_taq/i_adj_trd_high", shape=(self.max_dates_size, taq_sz, self.max_univ_size), null_value=np.nan)
    self.i_adj_trd_low_ = self.write_array("sup_taq/i_adj_trd_low", shape=(self.max_dates_size, taq_sz, self.max_univ_size), null_value=np.nan)
    self.i_adj_trd_cumvol_ = self.write_array("sup_taq/i_adj_trd_cumvol", shape=(self.max_dates_size, taq_sz, self.max_univ_size), null_value=np.nan)
    self.i_adj_trd_cumvwap_ = self.write_array("sup_taq/i_adj_trd_cumvwap", shape=(self.max_dates_size, taq_sz, self.max_univ_size), null_value=np.nan)
    self.i_adj_trd_vol_ = self.write_array("sup_taq/i_adj_trd_vol", shape=(self.max_dates_size, taq_sz, self.max_univ_size), null_value=np.nan)
    self.i_adj_trd_vwap_ = self.write_array("sup_taq/i_adj_trd_vwap", shape=(self.max_dates_size, taq_sz, self.max_univ_size), null_value=np.nan)

    stock_taq_path_fmt = self.config["stock_taq_path"]
    idx_taq_path_fmt = self.config["idx_taq_path"]
    for di in range(self.start_di, self.end_di):
      date = self.dates[di]
      live_mode = self.env.live and di == self.end_di - 1
      if live_mode:
        logging.info(f"taq live mode at {self.env.current_time}")
      stock_taq_path = datetime.datetime.strptime(str(date), '%Y%m%d').strftime(stock_taq_path_fmt)
      idx_taq_path = datetime.datetime.strptime(str(date), '%Y%m%d').strftime(idx_taq_path_fmt)
      stock_cnt = 0
      idx_cnt = 0
      for taqi in range(taq_sz):
        if taqi != 0:
          
          self.i_trd_last_[di, taqi, :] = self.i_trd_last_[di, taqi - 1, :]
          self.i_mid_last_[di, taqi, :] = self.i_mid_last_[di, taqi - 1, :]
          self.i_trd_high_[di, taqi, :] = self.i_trd_high_[di, taqi - 1, :]
          self.i_trd_low_[di, taqi, :] = self.i_trd_low_[di, taqi - 1, :]

          self.i_trd_cumdvol_[di, taqi, :] = self.i_trd_cumdvol_[di, taqi - 1, :]
          self.i_trd_cumvol_[di, taqi, :] = self.i_trd_cumvol_[di, taqi - 1, :]
          self.i_trd_cumvwap_[di, taqi, :] = self.i_trd_cumvwap_[di, taqi - 1, :]
        else:
          self.i_trd_cumdvol_[di, taqi, :] = 0
          self.i_trd_cumvol_[di, taqi, :] = 0
          self.i_trd_dvol_[di, taqi, :] = 0
          self.i_trd_vol_[di, taqi, :] = 0

        if live_mode and self.env.current_time < self.taq_times_[taqi]:
          stock_cnt += 1
          idx_cnt += 1
          continue
        stock_cnt += self.LoadFile(stock_taq_path, di, taqi, False)
        idx_cnt += self.LoadFile(idx_taq_path, di, taqi, True)
        
      if stock_cnt < taq_sz:
        logging.error(f"missing stock taq on {date}: {stock_cnt} / {taq_sz}")
      if idx_cnt < taq_sz:
        logging.error(f"missing idx taq on {date}: {idx_cnt} / {taq_sz}")
      logging.info(f"[{self.name}] Update information on date {self.dates[di]}. (finished {di + 1}/{self.dates_size})")
      return

  def LoadFile(self, taq_date_path, di, taqi, is_idx):
    taq_path = taq_date_path.format(self.taq_times_[taqi])
    if not os.path.exists(taq_path):
      if self.taq_times_[taqi] in self.ignore_missing_:
        return True
      else:
        logging.error(f"Missing {taq_path}")
        return False
    logging.info(f"Loading {taq_path}")
    # try:
    if True:
      with gzip.open(taq_path, 'rt') as f:
        header_mp = read_header(f)
        for raw_line in f:
          line = read_line(raw_line, header_mp, float_fields = ["close", "high", "low", "vol", "amt"])

          sid = line["sid"]
          if len(sid) != 9:
            logging.fatal(f"sid {sid} error")
          ii = self.univ.find(sid)
          if ii > -1:
            self.i_trd_last_[di, taqi, ii] = line["close"]
            self.i_mid_last_[di, taqi, ii] = self.i_trd_last_[di, taqi, ii]
            self.i_trd_high_[di, taqi, ii] = line["high"]
            self.i_trd_low_[di, taqi, ii] = line["low"]

            cumdvol = self.i_trd_cumdvol_[di, taqi, ii]
            cumvol = self.i_trd_cumvol_[di, taqi, ii]
            if not np.isfinite(cumdvol): cumdvol = 0
            if not np.isfinite(cumvol): cumvol = 0
            interval_dvol = line["amt"]
            interval_vol = line["vol"]
            if np.isfinite(interval_dvol):
              cumdvol += interval_dvol
              self.i_trd_dvol_[di, taqi, ii] = interval_dvol
            else:
              self.i_trd_dvol_[di, taqi, ii] = 0
            if interval_vol > 0:
              cumvol += interval_vol
              self.i_trd_vol_[di, taqi, ii] = interval_vol
              self.i_trd_vwap_[di, taqi, ii] = interval_dvol / interval_vol
            else:
              self.i_trd_vol_[di, taqi, ii] = 0
              self.i_trd_vwap_[di, taqi, ii] = self.i_trd_last_[di, taqi, ii]
            if is_idx:
              self.i_trd_cumvwap_[di, taqi, ii] = self.i_trd_last_[di, taqi, ii]
            else:
              if cumvol > 0:
                self.i_trd_cumvwap_[di, taqi, ii] = cumdvol / cumvol
              else:
                self.i_trd_cumvwap_[di, taqi, ii] = self.i_trd_last_[di, taqi, ii]
            self.i_adj_trd_last_[di, taqi, ii] = self.i_trd_last_[di, taqi, ii] * self.b_cumadj[di, ii]
            self.i_adj_mid_last_[di, taqi, ii] = self.i_mid_last_[di, taqi, ii] * self.b_cumadj[di, ii]
            self.i_adj_trd_high_[di, taqi, ii] = self.i_trd_high_[di, taqi, ii] * self.b_cumadj[di, ii]
            self.i_adj_trd_low_[di, taqi, ii] = self.i_trd_low_[di, taqi, ii] * self.b_cumadj[di, ii]
            self.i_adj_trd_cumvol_[di, taqi, ii] = self.i_trd_cumvol_[di, taqi, ii] / self.b_cumadj[di, ii]
            self.i_adj_trd_cumvwap_[di, taqi, ii] = self.i_trd_cumvwap_[di, taqi, ii] * self.b_cumadj[di, ii]
            self.i_adj_trd_vol_[di, taqi, ii] = self.i_trd_vol_[di, taqi, ii] / self.b_cumadj[di, ii]
            self.i_adj_trd_vwap_[di, taqi, ii] = self.i_trd_vwap_[di, taqi, ii] * self.b_cumadj[di, ii]
    # catch Exception as e:
    #   logging.error(f"Failed to load {taq_path}: {e}")
    return True

