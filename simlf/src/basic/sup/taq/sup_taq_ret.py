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

class SupTaqRet(Module):
  def run_impl(self):

    self.taq_times_ = self.env.config["taq"].get("times")
    taq_sz = len(self.taq_times_) 

    self.i_ret = self.write_array("sup_taq/i_ret_trd_last", shape=(self.max_dates_size, taq_sz, self.max_univ_size), null_value=np.nan)
    self.i_adj_trd_last = self.read_array("sup_taq/i_adj_trd_last")

    self.i_ret[self.start_di:self.end_di, 1:, :] = 1. - self.i_adj_trd_last[self.start_di:self.end_di, :-1, :] / self.i_adj_trd_last[self.start_di:self.end_di, 1:, :]
    if self.start_di > 0:
      self.i_ret[self.start_di:self.end_di, 0, :] = 1. - self.i_adj_trd_last[self.start_di - 1, -1, :] / self.i_adj_trd_last[self.start_di:self.end_di, 0, :] 
    else:
      self.i_ret[self.start_di+1:self.end_di, 0, :] = 1. - self.i_adj_trd_last[self.start_di, -1, :] / self.i_adj_trd_last[self.start_di+1:self.end_di, 0, :] 
      self.i_ret[0, 0, :] = 0. 
