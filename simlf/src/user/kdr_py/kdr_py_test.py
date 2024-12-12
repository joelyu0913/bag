import logging
import sys

import numba
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sim import Module
from basic.lib.pycommon.oper import ts_mean, c_demean
from basic.lib.simm import Sim, apply_ops

def rolling_sum(nums, window_size=5):
    # Use numpy's convolve to compute the rolling sum
    return np.convolve(nums, np.ones(window_size), 'fill')[:len(nums)]

def c_scale(arr, sum_to=1):
    arr[~np.isfinite(arr)] = 0
    x = np.abs(arr).sum(axis = 1)
    return (arr.T / x).T * sum_to

class KdrPyTest(Module):
    def run_impl(self):
        sim = Sim({"sys_cache": self.cache_dir.sys_dir, "user_cache": self.cache_dir.user_dir})
        # sim.load_base(["ret"])
        sim.load_base(["close"])
        lookback = self.config['lookback']
        sim.start_di = self.start_di
        sim.end_di = self.end_di
        run_start_di = max(sim.start_di, lookback)
        du = (slice(sim.start_di, sim.end_di), slice(sim.univ_size))
        univ_all = sim.load_mod('base/univ_all')

        if "output" in self.config:
            output = self.config["output"]
        else:
            output = f"{self.config['name']}/b_sig"
        alpha = sim.write_mod(output)

        alpha[sim.start_di:sim.end_di] = np.nan

        b_close = sim.b_close
        alpha[du] = b_close[du]* 100000
        if self.config.get('univ', False):
            alpha[~univ_all] = np.nan
        
        # ret_mean = ts_mean(alpha[du_pre].T, lookback).T

        # alpha[du] = ret_mean[(slice(sim.end_di - sim.start_di), slice(sim.univ_size))]
        alpha[du] = c_scale(c_demean(alpha[du]), 2e5)

