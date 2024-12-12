import logging
import sys

import numba
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sim import Module
from basic.lib.pycommon.oper import ts_mean, c_demean, c_scale
from basic.lib.simm import Sim, apply_ops

def rolling_sum(nums, window_size=5):
    # Use numpy's convolve to compute the rolling sum
    return np.convolve(nums, np.ones(window_size), 'fill')[:len(nums)]


class KdrPy(Module):
    def run_impl(self):
        sim = Sim({"sys_cache": self.cache_dir.sys_dir, "user_cache": self.cache_dir.user_dir})
        sim.load_base(["ret"])
        lookback = self.config['lookback']
        sim.start_di = max(self.start_di, lookback)
        sim.end_di = self.end_di
        du = (slice(sim.start_di, sim.end_di), slice(sim.univ_size))
        univ_all = sim.load_mod('base/univ_all')

        if "output" in self.config:
            output = self.config["output"]
        else:
            output = f"{self.config['name']}/b_sig"
        alpha = sim.write_mod(output)

        alpha[du] = np.nan

        du_pre = (slice(sim.start_di - lookback, sim.end_di), slice(sim.univ_size))
        b_ret = sim.b_ret[du_pre]
        ret_mean = ts_mean(b_ret.T, lookback).T

        alpha[du] = ret_mean[(slice(sim.end_di - sim.start_di), slice(sim.univ_size))]
        alpha[du] = c_scale(c_demean(alpha[du]), 2e5)

