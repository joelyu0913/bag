import importlib
import logging
import sys

import numba
import numpy as np

from sim import Module
from yao.lib.pycommon.oper import ts_mean
from yao.lib.simm import Sim, apply_ops


class DemoPy(Module):
    def run_impl(self):
        sim = Sim({"sys_cache": self.cache_dir.sys_dir, "user_cache": self.cache_dir.user_dir})
        sim.load_base(["dvol"])
        sim.register_crypto()
        lookback = 24
        sim.start_di = max(self.start_di, lookback)
        sim.end_di = self.end_di
        du = (slice(sim.start_di, sim.end_di), slice(sim.univ_size))

        if "output" in self.config:
            output = self.config["output"]
        else:
            output = f"{self.config['name']}/b_sig"
        alpha = sim.write_mod(output)

        alpha[du] = np.nan

        du_pre = (slice(sim.start_di - lookback, sim.end_di), slice(sim.univ_size))
        b_dvol = sim.b_dvol[du_pre]
        scale_dvol = b_dvol / ts_mean(np.abs(b_dvol.T), lookback).T

        mask_ = scale_dvol[lookback:, :] > 0
        alpha[du][mask_] = 1.0

        apply_ops(self, sim, alpha)
        print(" .  xxx  ", self.start_di, self.end_di)
