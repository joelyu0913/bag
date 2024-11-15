import importlib
import logging
import sys

import numba
import numpy as np

from sim import Module
from yao.lib.pycommon.oper import ts_mean
from yao.lib.simm import Sim, apply_ops


class LongOne(Module):
    def run_impl(self):
        sim = Sim({"sys_cache": self.cache_dir.sys_dir, "user_cache": self.cache_dir.user_dir})
        # load params
        lookback = 24

        if "output" in self.config:
            output = self.config["output"]
        else:
            output = f"{self.config['name']}/b_sig"

        sids = self.config["sids"]

        ### pre_run
        sim.start_di = max(self.start_di, lookback)
        sim.end_di = self.end_di
        du = (slice(sim.start_di, sim.end_di), slice(sim.univ_size))
        alpha = sim.write_mod(output)
        alpha[sim.start_di : sim.end_di] = np.nan

        ### run
        for sid, wt in sids:
            alpha[sim.start_di : sim.end_di, sim.univ.find(sid)] = wt

        ##
        apply_ops(self, sim, alpha)
