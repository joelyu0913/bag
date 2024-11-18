import importlib
import logging
import sys

import numpy as np

from yang.sim import Module
from yao.lib.simm import Sim, apply_ops


class AlphaLoader(Module):
    def run_impl(self):
        sim = Sim({"sys_cache": self.cache_dir.sys_dir, "user_cache": self.cache_dir.user_dir})
        sim.start_di = self.start_di
        sim.end_di = self.end_di

        user_alpha = importlib.import_module(self.config["code_path"])
        if "no_sig" in self.config:
            user_alpha.cal_alpha(sim, self.config)
        else:
            if "output" in self.config:
                alpha = sim.write_mod(self.config["output"])
            else:
                alpha = sim.write_mod(f"{self.config['name']}/b_sig")
            du = (slice(sim.start_di, sim.end_di), slice(sim.univ_size))

            alpha[du] = np.nan
            user_alpha.cal_alpha(sim, self.config, alpha)
            apply_ops(self, sim, alpha)
