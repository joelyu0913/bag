from sim import Module
from basic.lib.simm import Sim, apply_ops
import numpy as np
import logging

class SupFretRawCNEQ(Module):
    def run_impl(self):
      
        window = int(self.config["window"])
        par_type = self.config.get("par_type", "raw")
        v = par_type.split('_')

        sim = Sim({"sys_cache": self.cache_dir.sys_dir})
        sim.start_di = max(self.start_di, self.config['window'])
        sim.end_di = self.end_di
        du_s = (slice(sim.start_di, sim.end_di - window), slice(sim.max_univ_size))
        du_e = (slice(sim.start_di+ window, sim.end_di), slice(sim.max_univ_size))
        adj_close = sim.load_mod('base/adj_close')
        close = sim.load_mod('base/close')
        ret = sim.load_mod('base/ret')
        if not self.config.get("limit_filter", False):
            fret = sim.write_mod(f"ext/fret_{par_type}_{window}")
        else:
            qtl = self.config.get("quantile", 1.0)
            trun = self.config.get("truncate", 10.)
            fret = sim.write_mod(f"ext/fret_{par_type}_{window}_{qtl}_{trun}")

        if v[0] == "raw":
            # for di in range(sim.start_di, sim.end_di - window):
            #     for ii in range(sim.max_univ_size):
            #         fret[di, ii] = adj_close[di + window, ii] / adj_close[di, ii] - 1
            fret[du_s] = adj_close[du_e] / adj_close[du_s] - 1.
        elif v[0] == "index":
            idx = sim.univ.index_id_start() + int(v[1])
            for di in range(sim.start_di, sim.end_di - window):
                hedge_val = close[di + window, idx] / close[di, idx] - 1
                for ii in range(sim.max_univ_size):
                    fret[di, ii] = adj_close[di + window, ii] / adj_close[di, ii] - 1 - hedge_val
        else:
            logging.fatal(f"Invalid par_type: {par_type}")


        if self.config.get("limit_filter", False):
            adj = sim.load_mod('base/adj')
            limit_up = sim.load_mod('base/limit_up')
            limit_down = sim.load_mod('base/limit_down')
            qtl = self.config.get("quantile", 1.0)
            if qtl == 1.0:
                for di in range(sim.start_di, sim.end_di - window):
                    for ii in range(sim.univ_size):
                        if close[di, ii] > limit_up[di, ii] - 1e-3 or close[di, ii] < limit_down[di, ii] + 1e-3:
                            fret[di, ii] = np.nan
            else:
                for di in range(sim.start_di + 1, sim.end_di - window):
                    for ii in range(sim.univ_size):
                        ret_ = ret[di, ii]
                        ret_up = limit_up[di, ii] * adj[di, ii] / close[di - 1, ii] - 1
                        ret_down = limit_down[di, ii] * adj[di, ii] / close[di - 1, ii] - 1
                        if ret_ > ret_up * qtl or ret_ < ret_down * qtl:
                            fret[di, ii] = np.nan

            trun = self.config.get("truncate", 10.)
            for di in range(sim.start_di, sim.end_di - self.config['window']):
                for ii in range(sim.max_univ_size):
                    if fret[di, ii] > trun:
                        fret[di, ii] = trun
                    elif fret[di, ii] < -trun:
                        fret[di, ii] = -trun
