from sim import Module
from basic.lib.simm import Sim, apply_ops

class SupUnivClone(Module):
    def run_impl(self):
        sim = Sim({"sys_cache": self.cache_dir.sys_dir, "user_cache": self.cache_dir.user_dir})
        univ_to = sim.write_mod(self.config["par_to"], dtype=bool)
        univ_from = sim.load_mod(self.config["par_from"])
        univ_to[sim.start_di:sim.end_di] = univ_from[sim.start_di:sim.end_di]