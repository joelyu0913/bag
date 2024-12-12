import logging

from sim import Module
from basic.lib.simm import Sim, apply_ops

class SupUnivSt(Module):
    def run_impl(self):
        st_univ = self.write_array(self.config["output"], dtype=bool)
        st_arr = self.read_array(self.config["st_path"])
        for di in range(self.start_di, self.end_di):
            mask1 = (st_arr[di] == 1)
            mask2 = (st_arr[di] == 0)
            mask3 = ~(mask1 | mask2)
            st_univ[di][mask1] = True
            st_univ[di][mask2] = False
            if di > 0:
                st_univ[di][mask3] = st_univ[di - 1][mask3]
