import logging

from sim import Module
from basic.lib.simm import Sim, apply_ops

class SupUnivMerge(Module):
    def run_impl(self):
      self.cal()

    def include(self, a, b):
        a[self.start_di:self.end_di][b[self.start_di:self.end_di]] = True

    def intersect(self, a, b):
        a[self.start_di:self.end_di][~b[self.start_di:self.end_di]] = False

    def exclude(self, a, b):
        a[self.start_di:self.end_di][b[self.start_di:self.end_di]] = False

    def cal(self):
        univ = self.config["par_univ"]
        b_univ_custom = self.write_array(univ,dtype=bool)
        vs_include = self.config.get("par_include", [])
        vs_intersect = self.config.get("par_intersect", [])
        vs_exclude = self.config.get("par_exclude", [])
        b_univ_custom[self.start_di:self.end_di] = False

        for str_univ in vs_include:
            logging.info(f"Include: {str_univ}")
            if str_univ:
                b_univ_tmp = self.read_array(str_univ)
                self.include(b_univ_custom, b_univ_tmp)
        for str_univ in vs_intersect:
            logging.info(f"Intersect: {str_univ}")
            if str_univ:
                b_univ_tmp = self.read_array(str_univ)
                self.intersect(b_univ_custom, b_univ_tmp)
        for str_univ in vs_exclude:
            logging.info(f"Exclude: {str_univ}")
            if str_univ:
                b_univ_tmp = self.read_array(str_univ)
                self.exclude(b_univ_custom, b_univ_tmp)