from typing import Union

import numpy as np

from data import Array
from sim import Env, Module
# from basic.operation_manager import OperationManager


class BaseDaily(Module):
    def __init__(self, name: str, config: dict, env: Env):
        Module.__init__(self, name, config, env)
        self.b_sig = None
        self.data_loaded = set()

    def after_run(self):
        self._apply_ops()

    def base_load(self, *names: list[str]):
        for name in names:
            if name in self.data_loaded:
                continue
            self.data_loaded.add(name)
            if name == "univ":
                self.b_univ = Array.load(self.cache_dir.get_path("sup_univ", self.config["__univ"]))
            elif name == "sig":
                self.b_sig = self.write_array(self.name, "b_sig", null_value=np.nan)
            else:
                raise RuntimeError(f"Unknown data: {name}")

    def _apply_ops(self):
        ops = self.config.get("ops", "")
        if ops == "":
            return

        assert self.b_sig is not None
        # self.base_load("sig")
        # sig_op = self.write_array(self.name, "b_sig_op", null_value=np.nan)
        # op_manager = OperationManager(self.env)
        # op_manager.apply(self.b_sig, sig_op, self.start_di, self.end_di, ops)
