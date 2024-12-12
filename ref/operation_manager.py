from __future__ import annotations

import numpy as np

import yang.sim.ext
from yang.data import Array
from yang.sim import Env

OperationDesc = tuple[str, list[str], dict[str, str]]


class OperationManager(object):
    def __init__(self, env: Env):
        def path_func(name: str):
            if "/" in name:
                return env.cache_dir.get_path(name)
            elif name.startswith("i_"):
                return env.cache_dir.get_path("ibase", name)
            else:
                return env.cache_dir.get_path("base", name)

        self.env = env
        self.op_manager = yang.sim.ext.OperationManager()
        self.op_manager.initialize(self.env.cpp_env)

    def parse_ops(self, raw_ops: str | list[dict]) -> list[OperationDesc]:
        ops = []
        if isinstance(raw_ops, str):
            for op_desc in raw_ops.split("|"):
                args = op_desc.split(":")
                ops.append((args[0], args[1:], {}))
        else:
            for op_desc in raw_ops:
                ops.append((op_desc["name"], op_desc.get("args", []), op_desc.get("kwargs", {})))
        return ops

    def apply(
        self,
        sig_in: Array | np.array,
        sig_out: Array | np.array,
        start_di: int,
        end_di: int,
        ops: str | list[dict],
    ) -> None:
        """
        Inplace application is supported: sig_in and sig_out can be the same array.
        """
        if isinstance(sig_in, Array):
            sig_in = sig_in.data
        if isinstance(sig_out, Array):
            sig_out = sig_out.data
        ops = self.parse_ops(ops)
        self.op_manager.apply(sig_in, sig_out, start_di, end_di, ops)
