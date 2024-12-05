from __future__ import annotations
import numpy as np

from data import Array
from sim import Env
import logging

OperationDesc = tuple[str, list[str], dict[str, str]]


class OperationManager:
    def __init__(self, env: Env):
        self.env = env

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
        logging.error(1312312312312) #todo
    
        if isinstance(sig_in, Array):
            sig_in = sig_in.data
        if isinstance(sig_out, Array):
            sig_out = sig_out.data

        ops = self.parse_ops(ops)
        for ops_ in ops:
            logging.error(ops_)

        # selfapply(sig_in, sig_out, start_di, end_di, ops)
