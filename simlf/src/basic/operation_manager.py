from __future__ import annotations
import numpy as np

from data import Array
from sim import Env
import logging

from basic.lib.pycommon.oper import ts_mean, c_demean, c_rank, c_scale

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
    
        if isinstance(sig_in, Array):
            sig_in = sig_in.data
        if isinstance(sig_out, Array):
            sig_out = sig_out.data

        sig_out[start_di:end_di] = sig_in[start_di:end_di]
        ops_list = self.parse_ops(ops)
        for ops_info in ops_list:
            ops_ = ops_info[0]
            if ops_ == 'rank':
                sig_out[start_di:end_di] = c_rank(sig_out[start_di:end_di])

            elif ops_ == 'scale':
                scale_to = 2e5
                if ops_info[1]:
                    scale_to = float(ops_info[1][0])
                sig_out[start_di:end_di] = c_scale(sig_out[start_di:end_di], scale_to)
            elif ops_ == 'neut':
                sig_out[start_di:end_di] = c_demean(sig_out[start_di:end_di])
            elif ops_ == 'univ':
                univ_str = ops_info[1][0]
                # b_univ = self.env.read_array(univ_str)
                b_univ= self.env.read_data(Array, f'sup_univ/{univ_str}').data
                sig_out[start_di:end_di][~b_univ[start_di:end_di]] = np.nan

            elif ops_ == 'ts_mean':
                pass
                # alpha[du] = ret_mean[(slice(sim.end_di - sim.start_di), slice(sim.univ_size))]
                #         ret_mean = ts_mean(b_ret.T, lookback).T
            else:
                logging.warning(f"Operation {ops_} not supported.")

