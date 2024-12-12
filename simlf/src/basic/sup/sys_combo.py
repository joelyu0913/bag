import numpy as np
from numba import float32, int64, njit, types, void
from numba.typed import List

from data import Array

# from yao.B.basic.base_daily import BaseDaily
from sim import Module
from basic.operation_manager import OperationManager


@njit(
    void(float32[:, ::1], types.Array(float32, 2, "C", readonly=True), float32),
)
def add_sig(x, y, weight):
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if not np.isfinite(x[i, j]):
                x[i, j] = y[i, j] * weight
            elif np.isfinite(y[i, j]):
                x[i, j] += y[i, j] * weight


class SysCombo(Module):
    def run_impl(self):
        ops = self.config.get("ops", "")
        output = self.config.get("output")
        if output and ops == "":
            sig_raw = self.write_array(output, null_value=np.nan)
        else:
            sig_raw = self.write_array(f"{self.name}/b_sig", null_value=np.nan)

        for signal in self.config.get("signals", []):
            wt = float32(signal["weight"])
            sig = Array.mmap(self.cache_dir.get_path(signal["signal"]))
            add_sig(
                sig_raw[self.start_di : self.end_di, :], sig[self.start_di : self.end_di, :], wt
            )

        if ops != "":
            if output:
                sig_op = self.write_array(output, null_value=np.nan)
            else:
                sig_op = self.write_array(f"{self.name}/b_sig_op", null_value=np.nan)
            op_manager = OperationManager(self.env)
            op_manager.apply(sig_raw, sig_op, self.start_di, self.end_di, ops)


@njit()
def combo(output, sigs, weights):
    for di in range(output.shape[0]):
        for ii in range(output.shape[1]):
            weight_sum = 0
            sig_sum = 0
            for i in range(len(sigs)):
                x = sigs[i][di, ii]
                if np.isfinite(x):
                    sig_sum += x * weights[i]
                    weight_sum += weights[i]
            if weight_sum > 0:
                output[di][ii] = sig_sum / weight_sum
            else:
                output[di][ii] = np.nan


class SysComboMean(Module):
    def run_impl(self):
        ops = self.config.get("ops", "")
        output = self.config.get("output")
        if output and ops == "":
            sig_raw = self.write_array(output, null_value=np.nan)
        else:
            sig_raw = self.write_array(f"{self.name}/b_sig", null_value=np.nan)

        weights = []
        sigs = List()
        for signal in self.config.get("signals", []):
            weights.append(signal["weight"])
            sig = Array.mmap(self.cache_dir.get_path(signal["signal"]))
            sigs.append(sig[self.start_di : self.end_di])
        combo(sig_raw[self.start_di : self.end_di], sigs, np.array(weights, dtype="float32"))

        if ops != "":
            if output:
                sig_op = self.write_array(output, null_value=np.nan)
            else:
                sig_op = self.write_array(f"{self.name}/b_sig_op", null_value=np.nan)
            op_manager = OperationManager(self.env)
            op_manager.apply(sig_raw, sig_op, self.start_di, self.end_di, ops)
