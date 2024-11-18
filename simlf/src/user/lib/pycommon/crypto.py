import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numba import guvectorize

from yao.lib.pycommon.data import round_float
from yao.lib.pycommon.math import compute_cos


@guvectorize(
    ["void(float64[:], float64[:])"],
    "(n)->(n)",
)
def ts_ffillna(a, out):

    flag = False
    for i in range(len(a)):
        if not flag and np.isfinite(a[i]):
            flag = True
            out[i] = a[i]
        elif flag and ~np.isfinite(a[i]):
            out[i] = 0.0
        else:
            out[i] = a[i]


def nan_to_zero(x):
    if ~np.isfinite(x):
        return 0.0
    return x


class Crypto:
    def __init__(self, sim, flag_print, freq=24):
        self.sim = sim
        self.flag_print = flag_print
        self.cost = {}
        self.freq = freq

    def register_extra(self):
        if not hasattr(self, "b_close"):
            self.load_base("close")
        self.first_di = {}
        for ii in range(self.sim.univ_size):
            for di in range(self.sim.dates_size):
                if np.isfinite(self.sim.b_close[di, ii]):
                    self.first_di[ii] = di
                    break
        return self

    def set_costs(self, sids, list_bps):
        self.cost = {}
        for sid, bps in zip(sids, list_bps):
            self.cost[sid] = bps

    def set_cost(self, bps):
        self.cost = {}
        self.cost["default"] = bps

    def get_ii_(self, sid_or_ii):
        if isinstance(sid_or_ii, str):
            return self.sim.univ.find(sid_or_ii.upper())
        else:
            return sid_or_ii

    def get_first_di(self, sid_or_ii):
        ii = self.get_ii_(sid_or_ii)
        if ii in self.first_di:
            return self.first_di[ii]
        else:
            return 100000

    def get_iis(self, sig, di=0):
        if isinstance(sig, str):
            sig = self.sim.load_mod(sig)

        return np.where(sig[di, : self.sim.univ_size])[0]

    def get_univ(self, sig, di=0):

        iis = self.get_iis(sig, di)
        return sorted([self.sim.univ[ii] for ii in iis])

    def _compute_pnl(self, sig, sid_or_ii, start_date=-1, end_date=-1, ret=None):
        warnings.filterwarnings("ignore")
        ii = self.get_ii_(sid_or_ii)
        if ii < 0:
            raise RuntimeError("Error:", sid_or_ii)

        if end_date == -1:
            end_di = self.sim.end_di
        else:
            end_di = self.sim.dates.less_equal_than(end_date)

        if start_date == -1:
            start_di = 0
        else:
            start_di = self.sim.dates.greater_equal_than(start_date)

        if isinstance(sig, str):
            ts_sig = self.sim.load_mod(sig)[start_di:end_di, ii].copy()
        else:
            ts_sig = sig[start_di:end_di, ii].copy()

        ts_sig = ts_ffillna(ts_sig)
        if ret is None:
            if not hasattr(self, "b_ret"):
                self.sim.load_base("ret")
            ts_ret = self.sim.b_ret[start_di:end_di, ii].copy()[1:]
        else:
            ts_ret = ret[start_di:end_di, ii].copy()[:-1]

        sid = self.sim.univ[ii]
        if sid in self.cost:
            fee_rate = self.cost[sid]
        elif "default" in self.cost:
            fee_rate = self.cost["default"]
        else:
            fee_rate = 0.0

        tvrs = np.abs(ts_sig[1:] - ts_sig[:-1])
        pnls = ts_sig[:-1] * ts_ret - tvrs * fee_rate

        info = {}
        info["cos"] = nan_to_zero(compute_cos(ts_sig[:-1], ts_ret))
        info["tvr"] = nan_to_zero(np.nanmean(tvrs))
        # print(info['tvr'])
        pnl_mean = nan_to_zero(np.nanmean(pnls))

        if np.abs(info["tvr"]) > 1e-20:
            info["pnl_bps"] = np.nansum(pnls) / np.nansum(tvrs) * 1e4
        else:
            info["pnl_bps"] = 0.0
        info["total_pnl"] = np.nansum(pnls)
        info["yearly_pnl"] = pnl_mean * self.freq * 365
        pnl_std = np.nanstd(pnls)
        if pnl_std > 1e-20:
            info["ir"] = pnl_mean / pnl_std * np.sqrt(self.freq * 365)
        else:
            info["ir"] = 0.0

        pnl_cum = pd.Series(np.nancumsum(pnls))
        info["dd"] = np.abs((pnl_cum - pnl_cum.cummax()).min())
        if self.flag_print and np.nansum(np.abs(pnls)) > 1e-20:
            for col in ["cos", "tvr", "pnl_bps", "total_pnl", "yearly_pnl", "ir", "dd"]:
                print(f"{col}: {round_float(float(info[col]), 6)}")

            fig, ax = plt.subplots(1, 2, figsize=(24, 10))
            plt.rc("font", size=30)
            ax[0].plot(np.nancumsum(pnls), label=sid_or_ii)
            ax[1].hist(ts_sig, bins=100)

            ax[0].legend(loc="upper left")
        return pnls, info

    def compute_pnl(self, sig, sids, start_date=-1, end_date=-1, ret=None):
        if isinstance(sids, list):
            return {sid: self._compute_pnl(sig, sid, start_date, end_date, ret) for sid in sids}
        else:
            return self._compute_pnl(sig, sids, start_date, end_date, ret)

    def compute_pnls(self, sig, sids, start_date=-1, end_date=-1, ret=None):
        warnings.filterwarnings("ignore")
        list_pnls = []
        infos = {}
        flag_print_ = self.flag_print
        self.flag_print = False
        for sid in sids:
            pnls_, infos[sid] = self.compute_pnl(sig, sid, start_date, end_date, ret)
            list_pnls.append((sid, pnls_))

        self.flag_print = flag_print_
        total_pnls = np.full([len(list_pnls[0][1])], 0.0)
        total_info = {"tvr": 0}
        total_pnl = 0.0
        for sid, pnls in list_pnls:
            pnls[~np.isfinite(pnls)] = 0.0
            total_pnls += pnls
            total_pnl += np.nansum(pnls)

        for sid, info in infos.items():
            total_info["tvr"] += info["tvr"]
        total_info["total_pnl"] = total_pnl
        total_info["yearly_pnl"] = total_pnl / len(total_pnls) * self.freq * 365
        total_pnl_mean = nan_to_zero(np.nanmean(total_pnls))
        total_yearly_std = np.nanstd(total_pnls) * np.sqrt(self.freq * 365)
        total_info["ir"] = total_info["yearly_pnl"] / total_yearly_std

        if total_info["tvr"] > 0.0:
            total_info["pnl_bps"] = total_pnl_mean / total_info["tvr"] * 1e4
        else:
            total_info["pnl_bps"] = 0.0
        pnl_cum = pd.Series(np.nancumsum(total_pnls))
        total_info["dd"] = np.abs((pnl_cum - pnl_cum.cummax()).min())
        if total_info["dd"] > 0:
            total_info["calmar_ratio"] = total_info["yearly_pnl"] / total_info["dd"]
        else:
            total_info["calmar_ratio"] = 0.0

        if len(total_pnls) > 0:
            total_info["win_rate"] = np.sum(total_pnls > 1e-15) / np.sum(np.abs(total_pnls) > 1e-15)
        else:
            total_info["win_rate"] = 0.0

        for key, val in total_info.items():
            total_info[key] = round_float(float(val), 4)

        return total_pnls, total_info, list_pnls, infos

    def sort(self, x, key, sids, start_date):
        mp_info = []
        flag_print_ = self.flag_print
        self.flag_print = False
        for sid in sids:
            pnls, info = self.comupte_pnl(x, sid, start_date=start_date)
            mp_info.append((sid, info[key]))
        self.flag_print = flag_print_
        return sorted(mp_info, key=lambda x: x[1], reverse=True)

    def filter(self, alpha, sig, sids, scalers=[]):
        du = (slice(self.sim.start_di, self.sim.end_di), slice(self.sim.univ_size))
        alpha[du] = np.nan
        sum_scaler = 0.0
        for idx, sid in enumerate(sids):
            ii = self.sim.univ.find(sid)
            if len(scalers) > 0:
                scaler = scalers[idx]
            else:
                scaler = 1.0

            sum_scaler += scaler
            alpha[self.sim.start_di : self.sim.end_di, ii] = (
                sig[self.sim.start_di : self.sim.end_di, ii] * scaler
            )

        alpha[du] /= sum_scaler

    def show_pnls(self, alpha_w, start_date=-1, end_date=-1, sids=[], ret=None):
        w = 1
        if isinstance(alpha_w, tuple) or isinstance(alpha_w, list):
            alpha, w = alpha_w
        else:
            alpha = alpha_w

        if isinstance(alpha, str):
            xx = self.sim.load_mod(alpha) * w
            alpha_name = alpha
        else:
            xx = alpha * w
            alpha_name = ""

        if start_date == -1:
            start_date = self.sim.dates[0]
        if end_date == -1:
            end_date = self.sim.dates[-1]

        _ = self.compute_pnls(xx, sids=sids, start_date=start_date, end_date=end_date, ret=ret)
        print(_[1])
        plt.plot(np.nancumsum(_[0]), label=alpha_name)
