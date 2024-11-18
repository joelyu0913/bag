import importlib
import os
import sys
from collections import OrderedDict
from typing import Optional, TextIO, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate

from yang.data import Array
from yang.sim import DISPLAY_BOOK_SIZE, Env
from yang.util import in_notebook
from yao.lib.pycommon.data import round_float


def cal_pnl_tvr(sig, ret, start_di, end_di):
    if start_di == 0:
        tvr_ts = np.full((end_di - start_di,), 0.0)
        tvr_ts[0] = np.sum(np.abs(sig[0]))
        tvr_ts[1 : end_di - start_di] = np.sum(
            np.abs(sig[start_di + 1 : end_di, :] - sig[start_di : end_di - 1, :]), axis=1
        )
    else:
        tvr_ts = np.sum(np.abs(sig[start_di:end_di, :] - sig[start_di - 1 : end_di - 1, :]), axis=1)

    if start_di == 0:
        pnl_ts = np.full((end_di - start_di,), 0.0)
        pnl_ts[1 : end_di - start_di] = np.nansum(
            sig[start_di : end_di - 1, :] * ret[start_di + 1 : end_di, :], axis=1
        )
    else:
        pnl_ts = np.nansum(sig[start_di - 1 : end_di - 1, :] * ret[start_di:end_di, :], axis=1)

    return {"tvr": tvr_ts, "pnl": pnl_ts}


def cal_pnl_ts_equity(sig, ret, start_di, end_di):
    ts_map = cal_pnl_stats(sig, ret, start_di, end_di)

    pos_mask = ret > 0
    neg_mask = ret < 0
    long_count_ts = np.sum(pos_mask, axis=1)
    short_count_ts = np.sum(neg_mask, axis=1)

    pos_sig, neg_sig = sig.copy(), sig.copy()
    sig_pos[neg_mask] = 0
    long_val_ts = np.sum(pos_sig, axis=1)
    sig_neg[pos_mask] = 0
    short_val_ts = np.sum(neg_sig, axis=1)
    return {
        **ts_map,
        "long_count": long_count_ts,
        "short_count": short_count_ts,
        "long_val": long_val_ts,
        "short_val": short_val_ts,
    }


def cal_pnl_stats(pnls, tvrs, yearly_intervals):
    info = {}
    info["tvr"] = np.mean(tvrs)
    info["total_pnl"] = np.sum(pnls)
    mean_pnl = np.mean(pnls)
    info["yearly_pnl"] = mean_pnl * yearly_intervals
    if info["tvr"] > 0.0:
        info["pnl_bps"] = mean_pnl / info["tvr"] * 1e4
    else:
        info["pnl_bps"] = 0.0

    yearly_std = np.nanstd(pnls) * np.sqrt(yearly_intervals)
    if yearly_std > 0.0:
        info["ir"] = info["yearly_pnl"] / yearly_std
    else:
        info["ir"] = 0.0

    pnl_cum = pd.Series(np.cumsum(pnls))
    info["dd"] = np.abs((pnl_cum - pnl_cum.cummax()).min())
    if info["dd"] > 0:
        info["calmar_ratio"] = info["yearly_pnl"] / info["dd"]
    else:
        info["calmar_ratio"] = 0.0

    bets = np.sum(np.abs(pnls) > 1e-15)
    if bets > 0:
        info["win_rate"] = np.sum(pnls > 1e-15) / bets
    else:
        info["win_rate"] = 0.0

    for key, val in info.items():
        info[key] = round_float(float(val), 4)
    return info


def cal_pnl_stats_g(ts_info, fields, yearly_intervals=1):
    info = {}
    if "tvr" in fields or "pnl_bps" in fields:
        info["tvr"] = np.mean(ts_info["tvr"])

    if "total_pnl" in fields:
        info["total_pnl"] = np.sum(ts_info["pnl"])

    if "yearly_pnl" in fields:
        mean_pnl = np.mean(ts_info["pnl"])
        info["yearly_pnl"] = mean_pnl * yearly_intervals

    if "pnl_bps" in fields:
        mean_pnl = np.mean(ts_info["pnl"])
        if info["tvr"] > 0.0:
            info["pnl_bps"] = mean_pnl / info["tvr"] * 1e4
        else:
            info["pnl_bps"] = 0.0

    if "ir" in fields:
        yearly_std = np.nanstd(ts_info["pnl"]) * np.sqrt(yearly_intervals)
        if yearly_std > 0.0:
            info["ir"] = info["yearly_pnl"] / yearly_std
        else:
            info["ir"] = 0.0

    if "dd" in fields:
        pnl_cum = pd.Series(np.cumsum(ts_info["pnl"]))
        info["dd"] = np.abs((pnl_cum - pnl_cum.cummax()).min())

    if "calmar_ratio" in fields:
        if info["dd"] > 0:
            info["calmar_ratio"] = info["yearly_pnl"] / info["dd"]
        else:
            info["calmar_ratio"] = 0.0

    if "win_rate" in fields:
        bets = np.sum(np.abs(ts_info["pnl"]) > 1e-15)
        if bets > 0:
            info["win_rate"] = np.sum(ts_info["pnl"] > 1e-15) / bets
        else:
            info["win_rate"] = 0.0

        for key, val in info.items():
            info[key] = round_float(float(val), 4)
    return info


def show_pnls(pnl_stats: dict, out: TextIO = None, columns=None) -> Optional[str]:
    if columns is None:
        columns = [
            ("from", "start_date", ".0f"),
            ("to", "end_date", ".0f"),
            ("long$", "long_val", ".0f"),
            ("short$", "short_val", ".0f"),
            ("long#", "long_count", ".1f"),
            ("short#", "short_count", ".1f"),
            ("PNL", "pnl", ".0f"),
            ("ret", "ret", ".3f"),
            ("long_ret", "long_ret", ".3f"),
            ("short_ret", "short_ret", ".3f"),
            ("hedge_ret", "hedge_ret", ".3f"),
            ("TVR", "tvr", ".3f"),
            ("IR", "ir", ".3f"),
            ("STD", "std", ".4f"),
            ("IC", "ic", ".4f"),
            ("DD", "max_dd", ".2f"),
            ("dd_start", "max_dd_start", ".0f"),
            ("dd_end", "max_dd_end", ".0f"),
            ("up_days", "up_days", ".0f"),
            ("down_days", "down_days", ".0f"),
        ]
        if "trade_cost" in pnl_stats:
            columns.insert(6, ("tcost", "trade_cost", ".0f"))
    columns = [c for c in columns if c[1] in pnl_stats]
    table = pd.DataFrame([{c[1]: pnl_stats[c[1]] for c in columns}])
    formats = [c[2] for c in columns]
    headers = [c[0] for c in columns]
    if out is None and in_notebook():
        return tabulate(table, headers=headers, floatfmt=formats, showindex=False, tablefmt="html")
    else:
        print(tabulate(table, headers=headers, floatfmt=formats, showindex=False), file=out)


class Pnl2:
    def __init__(self, sim, intervals=1):
        self.sim = sim
        self.cost = {"default": 0.0}
        self.yearly_intervals = 365 * intervals

    def set_costs(self, sids, list_bps):
        self.cost = {}
        for sid, bps in zip(sids, list_bps):
            self.cost[sid] = bps

    def set_cost(self, bps):
        self.cost = {"default": bps}

    def compute_pnls(
        self, sig_, sids=[], start_date=-1, end_date=-1, freq=None, ret=None, fee_rate=None
    ):
        if isinstance(sig_, str):
            sig_ = self.sim.load_mod(sig_)
        sig = sig_.copy()
        if sids != []:
            if isinstance(sids[0], str):
                sids = [self.sim.univ.find(sid) for sid in sids]

            x = [_ for _ in list(range(sig.shape[1])) if _ not in sids]
            sig[:, x] = 0

        np.nan_to_num(sig, 0)
        if start_date == -1:
            start_di = 0
        else:
            start_di = self.sim.dates.lower_bound(start_date)
        if end_date == -1:
            end_di = self.sim.dates_size
        else:
            end_di = self.sim.dates.upper_bound(end_date)

        if ret is None:
            if not hasattr(self.sim, "b_ret"):
                self.sim.load_base("ret")
            ret = self.sim.b_ret
        else:
            if isinstance(ret, str):
                ret = self.sim.load_mod(ret)

        mp = cal_pnl_tvr(sig, ret, start_di, end_di)
        tvrs, pnls = mp["tvr"], mp["pnl"]

        # ?? to do:update list of cost
        if fee_rate is not None:
            pnls = pnls - tvrs * fee_rate
        else:
            pnls = pnls - tvrs * self.cost["default"]

        info = cal_pnl_stats(pnls, tvrs, self.yearly_intervals)
        # info = cal_pnl_stats_g(mp, ['tvr', 'total_pnl', 'yearly_pnl', 'pnl_bps', 'ir', 'dd', 'calmar_ratio', 'win_rate'],  self.yearly_intervals)
        if freq is None:
            return pnls, info, None
        else:
            infos = []
            for i in range(0, len(pnls), freq):
                if len(pnls) - i > 3:
                    infos.append(
                        cal_pnl_stats(pnls[i : i + freq], tvrs[i : i + freq], self.yearly_intervals)
                    )

            return pnls, info, infos

    def display_pnls(
        self, sig_, sids=[], start_date=-1, end_date=-1, freq=None, ret=None, fee_rate=None
    ):
        _ = self.compute_pnls(sig_, sids, start_date, end_date, freq, ret, fee_rate)
        print(_[1])
        plt.plot(np.nancumsum(_[0]))
        return self
