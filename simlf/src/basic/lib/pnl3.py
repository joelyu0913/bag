import importlib
import os
import sys
from collections import OrderedDict
from typing import Optional, TextIO, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numba import njit
from tabulate import tabulate

from data import Array
from cli.util import in_notebook
from basic.lib.pycommon.data import round_float

DiRange = tuple[int, int]


@njit()
def compute_dd(ret: np.array):
    max_dd = 0
    max_dd_start = 0
    max_dd_end = 0

    dd = 0
    dd_start = 0
    dd_end = 0
    for di in range(len(ret)):
        dd += ret[di]
        if dd < 0:
            if dd_end != di - 1:
                dd_start = di
            dd_end = di
            if dd < max_dd:
                max_dd = dd
                max_dd_start = dd_start
                max_dd_end = dd_end
        else:
            dd = 0
    return (max_dd * 100, max_dd_start, max_dd_end)


def cal_ts_info(info, ts_fields, start_di, end_di):
    def _cal_ts_pnl(sig, ret, start_di, end_di):
        if start_di == 0:
            pnl_ts = np.full((end_di - start_di,), 0.0)
            pnl_ts[1 : end_di - start_di] = np.nansum(
                sig[start_di : end_di - 1, :] * ret[start_di + 1 : end_di, :], axis=1
            )
        else:
            pnl_ts = np.nansum(sig[start_di - 1 : end_di - 1, :] * ret[start_di:end_di, :], axis=1)
        return pnl_ts

    def _cal_ts_tvr(sig, start_di, end_di):
        if start_di == 0:
            tvr_ts = np.full((end_di - start_di,), 0.0)
            tvr_ts[0] = np.sum(np.abs(sig[0]))
            tvr_ts[1 : end_di - start_di] = np.sum(
                np.abs(sig[start_di + 1 : end_di, :] - sig[start_di : end_di - 1, :]), axis=1
            )
        else:
            tvr_ts = np.sum(
                np.abs(sig[start_di:end_di, :] - sig[start_di - 1 : end_di - 1, :]), axis=1
            )
        return tvr_ts

    for field in ts_fields:
        if field == "sig":
            sig = info["sig"]
            range_sig = sig[start_di:end_di]
        elif field == "ret.mat":
            ret = info["ret.mat"]
        elif field == "dates.ts":
            dates = info["dates.ts"]

    if "long_sig.ts" in ts_fields:
        short_mask = sig < 0
        long_sig = sig.copy()
        long_sig[short_mask] = 0
        range_long_sig = long_sig[start_di:end_di]

    if "short_sig.ts" in ts_fields:
        long_mask = sig > 0
        short_sig = sig.copy()
        short_sig[long_mask] = 0
        range_short_sig = short_sig[start_di:end_di]

    if "tvr.ts" in ts_fields:
        info["tvr.ts"] = _cal_ts_tvr(sig, start_di, end_di)

    if "tcost.ts" in ts_fields:
        info["tcost.ts"] = -info["tvr.ts"] * info["fee_rate"]

    if "pnl.ts" in ts_fields:
        info["pnl.ts"] = _cal_ts_pnl(sig, ret, start_di, end_di) + info["tcost.ts"]

    if "long_pnl.ts" in ts_fields:
        info["long_pnl.ts"] = _cal_ts_pnl(long_sig, ret, start_di, end_di)

    if "short_pnl.ts" in ts_fields:
        info["short_pnl.ts"] = _cal_ts_pnl(short_sig, ret, start_di, end_di)

    if "long_count.ts" in ts_fields:
        long_mask = range_sig > 0
        info["long_count.ts"] = np.sum(long_mask, axis=1)

    if "short_count.ts" in ts_fields:
        short_mask = range_sig < 0
        info["short_count.ts"] = np.sum(short_mask, axis=1)

    if "long_val.ts" in ts_fields:
        info["long_val.ts"] = np.sum(range_long_sig, axis=1)

    if "short_val.ts" in ts_fields:
        info["short_val.ts"] = np.sum(range_short_sig, axis=1)

    if "dates.ts" in ts_fields:
        info["dates.ts"] = dates[start_di:end_di]


def cal_ts_info_3d(info, ts_fields, start_di, end_di, intervals):
    def _cal_ts_pnl(sig, ret, start_di, end_di):
        sig_i = sig[start_di:end_di].copy()
        sig_i[:, 1:, :] = sig[start_di:end_di, :-1, :]
        sig_i[1:, 0, :] = sig[start_di : end_di - 1, -1, :]
        if start_di == 0:
            sig_i[0, 0, :] = 0.0
        else:
            sig_i[0, 0, :] = sig[start_di - 1, -1, :]
        pnl_i = sig_i * ret[start_di:end_di, :, :]
        return np.sum(pnl_i, axis=(1, 2))

    def _cal_ts_tvr(sig, start_di, end_di):
        tvr_ts = np.full((end_di - start_di,), 0.0)
        tvr_i = sig[start_di:end_di].copy()
        tvr_i[:, 1:, :] = sig[start_di:end_di, 1:, :] - sig[start_di:end_di, :-1, :]
        tvr_i[1:, 0, :] = sig[start_di + 1 : end_di, 0, :] - sig[start_di : end_di - 1, -1, :]
        if start_di == 0:
            tvr_i[0, 0, :] = sig[0, 0, :]
        else:
            tvr_i[0, 0, :] = sig[start_di, 0, :] - sig[start_di - 1, -1, :]

        tvr_ts = np.sum(np.abs(tvr_i), axis=(1, 2))
        return tvr_ts

    for field in ts_fields:
        if field == "sig":
            sig = info["sig"]
            range_sig = sig[start_di:end_di]
        elif field == "ret.mat":
            ret = info["ret.mat"]
        elif field == "dates.ts":
            dates = info["dates.ts"]

    if "long_sig.ts" in ts_fields:
        short_mask = sig < 0
        long_sig = sig.copy()
        long_sig[short_mask] = 0
        range_long_sig = long_sig[start_di:end_di]

    if "short_sig.ts" in ts_fields:
        long_mask = sig > 0
        short_sig = sig.copy()
        short_sig[long_mask] = 0
        range_short_sig = short_sig[start_di:end_di]

    if "tvr.ts" in ts_fields:
        info["tvr.ts"] = _cal_ts_tvr(sig, start_di, end_di)

    if "tcost.ts" in ts_fields:
        info["tcost.ts"] = -info["tvr.ts"] * info["fee_rate"]

    if "pnl.ts" in ts_fields:
        info["pnl.ts"] = _cal_ts_pnl(sig, ret, start_di, end_di) + info["tcost.ts"]

    if "long_pnl.ts" in ts_fields:
        info["long_pnl.ts"] = _cal_ts_pnl(long_sig, ret, start_di, end_di)

    if "short_pnl.ts" in ts_fields:
        info["short_pnl.ts"] = _cal_ts_pnl(short_sig, ret, start_di, end_di)

    if "long_count.ts" in ts_fields:
        long_mask = range_sig > 0
        info["long_count.ts"] = np.sum(long_mask, axis=(1, 2)) / intervals

    if "short_count.ts" in ts_fields:
        short_mask = range_sig < 0
        info["short_count.ts"] = np.sum(short_mask, axis=(1, 2)) / intervals

    if "long_val.ts" in ts_fields:
        info["long_val.ts"] = np.sum(range_long_sig, axis=(1, 2)) / intervals

    if "short_val.ts" in ts_fields:
        info["short_val.ts"] = np.sum(range_short_sig, axis=(1, 2)) / intervals

    if "dates.ts" in ts_fields:
        info["dates.ts"] = dates[start_di:end_di]


def cal_stats_info(info, stats_fields):
    if "tvr" in stats_fields:
        info["tvr"] = np.mean(info["tvr.ts"]) / info["booksize"]

    if "tcost" in stats_fields:
        info["tcost"] = np.sum(info["tcost.ts"])

    if "pnl" in stats_fields:
        info["pnl"] = np.sum(info["pnl.ts"])

    if "pnl_y" in stats_fields:
        mean_pnl = np.mean(info["pnl.ts"])
        info["pnl_y"] = mean_pnl * info["intervals_y"]

    if "pnl_bps" in stats_fields:
        mean_pnl = np.mean(info["pnl.ts"])
        if info["tvr"] > 0.0:
            info["pnl_bps"] = mean_pnl / info["tvr"] * 1e4
        else:
            info["pnl_bps"] = 0.0

    if "ir" in stats_fields:
        std_y = np.nanstd(info["pnl.ts"]) * np.sqrt(info["intervals_y"])
        if std_y > 0.0:
            info["ir"] = info["pnl_y"] / std_y
        else:
            info["ir"] = 0.0

    if "max_dd" in stats_fields:
        pnl_cum = pd.Series(np.cumsum(info["pnl.ts"]))
        max_dd, start_idx, end_idx = compute_dd(info["pnl.ts"])
        info["max_dd_start"] = info["dates.ts"][start_idx]
        info["max_dd_end"] = info["dates.ts"][end_idx]
        info["max_dd"] = np.abs(max_dd / info["booksize"])

    if "calmar_ratio" in stats_fields:
        if info["max_dd"] > 0:
            info["calmar_ratio"] = info["pnl_y"] / info["max_dd"]
        else:
            info["calmar_ratio"] = 0.0

    if "win_rate" in stats_fields:
        bets = np.sum(np.abs(info["pnl.ts"]) > 1e-15)
        if bets > 0:
            info["win_rate"] = np.sum(info["pnl.ts"] > 1e-15) / bets
        else:
            info["win_rate"] = 0.0

    if "start_date" in stats_fields:
        info["start_date"] = info["dates.ts"][0]

    if "end_date" in stats_fields:
        info["end_date"] = info["dates.ts"][-1]

    for field in ["long_val", "short_val", "long_count", "short_count"]:
        if field in stats_fields:
            info[field] = np.mean(info[field + ".ts"])

    if "ret" in stats_fields:
        info["ret"] = np.nansum(info["pnl.ts"] / info["booksize"])

    if "ret_y" in stats_fields:
        info["ret_y"] = np.nanmean(info["pnl.ts"] / info["booksize"]) * info["intervals_y"]

    if "long_ret_y" in stats_fields:
        info["long_ret_y"] = (
            np.nanmean(info["long_pnl.ts"] / info["booksize"]) * info["intervals_y"]
        )

    if "short_ret_y" in stats_fields:
        info["short_ret_y"] = (
            np.nanmean(info["short_pnl.ts"] / info["booksize"]) * info["intervals_y"]
        )


def show_pnls(pnl_stats: dict, out: TextIO = None, columns="cn") -> Optional[str]:
    if columns == "cn":
        columns = [
            ("from", "start_date", ".0f"),
            ("to", "end_date", ".0f"),
            ("long$", "long_val", ".0f"),
            ("short$", "short_val", ".0f"),
            ("long#", "long_count", ".1f"),
            ("short#", "short_count", ".1f"),
            ("tcost", "tcost", ".0f"),
            ("PNL", "pnl", ".0f"),
            ("ret", "ret_y", ".3f"),
            ("long_ret", "long_ret_y", ".3f"),
            ("short_ret", "short_ret_y", ".3f"),
            ("hedge_ret", "hedge_ret", ".3f"),
            ("TVR", "tvr", ".3f"),
            ("IR", "ir", ".3f"),
            ("STD", "std", ".4f"),
            ("IC", "ic", ".4f"),
            ("DD", "max_dd", ".2f"),
            ("dd_start", "max_dd_start", ".0f"),
            ("dd_end", "max_dd_end", ".0f"),
            ("win_rate", "win_rate", ".2f"),
        ]
    elif columns == "crypto":
        columns = [
            ("from", "start_date", ".0f"),
            ("to", "end_date", ".0f"),
            ("long$", "long_val", ".3f"),
            ("short$", "short_val", ".3f"),
            ("long#", "long_count", ".1f"),
            ("short#", "short_count", ".1f"),
            ("ret", "ret", ".3f"),
            ("ret_y", "ret_y", ".3f"),
            ("pnl_bps", "pnl_bps", ".3f"),
            ("TVR", "tvr", ".3f"),
            ("IR", "ir", ".3f"),
            ("CalmarR", "calmar_ratio", ".3f"),
            ("DD", "max_dd", ".2f"),
            ("dd_start", "max_dd_start", ".0f"),
            ("dd_end", "max_dd_end", ".0f"),
            ("win_rate", "win_rate", ".2f"),
        ]

    columns = [c for c in columns if c[1] in pnl_stats]
    table = pd.DataFrame({c[1]: pnl_stats[c[1]] for c in columns})
    formats = [c[2] for c in columns]
    headers = [c[0] for c in columns]
    if out is None and in_notebook():
        return tabulate(table, headers=headers, floatfmt=formats, showindex=False, tablefmt="html")
    else:
        print(tabulate(table, headers=headers, floatfmt=formats, showindex=False), file=out)


cn_stats_fields = [
    "start_date",
    "end_date",
    "long_val",
    "short_val",
    "long_count",
    "short_count",
    "tcost",
    "pnl",
    "ret_y",
    "long_ret_y",
    "short_ret_y",
    # "hedge_ret",
    "tvr",
    "ir",
    # "std",
    # "ic",
    "max_dd",
    "max_dd_start",
    "max_dd_end",
    "win_rate",
]

crypto_stats_fields = [
    "start_date",
    "end_date",
    "long_val",
    "short_val",
    "long_count",
    "short_count",
    "tvr",
    "pnl",
    "ret",
    "ret_y",
    "pnl_bps",
    "ir",
    "max_dd",
    "calmar_ratio",
    "win_rate",
]


def tree(node, ret_ts, grf):
    if node not in ret_ts:
        if node in grf:
            for sub_node in grf[node]:
                tree(sub_node, ret_ts, grf)

        ret_ts.append(node)


class DepTree:
    def __init__(self):
        input_dep = {
            "tvr.ts": ["sig"],
            "pnl.ts": ["sig", "ret.mat"],
            "long_pnl.ts": ["ret.mat"],
            "short_pnl.ts": ["ret.mat"],
            "long_count.ts": ["sig"],
            "short_count.ts": ["sig"],
            "long_sig.ts": ["sig"],
            "short_sig.ts": ["sig"],
            "long_ret_y": ["booksize", "intervals_y"],
            "short_ret_y": ["booksize", "intervals_y"],
            "ret.mat": ["booksize"],
            "ret": ["booksize"],
            "ret_y": ["booksize", "intervals_y"],
            "pnl_y": ["intervals_y"],
            "max_dd": ["booksize"],
            "tcost.ts": ["fee_rate"],
        }

        ts_dep = {
            "tvr": ["tvr.ts"],
            "pnl.ts": ["tcost.ts"],
            "pnl": ["pnl.ts"],
            "pnl_y": ["pnl.ts"],
            "pnl_bps": ["pnl.ts", "tvr.ts"],
            "ir": ["pnl.ts"],
            "max_dd": ["pnl.ts"],
            "win_rate": ["pnl.ts"],
            "start_date": ["dates.ts"],
            "end_date": ["dates.ts"],
            "long_val": ["long_val.ts"],
            "short_val": ["short_val.ts"],
            "long_val.ts": ["long_sig.ts"],
            "short_val.ts": ["short_sig.ts"],
            "long_pnl.ts": ["long_sig.ts"],
            "short_pnl.ts": ["short_sig.ts"],
            "long_count": ["long_count.ts"],
            "short_count": ["short_count.ts"],
            "long_ret_y": ["long_pnl.ts"],
            "short_ret_y": ["short_pnl.ts"],
            "ret_y": ["pnl.ts"],
            "tcost.ts": ["tvr.ts"],
            "tcost": ["tcost.ts"],
        }

        stats_dep = {
            "pnl_bps": ["tvr"],
            "ir": ["pnl_y"],
            "calmar_raio": ["max_dd"],
            "max_dd_start": ["max_dd"],
            "max_dd_end": ["max_dd"],
        }

        self.deps = input_dep
        for key in ts_dep:
            if key in self.deps:
                self.deps[key] += ts_dep[key]
            else:
                self.deps[key] = ts_dep[key]
        for key in stats_dep:
            if key in self.deps:
                self.deps[key] += stats_dep[key]
            else:
                self.deps[key] = stats_dep[key]


class Pnl3:
    def __init__(self, sim):
        self.sim = sim
        self.deps = DepTree().deps
        self.region = ""

    def set_fields(self, **kwargs):

        if "region" in kwargs:
            self.region = kwargs["region"]
            if self.region == "cn":
                self.stats_fields = cn_stats_fields
                self.default_info = {
                    "booksize": kwargs.get("booksize", 1e5),
                    # "fee_rate": kwargs.get("fee_rate", 6.5e-4),  # 'buy_fee':1e-4, 'sell_fee': 12e-4
                    "fee_rate": kwargs.get("fee_rate", {20010101: 0.00065, 20240101: 0.0004}),
                    "intervals_y": kwargs.get("intervals_y", 250),
                }
            elif self.region == "crypto":
                self.stats_fields = crypto_stats_fields
                self.default_info = {
                    "booksize": kwargs.get("booksize", 1),
                    "fee_rate": kwargs.get("fee_rate", 2e-4),
                    "intervals_y": kwargs.get("intervals_y", 365 * 24),
                }
        else:
            self.stats_fields = kwargs["fields"]

            self.default_info = {
                "booksize": kwargs.get("booksize", 1),
                "fee_rate": kwargs.get("fee_rate", 0),
                "intervals_y": kwargs["intervals_y"],
                "dates.ts": kwargs.get("dates.ts", self.sim.dates),
            }

        self.default_info["dates.ts"] = kwargs.get("dates.ts", self.sim.dates)

        if "ret" not in kwargs:
            if not hasattr(self.sim, "b_ret"):
                self.sim.load_base("ret")
            self.default_info["ret.mat"] = self.sim.b_ret
        else:
            ret = kwargs["ret"]
            if isinstance(ret, str):
                self.default_info["ret.mat"] = self.sim.load_mod(ret)
            else:
                self.default_info["ret.mat"] = ret

        self.fields = []
        for field in self.stats_fields:
            tree(field, self.fields, self.deps)

    def compute_pnls(self, sig_, ops=None, start_date=-1, end_date=-1, freq=None, update_info={}):

        if isinstance(sig_, str):
            sig_ = self.sim.load_mod(sig_).astype(np.float32)
        if ops is None:
            sig = sig_.copy()
        else:
            sig = self.sim.apply_ops(sig_, ops)

        np.nan_to_num(sig, 0)
        if start_date == -1:
            start_di = 0
        else:
            start_di = self.sim.dates.lower_bound(start_date)
        if end_date == -1:
            end_di = self.sim.dates_size
        else:
            end_di = self.sim.dates.upper_bound(end_date)

        self.default_info["sig"] = sig

        info = {
            field: self.default_info[field] for field in self.fields if field in self.default_info
        }
        info = {**info, **update_info}
        if "fee_rate" in info and isinstance(info["fee_rate"], dict):
                info["fee_rate"] = self.fee_config(info["fee_rate"], start_date, end_date)

        if sig.ndim == 3:
            cal_ts_info_3d(info, self.fields, start_di, end_di, sig.shape[1])
        else:
            cal_ts_info(info, self.fields, start_di, end_di)

        tvrs, pnls = info["tvr.ts"], info["pnl.ts"]

        self.whole_info = info.copy()
        cal_stats_info(self.whole_info, self.fields)

        if freq is None:
            return pnls, {field: self.whole_info[field] for field in self.stats_fields}, None
        elif freq == "yearly":

            def generate_yearly_ranges(dates: np.array, include_total: bool) -> list[DiRange]:

                year_multiplier = 10000 if len(str(dates[0])) == 8 else 100000000
                first_year = dates[0] // year_multiplier
                last_year = dates[-1] // year_multiplier
                ranges = []
                for year in range(first_year, last_year + 1):
                    rg = dates.searchsorted([year * year_multiplier, (year + 1) * year_multiplier])
                    ranges.append(rg)
                if include_total:
                    ranges.append((0, -1))
                return ranges

            infos = []
            ranges = generate_yearly_ranges(np.array(info["dates.ts"]), False)
            for start_, end_ in ranges:
                info_ = dict(
                    [
                        (key, val[start_:end_]) if ".ts" in key else (key, val)
                        for key, val in info.items()
                    ]
                )
                cal_stats_info(
                    info_,
                    self.fields,
                )
                infos.append({field: info_[field] for field in self.stats_fields})

        else:
            infos = []
            for i in range(0, len(pnls), freq):
                if len(pnls) - i > 3:
                    info_ = dict(
                        [
                            (key, val[i : i + freq]) if ".ts" in key else (key, val)
                            for key, val in info.items()
                        ]
                    )
                    cal_stats_info(info_, self.fields)
                    infos.append({field: info_[field] for field in self.stats_fields})

        return pnls, {field: self.whole_info[field] for field in self.stats_fields}, infos

    def show_pnls(
        self,
        sig_,
        start_date=-1,
        end_date=-1,
        freq="yearly",
        update_info={},
        plot=False,
        plot_type="ret",
        ops=None,
    ):
        _ = self.compute_pnls(sig_, ops, start_date, end_date, freq, update_info)
        mp_info = {key: [] for key in _[1]}

        for key in _[1]:
            for x in _[2]:
                mp_info[key].append(x[key])
            mp_info[key].append(_[1][key])

        if plot:
            if plot_type == "ret":
                plt.plot(np.nancumsum(_[0]) / self.whole_info["booksize"])
            else:
                plt.plot(np.nancumsum(_[0]))

        if self.region:
            return show_pnls(mp_info, columns=self.region)
        else:
            return show_pnls(mp_info, columns=self.stats_fields)

    def plot_pnls(
        self,
        prs,
        ops=None,
        start_date=-1,
        end_date=-1,
        freq=None,
        update_info={},
        color={},
        x_axis=None,
    ):
        if not isinstance(prs, list):
            prs = [prs]

        for pr in prs:
            if isinstance(pr, tuple):
                alpha_name = pr[0]
                wt = pr[1]
            else:
                alpha_name = pr
                wt = 1.0

            if isinstance(alpha_name, str):
                sig_ = self.sim.load_mod(alpha_name) * wt
            else:
                sig_ = alpha_name * wt

            _ = self.compute_pnls(sig_, ops, start_date, end_date, freq, update_info)
            label_, color_ = None, None
            if isinstance(alpha_name, str):
                label_ = alpha_name
                if alpha_name in color:
                    color_ = color[alpha_name]

            plt.plot(np.nancumsum(_[0]), label=label_, color=color_)

        plt.legend(loc="upper left", fontsize="small")

        if x_axis is not None:
            if start_date == -1:
                start_di = 0
            else:
                start_di = self.sim.env.dates.lower_bound(start_date)
            if end_date == -1:
                end_di = self.sim.env.dates_size
            else:
                end_di = self.sim.env.dates.upper_bound(end_date)

            input_dates = self.sim.env.dates[start_di:end_di]

            def get_pre(input_dates, trailing=100):
                odates = [input_dates[0] // trailing]

                for date in input_dates[1:]:
                    yyyymm = date // trailing
                    if yyyymm != odates[-1]:
                        odates.append(yyyymm)
                dis = [
                    self.sim.env.dates.lower_bound(date * trailing) - start_di for date in odates
                ]
                return odates, dis

            if x_axis == "monthly":
                if len(str(self.sim.env.dates[0])) == 12:
                    dates, dis = get_pre(input_dates, 1000000)
                    print(dates)
                else:
                    dates, dis = get_pre(input_dates, 100)
            elif x_axis == "yearly":
                if len(str(self.sim.env.dates[0])) == 12:
                    dates, dis = get_pre(input_dates, 100000000)
                else:
                    dates, dis = get_pre(input_dates, 10000)

            plt.xticks(dis, dates, rotation=30)

    def fee_config(self, mp, start_date=-1, end_date=-1):
        if start_date == -1:
            start_di = 0
        else:
            start_di = self.sim.dates.lower_bound(start_date)
        if end_date == -1:
            end_di = self.sim.dates_size
        else:
            end_di = self.sim.dates.upper_bound(end_date)
        ret_fees = np.zeros(end_di - start_di)
        for date, val in mp.items():
            di = self.sim.dates.lower_bound(date) - start_di
            if di < 0:
                di = 0
            ret_fees[di:] = val
        return ret_fees
