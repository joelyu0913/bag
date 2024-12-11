import importlib
import logging
import os
import sys
from collections import OrderedDict
from typing import Optional, TextIO, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate

from data import Array
from sim import DISPLAY_BOOK_SIZE, Env
from cli.util import in_notebook
from basic.operation_manager import OperationManager

"""
pnl = PnlStats(["base", "hedge", "tcost", "ic", "ir"], {"buy_fee": 1e-4, "sell_fee": 11e-4})
start_di = env.dates.lower_bound(20170101)
daily_stats = pnl.compute(env, "P__kirin2_rd/b_sig_op", start_di=start_di)
yearly_stats = pnl.summarize_yearly(env, daily_stats)
monthly_stats = pnl.summarize_monthly(env, daily_stats)
"""
DEFAULT_METRICS = ["base", "hedge", "ic", "ir"]

Signal = Union[np.array, str, Array]
DiRange = tuple[int, int]


def load_signal(sig: Signal, env: Env) -> np.array:
    if isinstance(sig, str):
        return Array.mmap(sig).data
    elif isinstance(sig, Array):
        return sig.data
    return sig


def generate_yearly_ranges(dates: np.array, daily: bool, include_total: bool) -> list[DiRange]:
    year_multiplier = 10000 if daily else 100000000
    first_year = dates[0] // year_multiplier
    last_year = dates[-1] // year_multiplier
    ranges = []
    for year in range(first_year, last_year + 1):
        rg = dates.searchsorted([year * year_multiplier, (year + 1) * year_multiplier])
        ranges.append(rg)
    if include_total:
        ranges.append((0, -1))
    return ranges


def generate_monthly_ranges(dates: np.array, daily: bool, include_total: bool) -> list[DiRange]:
    month_multiplier = 100 if daily else 1000000
    first_month = dates[0] // month_multiplier
    last_month = dates[-1] // month_multiplier
    ranges = []
    cur_month = first_month
    while cur_month <= last_month:
        next_month = cur_month + 1
        if next_month % 100 == 13:
            next_month = (next_month // 100 + 1) * 100 + 1
        rg = dates.searchsorted([cur_month * month_multiplier, next_month * month_multiplier])
        ranges.append(rg)
        cur_month = next_month
    if include_total:
        ranges.append((0, -1))
    return ranges


def generate_ranges(
    period: str, dates: np.array, daily: bool, include_total: bool
) -> list[DiRange]:
    if period == "monthly":
        return generate_monthly_ranges(dates, daily, include_total)
    if period == "yearly":
        return generate_yearly_ranges(dates, daily, include_total)
    raise RuntimeError(f"unknown period {period}")


def show_pnls(pnl_stats: pd.DataFrame, out: TextIO = None, columns=None) -> Optional[str]:
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
        if "trade_cost" in pnl_stats.columns:
            columns.insert(6, ("tcost", "trade_cost", ".0f"))
    columns = [c for c in columns if c[1] in pnl_stats.columns]
    table = pnl_stats[[c[1] for c in columns]]
    formats = [c[2] for c in columns]
    headers = [c[0] for c in columns]
    if out is None and in_notebook():
        return tabulate(table, headers=headers, floatfmt=formats, showindex=False, tablefmt="html")
    else:
        print(tabulate(table, headers=headers, floatfmt=formats, showindex=False), file=out)


class pnl_dict:
    def __init__(self, df, dict_global):
        self.data = {}
        self.yys = (df["start_date"] // 10000).to_list()
        self.fields = df.columns.to_list()
        for yy in self.yys:
            self.data[yy] = (
                df[np.logical_and(df["start_date"] // 10000 == yy, df["end_date"] // 10000 == yy)]
                .squeeze()
                .to_dict()
            )
        df_tmp = df[df["start_date"] // 10000 != df["end_date"] // 10000]
        if len(df_tmp) > 0:
            self.data["all"] = df_tmp.squeeze().to_dict()
        else:
            self.data["all"] = list(self.data.items())[0][1]
        self.yys.append("all")

        self.dict = {}

        for field in self.fields:
            for yy in self.yys:
                self.dict[f"{field}_{yy}"] = self.data[yy][field]

        self.dict_global = dict_global

    def eval(self, filter):
        return eval(filter, self.dict_global, self.dict)


class PnlStats(object):
    def __init__(self, metrics: list[str] = DEFAULT_METRICS, config: dict = {}):
        self.metrics = metrics
        self.metric_mods = [importlib.import_module(f"basic.lib.pnl_metrics.{m}") for m in metrics]
        self.metric_funcs = {}
        for func_name in ["compute", "summarize"]:
            funcs = []
            for mod in self.metric_mods:
                if not hasattr(mod, func_name):
                    continue
                funcs.append(getattr(mod, func_name))
            self.metric_funcs[func_name] = funcs

        default_config = {
            "book_size": DISPLAY_BOOK_SIZE,
            "ret": "base/ret",
            "ops": "",
        }
        self.config = {**default_config, **config}

    def compute(
        self,
        env: Env,
        sig: Signal,
        start_di: Optional[int] = None,
        end_di: Optional[int] = None,
        **kwargs,
    ) -> pd.DataFrame:
        config = {**self.config, **kwargs}

        if start_di is None:
            start_di = env.start_di
        if end_di is None:
            end_di = env.end_di
        ret = env.read_data(Array, *config["ret"].split("/")).data

        sig = load_signal(sig, env)
        assert sig.shape[0] >= end_di and sig.shape[1] == env.max_univ_size
        ops = config["ops"]
        if ops != "":
            sig_op = np.full_like(sig, np.nan)
            op_manager = OperationManager(env)
            op_manager.apply(sig, sig_op, start_di, end_di, ops)
            sig = sig_op
        sig = sig[start_di:end_di]
        ret = ret[start_di:end_di]
        metrics = pd.DataFrame(
            {
                "di": list(range(start_di, end_di)),
                "date": env.dates[start_di:end_di],
            }
        )

        config["ret"] = ret
        config["start_di"] = start_di
        config["end_di"] = end_di
        if env.short_book_size:
            config["book_size"] *= 2
        for mod in self.metric_mods:
            if not hasattr(mod, "compute"):
                continue
            compute_func = getattr(mod, "compute")
            compute_func(
                env=env,
                sig=sig,
                metrics=metrics,
                config=config,
            )
        return metrics

    def summarize_yearly(
        self, env: Env, metrics: pd.DataFrame, include_total=True, **kwargs
    ) -> pd.DataFrame:
        if len(metrics) == 0:
            return None
        sum_ranges = generate_yearly_ranges(metrics.date.array, env.daily, include_total)
        return self.summarize(env, metrics, sum_ranges, **kwargs)

    def summarize_monthly(
        self, env: Env, metrics: pd.DataFrame, include_total=True, **kwargs
    ) -> pd.DataFrame:
        if len(metrics) == 0:
            return None
        sum_ranges = generate_monthly_ranges(metrics.date.array, env.daily, include_total)
        return self.summarize(env, metrics, sum_ranges, **kwargs)

    def summarize(
        self, env: Env, metrics: pd.DataFrame, ranges: list[DiRange], **kwargs
    ) -> pd.DataFrame:
    
        if len(ranges) == 0:
            return None

        funcs = self.metric_funcs["summarize"]
        config = {**self.config, **kwargs}
        config["days_per_year"] = env.days_per_year
        config["intervals_per_day"] = env.intervals_per_day
        if env.short_book_size:
            config["book_size"] *= 2
        sum_metrics = []
        for begin, end in ranges:
            range_metrics = metrics.iloc[begin:end] if end != -1 else metrics.iloc[begin:]
            range_sum = OrderedDict(
                [
                    ("start_date", range_metrics.date.iloc[0]),
                    ("end_date", range_metrics.date.iloc[-1]),
                ]
            )
            for f in funcs:
                f(
                    metrics=range_metrics,
                    sum_metrics=range_sum,
                    config=config,
                )
            sum_metrics.append(range_sum)
        cols = list(sum_metrics[0].keys())
        data = {c: [m[c] for m in sum_metrics] for c in cols}
        return pd.DataFrame(data)

    def show(self, pnl_stats: pd.DataFrame, out: TextIO = None, columns=None) -> Optional[str]:
        return show_pnls(pnl_stats, out=out, columns=columns)

    def compute_cos(
        self,
        env: Env,
        sig: Signal,
        start_di: Optional[int] = None,
        end_di: Optional[int] = None,
        period: str = "monthly",
        include_total: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        config = {**self.config, **kwargs}

        if start_di is None:
            start_di = env.start_di
        if end_di is None:
            end_di = env.end_di
        ret = env.read_data(Array, *config["ret"].split("/")).data
        sig = load_signal(sig, env)
        assert sig.shape[0] >= end_di and sig.shape[1] == env.max_univ_size
        ops = config["ops"]
        if ops != "":
            sig_op = np.full_like(sig, np.nan)
            op_manager = OperationManager(env)
            op_manager.apply(sig, sig_op, start_di, end_di, ops)
            sig = sig_op
        if end_di == env.end_di:
            end_di -= 1
        sig = sig[start_di:end_di]
        ret = ret[start_di + 1 : end_di + 1]
        dates = env.dates[start_di:end_di]
        ranges = generate_ranges(period, dates, env.daily, include_total)
        if len(ranges) == 0:
            return None
        start_dates = []
        end_dates = []
        cos = []
        from .pnl_metrics.ic import compute_cos

        for begin, end in ranges:
            start_dates.append(dates[begin])
            end_dates.append(dates[end - 1])
            cos.append(compute_cos(sig[begin:end].reshape((-1,)), ret[begin:end].reshape((-1,))))
        return pd.DataFrame({"start_date": start_dates, "end_date": end_dates, "cos": cos})


class Pnl:
    def __init__(self, sim, fee_rate, fields):
        self.sim = sim
        self.fee_rate = fee_rate
        self.register_pnl_fields = fields

    def _compute_pnl_stats(self, mod_path, register_pnl_fields, **kwargs):
        if isinstance(mod_path, str):
            mod_path = self.sim._full_path(mod_path)
        self._update_pnl_pars(kwargs)
        self.pnl_ = PnlStats(self.register_pnl_fields, self.fee_rate)
        self.daily_stats = self.pnl_.compute(self.sim, mod_path, **kwargs)
        self.yearly_stats = self.pnl_.summarize_yearly(self.sim, self.daily_stats)

    def compute_pnls(self, mod_path, **kwargs):
        import warnings

        warnings.filterwarnings("ignore")
        register_pnl_fields = self.register_pnl_fields
        if "tpnl" in kwargs and kwargs["tpnl"] and "tcost" not in register_pnl_fields:
            register_pnl_fields.append("tcost")

        self._compute_pnl_stats(mod_path, register_pnl_fields, **kwargs)

        if "pnl_dict" in kwargs:

            pnl_fields = ["ret", "tvr", "ic", "ir"]
            if "pnl_fields" in kwargs:
                pnl_fields = kwargs["pnl_fields"]
            self.pnl_dict = {}
            targ = self.yearly_stats

            for i in range(targ.shape[0]):
                row = targ.iloc[i]

                if i < targ.shape[0] - 1:
                    time = int(row["start_date"])
                    if len(str(time)) == 8:
                        year = int(time) // 10000
                    elif len(str(time)) == 12:
                        year = int(time) // 100000000
                    else:
                        logging.error(f"time is not valid {time}")
                else:
                    year = "all"

                for field in pnl_fields:
                    self.pnl_dict[f"{field}_{year}"] = row[field]
        return self

    def _update_pnl_pars(self, kwargs):
        if "start_date" in kwargs and "start_di" not in kwargs:
            kwargs["start_di"] = self.sim.dates.lower_bound(kwargs["start_date"])
        if "end_date" in kwargs and "end_di" not in kwargs:
            kwargs["end_di"] = self.sim.dates.upper_bound(kwargs["end_date"])
        return kwargs

    def compute_cos(self, mod_path, **kwargs):
        pnlstats = PnlStats(self.register_pnl_fields, self.fee_rate)
        if isinstance(mod_path, str):
            mod_path = self.sim._full_path(mod_path)
            sig = self.load_mod(mod_path)
        else:
            sig = mod_path
        self._update_pnl_pars(kwargs)
        return pnlstats.compute_cos(self.sim, sig, **kwargs)["cos"].to_list()[:-1]

    def show_pnls(self):
        return show_pnls(self.yearly_stats)

    def plot_pnls(self, mod_path, **kwargs):
        self._update_pnl_pars(kwargs)
        self.compute_pnls(mod_path, aggregate=False, **kwargs)

        x = [str(_) for _ in self.sim.dates.items]
        y = np.cumsum(self.daily_stats["ret"])

        if "start_di" in kwargs:
            start_di = kwargs["start_di"]
        else:
            start_di = self.sim.start_di
        if "end_di" in kwargs:
            end_di = kwargs["end_di"]
        else:
            end_di = self.sim.end_di

        x_dates = x[start_di:end_di]
        pnl = y[0 : end_di - start_di]
        # print(pnl.shape, len(pnl), end_di, start_di)

        if "hold" not in kwargs or kwargs["hold"] == False:
            plt.rc("font", size=25)
            fig = plt.figure(figsize=(24, 16))

            x_stride = len(x_dates) // 30
            plt.xticks(ticks=range(0, len(x_dates), x_stride), rotation=90)
            self.ax = plt.subplot(111)
            self.ax.spines.right.set_visible(False)
            self.ax.spines.top.set_visible(False)

            if "y_stride" in kwargs:
                y_stride = kwargs["y_stride"]
            else:
                y_stride = 0.1

            range_in = range(-1, int(pnl[end_di - start_di - 1] / y_stride) + 1)
            for t in np.array(range_in) * y_stride:
                self.ax.axhline(y=t, color="black", linewidth=0.3)
        if "color" in kwargs:
            color = kwargs["color"]
        else:
            color = "red"
        self.ax.plot(x_dates, pnl, color=color, linewidth=2)
        return self

    def cal_implied_vol(self, sig, path_cov_mat, start_date=-1, end_date=-1, flag_scale=False):
        ll = []
        if isinstance(sig, str):
            sig = self.sim.load_mod(sig)

        if start_date == -1:
            start_di = 0
        else:
            start_di = self.sim.dates.less_equal_than(start_date)

        if end_date == -1:
            end_di = self.sim.dates_size
        else:
            end_di = self.sim.dates.less_equal_than(end_date)

        self.sim.register_cov_mat(path_cov_mat)

        for di in range(start_di, end_di):

            idx, mat = self.sim.cov_mat.load(di, "cov")
            alpha = sig[di, idx]
            alpha[~np.isfinite(alpha)] = 0
            if flag_scale:
                alpha = alpha / np.sum(np.abs(alpha))
            ll.append(alpha @ mat @ alpha.T)
        return ll


def compute_pnls(
    env: Env,
    sig: Signal,
    start_di: Optional[int] = None,
    end_di: Optional[int] = None,
    aggregate: bool = True,
    tpnl: bool = False,
    buy_fee: float = 1e-4,
    sell_fee: float = 11e-4,
    **kwargs,
) -> pd.DataFrame:
    pnl_metrics = ["base"]
    if env.benchmark_index != "":
        pnl_metrics.append("hedge")
    if tpnl:
        pnl_metrics.append("tcost")
    pnl_metrics += ["ic", "ir"]
    pnl = PnlStats(pnl_metrics, {"buy_fee": buy_fee, "sell_fee": sell_fee})
    daily_stats = pnl.compute(
        env=env,
        sig=sig,
        start_di=start_di,
        end_di=end_di,
        **kwargs,
    )
    if aggregate:
        return pnl.summarize_yearly(env, daily_stats, **kwargs)
    else:
        return daily_stats
