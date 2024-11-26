from typing import Any

import numpy as np
import numpy.ma as ma
import pandas as pd
from numba import njit

from sim import Env


def compute(
    env: Env,
    sig: np.array,
    metrics: pd.DataFrame,
    config: dict[str, Any],
):
    booksize = config["book_size"]
    ret = config["ret"]

    sig = np.copy(sig)
    sig[~np.isfinite(sig)] = 0
    prev_sig = np.zeros(sig.shape, dtype=sig.dtype)
    prev_sig[1:] = sig[:-1]

    long_idx = sig > 0
    short_idx = sig < 0
    long_count = np.sum(long_idx, axis=1)
    short_count = np.sum(short_idx, axis=1)

    long_val = ma.getdata(np.sum(ma.MaskedArray(sig, mask=~long_idx), axis=1))
    short_val = ma.getdata(np.sum(ma.MaskedArray(sig, mask=~short_idx), axis=1))
    trade_val = np.sum(np.abs(sig - prev_sig), axis=1)

    raw_pnl = prev_sig * ret
    raw_pnl[~np.isfinite(raw_pnl)] = 0
    long_pnl = ma.getdata(np.sum(ma.MaskedArray(raw_pnl, prev_sig <= 0), axis=1))
    short_pnl = ma.getdata(np.sum(ma.MaskedArray(raw_pnl, prev_sig >= 0), axis=1))

    pnl = long_pnl + short_pnl

    metrics["long_count"] = long_count
    metrics["short_count"] = short_count
    metrics["long_val"] = long_val
    metrics["short_val"] = short_val
    metrics["trade_val"] = trade_val
    metrics["long_pnl"] = long_pnl
    metrics["short_pnl"] = short_pnl
    metrics["pnl"] = pnl
    metrics["long_ret"] = long_pnl / booksize
    metrics["short_ret"] = short_pnl / booksize
    metrics["ret"] = pnl / booksize


def summarize(metrics: pd.DataFrame, sum_metrics: dict[str, float], config: dict[str, Any]):
    sum_metrics["long_count"] = metrics.long_count.mean()
    sum_metrics["short_count"] = metrics.short_count.mean()
    sum_metrics["long_val"] = metrics.long_val.mean()
    sum_metrics["short_val"] = metrics.short_val.mean()
    sum_metrics["trade_val"] = metrics.trade_val.mean()
    sum_metrics["long_pnl"] = metrics.long_pnl.sum()
    sum_metrics["short_pnl"] = metrics.short_pnl.sum()
    sum_metrics["pnl"] = metrics.pnl.sum()
    days_per_year = config["days_per_year"]
    intervals_per_day = config["intervals_per_day"]
    sum_metrics["long_ret"] = metrics.long_ret.mean() * days_per_year * intervals_per_day
    sum_metrics["short_ret"] = metrics.short_ret.mean() * days_per_year * intervals_per_day
    sum_metrics["ret"] = metrics.ret.mean() * days_per_year * intervals_per_day

    sum_metrics["tvr"] = sum_metrics["trade_val"] / config["book_size"]

    sum_metrics["up_days"] = np.sum(metrics.pnl > 0)
    sum_metrics["down_days"] = np.sum(metrics.pnl < 0)

    max_dd, max_dd_start, max_dd_end = compute_dd(metrics.ret.to_numpy())
    sum_metrics["max_dd"] = -max_dd
    sum_metrics["max_dd_start"] = metrics.date.iloc[max_dd_start]
    sum_metrics["max_dd_end"] = metrics.date.iloc[max_dd_end]


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
