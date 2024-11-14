from yang.data import DateIndex, Array
from typing import Union
import numpy as np
import pandas as pd
import yang.sim.ext

LOT_SIZE = 100

PNL_STATS_FIELDS = [
    "long_count",
    "short_count",
    "long_pnl",
    "short_pnl",
    "long_val",
    "short_val",
    "trade_val",
    "ret",
    "long_ret",
    "short_ret",
    "hedge_ret",
    "ic",
]


def format_yearly_stats(stats, dates: np.array):
    start_date = []
    end_date = []
    max_dd_start = []
    max_dd_end = []
    attrs = [
        "count",
        "long_count",
        "short_count",
        "long_pnl",
        "short_pnl",
        "long_val",
        "short_val",
        "trade_val",
        "trade_cost",
        "pnl",
        "ret",
        "long_ret",
        "short_ret",
        "hedge_ret",
        "tvr",
        "ir",
        "ic",
        "up_days",
        "down_days",
        "max_dd",
    ]
    values = {a: [] for a in attrs}
    for year_stats in stats:
        start_date.append(dates[year_stats.start])
        end_date.append(dates[year_stats.end])
        if year_stats.max_dd != 0:
            max_dd_start.append(dates[year_stats.max_dd_start])
            max_dd_end.append(dates[year_stats.max_dd_end])
        else:
            max_dd_start.append(0)
            max_dd_end.append(0)
        for a in attrs:
            values[a].append(getattr(year_stats, a))
    cols = ["start_date", "end_date"] + attrs + ["max_dd_start", "max_dd_end"]
    values["start_date"] = start_date
    values["end_date"] = end_date
    values["max_dd_start"] = max_dd_start
    values["max_dd_end"] = max_dd_end
    return pd.DataFrame(values, columns=cols)


def compute_pnl_stats(
    sig: Union[np.array, Array],
    ret: Union[np.array, Array],
    univ_size: int,
    book_size: int,
    hedge_ii: int = -1,
) -> pd.DataFrame:
    if isinstance(sig, Array):
        sig = sig.data
    if isinstance(ret, Array):
        ret = ret.data
    stats = yang.sim.ext.compute_pnl_stats(sig, ret, univ_size, book_size, hedge_ii)
    return pd.DataFrame({f: getattr(stats, f) for f in PNL_STATS_FIELDS})


def compute_yearly_pnl_stats(
    dates: Union[np.array, DateIndex],
    sig: Union[np.array, Array],
    ret: Union[np.array, Array],
    univ_size: int,
    book_size: int,
    hedge_ii: int = -1,
    include_total: bool = True,
) -> pd.DataFrame:
    if isinstance(dates, DateIndex):
        dates = dates.items
    if isinstance(sig, Array):
        sig = sig.data
    if isinstance(ret, Array):
        ret = ret.data
    base_stats = yang.sim.ext.compute_pnl_stats(sig, ret, univ_size, book_size, hedge_ii)
    stats = yang.sim.ext.compute_yearly_pnl_stats(base_stats, dates, book_size, include_total)
    return format_yearly_stats(stats, dates)


def compute_tpnl_stats(
    sig: Union[np.array, Array],
    ret: Union[np.array, Array],
    prc: Union[np.array, Array],
    univ_size: int,
    book_size: int,
    buy_fee: float,
    sell_fee: float,
    hedge_ii: int = -1,
) -> pd.DataFrame:
    if isinstance(sig, Array):
        sig = sig.data
    if isinstance(ret, Array):
        ret = ret.data
    if isinstance(prc, Array):
        prc = prc.data
    stats = yang.sim.ext.compute_tpnl_stats(
        sig, ret, prc, univ_size, book_size, buy_fee, sell_fee, hedge_ii
    )
    return pd.DataFrame({f: getattr(stats, f) for f in PNL_STATS_FIELDS})


def compute_yearly_tpnl_stats(
    dates: Union[np.array, DateIndex],
    sig: Union[np.array, Array],
    ret: Union[np.array, Array],
    prc: Union[np.array, Array],
    univ_size: int,
    book_size: int,
    buy_fee: float,
    sell_fee: float,
    hedge_ii: int = -1,
    include_total: bool = True,
) -> pd.DataFrame:
    if isinstance(dates, DateIndex):
        dates = dates.items
    if isinstance(sig, Array):
        sig = sig.data
    if isinstance(ret, Array):
        ret = ret.data
    if isinstance(prc, Array):
        prc = prc.data
    base_stats = yang.sim.ext.compute_tpnl_stats(
        sig, ret, prc, univ_size, book_size, buy_fee, sell_fee, hedge_ii
    )
    stats = yang.sim.ext.compute_yearly_pnl_stats(base_stats, dates, book_size, include_total)
    return format_yearly_stats(stats, dates)
