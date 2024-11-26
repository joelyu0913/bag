from typing import Any

import numpy as np
import pandas as pd

from sim import Env


def compute(
    env: Env,
    sig: np.array,
    metrics: pd.DataFrame,
    config: dict[str, Any],
):
    buy_fee = config.get("buy_fee", 1e-4)
    sell_fee = config.get("sell_fee", 11e-4)

    n = len(sig)
    trade_cost = np.zeros(n)

    prev_sig_row = np.zeros(sig.shape[1])
    for i in range(n):
        sig_row = np.copy(sig[i])
        sig_row[~np.isfinite(sig_row)] = 0

        if i > 0:
            trade_val = sig_row - prev_sig_row
            trade_cost[i] = -buy_fee * np.sum(trade_val[trade_val > 0]) + sell_fee * np.sum(
                trade_val[trade_val < 0]
            )

        prev_sig_row = sig_row

    metrics["trade_cost"] = trade_cost
    metrics["pnl"] = metrics.pnl + trade_cost
    metrics["ret"] = metrics["pnl"] / config["book_size"]


def summarize(metrics: pd.DataFrame, sum_metrics: dict[str, float], config: dict[str, Any]):
    sum_metrics["trade_cost"] = metrics.trade_cost.sum()
