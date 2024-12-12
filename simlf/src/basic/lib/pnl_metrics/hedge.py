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
    ret = config["ret"]
    hedge = config.get("hedge", env.benchmark_index)
    hedge_ii = env.univ.find(hedge)
    hedge_ret = metrics.long_ret - ret[:, hedge_ii]
    hedge_ret[0] = 0
    metrics["hedge_ret"] = hedge_ret


def summarize(metrics: pd.DataFrame, sum_metrics: dict[str, float], config: dict[str, Any]):
    sum_metrics["hedge_ret"] = (
        metrics.hedge_ret.mean() * config["days_per_year"] * config["intervals_per_day"]
    )
