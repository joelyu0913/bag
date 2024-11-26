from typing import Any

import numpy as np
import pandas as pd
from numba import njit

from sim import Env


@njit()
def compute_cos(sig: np.array, ret: np.array) -> float:
    sum_xy = 0
    sum_x2 = 0
    sum_y2 = 0
    for i in range(len(sig)):
        x = sig[i]
        y = ret[i]
        if np.isfinite(x) and np.isfinite(y):
            sum_xy += x * y
            sum_x2 += x * x
            sum_y2 += y * y
    if sum_x2 == 0 or sum_y2 == 0:
        return np.nan
    return sum_xy / np.sqrt(sum_x2 * sum_y2)


@njit()
def compute_daily_ic(sig: np.array, ret: np.array) -> np.array:
    ic = np.zeros(len(sig))
    for i in range(1, len(sig)):
        ic[i] = compute_cos(sig[i - 1], ret[i])
    return ic


def compute(
    env: Env,
    sig: np.array,
    metrics: pd.DataFrame,
    config: dict[str, Any],
):
    ret = config["ret"]
    metrics["ic"] = compute_daily_ic(sig[:, : env.univ_size], ret[:, : env.univ_size])


def summarize(metrics: pd.DataFrame, sum_metrics: dict[str, float], config: dict[str, Any]):
    sum_metrics["ic"] = metrics.ic.mean()
