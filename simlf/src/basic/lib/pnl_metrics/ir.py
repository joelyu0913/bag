from typing import Any

from cli.nan import nanstd, nanmean
import numpy as np
import pandas as pd


def summarize(metrics: pd.DataFrame, sum_metrics: dict[str, float], config: dict[str, Any]):
    intervals_per_day = config["intervals_per_day"]
    std = nanstd(metrics.ret, ddof=1)
    std = np.nanstd(metrics.ret, ddof=1)
    sum_metrics["std"] = std
    sum_metrics["ir"] = nanmean(metrics.ret) / std * (intervals_per_day ** 0.5)
    # sum_metrics["ir"] = np.nanmean(metrics.ret) / std * (intervals_per_day ** 0.5)
