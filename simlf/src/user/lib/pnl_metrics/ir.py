from typing import Any

import nu
import pandas as pd


def summarize(metrics: pd.DataFrame, sum_metrics: dict[str, float], config: dict[str, Any]):
    intervals_per_day = config["intervals_per_day"]
    std = nu.nanstd(metrics.ret, ddof=1)
    sum_metrics["std"] = std
    sum_metrics["ir"] = nu.nanmean(metrics.ret) / std * (intervals_per_day ** 0.5)
