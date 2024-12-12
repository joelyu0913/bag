import logging
import os
import time

import numpy as np
import pandas as pd
import yaml
from tabulate import tabulate

from yang.data import Array
from yang.sim import Env
from yao.lib.pnl2 import compute_pnls


def is_pnl_update_required(alpha_path, ops, registry_entry):
    if registry_entry["ops"] != ops:
        return True
    mtime = os.path.getmtime(alpha_path)
    if registry_entry["timestamp"] < mtime:
        return True
    return False


def register_corr(cache_dir, corr_config):
    env = Env({"cache": cache_dir})
    corr_dir = env.cache_dir.get_path("_" + corr_config.get("corr_dir", "corr"))
    os.makedirs(corr_dir, exist_ok=True)
    registry_path = os.path.join(corr_dir, "registry.yml")
    if os.path.exists(registry_path):
        with open(registry_path) as f:
            registry = yaml.safe_load(f)
    else:
        registry = {}

    metric = corr_config.get("metric", "ic")
    start_date = corr_config.get("start_date", None)
    end_date = corr_config.get("end_date", None)
    if start_date is not None:
        start_di = env.dates.lower_bound(start_date)
    else:
        start_di = 0
    if end_date is not None:
        end_di = env.dates.upper_bound(end_date)
    else:
        end_di = len(env.dates)
    for name, alpha_config in corr_config.get("alphas", {}).items():
        alpha = env.cache_dir.get_path(alpha_config["alpha"])
        if not os.path.exists(alpha):
            logging.error(f"Missing alpha file: {alpha}")
            continue
        ops = alpha_config.get("ops", "")
        tags = alpha_config.get("tags", [])
        if name not in registry or is_pnl_update_required(alpha, ops, registry[name]):
            daily_pnls = compute_pnls(
                env,
                Array.mmap(alpha),
                ops=ops,
                start_di=start_di,
                end_di=end_di,
                aggregate=False,
            )
            metric_values = daily_pnls[metric].to_numpy().astype(np.float32)
            Array(metric_values).save(os.path.join(corr_dir, name))
        registry[name] = {
            "alpha": alpha,
            "ops": ops,
            "tags": tags,
            "timestamp": int(time.time()),
        }

    with open(registry_path, "w") as f:
        yaml.dump(registry, f, Dumper=yaml.CDumper)


def corr(cache_dir, alpha, tags, start_date=None, end_date=None, corr_dir="corr", display_num=10):
    env = Env({"cache": cache_dir})
    corr_path = env.cache_dir.get_path("_" + corr_dir)
    registry_path = os.path.join(corr_path, "registry.yml")
    if not os.path.exists(registry_path):
        raise RuntimeError("corr registry does not exist")
    with open(registry_path) as f:
        registry = yaml.safe_load(f)

    if start_date is not None:
        start_di = env.dates.lower_bound(start_date)
    else:
        start_di = 0
    if end_date is not None:
        end_di = env.dates.upper_bound(end_date)
    else:
        end_di = len(env.dates)

    if "/" not in alpha:
        alpha_path = os.path.join(corr_path, alpha)
        if not os.path.exists(alpha_path):
            raise RuntimeError(f"alpha {alpha} is not in the registry")
        x_metric = Array.mmap(alpha_path).data[start_di:end_di]
    else:
        alpha_path = os.path.join(cache_dir, alpha)
        alpha_array = Array.mmap(alpha_path)
        daily_pnls = compute_pnls(
            env, alpha_array, start_di=start_di, end_di=end_di, aggregate=False
        )
        x_metric = daily_pnls["ic"].to_numpy().astype(np.float32)[start_di:end_di]

    corrs = []
    missing_alphas = []
    for alpha_name, alpha_config in registry.items():
        if alpha_name == alpha:
            continue
        alpha_tags = alpha_config["tags"]
        if all(t in alpha_tags for t in tags):
            alpha_path = os.path.join(corr_path, alpha_name)
            if os.path.exists(alpha_path):
                y_metric = Array.mmap(alpha_path).data[start_di:end_di]
                corrs.append(
                    (
                        np.corrcoef(x_metric, y_metric)[0, 1],
                        alpha_name + " - " + " - ".join(alpha_tags),
                    )
                )
            else:
                missing_alphas.append(alpha_name)
    corrs = list(reversed(sorted(corrs)))
    names = [c[1] for c in corrs]
    values = [c[0] for c in corrs]
    table = pd.DataFrame(data={"alpha": names, "corr": values})[:display_num]
    print(tabulate(table, headers=["alpha", "corr"], showindex=False))

    if len(missing_alphas) > 0:
        logging.debug("removing missing alphas from registry: %s", missing_alphas)
        for alpha in missing_alphas:
            del registry[alpha]
        with open(registry_path, "w") as f:
            yaml.dump(registry, f, Dumper=yaml.CDumper)
