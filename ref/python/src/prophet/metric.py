import numpy as np
import pandas as pd
import torch
from numba import njit


def normalize_vector(y):
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()
    if len(y.shape) == 2 and y.shape[1] == 1:
        y = y[:, 0]
    return y


def IC(y1, y2):
    y1 = normalize_vector(y1)
    y2 = normalize_vector(y2)
    return np.sum(y1 * y2) / np.sqrt(np.sum(y1 ** 2) * np.sum(y2 ** 2))


@njit()
def group_IC(x, y, g):
    num_groups = 0
    g_idx = {}
    for g_id in g:
        if g_id not in g_idx:
            g_idx[g_id] = num_groups
            num_groups += 1
    sum_xy = np.zeros(num_groups)
    sum_x2 = np.zeros(num_groups)
    sum_y2 = np.zeros(num_groups)
    for i in range(len(x)):
        idx = g_idx[g[i]]
        sum_xy[idx] += x[i] * y[i]
        sum_x2[idx] += x[i] * x[i]
        sum_y2[idx] += y[i] * y[i]
    ic_sum = 0
    for i in range(num_groups):
        ic_sum += sum_xy[i] / np.sqrt(sum_x2[i] * sum_y2[i])
    return ic_sum / num_groups


class ICAccumulator(object):
    def __init__(self):
        self.targets = []
        self.preds = []
        self.tags = []

    def add(self, target, pred, tag=None) -> None:
        self.targets.append(normalize_vector(target))
        self.preds.append(normalize_vector(pred))
        if tag is not None:
            self.tags.append(normalize_vector(tag))

    def get(self) -> float:
        targets = np.concatenate(self.targets)
        preds = np.concatenate(self.preds)
        if len(self.tags) == len(self.targets):
            tags = np.concatenate(self.tags)
            return group_IC(targets, preds, tags)
        else:
            return IC(targets, preds)


METRIC_MAP = {
    "IC": IC,
}

METRIC_ACC_MAP = {
    "IC": ICAccumulator,
}
