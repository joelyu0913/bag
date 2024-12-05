import os
from datetime import timedelta
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch.nn
import torch.utils.data
from numba import njit

from sim import DataDirectory
import logging


def load_features(
    paths: list[str],
    stock_id_feature: bool = False,
    start_di: Optional[int] = None,
    end_di: Optional[int] = None,
    univ_size: Optional[int] = None,
    shape_hint: Optional[list[int]] = None,
    dtype_hint: Optional[np.dtype] = None,
    filter_mask: Optional[np.array] = None,
) -> np.array:
    """
    Load features in date range [start_di, end_di). Output shape is
    (num_stocks, num_dates, num_features).
    """
    from data import Array

    assert len(paths) > 0

    if shape_hint is None:
        xs = [Array.mmap(p).data for p in paths]
    else:
        xs = [np.memmap(p, mode="r", dtype=dtype_hint).reshape(shape_hint) for p in paths]
    xs = [x[start_di:end_di, :univ_size] for x in xs]
    num_stocks = xs[0].shape[1]
    xs = [f.reshape((1, -1, num_stocks)) for f in xs]
    if stock_id_feature:
        stock_id = np.empty(xs[0].shape)
        for i in range(num_stocks):
            stock_id[:, :, i] = (i + 1) / num_stocks
        xs.append(stock_id)
    X = np.concatenate(xs, axis=0).transpose((2, 1, 0))
    if filter_mask is not None:
        X[~filter_mask] = np.nan
    return X


def load_target(
    path: str,
    start_di: Optional[int] = None,
    end_di: Optional[int] = None,
    univ_size: Optional[int] = None,
    filter_mask: Optional[np.array] = None,
) -> np.array:
    """
    Load target in date range [start_di, end_di). Output shape is (num_stocks, num_dates, 1).
    """
    from data import Array

    Y = Array.load(path).data[start_di:end_di, :univ_size]
    num_stocks = Y.shape[1]
    Y = Y.reshape((1, -1, num_stocks)).transpose((2, 1, 0))
    if filter_mask is not None:
        Y[~filter_mask] = np.nan
    return Y


def load_univ(
    data_dir: DataDirectory,
    univ_name: Optional[str] = None,
    start_di: Optional[int] = None,
    end_di: Optional[int] = None,
    univ_size: Optional[int] = None,
) -> np.array:
    if not univ_name:
        return None
    from data import Array

    univ = Array.mmap(data_dir.get_read_path(univ_name)).data[start_di:end_di, :univ_size]
    return univ.transpose()


def load_xy(
    data_dir: DataDirectory,
    x_paths: list[str],
    y_path: str,
    start_di: Optional[int] = None,
    end_di: Optional[int] = None,
    univ_size: Optional[int] = None,
    filter_mask: Optional[np.array] = None,
) -> tuple[np.array, np.array]:
    """
    Load features and target in date range [start_di, end_di)
    """
    y_path = data_dir.get_path(y_path)
    stock_id_feature = "STOCK_ID" in x_paths
    if stock_id_feature:
        x_paths.remove("STOCK_ID")
    x_paths = [data_dir.get_path(p) for p in x_paths]
    Y = load_target(y_path)
    X = load_features(
        x_paths,
        stock_id_feature,
        start_di=start_di,
        end_di=end_di,
        univ_size=univ_size,
        shape_hint=(-1, Y.shape[0]),
        dtype_hint=Y.dtype,
        filter_mask=filter_mask,
    )
    Y = Y[:univ_size, start_di:end_di]
    if filter_mask is not None:
        Y[filter_mask == False] = np.nan
    return X, Y


@njit()
def collect_valid_indices(
    X: np.array,
    Y: Optional[np.array],
    start_di: int,
    end_di: int,
    seq_len: int,
    valid_ratio: float = 0.999,
    check_y_seq: bool = False,
) -> np.array:
    """
    Collect indices with valid X sequence and Y in date range [start_di, end_di)
    """
    if Y is not None:
        assert X.shape[0] == Y.shape[0]
        assert X.shape[1] == Y.shape[1]

    indices = []
    valid_acc = np.zeros(X.shape[1])  # number of valid days from check_start
    check_start = max(start_di - seq_len + 1, 0)
    check_end = end_di
    for s in range(X.shape[0]):
        valid_acc[check_start:check_end] = 0
        for t in range(check_start, check_end):
            num_valids = np.sum(np.isfinite(X[s, t]))
            if num_valids / X.shape[2] >= valid_ratio and (
                not check_y_seq or Y is None or np.isfinite(Y[s, t, 0])
            ):
                valid_acc[t] = 1
            if t > 0:
                valid_acc[t] += valid_acc[t - 1]

            valid_t = valid_acc[t]
            if t >= seq_len:
                valid_t -= valid_acc[t - seq_len]
            if valid_t == seq_len and (Y is None or np.isfinite(Y[s, t, 0])):
                indices.append((s, t))
    if len(indices) == 0:
        return None
    return np.array(indices)


class IndexTsDataset(torch.utils.data.Dataset):
    def __init__(
        self, indices: np.array, X: np.array, Y: np.array, seq_len: int, time_tag: bool = False
    ):
        self.X = X
        self.Y = Y
        self.indices = indices
        self.seq_len = seq_len
        self.time_tag = time_tag

    def __getitem__(self, index: int):
        s, t = self.indices[index]
        assert t >= self.seq_len - 1
        x = self.X[s, t - self.seq_len + 1 : t + 1]
        x = torch.from_numpy(x.astype(np.float32)).float()
        if self.Y is not None:
            y = self.Y[s, t]
            y = torch.from_numpy(y.astype(np.float32)).float()
        else:
            y = torch.from_numpy(np.zeros(1, dtype=np.float32)).float()
        if self.time_tag:
            return x, y, t
        else:
            return x, y

    def __len__(self):
        return self.indices.shape[0]


def generate_roll_dates(start_date: int, end_date: int, intervals: dict[str, float]) -> list[dict]:
    start_date = pd.Timestamp(str(start_date))
    end_date = pd.Timestamp(str(end_date))
    train_interval = timedelta(days=int(365 * intervals["train"]))
    valid_interval = timedelta(days=int(365 * intervals["valid"]))
    pred_interval = timedelta(days=int(365 * intervals["pred"]))
    pred_start = start_date + train_interval + valid_interval
    pred_dates = pd.date_range(pred_start, end_date + timedelta(days=1), freq=pred_interval)
    train_from_start = intervals.get("train_from_start", True)
    if "max_train" in intervals:
        max_train_interval = timedelta(days=int(365 * intervals["max_train"]))
    else:
        max_train_interval = train_interval

    one_day = timedelta(days=1)
    gap_days = timedelta(days=intervals.get("pred_gap_days", 0))
    results = []
    conv_date = lambda d: int(d.strftime("%Y%m%d"))
    for pred_start in pred_dates:
        pred_end = pred_start + pred_interval - one_day
        valid_end = pred_start - one_day - gap_days
        valid_start = pred_start - valid_interval - gap_days
        train_end = valid_start - one_day
        if train_from_start:
            train_start = start_date
        else:
            train_start = max(start_date, valid_start - max_train_interval)
        results.append(
            dict(
                train=(conv_date(train_start), conv_date(train_end)),
                valid=(conv_date(valid_start), conv_date(valid_end)),
                pred=(conv_date(pred_start), conv_date(pred_end)),
            )
        )
    return results


def predict_nn(
    model: torch.nn.Module,
    X: np.array,
    indices: np.array,
    seq_len: int,
    device: str = "cpu",
    batch_size: int = 256,
    data_workers: int = 4,
    Y: np.array = None,
) -> np.array:
    dataset = IndexTsDataset(indices, X, None, seq_len)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=data_workers
    )
    if Y is None:
        Y = np.full((X.shape[0], X.shape[1], 1), np.nan)
    ys = []
    model.eval()
    for item in loader:
        pred_y = model(item[0].to(device))
        ys.append(pred_y.cpu().detach().numpy())
    ys = np.concatenate(ys)
    for i in range(len(ys)):
        s, t = indices[i]
        Y[s, t, 0] = ys[i]
    return Y


def predict_ml(
    model,
    X: np.array,
    indices: Optional[np.array],
    Y: np.array = None,
    **kwargs,
) -> np.array:
    if len(X.shape) == 3:
        if Y is None:
            Y = np.full((X.shape[0], X.shape[1], 1), np.nan)
        X = X.reshape(-1, X.shape[2])  # (S, T, F) -> (S x T, F)
        indices_flatten = flatten_indices(indices, Y.shape[1])
        ys = model.predict(X[indices_flatten], **kwargs)
        for i in range(len(ys)):
            s, t = indices[i]
            Y[s, t, 0] = ys[i]
    else:
        if Y is None:
            Y = np.full((X.shape[0], 1), np.nan)
        if indices is not None:
            X = X[indices]
        ys = model.predict(X, **kwargs)
        if indices is None:
            Y[:, 0] = ys
        else:
            for i in range(len(ys)):
                t = indices[i]
                Y[t, 0] = ys[i]
    return Y


def get_feature_paths(config) -> list[str]:
    features = []
    for i in config:
        if isinstance(i, str):
            features.append(i)
        elif "path" in i:
            features.append(i["path"])
    return features


@njit()
def flatten_indices(indices: np.array, row_len: int) -> np.array:
    ret = np.zeros(len(indices), dtype=np.int64)
    for i in range(len(indices)):
        ret[i] = indices[i][0] * row_len + indices[i][1]
    return ret


@njit()
def generate_sample_weights(indices: np.array, tsample_weights: np.array) -> np.array:
    ret = np.zeros(len(indices))
    for i in range(len(indices)):
        ret[i] = tsample_weights[indices[i][1]]
    return ret


def load_features_grf(
    paths: list[str],
    start_di: Optional[int] = None,
    end_di: Optional[int] = None,
    min_valid_count: int = 1,
) -> tuple[np.array, np.array, np.array]:
    """
    Load features in date range [start_di, end_di). Output shape is
    (N, num_features).
    Returns (features, orig_mask, blocks)
    """
    from data import BlockVector

    assert len(paths) > 0

    vecs = [BlockVector.mmap(p) for p in paths]
    if start_di is None:
        start_idx = 0
        start_di = 0
    else:
        start_idx = vecs[0].block_begin(start_di)
    if end_di is None:
        end_idx = None
        end_di = len(vecs[0].blocks)
    else:
        end_idx = vecs[0].block_end(end_di - 1)
    xs = [vec[start_idx:end_idx] for vec in vecs]
    valid_counts = np.zeros(len(xs[0]))
    for x in xs:
        valid_counts += np.isfinite(x)
    indices = valid_counts >= min_valid_count
    X = np.concatenate([x[indices].reshape(-1, 1) for x in xs], 1)
    blocks = [0]
    for di in range(start_di, end_di):
        idx1 = int(vecs[0].block_begin(di) - vecs[0].block_begin(start_di))
        idx2 = int(vecs[0].block_end(di) - vecs[0].block_begin(start_di))
        blocks.append(blocks[-1] + int(np.sum(indices[idx1:idx2])))
    return X, indices, np.array(blocks, dtype=np.uint64)


def load_blocks_grf(
    path: str,
    start_di: Optional[int] = None,
    end_di: Optional[int] = None,
) -> np.array:
    from data import BlockVector

    blocks = BlockVector.mmap(path).blocks
    if start_di is not None and start_di > 0:
        block_off = blocks[start_di - 1]
    else:
        block_off = 0
    blocks = blocks[start_di:end_di] - block_off
    blocks = np.concatenate(([0], blocks))
    return blocks.astype(np.uint64)


def load_xy_grf(
    data_dir: DataDirectory,
    x_paths: list[str],
    y_path: str,
    start_di: Optional[int] = None,
    end_di: Optional[int] = None,
    min_valid_count: int = 1,
) -> tuple[np.array, np.array, np.array, np.array]:
    """
    Load features and target in date range [start_di, end_di).
    Returns (X, Y, orig_mask, blocks)
    """
    y_path = data_dir.get_path(y_path)
    x_paths = [data_dir.get_path(p) for p in x_paths]
    data, mask, blocks = load_features_grf(
        [y_path] + x_paths,
        start_di=start_di,
        end_di=end_di,
        min_valid_count=min_valid_count,
    )
    Y = data[:, :1]
    X = data[:, 1:]
    return X, Y, mask, blocks


@njit()
def collect_valid_indices_grf(
    X: np.array,
    Y: Optional[np.array],
    blocks: np.array,
    start_di: int,
    end_di: int,
    valid_ratio: float = 0.999,
) -> np.array:
    """
    Collect indices with valid X sequence and Y in date range [start_di, end_di)
    """
    if Y is not None:
        assert X.shape[0] == Y.shape[0]

    indices = []
    check_start = max(start_di, 0)
    check_end = end_di
    for t in range(blocks[check_start], blocks[check_end]):
        num_valids = np.sum(np.isfinite(X[t]))
        valid_t = (num_valids / X.shape[1]) >= valid_ratio
        if valid_t and (Y is None or np.isfinite(Y[t, 0])):
            indices.append(t)
    if len(indices) == 0:
        return None
    return np.array(indices)


class DateTimeIndex:
    def __init__(self, datetimes, daily: bool):
        self.datetimes = datetimes
        self.daily = daily

    def __len__(self):
        return len(self.datetimes)

    def __getitem__(self, i) -> Any:
        return self.datetimes[i]

    def get_date(self, i) -> Any:
        if self.daily:
            return self.datetimes[i]
        return self.datetimes[i] // 10000

    def lower_bound_date(self, date: int) -> int:
        if self.daily:
            return self.datetimes.lower_bound(date)
        return self.datetimes.lower_bound(date * 10000)

    def upper_bound_date(self, date: int) -> int:
        if self.daily:
            return self.datetimes.upper_bound(date)
        return self.datetimes.upper_bound(date * 10000 + 2359)

    def lower_bound(self, date: int) -> int:
        return self.datetimes.lower_bound(date)

    def upper_bound(self, date: int) -> int:
        return self.datetimes.upper_bound(date)
