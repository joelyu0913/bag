import concurrent
import glob
import gzip
import multiprocessing as mpro
import os
import random

import numba
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml


def my_create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


@numba.njit()
def j_cosine(a, b):
    lower_ = np.linalg.norm(a) * np.linalg.norm(b)
    if np.abs(lower_) < 1e-20:
        return 0.0
    return np.dot(a, b) / lower_


@numba.njit()
def j_cosine_period(a, b, periods):
    cos_ = 0.0
    for start, end in periods:
        cos_ += j_cosine(a[start:end], b[start:end])
    return cos_ / len(periods)


@numba.njit()
def j_corr(a, b):
    lower_ = np.linalg.norm(a) * np.linalg.norm(b)
    if np.abs(lower_) < 1e-20:
        return 0.0
    return (np.dot(a, b) - np.sum(a) * np.sum(b) / len(a)) / lower_


@numba.njit()
def j_compute_pnl(alphas, ret):
    return np.sum(alphas * ret)


@numba.njit()
def j_compute_tvr(alphas):
    return np.sum(np.abs(alphas - j_shift(alphas, 1, 0.0)))


@numba.njit()
def compute_accuracy(alphas, ret):
    pnls = alphas * ret
    return np.sum(pnls > 0) / np.sum(pnls != 0)


@numba.njit()
def cosine_threshold(a, b, threshold):
    a = pd.Series(a).copy().reset_index(drop=True)
    b = pd.Series(b).copy().reset_index(drop=True).fillna(0)
    b[np.abs(a) < threshold] = np.nan
    a[np.abs(a) < threshold] = np.nan
    return (j_cosine(a.dropna(), b.dropna()), a.count())


def my_shift(arr, num, fill_value=np.nan):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result


@numba.njit()
def j_shift(arr, num, fill_value=np.nan):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result


def load_np(path):
    with open(path, "rb") as f:
        return np.load(f)


def save_np(path, obj_):
    with open(path, "wb") as f:
        np.save(f, obj_)


def load_list(path, load_type=str):
    l = []
    with open(path, "r") as f:
        for ln in f:
            l.append(load_type(ln.strip()))
    return l


def save_list(path, list_):
    with open(path, "w") as f:
        for eli in list_:
            f.write(f"{eli}\n")


def load_yaml(path):
    with open(path, "r") as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)


def save_yaml(path, data, default_flow_style=True):
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=default_flow_style)


class DLoader:
    def __init__(self, path):
        self._path = path
        if len(self._path) == 0 or self._path[-1] != "/":
            self._path += "/"

    def rpath(self, name=""):
        return os.path.join(self._path, name)

    def load_np(self, name):
        return load_np(self.rpath(name))

    def save_np(self, name, obj_):
        return save_np(self.rpath(name), obj_)

    def load_list(self, name, load_type=str):
        return load_list(self.rpath(name), load_type)

    def load_csv(self, name, **args):
        return pd.read_csv(self.rpath(name), **args)

    def save_list(self, name, obj_):
        return save_list(self.rpath(name), obj_)

    def load_parquet(self, name):
        return pq.read_table(self.rpath(name))

    def save_parquet(self, name, obj_, key=None):
        if key is None:
            key = name
        return pq.write_table(pa.table({key: obj_}), self.rpath(name))

    def load_yaml(self, name):
        return load_yaml(self.rpath(name))

    def save_yaml(self, name, obj_, style=True):
        return save_yaml(self.rpath(name), obj_, style)

    def exists(self, name=""):
        return True if os.path.exists(self.rpath(name)) else False

    def makedirs(self, name=""):
        os.makedirs(self.rpath(name))

    def load_np_exists(self, name, target):
        assert type(target) == list, f"type error: target must be list! {target}"

        if self.exists(name):
            target[0] = self.load_np(self.rpath(name))
            return True
        else:
            return False

    @property
    def last_dir(self):
        return self._path.split("/")[-2]


def my_process_pool(func, pars, pool_process_num=6, pool_batch_size=None):
    if pool_batch_size == None:
        pool_batch_size = len(pars)

    for i in range(0, len(pars), pool_batch_size):
        pool = mpro.Pool(pool_process_num)
        pool.starmap(func, pars[i : min(len(pars), i + pool_batch_size)])
        pool.close()


def my_thread_pool(func, pars, pool_thread_num=6):
    # using context manager to enable auto .join()
    with concurrent.futures.ThreadPoolExecutor(max_workers=pool_thread_num) as executor:
        executor.map(lambda p: func(*p), pars)


def my_eq(np_a, np_b, pct=1e-10):
    return (
        (np.isnan(np_a) & np.isnan(np_b))
        | (np.isinf(np_a) & np.isinf(np_b))
        | (np.abs(np_a - np_b) < pct * (np.abs(np_a) + np.abs(np_b) + pct))
    )


INVALID_TIME = -1


class TimeScale:
    NS_PER_US = 1000
    NS_PER_MS = 1000000
    NS_PER_SEC = 1000000000
    NS_PER_MIN = 60 * 1000000000
    NS_PER_HOUR = 60 * 60 * 1000000000


def itv2ns(itv):
    if itv == "" or itv is None:
        return INVALID_TIME

    if itv.endswith("us"):
        return int(float(itv[:-2]) * TimeScale.NS_PER_US)
    elif itv.endswith("ms"):
        return int(float(itv[:-2]) * TimeScale.NS_PER_MS)
    elif itv.endswith("s"):
        return int(float(itv[:-1]) * TimeScale.NS_PER_SEC)
    elif itv.endswith("m"):
        return int(float(itv[:-1]) * TimeScale.NS_PER_MIN)
    elif itv.endswith("h"):
        return int(float(itv[:-1]) * TimeScale.NS_PER_HOUR)
    return int(itv)


def my_assert(a, b):
    if np.shape(a) != np.shape(b):
        assert False

    if len(np.shape(a)) == 0:
        assert a == b
    else:
        assert np.sum(a != b) == 0


"""
my_assert(get_upper([7, 14, 15, 64, 66, 68, 71, 73, 78, 91], 83), 9)
my_assert(get_upper([2, 6, 27, 36, 37, 40, 44, 52, 60, 81], 71), 9)
my_assert(get_upper([6, 16, 33, 42, 50, 54, 62, 83, 87, 91], 99), 10)
my_assert(get_upper([23, 28, 39, 60, 64, 71, 79, 87, 91, 97], 16), 0)
my_assert(get_upper([17, 25, 48, 51, 65, 66, 70, 70, 83, 91], 88), 9)

my_assert(get_lower([0, 4, 12, 22, 29, 46, 56, 62, 70, 90], 27), 3)
my_assert(get_lower([7, 21, 26, 31, 45, 49, 49, 66, 92, 100], 85), 7)
my_assert(get_lower([8, 9, 13, 15, 41, 49, 50, 88, 89, 93], 92), 8)
my_assert(get_lower([16, 20, 21, 25, 37, 48, 61, 72, 74, 94], 82), 8)
my_assert(get_lower([26, 39, 39, 45, 71, 82, 91, 93, 97, 97], 45), 3)

my_assert(get_upper([1,3,4,5], [0,4,6], True), np.array([0,3,4]))
my_assert(get_upper([1,3,4,5], [0,4,6]), np.array([0,2,4]))
my_assert(get_lower([1,3,4,5], [0,4,6], True), np.array([-1,1,3]))
my_assert(get_lower([1,3,4,5], [0,4,6]), np.array([-1,2,3]))
"""


@numba.njit()
def _get_upper_oend(srs_asc, seps):
    i = 0
    srs_idx = np.full(seps.shape, len(srs_asc))
    for j in range(len(seps)):
        while i < len(srs_asc):
            if srs_asc[i] > seps[j]:
                break
            i += 1
        srs_idx[j] = i

    return srs_idx


@numba.njit()
def _get_upper_cend(srs_asc, seps):
    i = 0
    srs_idx = np.full(seps.shape, len(srs_asc))
    for j in range(len(seps)):
        while i < len(srs_asc):
            if srs_asc[i] >= seps[j]:
                break
            i += 1
        srs_idx[j] = i

    return srs_idx


def get_upper(srs_asc, seps, flag_oend=False):
    """
    get_upper(ascending series, separation srs, flag_oend=False)
    get_upper([1,3,4,5], [0,4,6], True) --> np.array([0,3,4])
    """
    srs_seps = [seps] if len(np.shape(seps)) == 0 else seps
    ff = _get_upper_oend if flag_oend else _get_upper_cend
    x = ff(np.array(srs_asc), np.array(srs_seps))
    return x[0] if len(np.shape(seps)) == 0 else x


@numba.njit()
def _get_lower_oend(srs_asc, seps):
    i = len(srs_asc) - 1
    srs_idx = np.full(seps.shape, len(srs_asc))
    for j in range(len(seps) - 1, -1, -1):
        while i >= 0:
            if srs_asc[i] < seps[j]:
                break
            i -= 1
        srs_idx[j] = i

    return srs_idx


@numba.njit()
def _get_lower_cend(srs_asc, seps):
    i = len(srs_asc) - 1
    srs_idx = np.full(seps.shape, len(srs_asc))
    for j in range(len(seps) - 1, -1, -1):
        while i >= 0:
            if srs_asc[i] <= seps[j]:
                break
            i -= 1
        srs_idx[j] = i

    return srs_idx


def get_lower(srs_asc, seps, flag_oend=False):
    """
    get_lower(ascending series, separation series, flag_oend=False)
    get_lower([1,3,4,5], [0,4,6], True) --> np.array([-1,1,3])
    """
    srs_seps = [seps] if len(np.shape(seps)) == 0 else seps
    ff = _get_lower_oend if flag_oend else _get_lower_cend
    x = ff(np.array(srs_asc), np.array(srs_seps))
    return x[0] if len(np.shape(seps)) == 0 else x


# can handle duplicated cases
@numba.njit()
def j_select(srs, l, r, index):

    if r == l:
        return srs[l]
    pivot_index = random.randint(l, r)

    srs[l], srs[pivot_index] = srs[pivot_index], srs[l]
    i = l
    for j in range(l + 1, r + 1):
        if srs[j] < srs[l] or (srs[j] == srs[l] and (j % 2 == 0)):
            i += 1
            srs[i], srs[j] = srs[j], srs[i]

    srs[i], srs[l] = srs[l], srs[i]

    if index == i:
        return srs[i]
    elif index < i:
        return j_select(srs, l, i - 1, index)
    else:
        return j_select(srs, i + 1, r, index)


def select_nth(items, item_index):

    if items is None or len(items) < 1:
        return None

    if item_index < 0 or item_index > len(items) - 1:
        raise IndexError()

    return j_select(items, 0, len(items) - 1, item_index)


def my_open(path, mode=""):
    _, fext = os.path.splitext(path)
    if fext == ".gz":
        if mode == "":
            mode = "rt"
        return gzip.open(path, mode)
    else:
        if mode == "":
            mode = "r"
        return open(path, mode)
