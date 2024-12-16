import datetime
import decimal
import gzip
import os
import sys
from collections.abc import Iterable

import numpy as np
import pandas as pd

def read_header(f, sep ='|'):
    header = f.readline().strip().split(sep)
    return {header[i]: i for i in range(len(header))}


def read_line(raw_line, header_mp, sep='|', float_fields=[]):
    line_ = raw_line.strip().split(sep)
    line =  {x: line_[i] for x, i in header_mp.items()}
    for k in float_fields:
        if line[k] == "":
            line[k] = np.nan
        else:
            line[k] = float(line[k])
    return line

class EnvData:
    def __init__(self, env_dir, start_date, end_date):
        self.dates = []
        with open(f"{env_dir}/trade_dates") as f:
            for ln in f:
                date = int(ln)
                if date < start_date or date > end_date:
                    continue
                self.dates.append(date)
        secmaster_path = f"{env_dir}/sec_master/sec_master"
        if os.path.exists(secmaster_path):
            df_sec = pd.read_csv(secmaster_path, delimiter="|")
            self.delist = {a: b for a, b in zip(df_sec["sid"], df_sec["delist"])}


class BaseData(EnvData):
    def __init__(self, data_root, start_date, end_date, env_dir="env"):
        self.data_root = data_root
        super().__init__(f"{data_root}/{env_dir}/", start_date, end_date)

    def get_index(self, basedata_index_fmt):
        self.index_closes = []
        for date in self.dates[:]:
            index_path = format_date(self.data_root + "/" + basedata_index_fmt, date)
            index = DF(index_path, columns=["close"], index="sid", clean=True)
            index_close = float(index.rows["000905.SH"]["close"])
            self.index_closes.append(index_close)


def format_date(input, date):
    if not isinstance(date, str):
        date = str(date)
    yyyy, mm, dd = date[:4], date[4:6], date[6:8]
    HH, MM, SS = 0, 0, 0
    if len(date) >= 10:
        HH = date[8:10]
    if len(date) >= 12:
        MM = date[10:12]
    if len(date) >= 14:
        SS = date[12:14]
    return input.format(yyyy=yyyy, mm=mm, dd=dd, HH=HH, MM=MM, SS=SS)


def round_float(v, precision=2):
    PRECISION = decimal.Decimal(str(10 ** (-precision)))
    return float(decimal.Decimal(v).quantize(PRECISION, rounding=decimal.ROUND_HALF_UP))


class BaseEnv:
    def __init__(self, data_root, start_date=20110101, end_date=20999999):
        self.data_root = data_root
        date_path = self.data_root + "/env/trade_dates"
        self.dates = [
            _ for _ in list(pd.read_csv(date_path, header=None)[0]) if start_date <= _ <= end_date
        ]


class DF:
    def __init__(self, path=None, delimiter="|", columns=[], index="", clean=False):
        if index == "":
            self.rows = []
        else:
            self.rows = {}

        if path is None:
            self.cols = columns
        else:
            vv = path.split(".")
            if len(vv) > 0 and vv[-1] == "gz":
                rf = gzip.open(path, "rt")
            else:
                rf = open(path, "r")

            lines = list(rf)
            cols = lines[0].strip().split(delimiter)
            if columns == []:
                self.cols = cols
            else:
                self.cols = columns
            if not clean:
                if index == "":
                    self.rows = [dict(zip(cols, l.strip().split(delimiter))) for l in lines[1:]]
                else:
                    for l in lines[1:]:
                        row = dict(zip(cols, l.strip().split(delimiter)))
                        self.rows[row[index]] = row
            else:
                if index == "":
                    indices = [True if col in self.cols else False for col in cols]
                    for l in lines[1:]:
                        row = {
                            key: val
                            for key, val, is_in in zip(cols, l.strip().split(delimiter), indices)
                            if is_in
                        }
                        self.rows.append(row)
                else:
                    for l in lines[1:]:
                        row = dict(zip(cols, l.strip().split(delimiter)))
                        self.rows[row[index]] = {
                            key: val for key, val in row.items() if key in self.cols
                        }
                        # print({key:val for key,val in row.items() if key in self.cols})

    def rename(self, mp):
        self.cols = [mp[col] if col in mp else col for col in self.cols]
        if isinstance(self.rows, list):
            for row in self.rows:
                for k_old, k_new in mp.items():
                    row[k_new] = row.pop(k_old)
        else:
            for sid, row in self.rows.items():
                for k_old, k_new in mp.items():
                    row[k_new] = row.pop(k_old)

    def sort_values(self, col):
        self.rows = sorted(self.rows, key=lambda x: x[col])

    def drop(self, cols, type="soft"):
        if type == "soft":
            for col in cols:
                self.cols.remove(col)

    def to_gzip(self, path, delimiter="|"):
        with gzip.open(path, "wt") as f:
            f.write(delimiter.join(self.cols) + "\n")
            for row in self.rows:
                f.write(delimiter.join([row[col] for col in self.cols]) + "\n")

    def __setitem__(self, col, val):
        sz = len(self.rows)
        if col not in self.cols:
            self.cols.append(col)

        if isinstance(val, list):
            if sz != len(val):
                raise RuntimeError("Error: set value length not match!", sz, len(val), val)

            for i in range(sz):
                self.rows[i][col] = val[i]
        else:
            for i in range(sz):
                self.rows[i][col] = val

    def __getitem__(self, col):
        if isinstance(self.rows, list):
            return [r[col] for r in self.rows]
        else:
            return [r[col] for r in self.rows.values()]


def next_datetm(datetm):
    datetm = str(datetm)
    yyyy = int(datetm[:4])
    mm = int(datetm[4:6])
    dd = int(datetm[6:8])
    hh = int(datetm[8:10])
    hh += 1
    if hh == 24:
        hh = 0

        today_datetm = datetime.datetime(yyyy, mm, dd, hh)
        nextday_datetm = today_datetm + datetime.timedelta(1)
        return int(nextday_datetm.strftime("%Y%m%d%H"))
    else:
        today_datetm = datetime.datetime(yyyy, mm, dd, hh)
        return int(today_datetm.strftime("%Y%m%d%H"))


def get_datetms(start_datetm, end_datetm):
    ret_tms = []
    tm = start_datetm
    while tm <= end_datetm:
        ret_tms.append(tm)
        tm = next_datetm(tm)
    return ret_tms


def cmp_cache(mod1, mod2, dis=100000, iis=100000, verbose=False):
    univ_sz = min(mod1.shape[1], iis)
    dates_sz = min(mod1.shape[0], dis)
    a = mod1[:dates_sz, :univ_sz]
    b = mod2[:dates_sz, :univ_sz]

    aa = np.logical_xor(np.isfinite(mod1), np.isfinite(mod2))
    bb = np.logical_and(np.logical_and(np.isfinite(mod1), np.isfinite(mod2)), mod1 != mod2)
    # print('return [NaN not match], [val not match]')
    if verbose:
        return aa, bb
    else:
        return np.sum(aa), np.sum(bb)
