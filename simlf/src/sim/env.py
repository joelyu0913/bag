from __future__ import annotations

import os
from collections import deque
from enum import Enum
from typing import Any

import numpy as np
import yaml
import logging

from data import DataCache, DateTimeIndex, TimeIndex, UnivIndex
from sim.data_directory import DataDirectory
from sim.rerun_manager import RerunManager

DISPLAY_BOOK_SIZE = 100000


class RunStage(Enum):
    ERROR = 0
    PREPARE = 1
    OPEN = 2
    INTRADAY = 3
    EOD = 5

    @staticmethod
    def parse(s: str) -> RunStage:
        try:
            return RunStage[s.upper()]
        except KeyError:
            return RunStage.ERROR

    def __str__(self):
        return self.name.lower()


def _upgrade_meta(meta: dict):
    if "daily" in meta:
        return
    meta["daily"] = True
    meta["univ_start_datetime"] = meta["univ_start_date"]
    meta["univ_end_datetime"] = meta["univ_end_date"]
    meta["taq_times"] = meta["intraday_times"]
    meta["intraday_times"] = []
    meta["days_per_year"] = 250
    meta["short_book_size"] = False
    meta["benchmark_index"] = "000905.SH"


class Env(object):
    def __init__(self, config: dict, verbose: bool = True):
        self.config = config
        if "user_cache" in config:
            if not config["sys_cache"]:
                raise RuntimeError("sys_cache missing")
            self.cache_dir = DataDirectory(config["user_cache"], config["sys_cache"])
            self.user_mode = True
        elif "cache" in config:
            self.cache_dir = DataDirectory(config["cache"])
            self.user_mode = False
        elif "sys_cache" in config:
            self.cache_dir = DataDirectory(config["sys_cache"])
            self.user_mode = False
        else:
            raise RuntimeError("cache config missing")
        assert os.path.exists(self.cache_dir.user_dir), "user cache dir does not exist"
        assert os.path.exists(self.cache_dir.sys_dir), "sys cache dir does not exist"
        self.data_cache = DataCache()

        if self.user_mode:
            self.rerun_manager = RerunManager(self.cache_dir.user_dir + "/_rerun")
        else:
            self.rerun_manager = RerunManager(self.cache_dir.sys_dir + "/_rerun")
        
        if verbose:
            logging.info(f'Running in {"user" if self.user_mode else "sys"} mode')

        meta_path = self.cache_dir.get_path("env", "meta.yml")
        with open(meta_path) as f:
            meta = yaml.safe_load(f)
        datetimes_path = "datetimes"
        _upgrade_meta(meta)
        self.univ = UnivIndex.load(self.cache_dir.get_path("env", "univ"))
        self.univ_size = len(self.univ)
        self.max_univ_size = meta["max_univ_size"]
        self.datetimes = DateTimeIndex.load(self.cache_dir.get_path("env", datetimes_path))
        self.datetimes_size = len(self.datetimes)

        ROUND = 64
        ROUND_MASK = ~(ROUND - 1)
        self.max_datetimes_size = (len(self.datetimes) + ROUND - 1) & ROUND_MASK

        self.daily = meta["daily"]

        self.univ_start_datetime = meta["univ_start_datetime"]
        self.univ_end_datetime = meta["univ_end_datetime"]
        if self.daily:
            self.sim_start_datetime = config.get("sim_start_date", self.univ_start_datetime)
            self.sim_end_datetime = config.get("sim_end_date", self.univ_end_datetime)
        else:
            self.sim_start_datetime = config.get("sim_start_datetime", self.univ_start_datetime)
            self.sim_end_datetime = config.get("sim_end_datetime", self.univ_end_datetime)

        self.live = False
        self.prod = False
        self.start_dti = self.datetimes.lower_bound(self.sim_start_datetime)
        self.end_dti = self.datetimes.upper_bound(self.sim_end_datetime)
        self.intraday_times = TimeIndex(meta["intraday_times"])
        self.taq_times = TimeIndex(meta["taq_times"])
        self.days_per_year = meta["days_per_year"]
        self.short_book_size = meta["short_book_size"]
        self.benchmark_index = meta["benchmark_index"]
        self.rerun_manager.set_dates(int(self.datetimes[self.start_dti]), int(self.datetimes[self.end_dti - 1]))

    @property
    def dates(self):
        return self.datetimes

    @property
    def max_dates_size(self):
        return self.max_datetimes_size

    @property
    def dates_size(self):
        return self.datetimes_size

    @property
    def start_di(self):
        return self.start_dti

    @start_di.setter
    def start_di(self, di):
        self.start_dti = di

    @property
    def end_di(self):
        return self.end_dti

    @end_di.setter
    def end_di(self, di):
        self.end_dti = di

    def find_indx_id(self, idx) -> int:
        return self.univ.index_id_start + idx

    @staticmethod
    def load(path: str) -> Env:
        with open(path) as f:
            config = yaml.safe_load(f)
        return Env(config)

    def read_data(self, cls: type, mod: str, name: str = None) -> Any:
        def make_data():
            path = self.cache_dir.get_path(mod, name)
            if hasattr(cls, "mmap"):
                return cls.mmap(path)
            else:
                return cls.load(path)

        return self.data_cache.get_or_make(f"{mod}.{name}", make_data)

    @property
    def trade_book_size(self) -> int:
        return self.config.get("trade_book_size", DISPLAY_BOOK_SIZE)

    @property
    def univ_indices(self) -> UnivIndex:
        return self.univ.indices

    @property
    def intervals_per_day(self) -> int:
        return 1 if self.daily else len(self.intraday_times)
