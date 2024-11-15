import logging
from typing import Any, Optional

import numpy as np

from data import Array, DataCache, DateIndex, UnivIndex
from sim.data_directory import DataDirectory
from sim.env import Env, RunStage


class Module(object):
    def __init__(self, name: str, config: dict, env: Env):
        self.name = name
        self.config = config
        self.env = env
        self.stage = RunStage.INTRADAY

    @property
    def cache_dir(self) -> DataDirectory:
        return self.env.cache_dir

    @property
    def data_cache(self) -> DataCache:
        return self.env.data_cache

    @property
    def univ(self) -> UnivIndex:
        return self.env.univ

    @property
    def datetimes(self) -> DateIndex:
        return self.env.datetimes

    @property
    def datetimes_size(self) -> int:
        return self.env.datetimes_size

    @property
    def max_datetimes_size(self) -> int:
        return self.env.max_datetimes_size

    @property
    def dates(self) -> DateIndex:
        return self.env.dates

    @property
    def dates_size(self) -> int:
        return self.env.dates_size

    @property
    def max_dates_size(self) -> int:
        return self.env.max_dates_size

    @property
    def start_di(self) -> int:
        return self.env.start_di

    @property
    def end_di(self) -> int:
        return self.env.end_di

    @property
    def univ_size(self) -> int:
        return self.env.univ_size

    @property
    def max_univ_size(self) -> int:
        return self.env.max_univ_size

    @property
    def sys(self) -> bool:
        return self.config.get("sys", False)

    def run(self):
        self.before_run()
        self.run_impl()
        self.after_run()

    def before_run(self):
        pass

    def run_impl(self):
        pass

    def after_run(self):
        pass

    def write_array(
        self,
        mod: str,
        name: str = None,
        dtype=np.float32,
        shape: Optional[list[int]] = None,
        null_value: Any = None,
        fill_null: Optional[bool] = None,
    ) -> Array:
        if shape is None:
            shape = (self.max_dates_size, self.max_univ_size)
            if fill_null is None:
                fill_null = True
        arr = Array.mmap(
            self.cache_dir.get_write_path(mod, name),
            writable=True,
            dtype=dtype,
            shape=shape,
            null_value=null_value,
        )
        if fill_null:
            arr[self.start_di : self.end_di].fill(null_value)
        return arr

    def read_data(self, cls: type, mod: str, name: str = None) -> Any:
        return self.env.read_data(cls, mod, name)

    def read_array(self, mod: str, name: str = None) -> Any:
        return self.env.read_data(Array, mod, name)


class DemoModule(Module):
    def run_impl(self):
        logging.info(f"Run DemoModule ({self.start_di} - {self.end_di})")
