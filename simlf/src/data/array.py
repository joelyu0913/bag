from __future__ import annotations

import logging
import os
from typing import Any, Optional

import numpy as np
import yaml

from data.null import get_null_value

dtype_map = {
    "float": np.dtype(np.float32),
    "double": np.dtype(np.float64),
    "bool": np.dtype(np.bool8),
    "int16": np.dtype(np.int16),
    "int32": np.dtype(np.int32),
    "int64": np.dtype(np.int64),
    "uint16": np.dtype(np.uint16),
    "uint32": np.dtype(np.uint32),
    "uint64": np.dtype(np.uint64),
    "int": np.dtype(np.int32),
}
meta_type_map = {np.dtype(v).name: k for k, v in dtype_map.items()}


class ArrayMeta(object):
    def __init__(self, item_type: np.dtype, item_size: int, shape: tuple):
        self.item_type = item_type
        self.item_size = item_size
        self.shape = shape

    def save(self, path: str) -> None:
        data = {
            "item_type": meta_type_map[self.item_type.name],
            "item_size": self.item_size,
            "shape": list(self.shape),
        }
        with open(path, "w") as f:
            yaml.safe_dump(data, f)

    @staticmethod
    def load(path: str) -> ArrayMeta:
        with open(path) as f:
            raw_meta = yaml.safe_load(f)
        cpp_type = raw_meta["item_type"]
        if cpp_type.startswith("std::array"):
            # std::array<T, N>
            etype, n = cpp_type[11:-1].split(", ")
            dtype = np.dtype(f"{n}{dtype_map[etype]}")
        else:
            dtype = dtype_map[cpp_type]
        return ArrayMeta(dtype, raw_meta["item_size"], tuple(raw_meta["shape"]))


# TODO: resize, clone
class Array(object):
    def __init__(self, data: np.array, path: str = None, null_value: Any = None):
        self.data = data
        self.path = path
        self.null_value = null_value

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i) -> Any:
        return self.data[i]

    def __setitem__(self, i, value) -> None:
        self.data[i] = value

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def shape(self) -> tuple:
        return self.data.shape

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    def save(self, path: str = None) -> None:
        if path is not None:
            self.path = path
        assert self.path is not None
        dir_path = os.path.dirname(self.path)
        if dir_path != "":
            os.makedirs(dir_path, exist_ok=True)
        meta = ArrayMeta(self.data.dtype, self.data.itemsize, self.data.shape)
        meta.save(self.path + ".meta")
        self.data.tofile(self.path)

    @staticmethod
    def mmap(
        path: str,
        writable: bool = False,
        shape: Optional[tuple[int]] = None,
        dtype: Optional[np.dtype] = None,
        null_value: Any = None,
    ) -> Array:
        meta_path = path + ".meta"
        if writable:
            if dtype is not None:
                dtype = np.dtype(dtype)
            if os.path.exists(meta_path):
                # update existing
                old_meta = ArrayMeta.load(meta_path)
                if dtype is None:
                    dtype = old_meta.item_type
                else:
                    assert (
                        dtype == old_meta.item_type
                    ), f"array dtype mismatch, new:{dtype} old:{old_meta.item_type}"
                if shape is None:
                    shape = old_meta.shape
                else:
                    assert len(shape) == len(old_meta.shape)
            else:
                dir_path = os.path.dirname(path)
                if dir_path != "":
                    os.makedirs(dir_path, exist_ok=True)
                old_meta = None
            if null_value is None:
                null_value = get_null_value(dtype)
            if old_meta is None:
                data = np.memmap(path, dtype=dtype, mode="w+", shape=shape)
                data.fill(null_value)
                logging.debug(f"Created {path} {data.shape}")
            elif shape is None or old_meta.shape == shape:
                data = np.memmap(path, dtype=dtype, mode="r+", shape=shape)
                logging.debug(f"Loaded {path} {data.shape}")
            elif old_meta.shape[1:] == shape[1:]:
                if old_meta.shape[0] > shape[0]:
                    os.truncate(path, dtype.itemsize * np.product(shape))
                data = np.memmap(path, dtype=dtype, mode="r+", shape=shape)
                data[old_meta.shape[0] :, ...].fill(null_value)
                logging.debug(f"Resized {path} {old_meta.shape} -> {data.shape}")
            else:
                old_data = np.fromfile(path, dtype=dtype).reshape(old_meta.shape)
                data = np.memmap(path, dtype=dtype, mode="w+", shape=shape)
                data.fill(null_value)
                slices = tuple(slice(0, min(n1, n2)) for n1, n2 in zip(old_data.shape, shape))
                data[slices] = old_data[slices]
                logging.debug(f"Resized {path} {old_data.shape} -> {data.shape}")
            if not old_meta or old_meta.shape != shape:
                meta = ArrayMeta(dtype, dtype.itemsize, shape)
                meta.save(meta_path)
        else:
            # read
            mode = "r"
            meta = ArrayMeta.load(meta_path)
            data = np.memmap(path, dtype=meta.item_type, mode=mode, shape=meta.shape)
            logging.debug(f"Loaded {path} {data.shape}")
        return Array(data, path, null_value)

    @staticmethod
    def load(path: str) -> Array:
        meta_path = path + ".meta"
        meta = ArrayMeta.load(meta_path)
        data = np.fromfile(path, dtype=meta.item_type).reshape(meta.shape)
        return Array(data, path)
