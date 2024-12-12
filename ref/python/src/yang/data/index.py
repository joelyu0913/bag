from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import numpy as np


class Index(object):
    def __init__(self, items: list[Any], path: str = None):
        self.items = list(items)
        self.path = path
        self.idx = {item: i for i, item in enumerate(self.items)}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i) -> Any:
        return self.items[i]

    def __iter__(self):
        return iter(self.items)

    def find(self, key: Any) -> int:
        return self.idx.get(key, -1)

    @classmethod
    def load(self, path: str, item_type: Callable[[str], Any] = str) -> Index:
        with open(path) as f:
            items = [item_type(l) for l in f]
        return Index(items, path)


class IntIndex(object):
    def __init__(self, items: np.array, path: str = None):
        self.items = items
        self.path = path

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i) -> int:
        return self.items[i]

    def __iter__(self):
        return iter(self.items)

    def find(self, key: int) -> int:
        idx = np.searchsorted(self.items, key)
        if idx < len(self.items) and self.items[idx] == key:
            return idx
        return -1

    def lower_bound(self, key: int, inclusive: bool = True) -> int:
        # inclusive: a[i-1] < v <= a[i]
        # exclusive: a[i] < v <= a[i + 1]
        idx = np.searchsorted(self.items, key, side="left")
        if not inclusive and idx < len(self.items) and self.items[idx] == key:
            idx -= 1
        return idx

    def upper_bound(self, key: int, inclusive: bool = False) -> int:
        # exclusive: a[i-1] <= v < a[i]
        # inclusive: a[i] <= v < a[i + 1]
        idx = np.searchsorted(self.items, key, side="right")
        if inclusive and idx > 0 and self.items[idx - 1] == key:
            idx -= 1
        return idx

    def greater_than(self, key: int) -> int:
        return self.upper_bound(key)

    def greater_equal_than(self, key: int) -> int:
        return self.lower_bound(key)

    def less_than(self, key: int) -> int:
        idx = np.searchsorted(self.items, key, side="left")
        if idx == len(self.items) or self.items[idx] >= key:
            idx -= 1
        return idx

    def less_equal_than(self, key: int) -> int:
        idx = np.searchsorted(self.items, key, side="left")
        if idx == len(self.items) or self.items[idx] > key:
            idx -= 1
        return idx

    @classmethod
    def load(self, path: str) -> IntIndex:
        with open(path) as f:
            items = np.array([int(l) for l in f], dtype=np.int64)
        return IntIndex(items, path)


DateTimeIndex = IntIndex
DateIndex = IntIndex
TimeIndex = IntIndex


class UnivIndex(Index):
    def __init__(
        self,
        items: list[str],
        list_dis: np.array,
        indices: list[str],
        index_id_start: int,
        path: str = None,
    ):
        Index.__init__(self, items, path)
        self.list_dis = list_dis
        self.indices = indices
        self.index_id_start = index_id_start
        for i, symbol in enumerate(self.indices):
            self.idx[symbol] = i + index_id_start

    def find_on(self, di: int, key: str) -> int:
        idx = self.find(key)
        if idx >= 0 and (idx >= self.index_id_start or self.list_dis[idx] <= di):
            return idx
        return -1

    def __getitem__(self, idx) -> Any:
        if isinstance(idx, int):
            if idx < self.index_id_start:
                return self.items[idx]
            else:
                return self.indices[idx - self.index_id_start]
        else:
            return self.items[idx]

    @classmethod
    def load(self, path: str) -> UnivIndex:
        univ = []
        list_dis = []
        with open(path) as f:
            for line in f:
                symbol, date = line.split(" ")
                univ.append(symbol)
                list_dis.append(int(date))
        indices_path = path + ".indices"
        if os.path.exists(indices_path):
            with open(indices_path) as f:
                index_id_start = int(f.readline())
                indices = [l.strip() for l in f]
        else:
            index_id_start = 10000
            indices = []
        return UnivIndex(univ, np.array(list_dis, dtype=np.int32), indices, index_id_start, path)
