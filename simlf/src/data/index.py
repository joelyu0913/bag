from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

UNKNOWN_LIST_DI = 10000000

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
    
    def insert(self, key: Any) -> int:
        if key in self.idx:
            return self.idx[key]
        self.idx[key] = len(self.items)
        self.items.append(key)
        return len(self.items) - 1

    @classmethod
    def load(self, path: str, item_type: Callable[[str], Any] = str) -> Index:
        if not os.path.exists(path):
            return Index([], path)
        else:
            with open(path) as f:
                items = [item_type(l) for l in f]
            return Index(items, path)
    
#       virtual void Save() {
#     fs::create_directories(fs::path(path_).parent_path());
#     std::ofstream ofs(path_);
#     for (auto &x : items_) ofs << x << '\n';
#     ofs.flush();
#     ENSURE(ofs.good(), "Failed to save {}", path_);
#   }
    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, 'w') as wf:
            for item in self.items:
                wf.write(f'{item}\n')



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
        if os.path.exists(path):
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

    def set_indices(self, indices, id_start):
        self.index_id_start = id_start
        for idx in self.indices: 
            del self.idx[idx]
        self.indices = indices
        for i in range(len(self.indices)):
            self.idx[self.indices[i]] = i + self.index_id_start
    
    
    def max_id(self): 
        return self.index_id_start + len(self.indices) - 1

    def get_or_insert(self, di, symbol):
        if symbol in self.idx:
            ii = self.idx[symbol]
            old_di = self.list_dis[ii]
            assert (old_di <= di or old_di == UNKNOWN_LIST_DI), f"list date changed, symbol: {symbol}, old_di: {old_di}, new_di: {di}"
            if old_di == UNKNOWN_LIST_DI: 
                self.list_dis[ii] = di
            return ii
        self.idx[symbol] = len(self.items)
        self.items.append(symbol)
        self.list_dis = np.append(self.list_dis, di)
        assert len(self.items) < self.index_id_start, "UnivIndex size exceeds index_id_start"
        return len(self.items) - 1

    def save(self, path):
        pd.DataFrame({0:self.items, 1:self.list_dis}).to_csv(path, sep=' ', header=False, index=False)

        if (len(self.indices) > 0):
            indices_path = path + ".indices"
            with open(indices_path, 'w') as wf:
                wf.write(f'{self.index_id_start}\n')
                for idx in self.indices:
                    wf.write(f'{idx}\n')

   
