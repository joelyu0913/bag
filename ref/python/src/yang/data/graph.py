from typing import Optional

import numpy as np

from yang.data.block_struct_vector import BlockStructVector
from yang.data.block_vector import BlockVector

"""
from yang.data import BlockStructVector, SparseGraph
v = BlockStructVector.mmap("tmp/cache_os/B_grf_mx_corr__10__504_5_cty_ret/grf")
grf = SparseGraph()
grf.clear()
grf.load(v, 1000)
"""


class SparseGraph(object):
    def __init__(self):
        self._out_edges = {}
        self._in_edges = {}
        self._weight_map = {}

    def out_edges(self, from_v: int) -> list[int]:
        return self._out_edges.get(from_v, [])

    def in_edges(self, to_v: int) -> list[int]:
        return self._in_edges.get(to_v, [])

    def weight(self, from_v: int, to_v: int) -> float:
        return self._weight_map.get((from_v, to_v), np.nan)

    def clear(self) -> None:
        self._out_edges.clear()
        self._in_edges.clear()
        self._weight_map.clear()

    def load(
        self,
        vec: BlockStructVector,
        bi: int,
        max_rank: int = -1,
        update: bool = True,
        weights: Optional[BlockVector] = None,
    ) -> None:
        from_arr = vec.field("from")
        to_arr = vec.field("to")
        if weights is None:
            weight_arr = vec.field("weight")
        else:
            weight_arr = weights.vec
        rank_arr = vec.field("rank")
        for i in vec.block_range(bi):
            if max_rank < 0 or rank_arr[i] <= max_rank:
                from_v = from_arr[i]
                to_v = to_arr[i]
                w = weight_arr[i]
                if (from_v, to_v) in self._weight_map:
                    if update:
                        self._weight_map[(from_v, to_v)] = w
                else:
                    self._weight_map[(from_v, to_v)] = w

                    out_edges = self._out_edges.get(from_v)
                    if out_edges is not None:
                        out_edges.append(to_v)
                    else:
                        self._out_edges[from_v] = [to_v]

                    in_edges = self._in_edges.get(to_v)
                    if in_edges is not None:
                        in_edges.append(from_v)
                    else:
                        self._in_edges[to_v] = [from_v]
