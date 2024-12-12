from __future__ import annotations

import os
from typing import Any

from data.array import Array
from data.block_vector import BlockVector

"""
mat = BlockMat.mmap(path)
for di in range(start_di, end_di):
    if not mat.load_mat(di): continue
    for idx_ii in range(mat.mat_size(di)):
        ii = mat.get_id(di, idx_ii)
        for idx_jj in range(mat.mat_size(di)):
            if idx_ii == idx_jj: continue
        jj = mat.get_id(di, idx_jj)
        val = mat[idx_ii, idx_jj]
"""


class BlockMat(object):
    def __init__(self, base_path: str, ids: BlockVector):
        self.base_path = base_path
        self.ids = ids
        self.mat = None

    def mat_size(self, bi: int) -> int:
        return self.ids.block_size(bi)

    def get_id(self, bi: int, offset: int) -> int:
        return self.ids[int(self.ids.block_begin(bi)) + offset]

    def load_mat(self, bi: int) -> bool:
        path = self.mat_path(bi)
        if os.path.exists(path):
            self.mat = Array.mmap(path)
            return True
        else:
            self.mat = None
            return False

    def mat_path(self, bi: int) -> str:
        return os.path.join(self.base_path + ".mat", str(bi))

    def __getitem__(self, i) -> Any:
        return self.mat[i]

    @staticmethod
    def mmap(path: str) -> BlockMat:
        return BlockMat(path, BlockVector.mmap(path + ".id"))
