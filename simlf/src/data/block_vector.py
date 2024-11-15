from __future__ import annotations

from typing import Any, Optional

from data.array import Array


class BlockVector(object):
    def __init__(self, vec: Array, blocks: Array, path: str = None):
        assert vec.ndim == 1
        self.vec = vec
        self.blocks = blocks

    def __len__(self):
        return self.vec.shape[0]

    def __getitem__(self, i) -> Any:
        return self.vec[i]

    @property
    def path(self) -> str:
        return self.vec.path

    @property
    def num_blocks(self) -> int:
        return len(self.blocks)

    def block_begin(self, bi: int) -> int:
        return self.blocks[bi - 1] if bi > 0 else 0

    def block_end(self, bi: int) -> int:
        return self.blocks[bi]

    def block_range(self, bi: int) -> range:
        return range(self.block_begin(bi), self.block_end(bi))

    def block_size(self, bi: int) -> int:
        begin = self.block_begin(bi)
        end = self.block_end(bi)
        if end > begin:
            return end - begin
        return 0

    @staticmethod
    def mmap(path: str) -> BlockVector:
        return BlockVector(Array.mmap(path), Array.mmap(path + ".blocks"), path=path)

    def save(self, path: str = None) -> None:
        self.vec.save(self.path)
        self.blocks.save(self.path + ".blocks")
