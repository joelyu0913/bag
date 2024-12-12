from __future__ import annotations

from typing import Any, Optional

from yang.data.array import Array
from yang.data.struct_array import StructArray


class BlockStructVector(object):
    def __init__(self, vec: StructArray, blocks: Array):
        assert vec.ndim == 1
        self.vec = vec
        self.blocks = blocks
        for i in range(len(self.vec.fields)):
            self.__setattr__(self.field_names[i], self.vec.fields[i])

    def __len__(self):
        return self.vec.shape[0]

    @property
    def num_fields(self) -> int:
        return self.vec.num_fields

    def field(self, i: Any) -> Array:
        return self.vec.field(i)

    def find_field(self, name: str) -> int:
        return self.vec.find_field(name)

    @property
    def field_names(self) -> list[str]:
        return self.vec.field_names

    @property
    def columns(self) -> list[str]:
        return self.vec.field_names

    def to_xarray(self):
        return self.vec.to_xarray()

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

    @staticmethod
    def mmap(path: str, fields: Optional[list[str]] = None) -> BlockStructVector:
        return BlockStructVector(StructArray.mmap(path, fields), Array.mmap(path + ".blocks"))
