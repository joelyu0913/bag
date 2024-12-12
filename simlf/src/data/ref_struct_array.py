from typing import Any, Optional

import numpy as np

from data.array import Array
from data.block_struct_vector import BlockStructVector
from data.null import get_null_value


class RefFieldArray(object):
    def __init__(self, values: Array, ref: Array, null_value: Any):
        self.values = values.data
        self.ref = ref.data
        self.null_value = null_value

    def __len__(self):
        return len(self.ref)

    @property
    def shape(self):
        return self.ref.shape

    @property
    def ndim(self) -> int:
        return self.ref.ndim

    def __getitem__(self, i):
        indices = self.ref[i]
        if indices.ndim == 0:
            return self.get_value(indices)
        else:
            values = np.empty(indices.shape + self.values.shape[1:], dtype=self.values.dtype)
            flat_indices = indices.reshape((-1,))
            flat_values = values.reshape((-1,) + self.values.shape[1:])
            for i in range(len(flat_indices)):
                flat_values[i] = self.get_value(flat_indices[i])
            return values

    def get_value(self, idx: int) -> Any:
        if idx >= 0:
            return self.values[idx]
        else:
            return self.null_value


class RefStructArray(object):
    def __init__(self, data: BlockStructVector, ref: Array):
        self.data = data
        self.ref = ref

    def __len__(self):
        return len(self.ref)

    @property
    def num_fields(self) -> int:
        return self.data.num_fields

    def field(self, i: Any, null_value: Any = None) -> Array:
        values = self.data.field(i)
        if null_value is None:
            if values.ndim == 1:
                null_value = get_null_value(values.dtype)
            else:
                null_value = np.full(values.shape[1:], get_null_value(values.dtype))
        return RefFieldArray(values, self.ref, null_value)

    def find_field(self, name: str) -> int:
        return self.data.find_field(name)

    @property
    def field_names(self) -> list[str]:
        return self.data.field_names

    @property
    def columns(self) -> list[str]:
        return self.vec.field_names

    @property
    def shape(self):
        return self.ref.shape

    @property
    def ndim(self) -> int:
        return self.ref.ndim

    @property
    def path(self) -> str:
        return self.data.path[-5:]

    @staticmethod
    def mmap(path: str, fields: Optional[list[str]] = None):
        return RefStructArray(
            BlockStructVector.mmap(path + ".data", fields), Array.mmap(path + ".ref")
        )
