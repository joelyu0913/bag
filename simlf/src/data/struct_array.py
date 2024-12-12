from __future__ import annotations
from data.array import Array
from typing import Union, Optional
import yaml


class StructArrayMeta(object):
    def __init__(self, shape: tuple, fields: list[str]):
        self.shape = shape
        self.fields = fields

    def save(self, path: str) -> None:
        data = {
            "shape": self.shape,
            "fields": self.fields,
        }
        with open(path, "w") as f:
            yaml.safe_dump(data, f)

    @staticmethod
    def load(path: str) -> StructArrayMeta:
        with open(path) as f:
            raw_meta = yaml.safe_load(f)
        return StructArrayMeta(tuple(raw_meta["shape"]), raw_meta["fields"])


# TODO: writable
class StructArray(object):
    def __init__(
        self, fields: list[Array], field_names: list[str], meta: StructArrayMeta, path: str
    ):
        self.fields = fields
        self.field_names = field_names
        self.meta = meta
        self.path = path
        self.field_idx = {f: i for i, f in enumerate(field_names)}
        for i in range(len(self.fields)):
            self.__setattr__(self.field_names[i], self.fields[i])

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def shape(self) -> tuple:
        return self.meta.shape

    @property
    def num_fields(self) -> int:
        return len(self.fields)

    def field(self, i: Union[int, str]) -> Array:
        if isinstance(i, int):
            return self.fields[i]
        elif i in self.field_idx:
            return self.fields[self.field_idx[i]]
        else:
            raise RuntimeError(f"Unknown field: {i}")

    def find_field(self, name: str) -> int:
        return self.field_idx.get(name, -1)

    @property
    def columns(self) -> list[str]:
        return self.field_names

    def to_xarray(self):
        import xarray as xr

        fields = {
            self.columns[i]: xr.DataArray(self.fields[i].data) for i in range(len(self.fields))
        }
        return xr.Dataset(fields)

    @staticmethod
    def mmap(path: str, fields: Optional[list[str]] = None) -> StructArray:
        meta_path = path + ".meta"
        meta = StructArrayMeta.load(meta_path)
        if fields is None:
            fields = meta.fields
        field_names = fields
        fields = [Array.mmap(f"{path}._{f}") for f in field_names]
        return StructArray(fields, field_names, meta, path)
