from collections.abc import Callable
from typing import Any, Union


class DataCache(object):
    data: dict[str, Any]

    def __init__(self):
        self.data = {}

    def get_or_make(self, name: str, f: Callable[[str], Any]) -> Any:
        v = self.data.get(name, None)
        if v is not None:
            return v
        v = f()
        self.data[name] = v
        return v

    def clear(self) -> None:
        self.data.clear()
