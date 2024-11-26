import logging

import nu
import numpy as np
from numba import njit


@njit()
def _ffill_smooth(a: np.array, horizon: int) -> None:
    assert len(a.shape) == 3

    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            last_valid = 0
            if not np.isfinite(a[i, j, 0]):
                a[i, j, 0] = 0
            for t in range(1, a.shape[2]):
                if np.isfinite(a[i, j, t]):
                    last_valid = t
                elif horizon < 0 or last_valid + horizon <= t:
                    a[i, j, t] = a[i, j, t - 1]
                else:
                    a[i, j, t] = 0


def ffill_smooth(a: np.array, axis: int = -1, horizon: int = -1) -> None:
    assert len(a.shape) == 3
    if axis != -1 and axis != len(a.shape) - 1:
        a = a.swapaxes(-1, axis)
    _ffill_smooth(a, horizon)


def move_axis_inner(a: np.array, axis: int) -> np.array:
    # optimize memory access pattern
    if np.argmin(a.strides) != axis:
        a = np.copy(a.swapaxes(axis, -1), order="c")
        return a.swapaxes(axis, -1)
    else:
        return a


class Preprocessor(object):
    def fit(self, data: np.array) -> None:
        pass

    def transform(self, data: np.array) -> np.array:
        return data

    def inverse_transform(self, data: np.array) -> np.array:
        raise RuntimeError("Unsupported")

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state_dict: dict) -> None:
        pass

    def requires_fit(self) -> bool:
        return False


class CsStdScaler(Preprocessor):
    def __init__(self, axis: int = 0, center: bool = False):
        self.axis = axis
        self.center = center

    def transform(self, data: np.array) -> np.array:
        data = move_axis_inner(data, self.axis)
        if self.axis != 0:
            data = data.swapaxes(0, self.axis)
        stds = nu.nanstd(data, axis=0, ddof=0)
        data /= stds
        if self.center:
            means = nu.nanmean(data, axis=0)
            data -= means
        if self.axis != 0:
            data = data.swapaxes(0, self.axis)
        return data

    def state_dict(self) -> dict:
        return dict(axis=self.axis, center=self.center)

    def load_state_dict(self, state_dict: dict) -> None:
        self.axis = state_dict["axis"]
        self.center = state_dict["center"]


class FFillSmoother(Preprocessor):
    def __init__(self, axis: int = -1, horizon: int = -1):
        self.axis = axis
        self.horizon = horizon

    def transform(self, data: np.array) -> np.array:
        data = move_axis_inner(data, self.axis)
        ffill_smooth(data, axis=self.axis, horizon=self.horizon)
        return data

    def state_dict(self) -> dict:
        return dict(axis=self.axis, horizon=self.horizon)

    def load_state_dict(self, state_dict: dict) -> None:
        self.axis = state_dict["axis"]
        self.horizon = state_dict["horizon"]


class StdScaler(Preprocessor):
    def __init__(self, axis: int = 0, center: bool = False):
        self.axis = axis
        self.center = center
        self.stds = None
        self.means = None

    def fit(self, data: np.array) -> None:
        if self.axis != 0:
            data = data.swapaxes(0, self.axis)
        self.stds = nu.nanstd(data, axis=0, ddof=1)
        if self.center:
            self.means = nu.nanmean(data, axis=0)

    def transform(self, data: np.array) -> np.array:
        data = move_axis_inner(data, self.axis)
        if self.axis != 0:
            data = data.swapaxes(0, self.axis)
        data /= self.stds
        if self.center:
            data -= self.means
        if self.axis != 0:
            data = data.swapaxes(0, self.axis)
        return data

    def inverse_transform(self, data: np.array) -> np.array:
        if self.axis != 0:
            data = data.swapaxes(0, self.axis)
        if self.center:
            data += self.means
        data *= self.stds
        if self.axis != 0:
            data = data.swapaxes(0, self.axis)
        return data

    def state_dict(self) -> dict:
        return dict(axis=self.axis, center=self.center, stds=self.stds, means=self.means)

    def load_state_dict(self, state_dict: dict) -> None:
        self.axis = state_dict["axis"]
        self.center = state_dict["center"]
        self.stds = state_dict["stds"]
        self.means = state_dict["means"]

    def requires_fit(self) -> bool:
        return True


class FillNA(Preprocessor):
    def __init__(self, fill_value: float = 0):
        self.fill_value = fill_value

    def transform(self, data: np.array) -> np.array:
        data[~np.isfinite(data)] = self.fill_value
        return data

    def state_dict(self) -> dict:
        return dict(fill_value=self.fill_value)

    def load_state_dict(self, state_dict: dict) -> None:
        self.fill_value = state_dict["fill_value"]


class CsDemean(Preprocessor):
    def __init__(self, axis: int = 0):
        self.axis = axis

    def transform(self, data: np.array) -> np.array:
        data = move_axis_inner(data, self.axis)
        if self.axis != 0:
            data = data.swapaxes(0, self.axis)
        means = nu.nanmean(data, axis=0)
        data -= means
        if self.axis != 0:
            data = data.swapaxes(0, self.axis)
        return data

    def state_dict(self) -> dict:
        return dict(axis=self.axis)

    def load_state_dict(self, state_dict: dict) -> None:
        self.axis = state_dict["axis"]


def make_preprocesor(type: str, axis_map: dict[str, int], **kwargs) -> Preprocessor:
    if "axis" in kwargs:
        axis = kwargs["axis"]
        if isinstance(axis, str):
            if axis not in axis_map:
                raise RuntimeError("Unknown axis " + axis)
            kwargs["axis"] = axis_map[axis]
    return globals()[type](**kwargs)


class Pipeline(Preprocessor):
    processors: list[tuple[str, Preprocessor]]

    def __init__(self, config: list[dict], axis_names: list[int] = []):
        self.processors = []
        axis_map = {a: i for i, a in enumerate(axis_names)}
        for i, proc_config in enumerate(config):
            proc = make_preprocesor(axis_map=axis_map, **proc_config)
            self.processors.append((proc_config["type"], proc))

    def requires_fit(self) -> bool:
        return any(p.requires_fit() for _, p in self.processors)

    def fit(self, data: np.array) -> None:
        for _, p in self.processors:
            p.fit(data)

    def transform(self, data: np.array) -> np.array:
        for _, p in self.processors:
            data = p.transform(data)
        return data

    def inverse_transform(self, data: np.array) -> np.array:
        for _, p in reversed(self.processors):
            data = p.inverse_transform(data)
        return data

    def state_dict(self) -> dict:
        return dict(processors=[(t, p.state_dict()) for t, p in self.processors])

    def load_state_dict(self, state_dict: dict) -> None:
        for i in range(len(self.processors)):
            self.processors[i][1].load_state_dict(state_dict["processors"][i][1])
