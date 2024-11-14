import nu.nan
import numpy as np
from numba import njit


@njit()
def std_scale(data: np.array, blocks: np.array, center: bool) -> np.array:
    for i in range(len(blocks) - 1):
        for fi in range(data.shape[1]):
            idx1 = blocks[i]
            idx2 = blocks[i + 1]
            std = nu.nan._nanstd(data[idx1:idx2, fi], 0)
            data[idx1:idx2, fi] /= std
            if center:
                m = nu.nan._nanmean(data[idx1:idx2, fi])
                data[idx1:idx2, fi] -= m
    return data


class Preprocessor(object):
    def fit(self, data: np.array) -> None:
        pass

    def transform(self, data: np.array, blocks: np.array) -> np.array:
        return data

    def inverse_transform(self, data: np.array) -> np.array:
        return data

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state_dict: dict) -> None:
        pass

    def requires_fit(self) -> bool:
        return False


class CsStdScaler(Preprocessor):
    def __init__(self, center: bool = False):
        self.center = center

    def transform(self, data: np.array, blocks: np.array) -> np.array:
        return std_scale(data, blocks, self.center)

    def state_dict(self) -> dict:
        return dict(center=self.center)

    def load_state_dict(self, state_dict: dict) -> None:
        self.center = state_dict["center"]


class FillNA(Preprocessor):
    def __init__(self, fill_value: float = 0):
        self.fill_value = fill_value

    def transform(self, data: np.array, blocks: np.array) -> np.array:
        data[~np.isfinite(data)] = self.fill_value
        return data

    def state_dict(self) -> dict:
        return dict(fill_value=self.fill_value)

    def load_state_dict(self, state_dict: dict) -> None:
        self.fill_value = state_dict["fill_value"]


def make_preprocesor(type: str, **kwargs) -> Preprocessor:
    return globals()[type](**kwargs)


class Pipeline(Preprocessor):
    processors: list[tuple[str, Preprocessor]]

    def __init__(self, config: list[dict]):
        self.processors = []
        for i, proc_config in enumerate(config):
            proc = make_preprocesor(**proc_config)
            self.processors.append((proc_config["type"], proc))

    def requires_fit(self) -> bool:
        return False

    def fit(self, data: np.array) -> None:
        pass

    def transform(self, data: np.array, blocks: np.array) -> np.array:
        for _, p in self.processors:
            data = p.transform(data, blocks)
        return data

    def inverse_transform(self, data: np.array) -> np.array:
        return data

    def state_dict(self) -> dict:
        return dict(processors=[(t, p.state_dict()) for t, p in self.processors])

    def load_state_dict(self, state_dict: dict) -> None:
        for i in range(len(self.processors)):
            self.processors[i][1].load_state_dict(state_dict["processors"][i][1])
