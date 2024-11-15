from typing import Any
import numpy as np


def get_null_value(dtype: np.dtype) -> Any:
    if dtype.ndim > 0:
        return np.full(dtype.shape, get_null_value(dtype.base), dtype=dtype.base)

    if np.issubdtype(dtype, np.floating):
        return np.NAN
    if np.issubdtype(dtype, np.bool8):
        return False
    if np.issubdtype(dtype, np.signedinteger):
        return np.iinfo(dtype).min
    if np.issubdtype(dtype, np.unsignedinteger):
        return np.iinfo(dtype).max
    return None
