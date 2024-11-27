import numpy as np
from numba import float32, float64, guvectorize, int32, njit


@njit()
def _nansum(a):
    s = 0
    for i in range(len(a)):
        if np.isfinite(a[i]):
            s += a[i]
    return s


@njit()
def _nanmean(a):
    s = 0
    cnt = 0
    for i in range(len(a)):
        if np.isfinite(a[i]):
            s += a[i]
            cnt += 1
    return s / cnt


@njit()
def _nanvar(a, ddof=0):
    n = 0
    mean = _nanmean(a)
    var = 0
    for i in range(len(a)):
        if np.isfinite(a[i]):
            diff = a[i] - mean
            var += diff * diff
            n += 1
    if n < 1 + ddof:
        return np.nan
    return var / (n - ddof)


@njit(cache=True)
def _nanstd(a, ddof=0):
    return np.sqrt(_nanvar(a, ddof))


@guvectorize([(float32[:], float32[:]), (float64[:], float64[:])], "(n)->()", nopython=True)
def nansum(a, res):
    res[0] = _nansum(a)


@guvectorize([(float32[:], float32[:]), (float64[:], float64[:])], "(n)->()", nopython=True)
def nanmean(a, res):
    res[0] = _nanmean(a)


@guvectorize(
    [(float32[:], int32, float32[:]), (float64[:], int32, float64[:])], "(n), ()->()", nopython=True
)
def _nanvar_wrap(a, ddof, res):
    res[0] = _nanvar(a, ddof)


def nanvar(a, ddof=0, **kwargs):
    return _nanvar_wrap(a, ddof, **kwargs)


@guvectorize(
    [(float32[:], int32, float32[:]), (float64[:], int32, float64[:])], "(n), ()->()", nopython=True
)
def _nanstd_wrap(a, ddof, res):
    res[0] = _nanstd(a, ddof)


def nanstd(a, ddof=0, **kwargs):
    return _nanstd_wrap(a, ddof, **kwargs)
