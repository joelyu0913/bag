#cython: language_level=3

from libcpp cimport bool

from typing import Any, Optional
import numpy as np

cdef extern from "python/src/yang/math/np_ufuncs.h" namespace "yang::math::ops" nogil:
    object get_rank_ufunc()
    object get_demean_ufunc(bool)
    object get_group_demean_ufunc(bool)
    object get_filter_ufunc()
    object get_hedge_ufunc()
    object get_scale_ufunc()
    object get_ewa_ufunc(bool)
    object get_mean_ufunc(bool)
    object get_variance_ufunc(bool)
    object get_stdev_ufunc(bool)
    object get_sum_ufunc(bool)

rank = get_rank_ufunc()

demean = get_demean_ufunc(True)
demean_unchecked = get_demean_ufunc(False)

group_demean  = get_group_demean_ufunc(True)
group_demean_unchecked = get_group_demean_ufunc(False)

filter = get_filter_ufunc()

_hedge = get_hedge_ufunc()
def hedge(arr: Any, univ_size: int, hedge_idx: int, out: Optional[np.array] = None,
          **kwargs) -> np.array:
    return _hedge(arr, univ_size, hedge_idx, out=out, **kwargs)

_scale = get_scale_ufunc()
def scale(arr: Any, scale_size: float, eps: float = 0, out: Optional[np.array] = None,
          **kwargs) -> np.array:
    return _scale(arr, scale_size, eps, out=out, **kwargs)

_ewa = get_ewa_ufunc(True)
def ewa(arr: Any, ratio: float, out: Optional[np.array] = None, **kwargs) -> np.array:
    return _ewa(arr, ratio, out=out, **kwargs)

_ewa_unchecked = get_ewa_ufunc(False)
def ewa_unchecked(arr: Any, ratio: float, out: Optional[np.array] = None, **kwargs) -> np.array:
    return _ewa_unchecked(arr, ratio, out=out, **kwargs)

mean = get_mean_ufunc(True)
mean_unchecked = get_mean_ufunc(False)
sum = get_sum_ufunc(True)
sum_unchecked = get_sum_ufunc(False)

variance = get_variance_ufunc(True)
variance_unchecked = get_variance_ufunc(False)
var = variance
var_unchecked = variance_unchecked

stdev = get_stdev_ufunc(True)
stdev_unchecked = get_stdev_ufunc(False)
std = stdev
std_unchecked = stdev_unchecked
