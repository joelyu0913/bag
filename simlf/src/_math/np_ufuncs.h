#pragma once

#include <Python.h>

namespace yang::math::ops {

PyObject *get_rank_ufunc();
PyObject *get_demean_ufunc(bool check_valid);
PyObject *get_group_demean_ufunc(bool check_valid);
PyObject *get_filter_ufunc();
PyObject *get_hedge_ufunc();
PyObject *get_scale_ufunc();
PyObject *get_ewa_ufunc(bool check_valid);

PyObject *get_mean_ufunc(bool check_valid);
PyObject *get_sum_ufunc(bool check_valid);
PyObject *get_variance_ufunc(bool check_valid);
PyObject *get_stdev_ufunc(bool check_valid);

}  // namespace yang::math::ops
