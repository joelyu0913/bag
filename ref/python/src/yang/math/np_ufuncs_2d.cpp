#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <numpy/ndarraytypes.h>
#include <numpy/ufuncobject.h>

#include "helpers.h"
#include "yang/math/mat_ops.h"
#include "yang/math/mat_view.h"

namespace yang::math::ops {

template <class... Us, size_t... Is, size_t... Js, class Func>
void apply2d_impl(char **args, const npy_intp *dimensions, const npy_intp *steps,
                  std::index_sequence<Is...>, std::index_sequence<Js...>, Func &&f) {
  auto n = dimensions[0];
  auto m = dimensions[1];
  auto cont_stride = (true && ... && (steps[Js] == 0 || steps[Js] == sizeof(Us)));
  if (cont_stride) {
    f(MatView<Us>(reinterpret_cast<Us *>(args[Is]), n, m, steps[Is] / sizeof(Us), 1)...);
  } else {
    f(MatView<Us, MatShape<>, MatStride<>>(reinterpret_cast<Us *>(args[Is]), n, m,
                                           steps[Is] / sizeof(Us), steps[Js] / sizeof(Us))...);
  }
}

template <class... Us, size_t... Js, class Func>
void apply2d(char **args, const npy_intp *dimensions, const npy_intp *steps,
             std::index_sequence<Js...> js, Func &&f) {
  apply2d_impl<Us...>(args, dimensions, steps, std::index_sequence_for<Us...>(), js, std::move(f));
}

template <class... Us, class Func>
void apply2d(char **args, const npy_intp *dimensions, const npy_intp *steps, Func &&f) {
  constexpr auto N = sizeof...(Us);
  auto seq = std::make_index_sequence<N>();
  apply2d_impl<Us...>(args, dimensions, steps, seq, add_to_index_sequence<N>(seq), std::move(f));
}

template <class T>
void np_rank(char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
  apply2d<T, T>(args, dimensions, steps, [](auto in_mat, auto out_mat) { rank(in_mat, out_mat); });
}

PyObject *get_rank_ufunc() {
  import_array();
  import_umath();
  static PyUFuncGenericFunction funcs[] = {&np_rank<float>, &np_rank<double>};
  static char types[] = {NPY_FLOAT, NPY_FLOAT, NPY_DOUBLE, NPY_DOUBLE};
  static void *data[] = {NULL, NULL};
  return PyUFunc_FromFuncAndDataAndSignature(funcs, data, types, 2, 1, 1, PyUFunc_None, "rank", "",
                                             0, "(i)->(i)");
}

template <class T, class ValidCheck>
void np_demean(char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
  apply2d<T, T>(args, dimensions, steps,
                [](auto in_mat, auto out_mat) { demean<ValidCheck>(in_mat, out_mat); });
}

template <class ValidCheck>
PyObject *get_demean_ufunc() {
  import_array();
  import_umath();
  static PyUFuncGenericFunction funcs[] = {&np_demean<float, ValidCheck>,
                                           &np_demean<double, ValidCheck>};
  static char types[] = {NPY_FLOAT, NPY_FLOAT, NPY_DOUBLE, NPY_DOUBLE};
  static void *data[] = {NULL, NULL};
  return PyUFunc_FromFuncAndDataAndSignature(funcs, data, types, 2, 1, 1, PyUFunc_None, "demean",
                                             "", 0, "(i)->(i)");
}

PyObject *get_demean_ufunc(bool check_valid) {
  if (check_valid) return get_demean_ufunc<DefaultCheck>();
  return get_demean_ufunc<CheckNothing>();
}

template <class T, class U, class ValidCheck>
void np_group_demean(char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
  apply2d<T, U, T>(args, dimensions, steps, [](auto in_mat, auto g_mat, auto out_mat) {
    group_demean<ValidCheck>(in_mat, g_mat, out_mat);
  });
}

template <class ValidCheck>
PyObject *get_group_demean_ufunc() {
  import_array();
  import_umath();
  // numpy cast python int to int64_t by default
  static PyUFuncGenericFunction funcs[] = {
      &np_group_demean<float, int32_t, ValidCheck>,
      &np_group_demean<double, int32_t, ValidCheck>,
      &np_group_demean<float, int64_t, ValidCheck>,
      &np_group_demean<double, int64_t, ValidCheck>,
  };
  // clang-format off
  static char types[] = {
      NPY_FLOAT,  NPY_INT32, NPY_FLOAT,
      NPY_DOUBLE, NPY_INT32, NPY_DOUBLE,
      NPY_FLOAT,  NPY_INT64, NPY_FLOAT,
      NPY_DOUBLE, NPY_INT64, NPY_DOUBLE,
  };
  // clang-format on
  static void *data[] = {NULL, NULL, NULL, NULL};
  return PyUFunc_FromFuncAndDataAndSignature(funcs, data, types, 4, 2, 1, PyUFunc_None,
                                             "group_demean", "", 0, "(i),(i)->(i)");
}

PyObject *get_group_demean_ufunc(bool check_valid) {
  if (check_valid) return get_group_demean_ufunc<DefaultCheck>();
  return get_group_demean_ufunc<CheckNothing>();
}

template <class T, class U>
void np_hedge(char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
  ENSURE2(steps[1] == 0);
  apply2d<T, U, U, T>(args, dimensions, steps, std::index_sequence<4, 1, 2, 5>(),
                      [](auto in_mat, auto univ_size, auto hedge_idx, auto out_mat) {
                        hedge(in_mat, out_mat, univ_size(0, 0), hedge_idx(0, 0));
                      });
}

PyObject *get_hedge_ufunc() {
  import_array();
  import_umath();
  // numpy cast python int to int64_t by default
  static PyUFuncGenericFunction funcs[] = {
      &np_hedge<float, int32_t>,
      &np_hedge<double, int32_t>,
      &np_hedge<float, int64_t>,
      &np_hedge<double, int64_t>,
  };
  // clang-format off
  static char types[] = {
      NPY_FLOAT,  NPY_INT32, NPY_INT32, NPY_FLOAT,
      NPY_DOUBLE, NPY_INT32, NPY_INT32, NPY_DOUBLE,
      NPY_FLOAT,  NPY_INT64, NPY_INT64, NPY_FLOAT,
      NPY_DOUBLE, NPY_INT64, NPY_INT64, NPY_DOUBLE,
  };
  // clang-format on
  static void *data[] = {NULL, NULL, NULL, NULL};
  return PyUFunc_FromFuncAndDataAndSignature(funcs, data, types, 4, 3, 1, PyUFunc_None, "hedge", "",
                                             0, "(i),(),()->(i)");
}

template <class T, class U>
void np_scale(char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
  ENSURE2(steps[1] == 0);
  ENSURE2(steps[2] == 0);
  apply2d<T, U, U, T>(args, dimensions, steps, std::index_sequence<4, 1, 2, 5>(),
                      [](auto in_mat, auto scale_size, auto eps, auto out_mat) {
                        scale(in_mat, out_mat, scale_size(0, 0), eps(0, 0));
                      });
}

PyObject *get_scale_ufunc() {
  import_array();
  import_umath();
  // numpy cast python float to double (float64) by default, we don't want all
  // arrays to be converted to double
  static PyUFuncGenericFunction funcs[] = {
      &np_scale<float, float>,
      &np_scale<float, double>,
      &np_scale<double, double>,
  };
  // clang-format off
  static char types[] = {
      NPY_FLOAT,  NPY_FLOAT,  NPY_FLOAT,  NPY_FLOAT,
      NPY_FLOAT,  NPY_DOUBLE, NPY_DOUBLE, NPY_FLOAT,
      NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE,
  };
  // clang-format on
  static void *data[] = {NULL, NULL, NULL};
  return PyUFunc_FromFuncAndDataAndSignature(funcs, data, types, 3, 3, 1, PyUFunc_None, "scale", "",
                                             0, "(i),(),()->(i)");
}

}  // namespace yang::math::ops
