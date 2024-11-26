#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <numpy/ndarraytypes.h>
#include <numpy/ufuncobject.h>

#include "helpers.h"
#include "yang/math/mat_ops.h"
#include "yang/math/mat_view.h"

namespace yang::math::ops {

template <class... Us, size_t... Is, size_t... Js, size_t... Ks, class Func>
void apply3d_impl(char **args, const npy_intp *dimensions, const npy_intp *steps,
                  std::index_sequence<Is...>, std::index_sequence<Js...>,
                  std::index_sequence<Ks...>, Func &&f) {
  auto n = dimensions[0];
  auto cont_stride = (true && ... && (steps[Ks] == 0 || steps[Ks] == sizeof(Us)));
  for (npy_intp i = 0; i < n; ++i) {
    if (cont_stride) {
      f(MatView<Us>(reinterpret_cast<Us *>(args[Is] + steps[Is]), dimensions[1], dimensions[2],
                    steps[Js] / sizeof(Us), 1)...);
    } else {
      f(MatView<Us, MatShape<>, MatStride<>>(reinterpret_cast<Us *>(args[Is] + steps[Is]),
                                             dimensions[1], dimensions[2], steps[Js] / sizeof(Us),
                                             steps[Ks] / sizeof(Us))...);
    }
  }
}

template <class... Us, size_t... Js, size_t... Ks, class Func>
void apply3d(char **args, const npy_intp *dimensions, const npy_intp *steps,
             std::index_sequence<Js...> js, std::index_sequence<Ks...> ks, Func &&f) {
  apply3d_impl<Us...>(args, dimensions, steps, std::index_sequence_for<Us...>(), js, ks,
                      std::move(f));
}

template <class T, class ValidCheck>
void np_ewa(char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
  ENSURE2(steps[1] == 0);
  apply3d<T, T, T>(
      args, dimensions, steps, std::index_sequence<3, 1, 5>(), std::index_sequence<4, 1, 6>(),
      [](auto in_mat, auto ratio, auto out_mat) { ewa<ValidCheck>(in_mat, out_mat, ratio(0, 0)); });
}

template <class ValidCheck>
PyObject *get_ewa_ufunc() {
  import_array();
  import_umath();
  static PyUFuncGenericFunction funcs[] = {&np_ewa<float, ValidCheck>, &np_ewa<double, ValidCheck>};
  static char types[] = {NPY_FLOAT, NPY_FLOAT, NPY_FLOAT, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE};
  static void *data[] = {NULL, NULL};
  return PyUFunc_FromFuncAndDataAndSignature(funcs, data, types, 2, 2, 1, PyUFunc_None, "ewa", "",
                                             0, "(i,j),()->(i,j)");
}

PyObject *get_ewa_ufunc(bool check_valid) {
  if (check_valid)
    return get_ewa_ufunc<DefaultCheck>();
  else
    return get_ewa_ufunc<CheckNothing>();
}

}  // namespace yang::math::ops
