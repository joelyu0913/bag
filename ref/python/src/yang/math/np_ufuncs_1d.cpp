#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <numpy/ndarraytypes.h>
#include <numpy/ufuncobject.h>

#include "helpers.h"
#include "yang/math/ops.h"
#include "yang/math/vec_view.h"

namespace yang::math::ops {

template <class... Us, size_t... Is, class Func>
void apply1d_impl(char **args, const npy_intp *dimensions, const npy_intp *steps,
                  std::index_sequence<Is...>, Func &&f) {
  auto n = dimensions[0];
  auto cont_stride = (true && ... && (steps[Is] == 0 || steps[Is] == sizeof(Us)));
  if (cont_stride) {
    f(VecView<Us, 1>(reinterpret_cast<Us *>(args[Is]), n, 0)...);
  } else {
    f(VecView<Us, DYNAMIC_STRIDE>(reinterpret_cast<Us *>(args[Is]), n, steps[Is] / sizeof(Us))...);
  }
}

template <class... Us, class Func>
void apply1d(char **args, const npy_intp *dimensions, const npy_intp *steps, Func &&f) {
  apply1d_impl<Us...>(args, dimensions, steps, std::index_sequence_for<Us...>(), std::move(f));
}

template <class T>
void np_filter(char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
  apply1d<T, bool, T>(args, dimensions, steps, [](auto in_vec, auto c_vec, auto out_vec) {
    filter(in_vec.begin(), in_vec.end(), c_vec.begin(), out_vec.begin());
  });
}

PyObject *get_filter_ufunc() {
  import_array();
  import_umath();
  static PyUFuncGenericFunction funcs[] = {
      &np_filter<float>,
      &np_filter<double>,
      &np_filter<int32_t>,
      &np_filter<int64_t>,
  };
  static char types[] = {NPY_FLOAT, NPY_BOOL, NPY_FLOAT, NPY_DOUBLE, NPY_BOOL, NPY_DOUBLE,
                         NPY_INT32, NPY_BOOL, NPY_INT32, NPY_INT64,  NPY_BOOL, NPY_INT64};
  static void *data[] = {NULL, NULL, NULL, NULL};
  return PyUFunc_FromFuncAndData(funcs, data, types, 4, 2, 1, PyUFunc_None, "filter", "", 0);
}

}  // namespace yang::math::ops
