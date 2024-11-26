#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <numpy/ndarraytypes.h>
#include <numpy/ufuncobject.h>

#include "helpers.h"
#include "yang/math/mat_ops.h"
#include "yang/math/mat_view.h"
#include "yang/math/vec_view.h"

namespace yang::math::ops {

template <class RT, class... Us, size_t... Is, size_t... Js, class Func>
void apply_impl(char **args, const npy_intp *dimensions, const npy_intp *steps,
                std::index_sequence<Is...>, std::index_sequence<Js...>, Func &&f) {
  constexpr int N = sizeof...(Us) + 1;
  auto n = dimensions[0];
  auto m = dimensions[1];
  auto cont_stride = ((steps[N - 1] == 0 || steps[N - 1] == sizeof(RT)) && ... &&
                      (steps[Js] == 0 || steps[Js] == sizeof(Us)));
  if (cont_stride) {
    f(MatView<Us>(reinterpret_cast<Us *>(args[Is]), n, m, steps[Is] / sizeof(Us), 1)...,
      VecView<RT, 1>(reinterpret_cast<RT *>(args[N - 1]), n));
  } else {
    f(MatView<Us, MatShape<>, MatStride<>>(reinterpret_cast<Us *>(args[Is]), n, m,
                                           steps[Is] / sizeof(Us), steps[Js] / sizeof(Us))...,
      VecView<RT, DYNAMIC_STRIDE>(reinterpret_cast<RT *>(args[N - 1]), n,
                                  steps[N - 1] / sizeof(RT)));
  }
}

template <class RT, class... Us, size_t... Js, class Func>
void apply(char **args, const npy_intp *dimensions, const npy_intp *steps,
           std::index_sequence<Js...> js, Func &&f) {
  constexpr auto N = sizeof...(Us);
  auto seq = std::make_index_sequence<N>();
  apply_impl<RT, Us...>(args, dimensions, steps, seq, js, std::move(f));
}

template <class RT, class... Us, class Func>
void apply(char **args, const npy_intp *dimensions, const npy_intp *steps, Func &&f) {
  constexpr auto N = sizeof...(Us);
  auto seq = std::make_index_sequence<N>();
  apply_impl<RT, Us...>(args, dimensions, steps, seq, add_to_index_sequence<N + 1>(seq),
                        std::move(f));
}

#define DEFINE_REDUCTION(name, desc)                                                             \
  template <class T, class ValidCheck>                                                           \
  void np_##name(char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {   \
    apply<T, T>(args, dimensions, steps,                                                         \
                [](auto in_mat, auto out_vec) { name<ValidCheck>(in_mat, out_vec); });           \
  }                                                                                              \
  template <class ValidCheck>                                                                    \
  PyObject *get_##name##_ufunc() {                                                               \
    import_array();                                                                              \
    import_umath();                                                                              \
    static PyUFuncGenericFunction funcs[] = {&np_##name<float, ValidCheck>,                      \
                                             &np_##name<double, ValidCheck>};                    \
    static char types[] = {NPY_FLOAT, NPY_FLOAT, NPY_DOUBLE, NPY_DOUBLE};                        \
    static void *data[] = {NULL, NULL};                                                          \
    return PyUFunc_FromFuncAndDataAndSignature(funcs, data, types, 2, 1, 1, PyUFunc_None, #name, \
                                               desc, 0, "(i)->()");                              \
  }                                                                                              \
  PyObject *get_##name##_ufunc(bool check_valid) {                                               \
    if (check_valid)                                                                             \
      return get_##name##_ufunc<DefaultCheck>();                                                 \
    else                                                                                         \
      return get_##name##_ufunc<CheckNothing>();                                                 \
  }

DEFINE_REDUCTION(mean, "");
DEFINE_REDUCTION(variance, "");
DEFINE_REDUCTION(stdev, "");
DEFINE_REDUCTION(sum, "");

}  // namespace yang::math::ops
