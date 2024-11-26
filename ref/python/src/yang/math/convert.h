#pragma once

#include <pybind11/pybind11.h>

#include <string_view>

#include "yang/math/mat_view.h"
#include "yang/math/vec_view.h"
#include "yang/util/logging.h"

namespace yang::math {

template <class T>
yang::math::MatView<T> py_buffer_to_mat(pybind11::buffer buf, std::string_view format) {
  auto buf_info = buf.request();
  ENSURE(buf_info.format == format, "Wrong buffer format: {}, expected: {}", buf_info.format,
         format);
  ENSURE(buf_info.ndim == 2, "Wrong buffer ndim: {}, expected: 2", buf_info.ndim);
  ENSURE(buf_info.strides[1] == sizeof(T), "Wrong buffer strides[1]: {}, expected: {}",
         buf_info.strides[1], sizeof(T));
  return yang::math::MatView<T>(reinterpret_cast<T *>(buf_info.ptr), buf_info.shape[0],
                                buf_info.shape[1], buf_info.strides[0] / sizeof(T), 1);
}

template <class T>
yang::math::VecView<T> py_buffer_to_vec(pybind11::buffer buf, std::string_view format) {
  auto buf_info = buf.request();
  ENSURE(buf_info.format == format, "Wrong buffer format: {}, expected: {}", buf_info.format,
         format);
  ENSURE(buf_info.ndim == 1, "Wrong buffer ndim: {}, expected: 2", buf_info.ndim);
  ENSURE(buf_info.strides[0] == sizeof(T), "Wrong buffer strides[0]: {}, expected: {}",
         buf_info.strides[1], sizeof(T));
  return yang::math::VecView<T>(reinterpret_cast<T *>(buf_info.ptr), buf_info.shape[0]);
}

}  // namespace yang::math
