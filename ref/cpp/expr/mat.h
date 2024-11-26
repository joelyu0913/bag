#pragma once

#include <variant>

#include "yang/math/mat_view.h"
#include "yang/math/vec_view.h"

namespace yang::expr {

class FloatMat {
 public:
  FloatMat(yang::math::MatView<const float> m) : data_(m) {}
  FloatMat(yang::math::MatView<const double> m) : data_(m) {}
  FloatMat(yang::math::MatView<const int32_t> m) : data_(m) {}

  template <class T>
  void LoadRow(yang::math::VecView<T> vec, int row) const {
    std::visit([&](auto &&m) { vec.copy_from(m.row(row).to_vec()); }, data_);
  }

  int rows() const {
    return std::visit([](auto &&m) { return m.rows(); }, data_);
  }

  int cols() const {
    return std::visit([](auto &&m) { return m.cols(); }, data_);
  }

  int type_index() const {
    return data_.index();
  }

  template <class T>
  auto raw() const {
    return std::get<yang::math::MatView<const T>>(data_);
  }

 private:
  std::variant<yang::math::MatView<const float>, yang::math::MatView<const double>,
               yang::math::MatView<const int32_t>>
      data_;
};

class OutFloatMat {
 public:
  OutFloatMat(yang::math::MatView<float> m) : data_(m) {}
  OutFloatMat(yang::math::MatView<double> m) : data_(m) {}

  template <class T>
  void SaveRow(yang::math::VecView<const T> vec, int row) const {
    std::visit([&](auto &&m) { m.row(row).to_vec().slice(0, vec.size()).copy_from(vec); }, data_);
  }

  int rows() const {
    return std::visit([](auto &&m) { return m.rows(); }, data_);
  }

  int cols() const {
    return std::visit([](auto &&m) { return m.cols(); }, data_);
  }

  int type_index() const {
    return data_.index();
  }

  template <class T>
  auto raw() const {
    return std::get<yang::math::MatView<T>>(data_);
  }

 private:
  std::variant<yang::math::MatView<float>, yang::math::MatView<double>> data_;
};

}  // namespace yang::expr
