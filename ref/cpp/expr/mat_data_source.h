#pragma once

#include "yang/expr/data_source.h"
#include "yang/expr/mat.h"
#include "yang/math/mat_view.h"
#include "yang/util/logging.h"
#include "yang/util/unordered_map.h"

namespace yang::expr {

class MatDataSource : public DataSource {
 public:
  template <class T>
  using Mat = yang::math::MatView<const T>;

  MatDataSource() {}

  MatDataSource(int rows, int cols) {
    Initialize(rows, cols);
  }

  MatDataSource(const MatDataSource &other) = delete;
  MatDataSource &operator=(const MatDataSource &other) = delete;

  void Initialize(int rows, int cols) {
    rows_ = rows;
    cols_ = cols;
  }

  int rows() const {
    return rows_;
  }

  int cols() const {
    return cols_;
  }

  int vec_size() const final {
    return cols_;
  }

  int cur_row() const {
    return cur_row_;
  }

  void set_cur_row(int row);

  bool live() const {
    return live_;
  }

  void set_live(bool v) {
    live_ = v;
  }

  VecView<Float> GetData(std::string_view name) const final;

  VecView<int> GetGroup(std::string_view name) const final;

  VecView<bool> GetMask() const;

  void AddData(std::string_view name, FloatMat live_data, FloatMat eod_data);

  void AddGroup(std::string_view name, Mat<int> group) {
    groups_[name] = group;
  }

  void set_mask(Mat<bool> mask) {
    mask_ = mask;
  }

  bool has_data(std::string_view name) const {
    return data_map_.count(name);
  }

  bool has_group(std::string_view name) const {
    return groups_.count(name);
  }

 private:
  int rows_ = 0;
  int cols_ = 0;
  int cur_row_ = -1;
  bool live_ = true;

  unordered_map<std::string, int> data_map_;
  std::vector<FloatMat> live_data_;
  std::vector<FloatMat> eod_data_;
  mutable std::vector<std::vector<Float>> live_data_rows_;
  mutable std::vector<std::vector<Float>> eod_data_rows_;
  unordered_map<std::string, Mat<int>> groups_;
  Mat<bool> mask_;
};

}  // namespace yang::expr
