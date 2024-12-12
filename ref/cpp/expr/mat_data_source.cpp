#include "yang/expr/mat_data_source.h"

namespace yang::expr {

VecView<Float> MatDataSource::GetData(std::string_view name) const {
  auto it = data_map_.find(name);
  ENSURE(it != data_map_.end(), "Missing data: {}", name);
  int idx = it->second;
  auto &data_row = live_ ? live_data_rows_[idx] : eod_data_rows_[idx];
  if (data_row.empty()) {
    auto &data = live_ ? live_data_[idx] : eod_data_[idx];
    data_row.resize(data.cols());
    data.LoadRow(OutVecView<Float>(data_row), cur_row_);
  }
  return VecView<Float>(data_row);
}

VecView<int> MatDataSource::GetGroup(std::string_view name) const {
  auto it = groups_.find(name);
  ENSURE(it != groups_.end(), "Missing group: {}", name);
  return it->second.row(cur_row_).to_vec();
}

VecView<bool> MatDataSource::GetMask() const {
  if (mask_.empty()) return VecView<bool>();
  return mask_.row(cur_row_).to_vec();
}

void MatDataSource::AddData(std::string_view name, FloatMat live_data, FloatMat eod_data) {
  ENSURE(data_map_.count(name) == 0, "Duplicate data {}", name);
  data_map_[name] = live_data_.size();
  live_data_.push_back(live_data);
  eod_data_.push_back(eod_data);
  live_data_rows_.push_back(std::vector<Float>(vec_size(), NAN));
  eod_data_rows_.push_back(std::vector<Float>(vec_size(), NAN));
}

void MatDataSource::set_cur_row(int row) {
  ENSURE(row >= 0 && row < rows_, "Invalid row: {}", row);
  if (row != cur_row_) {
    cur_row_ = row;
    for (auto &row : live_data_rows_) row.clear();
    for (auto &row : eod_data_rows_) row.clear();
  }
}

}  // namespace yang::expr
