#include "yang/data/univ_index.h"

namespace yang {

void UnivIndex::Save() {
  {
    std::ofstream ofs(path_);
    for (int i = 0; i < size(); ++i) {
      ofs << items_[i] << " " << list_dis_[i] << '\n';
    }
    ofs.flush();
    ENSURE(ofs.good(), "Failed to save {}", path_);
  }

  if (!indices_.empty()) {
    auto indices_path = GetIndicesPath();
    std::ofstream ofs(indices_path);
    ofs << index_id_start_ << '\n';
    for (auto &idx : indices_) ofs << idx << '\n';
    ofs.flush();
    ENSURE(ofs.good(), "Failed to save {}", indices_path);
  }
}

void UnivIndex::LoadImpl() {
  if (fs::exists(path_)) {
    std::ifstream ifs(path_);
    std::string symbol;
    int di;
    while (ifs >> symbol >> di) {
      items_.emplace_back(std::move(symbol));
      list_dis_.emplace_back(di);
      index_[items_.back()] = items_.size() - 1;
    }
  }
  auto indices_path = GetIndicesPath();
  if (fs::exists(indices_path)) {
    std::ifstream ifs(indices_path);
    ifs >> index_id_start_;
    std::string symbol;
    while (ifs >> symbol) {
      indices_.push_back(std::move(symbol));
      index_[indices_.back()] = indices_.size() - 1 + index_id_start_;
    }
  }
}

void UnivIndex::SetIndices(const std::vector<std::string> &indices, int id_start) {
  index_id_start_ = id_start;
  for (auto &idx : indices_) index_.erase(idx);
  indices_ = indices;
  for (int i = 0; i < static_cast<int>(indices_.size()); ++i) {
    index_[indices_[i]] = i + index_id_start_;
  }
}

}  // namespace yang
