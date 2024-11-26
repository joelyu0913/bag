#pragma once

#include <string>

#include "yang/base/size.h"
#include "yang/data/index.h"
#include "yang/util/logging.h"

namespace yang {

class UnivIndex : public IndexBase<std::string> {
 public:
  static constexpr int UNKNOWN_LIST_DI = 10000000;

  UnivIndex() {}

  UnivIndex(UnivIndex &&other) {
    this->operator=(std::move(other));
  }

  UnivIndex &operator=(UnivIndex &&other) {
    index_ = std::move(other.index_);
    list_dis_ = std::move(other.list_dis_);
    index_id_start_ = other.index_id_start_;
    indices_ = std::move(other.indices_);
    IndexBase::operator=(std::move(other));
    return *this;
  }

  const std::string &operator()(SizeType i) const {
    return Get(i);
  }

  const std::string &operator[](SizeType i) const {
    return Get(i);
  }

  const std::string &Get(SizeType i) const {
    if (i >= index_id_start_) return indices_[i - index_id_start_];
    return items_[i];
  }

  template <class K>
  int Find(const K &k) const {
    auto it = index_.find(k);
    if (it != index_.end()) {
      return it->second;
    } else {
      return -1;
    }
  }

  template <class K>
  int Find(int di, const K &k) const {
    auto ret = Find(k);
    if (ret >= 0 && (ret >= index_id_start_ || list_dis_[ret] <= di)) return ret;
    return -1;
  }

  int GetOrInsert(int di, const std::string &symbol) {
    auto it = index_.find(symbol);
    if (it != index_.end()) {
      auto old_di = list_dis_[it->second];
      ENSURE(old_di <= di || old_di == UNKNOWN_LIST_DI,
             "list date changed, symbol: {}, old_di: {}, new_di: {}", symbol, old_di, di);
      if (old_di == UNKNOWN_LIST_DI) list_dis_[it->second] = di;
      return it->second;
    }
    index_.emplace_hint(it, symbol, items_.size());
    items_.push_back(symbol);
    list_dis_.push_back(di);
    ENSURE(static_cast<int>(items_.size()) < index_id_start_,
           "UnivIndex size exceeds index_id_start_");
    return items_.size() - 1;
  }

  const std::vector<std::string> &indices() const {
    return indices_;
  }

  // Find ii for indices_[idx]
  int FindIndexId(int idx) const {
    return index_id_start_ + idx;
  }

  int index_id_start() const {
    return index_id_start_;
  }

  int max_id() const {
    return index_id_start_ + indices_.size() - 1;
  }

  void Save() final;

  void SetIndices(const std::vector<std::string> &indices, int id_start);

  [[nodiscard]] static UnivIndex Load(const std::string &path, bool ensure_exists = false) {
    if (ensure_exists) {
      ENSURE(fs::exists(path), "Missing {}", path);
    }
    UnivIndex index;
    index.InternalLoad(path);
    return index;
  }

  int GetListDi(int ii) const {
    return list_dis_[ii];
  }

 private:
  unordered_map<std::string, int> index_;
  std::vector<int> list_dis_;

  int index_id_start_ = 10000;
  std::vector<std::string> indices_;

  void LoadImpl() final;

  std::string GetIndicesPath() const {
    return path_ + ".indices";
  }
};

}  // namespace yang
