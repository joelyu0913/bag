#pragma once

#include "yang/data/block_data.h"
#include "yang/data/index.h"
#include "yang/util/logging.h"

namespace yang {

template <class T>
class BlockIndex : public BlockData<BlockIndex<T>> {
 public:
  BlockIndex() {}

  BlockIndex(BlockIndex &&other) {
    this->operator=(std::move(other));
  }

  BlockIndex &operator=(BlockIndex &&other) {
    index_ = std::move(other.index_);
    blocks_ = std::move(other.blocks_);
    return *this;
  }

  const std::string &path() const {
    return index_.path();
  }

  bool empty() const {
    return index_.empty();
  }

  SizeType size() const {
    return index_.size();
  }

  const T &operator()(SizeType i) const {
    return Get(i);
  }

  const T &operator[](SizeType i) const {
    return Get(i);
  }

  const T &Get(SizeType i) const {
    return index_.Get(i);
  }

  const std::vector<T> &items() const {
    return index_.items();
  }

  std::vector<int> &blocks() {
    return blocks_;
  }

  const std::vector<int> &blocks() const {
    return blocks_;
  }

  void ResizeBlocks(int size) {
    if (size < this->num_blocks()) {
      index_.Shrink(this->block_begin(size));
    }
    blocks_.resize(size);
  }

  int Insert(const T &v) {
    auto ret = TryInsert(v);
    return ret.first;
  }

  std::pair<int, bool> TryInsert(const T &v) {
    auto ret = index_.TryInsert(v);
    if (ret.second) {
      ++blocks_[this->current_block()];
    }
    return ret;
  }

  template <class K>
  int Find(int max_block_idx, const K &k) const {
    int i = index_.Find(k);
    if (max_block_idx >= static_cast<int>(blocks_.size()) || i < blocks_[max_block_idx]) {
      return i;
    }
    return -1;
  }

  template <class K>
  int Find(const K &k) const {
    return index_.Find(k);
  }

  [[nodiscard]] static BlockIndex Load(const std::string &path) {
    BlockIndex index;
    index.InternalLoad(path);
    return index;
  }

  void Save() {
    index_.Save();
    std::string blocks_path = this->blocks_path();
    std::ofstream ofs(blocks_path);
    for (auto &x : blocks_) ofs << x << '\n';
    ofs.flush();
    ENSURE(ofs.good(), "Failed to save {}", blocks_path);
  }

 private:
  Index<T> index_;
  std::vector<int> blocks_;

  std::string blocks_path() const {
    return index_.path() + ".blocks";
  }

  void InternalLoad(const std::string &path) {
    index_ = Index<T>::Load(path);
    std::string blocks_path = this->blocks_path();
    if (std::filesystem::exists(blocks_path)) {
      std::ifstream ifs(blocks_path);
      int n;
      while (ifs >> n) {
        blocks_.push_back(n);
      }
      ENSURE(blocks_.empty() || blocks_.back() <= static_cast<int>(index_.size()),
             "BlockIndex corrupted: {}", blocks_path);
    }
  }
};

}  // namespace yang
