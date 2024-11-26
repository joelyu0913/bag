#pragma once

#include <span>
#include <string>

#include "yang/base/size.h"
#include "yang/data/array.h"
#include "yang/data/block_data.h"
#include "yang/math/vec_view.h"
#include "yang/util/logging.h"

namespace yang {

template <class T>
class BlockVector : public BlockData<BlockVector<T>> {
 public:
  BlockVector() {}

  BlockVector(BlockVector &&other) {
    this->operator=(std::move(other));
  }

  BlockVector &operator=(BlockVector &&other) {
    array_ = std::move(other.array_);
    blocks_ = std::move(other.blocks_);
    this->current_block_ = other.current_block_;
    extend_size_ = other.extend_size_;
    path_ = std::move(other.path_);
    other.path_ = "";
    return *this;
  }

  const std::string &path() const {
    return path_;
  }

  SizeType size() const {
    return array_.shape(0);
  }

  const T &operator[](SizeType i) const {
    return Get(i);
  }

  T &operator[](SizeType i) {
    return Get(i);
  }

  const T &operator()(SizeType i) const {
    return Get(i);
  }

  T &operator()(SizeType i) {
    return Get(i);
  }

  const T &Get(SizeType i) const {
    return array_(i);
  }

  T &Get(SizeType i) {
    return array_(i);
  }

  const T &at(SizeType i) const {
    return array_.at(i);
  }

  T &at(SizeType i) {
    return array_.at(i);
  }

  auto blocks() {
    return blocks_.vec_view();
  }

  auto blocks() const {
    return blocks_.vec_view();
  }

  SizeType extend_size() const {
    return extend_size_;
  }

  void set_extend_size(SizeType v) const {
    extend_size_ = v;
  }

  const T *begin(int bi) const {
    return &Get(this->block_begin(bi));
  }

  const T *end(int bi) const {
    return &Get(this->block_end(bi));
  }

  std::span<const T> block_values(int bi) const {
    return std::span<const T>(begin(bi), end(bi));
  }

  void PushBack(const T &v) {
    ENSURE(this->current_block_ >= 0, "current_block_ not set: {}", this->current_block_);
    auto &last_idx = blocks_(this->current_block());
    if (size() == static_cast<SizeType>(last_idx)) {
      Extend();
    }
    Get(last_idx) = v;
    ++last_idx;
  }

  void Extend() {
    Extend(extend_size_);
  }

  void Extend(SizeType extend_by) {
    ENSURE2(extend_by > 0);
    LOG_DEBUG("Extending {}: {} -> {}", path_, size(), size() + extend_by);
    array_.Resize(size() + extend_by);
  }

  void ResizeBlocks(SizeType size) {
    blocks_.Resize(size);
  }

  [[nodiscard]] static BlockVector Load(const std::string &path) {
    return MMap(path);
  }

  static BlockVector MMap(const std::string &path, bool writable = false) {
    BlockVector vec;
    vec.array_ = Array<T>::MMap(path, writable);
    vec.blocks_ = Array<uint64_t>::MMap(path + ".blocks", writable, 0);
    vec.path_ = path;
    return vec;
  }

  static BlockVector MMap(const std::string &path, SizeType num_blocks, SizeType init_size,
                          SizeType extend_size) {
    BlockVector vec;
    vec.extend_size_ = extend_size;
    vec.blocks_ = Array<uint64_t>::MMap(path + ".blocks", {num_blocks}, 0);
    vec.array_ = Array<T>::MMap(path, true);
    vec.path_ = path;
    if (vec.array_.ndim() == 0) vec.array_.Resize(init_size);
    return vec;
  }

 private:
  Array<T> array_;
  Array<uint64_t> blocks_;  // TODO: change uint64_t to int64_t, but it'd break existing data
  SizeType extend_size_ = 1024;
  std::string path_;
};

}  // namespace yang
