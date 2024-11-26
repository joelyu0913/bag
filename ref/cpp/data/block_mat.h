#pragma once

#include "yang/data/array.h"
#include "yang/data/block_vector.h"
#include "yang/util/fs.h"
#include "yang/util/logging.h"

namespace yang {

template <class T>
class BlockMat {
 public:
  BlockMat() {}

  BlockMat(BlockMat &&other) {
    this->operator=(std::move(other));
  }

  BlockMat &operator=(BlockMat &&other) {
    base_path_ = std::move(other.base_path_);
    writable_ = other.writable_;
    ids_ = std::move(other.ids_);
    mat_ = std::move(other.mat_);
    return *this;
  }

  BlockVector<int> &id() {
    return ids_;
  }

  Array<T> &mat() {
    return mat_;
  }

  void StartId(int bi) {
    ENSURE2(writable_);
    ids_.StartBlock(bi);
  }

  void AddId(int id) {
    ids_.PushBack(id);
  }

  int GetId(int bi, int offset) const {
    return ids_(ids_.block_begin(bi) + offset);
  }

  int mat_size(int bi) {
    return ids_.block_size(bi);
  }

  bool StartMat(int bi) {
    ENSURE2(writable_);
    int size = mat_size(bi);
    if (size > 0) {
      mat_ = Array<T>::MMap(GetMatPath(bi), {size, size});
      return true;
    } else {
      mat_ = Array<T>();
      return false;
    }
  }

  void Set(int r, int c, const T &v) {
    mat_(r, c) = v;
  }

  bool LoadMat(int bi) {
    ENSURE2(!writable_);
    auto mat_path = GetMatPath(bi);
    if (!fs::exists(mat_path)) return false;
    mat_ = Array<T>::MMap(mat_path);
    return true;
  }

  const T &Get(int r, int c) const {
    return mat_(r, c);
  }

  const T &operator()(int r, int c) const {
    return Get(r, c);
  }

  [[nodiscard]] static BlockMat Load(const std::string &path) {
    return MMap(path);
  }

  static BlockMat MMap(const std::string &path) {
    BlockMat mat;
    mat.base_path_ = path;
    mat.writable_ = false;
    mat.ids_ = BlockVector<int>::MMap(mat.GetIdPath());
    return mat;
  }

  static BlockMat MMap(const std::string &path, SizeType num_blocks, SizeType id_init_size,
                       SizeType id_extend_size) {
    BlockMat mat;
    mat.base_path_ = path;
    mat.writable_ = true;
    mat.ids_ = BlockVector<int>::MMap(mat.GetIdPath(), num_blocks, id_init_size, id_extend_size);
    return mat;
  }

 private:
  std::string base_path_;
  bool writable_;
  BlockVector<int> ids_;
  Array<T> mat_;

  std::string GetIdPath() const {
    return (fs::path(base_path_ + ".id")).string();
  }

  std::string GetMatPath(int bi) const {
    return (fs::path(base_path_ + ".mat") / std::to_string(bi)).string();
  }
};

}  // namespace yang
