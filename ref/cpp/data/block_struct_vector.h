#pragma once

#include "yang/data/block_data.h"
#include "yang/data/struct_array.h"
#include "yang/util/logging.h"

namespace yang {

template <class T>
class BlockStructVector;

template <class... Ts>
class BlockStructVector<std::tuple<Ts...>>
    : public BlockData<BlockStructVector<std::tuple<Ts...>>> {
 public:
  using Struct = std::tuple<Ts...>;
  using Vector = StructArray<Struct>;

  static constexpr int NUM_FIELDS = Vector::NUM_FIELDS;

  BlockStructVector() {}

  BlockStructVector(BlockStructVector &&other) {
    this->operator=(std::move(other));
  }

  BlockStructVector &operator=(BlockStructVector &&other) {
    vec_ = std::move(other.vec_);
    blocks_ = std::move(other.blocks_);
    extend_size_ = other.extend_size_;
    return *this;
  }

  const std::string &path() const {
    return vec_.path();
  }

  SizeType size() const {
    return vec_.shape(0);
  }

  SizeType num_fields() const {
    return vec_.num_fields();
  }

  const std::vector<std::string> &field_names() const {
    return vec_.field_names();
  }

  const std::string &field_name(int i) const {
    return vec_.field_name(i);
  }

  template <class T>
  Array<T> &field(int i) {
    return vec_.template field<T>(i);
  }

  template <class T>
  const Array<T> &field(int i) const {
    return vec_.template field<T>(i);
  }

  template <class T>
  Array<T> &field(std::string_view name) {
    return vec_.template field<T>(name);
  }

  template <class T>
  const Array<T> &field(std::string_view name) const {
    return vec_.template field<T>(name);
  }

  template <class T>
  T &Get(SizeType idx, int f_idx) {
    return field<T>(f_idx)(idx);
  }

  template <class T>
  const T &Get(SizeType idx, int f_idx) const {
    return field<T>(f_idx)(idx);
  }

  int FindField(std::string_view name) const {
    return vec_.FindField(name);
  }

  Struct operator[](SizeType i) const {
    return vec_(i);
  }

  auto blocks() {
    return blocks_.vec_view();
  }

  auto blocks() const {
    return blocks_.vec_view();
  }

  void PushBack(const Ts &...vs) {
    PushBack(vs..., std::make_index_sequence<sizeof...(Ts)>());
  }

  void Extend() {
    Extend(extend_size_);
  }

  void Extend(SizeType extend_by) {
    ENSURE2(extend_by > 0);
    vec_.Resize(size() + extend_by);
  }

  void ResizeBlocks(SizeType size) {
    blocks_.Resize(size);
  }

  [[nodiscard]] static BlockStructVector Load(const std::string &path) {
    return MMap(path);
  }

  static BlockStructVector MMap(const std::string &path,
                                const std::vector<std::string_view> &field_names = {}) {
    BlockStructVector vec;
    vec.vec_ = Vector::MMap(path, field_names);
    vec.blocks_ = Array<int64_t>::MMap(path + ".blocks");
    return vec;
  }

  static BlockStructVector MMap(const std::string &path,
                                const std::array<std::string_view, Vector::NUM_FIELDS> &field_names,
                                SizeType num_blocks, SizeType init_size, SizeType extend_size) {
    BlockStructVector vec;
    vec.blocks_ = Array<int64_t>::MMap(path + ".blocks", {num_blocks}, 0);
    vec.vec_ = Vector::MMap(path, field_names, {});
    if (vec.vec_.ndim() == 0) {
      vec.vec_.Resize(init_size);
    }
    ENSURE2(vec.vec_.ndim() == 1);
    vec.extend_size_ = extend_size;
    return vec;
  }

 private:
  Vector vec_;
  Array<int64_t> blocks_;
  SizeType extend_size_ = 1024;

  template <size_t F, class T>
  void Set(SizeType idx, const T &v) {
    field<T>(F)(idx) = v;
  }

  template <size_t... Is>
  void PushBack(const Ts &...vs, std::index_sequence<Is...>) {
    ENSURE(this->current_block_ >= 0, "current_block_ not set: {}", this->current_block_);
    auto &last_idx = blocks_(this->current_block());
    if (size() == static_cast<SizeType>(last_idx)) {
      Extend();
    }
    (Set<Is, Ts>(last_idx, vs), ...);
    ++last_idx;
  }
};

}  // namespace yang
