#pragma once

#include <fstream>
#include <functional>
#include <memory>
#include <string>
#include <type_traits>

#include "yang/base/likely.h"
#include "yang/base/size.h"
#include "yang/data/null.h"
#include "yang/math/mat_view.h"
#include "yang/math/vec_view.h"
#include "yang/util/fs.h"
#include "yang/util/logging.h"
#include "yang/util/mmap_file.h"
#include "yang/util/small_vector.h"
#include "yang/util/type_name.h"

namespace yang {

using ArrayShape = small_vector<SizeType, 4>;

namespace detail {

struct ArrayMeta {
  std::string item_type;
  SizeType item_size;
  ArrayShape shape;

  bool Load(const std::string &path);
  void Save(const std::string &path) const;
  void Check(const std::string_view &expected_item_type, SizeType expected_item_size);
};

struct ArrayBackend {
  using Filler = std::function<void(void *, SizeType)>;

  virtual ~ArrayBackend() {}
  virtual void Resize(const ArrayShape &old_shape, const ArrayShape &new_shape, SizeType item_size,
                      const Filler &filler) = 0;
  virtual void ResizeRaw(const ArrayShape &new_shape, SizeType item_size) = 0;

  virtual void *data() const = 0;

  static void CopyArray(void *dest, const ArrayShape &dest_shape, void *src,
                        const ArrayShape &src_shape, SizeType item_size, const Filler &filler);
};

class DefaultArrayBackend : public ArrayBackend {
 public:
  ~DefaultArrayBackend() {
    Free();
  }

  void *data() const final {
    return data_;
  }

  void Resize(const ArrayShape &old_shape, const ArrayShape &new_shape, SizeType item_size,
              const Filler &filler) final;

  void ResizeRaw(const ArrayShape &new_shape, SizeType item_size) final;

 private:
  void *data_ = nullptr;
  SizeType size_ = 0;

  void Free();
};

class MMapArrayBackend : public ArrayBackend {
 public:
  MMapArrayBackend(const std::string &path, bool writable, const std::string &item_type,
                   SizeType item_size, const ArrayShape &shape, const Filler &filler);

  void *data() const final {
    return file_.addr();
  }

  const ArrayShape &shape() const {
    return meta_.shape;
  }

  void Resize(const ArrayShape &old_shape, const ArrayShape &new_shape, SizeType item_size,
              const Filler &filler) final;

  void ResizeRaw(const ArrayShape &new_shape, SizeType item_size) final;

 private:
  MMapFile file_;
  std::string path_;
  bool writable_;
  ArrayMeta meta_;

  void LoadFile();

  std::string GetMetaPath() const {
    return path_ + ".meta";
  }

  void Resize(const ArrayShape &old_shape, const ArrayShape &new_shape, SizeType item_size,
              const Filler &filler, bool create);
};

}  // namespace detail

class ArrayBase {
 public:
  virtual ~ArrayBase() {}

  virtual std::string_view item_type() const = 0;

  static void Copy(const std::string &from, const std::string &to);

  static std::string GetItemType(std::string_view array_path);
};

template <class T>
class Array : public ArrayBase {
  static_assert(std::is_trivially_copyable_v<T>);

 public:
  Array(const T &null = GetNullValue<T>()) : ptr_(nullptr), null_value_(null) {}

  Array(const ArrayShape &shape, const T &null = GetNullValue<T>()) : Array(null) {
    Resize(shape);
  }

  template <class Int>
  Array(std::initializer_list<Int> shape, const T &null = GetNullValue<T>())
      : Array(ArrayShape(shape), null) {}

  Array(Array &&other) {
    this->operator=(std::move(other));
  }

  Array(const Array &other) {
    this->operator=(other);
  }

  Array &operator=(Array &&other) {
    ptr_ = other.ptr_;
    shape_ = other.shape_;
    null_value_ = other.null_value_;
    backend_ = std::move(other.backend_);

    other.ptr_ = nullptr;
    other.shape_.clear();
    return *this;
  }

  Array &operator=(const Array &other) {
    ptr_ = other.ptr_;
    shape_ = other.shape_;
    null_value_ = other.null_value_;
    backend_ = other.backend_;
    return *this;
  }

  bool operator==(const Array &other) const {
    if (shape() != other.shape()) return false;
    SizeType size = GetNumItems();
    for (SizeType i = 0; i < size; ++i) {
      if (ptr_[i] != other.ptr_[i]) return false;
    }
    return true;
  }

  static Array MMap(const std::string &path, bool writable = false) {
    return MMapInternal(path, writable, ArrayShape({}));
  }

  static Array MMap(const std::string &path, const ArrayShape &shape,
                    const T &null = GetNullValue<T>()) {
    return MMapInternal(path, true, shape, null);
  }

  std::string_view item_type() const final {
    return GetTypeName<T>();
  }

  SizeType ndim() const {
    return shape_.size();
  }

  const ArrayShape &shape() const {
    return shape_;
  }

  SizeType shape(int i) const {
    return shape_[i];
  }

  const T &null_value() const {
    return null_value_;
  }

  void set_null_value(const T &v) {
    null_value_ = v;
  }

  template <class... Args>
  const T &operator()(Args... indexes) const {
    return ptr_[GetOffset(indexes...)];
  }

  template <class... Args>
  T &operator()(Args... indexes) {
    return ptr_[GetOffset(indexes...)];
  }

  template <class... Args>
  const T &Get(Args... indexes) const {
    return this->operator()(indexes...);
  }

  template <class... Args>
  T &Get(Args... indexes) {
    return this->operator()(indexes...);
  }

  template <class... Args>
  const T &at(Args... indexes) const {
    return ptr_[GetOffsetChecked(indexes...)];
  }

  template <class... Args>
  T &at(Args... indexes) {
    return ptr_[GetOffsetChecked(indexes...)];
  }

  T &operator[](SizeType i) {
    return ptr_[GetOffset(i)];
  }

  const T &operator[](SizeType i) const {
    return ptr_[GetOffset(i)];
  }

  template <class... Args>
  void Resize(Args... sizes) {
    Resize({static_cast<SizeType>(sizes)...});
  }

  const T *data() const {
    return ptr_;
  }

  T *data() {
    return ptr_;
  }

  math::MatView<T> mat_view() {
    ENSURE2(ndim() == 2);
    return math::MatView<T>(ptr_, {shape_[0], shape_[1]});
  }

  math::MatView<const T> mat_view() const {
    ENSURE2(ndim() == 2);
    return math::MatView<const T>(ptr_, {shape_[0], shape_[1]});
  }

  math::VecView<T> row_vec(int row) {
    return mat_view().row(row).to_vec();
  }

  math::VecView<const T> row_vec(int row) const {
    return mat_view().row(row).to_vec();
  }

  auto col_vec(int col) {
    return mat_view().col(col).to_vec();
  }

  auto col_vec(int col) const {
    return mat_view().col(col).to_vec();
  }

  math::VecView<T> vec_view() {
    ENSURE2(ndim() == 1);
    return math::VecView<T>(ptr_, shape_[0]);
  }

  math::VecView<const T> vec_view() const {
    ENSURE2(ndim() == 1);
    return math::VecView<const T>(ptr_, shape_[0]);
  }

  void Resize(const ArrayShape &new_shape) {
    ENSURE(shape_.empty() || shape_.size() == new_shape.size(),
           "Resize cannot change array dimension, old: ({}), new: ({})", shape_, new_shape);
    EnsureDefaultBackend();
    backend_->Resize(shape_, new_shape, sizeof(T),
                     [this](void *dest, SizeType n) { FillItems(dest, n); });
    ptr_ = reinterpret_cast<T *>(backend_->data());
    shape_ = new_shape;
  }

  [[nodiscard]] static Array Load(const std::string &path) {
    auto meta_path = path + ".meta";
    detail::ArrayMeta meta;
    if (!meta.Load(meta_path)) throw MakeExcept<IoError>("Array meta missing: {}", meta_path);
    meta.Check(GetTypeName<T>(), sizeof(T));

    Array array;
    array.backend_ = std::make_shared<detail::DefaultArrayBackend>();
    array.backend_->ResizeRaw(meta.shape, sizeof(T));
    array.ptr_ = reinterpret_cast<T *>(array.backend_->data());
    array.shape_ = meta.shape;
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.good()) throw MakeExcept<IoError>("Failed to read {}", path);
    ifs.read(reinterpret_cast<char *>(array.ptr_), array.GetNumItems() * sizeof(T));
    LOG_DEBUG("Loaded {} ({})", path, array.shape_);
    return array;
  }

  void Save(const std::string &path) const {
    fs::create_directories(fs::path(path).parent_path());

    std::ofstream ofs(path, std::ios::binary);
    if (!ofs.good()) throw MakeExcept<IoError>("Failed to write {}", path);
    ofs.write(reinterpret_cast<const char *>(ptr_), GetNumItems() * sizeof(T));

    detail::ArrayMeta meta;
    meta.item_type = GetTypeName<T>();
    meta.item_size = sizeof(T);
    meta.shape = shape_;
    meta.Save(path + ".meta");
    LOG_DEBUG("Saved {}", path);
  }

  void Clone(const Array &src) {
    EnsureDefaultBackend();
    backend_->ResizeRaw(src.shape_, sizeof(T));
    shape_ = src.shape_;
    ptr_ = reinterpret_cast<T *>(backend_->data());
    std::memcpy(ptr_, src.ptr_, GetNumItems() * sizeof(T));
    null_value_ = src.null_value_;
  }

  void CopyFrom(const Array &src, int row_begin, int row_end) {
    mat_view().slice(row_begin, row_end).copy_from(src.mat_view().slice(row_begin, row_end));
  }

  void FillNull(int row_begin, int row_end, int col_begin, int col_end) {
    mat_view().slice(row_begin, row_end, col_begin, col_end).fill(null_value_);
  }

  void FillNull(int row_begin, int row_end) {
    auto rows = row_end - row_begin;
    auto cols = 1;
    for (int i = 1; i < ndim(); ++i) {
      cols *= shape_[i];
    }
    math::VecView<T>(ptr_ + cols * row_begin, cols * rows).fill(null_value_);
  }

 private:
  T *ptr_;
  ArrayShape shape_;
  T null_value_;
  std::shared_ptr<detail::ArrayBackend> backend_;

  void FillItems(void *dest, SizeType n) {
    auto *ptr = reinterpret_cast<T *>(dest);
    std::fill_n(ptr, n, null_value_);
  }

  SizeType GetOffset() const {
#ifndef YANG_FORCE_BOUNDS_CHECK
    return 0;
#else
    return GetOffsetChecked();
#endif
  }

  template <class... Args>
  SizeType GetOffset(SizeType idx, Args... indexes) const {
#ifndef YANG_FORCE_BOUNDS_CHECK
    return ComputeOffset<false>(idx, 1, indexes...);
#else
    return GetOffsetChecked(idx, indexes...);
#endif
  }

  SizeType GetOffsetChecked() const {
    if UNLIKELY (!shape_.empty()) throw OutOfRange("Empty indexes");
    return 0;
  }

  template <class... Args>
  SizeType GetOffsetChecked(SizeType idx, Args... indexes) const {
    constexpr SizeType indexes_size = sizeof...(indexes) + 1;
    if UNLIKELY (indexes_size != shape_.size())
      throw MakeExcept<OutOfRange>("Number of indexes ({}) != array dim ({})", indexes_size,
                                   shape_.size());
    if UNLIKELY (idx >= shape_[0] || idx < 0)
      throw MakeExcept<OutOfRange>("Invalid index at dim 0, index:{}, size:{}", idx, shape_[0]);

    return ComputeOffset<true>(idx, 1, indexes...);
  }

  template <bool BoundsCheck, class... Args>
  SizeType ComputeOffset(SizeType acc, SizeType i, SizeType idx, Args... indexes) const {
    if constexpr (BoundsCheck) {
      if UNLIKELY (idx >= shape_[i] || idx < 0)
        throw MakeExcept<OutOfRange>("Invalid index at dim {}, index:{}, size:{}", i, idx,
                                     shape_[i]);
    }
    return ComputeOffset<BoundsCheck>(acc * shape_[i] + idx, i + 1, indexes...);
  }

  template <bool BoundsCheck>
  SizeType ComputeOffset(SizeType acc, SizeType) const {
    return acc;
  }

  SizeType GetNumItems() const {
    SizeType size = 1;
    for (auto &d : shape_) size *= d;
    return size;
  }

  void EnsureDefaultBackend() {
    if (!backend_) {
      backend_ = std::make_shared<detail::DefaultArrayBackend>();
    }
  }

  static Array MMapInternal(const std::string &path, bool writable, const ArrayShape &shape,
                            const T &null = GetNullValue<T>()) {
    Array array(null);
    auto mmap = std::make_shared<detail::MMapArrayBackend>(
        path, writable, std::string(GetTypeName<T>()), sizeof(T), shape,
        [&array](void *dest, SizeType n) { array.FillItems(dest, n); });
    array.shape_ = mmap->shape();
    array.ptr_ = reinterpret_cast<T *>(mmap->data());
    array.backend_ = std::move(mmap);
    return array;
  }
};

}  // namespace yang
