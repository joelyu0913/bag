#pragma once

#include <array>
#include <memory>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "yang/base/exception.h"
#include "yang/base/size.h"
#include "yang/data/array.h"
#include "yang/util/logging.h"
#include "yang/util/type_name.h"
#include "yang/util/unordered_map.h"

namespace yang {

struct StructVectorMeta {
  SizeType size = 0;
  std::vector<std::string> fields;

  bool Load(const std::string &path);
  void Save(const std::string &path) const;
};

class StructVectorBase {
 public:
  StructVectorBase() {}

  StructVectorBase(StructVectorBase &&other) {
    this->operator=(std::move(other));
  }

  StructVectorBase &operator=(StructVectorBase &&other) {
    meta_ = std::move(other.meta_);
    fields_ = std::move(other.fields_);
    field_idx_ = std::move(other.field_idx_);
    path_ = std::move(other.path_);
    extend_size_ = other.extend_size_;
    other.path_ = "";
    return *this;
  }

  const std::string &path() const {
    return path_;
  }

  SizeType size() const {
    return meta_.size;
  }

  SizeType num_fields() const {
    return field_names().size();
  }

  const std::vector<std::string> &field_names() const {
    return meta_.fields;
  }

  const std::string &field_name(int i) const {
    return field_names()[i];
  }

  template <class T>
  Array<T> &field(int idx) {
    auto &f = fields_[idx];
    ENSURE2(f->data_type() == GetTypeName<T>());
    return static_cast<Field<T> &>(*f).data;
  }

  template <class T>
  const Array<T> &field(int idx) const {
    auto &f = fields_[idx];
    ENSURE2(f->data_type() == GetTypeName<T>());
    return static_cast<const Field<T> &>(*f).data;
  }

  template <class T>
  Array<T> &field(std::string_view name) {
    auto it = field_idx_.find(name);
    if (it != field_idx_.end()) {
      return field<T>(it->second);
    } else {
      throw MakeExcept<OutOfRange>("Field {} not found", name);
    }
  }

  template <class T>
  const Array<T> &field(std::string_view name) const {
    auto it = field_idx_.find(name);
    if (it != field_idx_.end()) {
      return field<T>(it->second);
    } else {
      throw MakeExcept<OutOfRange>("Field {} not found", name);
    }
  }

  template <class T>
  T &Get(SizeType idx, int f_idx) {
    return static_cast<Field<T> &>(*fields_[f_idx]).data(idx);
  }

  template <class T>
  const T &Get(SizeType idx, int f_idx) const {
    return static_cast<const Field<T> &>(*fields_[f_idx]).data(idx);
  }

  int FindField(std::string_view name) const {
    auto it = field_idx_.find(name);
    if (it != field_idx_.end()) {
      return it->second;
    } else {
      return -1;
    }
  }

  void Extend() {
    Extend(extend_size_);
  }

  void Extend(SizeType extend_by);

 protected:
  struct FieldBase {
    virtual ~FieldBase() {}
    virtual std::string_view data_type() const = 0;
    virtual void Resize(const ArrayShape &shape) = 0;
  };

  template <class T>
  struct Field : FieldBase {
    Array<T> data;

    Field(const std::string &path, bool writable) : data(Array<T>::MMap(path, writable)) {}

    std::string_view data_type() const final {
      return GetTypeName<T>();
    }

    void Resize(const ArrayShape &shape) final {
      data.Resize(shape);
    }
  };

  StructVectorMeta meta_;
  std::vector<std::unique_ptr<FieldBase>> fields_;
  unordered_map<std::string_view, int> field_idx_;

  std::string path_;
  SizeType extend_size_ = 1024;

  std::string GetMetaPath() const {
    return path_ + ".meta";
  }
};

template <class Struct>
class StructVector;

template <class... Ts>
class StructVector<std::tuple<Ts...>> : public StructVectorBase {
 public:
  static constexpr int NUM_FIELDS = sizeof...(Ts);
  static_assert(NUM_FIELDS > 0);

  using Struct = std::tuple<Ts...>;

  Struct operator[](SizeType idx) const {
    return GetRecord(idx, std::make_index_sequence<NUM_FIELDS>());
  }

  [[nodiscard]] static StructVector Load(const std::string &path) {
    return MMap(path);
  }

  // Open StructVector for read
  static StructVector MMap(const std::string &path,
                           const std::vector<std::string_view> &field_names = {}) {
    return MMap<false>(path, field_names, 0, 0, std::make_index_sequence<NUM_FIELDS>());
  }

  // Open StructVector for write
  static StructVector MMap(const std::string &path,
                           const std::array<std::string_view, NUM_FIELDS> &field_names,
                           SizeType init_size, SizeType extend_size) {
    return MMap<true>(path, field_names, init_size, extend_size,
                      std::make_index_sequence<NUM_FIELDS>());
  }

 private:
  template <bool Writable, class Names, size_t... Is>
  static StructVector MMap(const std::string &path, const Names &names, SizeType init_size,
                           SizeType extend_size, std::index_sequence<Is...>) {
    StructVector vec;
    vec.path_ = path;
    vec.extend_size_ = extend_size;
    vec.fields_.reserve(NUM_FIELDS);
    auto meta_path = vec.GetMetaPath();
    std::vector<std::string> field_names;
    if constexpr (Writable) {
      field_names = {std::string(std::get<Is>(names))...};
      if (vec.meta_.Load(meta_path)) {
        ENSURE(vec.meta_.fields == field_names,
               "StructVector fields mismatch, in meta: {}, expected: {} ({}). "
               "Please delete the corrupted data and rebuild.",
               vec.meta_.fields, field_names, path);
      }

      (vec.fields_.emplace_back(std::make_unique<Field<Ts>>(path + "._" + field_names[Is], true)),
       ...);
      auto init = [&](auto &f) {
        if (f.ndim() == 0) f.Resize(init_size);
      };
      (init(vec.field<Ts>(Is)), ...);
    } else {
      ENSURE(vec.meta_.Load(meta_path), "Failed to load meta {}", meta_path);
      if (names.empty()) {
        field_names = vec.meta_.fields;
        ENSURE2(static_cast<int>(field_names.size()) == NUM_FIELDS);
      } else {
        for (auto &name : names) field_names.push_back(std::string(name));
        vec.meta_.fields = field_names;
      }
      (vec.fields_.emplace_back(std::make_unique<Field<Ts>>(path + "._" + field_names[Is], false)),
       ...);
    }

    auto field_sizes = std::vector<SizeType>{vec.field<Ts>(Is).shape(0)...};
    SizeType vec_size = field_sizes[0];
    for (int i = 1; i < NUM_FIELDS; ++i) {
      ENSURE(field_sizes[i] == vec_size,
             "StructVector fields have different size: {}({}) {}({}) ({}). "
             "Please delete the corrupted data and rebuild.",
             field_names[0], vec_size, field_names[i], field_sizes[i], path);
    }
    if (vec.meta_.size != 0) {
      ENSURE(vec.meta_.size == vec_size,
             "StructVector size mismatch, in meta: {}, from fields: {} ({}). "
             "Please delete the corrupted data and rebuild.",
             vec.meta_.size, vec_size, path);
    }

    if constexpr (Writable) {
      vec.meta_.size = vec_size;
      vec.meta_.fields = field_names;
      vec.meta_.Save(meta_path);
    }
    for (int i = 0; i < vec.num_fields(); ++i) {
      vec.field_idx_[vec.field_name(i)] = i;
    }
    return vec;
  }

  template <size_t... Is>
  Struct GetRecord(SizeType idx, std::index_sequence<Is...>) const {
    return std::make_tuple(field<Ts>(Is)(idx)...);
  }
};

}  // namespace yang
