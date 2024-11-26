#pragma once

#include <array>
#include <memory>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "yang/base/exception.h"
#include "yang/base/size.h"
#include "yang/data/array.h"
#include "yang/util/fs.h"
#include "yang/util/logging.h"
#include "yang/util/type_name.h"
#include "yang/util/unordered_map.h"

namespace yang {

struct StructArrayMeta {
  ArrayShape shape;
  std::vector<std::string> fields;

  [[nodiscard]] static StructArrayMeta Load(const std::string &path);
  void Save(const std::string &path) const;
};

class StructArrayBase {
 public:
  StructArrayBase() {}

  StructArrayBase(StructArrayBase &&other) {
    this->operator=(std::move(other));
  }

  StructArrayBase &operator=(StructArrayBase &&other) {
    meta_ = std::move(other.meta_);
    fields_ = std::move(other.fields_);
    field_idx_ = std::move(other.field_idx_);
    path_ = std::move(other.path_);
    other.path_ = "";
    return *this;
  }

  const std::string &path() const {
    return path_;
  }

  int ndim() const {
    return shape().size();
  }

  const ArrayShape &shape() const {
    return meta_.shape;
  }

  SizeType shape(int i) const {
    return shape()[i];
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
  Array<T> &field(int i) {
    auto &f = fields_[i];
    ENSURE(f->data_type() == GetTypeName<T>(), "Field {} has type {} but {} requested",
           field_name(i), f->data_type(), GetTypeName<T>());
    return static_cast<Field<T> &>(*f).data;
  }

  template <class T>
  const Array<T> &field(int i) const {
    auto &f = fields_[i];
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

  int FindField(std::string_view name) const {
    auto it = field_idx_.find(name);
    if (it != field_idx_.end()) {
      return it->second;
    } else {
      return -1;
    }
  }

  template <class... Args>
  void Resize(Args... sizes) {
    Resize({static_cast<SizeType>(sizes)...});
  }

  void Resize(const ArrayShape &new_shape);

 protected:
  struct FieldBase {
    virtual ~FieldBase() {}
    virtual std::string_view data_type() const = 0;
    virtual void Resize(const ArrayShape &shape) = 0;
  };

  template <class T>
  struct Field : FieldBase {
    Array<T> data;

    Field(const std::string &path) : data(Array<T>::MMap(path)) {}

    Field(const std::string &path, const ArrayShape &shape) : data(Array<T>::MMap(path, shape)) {}

    std::string_view data_type() const final {
      return GetTypeName<T>();
    }

    void Resize(const ArrayShape &shape) final {
      data.Resize(shape);
    }
  };

  StructArrayMeta meta_;
  std::vector<std::unique_ptr<FieldBase>> fields_;
  unordered_map<std::string_view, int> field_idx_;
  std::string path_;

  std::string GetMetaPath() const {
    return path_ + ".meta";
  }
};

template <class Struct>
class StructArray;

template <class... Ts>
class StructArray<std::tuple<Ts...>> : public StructArrayBase {
 public:
  static constexpr int NUM_FIELDS = sizeof...(Ts);
  static_assert(NUM_FIELDS > 0);

  using Struct = std::tuple<Ts...>;

  using StructArrayBase::field;

  template <size_t I>
  auto &field() {
    using T = std::remove_reference_t<decltype(std::get<I>(std::declval<Struct>()))>;
    return field<T>(I);
  }

  template <class... Args>
  Struct operator()(Args... indexes) const {
    return Get<Args...>(indexes...);
  }

  template <class... Args>
  Struct Get(Args... indexes) const {
    return GetRecord(std::make_index_sequence<NUM_FIELDS>(), indexes...);
  }

  template <class T, size_t N>
  void Set(const std::array<T, N> &indexes, const Struct &value) {
    Set(indexes, value, std::make_index_sequence<N>(), std::make_index_sequence<NUM_FIELDS>());
  }

  template <class Func>
  void ForEachField(Func &&f) {
    ForEachField(std::move(f), std::make_index_sequence<NUM_FIELDS>());
  }

  [[nodiscard]] static StructArray Load(const std::string &path) {
    return MMap(path);
  }

  // Open StructArray for read
  static StructArray MMap(const std::string &path,
                          const std::vector<std::string_view> &field_names = {}) {
    return MMap<false>(path, field_names, {}, std::make_index_sequence<NUM_FIELDS>());
  }

  // Open StructArray for write
  static StructArray MMap(const std::string &path,
                          const std::array<std::string_view, NUM_FIELDS> &field_names,
                          const ArrayShape &shape) {
    return MMap<true>(path, field_names, shape, std::make_index_sequence<NUM_FIELDS>());
  }

 private:
  template <bool writable, class Names, size_t... Is>
  static StructArray MMap(const std::string &path, const Names &names, const ArrayShape &shape,
                          std::index_sequence<Is...>) {
    StructArray array;
    array.path_ = path;
    array.fields_.reserve(NUM_FIELDS);
    auto meta_path = array.GetMetaPath();
    std::vector<std::string> field_names;
    if constexpr (writable) {
      field_names = {std::string(std::get<Is>(names))...};
      if (fs::exists(meta_path)) {
        array.meta_ = StructArrayMeta::Load(meta_path);
        ENSURE(array.meta_.fields == field_names,
               "StructArray fields mismatch, in meta: {}, expected: {} ({}). "
               "Please delete the corrupted data and rebuild.",
               array.meta_.fields, field_names, path);
      }

      (array.fields_.emplace_back(
           std::make_unique<Field<Ts>>(path + "._" + field_names[Is], shape)),
       ...);
    } else {
      ENSURE(fs::exists(meta_path), "Missing meta {}", meta_path);
      array.meta_ = StructArrayMeta::Load(meta_path);
      if (names.empty()) {
        field_names = array.meta_.fields;
      } else {
        for (auto &name : names) field_names.push_back(std::string(name));
        array.meta_.fields = field_names;
      }
      ENSURE2(static_cast<int>(field_names.size()) == NUM_FIELDS);
      (array.fields_.emplace_back(std::make_unique<Field<Ts>>(path + "._" + field_names[Is])), ...);
    }

    auto field_shapes = std::vector<ArrayShape>{array.field<Ts>(Is).shape()...};
    auto &array_shape = field_shapes[0];
    for (int i = 1; i < NUM_FIELDS; ++i) {
      ENSURE(field_shapes[i] == array_shape,
             "StructArray fields have different shape: {}({}) {}({}) ({}). "
             "Please delete the corrupted data and rebuild.",
             field_names[0], array_shape, field_names[i], field_shapes[i], path);
    }

    if constexpr (writable) {
      array.meta_.shape = array_shape;
      array.meta_.fields = field_names;
      array.meta_.Save(meta_path);
    } else if (!array.meta_.shape.empty()) {
      ENSURE(array.meta_.shape == array_shape,
             "StructArray shape mismatch, in meta: {}, from fields: {} ({}). "
             "Please delete the corrupted data and rebuild.",
             array.meta_.shape, array_shape, path);
    }
    for (int i = 0; i < array.num_fields(); ++i) {
      array.field_idx_[array.field_name(i)] = i;
    }
    return array;
  }

  template <size_t... Is, class... Args>
  Struct GetRecord(std::index_sequence<Is...>, Args... indexes) const {
    return std::make_tuple(field<Ts>(Is)(indexes...)...);
  }

  template <class T, size_t N, size_t... Is, size_t... Fs>
  void Set(const std::array<T, N> &indexes, const Struct &value, std::index_sequence<Is...>,
           std::index_sequence<Fs...>) {
    auto set_f = [&](auto &f, auto &v) { f(std::get<Is>(indexes)...) = v; };
    (set_f(field<Ts>(Fs), std::get<Fs>(value)), ...);
  }

  template <class Func, size_t... Is>
  void ForEachField(Func &&f, std::index_sequence<Is...>) {
    (f(field<Is>()), ...);
  }
};

}  // namespace yang
