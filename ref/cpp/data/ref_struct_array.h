#pragma once

#include <optional>

#include "yang/data/array.h"
#include "yang/data/block_struct_vector.h"
#include "yang/data/null.h"

namespace yang {

template <class T>
class RefFieldArray {
 public:
  using ValueArray = Array<T>;
  using RefArray = Array<int32_t>;

  RefFieldArray() : values_(nullptr), ref_(nullptr) {}

  RefFieldArray(const Array<T> &values, const RefArray &ref, const T &null_value)
      : values_(&values), ref_(&ref), null_value_(null_value) {}

  template <class... Args>
  const T &operator()(Args... indexes) const {
    auto data_idx = (*ref_)(indexes...);
    if (!IsNull(data_idx)) {
      return (*values_)(data_idx);
    } else {
      return null_value_;
    }
  }

  template <class... Args>
  const T &Get(Args... indexes) const {
    return (*this)(indexes...);
  }

  operator bool() const {
    return values_ != nullptr;
  }

 private:
  const Array<T> *values_;
  const RefArray *ref_;
  T null_value_;
};

template <class T>
class RefStructArray {
 public:
  using Struct = T;
  using Vector = BlockStructVector<T>;
  using RefArray = Array<int32_t>;

  static constexpr int NUM_FIELDS = Vector::NUM_FIELDS;

  RefStructArray() {}

  RefStructArray(RefStructArray &&other) {
    this->operator=(std::move(other));
  }

  RefStructArray &operator=(RefStructArray &&other) {
    data_ = std::move(other.data_);
    ref_ = std::move(other.ref_);
    return *this;
  }

  SizeType ndim() const {
    return ref_.ndim();
  }

  const ArrayShape &shape() const {
    return ref_.shape();
  }

  SizeType shape(int i) const {
    return ref_.shape(i);
  }

  SizeType num_fields() const {
    return data_.num_fields();
  }

  const std::vector<std::string> &field_names() const {
    return data_.field_names();
  }

  const std::string &field_name(int i) const {
    return data_.field_name(i);
  }

  int FindField(std::string_view name) const {
    return data_.FindField(name);
  }

  template <class... Args>
  bool HasValue(Args... indexes) const {
    return !IsNull(ref_(indexes...));
  }

  template <class... Args>
  std::optional<T> operator()(Args... indexes) const {
    auto data_idx = ref_(indexes...);
    if (!IsNull(data_idx)) {
      return data_[data_idx];
    } else {
      return {};
    }
  }

  template <class U, class... Args>
  const U &Get(SizeType f_idx, Args... indexes) const {
    auto data_idx = ref_(indexes...);
    if (!IsNull(data_idx)) {
      return data_.template Get<U>(data_idx, f_idx);
    } else {
      return null_v<U>;
    }
  }

  template <size_t F, class... Args>
  auto Get(Args... indexes) const {
    return Get<std::tuple_element_t<F, T>>(F, indexes...);
  }

  void StartBlock(int di) {
    data_.StartBlock(di);
  }

  void NextBlock() {
    data_.NextBlock();
  }

  void Set(int di, int ii, const T &v) {
    ENSURE2(ndim() == 2);
    if (di > 0) {
      auto data_idx = ref_(di - 1, ii);
      if (!IsNull(data_idx) && data_[data_idx] == v) {
        ref_(di, ii) = data_idx;
        return;
      }
    }
    ref_(di, ii) = PushBack(di, v);
  }

  void Set(int di, int ti, int ii, const T &v) {
    ENSURE2(ndim() == 3);
    if (di > 0 || ti > 0) {
      int prev_di;
      int prev_ti;
      if (ti > 0) {
        prev_di = di;
        prev_ti = ti - 1;
      } else {
        prev_di = di - 1;
        prev_ti = shape(1) - 1;
      }
      auto data_idx = ref_(prev_di, prev_ti, ii);
      if (!IsNull(data_idx) && data_[data_idx] == v) {
        ref_(di, ti, ii) = data_idx;
        return;
      }
    }
    ref_(di, ti, ii) = PushBack(di, v);
  }

  void FillNull(int di_begin, int di_end) {
    ref_.FillNull(di_begin, di_end);
  }

  template <class... Args>
  void Resize(Args... sizes) {
    ref_.Resize({static_cast<SizeType>(sizes)...});
  }

  // Open RefStructArray for read
  static RefStructArray MMap(const std::string &path,
                             const std::vector<std::string_view> &field_names = {}) {
    RefStructArray array;
    array.data_ = Vector::MMap(path + ".data", field_names);
    array.ref_ = RefArray::MMap(path + ".ref");
    return array;
  }

  // Open RefStructArray for write
  static RefStructArray MMap(const std::string &path,
                             const std::array<std::string_view, NUM_FIELDS> &field_names,
                             const ArrayShape &shape, SizeType init_size, SizeType extend_size) {
    ENSURE2(shape.size() >= 2);
    RefStructArray array;
    array.data_ = Vector::MMap(path + ".data", field_names, shape[0], init_size, extend_size);
    array.ref_ = RefArray::MMap(path + ".ref", shape);
    return array;
  }

  const Vector &data() const {
    return data_;
  }

  const RefArray &ref() const {
    return ref_;
  }

  template <class U>
  RefFieldArray<U> field(int i, const U &null_value = null_v<U>) const {
    return RefFieldArray<U>(data_.template field<U>(i), ref_, null_value);
  }

  template <class U>
  RefFieldArray<U> field(std::string_view name, const U &null_value = null_v<U>) const {
    return RefFieldArray<U>(data_.template field<U>(name), ref_, null_value);
  }

 private:
  Vector data_;
  RefArray ref_;

  SizeType PushBack(int di, const T &v) {
    return PushBack(di, v, std::make_index_sequence<NUM_FIELDS>());
  }

  template <size_t... Is>
  SizeType PushBack(int di, const T &v, std::index_sequence<Is...>) {
    if (di != data_.current_block()) StartBlock(di);
    data_.PushBack(std::get<Is>(v)...);
    return data_.block_end(di) - 1;
  }
};

}  // namespace yang
