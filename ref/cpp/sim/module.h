#pragma once

#include <string>

#include "yang/base/valid.h"
#include "yang/data/array.h"
#include "yang/data/ref_struct_array.h"
#include "yang/data/struct_array.h"
#include "yang/sim/env.h"
#include "yang/util/config.h"
#include "yang/util/factory_registry.h"
#include "yang/util/logging.h"

namespace yang {

class Module {
 public:
  virtual ~Module() {}

  const std::string &name() const {
    return name_;
  }

  const Env &env() const {
    return *env_;
  }

  const DataDirectory &cache_dir() const {
    return env_->cache_dir();
  }

  DataCache &data_cache() const {
    return env_->data_cache();
  }

  const UnivIndex &univ() const {
    return env_->univ();
  }

  int univ_size() const {
    return univ().size();
  }

  int max_univ_size() const {
    return env().max_univ_size();
  }

  const DateTimeIndex &datetimes() const {
    return env_->datetimes();
  }

  int64_t datetime(int i) const {
    return datetimes()[i];
  }

  int datetimes_size() const {
    return datetimes().size();
  }

  int max_datetimes_size() const {
    return env().max_datetimes_size();
  }

  const DateTimeIndex &dates() const {
    return env_->dates();
  }

  int64_t date(int i) const {
    return dates()[i];
  }

  int dates_size() const {
    return dates().size();
  }

  int max_dates_size() const {
    return env().max_dates_size();
  }

  const TimeIndex &intraday_times() const {
    return env_->intraday_times();
  }

  int intraday_time(int i) const {
    return intraday_times()[i];
  }

  const TimeIndex &taq_times() const {
    return env_->taq_times();
  }

  int taq_time(int i) const {
    return taq_times()[i];
  }

  const Config &config() const {
    return config_;
  }

  template <class T>
  T config(const std::string &key) const {
    return config_.Get<T>(key);
  }

  template <class T>
  T config(const std::string &key, const T &default_value) const {
    return config_.Get<T>(key, default_value);
  }

  int start_di() const {
    return env_->start_di();
  }

  int end_di() const {
    return env_->end_di();
  }

  RunStage stage() const {
    return stage_;
  }

  void set_stage(RunStage stage) {
    stage_ = stage;
  }

  virtual void Initialize(const std::string &name, const Config &config, const Env *env);

  void Run();

  template <class T>
  Array<T> WriteArray(std::string_view mod, std::string_view array_name, const ArrayShape &shape,
                      const T &null = null_v<T>, bool fill_null = false) {
    return WriteArrayImpl<T>(cache_dir().GetWritePath(mod, array_name), shape, null, fill_null);
  }

  template <class T>
  Array<T> WriteArray(std::string_view array_name, const ArrayShape &shape,
                      const T &null = null_v<T>, bool fill_null = false) {
    return WriteArrayImpl<T>(GetDataPath<false>(array_name), shape, null, fill_null);
  }

  template <class T>
  Array<T> WriteArray(std::string_view mod, std::string_view array_name, const T &null = null_v<T>,
                      bool fill_null = true) {
    return WriteArray<T>(mod, array_name, {env_->max_dates_size(), env_->max_univ_size()}, null,
                         fill_null);
  }

  template <class T>
  Array<T> WriteArray(std::string_view array_name, const T &null = null_v<T>,
                      bool fill_null = true) {
    return WriteArray<T>(array_name, {env_->max_dates_size(), env_->max_univ_size()}, null,
                         fill_null);
  }

  template <class T, class Fields>
  StructArray<T> WriteStructArray(std::string_view mod, std::string_view array_name,
                                  const Fields &fields, const ArrayShape &shape,
                                  bool fill_null = false) {
    return WriteStructArrayImpl<T>(cache_dir().GetWritePath(mod, array_name), fields, shape,
                                   fill_null);
  }

  template <class T, class Fields>
  StructArray<T> WriteStructArray(std::string_view array_name, const Fields &fields,
                                  const ArrayShape &shape, bool fill_null = false) {
    return WriteStructArrayImpl<T>(GetDataPath<false>(array_name), fields, shape, fill_null);
  }

  template <class T, class Fields>
  StructArray<T> WriteStructArray(std::string_view mod, std::string_view array_name,
                                  const Fields &fields, bool fill_null = true) {
    return WriteStructArray<T>(mod, array_name, fields,
                               {env_->max_dates_size(), env_->max_univ_size()}, fill_null);
  }

  template <class T, class Fields>
  StructArray<T> WriteStructArray(std::string_view array_name, const Fields &fields,
                                  bool fill_null = true) {
    return WriteStructArray<T>(array_name, fields, {env_->max_dates_size(), env_->max_univ_size()},
                               fill_null);
  }

  template <class T, class Fields>
  RefStructArray<T> WriteRefStructArray(std::string_view mod, std::string_view array_name,
                                        const Fields &fields, const ArrayShape &shape,
                                        SizeType init_size, SizeType extend_size,
                                        bool fill_null = false) {
    return WriteRefStructArrayImpl<T>(cache_dir().GetWritePath(mod, array_name), fields, shape,
                                      init_size, extend_size, fill_null);
  }

  template <class T, class Fields>
  RefStructArray<T> WriteRefStructArray(std::string_view array_name, const Fields &fields,
                                        const ArrayShape &shape, SizeType init_size,
                                        SizeType extend_size, bool fill_null = false) {
    return WriteRefStructArrayImpl<T>(GetDataPath<false>(array_name), fields, shape, init_size,
                                      extend_size, fill_null);
  }

  template <class T, class Fields>
  RefStructArray<T> WriteRefStructArray(std::string_view mod, std::string_view array_name,
                                        const Fields &fields, bool fill_null = true) {
    return WriteRefStructArray<T>(mod, array_name, fields,
                                  {env_->max_dates_size(), env_->max_univ_size()}, 100000, 10000,
                                  fill_null);
  }

  template <class T, class Fields>
  RefStructArray<T> WriteRefStructArray(std::string_view array_name, const Fields &fields,
                                        bool fill_null = true) {
    return WriteRefStructArray<T>(array_name, fields,
                                  {env_->max_dates_size(), env_->max_univ_size()}, 100000, 10000,
                                  fill_null);
  }

  template <class T, class... Args>
  const T *ReadData(Args &&...args) {
    return env_->ReadData<T>(std::forward<Args>(args)...);
  }

  template <class T, class... Args>
  const auto *ReadArray(Args &&...args) {
    return env_->ReadData<yang::Array<T>>(std::forward<Args>(args)...);
  }

  template <class T>
  static bool IsValid(T v) {
    return ::yang::IsValid(v);
  }

 protected:
  std::string name_;
  Config config_;
  const Env *env_ = nullptr;
  RunStage stage_ = RunStage::INTRADAY;

  virtual void BeforeRun() {}

  virtual void RunImpl() {}

  virtual void AfterRun() {}

  template <bool READ_ONLY>
  std::string GetDataPath(std::string_view data_name) {
    if (data_name.find('/') == std::string_view::npos) {
      if constexpr (READ_ONLY) {
        return cache_dir().GetReadPath(name(), data_name);
      } else {
        return cache_dir().GetWritePath(name(), data_name);
      }
    } else {
      if constexpr (READ_ONLY) {
        return cache_dir().GetReadPath(data_name);
      } else {
        return cache_dir().GetWritePath(data_name);
      }
    }
  }

  template <class T>
  Array<T> WriteArrayImpl(const std::string &path, const ArrayShape &shape,
                          const T &null = null_v<T>, bool fill_null = false) {
    auto arr = Array<T>::MMap(path, shape, null);
    if (fill_null) arr.FillNull(start_di(), end_di());
    return arr;
  }

  template <class T, class Fields>
  StructArray<T> WriteStructArrayImpl(const std::string &path, const Fields &fields,
                                      const ArrayShape &shape, bool fill_null = false) {
    auto arr = StructArray<T>::MMap(path, fields, shape);
    if (fill_null) {
      arr.ForEachField([this](auto &f) { f.FillNull(start_di(), end_di()); });
    }
    return arr;
  }

  template <class T, class Fields>
  RefStructArray<T> WriteRefStructArrayImpl(const std::string &path, const Fields &fields,
                                            const ArrayShape &shape, SizeType init_size,
                                            SizeType extend_size, bool fill_null = false) {
    auto arr = RefStructArray<T>::MMap(path, fields, shape, init_size, extend_size);
    if (fill_null) {
      arr.FillNull(start_di(), end_di());
    }
    return arr;
  }
};

#define REGISTER_MODULE(cls) REGISTER_FACTORY("module", cls, cls, #cls)
}  // namespace yang
