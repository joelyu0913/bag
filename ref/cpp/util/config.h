#pragma once
#include <yaml-cpp/yaml.h>

#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "yang/base/exception.h"
#include "yang/util/fmt.h"

namespace yang {

struct ConfigException : Exception {
  using Exception::Exception;
};

namespace detail {
class ConfigIterator;
class ConfigIteratorValue;
}  // namespace detail

// A wrapper class around YAML::Node to provide encapsulation and better error messages
class Config {
 public:
  using Iterator = detail::ConfigIterator;

  Config() {}

  template <typename T>
  explicit Config(const T &value) : node_(value) {}

  static Config LoadFile(const std::string &filename);

  static Config LoadFile(std::istream &is);

  static Config Load(const std::string &yaml);

  const std::string &path() const {
    return path_;
  }

  void StripPath() {
    path_ = "";
  }

  template <typename Key>
  const Config operator[](const Key &key) const {
    if (node_.IsDefined()) {
      return Config(node_[key], GetSubPath(path_, key));
    } else {
      return Config(node_, GetSubPath(path_, key));
    }
  }

  template <typename T>
  T Get(const std::string &key) const {
    auto n = node_[key];
    if (!n) throw MakeExcept<std::runtime_error>("Config not found: {}", GetSubPath(path_, key));
    try {
      return n.as<T>();
    } catch (const YAML::Exception &e) {
      throw MakeExcept<ConfigException>("Failed to get config for key {}: {}",
                                        GetSubPath(path_, key), e.what());
    }
  }

  template <typename T>
  T Get(const std::string &key, const T &default_value) const {
    if (!node_.IsDefined()) {
      return default_value;
    }
    try {
      return node_[key].as<T>(default_value);
    } catch (const YAML::Exception &e) {
      throw MakeExcept<ConfigException>("Failed to get config for key {}: {}",
                                        GetSubPath(path_, key), e.what());
    }
  }

  template <typename T>
  void Set(const std::string &key, const T &value) {
    if (!node_.IsDefined()) {
      throw MakeExcept<ConfigException>("Cannot set value on invalid node {}", path_);
    }
    node_[key] = value;
  }

  template <typename T>
  void Set(const std::string &key, T &&value) {
    if (!node_.IsDefined()) {
      throw MakeExcept<ConfigException>("Cannot set value on invalid node {}", path_);
    }
    node_[key] = std::move(value);
  }

  template <typename T>
  T as() const {
    try {
      return node_.as<T>();
    } catch (const YAML::Exception &e) {
      throw MakeExcept<ConfigException>("Failed to get config {}: {}", path_, e.what());
    }
  }

  template <typename T>
  T as(const T &default_value) const {
    try {
      return node_.as<T>(default_value);
    } catch (const YAML::Exception &e) {
      throw MakeExcept<ConfigException>("Failed to get config {}: {}", path_, e.what());
    }
  }

  operator bool() const {
    return node_.IsDefined();
  }

  bool operator!() const {
    return !node_.IsDefined();
  }

  size_t size() const {
    return node_.size();
  }

  bool empty() const {
    return size() == 0;
  }

  Iterator begin() const;
  Iterator end() const;

  bool IsDefined() const {
    return node_.IsDefined();
  }

  bool IsNull() const {
    return node_.IsNull();
  }

  bool IsScalar() const {
    return node_.IsScalar();
  }

  bool IsSequence() const {
    return node_.IsSequence();
  }

  bool IsMap() const {
    return node_.IsMap();
  }

  std::string ToYamlString() const;

  // Access the underlying YAML Node

  const YAML::Node &node() const {
    return node_;
  }

  YAML::Node &mutable_node() {
    return node_;
  }

  // Rercusive merging
  void Merge(const Config &other);

  static Config LoadFiles(const std::vector<std::string> &files);

  template <class Key>
  static std::string GetSubPath(std::string_view path, const Key &key) {
    if (path.empty()) return fmt::format("{}", key);
    return fmt::format("{}/{}", path, key);
  }

 private:
  friend detail::ConfigIteratorValue;

  YAML::Node node_;
  std::string path_;

  template <typename T>
  Config(const T &value, std::string_view path) : node_(value), path_(path) {}
};

inline std::ostream &operator<<(std::ostream &os, const Config &config) {
  return os << config.ToYamlString();
}

namespace detail {

struct ConfigIteratorValue : public Config, public std::pair<Config, Config> {
  explicit ConfigIteratorValue(const YAML::Node::const_iterator::value_type &v,
                               std::string_view idx_path)
      : Config(v, idx_path), std::pair<Config, Config>(Config(v.first), Config(v.second)) {}
};

class ConfigIterator {
 public:
  struct ValueProxy {
    ConfigIteratorValue value;

    explicit ValueProxy(const YAML::Node::const_iterator::value_type &v, std::string_view idx_path)
        : value(v, idx_path) {}

    ConfigIteratorValue &operator*() {
      return value;
    }

    ConfigIteratorValue *operator->() {
      return &value;
    }
  };

  ConfigIterator() {}
  ConfigIterator(const ConfigIterator &other) = default;
  explicit ConfigIterator(YAML::Node::const_iterator it, std::string_view path, int idx = 0)
      : underlying_(it), path_(path), idx_(idx) {}

  ConfigIterator &operator=(const ConfigIterator &other) = default;

  ValueProxy operator->() const {
    return ValueProxy(*underlying_, Config::GetSubPath(path_, idx_));
  }

  ConfigIteratorValue operator*() const {
    return ConfigIteratorValue(*underlying_, Config::GetSubPath(path_, idx_));
  }

  ConfigIterator &operator++() {
    increment();
    return *this;
  }
  ConfigIterator operator++(int) {
    ConfigIterator temp(*this);
    increment();
    return temp;
  }

  bool operator==(const ConfigIterator &other) const {
    return this->equal(other);
  }
  bool operator!=(const ConfigIterator &other) const {
    return !this->equal(other);
  }

 private:
  YAML::Node::const_iterator underlying_;
  std::string path_;
  int idx_ = 0;

  void increment() {
    ++underlying_;
    ++idx_;
  }

  bool equal(const ConfigIterator &other) const {
    return underlying_ == other.underlying_;
  }
};

}  // namespace detail

inline Config::Iterator Config::begin() const {
  return Iterator(node_.begin(), path_);
}

inline Config::Iterator Config::end() const {
  return Iterator(node_.end(), path_);
}

}  // namespace yang

namespace YAML {
template <>
struct convert<yang::Config> {
  static Node encode(const yang::Config &rhs) {
    return rhs.node();
  }

  static bool decode(const Node &node, yang::Config &rhs) {
    rhs = yang::Config(node);
    return true;
  }
};
}  // namespace YAML

#ifdef FMT_OSTREAM_FORMATTER
template <>
struct fmt::formatter<yang::Config> : ostream_formatter {};
#endif
