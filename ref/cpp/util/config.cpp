#include "yang/util/config.h"

#include <filesystem>

#include "yang/util/logging.h"

namespace fs = std::filesystem;

namespace yang {

static YAML::Node MergeNodes(YAML::Node a, YAML::Node b) {
  if (!b.IsMap()) {
    // If b is not a map, merge result is b, unless b is null
    return b.IsNull() ? a : b;
  }
  if (!a.IsMap()) {
    // If a is not a map, merge result is b
    return b;
  }
  if (!b.size()) {
    // If a is a map, and b is an empty map, return a
    return a;
  }
  // Create a new map 'c' with the same mappings as a, merged with b
  auto c = YAML::Node(YAML::NodeType::Map);
  for (auto n : a) {
    ENSURE(n.first.IsScalar(), "Only string keys are allowed");
    auto key = n.first.Scalar();
    auto t = YAML::Node(b[key]);
    if (t.IsDefined()) {
      c[key] = MergeNodes(n.second, t);
    } else {
      c[key] = n.second;
    }
  }
  // Add the mappings from 'b' not already in 'c'
  for (auto n : b) {
    ENSURE(n.first.IsScalar(), "Only string keys are allowed");
    auto key = n.first.Scalar();
    if (!c[key].IsDefined()) {
      c[key] = n.second;
    }
  }
  return c;
}

Config Config::LoadFile(const std::string &filename) {
  return Config(YAML::LoadFile(filename));
}

Config Config::LoadFile(std::istream &is) {
  return Config(YAML::Load(is));
}

Config Config::Load(const std::string &yaml) {
  return Config(YAML::Load(yaml));
}

void Config::Merge(const Config &other) {
  node_ = MergeNodes(node_, other.node_);
}

Config Config::LoadFiles(const std::vector<std::string> &files) {
  Config config;
  for (auto &file : files) {
    auto c = Config::LoadFile(file);
    auto current_dir = fs::path(file).parent_path().string();
    config.Merge(c);
  }
  return config;
}

std::string Config::ToYamlString() const {
  YAML::Emitter out;
  out << node_;
  return out.c_str();
}

}  // namespace yang
