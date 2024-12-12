#include "yang/data/struct_array.h"

#include "yang/util/config.h"

namespace yang {

StructArrayMeta StructArrayMeta::Load(const std::string &path) {
  StructArrayMeta meta;
  try {
    auto meta_config = Config::LoadFile(path);
    meta.fields = meta_config.Get<std::vector<std::string>>("fields");
    for (auto d : meta_config.Get<std::vector<SizeType>>("shape")) {
      meta.shape.push_back(d);
    }
  } catch (const std::exception &ex) {
    throw MakeExcept<FatalError>("Failed to load meta {}: {}", path, ex.what());
  }
  return meta;
}

void StructArrayMeta::Save(const std::string &path) const {
  Config meta_config;
  meta_config.Set("shape", std::vector<SizeType>(shape.begin(), shape.end()));
  meta_config.Set("fields", fields);

  std::ofstream ofs(path);
  ofs << meta_config.ToYamlString();
  ENSURE2(ofs.good());
}

void StructArrayBase::Resize(const ArrayShape &new_shape) {
  for (auto &f : fields_) f->Resize(new_shape);
  meta_.shape = new_shape;
  meta_.Save(GetMetaPath());
}

}  // namespace yang
