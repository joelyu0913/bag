#include "yang/data/struct_vector.h"

#include "yang/util/config.h"

namespace yang {

bool StructVectorMeta::Load(const std::string &path) {
  if (fs::exists(path)) {
    auto meta_config = Config::LoadFile(path);
    size = meta_config.Get<SizeType>("size");
    fields = meta_config.Get<std::vector<std::string>>("fields");
    return true;
  }
  return false;
}

void StructVectorMeta::Save(const std::string &path) const {
  Config meta_config;
  meta_config.Set("size", size);
  meta_config.Set("fields", fields);

  std::ofstream ofs(path);
  ofs << meta_config.ToYamlString();
}

void StructVectorBase::Extend(SizeType extend_by) {
  ENSURE2(extend_by > 0);
  auto new_size = size() + extend_by;
  LOG_DEBUG("Extending {}: {} -> {}", path_, size(), new_size);
  for (auto &f : fields_) {
    f->Resize({new_size});
  }
  meta_.size = new_size;
  meta_.Save(GetMetaPath());
}

}  // namespace yang
