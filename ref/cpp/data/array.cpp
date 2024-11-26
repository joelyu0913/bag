#include "yang/data/array.h"

#include <cstring>
#include <fstream>
#include <vector>

#include "yang/base/exception.h"
#include "yang/util/config.h"

namespace yang {
namespace detail {

static void CopyArrayImpl(uint8_t *dest, const ArrayShape &dest_shape,
                          const ArrayShape &dest_stride, uint8_t *src, const ArrayShape &src_shape,
                          const ArrayShape &src_stride, SizeType item_size,
                          const ArrayBackend::Filler &filler, int dim = 0) {
  int n = std::min(src_shape[dim], dest_shape[dim]);
  if (dim == static_cast<int>(dest_stride.size()) - 1) {
    std::memcpy(dest, src, n * dest_stride[dim] * item_size);
  } else {
    for (int i = 0; i < n; ++i) {
      CopyArrayImpl(dest + i * dest_stride[dim] * item_size, dest_shape, dest_stride,
                    src + i * src_stride[dim] * item_size, src_shape, src_stride, item_size, filler,
                    dim + 1);
    }
  }
  if (n < dest_shape[dim]) {
    filler(dest + n * dest_stride[dim] * item_size, (dest_shape[dim] - n) * dest_stride[dim]);
  }
}

void ArrayBackend::CopyArray(void *dest, const ArrayShape &dest_shape, void *src,
                             const ArrayShape &src_shape, SizeType item_size,
                             const ArrayBackend::Filler &filler) {
  if (dest == nullptr) return;

  if (src == nullptr) {
    SizeType n = 1;
    for (auto &d : dest_shape) n *= d;
    filler(dest, n);
    return;
  }

  ENSURE(dest_shape.size() == src_shape.size(), "CopyArray dimension mismatch");

  // if the inner dimensions are the same, treat them as a single block
  int stride_end = dest_shape.size() - 1;
  SizeType block_size = 1;
  while (stride_end > 0 && dest_shape[stride_end] == src_shape[stride_end]) {
    block_size *= dest_shape[stride_end];
    stride_end--;
  }
  if (stride_end < 0) return;  // same shape
  SizeType stride = block_size;
  ArrayShape dest_stride(stride_end + 1);
  for (int i = stride_end; i >= 0; --i) {
    dest_stride[i] = stride;
    stride *= dest_shape[i];
  }
  stride = block_size;
  ArrayShape src_stride(stride_end + 1);
  for (int i = stride_end; i >= 0; --i) {
    src_stride[i] = stride;
    stride *= src_shape[i];
  }
  CopyArrayImpl(reinterpret_cast<uint8_t *>(dest), dest_shape, dest_stride,
                reinterpret_cast<uint8_t *>(src), src_shape, src_stride, item_size, filler, 0);
}

void DefaultArrayBackend::Resize(const ArrayShape &old_shape, const ArrayShape &new_shape,
                                 SizeType item_size, const Filler &filler) {
  if (old_shape == new_shape) return;

  SizeType new_size = item_size;
  for (auto &d : new_shape) new_size *= d;
  ENSURE(new_size > 0, "Resize to 0");
  auto new_data = std::malloc(new_size);
  CopyArray(new_data, new_shape, data_, old_shape, item_size, filler);
  Free();
  data_ = new_data;
  size_ = new_size;
}

void DefaultArrayBackend::ResizeRaw(const ArrayShape &new_shape, SizeType item_size) {
  SizeType new_size = item_size;
  for (auto &d : new_shape) new_size *= d;
  if (size_ != new_size) {
    Free();
    if (new_size == 0) {
      return;
    }
    data_ = std::malloc(new_size);
    ENSURE(data_ != nullptr, "malloc failed");
    size_ = new_size;
  }
}

void DefaultArrayBackend::Free() {
  if (data_) {
    std::free(data_);
    data_ = nullptr;
    size_ = 0;
  }
}

MMapArrayBackend::MMapArrayBackend(const std::string &path, bool writable,
                                   const std::string &item_type, SizeType item_size,
                                   const ArrayShape &shape, const Filler &filler)
    : path_(path), writable_(writable) {
  if (fs::exists(path_)) {
    // existing file
    if (meta_.Load(GetMetaPath())) {
      if (!meta_.shape.empty()) {
        meta_.Check(item_type, item_size);
        if (writable && !shape.empty() && shape != meta_.shape) {
          // skip loading file to avoid unnecessary munmap
          Resize(ArrayShape(meta_.shape), shape, item_size, filler);
        } else {
          LoadFile();
        }
        return;
      }
    } else if (!writable) {
      throw IoError("Failed to load array from " + path_);
    }
  } else if (!writable) {
    throw IoError("Array not found: " + path_);
  }

  // new file
  meta_.item_type = item_type;
  meta_.item_size = item_size;
  if (writable && !shape.empty()) {
    Resize({}, shape, item_size, filler, true);
    LOG_DEBUG("Created {} ({})", path_, shape);
  }
}

void MMapArrayBackend::LoadFile() {
  MMapFile::Options options;
  options.filename = path_;
  options.writable = writable_;
  options.lock = false;
  options.create = false;
  file_.Initialize(options);
  LOG_DEBUG("Loaded {} ({})", path_, shape());
}

void MMapArrayBackend::Resize(const ArrayShape &old_shape, const ArrayShape &new_shape,
                              SizeType item_size, const Filler &filler) {
  Resize(old_shape, new_shape, item_size, filler, false);
}

void MMapArrayBackend::Resize(const ArrayShape &old_shape, const ArrayShape &new_shape,
                              SizeType item_size, const Filler &filler, bool create) {
  ENSURE(writable_, "not writable");
  ENSURE(!new_shape.empty(), "new_shape is empty");
  ENSURE(meta_.shape.empty() || item_size == meta_.item_size, "item_size mismatch");

  bool extend_only = true;
  if (old_shape.size() > 0) {
    ENSURE(old_shape.size() == new_shape.size(), "Resize dimension mismatch, old: {}, new: {}",
           old_shape.size(), new_shape.size());
    for (int i = 1; i < static_cast<int>(new_shape.size()); ++i) {
      if (old_shape[i] != new_shape[i]) {
        extend_only = false;
        break;
      }
    }
  }

  fs::create_directories(fs::path(path_).parent_path());
  MMapFile::Options options;
  options.filename = path_;
  options.writable = writable_;
  options.lock = false;
  options.create = true;
  SizeType new_size = 1;
  for (auto &d : new_shape) new_size *= d;
  options.size = new_size * item_size;
  if (extend_only) {
    file_.Reset();
    file_.Initialize(options);
    ENSURE(file_.size() == options.size, "file size mismatch");
    SizeType old_len = old_shape.empty() ? 0 : old_shape[0];
    if (new_shape[0] > old_len) {
      SizeType block_size = new_size / new_shape[0];
      SizeType n = (new_shape[0] - old_len) * block_size;
      filler(file_.addr() + old_len * block_size * item_size, n);
    }
  } else {
    if (!file_.addr()) {
      LoadFile();
    }
    MMapFile new_file;
    options.filename += ".new";
    new_file.Initialize(options);
    CopyArray(new_file.addr(), new_shape, file_.addr(), old_shape, item_size, filler);
    fs::rename(options.filename, path_);
    file_ = std::move(new_file);
  }
  meta_.shape = new_shape;
  meta_.Save(GetMetaPath());
  if (!create) {
    LOG_DEBUG("Resized {} ({}) to ({})", path_, old_shape, new_shape);
  }
}

void MMapArrayBackend::ResizeRaw(const ArrayShape &new_shape, SizeType item_size) {
  ENSURE(writable_, "not writable");
  if (meta_.shape == new_shape) return;

  SizeType new_size = item_size;
  for (auto &d : new_shape) new_size *= d;

  if (new_size != static_cast<SizeType>(file_.size())) {
    fs::create_directories(fs::path(path_).parent_path());
    MMapFile::Options options;
    options.filename = path_;
    options.writable = writable_;
    options.lock = false;
    options.create = true;
    options.size = new_size;
    file_.Reset();
    file_.Initialize(options);
  }
  meta_.shape = new_shape;
  meta_.Save(GetMetaPath());
}

bool ArrayMeta::Load(const std::string &path) {
  if (fs::exists(path)) {
    try {
      auto meta_config = Config::LoadFile(path);
      item_type = meta_config.Get<std::string>("item_type");
      item_size = meta_config.Get<SizeType>("item_size");
      shape.clear();
      for (auto d : meta_config.Get<std::vector<SizeType>>("shape")) {
        shape.push_back(d);
      }
      return true;
    } catch (const ConfigException &ex) {
      LOG_FATAL("array meta corrupted: {}, {}", ex.what(), path);
    }
  }
  return false;
}

void ArrayMeta::Save(const std::string &path) const {
  Config meta_config;
  meta_config.Set("item_type", item_type);
  meta_config.Set("item_size", item_size);
  meta_config.Set("shape", std::vector<SizeType>(shape.begin(), shape.end()));

  std::ofstream ofs(path);
  ofs << meta_config.ToYamlString();
  ENSURE2(ofs.good());
}

void ArrayMeta::Check(const std::string_view &ex_item_type, SizeType ex_item_size) {
  ENSURE(item_type == ex_item_type, "Array item_type mismatch, expected: {}, in meta: {}",
         ex_item_type, item_type);
  ENSURE(item_size == ex_item_size, "Array item_size mismatch, expected: {}, in meta: {}",
         ex_item_size, item_size);
}

}  // namespace detail

void ArrayBase::Copy(const std::string &from, const std::string &to) {
  fs::create_directories(fs::path(to).parent_path());
  fs::copy_file(from, to, fs::copy_options::overwrite_existing);
  fs::copy_file(from + ".meta", to + ".meta", fs::copy_options::overwrite_existing);
}

std::string ArrayBase::GetItemType(std::string_view array_path) {
  auto meta_path = std::string(array_path) + ".meta";
  detail::ArrayMeta meta;
  ENSURE(meta.Load(meta_path), "Missing array meta {}", meta_path);
  return meta.item_type;
}

}  // namespace yang
