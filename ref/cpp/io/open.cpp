#include "yang/io/open.h"

#include "yang/io/gzip_file.h"
#include "yang/io/local_file.h"
#include "yang/io/zstd_file.h"

namespace yang {
namespace io {

static std::unique_ptr<BufferedFile> OpenBufferedFilePlain(const std::string &filename) {
  return std::make_unique<LocalFile>(filename);
}

std::unique_ptr<BufferedFile> OpenBufferedFile(const std::string &filename) {
  auto file = OpenBufferedFilePlain(filename);
  if (filename.ends_with(".zst")) {
    return std::make_unique<ZstdFile>(std::move(file));
  } else if (filename.ends_with(".gz")) {
    return std::make_unique<GzipFile>(std::move(file));
  } else {
    return file;
  }
}

BufferedIStream OpenBufferedStream(const std::string &filename) {
  return BufferedIStream(OpenBufferedFile(filename));
}

bool Exists(const std::string &filename) {
  return LocalFile::Exists(filename);
}

}  // namespace io
}  // namespace yang
