#pragma once

#include <memory>

#include "yang/io/buffered_file.h"

namespace yang {
namespace io {

// Forward declaration
struct GzipStream;

class GzipFile : public BufferedFile {
 public:
  static constexpr size_t BLOCK_SIZE = 16 * 1024 * 1024;

  explicit GzipFile(std::unique_ptr<BufferedFile> compressed)
      : BufferedFile(BLOCK_SIZE), file_(std::move(compressed)) {
    Initialize();
  }

  ~GzipFile();

 private:
  std::unique_ptr<BufferedFile> file_;
  GzipStream *stream_ = nullptr;

  void Initialize();

  int InternalRead(void *buf, size_t size_to_read) override;
};

}  // namespace io
}  // namespace yang
