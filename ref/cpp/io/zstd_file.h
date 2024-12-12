#pragma once

#include <zstd.h>

#include <memory>

#include "yang/io/buffered_file.h"

namespace yang {
namespace io {

class ZstdFile : public BufferedFile {
 public:
  static constexpr size_t BLOCK_SIZE = 16 * 1024 * 1024;

  explicit ZstdFile(std::unique_ptr<BufferedFile> compressed);

  ~ZstdFile();

 private:
  std::unique_ptr<BufferedFile> file_;
  ZSTD_DStream *dstream_;
  size_t in_buf_size_;
  char *in_buf_;
  size_t next_block_size_;

  void Initialize();

  int InternalRead(void *buf, size_t size_to_read) override;

  bool CheckReturnCode(int ret);
};

}  // namespace io
}  // namespace yang
