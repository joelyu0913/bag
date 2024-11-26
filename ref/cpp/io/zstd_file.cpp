#include "yang/io/zstd_file.h"

#include "yang/util/logging.h"

namespace yang {
namespace io {

ZstdFile::ZstdFile(std::unique_ptr<BufferedFile> compressed)
    : BufferedFile(ZSTD_DStreamOutSize()), file_(std::move(compressed)) {
  Initialize();
}

ZstdFile::~ZstdFile() {
  if (!dstream_) return;

  ZSTD_freeDStream(dstream_);
  dstream_ = nullptr;
  free(in_buf_);
  in_buf_ = nullptr;
}

void ZstdFile::Initialize() {
  if (file_->error()) {
    SetError();
    return;
  }

  in_buf_size_ = ZSTD_DStreamInSize();
  in_buf_ = reinterpret_cast<char *>(malloc(in_buf_size_));
  dstream_ = ZSTD_createDStream();

  next_block_size_ = ZSTD_initDStream(dstream_);
  if (!CheckReturnCode(next_block_size_)) return;

  set_batch_read_size(128 * 1024);

  file_->set_min_buf_size(4 * 1024);
  file_->Forward(0);
}

int ZstdFile::InternalRead(void *buf, size_t size_to_read) {
  if (next_block_size_ == 0) return 0;
  int size = 0;
  while (size == 0) {
    ZSTD_inBuffer input = {file_->head(), file_->buf_size(), 0};
    ZSTD_outBuffer output = {buf, size_to_read, 0};
    next_block_size_ = ZSTD_decompressStream(dstream_, &output, &input);
    if (!CheckReturnCode(next_block_size_)) return -1;
    if (output.pos == 0 && next_block_size_ == 0) {
      return 0;
    }
    size = output.pos;
    file_->Forward(input.pos);
  }
  return size;
}

bool ZstdFile::CheckReturnCode(int ret) {
  if (ZSTD_isError(ret)) {
    LOG_ERROR("Zstd error: {}", ZSTD_getErrorName(ret));
    SetError();
    return false;
  }
  return true;
}

}  // namespace io
}  // namespace yang
