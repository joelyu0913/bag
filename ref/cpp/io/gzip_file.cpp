#include "yang/io/gzip_file.h"

#include <zlib.h>

#include "yang/util/logging.h"

namespace yang {
namespace io {

struct GzipStream : z_stream {
  bool initialized = false;

  GzipStream() {
    zalloc = Z_NULL;
    zfree = Z_NULL;
    opaque = Z_NULL;
    avail_in = 0;
    next_in = Z_NULL;
  }

  bool Initialize() {
    if (inflateInit2(this, 15 + 32) == Z_OK) {
      initialized = true;
    }
    return initialized;
  }

  ~GzipStream() {
    if (initialized) {
      inflateEnd(this);
    }
  }
};

GzipFile::~GzipFile() {
  if (stream_) {
    delete stream_;
  }
}

void GzipFile::Initialize() {
  if (file_->error()) {
    SetError();
    return;
  }

  set_batch_read_size(32 * 1024);
  file_->set_min_buf_size(4 * 1024);
}

int GzipFile::InternalRead(void *buf, size_t size_to_read) {
  if (stream_ == nullptr) {
    stream_ = new GzipStream;
    if (!stream_->Initialize()) {
      SetError();
      LOG_ERROR("zlib error: {}", stream_->msg);
    }
  }
  if (file_->buf_size() == 0 && !file_->eof()) {
    file_->Forward(0);
  }

  size_t in_buf_size = file_->buf_size();
  // no more to read
  if (in_buf_size == 0) return 0;

  stream_->next_in = reinterpret_cast<decltype(stream_->next_in)>(file_->head());
  stream_->avail_in = in_buf_size;
  stream_->next_out = reinterpret_cast<decltype(stream_->next_out)>(buf);
  stream_->avail_out = size_to_read;
  // inflate will update next_in / avail_in / next_out / avail_out
  int ret = inflate(stream_, Z_NO_FLUSH);

  if (ret != Z_OK && ret != Z_STREAM_END) {
    SetError();
    LOG_ERROR("zlib error: {}", stream_->msg);
  }
  file_->Forward(in_buf_size - stream_->avail_in);
  int size_read = size_to_read - stream_->avail_out;
  if (ret == Z_STREAM_END) {
    delete stream_;
    stream_ = nullptr;
  }
  return size_read;
}

}  // namespace io
}  // namespace yang
