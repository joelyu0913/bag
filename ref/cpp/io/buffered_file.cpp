#include "yang/io/buffered_file.h"

#include "yang/util/error.h"

namespace yang {
namespace io {

void BufferedFile::ReadNext() {
  int ret = InternalRead(buf_ + tail_, buf_cap_ - tail_);
  if (ret < 0) {
    LOG_ERROR("Failed to read: {}", GetErrorString());
    SetError();
    return;
  }
  if (ret == 0) {
    eof_ = true;
  } else {
    tail_ += ret;
  }
}

bool BufferedFile::ReadLine(std::string &line) {
  if (eof()) {
    return false;
  }

  constexpr int block_size = 256;
  line.clear();
  line.reserve(block_size);

  while (true) {
    Forward(0, block_size);
    void *p = std::memchr(head(), '\n', buf_size());
    // handle last line without '\n'
    if (!p && eof_) {
      p = head() + buf_size();
    }
    char *ret = head();
    if (p) {
      int len = reinterpret_cast<char *>(p) - ret;
      line.append(ret, len);
      // move position pointer to after the newline char
      Forward(std::min<int>(len + 1, buf_size()), block_size);
      break;
    } else {
      // '\n' not found, continue
      line.append(ret, buf_size());
      Forward(buf_size(), block_size);
    }
  }
  return !line.empty();
}

void BufferedFile::Compact() {
  if (require_compaction()) {
    tail_ -= head_;
    std::memmove(buf_, buf_ + head_, tail_);
    head_ = 0;
  }
}

}  // namespace io
}  // namespace yang
