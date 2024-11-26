#pragma once

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

#include "yang/base/exception.h"
#include "yang/util/logging.h"

namespace yang {
namespace io {

struct BufferedFileException : Exception {
  using Exception::Exception;
};

class BufferedFile {
 public:
  explicit BufferedFile(size_t buf_cap) : buf_cap_(buf_cap), buf_(new char[buf_cap]) {}

  virtual ~BufferedFile() {
    delete[] buf_;
  }

  bool ReadLine(std::string &line);

  void Seek(size_t offset) {
    if (!eof()) {
      SeekImpl(offset);
      head_ = tail_ = 0;
      Forward(0);
    }
  }

  void Forward(int n) {
    Forward(n, min_buf_size());
  }

  // move head forward and ensure the buffer has at least min_size bytes when not eof
  void Forward(int n, size_t min_size) {
    if (n <= static_cast<int64_t>(buf_size())) {
      head_ += n;

      while (buf_size() < min_size && !eof_) {
        Compact();
        ReadNext();
      }
    } else {
      LOG_ERROR("Invalid forward: {} > {}", n, buf_size());
      throw BufferedFileException("Invalid forward");
    }
  }

  void Prefetch() {
    if (!eof_ && tail_ < buf_cap()) {
      ReadNext();
    }
  }

  char *head() const {
    return buf_ + head_;
  }

  size_t min_buf_size() const {
    return min_buf_size_;
  }
  void set_min_buf_size(size_t v) {
    min_buf_size_ = v;
  }

  size_t buf_size() const {
    return tail_ - head_;
  }

  size_t buf_cap() const {
    return buf_cap_;
  }

  bool eof() const {
    return eof_ && head_ == tail_;
  }

  bool last_buf() const {
    return eof_;
  }

  bool error() const {
    return error_;
  }

  size_t raw_head() const {
    return head_;
  }

 protected:
  static constexpr int DEFAULT_MOVE_SIZE = 1024;

  size_t batch_read_size_ = DEFAULT_MOVE_SIZE;
  size_t min_buf_size_ = DEFAULT_MOVE_SIZE;
  size_t buf_cap_;
  char *buf_;

  size_t head_ = 0;
  size_t tail_ = 0;

  // end of underlying file, there may still be data in buf_
  bool eof_ = false;
  bool error_ = false;

  virtual int InternalRead(void *buf, size_t size) = 0;

  size_t batch_read_size() const {
    return batch_read_size_;
  }

  void set_batch_read_size(size_t v) {
    batch_read_size_ = v;
  }

  bool require_compaction() const {
    return tail_ + batch_read_size_ > buf_cap();
  }

  void ReadNext();

  void Compact();

  virtual void SeekImpl(size_t) {
    throw BufferedFileException("Seek not implemented");
  }

  void SetError() {
    error_ = true;
    eof_ = true;
  }
};

}  // namespace io
}  // namespace yang
