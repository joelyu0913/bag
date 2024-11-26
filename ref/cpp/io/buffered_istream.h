#pragma once

#include <stdio.h>

#include <iostream>

#include "yang/io/buffered_file.h"

namespace yang {
namespace io {

// BufferedFile -> std::istream Adapter

class BufferedIStreamBuf : public std::streambuf {
 public:
  explicit BufferedIStreamBuf(std::unique_ptr<BufferedFile> file) : file_(std::move(file)) {
    setg(file_->head(), file_->head(), file_->head());
  }

  BufferedIStreamBuf(BufferedIStreamBuf &&other) {
    this->operator=(std::move(other));
  }

  BufferedIStreamBuf &operator=(BufferedIStreamBuf &&other) {
    file_ = std::move(other.file_);
    std::streambuf::swap(other);
    return *this;
  }

  int_type underflow() override {
    if (Forward()) {
      return traits_type::to_int_type(*gptr());
    } else {
      return traits_type::eof();
    }
  }

  BufferedFile *file() {
    return file_.get();
  }

 private:
  std::unique_ptr<BufferedFile> file_;

  bool Forward() {
    file_->Forward(gptr() - eback());
    setg(file_->head(), file_->head(), file_->head() + file_->buf_size());
    return !file_->eof();
  }
};

class BufferedIStream : public std::istream {
 public:
  explicit BufferedIStream(std::unique_ptr<BufferedFile> file)
      : std::istream(nullptr), streambuf_(std::move(file)) {
    rdbuf(&streambuf_);
    if (streambuf_.file()->error()) {
      setstate(std::ios_base::badbit);
    }
  }

  BufferedIStream(BufferedIStream &&other)
      : std::istream(nullptr), streambuf_(std::move(other.streambuf_)) {
    rdbuf(&streambuf_);
    clear(other.rdstate());
  }

  BufferedIStream &operator=(BufferedIStream &&other) {
    streambuf_ = std::move(other.streambuf_);
    rdbuf(&streambuf_);
    clear(other.rdstate());
    return *this;
  }

 private:
  BufferedIStreamBuf streambuf_;
};

}  // namespace io
}  // namespace yang
