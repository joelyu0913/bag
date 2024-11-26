#pragma once

#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>

#include "yang/io/buffered_file.h"

namespace yang {
namespace io {

class LocalFile : public BufferedFile {
 public:
  // 32KB, use a small buffer since we are using blocking read
  static constexpr size_t BUFFER_SIZE = 32 * 1024;

  explicit LocalFile(const std::string &filename, size_t buf_size = BUFFER_SIZE)
      : BufferedFile(buf_size) {
    fp_ = fopen(filename.c_str(), "r");
    if (fp_ == nullptr) {
      fd_ = -1;
      SetError();
    } else {
      fd_ = fileno(fp_);
    }
    Forward(0);
  }

  LocalFile(const LocalFile &other) = delete;
  LocalFile &operator=(const LocalFile &other) = delete;

  ~LocalFile() {
    if (fp_) {
      fclose(fp_);
    }
  }

  static bool Exists(const std::string &filename) {
    return std::filesystem::exists(filename);
  }

 protected:
  int InternalRead(void *buf, size_t size_to_read) override {
    return read(fd_, buf, size_to_read);
  }

  void SeekImpl(size_t offset) override {
    fseek(fp_, offset, SEEK_SET);
  }

 private:
  FILE *fp_ = nullptr;
  int fd_ = -1;
};

}  // namespace io
}  // namespace yang
