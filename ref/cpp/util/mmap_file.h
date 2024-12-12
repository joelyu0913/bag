#pragma once

#include <sys/mman.h>
#include <sys/types.h>

#include <cstdint>
#include <string>

namespace yang {

class MMapFile {
 public:
  struct Options {
    std::string filename;  // empty filename means anonymous mapping
    bool writable = true;
    bool lock = false;   // use mlock to lock the address into memory
    bool create = true;  // create if the file does not exist
    mode_t mode = 0644;  // access mode
    size_t size = 0;     // use a non-zero value to truncate the file
  };

  MMapFile() {}
  ~MMapFile();

  MMapFile(MMapFile &&other) {
    this->operator=(std::move(other));
  }

  MMapFile &operator=(MMapFile &&other);

  void Initialize(const Options &options);

  void Initialize(const Options &options, bool &truncated);

  uint8_t *addr() const {
    return addr_;
  }

  size_t size() const {
    return size_;
  }

  void Reset();

 private:
  uint8_t *addr_ = nullptr;
  size_t size_;
};

}  // namespace yang
