#include "yang/util/mmap_file.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>

#include "yang/util/logging.h"

namespace yang {

MMapFile::~MMapFile() {
  Reset();
}

MMapFile &MMapFile::operator=(MMapFile &&other) {
  Reset();
  addr_ = other.addr_;
  size_ = other.size_;
  other.addr_ = nullptr;
  return *this;
}

void MMapFile::Initialize(const Options &options) {
  bool truncated;
  Initialize(options, truncated);
}

void MMapFile::Initialize(const Options &options, bool &truncated) {
  ENSURE(addr_ == nullptr, "MMapFile already initialized");

  int prot = PROT_READ;
  if (options.writable) {
    prot |= PROT_WRITE;
  }

  if (options.filename.empty()) {
    int map_flags = MAP_ANONYMOUS | MAP_SHARED;
    size_ = options.size;
    addr_ = reinterpret_cast<uint8_t *>(mmap(nullptr, size_, prot, map_flags, 0, 0));
    if (addr_ == MAP_FAILED) LOG_FATAL("Failed to mmap: {}", GetErrorString());
    truncated = true;
  } else {
    int map_flags = MAP_SHARED;
    int file_flags = 0;
    if (options.writable) {
      file_flags = O_RDWR;
      if (options.create) {
        file_flags |= O_CREAT;
      }
    } else {
      file_flags = O_RDONLY;
    }
    int fd = open(options.filename.c_str(), file_flags, options.mode);
    if (fd == -1) LOG_FATAL("Failed to open file {}: {}", options.filename, GetErrorString());

    struct stat st;
    if (fstat(fd, &st)) LOG_FATAL("Failed to fstat {}: {}", options.filename, GetErrorString());
    if (options.writable && (st.st_mode & 0777) != options.mode) {
      if (fchmod(fd, options.mode) != 0)
        LOG_FATAL("Failed to chmod {}: {}", options.filename, GetErrorString());
    }
    if (options.writable && options.size != 0 && static_cast<int64_t>(options.size) != st.st_size) {
      size_ = options.size;
      if (ftruncate(fd, size_))
        LOG_FATAL("Failed to ftruncate {} (size: {}): {}", options.filename, size_,
                  GetErrorString());
      truncated = true;
    } else {
      size_ = st.st_size;
      truncated = false;
    }
    addr_ = reinterpret_cast<uint8_t *>(mmap(nullptr, size_, prot, map_flags, fd, 0));
    if (addr_ == MAP_FAILED) LOG_FATAL("Failed to mmap: {}", GetErrorString());
    close(fd);
  }
  if (options.lock && mlock(addr_, size_))
    LOG_FATAL("Failed to mlock (size: {}): {}", size_, GetErrorString());
}

void MMapFile::Reset() {
  if (addr_) {
    munmap(addr_, size_);
    addr_ = nullptr;
  }
}

}  // namespace yang
