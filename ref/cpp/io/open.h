#pragma once

#include <memory>

#include "yang/io/buffered_file.h"
#include "yang/io/buffered_istream.h"

namespace yang {
namespace io {

// Open a file on local filesystem, supports zstd and gzip compression
//
// zstd compressed files are detected via the ".zst" extension.
// gzip compressed files are detected via the ".gz" extension.
std::unique_ptr<BufferedFile> OpenBufferedFile(const std::string &filename);

// Similar to OpenBufferedFile but wrapped in a BufferedIStream
BufferedIStream OpenBufferedStream(const std::string &filename);

bool Exists(const std::string &filename);

}  // namespace io
}  // namespace yang
