#include "yang/util/simple_csv.h"

namespace yang {

unordered_map<std::string, int> SimpleCsvReader::ParseHeader(const std::string &line, char delim) {
  std::vector<std::string> header = yang::StrSplit(line, delim);
  yang::unordered_map<std::string, int> header_idx;
  for (int i = 0; i < static_cast<int>(header.size()); ++i) {
    header_idx.emplace(std::move(header[i]), i);
  }
  return header_idx;
}

SimpleCsvReader::SimpleCsvReader(const std::string &line) {
  if (line.find('|') != std::string::npos) {
    delim_ = '|';
  } else {
    delim_ = ',';
  }
  header_ = ParseHeader(line, delim_);
}

SimpleCsvReader::SimpleCsvReader(const std::string &line, char delim)
    : delim_(delim), header_(ParseHeader(line, delim)) {}

void SimpleCsvReader::ReadNext(const std::string &line) {
  row_ = yang::StrSplit(line, delim_);
}

}  // namespace yang
