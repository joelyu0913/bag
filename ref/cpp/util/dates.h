#pragma once

#include <string>

namespace yang {

std::string FormatDate(const std::string &pattern, int date);
std::string FormatDateTime(const std::string &pattern, int64_t datetime);

// date1 - date2, in days
int SubDate(int date1, int date2);

}  // namespace yang
