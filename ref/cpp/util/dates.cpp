#include "yang/util/dates.h"

#include <date/date.h>

#include <ctime>

#include "yang/util/datetime.h"

namespace yang {

static std::tuple<int, int, int> SplitDate(int date) {
  return {date / 10000, (date % 10000) / 100, date % 100};
}

std::string FormatDate(const std::string &pattern, int date) {
  char out[1024];
  struct tm t;
  t.tm_year = date / 10000 - 1900;
  t.tm_mon = (date % 10000) / 100 - 1;
  t.tm_mday = date % 100;
  std::strftime(out, sizeof(out), pattern.c_str(), &t);
  return out;
}

std::string FormatDateTime(const std::string &pattern, int64_t datetime) {
  char out[1024];
  struct tm t;
  auto [date, time] = SplitDateTime(datetime);
  t.tm_year = date / 10000 - 1900;
  t.tm_mon = (date % 10000) / 100 - 1;
  t.tm_mday = date % 100;
  t.tm_hour = time / 100;
  t.tm_min = time % 100;
  std::strftime(out, sizeof(out), pattern.c_str(), &t);
  return out;
}

int SubDate(int date1, int date2) {
  auto [y1, m1, d1] = SplitDate(date1);
  auto [y2, m2, d2] = SplitDate(date2);
  auto diff = date::sys_days(date::year{y1} / m1 / d1) - date::sys_days(date::year{y2} / m2 / d2);
  return diff.count();
}

}  // namespace yang
