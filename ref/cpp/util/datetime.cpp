#include "yang/util/datetime.h"

#include <date/date.h>
#include <date/iso_week.h>
#include <date/tz.h>

#include <cstdlib>

namespace yang {

LocalTimeZoneInfo::LocalTimeZoneInfo() {
  auto tz_env = std::getenv("TZ");
  Reset(tz_env ? tz_env : absl::LocalTimeZone().name());
}

void LocalTimeZoneInfo::Reset(const std::string &tz) {
  name = tz;
  if (!absl::LoadTimeZone(name, &absl_tz)) {
    absl_tz = absl::UTCTimeZone();
  }
}

Date Date::Next() const {
  absl::CivilDay cd(year, month, day);
  ++cd;
  return Date(cd.year(), cd.month(), cd.day());
}
Date Date::Prev() const {
  absl::CivilDay cd(year, month, day);
  --cd;
  return Date(cd.year(), cd.month(), cd.day());
}

YearWeekday YearWeekday::From(const Date &ywd) {
  auto tp = date::year{ywd.year} / ywd.month / ywd.day;
  auto iso_date = iso_week::year_weeknum_weekday{tp};
  return YearWeekday(static_cast<int>(iso_date.year()), static_cast<unsigned>(iso_date.weeknum()),
                     static_cast<unsigned>(iso_date.weekday()));
}

}  // namespace yang
