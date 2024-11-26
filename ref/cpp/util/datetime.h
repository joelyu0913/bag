#pragma once

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <utility>

#include "absl/time/time.h"
#include "yang/util/logging.h"

namespace yang {

template <class Precision = std::chrono::nanoseconds>
auto GetTimestamp() {
  auto now = std::chrono::system_clock::now();
  return std::chrono::duration_cast<Precision>(now.time_since_epoch()).count();
}

struct LocalTimeZoneInfo {
  std::string name;
  absl::TimeZone absl_tz;

  LocalTimeZoneInfo();
  LocalTimeZoneInfo(const LocalTimeZoneInfo &) = delete;
  LocalTimeZoneInfo &operator=(const LocalTimeZoneInfo &) = delete;

  static LocalTimeZoneInfo &instance() {
    static LocalTimeZoneInfo tz;
    return tz;
  }

  void Reset(const std::string &tz);
};

inline const LocalTimeZoneInfo &GetLocalTimeZone() {
  return LocalTimeZoneInfo::instance();
}

struct TimeOfDay {
  int8_t hour = 0;
  int8_t minute = 0;
  int8_t second = 0;
  int ns = 0;

  TimeOfDay() {}

  TimeOfDay(int8_t h, int8_t m, int8_t s, int ns = 0) : hour(h), minute(m), second(s), ns(ns) {}

  TimeOfDay(int hhmmss) {
    ENSURE2(hhmmss < 1000000);
    hour = hhmmss / 10000;
    minute = (hhmmss / 100) % 100;
    second = hhmmss % 100;
    ns = 0;
  }

  static TimeOfDay FromHHMM(int hhmm) {
    ENSURE2(hhmm < 10000);
    return TimeOfDay(hhmm / 100, hhmm % 100, 0);
  }

  auto tuple() const {
    return std::make_tuple(hour, minute, second, ns);
  }

  bool operator==(const TimeOfDay &other) const = default;
  bool operator!=(const TimeOfDay &other) const = default;
  bool operator>(const TimeOfDay &other) const {
    return tuple() > other.tuple();
  }
  bool operator>=(const TimeOfDay &other) const {
    return tuple() >= other.tuple();
  }
  bool operator<(const TimeOfDay &other) const {
    return tuple() < other.tuple();
  }
  bool operator<=(const TimeOfDay &other) const {
    return tuple() <= other.tuple();
  }
};

struct Date {
  int16_t year = 0;
  int8_t month = 0;
  int8_t day = 0;

  Date() {}
  Date(int16_t y, int8_t m, int8_t d) : year(y), month(m), day(d) {}

  Date(int yyyymmdd) {
    ENSURE2(yyyymmdd < 100000000);
    year = yyyymmdd / 10000;
    month = (yyyymmdd / 100) % 100;
    day = yyyymmdd % 100;
  }

  static Date From(int yyyymmdd) {
    return Date(yyyymmdd);
  }

  int ToYYYYMMDD() const {
    return year * 10000 + month * 100 + day;
  }

  auto tuple() const {
    return std::make_tuple(year, month, day);
  }

  bool operator==(const Date &other) const = default;
  bool operator!=(const Date &other) const = default;
  bool operator>(const Date &other) const {
    return tuple() > other.tuple();
  }
  bool operator>=(const Date &other) const {
    return tuple() >= other.tuple();
  }
  bool operator<(const Date &other) const {
    return tuple() < other.tuple();
  }
  bool operator<=(const Date &other) const {
    return tuple() <= other.tuple();
  }

  Date Next() const;
  Date Prev() const;

  Date &operator++() {
    *this = Next();
    return *this;
  }

  Date operator++(int) {
    return Next();
  }

  Date &operator--() {
    *this = Prev();
    return *this;
  }

  Date operator--(int) {
    return Prev();
  }
};

struct DateTime {
  Date date;
  TimeOfDay time;

  DateTime(Date d, TimeOfDay t = TimeOfDay()) : date(d), time(t) {}

  static DateTime FromTimestamp(int64_t ts) {
    auto t = absl::FromUnixNanos(ts);
    auto cs = absl::ToCivilSecond(t, GetLocalTimeZone().absl_tz);
    auto ns = absl::ToUnixNanos(t) % 1000000000ll;
    return DateTime(Date(cs.year(), cs.month(), cs.day()),
                    TimeOfDay(cs.hour(), cs.minute(), cs.second(), ns));
  }

  static DateTime Now() {
    return FromTimestamp(GetTimestamp());
  }

  auto tuple() const {
    return std::make_tuple(date, time);
  }

  bool operator==(const DateTime &other) const = default;
  bool operator!=(const DateTime &other) const = default;
  bool operator>(const DateTime &other) const {
    return tuple() > other.tuple();
  }
  bool operator>=(const DateTime &other) const {
    return tuple() >= other.tuple();
  }
  bool operator<(const DateTime &other) const {
    return tuple() < other.tuple();
  }
  bool operator<=(const DateTime &other) const {
    return tuple() <= other.tuple();
  }
};

inline TimeOfDay GetTimeOfDay() {
  return DateTime::Now().time;
}

inline Date GetToday() {
  return DateTime::Now().date;
}

// current time in HHMM format
inline int GetCurrentTime() {
  auto time = GetTimeOfDay();
  return time.hour * 100 + time.minute;
}

struct YearWeekday {
  int year;
  int weeknum;  // 1-53
  int weekday;  // 1-7, Mon-Sun

  YearWeekday() {}
  YearWeekday(int y, int wn, int wd) : year(y), weeknum(wn), weekday(wd) {}

  static YearWeekday From(const Date &ywd);
};

inline int64_t CombineDateTime(int64_t date, int64_t time) {
  if (time < 10000) {
    // hhmm
    return date * 10000 + time;
  } else {
    // hhmmsss
    return date * 1000000 + time;
  }
}

inline std::pair<int64_t, int64_t> SplitDateTime(int64_t dt) {
  if (dt < 100000000) {
    // date only
    return {dt, 0};
  } else if (dt < 1'0000'00'00'0000LL) {
    // date + hhmm
    auto ret = std::lldiv(dt, 10000);
    return {ret.quot, ret.rem};
  } else {
    // date + hhmmss
    auto ret = std::lldiv(dt, 1000000);
    return {ret.quot, ret.rem};
  }
}

}  // namespace yang
