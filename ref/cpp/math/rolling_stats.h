#pragma once

#include <cmath>
#include <cstdint>
#include <utility>

#include "yang/util/fixed_deque.h"

namespace yang::math {

template <class T, class Cmp>
class RollingMinMax {
 public:
  void Initialize(int window) {
    window_ = window;
    counter_ = 0;
    values_.reset(window + 1);
  }

  void Update(T v) {
    while (!values_.empty() && !Cmp{}(values_.back().value, v)) {
      values_.pop_back();
    }
    values_.push_back(Value{v, counter_});
    while (values_.front().index <= counter_ - window_) values_.pop_front();
    ++counter_;
  }

  T Get() const {
    if (values_.empty()) {
      if constexpr (std::is_floating_point<T>{}) {
        return NAN;
      } else {
        return {};
      }
    }
    return values_.front().value;
  }

 private:
#pragma pack(push, 1)
  struct Value {
    T value;
    std::int64_t index;
  };
#pragma pack(pop)
  std::int64_t counter_;
  int window_;
  fixed_deque<Value> values_;
};

template <class T>
using RollingMin = RollingMinMax<T, std::less<T>>;

template <class T>
using RollingMax = RollingMinMax<T, std::greater<T>>;

template <class T>
class RollingSum {
 public:
  RollingSum() {}

  RollingSum(int w) {
    Initialize(w);
  }

  void Initialize(int window) {
    window_ = window;
    sums_.reset(window + 1);
    sums_.push_back(0);
  }

  T Get() const {
    return sums_.back() - sums_.front();
  }

  T Get(int idx) const {
    return sums_[idx + 1] - sums_.front();
  }

  void Update(T v) {
    T acc = sums_.back() + v;
    if (static_cast<int>(sums_.size()) == window_ + 1) {
      sums_.pop_front();
    }
    sums_.push_back(acc);
  }

  int count() const {
    return sums_.size() - 1;
  }

  T last() const {
    if (sums_.size() == 1) return 0;
    return sums_[sums_.size() - 1] - sums_[sums_.size() - 2];
  }

 private:
  int window_;
  fixed_deque<T> sums_;
};

}  // namespace yang::math
