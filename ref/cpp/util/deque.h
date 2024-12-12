#pragma once

#include <cassert>
#include <cstdint>
#include <cstring>
#include <memory>
#include <type_traits>

#include "yang/util/logging.h"

namespace yang {

template <class T, class Container>
class deque_iterator {
 public:
  using this_type = deque_iterator<T, Container>;
  using size_type = std::size_t;

  deque_iterator() : container_(nullptr), idx_(0) {}
  deque_iterator(const this_type &other) = default;
  deque_iterator(Container *container, size_type idx) : container_(container), idx_(idx) {}

  this_type &operator=(const this_type &other) = default;

  T *operator->() const {
    return &dereference();
  }

  T &operator*() const {
    return dereference();
  }

  this_type &operator++() {
    if (idx_ != container_->end().idx_) {
      ++idx_;
    }
    return *this;
  }
  this_type operator++(int) {
    this_type temp(*this);
    operator++();
    return temp;
  }

  bool operator==(const this_type &other) const {
    return this->equal(other);
  }
  bool operator!=(const this_type &other) const {
    return !this->equal(other);
  }

 private:
  Container *container_;
  size_type idx_;

  bool equal(const deque_iterator &other) const {
    return container_ == other.container_ && idx_ == other.idx_;
  }

  T &dereference() const {
    return (*container_)[idx_];
  }
};

// A deque implemented as a circular buffer.
// It is mostly std::deque compatible, the most notable differences being
// `reset` and `reserve`.
template <class T, bool WarnRealloc = false, class Allocator = std::allocator<T>,
          bool Fixed = false>
class deque : private Allocator {
 public:
  using size_type = std::size_t;

  // Pads `front_idx_` and `back_idx_` to avoid underflow (going below 0).
  // This also ensure back_idx_ >= front_idx_ always holds.
  constexpr static size_type PADDING = 1LL << 62;

  using iterator = deque_iterator<T, deque>;
  using const_iterator = deque_iterator<const T, const deque>;

  deque() {}

  explicit deque(size_type cap) {
    reset(cap);
  }

  deque(const deque &other) {
    this->operator=(other);
  }

  deque &operator=(const deque &other) {
    destroy();

    cap_ = other.cap_;
    front_idx_ = other.front_idx_;
    back_idx_ = other.back_idx_;
    data_ = Allocator::allocate(cap_);
    std::copy(other.data_, other.data_ + cap_, data_);

    return *this;
  }

  deque(deque &&other) {
    this->operator=(std::move(other));
  }

  deque &operator=(deque &&other) {
    destroy();

    cap_ = other.cap_;
    front_idx_ = other.front_idx_;
    back_idx_ = other.back_idx_;
    data_ = other.data_;
    other.data_ = nullptr;

    return *this;
  }

  ~deque() {
    destroy();
  }

  void reset(size_type cap) {
    destroy();

    cap_ = 1;
    while (cap_ < cap) cap_ <<= 1;
    front_idx_ = PADDING;
    back_idx_ = front_idx_;
    data_ = Allocator::allocate(cap_);
  }

  void reserve(size_type cap) {
    deque copy(cap);
    ForEach([&](auto &i) { copy.push_back(std::move(i)); });
    this->operator=(std::move(copy));
  }

  void push_front(const T &v) {
    EnsureCapacity();
    // this is moved to the front to facilitate cpu caching & pipelining
    front_idx_--;
    if (std::is_trivially_copyable<T>::value) {
      at(front_idx_) = v;
    } else {
      new (&at(front_idx_)) T(v);
    }
  }

  void push_back(const T &v) {
    EnsureCapacity();
    if (std::is_trivially_copyable<T>::value) {
      at(back_idx_) = v;
    } else {
      new (&at(back_idx_)) T(v);
    }
    back_idx_++;
  }

  template <typename... Args>
  void emplace_front(Args &&...args) {
    EnsureCapacity();
    front_idx_--;
    new (&at(front_idx_)) T(std::forward<Args>(args)...);
  }

  template <typename... Args>
  void emplace_back(Args &&...args) {
    EnsureCapacity();
    new (&at(back_idx_)) T(std::forward<Args>(args)...);
    back_idx_++;
  }

  // caller must ensure the deque is not empty
  void pop_front() {
    pop_front(1);
  }

  // a faster way to pop n elements from the front
  void pop_front(int n) {
    assert(static_cast<int>(size()) >= n);
    if (!std::is_trivially_destructible<T>{}) {
      for (int i = 0; i < n; ++i) {
        at(front_idx_ + i).~T();
      }
    }
    front_idx_ += n;
  }

  // caller must ensure the deque is not empty
  void pop_back() {
    pop_back(1);
  }

  // a faster way to pop n elements from the back
  void pop_back(int n) {
    assert(static_cast<int>(size()) >= n);
    if (!std::is_trivially_destructible<T>{}) {
      for (int i = 1; i <= n; ++i) {
        at(back_idx_ - i).~T();
      }
    }
    back_idx_ -= n;
  }

  const T &front() const {
    return at(front_idx_);
  }
  T &front() {
    return at(front_idx_);
  }
  const T &back() const {
    return at(back_idx_ - 1);
  }
  T &back() {
    return at(back_idx_ - 1);
  }

  const T &operator[](size_type pos) const {
    return at(front_idx_ + pos);
  }

  T &operator[](size_type pos) {
    return at(front_idx_ + pos);
  }

  bool empty() const {
    return size() == 0;
  }

  size_type size() const {
    return back_idx_ - front_idx_;
  }

  size_type capacity() const {
    return cap_;
  }

  // get the underlying buffer
  const T *data_begin() const {
    return data_;
  }

  T *data_begin() {
    return data_;
  }

  const T *data_end() const {
    return data_ + cap_;
  }

  T *data_end() {
    return data_ + cap_;
  }

  // offset of the front entry in the underlying buffer
  size_type front_offset() const {
    return offset(front_idx_);
  }

  // offset of the back entry in the underlying buffer
  size_type back_offset() const {
    return offset(back_idx_);
  }

  template <class Func>
  inline __attribute__((__always_inline__)) void ForEach(Func &&f) const {
    if (empty()) return;
    const T *front_ptr = &front();
    const T *back_ptr = &back();
    if (front_ptr <= back_ptr) {
      for (auto *ptr = front_ptr; ptr <= back_ptr; ++ptr) {
        f(*ptr);
      }
    } else {
      for (auto *ptr = front_ptr; ptr < data_end(); ++ptr) {
        f(*ptr);
      }
      for (auto *ptr = data_begin(); ptr <= back_ptr; ++ptr) {
        f(*ptr);
      }
    }
  }

  void clear() {
    back_idx_ = front_idx_ = PADDING;
  }

  iterator begin() {
    return iterator(this, 0);
  }

  iterator end() {
    return iterator(this, size());
  }

  const_iterator begin() const {
    return const_iterator(this, 0);
  }

  const_iterator end() const {
    return const_iterator(this, size());
  }

 private:
  size_type cap_ = 0;
  size_type front_idx_ = 0;
  size_type back_idx_ = 0;
  T *data_ = nullptr;

  void destroy() {
    if (data_) {
      pop_front(size());
      Allocator::deallocate(data_, cap_);
    }
  }

  const T &at(size_type i) const {
    return data_[offset(i)];
  }

  T &at(size_type i) {
    return data_[offset(i)];
  }

  size_type offset(int abs_pos) const {
    // as long as we ensure `cap_` is a power of 2, this is much faster than
    // data_[i % cap_]
    //
    // NOTE: `cap_ - 1` is very fast, storing the value in a member doesn't pay off
    return abs_pos & (cap_ - 1);
  }

  void EnsureCapacity() {
    if (Fixed) {
      assert(size() < cap_);
    } else if (size() == cap_) {
      reserve(cap_ * 2);
      if (WarnRealloc) {
        LOG_WARN("deque reallocation, new cap: {}", cap_);
      }
    }
  }
};

}  // namespace yang
