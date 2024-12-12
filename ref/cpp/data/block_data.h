#pragma once

#include "yang/base/ranges.h"
#include "yang/base/size.h"
#include "yang/util/logging.h"

namespace yang {

template <class Self>
class BlockDataBase {
 public:
  SizeType num_blocks() const {
    return self().blocks().size();
  }

  SizeType block_begin(int bi) const {
    return bi == 0 ? 0 : self().blocks()[bi - 1];
  }

  SizeType block_end(int bi) const {
    return self().blocks()[bi];
  }

  auto block_range(int bi) const {
    return ranges::views::iota(block_begin(bi), std::max(block_begin(bi), block_end(bi)));
  }

  SizeType block_size(int bi) const {
    auto begin = block_begin(bi);
    auto end = block_end(bi);
    return end <= begin ? 0 : end - begin;
  }

  SizeType current_block_end() const {
    return block_end(self().current_block());
  }

  void StartBlock(int block) {
    ENSURE(block < num_blocks(), "block exceeds size");
    if (self().current_block() != -1) {
      for (int bi = self().current_block() + 1; bi <= block; ++bi) {
        if (bi == 0) {
          self().blocks()[bi] = 0;
        } else {
          self().blocks()[bi] = self().blocks()[bi - 1];
        }
      }
    } else {
      // keep existing data while maintaining consistency
      for (int bi = 1; bi < block; ++bi) {
        self().blocks()[bi] = std::max(self().blocks()[bi], self().blocks()[bi - 1]);
      }
      // restart current block
      self().blocks()[block] = block == 0 ? 0 : self().blocks()[block - 1];
    }
    self().set_current_block(block);
  }

  void NextBlock() {
    StartBlock(self().current_block() + 1);
  }

 private:
  Self &self() {
    return static_cast<Self &>(*this);
  }

  const Self &self() const {
    return static_cast<const Self &>(*this);
  }
};

template <class Self>
class BlockData : public BlockDataBase<Self> {
 public:
  int current_block() const {
    return current_block_;
  }

  void set_current_block(int v) {
    current_block_ = v;
  }

 protected:
  int current_block_ = -1;
};

}  // namespace yang
