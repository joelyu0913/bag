#pragma once

#include "yang/data/block_struct_vector.h"
#include "yang/util/unordered_map.h"

namespace yang {

// Data structures for directed graphs

constexpr std::array<std::string_view, 4> GRAPH_EDGE_FIELDS = {"from", "to", "weight", "rank"};
using GraphEdge = std::tuple<int16_t, int16_t, float, int16_t>;

class GraphEdgeVector : public BlockDataBase<GraphEdgeVector> {
 public:
  GraphEdgeVector() {}

  GraphEdgeVector(GraphEdgeVector &&other) {
    this->operator=(std::move(other));
  }

  GraphEdgeVector &operator=(GraphEdgeVector &&other) {
    data_ = std::move(other.data_);
    return *this;
  }

  const std::string &path() const {
    return data_.path();
  }

  SizeType size() const {
    return data_.size();
  }

  GraphEdge operator[](SizeType i) const {
    return data_[i];
  }

  auto blocks() {
    return data_.blocks();
  }

  auto blocks() const {
    return data_.blocks();
  }

  void PushBack(int16_t from, int16_t to, float weight, int16_t rank = null_v<int16_t>) {
    data_.PushBack(from, to, weight, rank);
  }

  void Extend() {
    data_.Extend();
  }

  void Extend(SizeType extend_by) {
    data_.Extend(extend_by);
  }

  void ResizeBlocks(SizeType size) {
    data_.ResizeBlocks(size);
  }

  [[nodiscard]] static GraphEdgeVector Load(const std::string &path) {
    return MMap(path);
  }

  static GraphEdgeVector MMap(const std::string &path);

  static GraphEdgeVector MMap(const std::string &path, SizeType num_blocks,
                              SizeType init_size = 5000000, SizeType extend_size = 500000);

  int current_block() const {
    return data_.current_block();
  }

  void set_current_block(int v) {
    data_.set_current_block(v);
  }

 private:
  BlockStructVector<GraphEdge> data_;
};

class SparseGraph {
 public:
  SparseGraph() {}

  SparseGraph(SparseGraph &&other) {
    this->operator=(std::move(other));
  }

  SparseGraph &operator=(SparseGraph &&other) {
    out_edges_ = std::move(other.out_edges_);
    in_edges_ = std::move(other.in_edges_);
    weight_map_ = std::move(other.weight_map_);
    return *this;
  }

  bool Add(int16_t from, int16_t to, float w, bool update = true) {
    auto it = weight_map_.find({from, to});
    if (it != weight_map_.end()) {
      if (update) {
        it->second = w;
        return true;
      } else {
        return false;
      }
    }
    weight_map_.emplace_hint(it, std::make_pair(from, to), w);
    out_edges_[from].push_back(to);
    in_edges_[to].push_back(from);
    return true;
  }

  void Update(int16_t from, int16_t to, float w) {
    Add(from, to, w, true);
  }

  const std::vector<int16_t> &out_edges(int16_t from) const {
    static const std::vector<int16_t> empty;
    auto it = out_edges_.find(from);
    return it != out_edges_.end() ? it->second : empty;
  }

  const std::vector<int16_t> &in_edges(int16_t to) const {
    static const std::vector<int16_t> empty;
    auto it = in_edges_.find(to);
    return it != in_edges_.end() ? it->second : empty;
  }

  float weight(int16_t from, int16_t to) const {
    auto it = weight_map_.find({from, to});
    return it != weight_map_.end() ? it->second : NAN;
  }

  void Clear() {
    out_edges_.clear();
    in_edges_.clear();
    weight_map_.clear();
  }

  void Load(const GraphEdgeVector &vec, int bi, int16_t max_rank = -1, bool update = true);

 private:
  unordered_map<int16_t, std::vector<int16_t>> out_edges_;
  unordered_map<int16_t, std::vector<int16_t>> in_edges_;
  unordered_map<std::pair<int16_t, int16_t>, float> weight_map_;
};

}  // namespace yang
