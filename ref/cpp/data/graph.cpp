#include "yang/data/graph.h"

#include <fstream>

#include "yang/util/config.h"
#include "yang/util/fs.h"

namespace yang {

void SparseGraph::Load(const GraphEdgeVector &vec, int bi, int16_t max_rank, bool update) {
  for (auto i : vec.block_range(bi)) {
    auto edge = vec[i];
    auto [from, to, weight, rank] = edge;
    if (max_rank < 0 || rank <= max_rank) Add(from, to, weight, update);
  }
}

GraphEdgeVector GraphEdgeVector::MMap(const std::string &path) {
  GraphEdgeVector vec;
  vec.data_ = BlockStructVector<GraphEdge>::MMap(path);
  return vec;
}

GraphEdgeVector GraphEdgeVector::MMap(const std::string &path, SizeType num_blocks,
                                      SizeType init_size, SizeType extend_size) {
  GraphEdgeVector vec;
  vec.data_ = BlockStructVector<GraphEdge>::MMap(path, GRAPH_EDGE_FIELDS, num_blocks, init_size,
                                                 extend_size);
  return vec;
}

}  // namespace yang
