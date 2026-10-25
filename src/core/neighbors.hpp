#pragma once

#include <utility>

#include "archive.hpp"
#include "distance.hpp"

namespace mlcore {

using Neighbor = std::pair<double, std::size_t>;  // (distance, sample index)

// K-d tree over the rows of a matrix (which the caller owns and must keep
// alive and unchanged). Splits on the widest dimension at the median and keeps
// a bounding box per node for pruning. Only valid for coordinate-bounded
// (Minkowski-family) distances.
class KDTree {
public:
    void build(const Matrix &X, std::size_t leaf_size);
    void nearest(const Matrix &X, const double *q, std::size_t k, const Distance &dist, std::vector<Neighbor> &out) const;
    void within(const Matrix &X, const double *q, double radius, const Distance &dist, std::vector<Neighbor> &out) const;
    bool built() const { return !nodes_.empty(); }

private:
    struct Node {
        std::size_t begin = 0, end = 0;
        std::int32_t left = -1, right = -1;
    };
    double box_distance(const double *q, std::size_t node, const Distance &dist, std::vector<double> &gap) const;

    std::size_t dims_ = 0;
    std::vector<Node> nodes_;
    std::vector<double> lo_, hi_;  // bounding boxes, nodes x dims
    std::vector<std::size_t> index_;
};

// Stores training samples and answers k-nearest and radius queries, either by
// brute force or with a K-d tree.
class NeighborIndex {
public:
    Distance metric;
    bool use_tree = false;
    std::size_t leaf_size = 30;
    Matrix samples;

    void build(const Matrix &X);
    // k nearest, sorted by distance (ties by index).
    void nearest(const double *q, std::size_t k, std::vector<Neighbor> &out) const;
    // All samples within `radius` (inclusive), sorted by distance.
    void within(const double *q, double radius, std::vector<Neighbor> &out) const;

    void save(ArchiveWriter &w) const;
    void load(ArchiveReader &r);

private:
    KDTree tree_;
};

// An index over X with `metric` that uses a K-d tree when it can help
// (low-dimensional data with a Minkowski-family distance).
NeighborIndex make_index(const Matrix &X, const Distance &metric);

// The k nearest neighbors of every indexed sample, itself excluded, nearest first
// (k is capped at samples - 1).
std::vector<std::vector<Neighbor>> neighbor_graph(const NeighborIndex &index, std::size_t k, unsigned threads);

// Weight of a neighbor at distance d: 1 when unweighted, else inverse distance.
inline double neighbor_weight(bool weighted, double d) { return weighted ? 1.0 / (d + 1e-9) : 1.0; }

// Majority (or distance-weighted) vote over neighbors sorted nearest-first.
// Ties go to the class whose votes accumulated first, i.e. the nearer one.
int vote(const std::vector<Neighbor> &neighbors, const std::vector<int> &labels, std::size_t classes, bool weighted);
void vote_distribution(const std::vector<Neighbor> &neighbors, const std::vector<int> &labels, std::size_t classes,
                       bool weighted, double *out);
double neighbor_mean(const std::vector<Neighbor> &neighbors, const std::vector<double> &targets, bool weighted);

}  // namespace mlcore
