#include "neighbors.hpp"

#include <numeric>
#include <queue>

namespace mlcore {

void KDTree::build(const Matrix &X, std::size_t leaf_size) {
    dims_ = X.cols;
    nodes_.clear();
    lo_.clear();
    hi_.clear();
    index_.resize(X.rows);
    std::iota(index_.begin(), index_.end(), 0);
    if (X.rows == 0) {
        return;
    }
    leaf_size = std::max<std::size_t>(leaf_size, 1);

    nodes_.push_back(Node{0, X.rows, -1, -1});
    std::vector<std::size_t> stack{0};
    while (!stack.empty()) {
        const std::size_t id = stack.back();
        stack.pop_back();
        const std::size_t begin = nodes_[id].begin;
        const std::size_t end = nodes_[id].end;

        // Bounding box, stored at node id.
        lo_.resize(nodes_.size() * dims_, std::numeric_limits<double>::infinity());
        hi_.resize(nodes_.size() * dims_, -std::numeric_limits<double>::infinity());
        const std::size_t base = id * dims_;
        for (std::size_t i = begin; i < end; ++i) {
            const double *x = X.row(index_[i]);
            for (std::size_t d = 0; d < dims_; ++d) {
                lo_[base + d] = std::min(lo_[base + d], x[d]);
                hi_[base + d] = std::max(hi_[base + d], x[d]);
            }
        }
        if (end - begin <= leaf_size) {
            continue;
        }
        std::size_t split_dim = 0;
        double widest = -1.0;
        for (std::size_t d = 0; d < dims_; ++d) {
            if (hi_[base + d] - lo_[base + d] > widest) {
                widest = hi_[base + d] - lo_[base + d];
                split_dim = d;
            }
        }
        if (!(widest > 0.0)) {
            continue;  // all points identical
        }
        const std::size_t mid = begin + (end - begin) / 2;
        std::nth_element(index_.begin() + static_cast<std::ptrdiff_t>(begin), index_.begin() + static_cast<std::ptrdiff_t>(mid),
                         index_.begin() + static_cast<std::ptrdiff_t>(end),
                         [&](std::size_t a, std::size_t b) { return X(a, split_dim) < X(b, split_dim); });
        const std::int32_t left = static_cast<std::int32_t>(nodes_.size());
        nodes_.push_back(Node{begin, mid, -1, -1});
        nodes_.push_back(Node{mid, end, -1, -1});
        nodes_[id].left = left;
        nodes_[id].right = left + 1;
        stack.push_back(static_cast<std::size_t>(left + 1));
        stack.push_back(static_cast<std::size_t>(left));
    }
    lo_.resize(nodes_.size() * dims_);
    hi_.resize(nodes_.size() * dims_);
}

double KDTree::box_distance(const double *q, std::size_t node, const Distance &dist, std::vector<double> &gap) const {
    const double *lo = &lo_[node * dims_];
    const double *hi = &hi_[node * dims_];
    // gap holds dims_ gaps followed by dims_ zeros: the distance from the gap
    // vector to the origin is a lower bound for Minkowski-family metrics.
    for (std::size_t d = 0; d < dims_; ++d) {
        gap[d] = q[d] < lo[d] ? lo[d] - q[d] : (q[d] > hi[d] ? q[d] - hi[d] : 0.0);
    }
    return dist(gap.data(), gap.data() + dims_, dims_);
}

void KDTree::nearest(const Matrix &X, const double *q, std::size_t k, const Distance &dist, std::vector<Neighbor> &out) const {
    out.clear();
    if (nodes_.empty() || k == 0) {
        return;
    }
    std::priority_queue<Neighbor> best;  // max-heap of the k best so far
    std::vector<double> gap(2 * dims_, 0.0);
    std::vector<std::pair<double, std::size_t>> stack{{0.0, 0}};
    while (!stack.empty()) {
        const auto [bound, id] = stack.back();
        stack.pop_back();
        if (best.size() == k && bound > best.top().first) {
            continue;
        }
        const Node &node = nodes_[id];
        if (node.left < 0) {
            for (std::size_t i = node.begin; i < node.end; ++i) {
                const std::size_t row = index_[i];
                const Neighbor candidate{dist(q, X.row(row), dims_), row};
                if (best.size() < k) {
                    best.push(candidate);
                } else if (candidate < best.top()) {
                    best.pop();
                    best.push(candidate);
                }
            }
            continue;
        }
        const double dl = box_distance(q, static_cast<std::size_t>(node.left), dist, gap);
        const double dr = box_distance(q, static_cast<std::size_t>(node.right), dist, gap);
        // Push the farther child first so the nearer one is explored next.
        if (dl <= dr) {
            stack.emplace_back(dr, static_cast<std::size_t>(node.right));
            stack.emplace_back(dl, static_cast<std::size_t>(node.left));
        } else {
            stack.emplace_back(dl, static_cast<std::size_t>(node.left));
            stack.emplace_back(dr, static_cast<std::size_t>(node.right));
        }
    }
    out.resize(best.size());
    for (std::size_t i = out.size(); i-- > 0;) {
        out[i] = best.top();
        best.pop();
    }
}

void KDTree::within(const Matrix &X, const double *q, double radius, const Distance &dist, std::vector<Neighbor> &out) const {
    out.clear();
    if (nodes_.empty()) {
        return;
    }
    std::vector<double> gap(2 * dims_, 0.0);
    std::vector<std::size_t> stack{0};
    while (!stack.empty()) {
        const std::size_t id = stack.back();
        stack.pop_back();
        if (box_distance(q, id, dist, gap) > radius) {
            continue;
        }
        const Node &node = nodes_[id];
        if (node.left < 0) {
            for (std::size_t i = node.begin; i < node.end; ++i) {
                const std::size_t row = index_[i];
                const double d = dist(q, X.row(row), dims_);
                if (d <= radius) {
                    out.emplace_back(d, row);
                }
            }
            continue;
        }
        stack.push_back(static_cast<std::size_t>(node.right));
        stack.push_back(static_cast<std::size_t>(node.left));
    }
    std::sort(out.begin(), out.end());
}

// ---- NeighborIndex --------------------------------------------------------------

void NeighborIndex::build(const Matrix &X) {
    if (use_tree && !metric.coordinate_bounded()) {
        throw std::invalid_argument("The K-d tree only supports euclidean, squared_euclidean, manhattan, chebyshev and "
                                    "minkowski distances, \"" + metric.spec() + "\" given");
    }
    samples = X;
    if (use_tree) {
        tree_.build(samples, leaf_size);
    }
}

void NeighborIndex::nearest(const double *q, std::size_t k, std::vector<Neighbor> &out) const {
    if (use_tree) {
        tree_.nearest(samples, q, k, metric, out);
        return;
    }
    const std::size_t n = samples.rows;
    // Euclidean ranks the same as squared euclidean: skip the sqrt until the end.
    const bool euclidean = metric.kind() == Distance::Kind::Euclidean;
    out.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = {euclidean ? squared_distance(q, samples.row(i), samples.cols) : metric(q, samples.row(i), samples.cols), i};
    }
    k = std::min(k, n);
    std::partial_sort(out.begin(), out.begin() + static_cast<std::ptrdiff_t>(k), out.end());
    out.resize(k);
    if (euclidean) {
        for (Neighbor &nb : out) {
            nb.first = std::sqrt(nb.first);
        }
    }
}

void NeighborIndex::within(const double *q, double radius, std::vector<Neighbor> &out) const {
    if (use_tree) {
        tree_.within(samples, q, radius, metric, out);
        return;
    }
    out.clear();
    for (std::size_t i = 0; i < samples.rows; ++i) {
        const double d = metric(q, samples.row(i), samples.cols);
        if (d <= radius) {
            out.emplace_back(d, i);
        }
    }
    std::sort(out.begin(), out.end());
}

void NeighborIndex::save(ArchiveWriter &w) const {
    metric.save(w);
    w.boolean(use_tree);
    w.size(leaf_size);
    w.matrix(samples);
}

void NeighborIndex::load(ArchiveReader &r) {
    metric = Distance::load(r);
    use_tree = r.boolean();
    leaf_size = r.size();
    Matrix X = r.matrix();
    if (use_tree) {
        ArchiveReader::require(metric.coordinate_bounded(), "K-d tree with unsupported metric");
    }
    build(X);  // the tree is rebuilt rather than trusted from the payload
}

// ---- voting -----------------------------------------------------------------------

int vote(const std::vector<Neighbor> &neighbors, const std::vector<int> &labels, std::size_t classes, bool weighted) {
    std::vector<double> votes(classes, 0.0);
    int best = -1;
    double best_votes = 0.0;
    for (const Neighbor &nb : neighbors) {
        const int label = labels[nb.second];
        votes[static_cast<std::size_t>(label)] += neighbor_weight(weighted, nb.first);
        if (votes[static_cast<std::size_t>(label)] > best_votes) {
            best_votes = votes[static_cast<std::size_t>(label)];
            best = label;
        }
    }
    return best;
}

void vote_distribution(const std::vector<Neighbor> &neighbors, const std::vector<int> &labels, std::size_t classes,
                       bool weighted, double *out) {
    std::fill(out, out + classes, 0.0);
    double total = 0.0;
    for (const Neighbor &nb : neighbors) {
        const double w = neighbor_weight(weighted, nb.first);
        out[labels[nb.second]] += w;
        total += w;
    }
    if (total > 0.0) {
        for (std::size_t k = 0; k < classes; ++k) {
            out[k] /= total;
        }
    }
}

double neighbor_mean(const std::vector<Neighbor> &neighbors, const std::vector<double> &targets, bool weighted) {
    if (neighbors.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    double sum = 0.0, total = 0.0;
    for (const Neighbor &nb : neighbors) {
        const double w = neighbor_weight(weighted, nb.first);
        sum += w * targets[nb.second];
        total += w;
    }
    return sum / total;
}

}  // namespace mlcore
