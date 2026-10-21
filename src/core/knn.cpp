#include "knn.hpp"

#include <algorithm>
#include <utility>

namespace mlcore {

void KNearestNeighbors::fit(const Matrix &X, const std::vector<int> &y, std::size_t n_classes) {
    require_training_set(X, y.size());
    if (k > X.rows) {
        throw std::invalid_argument("k (" + std::to_string(k) + ") cannot exceed the number of training samples (" +
                                    std::to_string(X.rows) + ")");
    }
    samples = X;
    labels = y;
    classes = n_classes;
}

int KNearestNeighbors::predict_one(const double *x) const {
    const std::size_t n = samples.rows;
    const std::size_t d = samples.cols;

    // Squared euclidean distance ranks neighbours the same as the real one.
    std::vector<std::pair<double, std::size_t>> distances(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double dist = metric == DistanceMetric::Euclidean ? squared_distance(x, samples.row(i), d)
                                                                : manhattan_distance(x, samples.row(i), d);
        distances[i] = {dist, i};
    }
    std::partial_sort(distances.begin(), distances.begin() + static_cast<std::ptrdiff_t>(k), distances.end());

    // Walking neighbours nearest-first, a class only takes the lead by
    // strictly exceeding the current count, which breaks ties by proximity.
    std::vector<std::size_t> votes(classes, 0);
    int best = -1;
    std::size_t best_votes = 0;
    for (std::size_t i = 0; i < k; ++i) {
        const int label = labels[distances[i].second];
        if (++votes[static_cast<std::size_t>(label)] > best_votes) {
            best_votes = votes[static_cast<std::size_t>(label)];
            best = label;
        }
    }
    return best;
}

}  // namespace mlcore
