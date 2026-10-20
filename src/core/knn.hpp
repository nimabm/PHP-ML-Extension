#pragma once

#include "common.hpp"

namespace mlcore {

enum class DistanceMetric { Euclidean, Manhattan };

// k-nearest-neighbours classifier with majority voting. Ties go to the class
// whose members are closest to the query.
class KNearestNeighbors {
public:
    std::size_t k = 3;
    DistanceMetric metric = DistanceMetric::Euclidean;

    Matrix samples;
    std::vector<int> labels;  // class index per training sample
    std::size_t classes = 0;

    void fit(const Matrix &X, const std::vector<int> &y, std::size_t n_classes);
    int predict_one(const double *x) const;

    bool fitted() const { return samples.rows > 0; }
    std::size_t features() const { return samples.cols; }
};

}  // namespace mlcore
