#pragma once

#include <cstdint>

#include "clustering.hpp"
#include "common.hpp"

namespace mlcore {

// Lloyd's k-means with k-means++ initialization.
class KMeans {
public:
    std::size_t clusters = 8;
    long max_iterations = 300;
    // Training stops once the summed squared movement of all centroids in one
    // iteration drops to this value or below.
    double tolerance = 1e-4;
    bool has_seed = false;
    std::uint64_t seed = 0;
    Seeder seeder = Seeder::PlusPlus;
    Matrix preset;  // initial centroids for Seeder::Preset

    Matrix centroids;
    std::vector<int> labels;  // cluster of each training sample
    double inertia = 0.0;     // sum of squared distances to the closest centroid
    long iterations = 0;

    void fit(const Matrix &X);
    // Index of the closest centroid; optionally reports its squared distance.
    int predict_one(const double *x, double *distance = nullptr) const;
    void probabilities(const double *x, double *out) const;

    bool fitted() const { return centroids.rows > 0; }
    std::size_t features() const { return centroids.cols; }
};

}  // namespace mlcore
