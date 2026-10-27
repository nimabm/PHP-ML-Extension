#pragma once

#include <cstdint>

#include "archive.hpp"
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

    void fit(const Matrix &X, unsigned threads = 1);
    // Index of the closest centroid; optionally reports its squared distance.
    int predict_one(const double *x, double *distance = nullptr) const;
    void probabilities(const double *x, double *out) const;

    bool fitted() const { return centroids.rows > 0; }
    std::size_t features() const { return centroids.cols; }
};

// Index of the closest row of `centroids`; optionally reports its squared distance.
int nearest_centroid(const Matrix &centroids, const double *x, double *distance = nullptr);
// Memberships proportional to inverse distance (an exact hit takes all).
void centroid_memberships(const Matrix &centroids, const double *x, double *out);

// Mini-batch k-means (Sculley, 2010): centroids follow running means of the
// samples assigned to them, one batch at a time. Supports incremental training.
class MiniBatchKMeans {
public:
    std::size_t clusters = 8;
    std::size_t batch_size = 256;
    long max_iterations = 100;  // epochs over the training set
    double tolerance = 1e-4;    // stop once an epoch moves the centroids less than this (summed squares)
    bool has_seed = false;
    std::uint64_t seed = 0;
    Seeder seeder = Seeder::PlusPlus;
    Matrix preset;

    Matrix centroids;
    std::vector<double> counts;  // samples absorbed by each centroid
    std::vector<int> labels;     // of the last fit() / partial_fit() data
    double inertia = 0.0;
    long iterations = 0;         // epochs of fit(), plus one per partial_fit()

    void fit(const Matrix &X, unsigned threads);
    void partial_fit(const Matrix &X, unsigned threads);
    int predict_one(const double *x) const { return nearest_centroid(centroids, x); }
    void probabilities(const double *x, double *out) const { centroid_memberships(centroids, x, out); }
    bool fitted() const { return centroids.rows > 0; }
    std::size_t features() const { return centroids.cols; }
    void save(ArchiveWriter &w) const;
    void load(ArchiveReader &r);

private:
    void initialize(const Matrix &X, Rng &rng);
    double update(const Matrix &X, const std::vector<std::size_t> &batch, unsigned threads);
    void assign(const Matrix &X, unsigned threads);
};

}  // namespace mlcore
