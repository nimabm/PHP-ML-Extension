#pragma once

#include "archive.hpp"
#include "distance.hpp"
#include "neighbors.hpp"

namespace mlcore {

enum class Seeder : std::uint8_t { PlusPlus, Random, KMC2, Preset };

Seeder parse_seeder(const std::string &name);
std::string seeder_name(Seeder s);

// Initial centers: k-means++, uniform random samples, K-MC2 (an MCMC
// approximation of k-means++) or `preset` (must be k x d).
Matrix seed_centers(const Matrix &X, std::size_t k, Seeder seeder, const Matrix &preset, const Distance &dist, Rng &rng);

// Partitioning around medoids (alternating / Voronoi iteration).
class KMedoids {
public:
    std::size_t clusters = 8;
    long max_iterations = 300;
    Distance metric;
    Seeder seeder = Seeder::PlusPlus;
    bool has_seed = false;
    std::uint64_t seed = 0;

    Matrix medoids;
    std::vector<int> labels;
    double inertia = 0.0;  // sum of distances to the medoid
    long iterations = 0;

    void fit(const Matrix &X);
    int predict_one(const double *x, double *distance = nullptr) const;
    bool fitted() const { return medoids.rows > 0; }
    void save(ArchiveWriter &w) const;
    void load(ArchiveReader &r);
};

// Fuzzy c-means: every sample belongs to every cluster with a membership grade.
class FuzzyCMeans {
public:
    std::size_t clusters = 8;
    double fuzz = 2.0;
    long max_iterations = 300;
    double min_change = 1e-4;
    bool has_seed = false;
    std::uint64_t seed = 0;

    Matrix centroids;
    double inertia = 0.0;
    long iterations = 0;

    void fit(const Matrix &X);
    void memberships(const double *x, double *out) const;
    bool fitted() const { return centroids.rows > 0; }
    void save(ArchiveWriter &w) const;
    void load(ArchiveReader &r);
};

// Gaussian mixture with diagonal covariances, fitted by expectation-maximization.
class GaussianMixture {
public:
    std::size_t components = 8;
    double smoothing = 1e-9;
    long max_iterations = 100;
    double min_change = 1e-3;
    bool has_seed = false;
    std::uint64_t seed = 0;

    Matrix means, variances;
    std::vector<double> log_priors;
    double log_likelihood = 0.0;
    long iterations = 0;

    void fit(const Matrix &X);
    void log_joint(const double *x, double *out) const;
    void probabilities(const double *x, double *out) const;
    bool fitted() const { return means.rows > 0; }
    void save(ArchiveWriter &w) const;
    void load(ArchiveReader &r);
};

// Mean shift with a flat kernel; the number of clusters is discovered.
class MeanShift {
public:
    double radius = 0.0;  // 0 = estimate from the data
    double percentile = 30.0;
    long max_iterations = 100;
    double min_shift = 1e-4;
    std::size_t max_seeds = 1000;
    bool has_seed = false;
    std::uint64_t seed = 0;

    Matrix centroids;
    double fitted_radius = 0.0;
    long iterations = 0;

    void fit(const Matrix &X, unsigned threads);
    int predict_one(const double *x) const;
    bool fitted() const { return centroids.rows > 0; }
    static double estimate_radius(const Matrix &X, double percentile, Rng &rng);
    void save(ArchiveWriter &w) const;
    void load(ArchiveReader &r);
};

// DBSCAN: density-based clusters; -1 marks noise.
std::vector<int> dbscan(const Matrix &X, double radius, std::size_t min_density, const Distance &dist);

// OPTICS ordering and reachability; clusters are extracted at `radius` like DBSCAN.
struct OpticsResult {
    std::vector<std::size_t> ordering;
    std::vector<double> reachability;  // per sample, infinity if undefined
    std::vector<double> core_distance;
    std::vector<int> labels;
};
OpticsResult optics(const Matrix &X, std::size_t min_density, double max_radius, double radius, const Distance &dist);

// Agglomerative clustering with Ward linkage (nearest-neighbor chain).
std::vector<int> ward(const Matrix &X, std::size_t clusters);

// Spectral clustering (Ng, Jordan & Weiss): k-means on the leading eigenvectors
// of the normalized affinity matrix D^(-1/2) W D^(-1/2).
struct SpectralParams {
    std::size_t clusters = 8;
    bool nearest = false;  // affinity: false = RBF kernel exp(-gamma ||a - b||^2), true = k-nearest-neighbor graph
    double gamma = 1.0;
    std::size_t neighbors = 10;
    std::size_t runs = 10;  // k-means restarts; the lowest inertia wins
    bool has_seed = false;
    std::uint64_t seed = 0;

    void save(ArchiveWriter &w) const;
    void load(ArchiveReader &r);
};
std::vector<int> spectral_clustering(const Matrix &X, const SpectralParams &p, unsigned threads);

}  // namespace mlcore
