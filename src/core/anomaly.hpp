#pragma once

#include "archive.hpp"
#include "distance.hpp"
#include "neighbors.hpp"

namespace mlcore {

// All detectors score samples so that higher means more anomalous, and flag
// a sample as an anomaly when its score exceeds `threshold`. Detectors with a
// `contamination` parameter set the threshold to the (1 - contamination)
// quantile of the training scores.

// Combined robust z-score: beta * max_j z_j + (1 - beta) * mean_j z_j, with
// z_j = 0.6745 |x_j - median_j| / MAD_j.
struct RobustZScore {
    double threshold = 3.5, beta = 0.5, smoothing = 1e-9;
    std::vector<double> medians, mads;

    void fit(const Matrix &X);
    double score(const double *x) const;
    std::size_t features() const { return medians.size(); }
    void save(ArchiveWriter &w) const;
    void load(ArchiveReader &r);
};

// Negative log-likelihood under a per-feature Gaussian.
struct GaussianMLE {
    double contamination = 0.1, smoothing = 1e-9;
    std::vector<double> means, variances;
    double threshold = 0.0;

    void fit(const Matrix &X);
    double score(const double *x) const;
    std::size_t features() const { return means.size(); }
    void save(ArchiveWriter &w) const;
    void load(ArchiveReader &r);
};

// Isolation Forest (Liu et al.): anomalies are isolated by fewer random splits.
// Scores are in (0, 1); without contamination the threshold is 0.5.
struct IsolationForest {
    struct Node {
        std::int32_t feature = -1;
        double split = 0.0;
        std::int32_t left = -1, right = -1;
        double size = 0.0;  // samples that reached a leaf
    };
    std::size_t estimators = 100;
    std::size_t max_samples = 256;
    double contamination = -1.0;  // < 0: threshold 0.5
    bool has_seed = false;
    std::uint64_t seed = 0;

    std::size_t features_ = 0, sample_size = 0;
    std::vector<std::vector<Node>> trees;
    double threshold = 0.5;

    void fit(const Matrix &X, unsigned threads);
    double score(const double *x) const;
    std::size_t features() const { return features_; }
    void save(ArchiveWriter &w) const;
    void load(ArchiveReader &r);
};

// Local Outlier Factor relative to the training set.
struct LocalOutlierFactor {
    std::size_t k = 20;
    double contamination = 0.1;
    NeighborIndex index;
    std::vector<double> k_distance, lrd;
    double threshold = 1.0;

    void fit(const Matrix &X);
    double score(const double *x) const;
    std::size_t features() const { return index.samples.cols; }
    void save(ArchiveWriter &w) const;
    void load(ArchiveReader &r);

private:
    double factor(const std::vector<Neighbor> &nb) const;
};

// Lightweight on-line detector of anomalies: random sparse projections with
// histogram density estimates.
struct Loda {
    std::size_t estimators = 100;
    std::size_t bins = 0;  // 0 = automatic
    double contamination = 0.1;
    bool has_seed = false;
    std::uint64_t seed = 0;

    std::size_t features_ = 0;
    Matrix projections;               // estimators x features
    std::vector<double> lows, widths;  // per estimator
    std::vector<double> densities;    // estimators x bins (log densities)
    std::size_t bin_count = 0;
    double samples_seen = 0.0;
    double threshold = 0.0;

    void fit(const Matrix &X);
    double score(const double *x) const;
    std::size_t features() const { return features_; }
    void save(ArchiveWriter &w) const;
    void load(ArchiveReader &r);
};

}  // namespace mlcore
