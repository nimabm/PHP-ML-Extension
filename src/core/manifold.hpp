#pragma once

// Non-linear dimensionality reduction: kernel PCA, Isomap and UMAP. Each one
// keeps its training samples so that new samples can be embedded later.

#include "archive.hpp"
#include "neighbors.hpp"
#include "svm.hpp"

namespace mlcore {

// Kernel principal component analysis: PCA in the feature space of a kernel.
class KernelPCA {
public:
    std::size_t dimensions = 2;
    Kernel kernel;  // gamma 0 = 1 / features, resolved into `active` at fit time

    Kernel active;
    Matrix samples;
    Matrix alphas;                     // n x k: eigenvectors scaled by 1 / sqrt(eigenvalue)
    std::vector<double> column_means;  // of the training kernel matrix
    double grand_mean = 0.0;
    std::vector<double> eigenvalues;   // of the centered kernel matrix, descending

    void fit(const Matrix &X, unsigned threads);
    void transform(const double *x, double *out) const;
    bool fitted() const { return samples.rows > 0; }
    std::size_t features() const { return samples.cols; }
    void save(ArchiveWriter &w) const;
    void load(ArchiveReader &r);
};

// Isomap: classical MDS on geodesic distances of the k-nearest-neighbor graph.
class Isomap {
public:
    std::size_t dimensions = 2;
    std::size_t neighbors = 5;
    Distance metric;

    NeighborIndex index;               // training samples
    Matrix geodesic;                   // n x n shortest-path distances
    Matrix vectors;                    // n x k eigenvectors of the double-centered squared geodesics
    std::vector<double> eigenvalues;
    std::vector<double> squared_means; // row means of the squared geodesic matrix

    void fit(const Matrix &X, unsigned threads);
    void transform(const double *x, double *out) const;
    bool fitted() const { return index.samples.rows > 0; }
    std::size_t features() const { return index.samples.cols; }
    void save(ArchiveWriter &w) const;
    void load(ArchiveReader &r);
};

constexpr std::size_t kMaxEpochs = 100000;

struct UMAPParams {
    std::size_t dimensions = 2;
    std::size_t neighbors = 15;
    double min_distance = 0.1;
    double spread = 1.0;
    std::size_t epochs = 0;  // 0 = 500 up to 10,000 samples, else 200
    double rate = 1.0;
    std::size_t negative_samples = 5;
    double repulsion = 1.0;
    bool random_init = false;  // false = PCA initialization
    Distance metric;
    bool has_seed = false;
    std::uint64_t seed = 0;

    void save(ArchiveWriter &w) const;
    void load(ArchiveReader &r);
};

// Uniform manifold approximation and projection (McInnes, Healy & Melville).
class UMAP {
public:
    UMAPParams params;

    NeighborIndex index;
    Matrix embedding;  // of the training samples
    double a = 0.0, b = 0.0;

    void fit(const Matrix &X, unsigned threads);
    // Embeds new samples against the fixed training embedding.
    Matrix transform(const Matrix &Q, unsigned threads) const;
    bool fitted() const { return embedding.rows > 0; }
    std::size_t features() const { return index.samples.cols; }
    void save(ArchiveWriter &w) const;
    void load(ArchiveReader &r);

    // Curve parameters of 1 / (1 + a d^(2b)) for a spread and minimum distance.
    static void fit_curve(double spread, double min_distance, double &a, double &b);
};

}  // namespace mlcore
