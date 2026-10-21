#pragma once

#include "archive.hpp"
#include "common.hpp"

namespace mlcore {

// y = (x - mean) W, with W a d x k matrix (mean empty = no centering).
struct LinearProjection {
    std::vector<double> mean;
    Matrix W;

    void transform(const double *x, double *out) const;
    std::size_t input_dim() const { return W.rows; }
    std::size_t output_dim() const { return W.cols; }
    void save(ArchiveWriter &w) const;
    void load(ArchiveReader &r);
};

// Principal component analysis (eigen-decomposition of the covariance matrix).
struct PCA {
    std::size_t dimensions = 2;
    LinearProjection projection;
    std::vector<double> explained;  // variance of each component
    double total_variance = 0.0;

    void fit(const Matrix &X);
    void save(ArchiveWriter &w) const;
    void load(ArchiveReader &r);
};

// Truncated SVD (no centering, suited to sparse/count data such as TF-IDF).
struct TruncatedSVD {
    std::size_t dimensions = 2;
    LinearProjection projection;
    std::vector<double> singular_values;
    double total = 0.0;

    void fit(const Matrix &X);
    void save(ArchiveWriter &w) const;
    void load(ArchiveReader &r);
};

// Fisher linear discriminant analysis (supervised).
struct LDA {
    std::size_t dimensions = 1;
    LinearProjection projection;
    std::vector<double> explained;

    void fit(const Matrix &X, const std::vector<int> &y, std::size_t classes);
    void save(ArchiveWriter &w) const;
    void load(ArchiveReader &r);
};

// Gaussian or sparse (Achlioptas) random projection.
struct RandomProjector {
    std::size_t dimensions = 2;
    bool sparse = false;
    double sparsity = 3.0;
    bool has_seed = false;
    std::uint64_t seed = 0;
    LinearProjection projection;

    void fit(const Matrix &X);
    // Johnson-Lindenstrauss bound: dimensions needed to keep distances within (1 +- distortion).
    static std::size_t min_dimensions(std::size_t samples, double distortion);
    void save(ArchiveWriter &w) const;
    void load(ArchiveReader &r);
};

struct TSNEParams {
    std::size_t dimensions = 2;
    double rate = 100.0;
    double perplexity = 30.0;
    double exaggeration = 12.0;
    std::size_t epochs = 1000;
    double min_gradient = 1e-7;
    bool has_seed = false;
    std::uint64_t seed = 0;

    void save(ArchiveWriter &w) const;
    void load(ArchiveReader &r);
};

// Exact t-SNE embedding (O(n^2) per iteration).
Matrix tsne(const Matrix &X, const TSNEParams &p, unsigned threads, std::vector<double> *losses = nullptr);

}  // namespace mlcore
