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

// Independent component analysis (symmetric FastICA on whitened data).
struct FastICA {
    enum class Contrast : std::uint8_t { LogCosh, Exp, Cube };

    std::size_t dimensions = 0;  // 0 = as many components as features
    Contrast contrast = Contrast::LogCosh;
    long max_iterations = 200;
    double tolerance = 1e-4;
    bool has_seed = false;
    std::uint64_t seed = 0;

    LinearProjection projection;  // mean and unmixing matrix (d x k)
    Matrix mixing;                // d x k: x ~ mean + mixing s
    long iterations = 0;

    void fit(const Matrix &X, unsigned threads);
    static Contrast parse(const std::string &name);
    static std::string name(Contrast c);
    void save(ArchiveWriter &w) const;
    void load(ArchiveReader &r);
};

// Non-negative matrix factorization X ~ W H (Frobenius loss, HALS updates).
struct NMF {
    std::size_t dimensions = 2;
    long max_iterations = 200;
    double tolerance = 1e-4;
    bool has_seed = false;
    std::uint64_t seed = 0;

    Matrix components;   // H: k x d
    Matrix gram;         // H H^T, cached for transform
    double error = 0.0;  // Frobenius norm of X - W H after fit
    long iterations = 0;

    // Fits H and returns the training coefficients W (n x k).
    Matrix fit(const Matrix &X, unsigned threads);
    // Non-negative coefficients of one sample with H fixed.
    void transform(const double *x, double *out) const;
    bool fitted() const { return components.rows > 0; }
    std::size_t features() const { return components.cols; }
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
