#pragma once

#include "archive.hpp"
#include "common.hpp"

namespace mlcore {

// Rescales each feature to [min, max].
struct MinMaxNormalizer {
    double min = 0.0, max = 1.0;
    std::vector<double> lows, ranges;

    void fit(const Matrix &X);
    void transform(const double *in, double *out) const;
    std::size_t features() const { return lows.size(); }
    void save(ArchiveWriter &w) const;
    void load(ArchiveReader &r);
};

// Divides each feature by its maximum absolute value (keeps sparsity and sign).
struct MaxAbsoluteScaler {
    std::vector<double> scales;

    void fit(const Matrix &X);
    void transform(const double *in, double *out) const;
    std::size_t features() const { return scales.size(); }
    void save(ArchiveWriter &w) const;
    void load(ArchiveReader &r);
};

// (x - median) / MAD, robust to outliers.
struct RobustStandardizer {
    bool center = true;
    std::vector<double> medians, mads;

    void fit(const Matrix &X);
    void transform(const double *in, double *out) const;
    std::size_t features() const { return medians.size(); }
    void save(ArchiveWriter &w) const;
    void load(ArchiveReader &r);
};

// Yeo-Johnson power transform; lambdas are estimated by maximum likelihood
// unless given. Optionally standardizes the result.
struct PowerTransformer {
    std::vector<double> fixed_lambdas;
    bool normalize = true;
    std::vector<double> lambdas, means, stds;

    void fit(const Matrix &X);
    void transform(const double *in, double *out) const;
    std::size_t features() const { return lambdas.size(); }
    static double yeo_johnson(double y, double lambda);
    void save(ArchiveWriter &w) const;
    void load(ArchiveReader &r);
};

}  // namespace mlcore
