#pragma once

#include "common.hpp"

namespace mlcore {

// Standardizes features to zero mean and unit variance. Constant features
// keep a scale of 1 so they are only centered.
class StandardScaler {
public:
    std::vector<double> means;
    std::vector<double> scales;

    void fit(const Matrix &X);
    void transform(double *x) const;

    bool fitted() const { return !means.empty(); }
    std::size_t features() const { return means.size(); }
};

}  // namespace mlcore
