#pragma once

#include "common.hpp"

namespace mlcore {

// Ordinary least squares with an optional L2 (ridge) penalty, solved in closed
// form through the normal equations on centered data.
class LinearRegression {
public:
    double l2 = 0.0;
    std::vector<double> coefficients;
    double intercept = 0.0;

    void fit(const Matrix &X, const std::vector<double> &y);
    double predict_one(const double *x) const;

    bool fitted() const { return !coefficients.empty(); }
    std::size_t features() const { return coefficients.size(); }
};

}  // namespace mlcore
