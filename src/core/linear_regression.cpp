#include "linear_regression.hpp"

#include <algorithm>

#include "linalg.hpp"

namespace mlcore {

void LinearRegression::fit(const Matrix &X, const std::vector<double> &y) {
    require_training_set(X, y.size());
    const std::size_t n = X.rows;
    const std::size_t d = X.cols;

    std::vector<double> mean(d, 0.0);
    double y_mean = 0.0;
    for (std::size_t r = 0; r < n; ++r) {
        const double *x = X.row(r);
        for (std::size_t c = 0; c < d; ++c) {
            mean[c] += x[c];
        }
        y_mean += y[r];
    }
    for (double &m : mean) {
        m /= static_cast<double>(n);
    }
    y_mean /= static_cast<double>(n);

    // Centering lets the intercept drop out of the system and keeps it
    // unpenalized when l2 > 0.
    std::vector<double> gram(d * d, 0.0);
    std::vector<double> rhs(d, 0.0);
    std::vector<double> centered(d);
    for (std::size_t r = 0; r < n; ++r) {
        const double *x = X.row(r);
        for (std::size_t c = 0; c < d; ++c) {
            centered[c] = x[c] - mean[c];
        }
        const double target = y[r] - y_mean;
        for (std::size_t i = 0; i < d; ++i) {
            const double xi = centered[i];
            rhs[i] += xi * target;
            double *gram_row = &gram[i * d];
            for (std::size_t j = 0; j <= i; ++j) {
                gram_row[j] += xi * centered[j];
            }
        }
    }
    for (std::size_t i = 0; i < d; ++i) {
        for (std::size_t j = 0; j < i; ++j) {
            gram[j * d + i] = gram[i * d + j];
        }
        gram[i * d + i] += l2;
    }

    if (!cholesky_solve(gram, rhs, d)) {
        throw std::runtime_error(
            "Cannot fit LinearRegression: the feature matrix is singular "
            "(constant or collinear features, or fewer samples than features); set l2 > 0 to regularize");
    }

    intercept = y_mean - dot(mean.data(), rhs.data(), d);
    coefficients = std::move(rhs);
}

double LinearRegression::predict_one(const double *x) const {
    return intercept + dot(coefficients.data(), x, coefficients.size());
}

}  // namespace mlcore
