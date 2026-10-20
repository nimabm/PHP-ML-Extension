#include "linear_regression.hpp"

#include <algorithm>

namespace mlcore {

namespace {

// Solves A x = b for a symmetric positive-definite n x n matrix A (row-major)
// with a Cholesky factorization. A is overwritten by its factor and b by the
// solution. Returns false when A is singular or not positive definite.
bool cholesky_solve(std::vector<double> &a, std::vector<double> &b, std::size_t n) {
    double max_diag = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        max_diag = std::max(max_diag, std::fabs(a[i * n + i]));
    }
    if (max_diag == 0.0) {
        return false;
    }
    const double tolerance = max_diag * 1e-12;

    for (std::size_t j = 0; j < n; ++j) {
        double pivot = a[j * n + j];
        for (std::size_t k = 0; k < j; ++k) {
            pivot -= a[j * n + k] * a[j * n + k];
        }
        if (!(pivot > tolerance)) {
            return false;
        }
        const double l_jj = std::sqrt(pivot);
        a[j * n + j] = l_jj;
        for (std::size_t i = j + 1; i < n; ++i) {
            double value = a[i * n + j];
            for (std::size_t k = 0; k < j; ++k) {
                value -= a[i * n + k] * a[j * n + k];
            }
            a[i * n + j] = value / l_jj;
        }
    }

    // L z = b
    for (std::size_t i = 0; i < n; ++i) {
        double value = b[i];
        for (std::size_t k = 0; k < i; ++k) {
            value -= a[i * n + k] * b[k];
        }
        b[i] = value / a[i * n + i];
    }
    // L^T x = z
    for (std::size_t i = n; i-- > 0;) {
        double value = b[i];
        for (std::size_t k = i + 1; k < n; ++k) {
            value -= a[k * n + i] * b[k];
        }
        b[i] = value / a[i * n + i];
    }
    return true;
}

}  // namespace

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
