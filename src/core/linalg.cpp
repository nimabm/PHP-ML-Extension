#include "linalg.hpp"

#include <numeric>

namespace mlcore {

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
    for (std::size_t i = 0; i < n; ++i) {
        double value = b[i];
        for (std::size_t k = 0; k < i; ++k) {
            value -= a[i * n + k] * b[k];
        }
        b[i] = value / a[i * n + i];
    }
    for (std::size_t i = n; i-- > 0;) {
        double value = b[i];
        for (std::size_t k = i + 1; k < n; ++k) {
            value -= a[k * n + i] * b[k];
        }
        b[i] = value / a[i * n + i];
    }
    return true;
}

void symmetric_eigen(std::vector<double> a, std::size_t n, std::vector<double> &values, std::vector<double> &vectors) {
    std::vector<double> v(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        v[i * n + i] = 1.0;
    }

    for (int sweep = 0; sweep < 100; ++sweep) {
        double off = 0.0;
        double total = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                const double x = a[i * n + j] * a[i * n + j];
                total += x;
                if (i != j) {
                    off += x;
                }
            }
        }
        if (off <= 1e-30 * std::max(total, 1e-300)) {
            break;
        }
        for (std::size_t p = 0; p + 1 < n; ++p) {
            for (std::size_t q = p + 1; q < n; ++q) {
                const double apq = a[p * n + q];
                if (std::fabs(apq) < 1e-300) {
                    continue;
                }
                const double theta = (a[q * n + q] - a[p * n + p]) / (2.0 * apq);
                const double t = (theta >= 0 ? 1.0 : -1.0) / (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
                const double c = 1.0 / std::sqrt(t * t + 1.0);
                const double s = t * c;
                for (std::size_t k = 0; k < n; ++k) {
                    const double akp = a[k * n + p];
                    const double akq = a[k * n + q];
                    a[k * n + p] = c * akp - s * akq;
                    a[k * n + q] = s * akp + c * akq;
                }
                for (std::size_t k = 0; k < n; ++k) {
                    const double apk = a[p * n + k];
                    const double aqk = a[q * n + k];
                    a[p * n + k] = c * apk - s * aqk;
                    a[q * n + k] = s * apk + c * aqk;
                }
                for (std::size_t k = 0; k < n; ++k) {
                    const double vkp = v[k * n + p];
                    const double vkq = v[k * n + q];
                    v[k * n + p] = c * vkp - s * vkq;
                    v[k * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }

    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](std::size_t x, std::size_t y) { return a[x * n + x] > a[y * n + y]; });
    values.assign(n, 0.0);
    vectors.assign(n * n, 0.0);
    for (std::size_t c = 0; c < n; ++c) {
        const std::size_t src = order[c];
        values[c] = a[src * n + src];
        // Deterministic sign: largest-magnitude component positive.
        std::size_t big = 0;
        for (std::size_t k = 1; k < n; ++k) {
            if (std::fabs(v[k * n + src]) > std::fabs(v[big * n + src])) {
                big = k;
            }
        }
        const double sign = v[big * n + src] < 0 ? -1.0 : 1.0;
        for (std::size_t k = 0; k < n; ++k) {
            vectors[k * n + c] = sign * v[k * n + src];
        }
    }
}

std::vector<double> column_means(const Matrix &X) {
    std::vector<double> mean(X.cols, 0.0);
    for (std::size_t r = 0; r < X.rows; ++r) {
        const double *x = X.row(r);
        for (std::size_t c = 0; c < X.cols; ++c) {
            mean[c] += x[c];
        }
    }
    for (double &m : mean) {
        m /= static_cast<double>(X.rows);
    }
    return mean;
}

std::vector<double> covariance(const Matrix &X, const std::vector<double> &means) {
    const std::size_t d = X.cols;
    std::vector<double> cov(d * d, 0.0);
    std::vector<double> centered(d);
    for (std::size_t r = 0; r < X.rows; ++r) {
        const double *x = X.row(r);
        for (std::size_t c = 0; c < d; ++c) {
            centered[c] = x[c] - means[c];
        }
        for (std::size_t i = 0; i < d; ++i) {
            for (std::size_t j = 0; j <= i; ++j) {
                cov[i * d + j] += centered[i] * centered[j];
            }
        }
    }
    for (std::size_t i = 0; i < d; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            cov[i * d + j] /= static_cast<double>(X.rows);
            cov[j * d + i] = cov[i * d + j];
        }
    }
    return cov;
}

}  // namespace mlcore
