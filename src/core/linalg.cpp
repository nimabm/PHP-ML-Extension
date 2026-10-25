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

namespace {

// Matrices up to this size use cyclic Jacobi (very accurate, O(n^3) per sweep);
// larger ones go through tridiagonalization.
constexpr std::size_t kJacobiLimit = 32;

// Flips `v` (n entries, `stride` apart) so that its largest-magnitude component is positive.
void normalize_sign(double *v, std::size_t n, std::size_t stride) {
    std::size_t big = 0;
    for (std::size_t k = 1; k < n; ++k) {
        if (std::fabs(v[k * stride]) > std::fabs(v[big * stride])) {
            big = k;
        }
    }
    if (v[big * stride] < 0) {
        for (std::size_t k = 0; k < n; ++k) {
            v[k * stride] = -v[k * stride];
        }
    }
}

void jacobi_eigen(std::vector<double> a, std::size_t n, std::vector<double> &values, std::vector<double> &vectors) {
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
        for (std::size_t k = 0; k < n; ++k) {
            vectors[k * n + c] = v[k * n + src];
        }
        normalize_sign(&vectors[c], n, n);
    }
}

// Householder reduction A = Q T Q^T of a symmetric matrix (overwritten). T has
// diagonal d and off-diagonal e (e[i] couples i and i + 1). Q = H_0 ... H_{n-3}
// with H_k = I - beta[k] v v^T, where v is stored in row k of `reflectors` and
// acts on indices k + 1 .. n - 1.
void tridiagonalize(std::vector<double> &a, std::size_t n, std::vector<double> &d, std::vector<double> &e,
                    std::vector<double> &reflectors, std::vector<double> &beta, unsigned threads) {
    d.assign(n, 0.0);
    e.assign(n > 0 ? n - 1 : 0, 0.0);
    beta.assign(n, 0.0);
    reflectors.assign(n * n, 0.0);
    std::vector<double> v(n), w(n);
    for (std::size_t k = 0; k + 2 < n; ++k) {
        const std::size_t off = k + 1, m = n - off;
        d[k] = a[k * n + k];
        const double x0 = a[off * n + k];
        double sigma = 0.0;
        for (std::size_t i = 1; i < m; ++i) {
            const double x = a[(off + i) * n + k];
            sigma += x * x;
        }
        if (sigma == 0.0) {
            e[k] = x0;  // the column is already reduced
            continue;
        }
        const double mu = std::sqrt(x0 * x0 + sigma);
        const double v0 = x0 <= 0.0 ? x0 - mu : -sigma / (x0 + mu);
        const double b = 2.0 * v0 * v0 / (sigma + v0 * v0);
        v[0] = 1.0;
        for (std::size_t i = 1; i < m; ++i) {
            v[i] = a[(off + i) * n + k] / v0;
        }
        e[k] = mu;
        beta[k] = b;
        std::copy(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(m), reflectors.begin() + static_cast<std::ptrdiff_t>(k * n));

        // A22 <- H A22 H = A22 - v w^T - w v^T with w = p - (b/2)(p.v) v, p = b A22 v.
        const unsigned workers = m >= 256 ? threads : 1;
        parallel_for(m, workers, [&](std::size_t i) { w[i] = b * dot(&a[(off + i) * n + off], v.data(), m); });
        const double half = 0.5 * b * dot(w.data(), v.data(), m);
        for (std::size_t i = 0; i < m; ++i) {
            w[i] -= half * v[i];
        }
        parallel_for(m, workers, [&](std::size_t i) {
            double *row = &a[(off + i) * n + off];
            const double vi = v[i], wi = w[i];
            for (std::size_t j = 0; j < m; ++j) {
                row[j] -= vi * w[j] + wi * v[j];
            }
        });
    }
    if (n >= 2) {
        d[n - 2] = a[(n - 2) * n + n - 2];
        e[n - 2] = a[(n - 1) * n + n - 2];
    }
    if (n >= 1) {
        d[n - 1] = a[(n - 1) * n + n - 1];
    }
}

// y <- Q y for the Q of tridiagonalize().
void apply_reflectors(const std::vector<double> &reflectors, const std::vector<double> &beta, std::size_t n, double *y) {
    for (std::size_t k = n >= 3 ? n - 2 : 0; k-- > 0;) {
        if (beta[k] == 0.0) {
            continue;
        }
        const std::size_t off = k + 1, m = n - off;
        const double *v = &reflectors[k * n];
        const double s = beta[k] * dot(v, y + off, m);
        for (std::size_t i = 0; i < m; ++i) {
            y[off + i] -= s * v[i];
        }
    }
}

// Number of eigenvalues of the tridiagonal (d, e^2) that are below x (Sturm sequence).
std::size_t count_below(const std::vector<double> &d, const std::vector<double> &e2, double x, double pivmin) {
    std::size_t count = 0;
    double q = d[0] - x;
    for (std::size_t i = 0;; ++i) {
        if (std::fabs(q) < pivmin) {
            q = -pivmin;
        }
        if (q < 0) {
            ++count;
        }
        if (i + 1 == d.size()) {
            return count;
        }
        q = d[i + 1] - x - e2[i] / q;
    }
}

// Gaussian elimination with partial pivoting of the shifted tridiagonal T - shift I,
// for inverse iteration. U has up to two super-diagonals.
struct TridiagonalLU {
    std::vector<double> u0, u1, u2, l;
    std::vector<char> swapped;

    void factor(const std::vector<double> &d, const std::vector<double> &e, double shift, double tiny) {
        const std::size_t n = d.size();
        u0.resize(n);
        u1.assign(n, 0.0);
        u2.assign(n, 0.0);
        l.assign(n, 0.0);
        swapped.assign(n, 0);
        for (std::size_t i = 0; i < n; ++i) {
            u0[i] = d[i] - shift;
            if (i + 1 < n) {
                u1[i] = e[i];
            }
        }
        for (std::size_t i = 0; i + 1 < n; ++i) {
            const double sub = e[i];
            if (std::fabs(u0[i]) >= std::fabs(sub)) {
                if (u0[i] == 0.0) {
                    u0[i] = tiny;
                }
                l[i] = sub / u0[i];
                u0[i + 1] -= l[i] * u1[i];
            } else {
                const double mult = u0[i] / sub;
                const double next_diag = u0[i + 1];
                const double next_super = i + 2 < n ? u1[i + 1] : 0.0;
                l[i] = mult;
                swapped[i] = 1;
                u0[i] = sub;
                u0[i + 1] = u1[i] - mult * next_diag;
                if (i + 2 < n) {
                    u1[i + 1] = -mult * next_super;
                }
                u1[i] = next_diag;
                u2[i] = next_super;
            }
        }
        for (double &u : u0) {
            if (std::fabs(u) < tiny) {
                u = u < 0 ? -tiny : tiny;
            }
        }
    }

    void solve(std::vector<double> &y) const {
        const std::size_t n = u0.size();
        for (std::size_t i = 0; i + 1 < n; ++i) {
            if (swapped[i]) {
                std::swap(y[i], y[i + 1]);
            }
            y[i + 1] -= l[i] * y[i];
        }
        for (std::size_t i = n; i-- > 0;) {
            double s = y[i];
            if (i + 1 < n) {
                s -= u1[i] * y[i + 1];
            }
            if (i + 2 < n) {
                s -= u2[i] * y[i + 2];
            }
            y[i] = s / u0[i];
        }
    }
};

bool normalize(std::vector<double> &x) {
    const double norm = std::sqrt(dot(x.data(), x.data(), x.size()));
    if (!(norm > 0.0) || !std::isfinite(norm)) {
        return false;
    }
    for (double &v : x) {
        v /= norm;
    }
    return true;
}

}  // namespace

void top_eigen(std::vector<double> &a, std::size_t n, std::size_t k, std::vector<double> &values, Matrix &vectors,
               unsigned threads) {
    k = std::min(k, n);
    values.assign(k, 0.0);
    vectors = Matrix(n, k);
    if (k == 0) {
        return;
    }
    if (n <= kJacobiLimit) {
        std::vector<double> all_values, all_vectors;
        jacobi_eigen(std::move(a), n, all_values, all_vectors);
        for (std::size_t c = 0; c < k; ++c) {
            values[c] = all_values[c];
            for (std::size_t r = 0; r < n; ++r) {
                vectors(r, c) = all_vectors[r * n + c];
            }
        }
        return;
    }

    std::vector<double> d, e, reflectors, beta;
    tridiagonalize(a, n, d, e, reflectors, beta, threads);
    std::vector<double> e2(e.size());
    double max_e2 = 0.0;
    for (std::size_t i = 0; i < e.size(); ++i) {
        e2[i] = e[i] * e[i];
        max_e2 = std::max(max_e2, e2[i]);
    }
    const double eps = std::numeric_limits<double>::epsilon();
    const double pivmin = std::numeric_limits<double>::min() * std::max(1.0, max_e2);
    // Gershgorin interval containing every eigenvalue.
    double lo = std::numeric_limits<double>::infinity(), hi = -lo;
    for (std::size_t i = 0; i < n; ++i) {
        const double radius = (i > 0 ? std::fabs(e[i - 1]) : 0.0) + (i + 1 < n ? std::fabs(e[i]) : 0.0);
        lo = std::min(lo, d[i] - radius);
        hi = std::max(hi, d[i] + radius);
    }
    const double tnorm = std::max(std::fabs(lo), std::fabs(hi));
    lo -= 2.0 * eps * tnorm + 2.0 * pivmin;
    hi += 2.0 * eps * tnorm + 2.0 * pivmin;

    // Eigenvalues by bisection: the c-th largest has ascending index n - 1 - c.
    parallel_for(k, threads, [&](std::size_t c) {
        const std::size_t index = n - 1 - c;
        double a_lo = lo, a_hi = hi;
        for (int step = 0; step < 128; ++step) {
            const double mid = 0.5 * (a_lo + a_hi);
            if (a_hi - a_lo <= 4.0 * eps * std::max(std::fabs(a_lo), std::fabs(a_hi)) + 2.0 * pivmin || mid == a_lo ||
                mid == a_hi) {
                break;
            }
            if (count_below(d, e2, mid, pivmin) > index) {
                a_hi = mid;
            } else {
                a_lo = mid;
            }
        }
        values[c] = 0.5 * (a_lo + a_hi);
    });

    // Eigenvectors of T by inverse iteration, on T scaled to unit norm so that
    // the near-singular solves cannot overflow. Vectors of eigenvalues closer than
    // 1e-3 ||T|| are kept orthogonal to each other (as LAPACK's dstein does).
    std::vector<std::vector<double>> t_vectors(k, std::vector<double>(n, 0.0));
    if (tnorm == 0.0) {
        for (std::size_t c = 0; c < k; ++c) {
            t_vectors[c][c] = 1.0;  // T = 0: any orthonormal basis
        }
    } else {
        for (double &v : d) v /= tnorm;
        for (double &v : e) v /= tnorm;
        Rng rng(0x9E3779B97F4A7C15ULL);  // fixed: results must not depend on a seed
        std::uniform_real_distribution<double> unit(-1.0, 1.0);
        TridiagonalLU lu;
        std::size_t cluster_start = 0;
        for (std::size_t c = 0; c < k; ++c) {
            if (c > 0 && std::fabs(values[c - 1] - values[c]) > 1e-3 * tnorm) {
                cluster_start = c;
            }
            lu.factor(d, e, values[c] / tnorm, eps);
            std::vector<double> &x = t_vectors[c];
            for (double &v : x) {
                v = unit(rng);
            }
            for (int iteration = 0; iteration < 5; ++iteration) {
                lu.solve(x);
                for (int attempt = 0;; ++attempt) {
                    for (std::size_t p = cluster_start; p < c; ++p) {
                        const double s = dot(x.data(), t_vectors[p].data(), n);
                        for (std::size_t i = 0; i < n; ++i) {
                            x[i] -= s * t_vectors[p][i];
                        }
                    }
                    if (normalize(x) || attempt == 3) {
                        break;
                    }
                    for (double &v : x) {
                        v = unit(rng);  // restart from a fresh direction
                    }
                }
            }
        }
    }

    parallel_for(k, threads, [&](std::size_t c) {
        std::vector<double> &y = t_vectors[c];
        apply_reflectors(reflectors, beta, n, y.data());
        normalize(y);
        normalize_sign(y.data(), n, 1);
        for (std::size_t r = 0; r < n; ++r) {
            vectors(r, c) = y[r];
        }
    });
}

void symmetric_eigen(std::vector<double> a, std::size_t n, std::vector<double> &values, std::vector<double> &vectors) {
    if (n <= kJacobiLimit) {
        jacobi_eigen(std::move(a), n, values, vectors);
        return;
    }
    Matrix columns;
    top_eigen(a, n, n, values, columns);
    vectors = std::move(columns.data);
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
