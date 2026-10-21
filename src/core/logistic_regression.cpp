#include "logistic_regression.hpp"

#include <algorithm>

namespace mlcore {

namespace {

double sigmoid(double z) {
    if (z >= 0.0) {
        return 1.0 / (1.0 + std::exp(-z));
    }
    const double e = std::exp(z);
    return e / (1.0 + e);
}

}  // namespace

void LogisticRegression::scores(const double *x, const double *w, const double *b, double *out) const {
    for (std::size_t k = 0; k < classes; ++k) {
        out[k] = b[k] + dot(w + k * features, x, features);
    }
}

void LogisticRegression::fit(const Matrix &X, const std::vector<int> &y, std::size_t n_classes) {
    require_training_set(X, y.size());
    if (n_classes < 2) {
        throw std::invalid_argument("LogisticRegression needs at least two distinct classes");
    }

    const std::size_t n = X.rows;
    const std::size_t d = X.cols;
    const std::size_t K = n_classes;
    const double inv_n = 1.0 / static_cast<double>(n);

    // Train into locals so a failed fit leaves the previous model intact.
    LogisticRegression next = *this;
    next.classes = K;
    next.features = d;
    next.weights.assign(K * d, 0.0);
    next.bias.assign(K, 0.0);

    if (K == 2) {
        // Two-class softmax is a sigmoid of the score difference v = w1 - w0, and
        // gradient descent keeps w0 = -w1 throughout. Training v alone halves the
        // work; it is stored back as the equivalent pair (-v/2, v/2).
        std::vector<double> v(d, 0.0);
        std::vector<double> grad_v(d);
        double c = 0.0;

        for (long it = 0; it < max_iterations; ++it) {
            std::fill(grad_v.begin(), grad_v.end(), 0.0);
            double grad_c = 0.0;

            for (std::size_t r = 0; r < n; ++r) {
                const double *x = X.row(r);
                const double err = sigmoid(c + dot(v.data(), x, d)) - static_cast<double>(y[r]);
                grad_c += err;
                for (std::size_t j = 0; j < d; ++j) {
                    grad_v[j] += err * x[j];
                }
            }

            // Each class's gradient is +-(g/n + l2 * v/2), so v moves by twice that
            // and the stopping rule sees the same largest component as the softmax form.
            const double gc = grad_c * inv_n;
            double max_grad = std::fabs(gc);
            c -= 2.0 * learning_rate * gc;
            for (std::size_t j = 0; j < d; ++j) {
                const double gv = grad_v[j] * inv_n + 0.5 * l2 * v[j];
                max_grad = std::max(max_grad, std::fabs(gv));
                v[j] -= 2.0 * learning_rate * gv;
            }

            if (!std::isfinite(max_grad)) {
                throw std::runtime_error(
                    "LogisticRegression diverged; lower the learning rate or scale the features (e.g. StandardScaler)");
            }
            if (max_grad < tolerance) {
                break;
            }
        }

        for (std::size_t j = 0; j < d; ++j) {
            next.weights[j] = -0.5 * v[j];
            next.weights[d + j] = 0.5 * v[j];
        }
        next.bias[0] = -0.5 * c;
        next.bias[1] = 0.5 * c;
        *this = std::move(next);
        return;
    }

    std::vector<double> grad_w(K * d);
    std::vector<double> grad_b(K);
    std::vector<double> p(K);

    for (long it = 0; it < max_iterations; ++it) {
        std::fill(grad_w.begin(), grad_w.end(), 0.0);
        std::fill(grad_b.begin(), grad_b.end(), 0.0);

        for (std::size_t r = 0; r < n; ++r) {
            const double *x = X.row(r);
            next.scores(x, next.weights.data(), next.bias.data(), p.data());
            softmax(p.data(), K);
            p[static_cast<std::size_t>(y[r])] -= 1.0;  // dLoss/dScore = p - onehot
            for (std::size_t k = 0; k < K; ++k) {
                const double err = p[k];
                grad_b[k] += err;
                double *g = &grad_w[k * d];
                for (std::size_t j = 0; j < d; ++j) {
                    g[j] += err * x[j];
                }
            }
        }

        double max_grad = 0.0;
        for (std::size_t k = 0; k < K; ++k) {
            const double gb = grad_b[k] * inv_n;
            max_grad = std::max(max_grad, std::fabs(gb));
            next.bias[k] -= learning_rate * gb;
            for (std::size_t j = 0; j < d; ++j) {
                const std::size_t idx = k * d + j;
                const double gw = grad_w[idx] * inv_n + l2 * next.weights[idx];
                max_grad = std::max(max_grad, std::fabs(gw));
                next.weights[idx] -= learning_rate * gw;
            }
        }

        if (!std::isfinite(max_grad)) {
            throw std::runtime_error(
                "LogisticRegression diverged; lower the learning rate or scale the features (e.g. StandardScaler)");
        }
        if (max_grad < tolerance) {
            break;
        }
    }

    *this = std::move(next);
}

void LogisticRegression::probabilities(const double *x, double *out) const {
    scores(x, weights.data(), bias.data(), out);
    softmax(out, classes);
}

int LogisticRegression::predict_one(const double *x) const {
    std::vector<double> s(classes);
    scores(x, weights.data(), bias.data(), s.data());
    return static_cast<int>(argmax(s.data(), classes));
}

}  // namespace mlcore
