#include "logistic_regression.hpp"

#include <algorithm>

namespace mlcore {

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
