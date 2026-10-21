#include "gaussian_nb.hpp"

#include <algorithm>

namespace mlcore {

namespace {
constexpr double kTwoPi = 6.283185307179586476925286766559;
}

void GaussianNB::fit(const Matrix &X, const std::vector<int> &y, std::size_t n_classes) {
    require_training_set(X, y.size());
    if (n_classes == 0) {
        throw std::invalid_argument("GaussianNB needs at least one class");
    }
    const std::size_t n = X.rows;
    const std::size_t d = X.cols;
    const std::size_t K = n_classes;

    std::vector<double> mu(K * d, 0.0);
    std::vector<double> var(K * d, 0.0);
    std::vector<double> counts(K, 0.0);

    for (std::size_t r = 0; r < n; ++r) {
        const std::size_t c = static_cast<std::size_t>(y[r]);
        counts[c] += 1.0;
        const double *x = X.row(r);
        for (std::size_t j = 0; j < d; ++j) {
            mu[c * d + j] += x[j];
        }
    }
    for (std::size_t c = 0; c < K; ++c) {
        if (counts[c] == 0.0) {
            throw std::invalid_argument("Every class must have at least one training sample");
        }
        for (std::size_t j = 0; j < d; ++j) {
            mu[c * d + j] /= counts[c];
        }
    }
    for (std::size_t r = 0; r < n; ++r) {
        const std::size_t c = static_cast<std::size_t>(y[r]);
        const double *x = X.row(r);
        for (std::size_t j = 0; j < d; ++j) {
            const double diff = x[j] - mu[c * d + j];
            var[c * d + j] += diff * diff;
        }
    }

    // Smoothing is relative to the largest per-feature variance of the whole set.
    double max_feature_var = 0.0;
    for (std::size_t j = 0; j < d; ++j) {
        double mean = 0.0;
        for (std::size_t r = 0; r < n; ++r) {
            mean += X(r, j);
        }
        mean /= static_cast<double>(n);
        double v = 0.0;
        for (std::size_t r = 0; r < n; ++r) {
            const double diff = X(r, j) - mean;
            v += diff * diff;
        }
        max_feature_var = std::max(max_feature_var, v / static_cast<double>(n));
    }
    double epsilon = var_smoothing * max_feature_var;
    if (!(epsilon > 0.0)) {
        epsilon = var_smoothing > 0.0 ? var_smoothing : 1e-300;
    }

    std::vector<double> priors(K);
    for (std::size_t c = 0; c < K; ++c) {
        for (std::size_t j = 0; j < d; ++j) {
            var[c * d + j] = var[c * d + j] / counts[c] + epsilon;
        }
        priors[c] = std::log(counts[c] / static_cast<double>(n));
    }

    classes = K;
    features = d;
    means = std::move(mu);
    variances = std::move(var);
    log_priors = std::move(priors);
}

void GaussianNB::log_likelihoods(const double *x, double *out) const {
    for (std::size_t c = 0; c < classes; ++c) {
        const double *mu = &means[c * features];
        const double *var = &variances[c * features];
        double sum = 0.0;
        for (std::size_t j = 0; j < features; ++j) {
            const double diff = x[j] - mu[j];
            sum += std::log(kTwoPi * var[j]) + diff * diff / var[j];
        }
        out[c] = log_priors[c] - 0.5 * sum;
    }
}

void GaussianNB::probabilities(const double *x, double *out) const {
    log_likelihoods(x, out);
    softmax(out, classes);
}

int GaussianNB::predict_one(const double *x) const {
    std::vector<double> ll(classes);
    log_likelihoods(x, ll.data());
    return static_cast<int>(argmax(ll.data(), classes));
}

}  // namespace mlcore
