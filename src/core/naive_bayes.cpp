#include "naive_bayes.hpp"

namespace mlcore {

namespace {

std::vector<double> class_log_priors(const std::vector<int> &y, std::size_t K, const std::vector<double> &fixed) {
    std::vector<double> priors(K, 0.0);
    if (!fixed.empty()) {
        double total = 0.0;
        for (double p : fixed) {
            total += p;
        }
        for (std::size_t c = 0; c < K; ++c) {
            priors[c] = std::log(fixed[c] / total);
        }
        return priors;
    }
    for (int label : y) {
        priors[static_cast<std::size_t>(label)] += 1.0;
    }
    for (double &p : priors) {
        p = std::log(p / static_cast<double>(y.size()));
    }
    return priors;
}

}  // namespace

void CategoricalNB::fit(const std::vector<int> &X, std::size_t n, std::size_t d, const std::vector<std::size_t> &n_categories,
                        const std::vector<int> &y, std::size_t n_classes) {
    if (n == 0 || d == 0 || y.size() != n) {
        throw std::invalid_argument("Training set must contain at least one sample with at least one feature");
    }
    if (!fixed_priors.empty() && fixed_priors.size() != n_classes) {
        throw std::invalid_argument("priors must contain one probability per class");
    }
    std::vector<std::vector<double>> probs(d);
    std::vector<double> class_counts(n_classes, 0.0);
    for (int label : y) {
        class_counts[static_cast<std::size_t>(label)] += 1.0;
    }
    for (std::size_t f = 0; f < d; ++f) {
        const std::size_t V = n_categories[f];
        std::vector<double> &table = probs[f];
        table.assign(n_classes * (V + 1), 0.0);
        for (std::size_t r = 0; r < n; ++r) {
            const int cat = X[r * d + f];
            table[static_cast<std::size_t>(y[r]) * (V + 1) + static_cast<std::size_t>(cat)] += 1.0;
        }
        for (std::size_t c = 0; c < n_classes; ++c) {
            // V known categories plus one slot for unseen ones.
            const double denominator = class_counts[c] + smoothing * static_cast<double>(V + 1);
            for (std::size_t v = 0; v <= V; ++v) {
                double &cell = table[c * (V + 1) + v];
                cell = std::log((cell + smoothing) / denominator);
            }
        }
    }
    classes = n_classes;
    features = d;
    categories = n_categories;
    log_probs = std::move(probs);
    log_priors = class_log_priors(y, n_classes, fixed_priors);
}

void CategoricalNB::log_likelihoods(const int *x, double *out) const {
    for (std::size_t c = 0; c < classes; ++c) {
        double sum = log_priors[c];
        for (std::size_t f = 0; f < features; ++f) {
            const std::size_t V = categories[f];
            const std::size_t v = x[f] < 0 ? V : static_cast<std::size_t>(x[f]);
            sum += log_probs[f][c * (V + 1) + v];
        }
        out[c] = sum;
    }
}

void CategoricalNB::probabilities(const int *x, double *out) const {
    log_likelihoods(x, out);
    softmax(out, classes);
}

int CategoricalNB::predict_one(const int *x) const {
    std::vector<double> ll(classes);
    log_likelihoods(x, ll.data());
    return static_cast<int>(argmax(ll.data(), classes));
}

void CategoricalNB::save(ArchiveWriter &w) const {
    w.f64(smoothing);
    w.doubles(fixed_priors);
    w.size(classes);
    w.size(features);
    w.sizes(categories);
    for (const std::vector<double> &table : log_probs) {
        w.doubles(table);
    }
    w.doubles(log_priors);
}

void CategoricalNB::load(ArchiveReader &r) {
    smoothing = r.f64();
    fixed_priors = r.doubles();
    classes = r.size(1u << 20);
    features = r.size(1u << 26);
    categories = r.sizes(1u << 30);
    ArchiveReader::require(categories.size() == features);
    log_probs.assign(features, {});
    for (std::size_t f = 0; f < features; ++f) {
        log_probs[f] = r.doubles();
        ArchiveReader::require(log_probs[f].size() == classes * (categories[f] + 1));
    }
    log_priors = r.doubles(classes);
}

void MultinomialNB::fit(const Matrix &X, const std::vector<int> &y, std::size_t n_classes) {
    require_training_set(X, y.size());
    if (!fixed_priors.empty() && fixed_priors.size() != n_classes) {
        throw std::invalid_argument("priors must contain one probability per class");
    }
    const std::size_t d = X.cols;
    std::vector<double> counts(n_classes * d, 0.0);
    for (std::size_t r = 0; r < X.rows; ++r) {
        const double *x = X.row(r);
        double *row = &counts[static_cast<std::size_t>(y[r]) * d];
        for (std::size_t j = 0; j < d; ++j) {
            if (x[j] < 0.0) {
                throw std::invalid_argument("MultinomialNB requires non-negative features (counts or frequencies)");
            }
            row[j] += x[j];
        }
    }
    for (std::size_t c = 0; c < n_classes; ++c) {
        double total = 0.0;
        for (std::size_t j = 0; j < d; ++j) {
            total += counts[c * d + j];
        }
        const double denominator = total + smoothing * static_cast<double>(d);
        for (std::size_t j = 0; j < d; ++j) {
            counts[c * d + j] = std::log((counts[c * d + j] + smoothing) / denominator);
        }
    }
    classes = n_classes;
    features = d;
    log_theta = std::move(counts);
    log_priors = class_log_priors(y, n_classes, fixed_priors);
}

void MultinomialNB::log_likelihoods(const double *x, double *out) const {
    for (std::size_t c = 0; c < classes; ++c) {
        out[c] = log_priors[c] + dot(&log_theta[c * features], x, features);
    }
}

void MultinomialNB::probabilities(const double *x, double *out) const {
    log_likelihoods(x, out);
    softmax(out, classes);
}

int MultinomialNB::predict_one(const double *x) const {
    std::vector<double> ll(classes);
    log_likelihoods(x, ll.data());
    return static_cast<int>(argmax(ll.data(), classes));
}

void MultinomialNB::save(ArchiveWriter &w) const {
    w.f64(smoothing);
    w.doubles(fixed_priors);
    w.size(classes);
    w.size(features);
    w.doubles(log_theta);
    w.doubles(log_priors);
}

void MultinomialNB::load(ArchiveReader &r) {
    smoothing = r.f64();
    fixed_priors = r.doubles();
    classes = r.size(1u << 20);
    features = r.size(1u << 26);
    log_theta = r.doubles();
    ArchiveReader::require(log_theta.size() == classes * features);
    log_priors = r.doubles(classes);
}

}  // namespace mlcore
