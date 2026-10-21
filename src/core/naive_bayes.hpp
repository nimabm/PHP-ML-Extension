#pragma once

#include "archive.hpp"
#include "common.hpp"

namespace mlcore {

// Naive Bayes over categorical features. Inputs are category indices per
// feature (encoded by the caller); -1 means a category unseen during training,
// which gets the smoothed probability of an empty count.
class CategoricalNB {
public:
    double smoothing = 1.0;
    std::vector<double> fixed_priors;  // per class; empty = learn from data

    std::size_t classes = 0;
    std::size_t features = 0;
    std::vector<std::size_t> categories;            // per feature
    std::vector<std::vector<double>> log_probs;     // per feature: classes x (categories + 1)
    std::vector<double> log_priors;

    // X: n x d category indices, row-major.
    void fit(const std::vector<int> &X, std::size_t n, std::size_t d, const std::vector<std::size_t> &n_categories,
             const std::vector<int> &y, std::size_t n_classes);
    void log_likelihoods(const int *x, double *out) const;
    void probabilities(const int *x, double *out) const;
    int predict_one(const int *x) const;
    bool fitted() const { return classes > 0; }

    void save(ArchiveWriter &w) const;
    void load(ArchiveReader &r);
};

// Multinomial naive Bayes for count features (word counts, TF-IDF).
class MultinomialNB {
public:
    double smoothing = 1.0;
    std::vector<double> fixed_priors;

    std::size_t classes = 0;
    std::size_t features = 0;
    std::vector<double> log_theta;  // classes x features
    std::vector<double> log_priors;

    void fit(const Matrix &X, const std::vector<int> &y, std::size_t n_classes);
    void log_likelihoods(const double *x, double *out) const;
    void probabilities(const double *x, double *out) const;
    int predict_one(const double *x) const;
    bool fitted() const { return classes > 0; }

    void save(ArchiveWriter &w) const;
    void load(ArchiveReader &r);
};

}  // namespace mlcore
