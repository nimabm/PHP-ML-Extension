#pragma once

#include "common.hpp"

namespace mlcore {

// Gaussian naive Bayes: each feature is modelled as an independent normal
// distribution per class.
class GaussianNB {
public:
    // Fraction of the largest feature variance added to every variance for
    // numerical stability (same meaning as scikit-learn's var_smoothing).
    double var_smoothing = 1e-9;

    std::size_t classes = 0;
    std::size_t features = 0;
    std::vector<double> means;      // classes x features
    std::vector<double> variances;  // classes x features
    std::vector<double> log_priors; // one per class

    void fit(const Matrix &X, const std::vector<int> &y, std::size_t n_classes);
    void log_likelihoods(const double *x, double *out) const;
    void probabilities(const double *x, double *out) const;
    int predict_one(const double *x) const;

    bool fitted() const { return classes > 0; }
};

}  // namespace mlcore
