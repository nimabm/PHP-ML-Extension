#pragma once

#include "common.hpp"

namespace mlcore {

// Multinomial (softmax) logistic regression trained with full-batch gradient
// descent. Binary problems are the two-class case of the same model.
class LogisticRegression {
public:
    double learning_rate = 0.1;
    long max_iterations = 1000;
    double l2 = 0.0;
    double tolerance = 1e-6;

    std::size_t classes = 0;
    std::size_t features = 0;
    std::vector<double> weights;  // classes x features, row-major
    std::vector<double> bias;     // one per class

    // `y` holds class indices in [0, n_classes).
    void fit(const Matrix &X, const std::vector<int> &y, std::size_t n_classes);
    void probabilities(const double *x, double *out) const;
    int predict_one(const double *x) const;

    bool fitted() const { return classes > 0; }

private:
    void scores(const double *x, const double *w, const double *b, double *out) const;
};

}  // namespace mlcore
