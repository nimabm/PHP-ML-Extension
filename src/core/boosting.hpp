#pragma once

#include "tree.hpp"

namespace mlcore {

// AdaBoost (SAMME, multiclass) with weighted decision trees as weak learners.
class AdaBoost {
public:
    TreeParams base;  // defaults to decision stumps
    std::size_t estimators = 100;
    double rate = 1.0;
    double ratio = 0.8;  // fraction of samples (without replacement) per round
    double min_change = 1e-4;
    bool has_seed = false;
    std::uint64_t seed = 0;

    std::size_t classes = 0, features = 0;
    std::vector<DecisionTree> trees;
    std::vector<double> alphas;
    std::vector<double> losses;  // weighted training error per round

    AdaBoost() { base.max_height = 1; base.max_leaf_size = 1; }
    void fit(const Matrix &X, const std::vector<int> &y, std::size_t n_classes);
    void probabilities(const double *x, double *out) const;
    bool fitted() const { return !trees.empty(); }
    void save(ArchiveWriter &w) const;
    void load(ArchiveReader &r);
};

// LogitBoost (Friedman et al.) for binary classification with regression trees.
class LogitBoost {
public:
    TreeParams base;
    std::size_t estimators = 100;
    double rate = 0.1;
    double ratio = 0.5;
    double min_change = 1e-4;
    bool has_seed = false;
    std::uint64_t seed = 0;

    std::size_t features = 0;
    std::vector<DecisionTree> trees;
    std::vector<double> losses;  // log loss per round

    LogitBoost() { base.max_height = 3; }
    void fit(const Matrix &X, const std::vector<int> &y);  // y in {0, 1}
    double positive_probability(const double *x) const;
    bool fitted() const { return !trees.empty(); }
    void save(ArchiveWriter &w) const;
    void load(ArchiveReader &r);
};

// Gradient-boosted regression trees (least squares) with early stopping on a hold-out set.
class GradientBoost {
public:
    TreeParams base;
    std::size_t estimators = 100;
    double rate = 0.1;
    double ratio = 0.5;
    double min_change = 1e-4;
    std::size_t window = 5;
    double hold_out = 0.1;
    bool has_seed = false;
    std::uint64_t seed = 0;

    std::size_t features = 0;
    double initial = 0.0;
    std::vector<DecisionTree> trees;
    std::vector<double> losses;      // training MSE per round
    std::vector<double> validation;  // hold-out MSE per round (empty without hold-out)

    GradientBoost() { base.max_height = 3; }
    void fit(const Matrix &X, const std::vector<double> &y);
    double predict(const double *x) const;
    bool fitted() const { return features > 0; }
    void save(ArchiveWriter &w) const;
    void load(ArchiveReader &r);
};

// Linear regression with combined L1 and L2 penalties, by coordinate descent.
class ElasticNet {
public:
    double alpha = 1.0;
    double l1_ratio = 0.5;
    long max_iterations = 1000;
    double tolerance = 1e-4;

    std::vector<double> coefficients;
    double intercept = 0.0;
    long iterations = 0;

    void fit(const Matrix &X, const std::vector<double> &y);
    double predict(const double *x) const { return intercept + dot(coefficients.data(), x, coefficients.size()); }
    bool fitted() const { return !coefficients.empty(); }
    void save(ArchiveWriter &w) const;
    void load(ArchiveReader &r);
};

}  // namespace mlcore
