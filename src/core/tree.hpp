#pragma once

#include "archive.hpp"
#include "common.hpp"

namespace mlcore {

struct TreeParams {
    std::size_t max_height = std::numeric_limits<std::size_t>::max();
    // A node with this many samples or fewer becomes a leaf.
    std::size_t max_leaf_size = 3;
    // Minimum impurity decrease for a split to be kept.
    double min_purity_increase = 1e-7;
    // Features considered per split; 0 = all.
    std::size_t max_features = 0;
    // Extremely randomized: one random threshold per candidate feature.
    bool extra = false;

    void save(ArchiveWriter &w) const;
    void load(ArchiveReader &r);
};

// CART decision tree. Classification uses Gini impurity and stores a class
// distribution per leaf; regression uses variance and stores the mean. Sample
// weights are supported (used by boosting).
class DecisionTree {
public:
    struct Node {
        std::int32_t feature = -1;  // -1 marks a leaf
        double threshold = 0.0;     // x[feature] < threshold goes left
        std::int32_t left = -1;
        std::int32_t right = -1;
        std::uint32_t leaf = 0;     // index of the leaf's values
        double weight = 0.0;        // total sample weight that reached the node
    };

    TreeParams params;
    bool classification = true;
    std::size_t classes = 0;  // width of a leaf value: classes, or 1 for regression
    std::size_t features = 0;
    std::vector<Node> nodes;
    std::vector<double> leaf_values;
    std::vector<double> importances;  // total weighted impurity decrease per feature
    std::size_t height = 0;

    // targets: class index (as double) or regression value per sample.
    // weights: empty for unit weights. rows: the samples to train on (repeats allowed).
    void fit(const Matrix &X, const std::vector<double> &targets, const std::vector<double> &weights,
             std::vector<std::size_t> rows, Rng &rng);

    const double *leaf_for(const double *x) const;
    double value(const double *x) const { return *leaf_for(x); }
    std::size_t leaves() const { return width() == 0 ? 0 : leaf_values.size() / width(); }
    std::size_t width() const { return classification ? classes : 1; }
    bool fitted() const { return !nodes.empty(); }
    std::vector<double> normalized_importances() const;

    void save(ArchiveWriter &w) const;
    void load(ArchiveReader &r);
};

// Bagged ensemble of decision trees (Random Forest / Extra Trees).
class Forest {
public:
    TreeParams params;
    bool classification = true;
    std::size_t estimators = 100;
    double ratio = 1.0;      // bootstrap sample size as a fraction of the training set
    bool balanced = false;   // classification: sample classes with equal probability
    bool has_seed = false;
    std::uint64_t seed = 0;

    std::size_t classes = 0;
    std::size_t features = 0;
    std::vector<DecisionTree> trees;

    void fit(const Matrix &X, const std::vector<double> &targets, std::size_t n_classes, unsigned threads);
    void probabilities(const double *x, double *out) const;
    double value(const double *x) const;
    std::vector<double> importances() const;
    bool fitted() const { return !trees.empty(); }

    void save(ArchiveWriter &w) const;
    void load(ArchiveReader &r);
};

}  // namespace mlcore
