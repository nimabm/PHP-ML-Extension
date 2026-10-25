#pragma once

// Collaborative filtering: matrix factorization for explicit ratings, implicit
// feedback ALS and neighborhood (item- or user-based) recommenders. Users and
// items are dense indices; the PHP layer maps them to labels.

#include "archive.hpp"
#include "common.hpp"

namespace mlcore {

// User-item interactions stored twice, as compressed rows by user and by item.
struct Interactions {
    std::size_t users = 0, items = 0;
    std::vector<std::size_t> user_start, user_items;  // CSR by user, items ascending
    std::vector<double> user_values;
    std::vector<std::size_t> item_start, item_users;  // CSR by item, users ascending
    std::vector<double> item_values;
    double mean = 0.0, min = 0.0, max = 0.0;

    // From (user, item, value) triples. A repeated pair keeps its last value, or
    // the sum of its values when `sum_duplicates` is set (implicit feedback).
    static Interactions build(const std::vector<std::size_t> &user, const std::vector<std::size_t> &item,
                              const std::vector<double> &value, std::size_t users, std::size_t items,
                              bool sum_duplicates);
    std::size_t size() const { return user_items.size(); }
    std::size_t popularity(std::size_t item) const { return item_start[item + 1] - item_start[item]; }
    // Value of (user, item), or false if there is none.
    bool find(std::size_t user, std::size_t item, double &value) const;

    void save(ArchiveWriter &w) const;
    void load(ArchiveReader &r);

private:
    void index_items();
};

// Every recommender scores items for a user (user = npos for an unknown user)
// and predicts single pairs; unknown users or items fall back to baselines.
constexpr std::size_t kUnknown = static_cast<std::size_t>(-1);

// Explicit ratings: r = mean + user bias + item bias + p_u . q_i, fitted by
// weighted-lambda alternating least squares (ALS-WR).
class MatrixFactorization {
public:
    std::size_t factors = 10;
    double l2 = 0.1;
    std::size_t epochs = 15;
    bool has_seed = false;
    std::uint64_t seed = 0;

    Interactions data;
    Matrix user_factors, item_factors;
    std::vector<double> user_bias, item_bias;
    std::vector<double> losses;  // training RMSE after each epoch

    void fit(Interactions interactions, unsigned threads);
    // Clipped to the range of the training ratings.
    double predict(std::size_t user, std::size_t item) const;
    double raw(std::size_t user, std::size_t item) const;
    void scores(std::size_t user, std::vector<double> &out) const;
    void item_similarities(std::size_t item, std::vector<double> &out) const;
    bool fitted() const { return data.users > 0; }
    void save(ArchiveWriter &w) const;
    void load(ArchiveReader &r);
};

// Implicit feedback (Hu, Koren & Volinsky): preference 1 for observed pairs with
// confidence 1 + alpha * value, 0 elsewhere.
class ImplicitALS {
public:
    std::size_t factors = 10;
    double l2 = 0.01;
    double alpha = 40.0;
    std::size_t epochs = 15;
    bool has_seed = false;
    std::uint64_t seed = 0;

    Interactions data;
    Matrix user_factors, item_factors;

    void fit(Interactions interactions, unsigned threads);
    double predict(std::size_t user, std::size_t item) const;
    void scores(std::size_t user, std::vector<double> &out) const;
    void item_similarities(std::size_t item, std::vector<double> &out) const;
    bool fitted() const { return data.users > 0; }
    void save(ArchiveWriter &w) const;
    void load(ArchiveReader &r);
};

// Neighborhood collaborative filtering around baseline estimates.
class KNNRecommender {
public:
    enum class Similarity : std::uint8_t { Cosine, Pearson, Jaccard };

    std::size_t neighbors = 20;
    Similarity similarity = Similarity::Cosine;
    bool user_based = false;
    double shrinkage = 0.0;  // similarity * n / (n + shrinkage), n = common raters

    Interactions data;
    std::vector<double> user_bias, item_bias;
    // Most similar items (or users) of each item (or user), most similar first.
    std::vector<std::vector<std::pair<std::size_t, double>>> similar;

    void fit(Interactions interactions, unsigned threads);
    double predict(std::size_t user, std::size_t item) const;
    void scores(std::size_t user, std::vector<double> &out) const;
    void item_similarities(std::size_t item, std::vector<double> &out) const;
    bool fitted() const { return data.users > 0; }
    static Similarity parse(const std::string &name);
    static std::string name(Similarity s);
    void save(ArchiveWriter &w) const;
    void load(ArchiveReader &r);
};

}  // namespace mlcore
