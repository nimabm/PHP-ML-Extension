#include "recommender.hpp"

#include <numeric>

#include "linalg.hpp"

namespace mlcore {

// ---- interactions ----------------------------------------------------------------------------

Interactions Interactions::build(const std::vector<std::size_t> &user, const std::vector<std::size_t> &item,
                                 const std::vector<double> &value, std::size_t users, std::size_t items,
                                 bool sum_duplicates) {
    std::vector<std::size_t> order(user.size());
    std::iota(order.begin(), order.end(), 0);
    // Stable, so that among repeated pairs the last one in input order comes last.
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        return user[a] != user[b] ? user[a] < user[b] : item[a] < item[b];
    });
    Interactions out;
    out.users = users;
    out.items = items;
    out.user_start.assign(users + 1, 0);
    for (std::size_t k = 0; k < order.size(); ++k) {
        const std::size_t o = order[k];
        if (k > 0 && user[order[k - 1]] == user[o] && item[order[k - 1]] == item[o]) {
            out.user_values.back() = sum_duplicates ? out.user_values.back() + value[o] : value[o];
            continue;
        }
        out.user_items.push_back(item[o]);
        out.user_values.push_back(value[o]);
        ++out.user_start[user[o] + 1];
    }
    for (std::size_t u = 0; u < users; ++u) {
        out.user_start[u + 1] += out.user_start[u];
    }
    out.index_items();
    return out;
}

void Interactions::index_items() {
    item_start.assign(items + 1, 0);
    for (std::size_t i : user_items) ++item_start[i + 1];
    for (std::size_t i = 0; i < items; ++i) item_start[i + 1] += item_start[i];
    item_users.resize(user_items.size());
    item_values.resize(user_items.size());
    std::vector<std::size_t> next(item_start.begin(), item_start.end() - 1);
    for (std::size_t u = 0; u < users; ++u) {
        for (std::size_t k = user_start[u]; k < user_start[u + 1]; ++k) {
            const std::size_t slot = next[user_items[k]]++;
            item_users[slot] = u;
            item_values[slot] = user_values[k];
        }
    }
    mean = 0.0;
    min = std::numeric_limits<double>::infinity();
    max = -min;
    for (double v : user_values) {
        mean += v;
        min = std::min(min, v);
        max = std::max(max, v);
    }
    mean /= static_cast<double>(std::max<std::size_t>(user_values.size(), 1));
}

bool Interactions::find(std::size_t user, std::size_t item, double &value) const {
    const auto begin = user_items.begin() + static_cast<std::ptrdiff_t>(user_start[user]);
    const auto end = user_items.begin() + static_cast<std::ptrdiff_t>(user_start[user + 1]);
    const auto it = std::lower_bound(begin, end, item);
    if (it == end || *it != item) {
        return false;
    }
    value = user_values[static_cast<std::size_t>(it - user_items.begin())];
    return true;
}

void Interactions::save(ArchiveWriter &w) const {
    w.size(users);
    w.size(items);
    w.sizes(user_start);
    w.sizes(user_items);
    w.doubles(user_values);
}

void Interactions::load(ArchiveReader &r) {
    users = r.size();
    items = r.size();
    user_start = r.sizes();
    user_items = r.sizes(items == 0 ? 0 : items - 1);
    user_values = r.doubles();
    if (users == 0) {
        ArchiveReader::require(items == 0 && user_start.empty() && user_items.empty() && user_values.empty());
        item_start.clear();
        return;
    }
    // Every user and item has at least one interaction, which also bounds the
    // allocations below by the payload size.
    ArchiveReader::require(items <= user_items.size() && user_start.size() == users + 1 && user_start[0] == 0 &&
                           user_start[users] == user_items.size() && user_values.size() == user_items.size());
    for (std::size_t u = 0; u < users; ++u) {
        ArchiveReader::require(user_start[u] < user_start[u + 1], "every user needs an interaction");
        for (std::size_t k = user_start[u] + 1; k < user_start[u + 1]; ++k) {
            ArchiveReader::require(user_items[k - 1] < user_items[k], "items must be sorted and unique");
        }
    }
    for (double v : user_values) ArchiveReader::require(std::isfinite(v), "values must be finite");
    index_items();
    for (std::size_t i = 0; i < items; ++i) {
        ArchiveReader::require(popularity(i) > 0, "every item needs an interaction");
    }
}

namespace {

void random_factors(Matrix &M, std::size_t rows, std::size_t k, double scale, Rng &rng) {
    std::normal_distribution<double> normal(0.0, scale);
    M = Matrix(rows, k);
    for (double &v : M.data) v = normal(rng);
}

double cosine(const double *a, const double *b, std::size_t k) {
    const double den = std::sqrt(dot(a, a, k) * dot(b, b, k));
    return den > 0.0 ? dot(a, b, k) / den : 0.0;
}

void factor_similarities(const Matrix &F, std::size_t item, std::vector<double> &out) {
    out.resize(F.rows);
    for (std::size_t j = 0; j < F.rows; ++j) out[j] = cosine(F.row(item), F.row(j), F.cols);
}

// Gram matrix F^T F (k x k).
std::vector<double> gram(const Matrix &F, unsigned threads) {
    const std::size_t k = F.cols;
    std::vector<double> G(k * k, 0.0);
    parallel_for(k, threads, [&](std::size_t a) {
        for (std::size_t b = 0; b < k; ++b) {
            double s = 0.0;
            for (std::size_t r = 0; r < F.rows; ++r) s += F(r, a) * F(r, b);
            G[a * k + b] = s;
        }
    });
    return G;
}

void check_factors(const Matrix &M, std::size_t rows, std::size_t k) {
    ArchiveReader::require(M.rows == rows && M.cols == k, "inconsistent factor matrix");
    for (double v : M.data) ArchiveReader::require(std::isfinite(v), "factors must be finite");
}

}  // namespace

// ---- matrix factorization ------------------------------------------------------------------

void MatrixFactorization::fit(Interactions interactions, unsigned threads) {
    const std::size_t U = interactions.users, I = interactions.items, k = factors, dim = k + 1;
    MatrixFactorization next = *this;
    next.data = std::move(interactions);
    const Interactions &d = next.data;
    Rng rng = make_rng(has_seed, seed);
    random_factors(next.user_factors, U, k, 0.1, rng);
    random_factors(next.item_factors, I, k, 0.1, rng);
    next.user_bias.assign(U, 0.0);
    next.item_bias.assign(I, 0.0);
    next.losses.clear();

    // One regularized least squares problem per row for [factors, bias], with the
    // other side fixed; the penalty grows with the row's number of ratings.
    auto solve_side = [&](std::size_t rows, const std::vector<std::size_t> &start, const std::vector<std::size_t> &index,
                          const std::vector<double> &values, const Matrix &fixed, const std::vector<double> &fixed_bias,
                          Matrix &factors_out, std::vector<double> &bias_out) {
        parallel_for(rows, threads, [&](std::size_t row) {
            std::vector<double> A(dim * dim, 0.0), rhs(dim, 0.0), z(dim, 1.0);
            for (std::size_t e = start[row]; e < start[row + 1]; ++e) {
                const std::size_t other = index[e];
                std::copy(fixed.row(other), fixed.row(other) + k, z.begin());
                const double target = values[e] - d.mean - fixed_bias[other];
                for (std::size_t a = 0; a < dim; ++a) {
                    rhs[a] += z[a] * target;
                    for (std::size_t b = 0; b < dim; ++b) A[a * dim + b] += z[a] * z[b];
                }
            }
            const double penalty = l2 * static_cast<double>(start[row + 1] - start[row]);
            for (std::size_t a = 0; a < dim; ++a) A[a * dim + a] += penalty;
            if (cholesky_solve(A, rhs, dim)) {
                std::copy(rhs.begin(), rhs.begin() + static_cast<std::ptrdiff_t>(k), factors_out.row(row));
                bias_out[row] = rhs[k];
            }
        });
    };
    for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
        solve_side(U, d.user_start, d.user_items, d.user_values, next.item_factors, next.item_bias, next.user_factors,
                   next.user_bias);
        solve_side(I, d.item_start, d.item_users, d.item_values, next.user_factors, next.user_bias, next.item_factors,
                   next.item_bias);
        std::vector<double> errors(U, 0.0);
        parallel_for(U, threads, [&](std::size_t u) {
            for (std::size_t e = d.user_start[u]; e < d.user_start[u + 1]; ++e) {
                const double diff = d.user_values[e] - next.predict(u, d.user_items[e]);
                errors[u] += diff * diff;
            }
        });
        double total = 0.0;
        for (double v : errors) total += v;
        next.losses.push_back(std::sqrt(total / static_cast<double>(d.size())));
    }
    *this = std::move(next);
}

double MatrixFactorization::raw(std::size_t user, std::size_t item) const {
    double r = data.mean;
    if (user != kUnknown) r += user_bias[user];
    if (item != kUnknown) r += item_bias[item];
    if (user != kUnknown && item != kUnknown) r += dot(user_factors.row(user), item_factors.row(item), factors);
    return r;
}

double MatrixFactorization::predict(std::size_t user, std::size_t item) const {
    return std::min(std::max(raw(user, item), data.min), data.max);
}

void MatrixFactorization::scores(std::size_t user, std::vector<double> &out) const {
    // Unclipped, so that items above the rating range still rank in order.
    out.resize(data.items);
    for (std::size_t i = 0; i < data.items; ++i) out[i] = raw(user, i);
}

void MatrixFactorization::item_similarities(std::size_t item, std::vector<double> &out) const {
    factor_similarities(item_factors, item, out);
}

void MatrixFactorization::save(ArchiveWriter &w) const {
    w.size(factors);
    w.f64(l2);
    w.size(epochs);
    w.boolean(has_seed);
    w.u64(seed);
    data.save(w);
    w.matrix(user_factors);
    w.matrix(item_factors);
    w.doubles(user_bias);
    w.doubles(item_bias);
    w.doubles(losses);
}

void MatrixFactorization::load(ArchiveReader &r) {
    factors = r.size();
    l2 = r.f64();
    epochs = r.size();
    has_seed = r.boolean();
    seed = r.u64();
    ArchiveReader::require(factors >= 1 && l2 >= 0.0 && epochs >= 1, "bad MatrixFactorization parameters");
    data.load(r);
    user_factors = r.matrix();
    item_factors = r.matrix();
    user_bias = r.doubles();
    item_bias = r.doubles();
    losses = r.doubles();
    if (fitted()) {
        check_factors(user_factors, data.users, factors);
        check_factors(item_factors, data.items, factors);
        ArchiveReader::require(user_bias.size() == data.users && item_bias.size() == data.items);
    }
}

// ---- implicit ALS --------------------------------------------------------------------------

void ImplicitALS::fit(Interactions interactions, unsigned threads) {
    const std::size_t U = interactions.users, I = interactions.items, k = factors;
    for (double v : interactions.user_values) {
        if (v < 0.0) {
            throw std::invalid_argument("ImplicitALS needs non-negative interaction values");
        }
    }
    ImplicitALS next = *this;
    next.data = std::move(interactions);
    const Interactions &d = next.data;
    Rng rng = make_rng(has_seed, seed);
    random_factors(next.user_factors, U, k, 0.01, rng);
    random_factors(next.item_factors, I, k, 0.01, rng);

    // x = (Y^T Y + Y^T (C - I) Y + l2 I)^-1 Y^T C p, using only the observed pairs.
    auto solve_side = [&](std::size_t rows, const std::vector<std::size_t> &start, const std::vector<std::size_t> &index,
                          const std::vector<double> &values, const Matrix &fixed, Matrix &out) {
        const std::vector<double> base = gram(fixed, threads);
        parallel_for(rows, threads, [&](std::size_t row) {
            std::vector<double> A = base, rhs(k, 0.0);
            for (std::size_t a = 0; a < k; ++a) A[a * k + a] += l2;
            for (std::size_t e = start[row]; e < start[row + 1]; ++e) {
                const double *y = fixed.row(index[e]);
                const double confidence = 1.0 + alpha * values[e];
                for (std::size_t a = 0; a < k; ++a) {
                    rhs[a] += confidence * y[a];
                    for (std::size_t b = 0; b < k; ++b) A[a * k + b] += (confidence - 1.0) * y[a] * y[b];
                }
            }
            if (cholesky_solve(A, rhs, k)) {
                std::copy(rhs.begin(), rhs.end(), out.row(row));
            }
        });
    };
    for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
        solve_side(U, d.user_start, d.user_items, d.user_values, next.item_factors, next.user_factors);
        solve_side(I, d.item_start, d.item_users, d.item_values, next.user_factors, next.item_factors);
    }
    *this = std::move(next);
}

double ImplicitALS::predict(std::size_t user, std::size_t item) const {
    if (item == kUnknown) return 0.0;
    if (user == kUnknown) {
        std::size_t top = 1;
        for (std::size_t i = 0; i < data.items; ++i) top = std::max(top, data.popularity(i));
        return static_cast<double>(data.popularity(item)) / static_cast<double>(top);
    }
    return dot(user_factors.row(user), item_factors.row(item), factors);
}

void ImplicitALS::scores(std::size_t user, std::vector<double> &out) const {
    out.resize(data.items);
    if (user == kUnknown) {
        std::size_t top = 1;
        for (std::size_t i = 0; i < data.items; ++i) top = std::max(top, data.popularity(i));
        for (std::size_t i = 0; i < data.items; ++i) out[i] = static_cast<double>(data.popularity(i)) / static_cast<double>(top);
        return;
    }
    for (std::size_t i = 0; i < data.items; ++i) out[i] = dot(user_factors.row(user), item_factors.row(i), factors);
}

void ImplicitALS::item_similarities(std::size_t item, std::vector<double> &out) const {
    factor_similarities(item_factors, item, out);
}

void ImplicitALS::save(ArchiveWriter &w) const {
    w.size(factors);
    w.f64(l2);
    w.f64(alpha);
    w.size(epochs);
    w.boolean(has_seed);
    w.u64(seed);
    data.save(w);
    w.matrix(user_factors);
    w.matrix(item_factors);
}

void ImplicitALS::load(ArchiveReader &r) {
    factors = r.size();
    l2 = r.f64();
    alpha = r.f64();
    epochs = r.size();
    has_seed = r.boolean();
    seed = r.u64();
    ArchiveReader::require(factors >= 1 && l2 > 0.0 && alpha >= 0.0 && epochs >= 1, "bad ImplicitALS parameters");
    data.load(r);
    user_factors = r.matrix();
    item_factors = r.matrix();
    if (fitted()) {
        check_factors(user_factors, data.users, factors);
        check_factors(item_factors, data.items, factors);
    }
}

// ---- neighborhood recommender ----------------------------------------------------------------

KNNRecommender::Similarity KNNRecommender::parse(const std::string &name) {
    if (name == "cosine") return Similarity::Cosine;
    if (name == "pearson") return Similarity::Pearson;
    if (name == "jaccard") return Similarity::Jaccard;
    throw std::invalid_argument("Unknown similarity \"" + name + "\" (expected cosine, pearson or jaccard)");
}

std::string KNNRecommender::name(Similarity s) {
    switch (s) {
        case Similarity::Cosine: return "cosine";
        case Similarity::Pearson: return "pearson";
        case Similarity::Jaccard: return "jaccard";
    }
    return "cosine";
}

namespace {

// One side of the interaction matrix in compressed rows.
struct Rows {
    const std::vector<std::size_t> &start, &index;
    const std::vector<double> &values;
    std::size_t size(std::size_t r) const { return start[r + 1] - start[r]; }
};

// Scratch space for co-occurrence sums, reset sparsely between entities.
struct Accumulator {
    std::vector<double> product, left, right;
    std::vector<std::size_t> common, touched;
    explicit Accumulator(std::size_t n) : product(n, 0.0), left(n, 0.0), right(n, 0.0), common(n, 0) {}
};

// Positive similarities of entity e (a row of `rows`) to every entity that shares
// a rater with it, found through `others` (the transposed side).
void co_similarities(std::size_t e, const Rows &rows, const Rows &others, KNNRecommender::Similarity kind,
                     double shrinkage, const std::vector<double> &norms, const std::vector<double> &means,
                     Accumulator &acc, std::vector<std::pair<std::size_t, double>> &out) {
    out.clear();
    for (std::size_t p = rows.start[e]; p < rows.start[e + 1]; ++p) {
        const std::size_t rater = rows.index[p];
        const double a = rows.values[p] - (kind == KNNRecommender::Similarity::Pearson ? means[e] : 0.0);
        for (std::size_t q = others.start[rater]; q < others.start[rater + 1]; ++q) {
            const std::size_t f = others.index[q];
            if (f == e) continue;
            if (acc.common[f]++ == 0) acc.touched.push_back(f);
            const double b = others.values[q] - (kind == KNNRecommender::Similarity::Pearson ? means[f] : 0.0);
            acc.product[f] += a * b;
            acc.left[f] += a * a;
            acc.right[f] += b * b;
        }
    }
    for (std::size_t f : acc.touched) {
        const double c = static_cast<double>(acc.common[f]);
        double s = 0.0;
        switch (kind) {
            case KNNRecommender::Similarity::Cosine:
                s = norms[e] > 0.0 && norms[f] > 0.0 ? acc.product[f] / (norms[e] * norms[f]) : 0.0;
                break;
            case KNNRecommender::Similarity::Pearson: {
                const double den = std::sqrt(acc.left[f] * acc.right[f]);
                s = den > 0.0 ? acc.product[f] / den : 0.0;
                break;
            }
            case KNNRecommender::Similarity::Jaccard:
                s = c / (static_cast<double>(rows.size(e) + rows.size(f)) - c);
                break;
        }
        if (shrinkage > 0.0) s *= c / (c + shrinkage);
        if (s > 0.0) out.push_back({f, std::min(s, 1.0)});
        acc.product[f] = acc.left[f] = acc.right[f] = 0.0;
        acc.common[f] = 0;
    }
    acc.touched.clear();
}

std::vector<double> row_norms(const Rows &rows, std::size_t n) {
    std::vector<double> norms(n, 0.0);
    for (std::size_t r = 0; r < n; ++r) {
        for (std::size_t p = rows.start[r]; p < rows.start[r + 1]; ++p) norms[r] += rows.values[p] * rows.values[p];
        norms[r] = std::sqrt(norms[r]);
    }
    return norms;
}

std::vector<double> row_means(const Rows &rows, std::size_t n) {
    std::vector<double> means(n, 0.0);
    for (std::size_t r = 0; r < n; ++r) {
        for (std::size_t p = rows.start[r]; p < rows.start[r + 1]; ++p) means[r] += rows.values[p];
        means[r] /= static_cast<double>(std::max<std::size_t>(rows.size(r), 1));
    }
    return means;
}

}  // namespace

void KNNRecommender::fit(Interactions interactions, unsigned threads) {
    KNNRecommender next = *this;
    next.data = std::move(interactions);
    const Interactions &d = next.data;
    const std::size_t U = d.users, I = d.items;
    // Baselines (Koren): item biases shrunk with 25 pseudo-ratings, user biases with 10.
    next.item_bias.assign(I, 0.0);
    next.user_bias.assign(U, 0.0);
    for (std::size_t i = 0; i < I; ++i) {
        double s = 0.0;
        for (std::size_t p = d.item_start[i]; p < d.item_start[i + 1]; ++p) s += d.item_values[p] - d.mean;
        next.item_bias[i] = s / (25.0 + static_cast<double>(d.popularity(i)));
    }
    for (std::size_t u = 0; u < U; ++u) {
        double s = 0.0;
        for (std::size_t p = d.user_start[u]; p < d.user_start[u + 1]; ++p) {
            s += d.user_values[p] - d.mean - next.item_bias[d.user_items[p]];
        }
        next.user_bias[u] = s / (10.0 + static_cast<double>(d.user_start[u + 1] - d.user_start[u]));
    }

    // With explicit ratings, cosine compares what is left after the baselines
    // (otherwise every pair of positive rating vectors would look similar).
    const bool centered = similarity == Similarity::Cosine && d.min < d.max;
    std::vector<double> user_residuals(d.user_values), item_residuals(d.item_values);
    if (centered) {
        for (std::size_t u = 0; u < U; ++u) {
            for (std::size_t p = d.user_start[u]; p < d.user_start[u + 1]; ++p) {
                user_residuals[p] -= d.mean + next.user_bias[u] + next.item_bias[d.user_items[p]];
            }
        }
        for (std::size_t i = 0; i < I; ++i) {
            for (std::size_t p = d.item_start[i]; p < d.item_start[i + 1]; ++p) {
                item_residuals[p] -= d.mean + next.user_bias[d.item_users[p]] + next.item_bias[i];
            }
        }
    }
    const Rows by_user{d.user_start, d.user_items, user_residuals};
    const Rows by_item{d.item_start, d.item_users, item_residuals};
    const Rows &rows = user_based ? by_user : by_item;
    const Rows &others = user_based ? by_item : by_user;
    const std::size_t E = user_based ? U : I;
    const std::vector<double> norms = row_norms(rows, E), means = row_means(rows, E);
    next.similar.assign(E, {});
    constexpr std::size_t kBlock = 64;
    parallel_for((E + kBlock - 1) / kBlock, threads, [&](std::size_t block) {
        Accumulator acc(E);
        std::vector<std::pair<std::size_t, double>> found;
        for (std::size_t e = block * kBlock; e < std::min(E, (block + 1) * kBlock); ++e) {
            co_similarities(e, rows, others, similarity, shrinkage, norms, means, acc, found);
            const auto better = [](const std::pair<std::size_t, double> &a, const std::pair<std::size_t, double> &b) {
                return a.second != b.second ? a.second > b.second : a.first < b.first;
            };
            const std::size_t keep = std::min(neighbors, found.size());
            std::partial_sort(found.begin(), found.begin() + static_cast<std::ptrdiff_t>(keep), found.end(), better);
            next.similar[e].assign(found.begin(), found.begin() + static_cast<std::ptrdiff_t>(keep));
        }
    });
    *this = std::move(next);
}

double KNNRecommender::predict(std::size_t user, std::size_t item) const {
    auto baseline = [&](std::size_t u, std::size_t i) {
        double b = data.mean;
        if (u != kUnknown) b += user_bias[u];
        if (i != kUnknown) b += item_bias[i];
        return b;
    };
    double estimate = baseline(user, item);
    if (user != kUnknown && item != kUnknown) {
        double num = 0.0, den = 0.0, r = 0.0;
        if (user_based) {
            for (const auto &[v, s] : similar[user]) {
                if (data.find(v, item, r)) {
                    num += s * (r - baseline(v, item));
                    den += s;
                }
            }
        } else {
            for (const auto &[j, s] : similar[item]) {
                if (data.find(user, j, r)) {
                    num += s * (r - baseline(user, j));
                    den += s;
                }
            }
        }
        if (den > 0.0) estimate += num / den;
    }
    return std::min(std::max(estimate, data.min), data.max);
}

void KNNRecommender::scores(std::size_t user, std::vector<double> &out) const {
    out.assign(data.items, 0.0);
    if (user == kUnknown) {
        std::size_t top = 1;
        for (std::size_t i = 0; i < data.items; ++i) top = std::max(top, data.popularity(i));
        for (std::size_t i = 0; i < data.items; ++i) out[i] = static_cast<double>(data.popularity(i)) / static_cast<double>(top);
        return;
    }
    if (data.min < data.max) {
        // Explicit ratings: rank by predicted rating.
        for (std::size_t i = 0; i < data.items; ++i) out[i] = predict(user, i);
        return;
    }
    // Implicit feedback (every value equal): rank by summed similarity.
    if (user_based) {
        for (const auto &[v, s] : similar[user]) {
            for (std::size_t p = data.user_start[v]; p < data.user_start[v + 1]; ++p) out[data.user_items[p]] += s;
        }
    } else {
        for (std::size_t p = data.user_start[user]; p < data.user_start[user + 1]; ++p) {
            for (const auto &[i, s] : similar[data.user_items[p]]) out[i] += s;
        }
    }
}

void KNNRecommender::item_similarities(std::size_t item, std::vector<double> &out) const {
    out.assign(data.items, 0.0);
    out[item] = 1.0;
    if (!user_based) {
        for (const auto &[j, s] : similar[item]) out[j] = s;
        return;
    }
    // User-based models keep no item neighbors: compute them for this item.
    const Rows by_user{data.user_start, data.user_items, data.user_values};
    const Rows by_item{data.item_start, data.item_users, data.item_values};
    Accumulator acc(data.items);
    std::vector<std::pair<std::size_t, double>> found;
    co_similarities(item, by_item, by_user, similarity, shrinkage, row_norms(by_item, data.items),
                    row_means(by_item, data.items), acc, found);
    for (const auto &[j, s] : found) out[j] = s;
}

void KNNRecommender::save(ArchiveWriter &w) const {
    w.size(neighbors);
    w.u8(static_cast<std::uint8_t>(similarity));
    w.boolean(user_based);
    w.f64(shrinkage);
    data.save(w);
    w.doubles(user_bias);
    w.doubles(item_bias);
    w.size(similar.size());
    for (const auto &list : similar) {
        w.size(list.size());
        for (const auto &[index, s] : list) {
            w.size(index);
            w.f64(s);
        }
    }
}

void KNNRecommender::load(ArchiveReader &r) {
    neighbors = r.size();
    const std::uint8_t kind = r.u8();
    ArchiveReader::require(kind <= static_cast<std::uint8_t>(Similarity::Jaccard), "bad similarity");
    similarity = static_cast<Similarity>(kind);
    user_based = r.boolean();
    shrinkage = r.f64();
    ArchiveReader::require(neighbors >= 1 && shrinkage >= 0.0, "bad KNNRecommender parameters");
    data.load(r);
    user_bias = r.doubles();
    item_bias = r.doubles();
    const std::size_t E = user_based ? data.users : data.items;
    const std::size_t lists = r.count(8);
    ArchiveReader::require(lists == (fitted() ? E : 0), "inconsistent neighbor lists");
    similar.assign(lists, {});
    for (auto &list : similar) {
        const std::size_t size = r.count(16);
        ArchiveReader::require(size <= neighbors, "too many neighbors");
        for (std::size_t m = 0; m < size; ++m) {
            const std::size_t index = r.size(E - 1);
            const double s = r.f64();
            ArchiveReader::require(s > 0.0 && s <= 1.0, "similarities must be in (0, 1]");
            list.push_back({index, s});
        }
    }
    if (fitted()) {
        ArchiveReader::require(user_bias.size() == data.users && item_bias.size() == data.items);
    }
}

}  // namespace mlcore
