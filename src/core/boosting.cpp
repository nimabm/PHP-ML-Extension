#include "boosting.hpp"

#include <numeric>

namespace mlcore {

namespace {

std::vector<std::size_t> subsample(const std::vector<std::size_t> &pool, double ratio, Rng &rng) {
    std::vector<std::size_t> rows = pool;
    std::shuffle(rows.begin(), rows.end(), rng);
    const std::size_t keep = std::max<std::size_t>(1, static_cast<std::size_t>(std::llround(ratio * static_cast<double>(rows.size()))));
    rows.resize(std::min(keep, rows.size()));
    return rows;
}

std::vector<std::size_t> all_rows(std::size_t n) {
    std::vector<std::size_t> rows(n);
    std::iota(rows.begin(), rows.end(), 0);
    return rows;
}

void save_trees(ArchiveWriter &w, const std::vector<DecisionTree> &trees) {
    w.size(trees.size());
    for (const DecisionTree &t : trees) {
        t.save(w);
    }
}

std::vector<DecisionTree> load_trees(ArchiveReader &r, bool classification, std::size_t width, std::size_t features) {
    std::vector<DecisionTree> trees(r.count(40));
    for (DecisionTree &t : trees) {
        t.load(r);
        ArchiveReader::require(t.classification == classification && t.width() == width && t.features == features);
    }
    return trees;
}

void save_common(ArchiveWriter &w, const TreeParams &base, std::size_t estimators, double rate, double ratio, double min_change,
                 bool has_seed, std::uint64_t seed) {
    base.save(w);
    w.size(estimators);
    w.f64(rate);
    w.f64(ratio);
    w.f64(min_change);
    w.boolean(has_seed);
    w.u64(seed);
}

void load_common(ArchiveReader &r, TreeParams &base, std::size_t &estimators, double &rate, double &ratio, double &min_change,
                 bool &has_seed, std::uint64_t &seed) {
    base.load(r);
    estimators = r.size();
    rate = r.f64();
    ratio = r.f64();
    min_change = r.f64();
    has_seed = r.boolean();
    seed = r.u64();
}

}  // namespace

// ---- AdaBoost ---------------------------------------------------------------------------

void AdaBoost::fit(const Matrix &X, const std::vector<int> &y, std::size_t K) {
    require_training_set(X, y.size());
    if (K < 2) {
        throw std::invalid_argument("AdaBoost needs at least two distinct classes");
    }
    const std::size_t n = X.rows;
    Rng rng = make_rng(has_seed, seed);
    const std::vector<double> targets(y.begin(), y.end());
    const std::vector<std::size_t> pool = all_rows(n);
    std::vector<double> w(n, 1.0 / static_cast<double>(n));

    AdaBoost next = *this;
    next.trees.clear();
    next.alphas.clear();
    next.losses.clear();
    for (std::size_t t = 0; t < estimators; ++t) {
        DecisionTree tree;
        tree.params = base;
        tree.classification = true;
        tree.classes = K;
        tree.fit(X, targets, w, subsample(pool, ratio, rng), rng);

        std::vector<bool> miss(n);
        double err = 0.0, total = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            miss[i] = static_cast<int>(argmax(tree.leaf_for(X.row(i)), K)) != y[i];
            err += miss[i] ? w[i] : 0.0;
            total += w[i];
        }
        err /= total;
        if (err >= 1.0 - 1.0 / static_cast<double>(K)) {
            if (next.trees.empty()) {
                next.trees.push_back(std::move(tree));  // no better than chance; keep one so the model is usable
                next.alphas.push_back(1.0);
                next.losses.push_back(err);
            }
            break;
        }
        const double alpha = rate * (std::log((1.0 - err) / std::max(err, 1e-10)) + std::log(static_cast<double>(K) - 1.0));
        next.trees.push_back(std::move(tree));
        next.alphas.push_back(alpha);
        next.losses.push_back(err);
        if (err <= 0.0) {
            break;
        }
        double sum = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            if (miss[i]) {
                w[i] *= std::exp(alpha);
            }
            sum += w[i];
        }
        for (double &v : w) {
            v /= sum;
        }
        if (t > 0 && std::fabs(next.losses[t] - next.losses[t - 1]) < min_change) {
            break;
        }
    }
    next.classes = K;
    next.features = X.cols;
    *this = std::move(next);
}

void AdaBoost::probabilities(const double *x, double *out) const {
    std::fill(out, out + classes, 0.0);
    double total = 0.0;
    for (std::size_t t = 0; t < trees.size(); ++t) {
        out[argmax(trees[t].leaf_for(x), classes)] += alphas[t];
        total += alphas[t];
    }
    for (std::size_t k = 0; k < classes; ++k) {
        out[k] = total > 0.0 ? out[k] / total : 1.0 / static_cast<double>(classes);
    }
}

void AdaBoost::save(ArchiveWriter &w) const {
    save_common(w, base, estimators, rate, ratio, min_change, has_seed, seed);
    w.size(classes);
    w.size(features);
    save_trees(w, trees);
    w.doubles(alphas);
    w.doubles(losses);
}

void AdaBoost::load(ArchiveReader &r) {
    load_common(r, base, estimators, rate, ratio, min_change, has_seed, seed);
    classes = r.size(1u << 20);
    features = r.size(1u << 26);
    trees = load_trees(r, true, classes, features);
    alphas = r.doubles(trees.size());
    losses = r.doubles();
}

// ---- LogitBoost -----------------------------------------------------------------------------

void LogitBoost::fit(const Matrix &X, const std::vector<int> &y) {
    require_training_set(X, y.size());
    const std::size_t n = X.rows;
    Rng rng = make_rng(has_seed, seed);
    const std::vector<std::size_t> pool = all_rows(n);
    std::vector<double> F(n, 0.0), p(n, 0.5), z(n), w(n);

    LogitBoost next = *this;
    next.trees.clear();
    next.losses.clear();
    for (std::size_t t = 0; t < estimators; ++t) {
        for (std::size_t i = 0; i < n; ++i) {
            const double var = std::max(p[i] * (1.0 - p[i]), 1e-10);
            z[i] = std::min(4.0, std::max(-4.0, (static_cast<double>(y[i]) - p[i]) / var));
            w[i] = var;
        }
        DecisionTree tree;
        tree.params = base;
        tree.classification = false;
        tree.classes = 1;
        tree.fit(X, z, w, subsample(pool, ratio, rng), rng);
        double loss = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            F[i] += rate * 0.5 * tree.value(X.row(i));
            p[i] = 1.0 / (1.0 + std::exp(-2.0 * F[i]));
            const double q = std::min(std::max(p[i], 1e-15), 1.0 - 1e-15);
            loss -= y[i] ? std::log(q) : std::log(1.0 - q);
        }
        loss /= static_cast<double>(n);
        next.trees.push_back(std::move(tree));
        next.losses.push_back(loss);
        if (t > 0 && next.losses[t - 1] - loss < min_change) {
            break;
        }
    }
    next.features = X.cols;
    *this = std::move(next);
}

double LogitBoost::positive_probability(const double *x) const {
    double F = 0.0;
    for (const DecisionTree &tree : trees) {
        F += rate * 0.5 * tree.value(x);
    }
    return 1.0 / (1.0 + std::exp(-2.0 * F));
}

void LogitBoost::save(ArchiveWriter &w) const {
    save_common(w, base, estimators, rate, ratio, min_change, has_seed, seed);
    w.size(features);
    save_trees(w, trees);
    w.doubles(losses);
}

void LogitBoost::load(ArchiveReader &r) {
    load_common(r, base, estimators, rate, ratio, min_change, has_seed, seed);
    features = r.size(1u << 26);
    trees = load_trees(r, false, 1, features);
    losses = r.doubles();
}

// ---- GradientBoost ---------------------------------------------------------------------------

void GradientBoost::fit(const Matrix &X, const std::vector<double> &y) {
    require_training_set(X, y.size());
    const std::size_t n = X.rows;
    Rng rng = make_rng(has_seed, seed);
    std::vector<std::size_t> order = all_rows(n);
    std::shuffle(order.begin(), order.end(), rng);
    std::size_t n_val = hold_out > 0.0 ? static_cast<std::size_t>(std::llround(hold_out * static_cast<double>(n))) : 0;
    if (n_val >= n) {
        n_val = 0;
    }
    const std::vector<std::size_t> val(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(n_val));
    const std::vector<std::size_t> train(order.begin() + static_cast<std::ptrdiff_t>(n_val), order.end());

    GradientBoost next = *this;
    next.trees.clear();
    next.losses.clear();
    next.validation.clear();
    next.initial = 0.0;
    for (std::size_t i : train) {
        next.initial += y[i];
    }
    next.initial /= static_cast<double>(train.size());
    std::vector<double> F(n, next.initial), residual(n, 0.0);

    double best = std::numeric_limits<double>::infinity();
    std::size_t best_round = 0;
    for (std::size_t t = 0; t < estimators; ++t) {
        for (std::size_t i : train) {
            residual[i] = y[i] - F[i];
        }
        DecisionTree tree;
        tree.params = base;
        tree.classification = false;
        tree.classes = 1;
        tree.fit(X, residual, {}, subsample(train, ratio, rng), rng);

        double train_mse = 0.0, val_mse = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            F[i] += rate * tree.value(X.row(i));
        }
        for (std::size_t i : train) {
            train_mse += (y[i] - F[i]) * (y[i] - F[i]);
        }
        train_mse /= static_cast<double>(train.size());
        next.trees.push_back(std::move(tree));
        next.losses.push_back(train_mse);

        if (!val.empty()) {
            for (std::size_t i : val) {
                val_mse += (y[i] - F[i]) * (y[i] - F[i]);
            }
            val_mse /= static_cast<double>(val.size());
            next.validation.push_back(val_mse);
            if (val_mse < best - min_change) {
                best = val_mse;
                best_round = t;
            } else if (t - best_round >= window) {
                break;
            }
        } else if (t > 0 && next.losses[t - 1] - train_mse < min_change) {
            break;
        }
    }
    if (!val.empty()) {
        next.trees.resize(best_round + 1);  // keep the rounds up to the best validation score
    }
    next.features = X.cols;
    *this = std::move(next);
}

double GradientBoost::predict(const double *x) const {
    double F = initial;
    for (const DecisionTree &tree : trees) {
        F += rate * tree.value(x);
    }
    return F;
}

void GradientBoost::save(ArchiveWriter &w) const {
    save_common(w, base, estimators, rate, ratio, min_change, has_seed, seed);
    w.size(window);
    w.f64(hold_out);
    w.size(features);
    w.f64(initial);
    save_trees(w, trees);
    w.doubles(losses);
    w.doubles(validation);
}

void GradientBoost::load(ArchiveReader &r) {
    load_common(r, base, estimators, rate, ratio, min_change, has_seed, seed);
    window = r.size();
    hold_out = r.f64();
    features = r.size(1u << 26);
    initial = r.f64();
    trees = load_trees(r, false, 1, features);
    losses = r.doubles();
    validation = r.doubles();
}

// ---- ElasticNet ----------------------------------------------------------------------------------

void ElasticNet::fit(const Matrix &X, const std::vector<double> &y) {
    require_training_set(X, y.size());
    const std::size_t n = X.rows, d = X.cols;
    const double nn = static_cast<double>(n);
    std::vector<double> mean(d, 0.0);
    double y_mean = 0.0;
    for (std::size_t r = 0; r < n; ++r) {
        for (std::size_t j = 0; j < d; ++j) {
            mean[j] += X(r, j) / nn;
        }
        y_mean += y[r] / nn;
    }
    // Column-major centered copy for fast coordinate updates.
    std::vector<double> xc(n * d);
    std::vector<double> norm(d, 0.0);
    for (std::size_t j = 0; j < d; ++j) {
        for (std::size_t r = 0; r < n; ++r) {
            const double v = X(r, j) - mean[j];
            xc[j * n + r] = v;
            norm[j] += v * v / nn;
        }
    }
    std::vector<double> residual(n), w(d, 0.0);
    for (std::size_t r = 0; r < n; ++r) {
        residual[r] = y[r] - y_mean;
    }
    const double l1 = alpha * l1_ratio, l2 = alpha * (1.0 - l1_ratio);
    long it = 0;
    for (; it < max_iterations; ++it) {
        double max_delta = 0.0, max_w = 0.0;
        for (std::size_t j = 0; j < d; ++j) {
            const double *col = &xc[j * n];
            double rho = 0.0;
            for (std::size_t r = 0; r < n; ++r) {
                rho += col[r] * residual[r];
            }
            rho = rho / nn + norm[j] * w[j];
            const double soft = rho > l1 ? rho - l1 : (rho < -l1 ? rho + l1 : 0.0);
            const double updated = norm[j] + l2 > 0.0 ? soft / (norm[j] + l2) : 0.0;
            const double delta = updated - w[j];
            if (delta != 0.0) {
                for (std::size_t r = 0; r < n; ++r) {
                    residual[r] -= col[r] * delta;
                }
                w[j] = updated;
            }
            max_delta = std::max(max_delta, std::fabs(delta));
            max_w = std::max(max_w, std::fabs(updated));
        }
        if (max_delta <= tolerance * std::max(max_w, 1e-12)) {
            ++it;
            break;
        }
    }
    intercept = y_mean - dot(mean.data(), w.data(), d);
    coefficients = std::move(w);
    iterations = it;
}

void ElasticNet::save(ArchiveWriter &w) const {
    w.f64(alpha);
    w.f64(l1_ratio);
    w.i64(max_iterations);
    w.f64(tolerance);
    w.doubles(coefficients);
    w.f64(intercept);
    w.i64(iterations);
}

void ElasticNet::load(ArchiveReader &r) {
    alpha = r.f64();
    l1_ratio = r.f64();
    max_iterations = static_cast<long>(r.i64());
    tolerance = r.f64();
    coefficients = r.doubles();
    intercept = r.f64();
    iterations = static_cast<long>(r.i64());
}

}  // namespace mlcore
