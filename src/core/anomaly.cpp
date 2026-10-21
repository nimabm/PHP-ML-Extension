#include "anomaly.hpp"

#include <numeric>

namespace mlcore {

namespace {

constexpr double kEuler = 0.5772156649015329;
constexpr double kTwoPi = 6.283185307179586476925286766559;

double contamination_threshold(std::vector<double> scores, double contamination) {
    if (contamination <= 0.0) {
        return *std::max_element(scores.begin(), scores.end());
    }
    return quantile(scores, 1.0 - contamination);
}

// Average path length of an unsuccessful search in a binary search tree of n nodes.
double average_path(double n) {
    if (n <= 1.0) return 0.0;
    if (n == 2.0) return 1.0;
    return 2.0 * (std::log(n - 1.0) + kEuler) - 2.0 * (n - 1.0) / n;
}

}  // namespace

// ---- RobustZScore -------------------------------------------------------------------------

void RobustZScore::fit(const Matrix &X) {
    require_training_set(X, X.rows);
    std::vector<double> med(X.cols), mad(X.cols), column(X.rows);
    for (std::size_t c = 0; c < X.cols; ++c) {
        for (std::size_t r = 0; r < X.rows; ++r) column[r] = X(r, c);
        med[c] = median(column);
        for (std::size_t r = 0; r < X.rows; ++r) column[r] = std::fabs(X(r, c) - med[c]);
        mad[c] = median(column);
    }
    medians = std::move(med);
    mads = std::move(mad);
}

double RobustZScore::score(const double *x) const {
    double max = 0.0, sum = 0.0;
    for (std::size_t c = 0; c < medians.size(); ++c) {
        const double z = 0.6745 * std::fabs(x[c] - medians[c]) / (mads[c] + smoothing);
        max = std::max(max, z);
        sum += z;
    }
    return beta * max + (1.0 - beta) * sum / static_cast<double>(medians.size());
}

void RobustZScore::save(ArchiveWriter &w) const {
    w.f64(threshold);
    w.f64(beta);
    w.f64(smoothing);
    w.doubles(medians);
    w.doubles(mads);
}

void RobustZScore::load(ArchiveReader &r) {
    threshold = r.f64();
    beta = r.f64();
    smoothing = r.f64();
    medians = r.doubles();
    mads = r.doubles(medians.size());
    ArchiveReader::require(smoothing > 0.0, "smoothing must be positive");
}

// ---- GaussianMLE ----------------------------------------------------------------------------

void GaussianMLE::fit(const Matrix &X) {
    require_training_set(X, X.rows);
    const double n = static_cast<double>(X.rows);
    std::vector<double> mu(X.cols, 0.0), var(X.cols, 0.0);
    for (std::size_t r = 0; r < X.rows; ++r) {
        for (std::size_t c = 0; c < X.cols; ++c) mu[c] += X(r, c) / n;
    }
    double max_var = 0.0;
    for (std::size_t r = 0; r < X.rows; ++r) {
        for (std::size_t c = 0; c < X.cols; ++c) var[c] += (X(r, c) - mu[c]) * (X(r, c) - mu[c]) / n;
    }
    for (double v : var) max_var = std::max(max_var, v);
    const double eps = std::max(smoothing * max_var, 1e-12);
    for (double &v : var) v += eps;
    means = std::move(mu);
    variances = std::move(var);
    std::vector<double> scores(X.rows);
    for (std::size_t r = 0; r < X.rows; ++r) scores[r] = score(X.row(r));
    threshold = contamination_threshold(std::move(scores), contamination);
}

double GaussianMLE::score(const double *x) const {
    double s = 0.0;
    for (std::size_t c = 0; c < means.size(); ++c) {
        const double d = x[c] - means[c];
        s += std::log(kTwoPi * variances[c]) + d * d / variances[c];
    }
    return 0.5 * s;
}

void GaussianMLE::save(ArchiveWriter &w) const {
    w.f64(contamination);
    w.f64(smoothing);
    w.doubles(means);
    w.doubles(variances);
    w.f64(threshold);
}

void GaussianMLE::load(ArchiveReader &r) {
    contamination = r.f64();
    smoothing = r.f64();
    means = r.doubles();
    variances = r.doubles(means.size());
    threshold = r.f64();
    for (double v : variances) ArchiveReader::require(v > 0.0, "variances must be positive");
}

// ---- IsolationForest ---------------------------------------------------------------------------

void IsolationForest::fit(const Matrix &X, unsigned threads) {
    require_training_set(X, X.rows);
    const std::size_t n = X.rows, d = X.cols;
    const std::size_t psi = std::min(max_samples, n);
    const std::size_t height_limit = static_cast<std::size_t>(std::ceil(std::log2(std::max<double>(2.0, static_cast<double>(psi)))));
    Rng master = make_rng(has_seed, seed);
    std::vector<std::uint64_t> seeds(estimators);
    for (auto &s : seeds) s = master();

    std::vector<std::vector<Node>> forest(estimators);
    parallel_for(estimators, threads, [&](std::size_t t) {
        Rng rng(seeds[t]);
        std::vector<std::size_t> rows(n);
        std::iota(rows.begin(), rows.end(), 0);
        std::shuffle(rows.begin(), rows.end(), rng);
        rows.resize(psi);

        std::vector<Node> &tree = forest[t];
        struct Task { std::int32_t node; std::size_t begin, end, depth; };
        tree.emplace_back();
        std::vector<Task> stack{{0, 0, rows.size(), 0}};
        std::vector<std::size_t> candidates;
        while (!stack.empty()) {
            const Task task = stack.back();
            stack.pop_back();
            const std::size_t count = task.end - task.begin;
            tree[task.node].size = static_cast<double>(count);
            if (task.depth >= height_limit || count <= 1) {
                continue;
            }
            candidates.clear();
            std::vector<double> lo(d, std::numeric_limits<double>::infinity()), hi(d, -std::numeric_limits<double>::infinity());
            for (std::size_t i = task.begin; i < task.end; ++i) {
                for (std::size_t f = 0; f < d; ++f) {
                    lo[f] = std::min(lo[f], X(rows[i], f));
                    hi[f] = std::max(hi[f], X(rows[i], f));
                }
            }
            for (std::size_t f = 0; f < d; ++f) {
                if (hi[f] > lo[f]) candidates.push_back(f);
            }
            if (candidates.empty()) {
                continue;
            }
            const std::size_t f = candidates[std::uniform_int_distribution<std::size_t>(0, candidates.size() - 1)(rng)];
            const double split = std::uniform_real_distribution<double>(lo[f], hi[f])(rng);
            const auto mid = std::partition(rows.begin() + static_cast<std::ptrdiff_t>(task.begin), rows.begin() + static_cast<std::ptrdiff_t>(task.end),
                                            [&](std::size_t row) { return X(row, f) < split; });
            const std::size_t m = static_cast<std::size_t>(mid - rows.begin());
            if (m == task.begin || m == task.end) {
                continue;
            }
            const std::int32_t left = static_cast<std::int32_t>(tree.size());
            tree.emplace_back();
            tree.emplace_back();
            tree[task.node].feature = static_cast<std::int32_t>(f);
            tree[task.node].split = split;
            tree[task.node].left = left;
            tree[task.node].right = left + 1;
            stack.push_back({left, task.begin, m, task.depth + 1});
            stack.push_back({left + 1, m, task.end, task.depth + 1});
        }
    });

    trees = std::move(forest);
    features_ = d;
    sample_size = psi;
    std::vector<double> scores(n);
    parallel_for(n, threads, [&](std::size_t r) { scores[r] = score(X.row(r)); });
    threshold = contamination < 0.0 ? 0.5 : contamination_threshold(std::move(scores), contamination);
}

double IsolationForest::score(const double *x) const {
    double total = 0.0;
    for (const std::vector<Node> &tree : trees) {
        std::size_t i = 0;
        double depth = 0.0;
        while (tree[i].feature >= 0) {
            i = static_cast<std::size_t>(x[tree[i].feature] < tree[i].split ? tree[i].left : tree[i].right);
            depth += 1.0;
        }
        total += depth + average_path(tree[i].size);
    }
    const double mean = total / static_cast<double>(trees.size());
    return std::pow(2.0, -mean / average_path(static_cast<double>(sample_size)));
}

void IsolationForest::save(ArchiveWriter &w) const {
    w.size(estimators);
    w.size(max_samples);
    w.f64(contamination);
    w.boolean(has_seed);
    w.u64(seed);
    w.size(features_);
    w.size(sample_size);
    w.f64(threshold);
    w.size(trees.size());
    for (const auto &tree : trees) {
        w.size(tree.size());
        for (const Node &node : tree) {
            w.i64(node.feature);
            w.f64(node.split);
            w.i64(node.left);
            w.i64(node.right);
            w.f64(node.size);
        }
    }
}

void IsolationForest::load(ArchiveReader &r) {
    estimators = r.size();
    max_samples = r.size();
    contamination = r.f64();
    has_seed = r.boolean();
    seed = r.u64();
    features_ = r.size(1u << 26);
    sample_size = r.size();
    threshold = r.f64();
    trees.assign(r.count(8), {});
    for (auto &tree : trees) {
        const std::size_t n = r.count(40);
        tree.assign(n, Node{});
        for (std::size_t i = 0; i < n; ++i) {
            Node &node = tree[i];
            const std::int64_t feature = r.i64();
            node.split = r.f64();
            const std::int64_t left = r.i64(), right = r.i64();
            node.size = r.f64();
            ArchiveReader::require(feature >= -1 && feature < static_cast<std::int64_t>(features_), "bad feature");
            node.feature = static_cast<std::int32_t>(feature);
            if (feature >= 0) {
                ArchiveReader::require(left > static_cast<std::int64_t>(i) && left < static_cast<std::int64_t>(n) &&
                                       right > static_cast<std::int64_t>(i) && right < static_cast<std::int64_t>(n), "bad child");
                node.left = static_cast<std::int32_t>(left);
                node.right = static_cast<std::int32_t>(right);
            }
        }
        ArchiveReader::require(!tree.empty(), "empty tree");
    }
    ArchiveReader::require(trees.empty() || sample_size >= 1, "bad sample size");
}

// ---- LocalOutlierFactor -----------------------------------------------------------------------------

double LocalOutlierFactor::factor(const std::vector<Neighbor> &nb) const {
    double reach = 0.0, neighbor_lrd = 0.0;
    for (const Neighbor &o : nb) {
        reach += std::max(k_distance[o.second], o.first);
        neighbor_lrd += lrd[o.second];
    }
    const double own = 1.0 / (reach / static_cast<double>(nb.size()) + 1e-10);
    return neighbor_lrd / static_cast<double>(nb.size()) / own;
}

void LocalOutlierFactor::fit(const Matrix &X) {
    require_training_set(X, X.rows);
    if (k >= X.rows) {
        throw std::invalid_argument("k (" + std::to_string(k) + ") must be less than the number of training samples (" +
                                    std::to_string(X.rows) + ")");
    }
    LocalOutlierFactor next = *this;
    next.index.use_tree = next.index.metric.coordinate_bounded();
    next.index.build(X);
    const std::size_t n = X.rows;
    std::vector<std::vector<Neighbor>> neighbors(n);
    next.k_distance.assign(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        std::vector<Neighbor> nb;
        next.index.nearest(X.row(i), k + 1, nb);
        // Drop the sample itself (or one exact duplicate standing in for it).
        auto self = std::find_if(nb.begin(), nb.end(), [&](const Neighbor &x) { return x.second == i; });
        nb.erase(self != nb.end() ? self : nb.begin());
        nb.resize(std::min(nb.size(), k));
        next.k_distance[i] = nb.back().first;
        neighbors[i] = std::move(nb);
    }
    next.lrd.assign(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        double reach = 0.0;
        for (const Neighbor &o : neighbors[i]) {
            reach += std::max(next.k_distance[o.second], o.first);
        }
        next.lrd[i] = 1.0 / (reach / static_cast<double>(neighbors[i].size()) + 1e-10);
    }
    std::vector<double> scores(n);
    for (std::size_t i = 0; i < n; ++i) {
        scores[i] = next.factor(neighbors[i]);
    }
    next.threshold = contamination_threshold(std::move(scores), contamination);
    *this = std::move(next);
}

double LocalOutlierFactor::score(const double *x) const {
    std::vector<Neighbor> nb;
    index.nearest(x, k, nb);
    return factor(nb);
}

void LocalOutlierFactor::save(ArchiveWriter &w) const {
    w.size(k);
    w.f64(contamination);
    index.save(w);
    w.doubles(k_distance);
    w.doubles(lrd);
    w.f64(threshold);
}

void LocalOutlierFactor::load(ArchiveReader &r) {
    k = r.size();
    contamination = r.f64();
    index.load(r);
    k_distance = r.doubles(index.samples.rows);
    lrd = r.doubles(index.samples.rows);
    threshold = r.f64();
    ArchiveReader::require(k >= 1 && (index.samples.rows == 0 || k < index.samples.rows), "bad k");
}

// ---- Loda ---------------------------------------------------------------------------------------------

void Loda::fit(const Matrix &X) {
    require_training_set(X, X.rows);
    const std::size_t n = X.rows, d = X.cols;
    Rng rng = make_rng(has_seed, seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    const std::size_t nonzero = std::max<std::size_t>(1, static_cast<std::size_t>(std::ceil(std::sqrt(static_cast<double>(d)))));
    const std::size_t B = bins > 0 ? bins : static_cast<std::size_t>(std::ceil(std::log2(static_cast<double>(n)))) + 1;

    Loda next = *this;
    next.projections = Matrix(estimators, d);
    next.lows.assign(estimators, 0.0);
    next.widths.assign(estimators, 1.0);
    next.densities.assign(estimators * B, 0.0);
    next.bin_count = B;
    std::vector<std::size_t> features(d);
    std::iota(features.begin(), features.end(), 0);
    std::vector<double> z(n);
    for (std::size_t e = 0; e < estimators; ++e) {
        std::shuffle(features.begin(), features.end(), rng);
        for (std::size_t i = 0; i < nonzero && i < d; ++i) {
            next.projections(e, features[i]) = normal(rng);
        }
        double lo = std::numeric_limits<double>::infinity(), hi = -lo;
        for (std::size_t r = 0; r < n; ++r) {
            z[r] = dot(next.projections.row(e), X.row(r), d);
            lo = std::min(lo, z[r]);
            hi = std::max(hi, z[r]);
        }
        const double width = hi > lo ? (hi - lo) / static_cast<double>(B) : 1.0;
        next.lows[e] = lo;
        next.widths[e] = width;
        std::vector<double> counts(B, 0.0);
        for (double v : z) {
            counts[std::min(B - 1, static_cast<std::size_t>((v - lo) / width))] += 1.0;
        }
        for (std::size_t b = 0; b < B; ++b) {
            // Laplace smoothing keeps empty bins finite.
            next.densities[e * B + b] = std::log((counts[b] + 1.0) / ((static_cast<double>(n) + static_cast<double>(B)) * width));
        }
    }
    next.features_ = d;
    next.samples_seen = static_cast<double>(n);
    std::vector<double> scores(n);
    for (std::size_t r = 0; r < n; ++r) scores[r] = next.score(X.row(r));
    next.threshold = contamination_threshold(std::move(scores), contamination);
    *this = std::move(next);
}

double Loda::score(const double *x) const {
    const std::size_t E = projections.rows;
    double total = 0.0;
    for (std::size_t e = 0; e < E; ++e) {
        const double z = dot(projections.row(e), x, features_);
        const double pos = (z - lows[e]) / widths[e];
        double density;
        if (pos < 0.0 || pos >= static_cast<double>(bin_count) + 1e-9) {
            // Outside the training range: the density of an empty (smoothed) bin.
            density = std::log(1.0 / ((samples_seen + static_cast<double>(bin_count)) * widths[e]));
        } else {
            density = densities[e * bin_count + std::min(bin_count - 1, static_cast<std::size_t>(pos))];
        }
        total -= density;
    }
    return total / static_cast<double>(E);
}

void Loda::save(ArchiveWriter &w) const {
    w.size(estimators);
    w.size(bins);
    w.f64(contamination);
    w.boolean(has_seed);
    w.u64(seed);
    w.size(features_);
    w.matrix(projections);
    w.doubles(lows);
    w.doubles(widths);
    w.size(bin_count);
    w.doubles(densities);
    w.f64(threshold);
    w.f64(samples_seen);
}

void Loda::load(ArchiveReader &r) {
    estimators = r.size();
    bins = r.size();
    contamination = r.f64();
    has_seed = r.boolean();
    seed = r.u64();
    features_ = r.size(1u << 26);
    projections = r.matrix();
    lows = r.doubles(projections.rows);
    widths = r.doubles(projections.rows);
    bin_count = r.size(1u << 20);
    densities = r.doubles(projections.rows * bin_count);
    threshold = r.f64();
    samples_seen = r.f64();
    ArchiveReader::require(projections.rows == 0 || (projections.cols == features_ && bin_count >= 1), "inconsistent Loda state");
    for (double width : widths) ArchiveReader::require(width > 0.0, "bad bin width");
}

}  // namespace mlcore
