#include "tree.hpp"

#include <numeric>

namespace mlcore {

void TreeParams::save(ArchiveWriter &w) const {
    w.size(max_height);
    w.size(max_leaf_size);
    w.f64(min_purity_increase);
    w.size(max_features);
    w.boolean(extra);
}

void TreeParams::load(ArchiveReader &r) {
    max_height = r.size();
    max_leaf_size = r.size();
    min_purity_increase = r.f64();
    max_features = r.size();
    extra = r.boolean();
}

namespace {

// Running statistics of one side of a split.
struct Stats {
    std::vector<double> counts;  // classification: weight per class
    double weight = 0.0;
    double sum = 0.0;     // regression
    double sumsq = 0.0;   // regression: sum of w*y^2; classification: sum of counts^2

    void reset(std::size_t classes) {
        counts.assign(classes, 0.0);
        weight = sum = sumsq = 0.0;
    }

    void add(bool classification, double target, double w) {
        weight += w;
        if (classification) {
            double &c = counts[static_cast<std::size_t>(target)];
            sumsq += (c + w) * (c + w) - c * c;
            c += w;
        } else {
            sum += w * target;
            sumsq += w * target * target;
        }
    }

    void remove(bool classification, double target, double w) {
        weight -= w;
        if (classification) {
            double &c = counts[static_cast<std::size_t>(target)];
            sumsq += (c - w) * (c - w) - c * c;
            c -= w;
        } else {
            sum -= w * target;
            sumsq -= w * target * target;
        }
    }

    // Impurity times weight (Gini * W, or sum of squared errors).
    double weighted_impurity(bool classification) const {
        if (weight <= 0.0) {
            return 0.0;
        }
        if (classification) {
            return weight - sumsq / weight;
        }
        return std::max(0.0, sumsq - sum * sum / weight);
    }
};

struct Task {
    std::int32_t node;
    std::size_t begin;
    std::size_t end;
    std::size_t depth;
};

}  // namespace

void DecisionTree::fit(const Matrix &X, const std::vector<double> &targets, const std::vector<double> &weights,
                       std::vector<std::size_t> rows, Rng &rng) {
    require_training_set(X, targets.size());
    if (rows.empty()) {
        throw std::invalid_argument("Cannot fit a tree on zero samples");
    }
    if (classification && classes == 0) {
        throw std::invalid_argument("Classification tree needs at least one class");
    }
    const std::size_t d = X.cols;
    const std::size_t width = classification ? classes : 1;
    const bool weighted = !weights.empty();
    auto weight_of = [&](std::size_t row) { return weighted ? weights[row] : 1.0; };

    features = d;
    nodes.clear();
    leaf_values.clear();
    importances.assign(d, 0.0);
    height = 0;

    std::vector<std::size_t> feature_pool(d);
    std::iota(feature_pool.begin(), feature_pool.end(), 0);
    const std::size_t try_features = params.max_features == 0 ? d : std::min(params.max_features, d);

    std::vector<std::pair<double, std::size_t>> sorted;
    Stats parent, left, right;

    // Exhaustive splits sort each feature once here, ordered by (value, row) like the
    // per-node sort, and keep every node's rows as one contiguous range per feature: a
    // split stable-partitions each range, so children stay sorted in O(n) instead of
    // O(n log n). Skipped when that index would be too large.
    const std::size_t total = rows.size();
    const bool presorted = !params.extra && X.rows <= std::numeric_limits<std::uint32_t>::max() &&
                           d <= (std::size_t{1} << 24) / total;
    std::vector<std::uint32_t> order;
    std::vector<std::uint32_t> scratch;
    std::vector<unsigned char> goes_left;
    if (presorted) {
        order.resize(d * total);
        scratch.resize(total);
        goes_left.resize(X.rows);
        for (std::size_t f = 0; f < d; ++f) {
            // Sorting contiguous pairs is much faster than sorting row ids through X.
            sorted.clear();
            for (const std::size_t row : rows) {
                sorted.emplace_back(X(row, f), row);
            }
            std::sort(sorted.begin(), sorted.end());
            std::uint32_t *block = &order[f * total];
            for (std::size_t i = 0; i < total; ++i) {
                block[i] = static_cast<std::uint32_t>(sorted[i].second);
            }
        }
    }

    auto make_leaf = [&](Node &node, const Stats &stats) {
        node.feature = -1;
        node.leaf = static_cast<std::uint32_t>(leaf_values.size() / width);
        if (classification) {
            for (std::size_t k = 0; k < width; ++k) {
                leaf_values.push_back(stats.weight > 0 ? stats.counts[k] / stats.weight : 1.0 / static_cast<double>(width));
            }
        } else {
            leaf_values.push_back(stats.weight > 0 ? stats.sum / stats.weight : 0.0);
        }
    };

    nodes.emplace_back();
    std::vector<Task> stack{{0, 0, rows.size(), 0}};
    while (!stack.empty()) {
        const Task task = stack.back();
        stack.pop_back();
        const std::size_t count = task.end - task.begin;
        height = std::max(height, task.depth);

        parent.reset(width);
        for (std::size_t i = task.begin; i < task.end; ++i) {
            parent.add(classification, targets[rows[i]], weight_of(rows[i]));
        }
        nodes[task.node].weight = parent.weight;
        const double parent_impurity = parent.weighted_impurity(classification);

        if (task.depth >= params.max_height || count <= params.max_leaf_size || parent_impurity <= 1e-12 ||
            parent.weight <= 0.0) {
            make_leaf(nodes[task.node], parent);
            continue;
        }

        // Partial Fisher-Yates picks `try_features` distinct candidates.
        for (std::size_t i = 0; i < try_features; ++i) {
            const std::size_t j = i + std::uniform_int_distribution<std::size_t>(0, d - 1 - i)(rng);
            std::swap(feature_pool[i], feature_pool[j]);
        }

        double best_gain = 0.0;
        std::int32_t best_feature = -1;
        double best_threshold = 0.0;

        for (std::size_t fi = 0; fi < try_features; ++fi) {
            const std::size_t f = feature_pool[fi];
            if (params.extra) {
                double lo = std::numeric_limits<double>::infinity();
                double hi = -lo;
                for (std::size_t i = task.begin; i < task.end; ++i) {
                    const double v = X(rows[i], f);
                    lo = std::min(lo, v);
                    hi = std::max(hi, v);
                }
                if (!(hi > lo)) {
                    continue;
                }
                const double threshold = std::uniform_real_distribution<double>(lo, hi)(rng);
                left.reset(width);
                right.reset(width);
                for (std::size_t i = task.begin; i < task.end; ++i) {
                    const std::size_t row = rows[i];
                    (X(row, f) < threshold ? left : right).add(classification, targets[row], weight_of(row));
                }
                if (left.weight <= 0.0 || right.weight <= 0.0) {
                    continue;
                }
                const double gain = parent_impurity - left.weighted_impurity(classification) -
                                    right.weighted_impurity(classification);
                if (gain > best_gain) {
                    best_gain = gain;
                    best_feature = static_cast<std::int32_t>(f);
                    best_threshold = threshold;
                }
                continue;
            }

            if (presorted) {
                const std::uint32_t *block = &order[f * total];
                if (!(X(block[task.end - 1], f) > X(block[task.begin], f))) {
                    continue;
                }
                left.reset(width);
                right = parent;
                for (std::size_t i = task.begin; i + 1 < task.end; ++i) {
                    const std::size_t row = block[i];
                    const double w = weight_of(row);
                    left.add(classification, targets[row], w);
                    right.remove(classification, targets[row], w);
                    const double value = X(row, f);
                    const double following = X(block[i + 1], f);
                    if (value == following) {
                        continue;
                    }
                    const double gain = parent_impurity - left.weighted_impurity(classification) -
                                        right.weighted_impurity(classification);
                    if (gain > best_gain) {
                        best_gain = gain;
                        best_feature = static_cast<std::int32_t>(f);
                        best_threshold = value + (following - value) / 2.0;
                        if (!(best_threshold > value)) {
                            best_threshold = following;  // adjacent doubles
                        }
                    }
                }
                continue;
            }

            sorted.clear();
            for (std::size_t i = task.begin; i < task.end; ++i) {
                sorted.emplace_back(X(rows[i], f), rows[i]);
            }
            std::sort(sorted.begin(), sorted.end());
            if (!(sorted.back().first > sorted.front().first)) {
                continue;
            }
            left.reset(width);
            right = parent;
            for (std::size_t i = 0; i + 1 < sorted.size(); ++i) {
                const std::size_t row = sorted[i].second;
                const double w = weight_of(row);
                left.add(classification, targets[row], w);
                right.remove(classification, targets[row], w);
                if (sorted[i].first == sorted[i + 1].first) {
                    continue;
                }
                const double gain = parent_impurity - left.weighted_impurity(classification) -
                                    right.weighted_impurity(classification);
                if (gain > best_gain) {
                    best_gain = gain;
                    best_feature = static_cast<std::int32_t>(f);
                    best_threshold = sorted[i].first + (sorted[i + 1].first - sorted[i].first) / 2.0;
                    if (!(best_threshold > sorted[i].first)) {
                        best_threshold = sorted[i + 1].first;  // adjacent doubles
                    }
                }
            }
        }

        // Purity increase is measured per unit of weight at this node.
        if (best_feature < 0 || best_gain / parent.weight < params.min_purity_increase) {
            make_leaf(nodes[task.node], parent);
            continue;
        }

        const auto middle = std::partition(rows.begin() + static_cast<std::ptrdiff_t>(task.begin),
                                           rows.begin() + static_cast<std::ptrdiff_t>(task.end),
                                           [&](std::size_t row) { return X(row, static_cast<std::size_t>(best_feature)) < best_threshold; });
        const std::size_t split = static_cast<std::size_t>(middle - rows.begin());
        if (split == task.begin || split == task.end) {
            make_leaf(nodes[task.node], parent);
            continue;
        }

        // Children that are certain to become leaves never read their sorted ranges.
        const auto leaf_child = [&](std::size_t n) { return task.depth + 1 >= params.max_height || n <= params.max_leaf_size; };
        if (presorted && !(leaf_child(split - task.begin) && leaf_child(task.end - split))) {
            for (std::size_t i = task.begin; i < task.end; ++i) {
                goes_left[rows[i]] = i < split;
            }
            for (std::size_t f = 0; f < d; ++f) {
                std::uint32_t *out = &order[f * total + task.begin];
                std::size_t moved = 0;
                for (std::size_t i = task.begin; i < task.end; ++i) {
                    const std::uint32_t row = order[f * total + i];
                    if (goes_left[row]) {
                        *out++ = row;
                    } else {
                        scratch[moved++] = row;
                    }
                }
                std::copy(scratch.begin(), scratch.begin() + static_cast<std::ptrdiff_t>(moved), out);
            }
        }

        importances[static_cast<std::size_t>(best_feature)] += best_gain;
        const std::int32_t left_index = static_cast<std::int32_t>(nodes.size());
        nodes.emplace_back();
        nodes.emplace_back();
        Node &node = nodes[task.node];
        node.feature = best_feature;
        node.threshold = best_threshold;
        node.left = left_index;
        node.right = left_index + 1;
        stack.push_back({left_index + 1, split, task.end, task.depth + 1});
        stack.push_back({left_index, task.begin, split, task.depth + 1});
    }
}

const double *DecisionTree::leaf_for(const double *x) const {
    std::size_t i = 0;
    while (nodes[i].feature >= 0) {
        const Node &node = nodes[i];
        i = static_cast<std::size_t>(x[node.feature] < node.threshold ? node.left : node.right);
    }
    return &leaf_values[nodes[i].leaf * width()];
}

std::vector<double> DecisionTree::normalized_importances() const {
    std::vector<double> out = importances;
    const double total = std::accumulate(out.begin(), out.end(), 0.0);
    if (total > 0.0) {
        for (double &v : out) {
            v /= total;
        }
    }
    return out;
}

void DecisionTree::save(ArchiveWriter &w) const {
    params.save(w);
    w.boolean(classification);
    w.size(classes);
    w.size(features);
    w.size(height);
    w.size(nodes.size());
    for (const Node &n : nodes) {
        w.i64(n.feature);
        w.f64(n.threshold);
        w.i64(n.left);
        w.i64(n.right);
        w.u32(n.leaf);
        w.f64(n.weight);
    }
    w.doubles(leaf_values);
    w.doubles(importances);
}

void DecisionTree::load(ArchiveReader &r) {
    params.load(r);
    classification = r.boolean();
    classes = r.size(1u << 20);
    features = r.size(1u << 26);
    height = r.size();
    const std::size_t n = r.count(44);
    nodes.assign(n, Node{});
    for (std::size_t i = 0; i < n; ++i) {
        Node &node = nodes[i];
        const std::int64_t feature = r.i64();
        node.threshold = r.f64();
        const std::int64_t left = r.i64();
        const std::int64_t right = r.i64();
        node.leaf = r.u32();
        node.weight = r.f64();
        ArchiveReader::require(feature >= -1 && feature < static_cast<std::int64_t>(features), "bad tree feature");
        node.feature = static_cast<std::int32_t>(feature);
        if (feature >= 0) {
            // Children always come after their parent, which also rules out cycles.
            ArchiveReader::require(left > static_cast<std::int64_t>(i) && left < static_cast<std::int64_t>(n) &&
                                       right > static_cast<std::int64_t>(i) && right < static_cast<std::int64_t>(n),
                                   "bad tree child");
            node.left = static_cast<std::int32_t>(left);
            node.right = static_cast<std::int32_t>(right);
        }
    }
    leaf_values = r.doubles();
    importances = r.doubles(features);
    ArchiveReader::require(classification ? classes > 0 : true, "bad class count");
    const std::size_t w = width();
    ArchiveReader::require(leaf_values.size() % w == 0, "bad leaf values");
    for (const Node &node : nodes) {
        if (node.feature < 0) {
            ArchiveReader::require(node.leaf < leaf_values.size() / w, "bad leaf index");
        }
    }
    ArchiveReader::require(!nodes.empty() && features > 0, "empty tree");
}

// ---- Forest -------------------------------------------------------------------

void Forest::fit(const Matrix &X, const std::vector<double> &targets, std::size_t n_classes, unsigned threads) {
    require_training_set(X, targets.size());
    if (estimators == 0) {
        throw std::invalid_argument("estimators must be at least 1");
    }
    const std::size_t n = X.rows;
    const std::size_t sample_size = std::max<std::size_t>(1, static_cast<std::size_t>(std::llround(ratio * static_cast<double>(n))));

    TreeParams tree_params = params;
    if (tree_params.max_features == 0) {
        const double d = static_cast<double>(X.cols);
        tree_params.max_features = static_cast<std::size_t>(std::max(1.0, classification ? std::round(std::sqrt(d)) : std::round(d / 3.0)));
    }

    // Balanced sampling: each sample's probability is inversely proportional to its class size.
    std::vector<double> cumulative;
    if (balanced && classification) {
        std::vector<double> class_size(n_classes, 0.0);
        for (double t : targets) {
            class_size[static_cast<std::size_t>(t)] += 1.0;
        }
        cumulative.resize(n);
        double acc = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            acc += 1.0 / class_size[static_cast<std::size_t>(targets[i])];
            cumulative[i] = acc;
        }
    }

    Rng master = make_rng(has_seed, seed);
    std::vector<std::uint64_t> seeds(estimators);
    for (std::uint64_t &s : seeds) {
        s = master();
    }

    std::vector<DecisionTree> grown(estimators);
    const std::vector<double> no_weights;
    parallel_for(estimators, threads, [&](std::size_t t) {
        Rng rng(seeds[t]);
        std::vector<std::size_t> rows(sample_size);
        if (cumulative.empty()) {
            std::uniform_int_distribution<std::size_t> pick(0, n - 1);
            for (std::size_t &row : rows) {
                row = pick(rng);
            }
        } else {
            std::uniform_real_distribution<double> pick(0.0, cumulative.back());
            for (std::size_t &row : rows) {
                const double u = pick(rng);
                row = std::min<std::size_t>(n - 1, static_cast<std::size_t>(std::upper_bound(cumulative.begin(), cumulative.end(), u) - cumulative.begin()));
            }
        }
        DecisionTree &tree = grown[t];
        tree.params = tree_params;
        tree.classification = classification;
        tree.classes = classification ? n_classes : 1;
        tree.fit(X, targets, no_weights, std::move(rows), rng);
    });

    trees = std::move(grown);
    classes = classification ? n_classes : 1;
    features = X.cols;
}

void Forest::probabilities(const double *x, double *out) const {
    std::fill(out, out + classes, 0.0);
    for (const DecisionTree &tree : trees) {
        const double *leaf = tree.leaf_for(x);
        for (std::size_t k = 0; k < classes; ++k) {
            out[k] += leaf[k];
        }
    }
    for (std::size_t k = 0; k < classes; ++k) {
        out[k] /= static_cast<double>(trees.size());
    }
}

double Forest::value(const double *x) const {
    double sum = 0.0;
    for (const DecisionTree &tree : trees) {
        sum += tree.value(x);
    }
    return sum / static_cast<double>(trees.size());
}

std::vector<double> Forest::importances() const {
    std::vector<double> total(features, 0.0);
    for (const DecisionTree &tree : trees) {
        const std::vector<double> imp = tree.normalized_importances();
        for (std::size_t f = 0; f < features; ++f) {
            total[f] += imp[f];
        }
    }
    const double sum = std::accumulate(total.begin(), total.end(), 0.0);
    if (sum > 0.0) {
        for (double &v : total) {
            v /= sum;
        }
    }
    return total;
}

void Forest::save(ArchiveWriter &w) const {
    params.save(w);
    w.boolean(classification);
    w.size(estimators);
    w.f64(ratio);
    w.boolean(balanced);
    w.boolean(has_seed);
    w.u64(seed);
    w.size(classes);
    w.size(features);
    w.size(trees.size());
    for (const DecisionTree &tree : trees) {
        tree.save(w);
    }
}

void Forest::load(ArchiveReader &r) {
    params.load(r);
    classification = r.boolean();
    estimators = r.size();
    ratio = r.f64();
    balanced = r.boolean();
    has_seed = r.boolean();
    seed = r.u64();
    classes = r.size(1u << 20);
    features = r.size(1u << 26);
    const std::size_t n = r.count(40);
    trees.assign(n, DecisionTree{});
    for (DecisionTree &tree : trees) {
        tree.load(r);
        ArchiveReader::require(tree.classification == classification && tree.features == features &&
                               tree.width() == (classification ? classes : 1));
    }
}

}  // namespace mlcore
