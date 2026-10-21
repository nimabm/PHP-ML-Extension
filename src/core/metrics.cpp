#include "metrics.hpp"

#include <numeric>

namespace mlcore {

std::vector<double> confusion(const std::vector<int> &expected, const std::vector<int> &predicted, std::size_t K) {
    std::vector<double> m(K * K, 0.0);
    for (std::size_t i = 0; i < expected.size(); ++i) {
        m[static_cast<std::size_t>(expected[i]) * K + static_cast<std::size_t>(predicted[i])] += 1.0;
    }
    return m;
}

double accuracy(const std::vector<int> &expected, const std::vector<int> &predicted) {
    std::size_t correct = 0;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        correct += expected[i] == predicted[i];
    }
    return static_cast<double>(correct) / static_cast<double>(expected.size());
}

std::vector<ClassStats> class_stats(const std::vector<double> &m, std::size_t K) {
    double total = 0.0;
    for (double v : m) {
        total += v;
    }
    std::vector<ClassStats> stats(K);
    for (std::size_t k = 0; k < K; ++k) {
        double row = 0.0, col = 0.0;
        for (std::size_t j = 0; j < K; ++j) {
            row += m[k * K + j];
            col += m[j * K + k];
        }
        ClassStats &s = stats[k];
        s.tp = m[k * K + k];
        s.fn = row - s.tp;
        s.fp = col - s.tp;
        s.tn = total - s.tp - s.fn - s.fp;
    }
    return stats;
}

namespace {
template <typename F>
double macro(const std::vector<ClassStats> &stats, F f) {
    double sum = 0.0;
    std::size_t n = 0;
    for (const ClassStats &s : stats) {
        if (s.tp + s.fp + s.fn > 0) {
            sum += f(s);
            ++n;
        }
    }
    return n == 0 ? 0.0 : sum / static_cast<double>(n);
}
}  // namespace

double macro_precision(const std::vector<ClassStats> &stats) {
    return macro(stats, [](const ClassStats &s) { return s.precision(); });
}
double macro_recall(const std::vector<ClassStats> &stats) {
    return macro(stats, [](const ClassStats &s) { return s.recall(); });
}
double macro_f_beta(const std::vector<ClassStats> &stats, double beta) {
    return macro(stats, [beta](const ClassStats &s) { return s.f_beta(beta); });
}
double macro_informedness(const std::vector<ClassStats> &stats) {
    return macro(stats, [](const ClassStats &s) { return s.recall() + s.specificity() - 1.0; });
}

double mcc(const std::vector<double> &m, std::size_t K) {
    double c = 0.0, s = 0.0;
    std::vector<double> t(K, 0.0), p(K, 0.0);
    for (std::size_t i = 0; i < K; ++i) {
        c += m[i * K + i];
        for (std::size_t j = 0; j < K; ++j) {
            t[i] += m[i * K + j];
            p[j] += m[i * K + j];
            s += m[i * K + j];
        }
    }
    double tp = 0.0, tt = 0.0, pp = 0.0;
    for (std::size_t k = 0; k < K; ++k) {
        tp += t[k] * p[k];
        tt += t[k] * t[k];
        pp += p[k] * p[k];
    }
    const double den = std::sqrt((s * s - pp) * (s * s - tt));
    return den == 0.0 ? 0.0 : (c * s - tp) / den;
}

double roc_auc(const std::vector<bool> &positive, const std::vector<double> &scores) {
    const std::size_t n = scores.size();
    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return scores[a] < scores[b]; });
    // Mann-Whitney U with average ranks for ties.
    double rank_sum = 0.0, positives = 0.0;
    std::size_t i = 0;
    while (i < n) {
        std::size_t j = i;
        while (j + 1 < n && scores[order[j + 1]] == scores[order[i]]) {
            ++j;
        }
        const double rank = (static_cast<double>(i) + static_cast<double>(j)) / 2.0 + 1.0;
        for (std::size_t k = i; k <= j; ++k) {
            if (positive[order[k]]) {
                rank_sum += rank;
                positives += 1.0;
            }
        }
        i = j + 1;
    }
    const double negatives = static_cast<double>(n) - positives;
    if (positives == 0.0 || negatives == 0.0) {
        throw std::invalid_argument("ROC AUC needs both positive and negative samples");
    }
    return (rank_sum - positives * (positives + 1.0) / 2.0) / (positives * negatives);
}

double average_precision(const std::vector<bool> &positive, const std::vector<double> &scores) {
    const std::size_t n = scores.size();
    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return scores[a] > scores[b]; });
    double total_pos = 0.0;
    for (bool p : positive) {
        total_pos += p;
    }
    if (total_pos == 0.0) {
        throw std::invalid_argument("Average precision needs at least one positive sample");
    }
    double tp = 0.0, ap = 0.0, prev_recall = 0.0;
    std::size_t i = 0;
    while (i < n) {
        std::size_t j = i;
        while (j < n && scores[order[j]] == scores[order[i]]) {
            tp += positive[order[j]];
            ++j;
        }
        const double precision = tp / static_cast<double>(j);
        const double recall = tp / total_pos;
        ap += (recall - prev_recall) * precision;
        prev_recall = recall;
        i = j;
    }
    return ap;
}

double mean_squared_error(const std::vector<double> &e, const std::vector<double> &p) {
    double s = 0.0;
    for (std::size_t i = 0; i < e.size(); ++i) {
        s += (e[i] - p[i]) * (e[i] - p[i]);
    }
    return s / static_cast<double>(e.size());
}

double mean_absolute_error(const std::vector<double> &e, const std::vector<double> &p) {
    double s = 0.0;
    for (std::size_t i = 0; i < e.size(); ++i) {
        s += std::fabs(e[i] - p[i]);
    }
    return s / static_cast<double>(e.size());
}

double median_absolute_error(const std::vector<double> &e, const std::vector<double> &p) {
    std::vector<double> errors(e.size());
    for (std::size_t i = 0; i < e.size(); ++i) {
        errors[i] = std::fabs(e[i] - p[i]);
    }
    return median(errors);
}

double r2_score(const std::vector<double> &e, const std::vector<double> &p) {
    const double mean = std::accumulate(e.begin(), e.end(), 0.0) / static_cast<double>(e.size());
    double ss_res = 0.0, ss_tot = 0.0;
    for (std::size_t i = 0; i < e.size(); ++i) {
        ss_res += (e[i] - p[i]) * (e[i] - p[i]);
        ss_tot += (e[i] - mean) * (e[i] - mean);
    }
    if (ss_tot == 0.0) {
        return ss_res == 0.0 ? 1.0 : 0.0;
    }
    return 1.0 - ss_res / ss_tot;
}

double smape(const std::vector<double> &e, const std::vector<double> &p) {
    double s = 0.0;
    for (std::size_t i = 0; i < e.size(); ++i) {
        const double den = std::fabs(e[i]) + std::fabs(p[i]);
        if (den > 0.0) {
            s += std::fabs(e[i] - p[i]) / den;
        }
    }
    return 100.0 * s / static_cast<double>(e.size());
}

std::vector<double> contingency(const std::vector<int> &labels, std::size_t L, const std::vector<int> &clusters, std::size_t C) {
    std::vector<double> table(L * C, 0.0);
    for (std::size_t i = 0; i < labels.size(); ++i) {
        table[static_cast<std::size_t>(labels[i]) * C + static_cast<std::size_t>(clusters[i])] += 1.0;
    }
    return table;
}

namespace {

double entropy(const std::vector<double> &counts, double n) {
    double h = 0.0;
    for (double c : counts) {
        if (c > 0) {
            h -= c / n * std::log(c / n);
        }
    }
    return h;
}

// H(A|B) where `table` is A x B.
double conditional_entropy(const std::vector<double> &table, std::size_t A, std::size_t B, double n) {
    double h = 0.0;
    for (std::size_t b = 0; b < B; ++b) {
        double col = 0.0;
        for (std::size_t a = 0; a < A; ++a) {
            col += table[a * B + b];
        }
        for (std::size_t a = 0; a < A; ++a) {
            const double v = table[a * B + b];
            if (v > 0) {
                h -= v / n * std::log(v / col);
            }
        }
    }
    return h;
}

std::vector<double> transpose(const std::vector<double> &t, std::size_t rows, std::size_t cols) {
    std::vector<double> out(rows * cols);
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < cols; ++c) {
            out[c * rows + r] = t[r * cols + c];
        }
    }
    return out;
}

void totals(const std::vector<double> &t, std::size_t L, std::size_t C, std::vector<double> &rows, std::vector<double> &cols,
            double &n) {
    rows.assign(L, 0.0);
    cols.assign(C, 0.0);
    n = 0.0;
    for (std::size_t l = 0; l < L; ++l) {
        for (std::size_t c = 0; c < C; ++c) {
            rows[l] += t[l * C + c];
            cols[c] += t[l * C + c];
            n += t[l * C + c];
        }
    }
}

}  // namespace

double homogeneity(const std::vector<double> &t, std::size_t L, std::size_t C) {
    std::vector<double> rows, cols;
    double n;
    totals(t, L, C, rows, cols, n);
    const double h_labels = entropy(rows, n);
    return h_labels == 0.0 ? 1.0 : 1.0 - conditional_entropy(t, L, C, n) / h_labels;
}

double completeness(const std::vector<double> &t, std::size_t L, std::size_t C) {
    std::vector<double> rows, cols;
    double n;
    totals(t, L, C, rows, cols, n);
    const double h_clusters = entropy(cols, n);
    return h_clusters == 0.0 ? 1.0 : 1.0 - conditional_entropy(transpose(t, L, C), C, L, n) / h_clusters;
}

double v_measure(const std::vector<double> &t, std::size_t L, std::size_t C, double beta) {
    const double h = homogeneity(t, L, C), c = completeness(t, L, C);
    return h + c == 0.0 ? 0.0 : (1.0 + beta) * h * c / (beta * h + c);
}

double adjusted_rand_index(const std::vector<double> &t, std::size_t L, std::size_t C) {
    auto comb2 = [](double x) { return x * (x - 1.0) / 2.0; };
    std::vector<double> rows, cols;
    double n;
    totals(t, L, C, rows, cols, n);
    double index = 0.0, a = 0.0, b = 0.0;
    for (double v : t) {
        index += comb2(v);
    }
    for (double v : rows) {
        a += comb2(v);
    }
    for (double v : cols) {
        b += comb2(v);
    }
    const double expected = a * b / comb2(n);
    const double max_index = (a + b) / 2.0;
    return max_index == expected ? 1.0 : (index - expected) / (max_index - expected);
}

double purity(const std::vector<double> &t, std::size_t L, std::size_t C) {
    double total = 0.0, n = 0.0;
    for (std::size_t c = 0; c < C; ++c) {
        double best = 0.0;
        for (std::size_t l = 0; l < L; ++l) {
            best = std::max(best, t[l * C + c]);
            n += t[l * C + c];
        }
        total += best;
    }
    return total / n;
}

double trustworthiness(const Matrix &X, const Matrix &Y, std::size_t k) {
    const std::size_t n = X.rows;
    if (n != Y.rows) {
        throw std::invalid_argument("original and embedded must contain the same number of samples");
    }
    if (k == 0 || k >= n / 2) {
        throw std::invalid_argument("k must be at least 1 and less than half the number of samples");
    }
    double penalty = 0.0;
    std::vector<std::pair<double, std::size_t>> dx(n), dy(n);
    std::vector<std::size_t> rank(n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            dx[j] = {j == i ? -1.0 : squared_distance(X.row(i), X.row(j), X.cols), j};
            dy[j] = {j == i ? -1.0 : squared_distance(Y.row(i), Y.row(j), Y.cols), j};
        }
        std::sort(dx.begin(), dx.end());
        std::sort(dy.begin(), dy.end());
        for (std::size_t r = 0; r < n; ++r) {
            rank[dx[r].second] = r;  // rank 0 is the point itself
        }
        for (std::size_t r = 1; r <= k; ++r) {
            const std::size_t original_rank = rank[dy[r].second];
            if (original_rank > k) {
                penalty += static_cast<double>(original_rank - k);
            }
        }
    }
    const double nn = static_cast<double>(n), kk = static_cast<double>(k);
    return 1.0 - 2.0 / (nn * kk * (2.0 * nn - 3.0 * kk - 1.0)) * penalty;
}

}  // namespace mlcore
