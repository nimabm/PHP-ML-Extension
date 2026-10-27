#include "manifold.hpp"

#include <queue>
#include <tuple>

#include "linalg.hpp"

namespace mlcore {

namespace {

constexpr double kInfinity = std::numeric_limits<double>::infinity();

void check_dimensions(std::size_t k, std::size_t n) {
    if (k > n) {
        throw std::invalid_argument("dimensions (" + std::to_string(k) + ") cannot exceed the number of samples (" +
                                    std::to_string(n) + ")");
    }
}

// The symmetric n x n matrix of f(i, j), computed for j <= i and mirrored.
template <typename F>
std::vector<double> symmetric_matrix(std::size_t n, unsigned threads, F f) {
    std::vector<double> M(n * n);
    parallel_for(n, threads, [&](std::size_t i) {
        for (std::size_t j = 0; j <= i; ++j) {
            M[i * n + j] = f(i, j);
        }
    });
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < i; ++j) {
            M[j * n + i] = M[i * n + j];
        }
    }
    return M;
}

// Row means of a symmetric matrix and their mean.
std::vector<double> row_means(const std::vector<double> &M, std::size_t n, double &grand) {
    std::vector<double> means(n, 0.0);
    grand = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            means[i] += M[i * n + j];
        }
        means[i] /= static_cast<double>(n);
        grand += means[i];
    }
    grand /= static_cast<double>(n);
    return means;
}

}  // namespace

// ---- kernel PCA --------------------------------------------------------------------------

void KernelPCA::fit(const Matrix &X, unsigned threads) {
    require_training_set(X, X.rows);
    check_dimensions(dimensions, X.rows);
    const std::size_t n = X.rows, k = dimensions;
    Kernel resolved = kernel;
    if (resolved.gamma <= 0.0) {
        resolved.gamma = 1.0 / static_cast<double>(X.cols);
    }
    std::vector<double> K = symmetric_matrix(n, threads, [&](std::size_t i, std::size_t j) {
        return resolved(X.row(i), X.row(j), X.cols);
    });
    double grand = 0.0;
    std::vector<double> means = row_means(K, n, grand);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            K[i * n + j] += grand - means[i] - means[j];
        }
    }
    std::vector<double> values;
    Matrix V;
    top_eigen(K, n, k, values, V, threads);

    // Components with a (numerically) zero eigenvalue carry no information and map to 0.
    const double floor = 1e-12 * std::max(values[0], 0.0);
    Matrix A(n, k);
    for (std::size_t c = 0; c < k; ++c) {
        if (values[c] > floor && values[c] > 0.0) {
            const double scale = 1.0 / std::sqrt(values[c]);
            for (std::size_t i = 0; i < n; ++i) {
                A(i, c) = V(i, c) * scale;
            }
        }
    }
    active = resolved;
    samples = X;
    alphas = std::move(A);
    column_means = std::move(means);
    grand_mean = grand;
    eigenvalues = std::move(values);
}

void KernelPCA::transform(const double *x, double *out) const {
    const std::size_t n = samples.rows, k = alphas.cols;
    std::vector<double> kx(n);
    double mean = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        kx[i] = active(x, samples.row(i), samples.cols);
        mean += kx[i];
    }
    mean /= static_cast<double>(n);
    std::fill(out, out + k, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        const double centered = kx[i] - mean - column_means[i] + grand_mean;
        const double *a = alphas.row(i);
        for (std::size_t c = 0; c < k; ++c) {
            out[c] += centered * a[c];
        }
    }
}

void KernelPCA::save(ArchiveWriter &w) const {
    w.size(dimensions);
    kernel.save(w);
    active.save(w);
    w.matrix(samples);
    w.matrix(alphas);
    w.doubles(column_means);
    w.f64(grand_mean);
    w.doubles(eigenvalues);
}

void KernelPCA::load(ArchiveReader &r) {
    dimensions = r.size();
    ArchiveReader::require(dimensions >= 1, "dimensions must be at least 1");
    kernel.load(r);
    active.load(r);
    samples = r.matrix();
    alphas = r.matrix();
    column_means = r.doubles();
    grand_mean = r.f64();
    eigenvalues = r.doubles();
    if (fitted()) {
        ArchiveReader::require(samples.cols > 0 && alphas.rows == samples.rows && alphas.cols == dimensions &&
                               column_means.size() == samples.rows && eigenvalues.size() == dimensions);
    } else {
        ArchiveReader::require(alphas.data.empty() && column_means.empty() && eigenvalues.empty());
    }
}

// ---- Isomap ------------------------------------------------------------------------------

namespace {

using Adjacency = std::vector<std::vector<std::pair<std::size_t, double>>>;

void dijkstra(const Adjacency &adjacency, std::size_t source, double *dist) {
    using Item = std::pair<double, std::size_t>;
    std::priority_queue<Item, std::vector<Item>, std::greater<Item>> heap;
    dist[source] = 0.0;
    heap.push({0.0, source});
    while (!heap.empty()) {
        const auto [d, u] = heap.top();
        heap.pop();
        if (d > dist[u]) {
            continue;
        }
        for (const auto &[v, w] : adjacency[u]) {
            const double candidate = d + w;
            if (candidate < dist[v]) {
                dist[v] = candidate;
                heap.push({candidate, v});
            }
        }
    }
}

std::size_t find_root(std::vector<std::size_t> &parent, std::size_t i) {
    while (parent[i] != i) {
        parent[i] = parent[parent[i]];
        i = parent[i];
    }
    return i;
}

}  // namespace

void Isomap::fit(const Matrix &X, unsigned threads) {
    require_training_set(X, X.rows);
    const std::size_t n = X.rows, k = dimensions;
    if (n < 2) {
        throw std::invalid_argument("Isomap needs at least 2 samples");
    }
    check_dimensions(k, n);
    NeighborIndex idx = make_index(X, metric);
    const auto graph = neighbor_graph(idx, neighbors, threads);

    Adjacency adjacency(n);
    std::vector<std::size_t> parent(n);
    for (std::size_t i = 0; i < n; ++i) {
        parent[i] = i;
    }
    std::size_t components = n;
    for (std::size_t i = 0; i < n; ++i) {
        for (const Neighbor &nb : graph[i]) {
            adjacency[i].push_back({nb.second, nb.first});
            adjacency[nb.second].push_back({i, nb.first});
            const std::size_t a = find_root(parent, i), b = find_root(parent, nb.second);
            if (a != b) {
                parent[a] = b;
                --components;
            }
        }
    }
    if (components > 1) {
        throw std::invalid_argument("The neighborhood graph has " + std::to_string(components) +
                                    " disconnected components; increase neighbors (currently " +
                                    std::to_string(neighbors) + ")");
    }

    Matrix G(n, n, kInfinity);
    parallel_for(n, threads, [&](std::size_t s) { dijkstra(adjacency, s, G.row(s)); });
    // Squared geodesics, symmetrized against rounding along reversed paths.
    std::vector<double> S = symmetric_matrix(n, threads, [&](std::size_t i, std::size_t j) {
        const double g = 0.5 * (G(i, j) + G(j, i));
        return g * g;
    });
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            G(i, j) = std::sqrt(S[i * n + j]);
        }
    }
    double grand = 0.0;
    std::vector<double> means = row_means(S, n, grand);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            S[i * n + j] = -0.5 * (S[i * n + j] - means[i] - means[j] + grand);
        }
    }
    std::vector<double> values;
    Matrix V;
    top_eigen(S, n, k, values, V, threads);

    index = std::move(idx);
    geodesic = std::move(G);
    vectors = std::move(V);
    eigenvalues = std::move(values);
    squared_means = std::move(means);
}

void Isomap::transform(const double *x, double *out) const {
    const std::size_t n = index.samples.rows, k = vectors.cols;
    std::vector<Neighbor> nearest;
    index.nearest(x, std::min(neighbors, n), nearest);
    // Geodesic distance through the closest training neighbors.
    std::vector<double> g(n, kInfinity);
    for (const Neighbor &nb : nearest) {
        const double *row = geodesic.row(nb.second);
        for (std::size_t j = 0; j < n; ++j) {
            g[j] = std::min(g[j], nb.first + row[j]);
        }
    }
    std::fill(out, out + k, 0.0);
    for (std::size_t j = 0; j < n; ++j) {
        const double t = g[j] * g[j] - squared_means[j];
        const double *v = vectors.row(j);
        for (std::size_t c = 0; c < k; ++c) {
            out[c] += v[c] * t;
        }
    }
    for (std::size_t c = 0; c < k; ++c) {
        out[c] = eigenvalues[c] > 0.0 ? -0.5 * out[c] / std::sqrt(eigenvalues[c]) : 0.0;
    }
}

void Isomap::save(ArchiveWriter &w) const {
    w.size(dimensions);
    w.size(neighbors);
    metric.save(w);
    index.save(w);
    w.matrix(geodesic);
    w.matrix(vectors);
    w.doubles(eigenvalues);
    w.doubles(squared_means);
}

void Isomap::load(ArchiveReader &r) {
    dimensions = r.size();
    neighbors = r.size();
    ArchiveReader::require(dimensions >= 1 && neighbors >= 1, "bad Isomap parameters");
    metric = Distance::load(r);
    index.load(r);
    geodesic = r.matrix();
    vectors = r.matrix();
    eigenvalues = r.doubles();
    squared_means = r.doubles();
    const std::size_t n = index.samples.rows;
    if (fitted()) {
        ArchiveReader::require(geodesic.rows == n && geodesic.cols == n && vectors.rows == n &&
                               vectors.cols == dimensions && eigenvalues.size() == dimensions &&
                               squared_means.size() == n);
        for (double g : geodesic.data) {
            ArchiveReader::require(g >= 0.0 && std::isfinite(g), "geodesic distances must be finite");
        }
    } else {
        ArchiveReader::require(geodesic.data.empty() && vectors.data.empty() && eigenvalues.empty() && squared_means.empty());
    }
}

// ---- UMAP --------------------------------------------------------------------------------

void UMAPParams::save(ArchiveWriter &w) const {
    w.size(dimensions);
    w.size(neighbors);
    w.f64(min_distance);
    w.f64(spread);
    w.size(epochs);
    w.f64(rate);
    w.size(negative_samples);
    w.f64(repulsion);
    w.boolean(random_init);
    metric.save(w);
    w.boolean(has_seed);
    w.u64(seed);
}

void UMAPParams::load(ArchiveReader &r) {
    dimensions = r.size();
    neighbors = r.size();
    min_distance = r.f64();
    spread = r.f64();
    epochs = r.size(kMaxEpochs);
    rate = r.f64();
    negative_samples = r.size(1000);
    repulsion = r.f64();
    random_init = r.boolean();
    metric = Distance::load(r);
    has_seed = r.boolean();
    seed = r.u64();
    ArchiveReader::require(dimensions >= 1 && neighbors >= 1 && spread > 0.0 && min_distance >= 0.0 &&
                               min_distance <= spread && rate > 0.0 && repulsion >= 0.0,
                           "bad UMAP parameters");
}

namespace {

struct Membership {
    double rho = 0.0, sigma = 1.0;
};

// UMAP's smooth k-NN distance: rho is the distance to the nearest neighbor and
// sigma is chosen so that the memberships of the neighbors sum to log2(k + 1).
Membership smooth_knn(const std::vector<Neighbor> &nbs, double fallback_scale) {
    Membership m;
    for (const Neighbor &nb : nbs) {
        if (nb.first > 0.0) {
            m.rho = nb.first;
            break;
        }
    }
    const double target = std::log2(static_cast<double>(nbs.size()) + 1.0);
    double lo = 0.0, hi = kInfinity, mid = 1.0;
    for (int step = 0; step < 64; ++step) {
        double sum = 0.0;
        for (const Neighbor &nb : nbs) {
            const double d = nb.first - m.rho;
            sum += d > 0.0 ? std::exp(-d / mid) : 1.0;
        }
        if (std::fabs(sum - target) < 1e-5) {
            break;
        }
        if (sum > target) {
            hi = mid;
            mid = (lo + hi) / 2.0;
        } else {
            lo = mid;
            mid = std::isinf(hi) ? mid * 2.0 : (lo + hi) / 2.0;
        }
    }
    double mean = 0.0;
    for (const Neighbor &nb : nbs) {
        mean += nb.first;
    }
    mean /= static_cast<double>(std::max<std::size_t>(nbs.size(), 1));
    m.sigma = std::max(mid, 1e-3 * (m.rho > 0.0 ? mean : fallback_scale));
    return m;
}

double membership(double distance, const Membership &m) {
    const double d = distance - m.rho;
    return d <= 0.0 || m.sigma <= 0.0 ? 1.0 : std::exp(-d / m.sigma);
}

struct Edges {
    std::vector<std::size_t> head, tail;
    std::vector<double> weight;

    void add(std::size_t h, std::size_t t, double w) {
        head.push_back(h);
        tail.push_back(t);
        weight.push_back(w);
    }
};

double clip(double v) { return v > 4.0 ? 4.0 : (v < -4.0 ? -4.0 : v); }

// UMAP's stochastic gradient descent over the fuzzy graph's edges. Head rows
// live in `head` and tail rows in `tail`; when `same_space` is set they are the
// same embedding and tails move too (fit), otherwise only heads move (transform).
void optimize(double *head, const double *tail, std::size_t tail_rows, bool same_space, std::size_t dims,
              const Edges &edges, std::size_t epochs, double a, double b, double rate, double repulsion,
              std::size_t negative, Rng &rng) {
    const std::size_t count = edges.weight.size();
    if (count == 0) {
        return;
    }
    double max_w = 0.0;
    for (double w : edges.weight) {
        max_w = std::max(max_w, w);
    }
    std::vector<double> period(count), next(count), negative_period(count), next_negative(count);
    for (std::size_t e = 0; e < count; ++e) {
        period[e] = max_w / edges.weight[e];
        next[e] = period[e];
        negative_period[e] = negative > 0 ? period[e] / static_cast<double>(negative) : kInfinity;
        next_negative[e] = negative_period[e];
    }
    double *moving_tail = same_space ? head : nullptr;
    std::uniform_int_distribution<std::size_t> pick(0, tail_rows - 1);

    for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
        const double alpha = rate * (1.0 - static_cast<double>(epoch) / static_cast<double>(epochs));
        const double now = static_cast<double>(epoch);
        for (std::size_t e = 0; e < count; ++e) {
            if (next[e] > now) {
                continue;
            }
            double *current = head + edges.head[e] * dims;
            const double *other = tail + edges.tail[e] * dims;
            const double d2 = squared_distance(current, other, dims);
            if (d2 > 0.0) {
                const double p = std::pow(d2, b - 1.0);  // d2^b = p * d2
                const double coef = -2.0 * a * b * p / (a * p * d2 + 1.0);
                for (std::size_t c = 0; c < dims; ++c) {
                    const double g = clip(coef * (current[c] - other[c])) * alpha;
                    current[c] += g;
                    if (moving_tail) {
                        moving_tail[edges.tail[e] * dims + c] -= g;
                    }
                }
            }
            next[e] += period[e];

            const double due = (now - next_negative[e]) / negative_period[e];
            const std::size_t samples = due > 0.0 ? static_cast<std::size_t>(due) : 0;
            for (std::size_t s = 0; s < samples; ++s) {
                const std::size_t k = pick(rng);
                if (same_space && k == edges.head[e]) {
                    continue;
                }
                const double *neg = tail + k * dims;
                const double nd2 = squared_distance(current, neg, dims);
                const double coef =
                    nd2 > 0.0 ? 2.0 * repulsion * b / ((0.001 + nd2) * (a * std::pow(nd2, b) + 1.0)) : 0.0;
                for (std::size_t c = 0; c < dims; ++c) {
                    current[c] += (coef > 0.0 ? clip(coef * (current[c] - neg[c])) : 4.0) * alpha;
                }
            }
            next_negative[e] += static_cast<double>(samples) * negative_period[e];
        }
    }
}

std::size_t default_epochs(std::size_t samples) { return samples <= 10000 ? 500 : 200; }

}  // namespace

void UMAP::fit_curve(double spread, double min_distance, double &a, double &b) {
    constexpr int kPoints = 300;
    std::vector<double> xs(kPoints), ys(kPoints);
    for (int i = 0; i < kPoints; ++i) {
        const double x = 3.0 * spread * static_cast<double>(i) / (kPoints - 1);
        xs[i] = x;
        ys[i] = x < min_distance ? 1.0 : std::exp(-(x - min_distance) / spread);
    }
    auto cost = [&](double pa, double pb) {
        double sum = 0.0;
        for (int i = 0; i < kPoints; ++i) {
            const double r = 1.0 / (1.0 + pa * std::pow(xs[i], 2.0 * pb)) - ys[i];
            sum += r * r;
        }
        return sum;
    };
    // Levenberg-Marquardt on the two curve parameters.
    double pa = 1.0, pb = 1.0, lambda = 1e-3, current = cost(pa, pb);
    for (int step = 0; step < 500 && lambda < 1e10; ++step) {
        double jaa = 0.0, jab = 0.0, jbb = 0.0, ga = 0.0, gb = 0.0;
        for (int i = 0; i < kPoints; ++i) {
            if (xs[i] <= 0.0) {
                continue;  // the curve is 1 at 0 whatever a and b are
            }
            const double t = std::pow(xs[i], 2.0 * pb), q = 1.0 + pa * t;
            const double r = 1.0 / q - ys[i];
            const double da = -t / (q * q), db = -pa * t * 2.0 * std::log(xs[i]) / (q * q);
            jaa += da * da;
            jab += da * db;
            jbb += db * db;
            ga += da * r;
            gb += db * r;
        }
        const double h11 = jaa * (1.0 + lambda), h22 = jbb * (1.0 + lambda), det = h11 * h22 - jab * jab;
        if (!(std::fabs(det) > 0.0)) {
            break;
        }
        const double na = pa + (-ga * h22 + gb * jab) / det, nb = pb + (-gb * h11 + ga * jab) / det;
        const double candidate = na > 0.0 && nb > 0.0 ? cost(na, nb) : kInfinity;
        if (candidate < current) {
            const bool converged = current - candidate < 1e-14 * std::max(current, 1e-300);
            pa = na;
            pb = nb;
            current = candidate;
            lambda /= 10.0;
            if (converged) {
                break;
            }
        } else {
            lambda *= 10.0;
        }
    }
    a = pa;
    b = pb;
}

void UMAP::fit(const Matrix &X, unsigned threads) {
    require_training_set(X, X.rows);
    const std::size_t n = X.rows, dims = params.dimensions;
    if (n < 3) {
        throw std::invalid_argument("UMAP needs at least 3 samples");
    }
    if (dims >= n) {
        throw std::invalid_argument("dimensions (" + std::to_string(dims) + ") must be less than the number of samples (" +
                                    std::to_string(n) + ")");
    }
    NeighborIndex idx = make_index(X, params.metric);
    const auto graph = neighbor_graph(idx, params.neighbors, threads);

    double scale = 0.0;
    std::size_t pairs = 0;
    for (const auto &nbs : graph) {
        for (const Neighbor &nb : nbs) {
            scale += nb.first;
            ++pairs;
        }
    }
    scale /= static_cast<double>(std::max<std::size_t>(pairs, 1));

    // Directed memberships, then the fuzzy union w = a + b - ab of both directions.
    std::vector<std::vector<double>> directed(n);
    parallel_for(n, threads, [&](std::size_t i) {
        const Membership m = smooth_knn(graph[i], scale);
        for (const Neighbor &nb : graph[i]) {
            directed[i].push_back(membership(nb.first, m));
        }
    });
    std::vector<std::tuple<std::size_t, std::size_t, bool, double>> entries;  // (lo, hi, forward, w)
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t m = 0; m < graph[i].size(); ++m) {
            const std::size_t j = graph[i][m].second;
            entries.emplace_back(std::min(i, j), std::max(i, j), i < j, directed[i][m]);
        }
    }
    std::sort(entries.begin(), entries.end());
    const std::size_t epochs = params.epochs ? params.epochs : default_epochs(n);
    Edges undirected;
    for (std::size_t e = 0; e < entries.size();) {
        const std::size_t lo = std::get<0>(entries[e]), hi = std::get<1>(entries[e]);
        double forward = 0.0, backward = 0.0;
        for (; e < entries.size() && std::get<0>(entries[e]) == lo && std::get<1>(entries[e]) == hi; ++e) {
            (std::get<2>(entries[e]) ? forward : backward) = std::get<3>(entries[e]);
        }
        undirected.add(lo, hi, forward + backward - forward * backward);
    }
    double max_w = 0.0;
    for (double w : undirected.weight) {
        max_w = std::max(max_w, w);
    }
    Edges edges;
    for (std::size_t e = 0; e < undirected.weight.size(); ++e) {
        const double w = undirected.weight[e];
        if (w > 0.0 && w >= max_w / static_cast<double>(epochs)) {
            edges.add(undirected.head[e], undirected.tail[e], w);
            edges.add(undirected.tail[e], undirected.head[e], w);
        }
    }

    double ca = 0.0, cb = 0.0;
    fit_curve(params.spread, params.min_distance, ca, cb);

    Rng rng = make_rng(params.has_seed, params.seed);
    Matrix Y(n, dims);
    bool initialized = false;
    if (!params.random_init && dims <= X.cols) {
        // PCA initialization, scaled so that the largest coordinate is 10.
        const std::vector<double> mean = column_means(X);
        std::vector<double> values, vectors;
        symmetric_eigen(covariance(X, mean), X.cols, values, vectors);
        double largest = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t c = 0; c < dims; ++c) {
                double s = 0.0;
                for (std::size_t j = 0; j < X.cols; ++j) {
                    s += (X(i, j) - mean[j]) * vectors[j * X.cols + c];
                }
                Y(i, c) = s;
                largest = std::max(largest, std::fabs(s));
            }
        }
        if (largest > 0.0) {
            for (double &v : Y.data) {
                v *= 10.0 / largest;
            }
            initialized = true;
        }
    }
    if (!initialized) {
        std::uniform_real_distribution<double> uniform(-10.0, 10.0);
        for (double &v : Y.data) {
            v = uniform(rng);
        }
    }
    optimize(Y.data.data(), Y.data.data(), n, true, dims, edges, epochs, ca, cb, params.rate, params.repulsion,
             params.negative_samples, rng);

    index = std::move(idx);
    embedding = std::move(Y);
    a = ca;
    b = cb;
}

Matrix UMAP::transform(const Matrix &Q, unsigned threads) const {
    const std::size_t n = index.samples.rows, dims = params.dimensions;
    const std::size_t k = std::min(params.neighbors, n);
    const std::size_t epochs = params.epochs ? std::max<std::size_t>(params.epochs / 3, 1) : (n <= 10000 ? 100 : 30);
    // One generator per query, derived from a single draw, so that the result
    // does not depend on the number of threads.
    Rng master = make_rng(params.has_seed, params.seed);
    const std::uint64_t base = master();
    Matrix out(Q.rows, dims);
    parallel_for(Q.rows, threads, [&](std::size_t q) {
        std::vector<Neighbor> nbs;
        index.nearest(Q.row(q), k, nbs);
        double scale = 0.0;
        for (const Neighbor &nb : nbs) {
            scale += nb.first;
        }
        scale /= static_cast<double>(std::max<std::size_t>(nbs.size(), 1));
        const Membership m = smooth_knn(nbs, scale);
        Edges edges;
        double *y = out.row(q);
        double total = 0.0, max_w = 0.0;
        for (const Neighbor &nb : nbs) {
            const double w = membership(nb.first, m);
            const double *e = embedding.row(nb.second);
            for (std::size_t c = 0; c < dims; ++c) {
                y[c] += w * e[c];
            }
            total += w;
            max_w = std::max(max_w, w);
            edges.add(0, nb.second, w);
        }
        for (std::size_t c = 0; c < dims; ++c) {
            y[c] /= total;
        }
        Edges kept;
        for (std::size_t e = 0; e < edges.weight.size(); ++e) {
            if (edges.weight[e] >= max_w / static_cast<double>(epochs)) {
                kept.add(0, edges.tail[e], edges.weight[e]);
            }
        }
        Rng rng(base ^ (0x9E3779B97F4A7C15ULL * (q + 1)));
        optimize(y, embedding.data.data(), n, false, dims, kept, epochs, a, b, params.rate / 4.0, params.repulsion,
                 params.negative_samples, rng);
    });
    return out;
}

void UMAP::save(ArchiveWriter &w) const {
    params.save(w);
    index.save(w);
    w.matrix(embedding);
    w.f64(a);
    w.f64(b);
}

void UMAP::load(ArchiveReader &r) {
    params.load(r);
    index.load(r);
    embedding = r.matrix();
    a = r.f64();
    b = r.f64();
    if (fitted()) {
        ArchiveReader::require(embedding.rows == index.samples.rows && embedding.cols == params.dimensions &&
                               index.samples.rows >= 3 && a > 0.0 && b > 0.0);
    } else {
        ArchiveReader::require(index.samples.rows == 0);
    }
}

}  // namespace mlcore
