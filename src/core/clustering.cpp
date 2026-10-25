#include "clustering.hpp"

#include <functional>
#include <numeric>
#include <queue>

namespace mlcore {

Seeder parse_seeder(const std::string &name) {
    if (name == "plusplus") return Seeder::PlusPlus;
    if (name == "random") return Seeder::Random;
    if (name == "kmc2") return Seeder::KMC2;
    if (name == "preset") return Seeder::Preset;
    throw std::invalid_argument("Unknown seeder \"" + name + "\"; expected plusplus, random, kmc2 or preset");
}

std::string seeder_name(Seeder s) {
    switch (s) {
        case Seeder::PlusPlus: return "plusplus";
        case Seeder::Random: return "random";
        case Seeder::KMC2: return "kmc2";
        case Seeder::Preset: return "preset";
    }
    return "plusplus";
}

Matrix seed_centers(const Matrix &X, std::size_t k, Seeder seeder, const Matrix &preset, const Distance &dist, Rng &rng) {
    const std::size_t n = X.rows, d = X.cols;
    if (k > n) {
        throw std::invalid_argument("Number of clusters (" + std::to_string(k) + ") cannot exceed the number of samples (" +
                                    std::to_string(n) + ")");
    }
    if (seeder == Seeder::Preset) {
        if (preset.rows != k || preset.cols != d) {
            throw std::invalid_argument("Preset centroids must be " + std::to_string(k) + " samples of " +
                                        std::to_string(d) + " features");
        }
        return preset;
    }
    Matrix centers(k, d);
    auto copy = [&](std::size_t c, std::size_t row) { std::copy(X.row(row), X.row(row) + d, centers.row(c)); };
    std::uniform_int_distribution<std::size_t> pick(0, n - 1);

    if (seeder == Seeder::Random) {
        std::vector<std::size_t> idx(n);
        std::iota(idx.begin(), idx.end(), 0);
        for (std::size_t c = 0; c < k; ++c) {
            std::swap(idx[c], idx[c + std::uniform_int_distribution<std::size_t>(0, n - 1 - c)(rng)]);
            copy(c, idx[c]);
        }
        return centers;
    }

    copy(0, pick(rng));
    auto sq = [&](const double *a, const double *b) {
        const double v = dist(a, b, d);
        return v * v;
    };
    if (seeder == Seeder::KMC2) {
        // Markov chain Monte Carlo approximation of D^2 sampling (Bachem et al.).
        const std::size_t chain = 50;
        std::uniform_real_distribution<double> unit(0.0, 1.0);
        for (std::size_t c = 1; c < k; ++c) {
            auto nearest_sq = [&](std::size_t row) {
                double best = std::numeric_limits<double>::infinity();
                for (std::size_t j = 0; j < c; ++j) {
                    best = std::min(best, sq(X.row(row), centers.row(j)));
                }
                return best;
            };
            std::size_t x = pick(rng);
            double dx = nearest_sq(x);
            for (std::size_t step = 1; step < chain; ++step) {
                const std::size_t y = pick(rng);
                const double dy = nearest_sq(y);
                if (dx == 0.0 || dy / dx > unit(rng)) {
                    x = y;
                    dx = dy;
                }
            }
            copy(c, x);
        }
        return centers;
    }

    std::vector<double> closest(n);
    for (std::size_t r = 0; r < n; ++r) {
        closest[r] = sq(X.row(r), centers.row(0));
    }
    for (std::size_t c = 1; c < k; ++c) {
        double total = 0.0;
        for (double v : closest) {
            total += v;
        }
        std::size_t chosen = pick(rng);
        if (total > 0.0) {
            const double target = std::uniform_real_distribution<double>(0.0, total)(rng);
            double acc = 0.0;
            for (std::size_t r = 0; r < n; ++r) {
                if (closest[r] <= 0.0) {
                    continue;
                }
                chosen = r;
                acc += closest[r];
                if (acc >= target) {
                    break;
                }
            }
        }
        copy(c, chosen);
        for (std::size_t r = 0; r < n; ++r) {
            closest[r] = std::min(closest[r], sq(X.row(r), centers.row(c)));
        }
    }
    return centers;
}

// ---- KMedoids --------------------------------------------------------------------------

int KMedoids::predict_one(const double *x, double *distance) const {
    std::size_t best = 0;
    double best_d = std::numeric_limits<double>::infinity();
    for (std::size_t c = 0; c < medoids.rows; ++c) {
        const double v = metric(x, medoids.row(c), medoids.cols);
        if (v < best_d) {
            best_d = v;
            best = c;
        }
    }
    if (distance) {
        *distance = best_d;
    }
    return static_cast<int>(best);
}

void KMedoids::fit(const Matrix &X) {
    require_training_set(X, X.rows);
    Rng rng = make_rng(has_seed, seed);
    KMedoids next = *this;
    next.medoids = seed_centers(X, clusters, seeder == Seeder::Preset ? Seeder::PlusPlus : seeder, Matrix(), metric, rng);
    next.labels.assign(X.rows, 0);
    next.iterations = 0;
    const std::size_t d = X.cols;
    for (long it = 0; it < max_iterations; ++it) {
        for (std::size_t r = 0; r < X.rows; ++r) {
            next.labels[r] = next.predict_one(X.row(r));
        }
        bool changed = false;
        for (std::size_t c = 0; c < clusters; ++c) {
            std::vector<std::size_t> members;
            for (std::size_t r = 0; r < X.rows; ++r) {
                if (next.labels[r] == static_cast<int>(c)) {
                    members.push_back(r);
                }
            }
            if (members.empty()) {
                continue;
            }
            std::size_t best = members[0];
            double best_cost = std::numeric_limits<double>::infinity();
            for (std::size_t a : members) {
                double cost = 0.0;
                for (std::size_t b : members) {
                    cost += metric(X.row(a), X.row(b), d);
                    if (cost >= best_cost) {
                        break;
                    }
                }
                if (cost < best_cost) {
                    best_cost = cost;
                    best = a;
                }
            }
            if (!std::equal(X.row(best), X.row(best) + d, next.medoids.row(c))) {
                std::copy(X.row(best), X.row(best) + d, next.medoids.row(c));
                changed = true;
            }
        }
        next.iterations = it + 1;
        if (!changed) {
            break;
        }
    }

    // PAM SWAP phase: the alternating steps above can stall in a local optimum
    // (e.g. when an outlier was picked as a seed). Try replacing each medoid by
    // each sample and apply the best improving swap until none is left. Uses
    // each sample's nearest/second-nearest medoid distances, so evaluating one
    // swap costs O(n). Skipped for very large inputs, where it is O(n^2) per pass.
    const std::size_t n = X.rows;
    if (n <= 5000 && clusters > 0) {
        std::vector<std::size_t> medoid_rows(clusters);
        for (std::size_t c = 0; c < clusters; ++c) {
            for (std::size_t r = 0; r < n; ++r) {
                if (std::equal(X.row(r), X.row(r) + d, next.medoids.row(c))) {
                    medoid_rows[c] = r;
                    break;
                }
            }
        }
        std::vector<double> nearest(n), second(n), candidate(n);
        std::vector<std::size_t> owner(n);
        for (long pass = 0; pass < max_iterations; ++pass) {
            for (std::size_t r = 0; r < n; ++r) {
                nearest[r] = second[r] = std::numeric_limits<double>::infinity();
                for (std::size_t c = 0; c < clusters; ++c) {
                    const double v = metric(X.row(r), X.row(medoid_rows[c]), d);
                    if (v < nearest[r]) {
                        second[r] = nearest[r];
                        nearest[r] = v;
                        owner[r] = c;
                    } else if (v < second[r]) {
                        second[r] = v;
                    }
                }
            }
            double best_delta = -1e-12;
            std::size_t best_c = 0, best_h = n;
            for (std::size_t h = 0; h < n; ++h) {
                if (std::find(medoid_rows.begin(), medoid_rows.end(), h) != medoid_rows.end()) {
                    continue;
                }
                for (std::size_t r = 0; r < n; ++r) {
                    candidate[r] = metric(X.row(r), X.row(h), d);
                }
                for (std::size_t c = 0; c < clusters; ++c) {
                    double delta = 0.0;
                    for (std::size_t r = 0; r < n; ++r) {
                        const double kept = owner[r] == c ? second[r] : nearest[r];
                        delta += std::min(kept, candidate[r]) - nearest[r];
                    }
                    if (delta < best_delta) {
                        best_delta = delta;
                        best_c = c;
                        best_h = h;
                    }
                }
            }
            if (best_h == n) {
                break;
            }
            medoid_rows[best_c] = best_h;
            std::copy(X.row(best_h), X.row(best_h) + d, next.medoids.row(best_c));
            ++next.iterations;
        }
    }

    next.inertia = 0.0;
    for (std::size_t r = 0; r < n; ++r) {
        double dr = 0.0;
        next.labels[r] = next.predict_one(X.row(r), &dr);
        next.inertia += dr;
    }
    *this = std::move(next);
}

void KMedoids::save(ArchiveWriter &w) const {
    w.size(clusters);
    w.i64(max_iterations);
    metric.save(w);
    w.u8(static_cast<std::uint8_t>(seeder));
    w.boolean(has_seed);
    w.u64(seed);
    w.matrix(medoids);
    w.ints(labels);
    w.f64(inertia);
    w.i64(iterations);
}

void KMedoids::load(ArchiveReader &r) {
    clusters = r.size();
    max_iterations = static_cast<long>(r.i64());
    metric = Distance::load(r);
    const std::uint8_t s = r.u8();
    ArchiveReader::require(s <= 3, "bad seeder");
    seeder = static_cast<Seeder>(s);
    has_seed = r.boolean();
    seed = r.u64();
    medoids = r.matrix();
    labels = r.ints_in(0, static_cast<int>(medoids.rows) - 1);
    inertia = r.f64();
    iterations = static_cast<long>(r.i64());
    ArchiveReader::require(medoids.rows == 0 || medoids.rows == clusters);
}

// ---- FuzzyCMeans ------------------------------------------------------------------------

void FuzzyCMeans::memberships(const double *x, double *out) const {
    const std::size_t k = centroids.rows, d = centroids.cols;
    std::vector<double> dist(k);
    for (std::size_t c = 0; c < k; ++c) {
        dist[c] = std::sqrt(squared_distance(x, centroids.row(c), d));
        if (dist[c] == 0.0) {
            std::fill(out, out + k, 0.0);
            out[c] = 1.0;
            return;
        }
    }
    const double power = 2.0 / (fuzz - 1.0);
    for (std::size_t c = 0; c < k; ++c) {
        double sum = 0.0;
        for (std::size_t j = 0; j < k; ++j) {
            sum += std::pow(dist[c] / dist[j], power);
        }
        out[c] = 1.0 / sum;
    }
}

void FuzzyCMeans::fit(const Matrix &X) {
    require_training_set(X, X.rows);
    Rng rng = make_rng(has_seed, seed);
    FuzzyCMeans next = *this;
    next.centroids = seed_centers(X, clusters, Seeder::PlusPlus, Matrix(), Distance(), rng);
    const std::size_t n = X.rows, d = X.cols, k = clusters;
    std::vector<double> u(k);
    next.iterations = 0;
    for (long it = 0; it < max_iterations; ++it) {
        Matrix sums(k, d);
        std::vector<double> weights(k, 0.0);
        next.inertia = 0.0;
        for (std::size_t r = 0; r < n; ++r) {
            next.memberships(X.row(r), u.data());
            for (std::size_t c = 0; c < k; ++c) {
                const double w = std::pow(u[c], fuzz);
                weights[c] += w;
                next.inertia += w * squared_distance(X.row(r), next.centroids.row(c), d);
                for (std::size_t j = 0; j < d; ++j) {
                    sums(c, j) += w * X(r, j);
                }
            }
        }
        double shift = 0.0;
        for (std::size_t c = 0; c < k; ++c) {
            if (weights[c] <= 0.0) {
                continue;
            }
            double delta = 0.0;
            for (std::size_t j = 0; j < d; ++j) {
                const double v = sums(c, j) / weights[c];
                delta += (v - next.centroids(c, j)) * (v - next.centroids(c, j));
                next.centroids(c, j) = v;
            }
            shift = std::max(shift, std::sqrt(delta));
        }
        next.iterations = it + 1;
        if (shift < min_change) {
            break;
        }
    }
    *this = std::move(next);
}

void FuzzyCMeans::save(ArchiveWriter &w) const {
    w.size(clusters);
    w.f64(fuzz);
    w.i64(max_iterations);
    w.f64(min_change);
    w.boolean(has_seed);
    w.u64(seed);
    w.matrix(centroids);
    w.f64(inertia);
    w.i64(iterations);
}

void FuzzyCMeans::load(ArchiveReader &r) {
    clusters = r.size();
    fuzz = r.f64();
    max_iterations = static_cast<long>(r.i64());
    min_change = r.f64();
    has_seed = r.boolean();
    seed = r.u64();
    centroids = r.matrix();
    inertia = r.f64();
    iterations = static_cast<long>(r.i64());
    ArchiveReader::require(fuzz > 1.0 && (centroids.rows == 0 || centroids.rows == clusters));
}

// ---- GaussianMixture --------------------------------------------------------------------

namespace {
constexpr double kLogTwoPi = 1.8378770664093454835606594728112;
}

void GaussianMixture::log_joint(const double *x, double *out) const {
    const std::size_t k = means.rows, d = means.cols;
    for (std::size_t c = 0; c < k; ++c) {
        double sum = 0.0;
        for (std::size_t j = 0; j < d; ++j) {
            const double v = variances(c, j);
            const double diff = x[j] - means(c, j);
            sum += kLogTwoPi + std::log(v) + diff * diff / v;
        }
        out[c] = log_priors[c] - 0.5 * sum;
    }
}

void GaussianMixture::probabilities(const double *x, double *out) const {
    log_joint(x, out);
    softmax(out, means.rows);
}

void GaussianMixture::fit(const Matrix &X) {
    require_training_set(X, X.rows);
    const std::size_t n = X.rows, d = X.cols, k = components;
    Rng rng = make_rng(has_seed, seed);
    GaussianMixture next = *this;
    next.means = seed_centers(X, k, Seeder::PlusPlus, Matrix(), Distance(), rng);

    std::vector<double> global_var(d, 0.0), mean(d, 0.0);
    for (std::size_t r = 0; r < n; ++r) {
        for (std::size_t j = 0; j < d; ++j) {
            mean[j] += X(r, j) / static_cast<double>(n);
        }
    }
    double max_var = 0.0;
    for (std::size_t r = 0; r < n; ++r) {
        for (std::size_t j = 0; j < d; ++j) {
            global_var[j] += (X(r, j) - mean[j]) * (X(r, j) - mean[j]) / static_cast<double>(n);
        }
    }
    for (double v : global_var) {
        max_var = std::max(max_var, v);
    }
    const double epsilon = std::max(smoothing * max_var, 1e-12);
    next.variances = Matrix(k, d);
    for (std::size_t c = 0; c < k; ++c) {
        for (std::size_t j = 0; j < d; ++j) {
            next.variances(c, j) = global_var[j] + epsilon;
        }
    }
    next.log_priors.assign(k, -std::log(static_cast<double>(k)));

    Matrix resp(n, k);
    double previous = -std::numeric_limits<double>::infinity();
    std::uniform_int_distribution<std::size_t> pick(0, n - 1);
    next.iterations = 0;
    for (long it = 0; it < max_iterations; ++it) {
        double ll = 0.0;
        for (std::size_t r = 0; r < n; ++r) {
            double *p = resp.row(r);
            next.log_joint(X.row(r), p);
            const double m = *std::max_element(p, p + k);
            double s = 0.0;
            for (std::size_t c = 0; c < k; ++c) {
                s += std::exp(p[c] - m);
            }
            const double lse = m + std::log(s);
            ll += lse;
            for (std::size_t c = 0; c < k; ++c) {
                p[c] = std::exp(p[c] - lse);
            }
        }
        for (std::size_t c = 0; c < k; ++c) {
            double nk = 0.0;
            for (std::size_t r = 0; r < n; ++r) {
                nk += resp(r, c);
            }
            if (nk < 1e-10) {
                // Collapsed component: restart it on a random sample.
                const std::size_t row = pick(rng);
                for (std::size_t j = 0; j < d; ++j) {
                    next.means(c, j) = X(row, j);
                    next.variances(c, j) = global_var[j] + epsilon;
                }
                next.log_priors[c] = std::log(1.0 / static_cast<double>(n));
                continue;
            }
            for (std::size_t j = 0; j < d; ++j) {
                double mu = 0.0;
                for (std::size_t r = 0; r < n; ++r) {
                    mu += resp(r, c) * X(r, j);
                }
                mu /= nk;
                double var = 0.0;
                for (std::size_t r = 0; r < n; ++r) {
                    var += resp(r, c) * (X(r, j) - mu) * (X(r, j) - mu);
                }
                next.means(c, j) = mu;
                next.variances(c, j) = var / nk + epsilon;
            }
            next.log_priors[c] = std::log(nk / static_cast<double>(n));
        }
        next.log_likelihood = ll;
        next.iterations = it + 1;
        if (std::fabs(ll - previous) / static_cast<double>(n) < min_change) {
            break;
        }
        previous = ll;
    }
    *this = std::move(next);
}

void GaussianMixture::save(ArchiveWriter &w) const {
    w.size(components);
    w.f64(smoothing);
    w.i64(max_iterations);
    w.f64(min_change);
    w.boolean(has_seed);
    w.u64(seed);
    w.matrix(means);
    w.matrix(variances);
    w.doubles(log_priors);
    w.f64(log_likelihood);
    w.i64(iterations);
}

void GaussianMixture::load(ArchiveReader &r) {
    components = r.size();
    smoothing = r.f64();
    max_iterations = static_cast<long>(r.i64());
    min_change = r.f64();
    has_seed = r.boolean();
    seed = r.u64();
    means = r.matrix();
    variances = r.matrix(means.rows, means.cols);
    log_priors = r.doubles(means.rows);
    log_likelihood = r.f64();
    iterations = static_cast<long>(r.i64());
    for (double v : variances.data) {
        ArchiveReader::require(v > 0.0, "variances must be positive");
    }
}

// ---- MeanShift -----------------------------------------------------------------------

double MeanShift::estimate_radius(const Matrix &X, double percentile, Rng &rng) {
    const std::size_t n = X.rows;
    std::vector<std::size_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::shuffle(idx.begin(), idx.end(), rng);
    idx.resize(std::min<std::size_t>(n, 500));
    std::vector<double> distances;
    for (std::size_t a = 0; a < idx.size(); ++a) {
        for (std::size_t b = a + 1; b < idx.size(); ++b) {
            distances.push_back(std::sqrt(squared_distance(X.row(idx[a]), X.row(idx[b]), X.cols)));
        }
    }
    if (distances.empty()) {
        return 1.0;
    }
    const double r = quantile(distances, percentile / 100.0);
    return r > 0.0 ? r : 1.0;
}

int MeanShift::predict_one(const double *x) const {
    std::size_t best = 0;
    double best_d = std::numeric_limits<double>::infinity();
    for (std::size_t c = 0; c < centroids.rows; ++c) {
        const double v = squared_distance(x, centroids.row(c), centroids.cols);
        if (v < best_d) {
            best_d = v;
            best = c;
        }
    }
    return static_cast<int>(best);
}

void MeanShift::fit(const Matrix &X, unsigned threads) {
    require_training_set(X, X.rows);
    Rng rng = make_rng(has_seed, seed);
    const double r = radius > 0.0 ? radius : estimate_radius(X, percentile, rng);
    const std::size_t n = X.rows, d = X.cols;

    NeighborIndex index;
    index.use_tree = true;
    index.build(X);

    std::vector<std::size_t> seeds(n);
    std::iota(seeds.begin(), seeds.end(), 0);
    if (n > max_seeds) {
        std::shuffle(seeds.begin(), seeds.end(), rng);
        seeds.resize(max_seeds);
    }
    Matrix shifted(seeds.size(), d);
    std::vector<std::size_t> density(seeds.size(), 0);
    std::vector<long> used(seeds.size(), 0);
    parallel_for(seeds.size(), threads, [&](std::size_t s) {
        std::vector<double> point(X.row(seeds[s]), X.row(seeds[s]) + d), mean(d);
        std::vector<Neighbor> nb;
        long it = 0;
        for (; it < max_iterations; ++it) {
            index.within(point.data(), r, nb);
            if (nb.empty()) {
                break;
            }
            std::fill(mean.begin(), mean.end(), 0.0);
            for (const Neighbor &x : nb) {
                for (std::size_t j = 0; j < d; ++j) {
                    mean[j] += X(x.second, j);
                }
            }
            for (double &v : mean) {
                v /= static_cast<double>(nb.size());
            }
            const double shift = std::sqrt(squared_distance(mean.data(), point.data(), d));
            point = mean;
            density[s] = nb.size();
            if (shift < min_shift) {
                break;
            }
        }
        used[s] = it + 1;
        std::copy(point.begin(), point.end(), shifted.row(s));
    });

    // Keep the densest modes; drop modes within the radius of a kept one.
    std::vector<std::size_t> order(seeds.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return density[a] > density[b]; });
    std::vector<std::size_t> kept;
    for (std::size_t s : order) {
        bool duplicate = false;
        for (std::size_t k : kept) {
            if (std::sqrt(squared_distance(shifted.row(s), shifted.row(k), d)) < r) {
                duplicate = true;
                break;
            }
        }
        if (!duplicate) {
            kept.push_back(s);
        }
    }
    MeanShift next = *this;
    next.centroids = Matrix(kept.size(), d);
    for (std::size_t c = 0; c < kept.size(); ++c) {
        std::copy(shifted.row(kept[c]), shifted.row(kept[c]) + d, next.centroids.row(c));
    }
    next.fitted_radius = r;
    next.iterations = used.empty() ? 0 : *std::max_element(used.begin(), used.end());
    *this = std::move(next);
}

void MeanShift::save(ArchiveWriter &w) const {
    w.f64(radius);
    w.f64(percentile);
    w.i64(max_iterations);
    w.f64(min_shift);
    w.size(max_seeds);
    w.boolean(has_seed);
    w.u64(seed);
    w.matrix(centroids);
    w.f64(fitted_radius);
    w.i64(iterations);
}

void MeanShift::load(ArchiveReader &r) {
    radius = r.f64();
    percentile = r.f64();
    max_iterations = static_cast<long>(r.i64());
    min_shift = r.f64();
    max_seeds = r.size();
    has_seed = r.boolean();
    seed = r.u64();
    centroids = r.matrix();
    fitted_radius = r.f64();
    iterations = static_cast<long>(r.i64());
}

// ---- DBSCAN / OPTICS -------------------------------------------------------------------

namespace {
NeighborIndex index_for(const Matrix &X, const Distance &dist) {
    NeighborIndex index;
    index.metric = dist;
    index.use_tree = dist.coordinate_bounded();
    index.build(X);
    return index;
}
}  // namespace

std::vector<int> dbscan(const Matrix &X, double radius, std::size_t min_density, const Distance &dist) {
    const std::size_t n = X.rows;
    const NeighborIndex index = index_for(X, dist);
    constexpr int kUnvisited = -2, kNoise = -1;
    std::vector<int> labels(n, kUnvisited);
    std::vector<Neighbor> nb, nb2;
    int cluster = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (labels[i] != kUnvisited) {
            continue;
        }
        index.within(X.row(i), radius, nb);
        if (nb.size() < min_density) {
            labels[i] = kNoise;
            continue;
        }
        labels[i] = cluster;
        std::vector<std::size_t> queue;
        for (const Neighbor &x : nb) {
            queue.push_back(x.second);
        }
        for (std::size_t q = 0; q < queue.size(); ++q) {
            const std::size_t j = queue[q];
            if (labels[j] == kNoise) {
                labels[j] = cluster;  // border point
            }
            if (labels[j] != kUnvisited) {
                continue;
            }
            labels[j] = cluster;
            index.within(X.row(j), radius, nb2);
            if (nb2.size() >= min_density) {
                for (const Neighbor &x : nb2) {
                    if (labels[x.second] == kUnvisited || labels[x.second] == kNoise) {
                        queue.push_back(x.second);
                    }
                }
            }
        }
        ++cluster;
    }
    return labels;
}

OpticsResult optics(const Matrix &X, std::size_t min_density, double max_radius, double radius, const Distance &dist) {
    const std::size_t n = X.rows;
    const NeighborIndex index = index_for(X, dist);
    const double inf = std::numeric_limits<double>::infinity();
    OpticsResult out;
    out.reachability.assign(n, inf);
    out.core_distance.assign(n, inf);
    std::vector<bool> processed(n, false);
    std::vector<Neighbor> nb;

    auto neighbors_of = [&](std::size_t p) {
        index.within(X.row(p), max_radius, nb);
        out.core_distance[p] = nb.size() >= min_density ? nb[min_density - 1].first : inf;
    };

    using Entry = std::pair<double, std::size_t>;
    for (std::size_t start = 0; start < n; ++start) {
        if (processed[start]) {
            continue;
        }
        std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> seeds;
        seeds.emplace(inf, start);
        while (!seeds.empty()) {
            const auto [reach, p] = seeds.top();
            seeds.pop();
            if (processed[p] || (reach > out.reachability[p] && reach != inf)) {
                continue;  // stale queue entry
            }
            processed[p] = true;
            out.ordering.push_back(p);
            neighbors_of(p);
            if (out.core_distance[p] == inf) {
                continue;
            }
            for (const Neighbor &q : nb) {
                if (processed[q.second]) {
                    continue;
                }
                const double candidate = std::max(out.core_distance[p], q.first);
                if (candidate < out.reachability[q.second]) {
                    out.reachability[q.second] = candidate;
                    seeds.emplace(candidate, q.second);
                }
            }
        }
    }

    // DBSCAN-equivalent extraction at `radius`.
    out.labels.assign(n, -1);
    int cluster = -1;
    for (std::size_t p : out.ordering) {
        if (out.reachability[p] > radius) {
            if (out.core_distance[p] <= radius) {
                out.labels[p] = ++cluster;
            }
        } else if (cluster >= 0) {
            out.labels[p] = cluster;
        }
    }
    return out;
}

// ---- Ward ---------------------------------------------------------------------------------

std::vector<int> ward(const Matrix &X, std::size_t clusters) {
    const std::size_t n = X.rows, d = X.cols;
    if (clusters < 1 || clusters > n) {
        throw std::invalid_argument("clusters must be between 1 and the number of samples");
    }
    Matrix centers = X;
    std::vector<double> size(n, 1.0);
    std::vector<bool> active(n, true);
    auto cost = [&](std::size_t a, std::size_t b) {
        return size[a] * size[b] / (size[a] + size[b]) * squared_distance(centers.row(a), centers.row(b), d);
    };

    struct Merge {
        double height;
        std::size_t a, b;
    };
    std::vector<Merge> merges;
    std::vector<std::size_t> chain;
    std::size_t remaining = n;
    while (remaining > 1) {
        if (chain.empty()) {
            for (std::size_t i = 0; i < n; ++i) {
                if (active[i]) {
                    chain.push_back(i);
                    break;
                }
            }
        }
        const std::size_t a = chain.back();
        const std::size_t previous = chain.size() > 1 ? chain[chain.size() - 2] : n;
        std::size_t best = n;
        double best_cost = std::numeric_limits<double>::infinity();
        if (previous != n) {
            best = previous;
            best_cost = cost(a, previous);
        }
        for (std::size_t i = 0; i < n; ++i) {
            if (!active[i] || i == a) {
                continue;
            }
            const double c = cost(a, i);
            if (c < best_cost) {
                best_cost = c;
                best = i;
            }
        }
        if (best == previous) {
            chain.pop_back();
            chain.pop_back();
            // Merge `previous` into `a`.
            const double total = size[a] + size[best];
            for (std::size_t j = 0; j < d; ++j) {
                centers(a, j) = (centers(a, j) * size[a] + centers(best, j) * size[best]) / total;
            }
            size[a] = total;
            active[best] = false;
            merges.push_back({best_cost, a, best});
            --remaining;
        } else {
            chain.push_back(best);
        }
    }

    // Replay the lowest n - clusters merges (Ward heights are monotone).
    std::stable_sort(merges.begin(), merges.end(), [](const Merge &x, const Merge &y) { return x.height < y.height; });
    std::vector<std::size_t> parent(n);
    std::iota(parent.begin(), parent.end(), 0);
    std::function<std::size_t(std::size_t)> find = [&](std::size_t x) { return parent[x] == x ? x : parent[x] = find(parent[x]); };
    // Merges reference cluster representatives at merge time; replaying in height
    // order is consistent because a cluster's later merges are always higher.
    for (std::size_t m = 0; m + clusters < n; ++m) {
        parent[find(merges[m].b)] = find(merges[m].a);
    }
    std::vector<int> labels(n);
    std::vector<int> id(n, -1);
    int next = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t root = find(i);
        if (id[root] < 0) {
            id[root] = next++;
        }
        labels[i] = id[root];
    }
    return labels;
}

}  // namespace mlcore
