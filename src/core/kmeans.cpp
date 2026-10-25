#include "kmeans.hpp"

#include <algorithm>
#include <limits>
#include <random>

namespace mlcore {

namespace {

Matrix init_plus_plus(const Matrix &X, std::size_t k, std::mt19937_64 &rng) {
    const std::size_t n = X.rows;
    const std::size_t d = X.cols;
    Matrix centers(k, d);
    std::uniform_int_distribution<std::size_t> pick(0, n - 1);

    auto copy_row = [&](std::size_t center, std::size_t sample) {
        std::copy(X.row(sample), X.row(sample) + d, centers.row(center));
    };

    copy_row(0, pick(rng));
    std::vector<double> closest(n);
    for (std::size_t r = 0; r < n; ++r) {
        closest[r] = squared_distance(X.row(r), centers.row(0), d);
    }

    for (std::size_t c = 1; c < k; ++c) {
        double total = 0.0;
        for (double v : closest) {
            total += v;
        }
        std::size_t chosen = 0;
        if (total > 0.0) {
            // Sample proportionally to the squared distance to the nearest center.
            const double target = std::uniform_real_distribution<double>(0.0, total)(rng);
            double acc = 0.0;
            for (std::size_t r = 0; r < n; ++r) {
                if (closest[r] <= 0.0) {
                    continue;
                }
                chosen = r;  // also the fallback if rounding leaves acc just below target
                acc += closest[r];
                if (acc >= target) {
                    break;
                }
            }
        } else {
            chosen = pick(rng);  // all points coincide with existing centers
        }
        copy_row(c, chosen);
        for (std::size_t r = 0; r < n; ++r) {
            closest[r] = std::min(closest[r], squared_distance(X.row(r), centers.row(c), d));
        }
    }
    return centers;
}

}  // namespace

void centroid_memberships(const Matrix &centroids, const double *x, double *out) {
    // Membership proportional to inverse distance; an exact hit takes it all.
    const std::size_t k = centroids.rows;
    double total = 0.0;
    for (std::size_t c = 0; c < k; ++c) {
        const double dist = std::sqrt(squared_distance(x, centroids.row(c), centroids.cols));
        if (dist == 0.0) {
            std::fill(out, out + k, 0.0);
            out[c] = 1.0;
            return;
        }
        out[c] = 1.0 / dist;
        total += out[c];
    }
    for (std::size_t c = 0; c < k; ++c) {
        out[c] /= total;
    }
}

int nearest_centroid(const Matrix &centroids, const double *x, double *distance) {
    std::size_t best = 0;
    double best_dist = std::numeric_limits<double>::infinity();
    for (std::size_t c = 0; c < centroids.rows; ++c) {
        const double dist = squared_distance(x, centroids.row(c), centroids.cols);
        if (dist < best_dist) {
            best_dist = dist;
            best = c;
        }
    }
    if (distance) {
        *distance = best_dist;
    }
    return static_cast<int>(best);
}

void KMeans::probabilities(const double *x, double *out) const { centroid_memberships(centroids, x, out); }

int KMeans::predict_one(const double *x, double *distance) const { return nearest_centroid(centroids, x, distance); }

namespace {

void require_clusters(std::size_t clusters, std::size_t samples) {
    if (clusters > samples) {
        throw std::invalid_argument("Number of clusters (" + std::to_string(clusters) +
                                    ") cannot exceed the number of samples (" + std::to_string(samples) + ")");
    }
}

}  // namespace

void KMeans::fit(const Matrix &X, unsigned threads) {
    require_training_set(X, X.rows);
    require_clusters(clusters, X.rows);
    const std::size_t n = X.rows;
    const std::size_t d = X.cols;
    const std::size_t k = clusters;

    Rng rng = make_rng(has_seed, seed);

    // Work on a copy so a failed fit leaves the previous model intact.
    KMeans next = *this;
    // k-means++ keeps its original implementation so seeded results match 0.1.x.
    next.centroids = seeder == Seeder::PlusPlus ? init_plus_plus(X, k, rng) : seed_centers(X, k, seeder, preset, Distance(), rng);
    next.labels.assign(n, 0);
    next.iterations = 0;

    std::vector<double> dist(n);
    Matrix sums(k, d);
    std::vector<std::size_t> counts(k);

    for (long it = 0; it < max_iterations; ++it) {
        parallel_for(n, threads, [&](std::size_t r) { next.labels[r] = next.predict_one(X.row(r), &dist[r]); });

        std::fill(sums.data.begin(), sums.data.end(), 0.0);
        std::fill(counts.begin(), counts.end(), 0);
        for (std::size_t r = 0; r < n; ++r) {
            const std::size_t c = static_cast<std::size_t>(next.labels[r]);
            ++counts[c];
            double *s = sums.row(c);
            const double *x = X.row(r);
            for (std::size_t j = 0; j < d; ++j) {
                s[j] += x[j];
            }
        }

        double shift = 0.0;
        for (std::size_t c = 0; c < k; ++c) {
            double *center = next.centroids.row(c);
            if (counts[c] == 0) {
                // Re-seed an empty cluster with the point worst served by its centroid.
                const std::size_t far = static_cast<std::size_t>(std::max_element(dist.begin(), dist.end()) - dist.begin());
                shift += squared_distance(center, X.row(far), d);
                std::copy(X.row(far), X.row(far) + d, center);
                dist[far] = 0.0;
                continue;
            }
            const double *s = sums.row(c);
            for (std::size_t j = 0; j < d; ++j) {
                const double updated = s[j] / static_cast<double>(counts[c]);
                const double delta = updated - center[j];
                shift += delta * delta;
                center[j] = updated;
            }
        }

        next.iterations = it + 1;
        if (shift <= tolerance) {
            break;
        }
    }

    // Final assignment so labels and inertia match the returned centroids.
    parallel_for(n, threads, [&](std::size_t r) { next.labels[r] = next.predict_one(X.row(r), &dist[r]); });
    next.inertia = 0.0;
    for (double dr : dist) {
        next.inertia += dr;
    }

    *this = std::move(next);
}

// ---- mini-batch k-means ------------------------------------------------------------------

void MiniBatchKMeans::initialize(const Matrix &X, Rng &rng) {
    // Seed on a random subset of three batches (at least 3k samples) to bound the cost.
    const std::size_t n = X.rows;
    const std::size_t size = std::min(n, std::max(3 * batch_size, 3 * clusters));
    if (size < n) {
        std::vector<std::size_t> order(n);
        for (std::size_t i = 0; i < n; ++i) order[i] = i;
        Matrix subset(size, X.cols);
        for (std::size_t i = 0; i < size; ++i) {
            std::swap(order[i], order[std::uniform_int_distribution<std::size_t>(i, n - 1)(rng)]);
            std::copy(X.row(order[i]), X.row(order[i]) + X.cols, subset.row(i));
        }
        centroids = seed_centers(subset, clusters, seeder, preset, Distance(), rng);
    } else {
        centroids = seed_centers(X, clusters, seeder, preset, Distance(), rng);
    }
    counts.assign(clusters, 0.0);
}

double MiniBatchKMeans::update(const Matrix &X, const std::vector<std::size_t> &batch, unsigned threads) {
    const std::size_t d = X.cols, k = centroids.rows;
    std::vector<int> assigned(batch.size());
    parallel_for(batch.size(), threads, [&](std::size_t b) { assigned[b] = nearest_centroid(centroids, X.row(batch[b])); });
    Matrix sums(k, d);
    std::vector<double> added(k, 0.0);
    for (std::size_t b = 0; b < batch.size(); ++b) {
        const std::size_t c = static_cast<std::size_t>(assigned[b]);
        added[c] += 1.0;
        const double *x = X.row(batch[b]);
        double *s = sums.row(c);
        for (std::size_t j = 0; j < d; ++j) s[j] += x[j];
    }
    // Each centroid becomes the running mean of every sample it has absorbed.
    double shift = 0.0;
    for (std::size_t c = 0; c < k; ++c) {
        if (added[c] == 0.0) continue;
        const double total = counts[c] + added[c];
        double *center = centroids.row(c);
        const double *s = sums.row(c);
        for (std::size_t j = 0; j < d; ++j) {
            const double updated = (center[j] * counts[c] + s[j]) / total;
            shift += (updated - center[j]) * (updated - center[j]);
            center[j] = updated;
        }
        counts[c] = total;
    }
    return shift;
}

void MiniBatchKMeans::assign(const Matrix &X, unsigned threads) {
    labels.assign(X.rows, 0);
    std::vector<double> dist(X.rows);
    parallel_for(X.rows, threads, [&](std::size_t r) { labels[r] = nearest_centroid(centroids, X.row(r), &dist[r]); });
    inertia = 0.0;
    for (double dr : dist) inertia += dr;
}

void MiniBatchKMeans::fit(const Matrix &X, unsigned threads) {
    require_training_set(X, X.rows);
    require_clusters(clusters, X.rows);
    const std::size_t n = X.rows, size = std::min(batch_size, n);
    MiniBatchKMeans next = *this;
    Rng rng = make_rng(has_seed, seed);
    next.initialize(X, rng);
    next.iterations = 0;
    std::vector<std::size_t> order(n), batch;
    for (std::size_t i = 0; i < n; ++i) order[i] = i;
    std::uniform_int_distribution<std::size_t> pick(0, n - 1);
    for (long epoch = 0; epoch < max_iterations; ++epoch) {
        std::shuffle(order.begin(), order.end(), rng);
        double shift = 0.0;
        for (std::size_t start = 0; start < n; start += size) {
            batch.assign(order.begin() + static_cast<std::ptrdiff_t>(start),
                         order.begin() + static_cast<std::ptrdiff_t>(std::min(n, start + size)));
            shift += next.update(X, batch, threads);
        }
        // A centroid that attracted nothing restarts at a random sample.
        for (std::size_t c = 0; c < next.clusters; ++c) {
            if (next.counts[c] == 0.0) {
                const double *x = X.row(pick(rng));
                std::copy(x, x + X.cols, next.centroids.row(c));
                shift = std::numeric_limits<double>::infinity();
            }
        }
        next.iterations = epoch + 1;
        if (shift <= tolerance) {
            break;
        }
    }
    next.assign(X, threads);
    *this = std::move(next);
}

void MiniBatchKMeans::partial_fit(const Matrix &X, unsigned threads) {
    require_training_set(X, X.rows);
    MiniBatchKMeans next = *this;
    if (!next.fitted()) {
        if (clusters > X.rows) {
            throw std::invalid_argument("The first partialFit() batch needs at least as many samples as clusters (" +
                                        std::to_string(clusters) + "), " + std::to_string(X.rows) + " given");
        }
        Rng rng = make_rng(has_seed, seed);
        next.initialize(X, rng);
    }
    std::vector<std::size_t> all(X.rows);
    for (std::size_t i = 0; i < X.rows; ++i) all[i] = i;
    next.update(X, all, threads);
    ++next.iterations;
    next.assign(X, threads);
    *this = std::move(next);
}

void MiniBatchKMeans::save(ArchiveWriter &w) const {
    w.size(clusters);
    w.size(batch_size);
    w.i64(max_iterations);
    w.f64(tolerance);
    w.boolean(has_seed);
    w.u64(seed);
    w.u8(static_cast<std::uint8_t>(seeder));
    w.matrix(preset);
    w.matrix(centroids);
    w.doubles(counts);
    w.ints(labels);
    w.f64(inertia);
    w.i64(iterations);
}

void MiniBatchKMeans::load(ArchiveReader &r) {
    clusters = r.size();
    batch_size = r.size();
    max_iterations = static_cast<long>(r.i64());
    tolerance = r.f64();
    has_seed = r.boolean();
    seed = r.u64();
    const std::uint8_t s = r.u8();
    ArchiveReader::require(s <= static_cast<std::uint8_t>(Seeder::Preset), "bad seeder");
    seeder = static_cast<Seeder>(s);
    preset = r.matrix();
    centroids = r.matrix();
    counts = r.doubles();
    ArchiveReader::require(clusters >= 1 && batch_size >= 1 && max_iterations >= 1 && tolerance >= 0.0,
                           "bad MiniBatchKMeans parameters");
    labels = r.ints_in(0, static_cast<int>(std::min<std::size_t>(clusters, 1u << 30)) - 1);
    inertia = r.f64();
    iterations = static_cast<long>(r.i64());
    ArchiveReader::require(seeder != Seeder::Preset || (preset.rows == clusters && preset.cols > 0), "preset seeder without centroids");
    if (fitted()) {
        ArchiveReader::require(centroids.rows == clusters && centroids.cols > 0 && counts.size() == clusters);
    } else {
        ArchiveReader::require(counts.empty() && labels.empty());
    }
}

}  // namespace mlcore
