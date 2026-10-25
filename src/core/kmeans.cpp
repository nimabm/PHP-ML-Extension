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

void KMeans::probabilities(const double *x, double *out) const {
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

int KMeans::predict_one(const double *x, double *distance) const {
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

void KMeans::fit(const Matrix &X) {
    require_training_set(X, X.rows);
    if (clusters > X.rows) {
        throw std::invalid_argument("Number of clusters (" + std::to_string(clusters) +
                                    ") cannot exceed the number of samples (" + std::to_string(X.rows) + ")");
    }
    const std::size_t n = X.rows;
    const std::size_t d = X.cols;
    const std::size_t k = clusters;

    std::mt19937_64 rng(has_seed ? seed : static_cast<std::uint64_t>(std::random_device{}()));

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
        for (std::size_t r = 0; r < n; ++r) {
            next.labels[r] = next.predict_one(X.row(r), &dist[r]);
        }

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
    next.inertia = 0.0;
    for (std::size_t r = 0; r < n; ++r) {
        double dr = 0.0;
        next.labels[r] = next.predict_one(X.row(r), &dr);
        next.inertia += dr;
    }

    *this = std::move(next);
}

}  // namespace mlcore
