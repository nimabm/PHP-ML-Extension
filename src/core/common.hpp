#pragma once

// Plain C++17 building blocks shared by every model. Nothing in src/core
// depends on PHP, so the algorithms can be reused or unit-tested on their own.

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace mlcore {

using Rng = std::mt19937_64;

// A generator seeded with `seed`, or from std::random_device when no seed is given.
inline Rng make_rng(bool has_seed, std::uint64_t seed) {
    if (has_seed) {
        return Rng(seed);
    }
    std::random_device device;
    return Rng((static_cast<std::uint64_t>(device()) << 32) ^ device());
}

// Runs fn(i) for i in [0, n) on up to `threads` threads (1 = inline). The first
// exception thrown by any task is rethrown in the caller after all threads join.
template <typename F>
void parallel_for(std::size_t n, unsigned threads, F &&fn) {
    if (threads <= 1 || n <= 1) {
        for (std::size_t i = 0; i < n; ++i) {
            fn(i);
        }
        return;
    }
    const unsigned workers = static_cast<unsigned>(std::min<std::size_t>(threads, n));
    std::atomic<std::size_t> next{0};
    std::exception_ptr error;
    std::mutex error_mutex;
    auto worker = [&] {
        for (std::size_t i = next++; i < n; i = next++) {
            try {
                fn(i);
            } catch (...) {
                std::lock_guard<std::mutex> lock(error_mutex);
                if (!error) {
                    error = std::current_exception();
                }
                next = n;  // stop handing out work
            }
        }
    };
    std::vector<std::thread> pool;
    pool.reserve(workers - 1);
    for (unsigned t = 1; t < workers; ++t) {
        pool.emplace_back(worker);
    }
    worker();
    for (std::thread &t : pool) {
        t.join();
    }
    if (error) {
        std::rethrow_exception(error);
    }
}

// Median of `values` (reorders them). `values` must not be empty.
inline double median(std::vector<double> &values) {
    const std::size_t mid = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(mid), values.end());
    double m = values[mid];
    if (values.size() % 2 == 0) {
        m = (m + *std::max_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(mid))) / 2.0;
    }
    return m;
}

// Linear-interpolated quantile q in [0, 1] (reorders them). `values` must not be empty.
inline double quantile(std::vector<double> &values, double q) {
    std::sort(values.begin(), values.end());
    const double pos = q * static_cast<double>(values.size() - 1);
    const std::size_t lo = static_cast<std::size_t>(std::floor(pos));
    const std::size_t hi = std::min(lo + 1, values.size() - 1);
    return values[lo] + (values[hi] - values[lo]) * (pos - static_cast<double>(lo));
}

inline std::string describe(std::size_t value) { return std::to_string(value); }

// Thrown when a model is used before it has been fitted (or restored).
class NotFittedError : public std::logic_error {
public:
    using std::logic_error::logic_error;
};

// Dense row-major matrix of doubles: rows are samples, columns are features.
struct Matrix {
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::vector<double> data;

    Matrix() = default;
    Matrix(std::size_t r, std::size_t c, double fill = 0.0) : rows(r), cols(c), data(r * c, fill) {}

    double *row(std::size_t r) { return data.data() + r * cols; }
    const double *row(std::size_t r) const { return data.data() + r * cols; }
    double &operator()(std::size_t r, std::size_t c) { return data[r * cols + c]; }
    double operator()(std::size_t r, std::size_t c) const { return data[r * cols + c]; }
};

inline double dot(const double *a, const double *b, std::size_t n) {
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

inline double squared_distance(const double *a, const double *b, std::size_t n) {
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

inline double manhattan_distance(const double *a, const double *b, std::size_t n) {
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        sum += std::fabs(a[i] - b[i]);
    }
    return sum;
}

// Turns scores into probabilities in place (numerically stable).
inline void softmax(double *v, std::size_t n) {
    double max = v[0];
    for (std::size_t i = 1; i < n; ++i) {
        if (v[i] > max) {
            max = v[i];
        }
    }
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        v[i] = std::exp(v[i] - max);
        sum += v[i];
    }
    for (std::size_t i = 0; i < n; ++i) {
        v[i] /= sum;
    }
}

inline std::size_t argmax(const double *v, std::size_t n) {
    std::size_t best = 0;
    for (std::size_t i = 1; i < n; ++i) {
        if (v[i] > v[best]) {
            best = i;
        }
    }
    return best;
}

// Validates a training set: at least one sample/feature and one target per sample.
inline void require_training_set(const Matrix &X, std::size_t targets) {
    if (X.rows == 0 || X.cols == 0) {
        throw std::invalid_argument("Training set must contain at least one sample with at least one feature");
    }
    if (targets != X.rows) {
        throw std::invalid_argument("Number of samples (" + std::to_string(X.rows) +
                                    ") does not match number of targets (" + std::to_string(targets) + ")");
    }
}

}  // namespace mlcore
