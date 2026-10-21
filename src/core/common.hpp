#pragma once

// Plain C++17 building blocks shared by every model. Nothing in src/core
// depends on PHP, so the algorithms can be reused or unit-tested on their own.

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace mlcore {

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
