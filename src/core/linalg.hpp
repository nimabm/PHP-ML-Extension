#pragma once

#include "common.hpp"

namespace mlcore {

// Solves A x = b for a symmetric positive-definite n x n matrix A (row-major).
// A is overwritten by its Cholesky factor and b by the solution. Returns false
// when A is not positive definite.
bool cholesky_solve(std::vector<double> &a, std::vector<double> &b, std::size_t n);

// Eigen-decomposition of a symmetric n x n matrix. Eigenvalues are returned in
// descending order; eigenvectors are the columns of `vectors` (row-major n x n),
// in the same order, each with its largest-magnitude component positive.
void symmetric_eigen(std::vector<double> a, std::size_t n, std::vector<double> &values, std::vector<double> &vectors);

// The k largest eigenpairs of a symmetric n x n matrix (row-major; `a` is
// overwritten). `values` is descending and `vectors` is n x k (column c belongs
// to values[c], sign-normalized as above). Uses Householder tridiagonalization,
// bisection and inverse iteration: O(n^3) but much faster than a full
// decomposition when k is small, and exact multiplicities are handled.
void top_eigen(std::vector<double> &a, std::size_t n, std::size_t k, std::vector<double> &values, Matrix &vectors,
               unsigned threads = 1);

// Column means of X.
std::vector<double> column_means(const Matrix &X);

// Covariance (population) of the columns of X around `means`.
std::vector<double> covariance(const Matrix &X, const std::vector<double> &means);

}  // namespace mlcore
