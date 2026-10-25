#pragma once

#include "common.hpp"

namespace mlcore {

// Solves A x = b for a symmetric positive-definite n x n matrix A (row-major).
// A is overwritten by its Cholesky factor and b by the solution. Returns false
// when A is not positive definite.
bool cholesky_solve(std::vector<double> &a, std::vector<double> &b, std::size_t n);

// Eigen-decomposition of a symmetric n x n matrix (cyclic Jacobi). Eigenvalues
// are returned in descending order; eigenvectors are the columns of `vectors`
// (row-major n x n), in the same order.
void symmetric_eigen(std::vector<double> a, std::size_t n, std::vector<double> &values, std::vector<double> &vectors);

// Column means of X.
std::vector<double> column_means(const Matrix &X);

// Covariance (population) of the columns of X around `means`.
std::vector<double> covariance(const Matrix &X, const std::vector<double> &means);

}  // namespace mlcore
