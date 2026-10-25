#include "decomposition.hpp"

#include "linalg.hpp"

namespace mlcore {

void LinearProjection::transform(const double *x, double *out) const {
    const std::size_t d = W.rows, k = W.cols;
    std::fill(out, out + k, 0.0);
    for (std::size_t j = 0; j < d; ++j) {
        const double v = mean.empty() ? x[j] : x[j] - mean[j];
        const double *w = W.row(j);
        for (std::size_t c = 0; c < k; ++c) {
            out[c] += v * w[c];
        }
    }
}

void LinearProjection::save(ArchiveWriter &w) const {
    w.doubles(mean);
    w.matrix(W);
}

void LinearProjection::load(ArchiveReader &r) {
    mean = r.doubles();
    W = r.matrix();
    ArchiveReader::require(mean.empty() || mean.size() == W.rows, "inconsistent projection");
}

namespace {

// First k eigenvectors (columns) of a symmetric d x d matrix as a d x k matrix.
Matrix top_vectors(const std::vector<double> &vectors, std::size_t d, std::size_t k) {
    Matrix W(d, k);
    for (std::size_t j = 0; j < d; ++j) {
        for (std::size_t c = 0; c < k; ++c) {
            W(j, c) = vectors[j * d + c];
        }
    }
    return W;
}

void check_dimensions(std::size_t k, std::size_t limit, const char *what) {
    if (k > limit) {
        throw std::invalid_argument(std::string("dimensions (") + std::to_string(k) + ") cannot exceed " + what + " (" +
                                    std::to_string(limit) + ")");
    }
}

}  // namespace

void PCA::fit(const Matrix &X) {
    require_training_set(X, X.rows);
    check_dimensions(dimensions, X.cols, "the number of features");
    const std::vector<double> mean = column_means(X);
    std::vector<double> values, vectors;
    symmetric_eigen(covariance(X, mean), X.cols, values, vectors);
    PCA next = *this;
    next.projection.mean = mean;
    next.projection.W = top_vectors(vectors, X.cols, dimensions);
    next.explained.assign(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(dimensions));
    next.total_variance = 0.0;
    for (double v : values) next.total_variance += std::max(0.0, v);
    *this = std::move(next);
}

void PCA::save(ArchiveWriter &w) const {
    w.size(dimensions);
    projection.save(w);
    w.doubles(explained);
    w.f64(total_variance);
}

void PCA::load(ArchiveReader &r) {
    dimensions = r.size();
    projection.load(r);
    explained = r.doubles();
    total_variance = r.f64();
}

void TruncatedSVD::fit(const Matrix &X) {
    require_training_set(X, X.rows);
    check_dimensions(dimensions, X.cols, "the number of features");
    const std::size_t d = X.cols;
    std::vector<double> gram(d * d, 0.0);
    for (std::size_t r = 0; r < X.rows; ++r) {
        const double *x = X.row(r);
        for (std::size_t i = 0; i < d; ++i) {
            for (std::size_t j = 0; j <= i; ++j) {
                gram[i * d + j] += x[i] * x[j];
            }
        }
    }
    for (std::size_t i = 0; i < d; ++i) {
        for (std::size_t j = 0; j < i; ++j) {
            gram[j * d + i] = gram[i * d + j];
        }
    }
    std::vector<double> values, vectors;
    symmetric_eigen(gram, d, values, vectors);
    TruncatedSVD next = *this;
    next.projection.mean.clear();
    next.projection.W = top_vectors(vectors, d, dimensions);
    next.singular_values.clear();
    next.total = 0.0;
    for (std::size_t c = 0; c < values.size(); ++c) {
        const double s = std::sqrt(std::max(0.0, values[c]));
        next.total += s * s;
        if (c < dimensions) next.singular_values.push_back(s);
    }
    *this = std::move(next);
}

void TruncatedSVD::save(ArchiveWriter &w) const {
    w.size(dimensions);
    projection.save(w);
    w.doubles(singular_values);
    w.f64(total);
}

void TruncatedSVD::load(ArchiveReader &r) {
    dimensions = r.size();
    projection.load(r);
    singular_values = r.doubles();
    total = r.f64();
}

void LDA::fit(const Matrix &X, const std::vector<int> &y, std::size_t K) {
    require_training_set(X, y.size());
    if (K < 2) {
        throw std::invalid_argument("LinearDiscriminantAnalysis needs at least two classes");
    }
    const std::size_t d = X.cols, n = X.rows;
    check_dimensions(dimensions, std::min(d, K - 1), "min(features, classes - 1)");
    const std::vector<double> mean = column_means(X);
    Matrix class_mean(K, d);
    std::vector<double> counts(K, 0.0);
    for (std::size_t r = 0; r < n; ++r) {
        const std::size_t c = static_cast<std::size_t>(y[r]);
        counts[c] += 1.0;
        for (std::size_t j = 0; j < d; ++j) class_mean(c, j) += X(r, j);
    }
    for (std::size_t c = 0; c < K; ++c) {
        for (std::size_t j = 0; j < d; ++j) class_mean(c, j) /= counts[c];
    }
    std::vector<double> sw(d * d, 0.0), sb(d * d, 0.0), diff(d);
    for (std::size_t r = 0; r < n; ++r) {
        const std::size_t c = static_cast<std::size_t>(y[r]);
        for (std::size_t j = 0; j < d; ++j) diff[j] = X(r, j) - class_mean(c, j);
        for (std::size_t a = 0; a < d; ++a) {
            for (std::size_t b = 0; b < d; ++b) sw[a * d + b] += diff[a] * diff[b];
        }
    }
    for (std::size_t c = 0; c < K; ++c) {
        for (std::size_t j = 0; j < d; ++j) diff[j] = class_mean(c, j) - mean[j];
        for (std::size_t a = 0; a < d; ++a) {
            for (std::size_t b = 0; b < d; ++b) sb[a * d + b] += counts[c] * diff[a] * diff[b];
        }
    }
    // Regularize Sw slightly so it is positive definite, then whiten: Sw = L L^T.
    double trace = 0.0;
    for (std::size_t i = 0; i < d; ++i) trace += sw[i * d + i];
    const double ridge = std::max(trace / static_cast<double>(d), 1.0) * 1e-6;
    for (std::size_t i = 0; i < d; ++i) sw[i * d + i] += ridge;
    std::vector<double> L(d * d, 0.0);
    for (std::size_t j = 0; j < d; ++j) {
        double s = sw[j * d + j];
        for (std::size_t k = 0; k < j; ++k) s -= L[j * d + k] * L[j * d + k];
        if (!(s > 0.0)) throw std::runtime_error("Within-class scatter matrix is singular");
        L[j * d + j] = std::sqrt(s);
        for (std::size_t i = j + 1; i < d; ++i) {
            double t = sw[i * d + j];
            for (std::size_t k = 0; k < j; ++k) t -= L[i * d + k] * L[j * d + k];
            L[i * d + j] = t / L[j * d + j];
        }
    }
    // Linv = L^-1 (lower triangular).
    std::vector<double> Linv(d * d, 0.0);
    for (std::size_t col = 0; col < d; ++col) {
        for (std::size_t i = 0; i < d; ++i) {
            double s = i == col ? 1.0 : 0.0;
            for (std::size_t k = 0; k < i; ++k) s -= L[i * d + k] * Linv[k * d + col];
            Linv[i * d + col] = s / L[i * d + i];
        }
    }
    // M = Linv Sb Linv^T
    std::vector<double> tmp(d * d, 0.0), M(d * d, 0.0);
    for (std::size_t i = 0; i < d; ++i)
        for (std::size_t j = 0; j < d; ++j)
            for (std::size_t k = 0; k < d; ++k) tmp[i * d + j] += Linv[i * d + k] * sb[k * d + j];
    for (std::size_t i = 0; i < d; ++i)
        for (std::size_t j = 0; j < d; ++j)
            for (std::size_t k = 0; k < d; ++k) M[i * d + j] += tmp[i * d + k] * Linv[j * d + k];
    std::vector<double> values, vectors;
    symmetric_eigen(M, d, values, vectors);
    // W = Linv^T U
    LDA next = *this;
    next.projection.mean = mean;
    next.projection.W = Matrix(d, dimensions);
    for (std::size_t j = 0; j < d; ++j) {
        for (std::size_t c = 0; c < dimensions; ++c) {
            double s = 0.0;
            for (std::size_t k = 0; k < d; ++k) s += Linv[k * d + j] * vectors[k * d + c];
            next.projection.W(j, c) = s;
        }
    }
    next.explained.assign(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(dimensions));
    *this = std::move(next);
}

void LDA::save(ArchiveWriter &w) const {
    w.size(dimensions);
    projection.save(w);
    w.doubles(explained);
}

void LDA::load(ArchiveReader &r) {
    dimensions = r.size();
    projection.load(r);
    explained = r.doubles();
}

void RandomProjector::fit(const Matrix &X) {
    require_training_set(X, X.rows);
    const std::size_t d = X.cols, k = dimensions;
    Rng rng = make_rng(has_seed, seed);
    Matrix W(d, k);
    if (sparse) {
        const double s = sparsity, scale = std::sqrt(s / static_cast<double>(k));
        std::uniform_real_distribution<double> unit(0.0, 1.0);
        for (double &v : W.data) {
            const double u = unit(rng);
            v = u < 0.5 / s ? scale : (u < 1.0 / s ? -scale : 0.0);
        }
    } else {
        std::normal_distribution<double> normal(0.0, 1.0 / std::sqrt(static_cast<double>(k)));
        for (double &v : W.data) v = normal(rng);
    }
    projection.mean.clear();
    projection.W = std::move(W);
}

std::size_t RandomProjector::min_dimensions(std::size_t samples, double distortion) {
    const double e = distortion;
    return static_cast<std::size_t>(std::ceil(4.0 * std::log(static_cast<double>(samples)) / (e * e / 2.0 - e * e * e / 3.0)));
}

void RandomProjector::save(ArchiveWriter &w) const {
    w.size(dimensions);
    w.boolean(sparse);
    w.f64(sparsity);
    w.boolean(has_seed);
    w.u64(seed);
    projection.save(w);
}

void RandomProjector::load(ArchiveReader &r) {
    dimensions = r.size();
    sparse = r.boolean();
    sparsity = r.f64();
    has_seed = r.boolean();
    seed = r.u64();
    projection.load(r);
}

// ---- t-SNE ----------------------------------------------------------------------------------------

void TSNEParams::save(ArchiveWriter &w) const {
    w.size(dimensions);
    w.f64(rate);
    w.f64(perplexity);
    w.f64(exaggeration);
    w.size(epochs);
    w.f64(min_gradient);
    w.boolean(has_seed);
    w.u64(seed);
}

void TSNEParams::load(ArchiveReader &r) {
    dimensions = r.size();
    rate = r.f64();
    perplexity = r.f64();
    exaggeration = r.f64();
    epochs = r.size();
    min_gradient = r.f64();
    has_seed = r.boolean();
    seed = r.u64();
    ArchiveReader::require(dimensions >= 1 && perplexity > 0.0, "bad t-SNE parameters");
}

Matrix tsne(const Matrix &X, const TSNEParams &p, unsigned threads, std::vector<double> *losses) {
    const std::size_t n = X.rows, k = p.dimensions;
    if (n < 4) {
        throw std::invalid_argument("t-SNE needs at least 4 samples");
    }
    if (p.perplexity >= static_cast<double>(n - 1)) {
        throw std::invalid_argument("perplexity must be less than the number of samples - 1");
    }
    // Pairwise squared distances.
    std::vector<double> D(n * n);
    parallel_for(n, threads, [&](std::size_t i) {
        for (std::size_t j = 0; j < n; ++j) D[i * n + j] = squared_distance(X.row(i), X.row(j), X.cols);
    });
    // Conditional probabilities with a per-point precision matching the perplexity.
    std::vector<double> P(n * n, 0.0);
    const double target = std::log(p.perplexity);
    parallel_for(n, threads, [&](std::size_t i) {
        double beta = 1.0, lo = 0.0, hi = std::numeric_limits<double>::infinity();
        double *row = &P[i * n];
        for (int step = 0; step < 64; ++step) {
            double sum = 0.0;
            for (std::size_t j = 0; j < n; ++j) {
                row[j] = j == i ? 0.0 : std::exp(-D[i * n + j] * beta);
                sum += row[j];
            }
            sum = std::max(sum, 1e-300);
            double entropy = 0.0;
            for (std::size_t j = 0; j < n; ++j) {
                row[j] /= sum;
                if (row[j] > 1e-300) entropy -= row[j] * std::log(row[j]);
            }
            const double diff = entropy - target;
            if (std::fabs(diff) < 1e-5) break;
            if (diff > 0) {
                lo = beta;
                beta = std::isinf(hi) ? beta * 2.0 : (beta + hi) / 2.0;
            } else {
                hi = beta;
                beta = (beta + lo) / 2.0;
            }
        }
    });
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            const double v = std::max((P[i * n + j] + P[j * n + i]) / (2.0 * static_cast<double>(n)), 1e-12);
            P[i * n + j] = P[j * n + i] = v;
        }
    }

    Rng rng = make_rng(p.has_seed, p.seed);
    std::normal_distribution<double> normal(0.0, 1e-4);
    Matrix Y(n, k), update(n, k), gains(n, k, 1.0), grad(n, k);
    for (double &v : Y.data) v = normal(rng);
    std::vector<double> num(n * n);
    const std::size_t exaggerated = std::min<std::size_t>(250, p.epochs / 4);

    for (std::size_t epoch = 0; epoch < p.epochs; ++epoch) {
        const double ex = epoch < exaggerated ? p.exaggeration : 1.0;
        const double momentum = epoch < exaggerated ? 0.5 : 0.8;
        std::vector<double> partial(n, 0.0);
        parallel_for(n, threads, [&](std::size_t i) {
            for (std::size_t j = 0; j < n; ++j) {
                const double q = j == i ? 0.0 : 1.0 / (1.0 + squared_distance(Y.row(i), Y.row(j), k));
                num[i * n + j] = q;
                partial[i] += q;
            }
        });
        double sum_q = 0.0;
        for (double v : partial) sum_q += v;
        sum_q = std::max(sum_q, 1e-300);
        parallel_for(n, threads, [&](std::size_t i) {
            double *g = grad.row(i);
            std::fill(g, g + k, 0.0);
            for (std::size_t j = 0; j < n; ++j) {
                const double mult = (ex * P[i * n + j] - num[i * n + j] / sum_q) * num[i * n + j];
                for (std::size_t c = 0; c < k; ++c) g[c] += 4.0 * mult * (Y(i, c) - Y(j, c));
            }
        });
        double max_grad = 0.0;
        for (std::size_t idx = 0; idx < Y.data.size(); ++idx) {
            const double g = grad.data[idx];
            max_grad = std::max(max_grad, std::fabs(g));
            double &gain = gains.data[idx];
            gain = (g > 0) != (update.data[idx] > 0) ? gain + 0.2 : gain * 0.8;
            gain = std::max(gain, 0.01);
            update.data[idx] = momentum * update.data[idx] - p.rate * gain * g;
            Y.data[idx] += update.data[idx];
        }
        // Keep the embedding centered.
        for (std::size_t c = 0; c < k; ++c) {
            double mean = 0.0;
            for (std::size_t i = 0; i < n; ++i) mean += Y(i, c);
            mean /= static_cast<double>(n);
            for (std::size_t i = 0; i < n; ++i) Y(i, c) -= mean;
        }
        if (losses && (epoch % 10 == 0 || epoch + 1 == p.epochs)) {
            double kl = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                for (std::size_t j = 0; j < n; ++j) {
                    if (i != j) kl += P[i * n + j] * std::log(P[i * n + j] / std::max(num[i * n + j] / sum_q, 1e-300));
                }
            }
            losses->push_back(kl);
        }
        if (epoch >= exaggerated && max_grad < p.min_gradient) {
            break;
        }
    }
    return Y;
}

}  // namespace mlcore
