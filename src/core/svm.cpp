#include "svm.hpp"

#include <functional>
#include <list>
#include <unordered_map>

namespace mlcore {

// ---- kernels --------------------------------------------------------------------------

Kernel::Type Kernel::parse(const std::string &name) {
    if (name == "linear") return Type::Linear;
    if (name == "polynomial") return Type::Polynomial;
    if (name == "rbf") return Type::RBF;
    if (name == "sigmoid") return Type::Sigmoid;
    throw std::invalid_argument("Unknown kernel \"" + name + "\"; expected linear, polynomial, rbf or sigmoid");
}

std::string Kernel::name(Type t) {
    switch (t) {
        case Type::Linear: return "linear";
        case Type::Polynomial: return "polynomial";
        case Type::RBF: return "rbf";
        case Type::Sigmoid: return "sigmoid";
    }
    return "rbf";
}

double Kernel::operator()(const double *a, const double *b, std::size_t n) const {
    switch (type) {
        case Type::Linear: return dot(a, b, n);
        case Type::Polynomial: return std::pow(gamma * dot(a, b, n) + coef0, degree);
        case Type::RBF: return std::exp(-gamma * squared_distance(a, b, n));
        case Type::Sigmoid: return std::tanh(gamma * dot(a, b, n) + coef0);
    }
    return 0.0;
}

void Kernel::save(ArchiveWriter &w) const {
    w.u8(static_cast<std::uint8_t>(type));
    w.f64(gamma);
    w.f64(degree);
    w.f64(coef0);
}

void Kernel::load(ArchiveReader &r) {
    const std::uint8_t t = r.u8();
    ArchiveReader::require(t <= 3, "bad kernel");
    type = static_cast<Type>(t);
    gamma = r.f64();
    degree = r.f64();
    coef0 = r.f64();
}

double SVMFunction::decision(const Kernel &k, const double *x) const {
    double s = -rho;
    for (std::size_t i = 0; i < support.rows; ++i) {
        s += coef[i] * k(support.row(i), x, support.cols);
    }
    return s;
}

void SVMFunction::save(ArchiveWriter &w) const {
    w.matrix(support);
    w.doubles(coef);
    w.f64(rho);
}

void SVMFunction::load(ArchiveReader &r, std::size_t features) {
    support = r.matrix();
    coef = r.doubles(support.rows);
    rho = r.f64();
    ArchiveReader::require(support.rows == 0 || support.cols == features, "inconsistent support vectors");
}

void SVMParams::save(ArchiveWriter &w) const {
    kernel.save(w);
    w.f64(c);
    w.f64(epsilon);
    w.f64(nu);
    w.f64(tolerance);
    w.size(cache_mb);
}

void SVMParams::load(ArchiveReader &r) {
    kernel.load(r);
    c = r.f64();
    epsilon = r.f64();
    nu = r.f64();
    tolerance = r.f64();
    cache_mb = r.size();
}

// ---- SMO solver (after libsvm's Solver, WSS2, without shrinking) ------------------------

namespace {

constexpr double kTau = 1e-12;

// Rows of the signed kernel matrix Q_ij = y_i y_j K(x_{idx_i}, x_{idx_j}), cached LRU.
class QMatrix {
public:
    QMatrix(const Matrix &X, std::vector<std::size_t> index, std::vector<signed char> y, const Kernel &k, std::size_t cache_mb)
        : X_(X), index_(std::move(index)), y_(std::move(y)), k_(k) {
        const std::size_t l = index_.size();
        capacity_ = std::max<std::size_t>(2, cache_mb * 1024 * 1024 / (sizeof(double) * std::max<std::size_t>(l, 1)));
        diag_.resize(l);
        for (std::size_t i = 0; i < l; ++i) {
            diag_[i] = k_(X_.row(index_[i]), X_.row(index_[i]), X_.cols);
        }
    }

    const std::vector<double> &row(std::size_t i) {
        auto it = cache_.find(i);
        if (it != cache_.end()) {
            order_.splice(order_.begin(), order_, it->second.second);
            return it->second.first;
        }
        if (cache_.size() >= capacity_) {
            cache_.erase(order_.back());
            order_.pop_back();
        }
        const std::size_t l = index_.size();
        std::vector<double> values(l);
        const double *xi = X_.row(index_[i]);
        for (std::size_t j = 0; j < l; ++j) {
            values[j] = static_cast<double>(y_[i] * y_[j]) * k_(xi, X_.row(index_[j]), X_.cols);
        }
        order_.push_front(i);
        auto &entry = cache_[i];
        entry.first = std::move(values);
        entry.second = order_.begin();
        return entry.first;
    }

    double diag(std::size_t i) const { return diag_[i]; }

private:
    const Matrix &X_;
    std::vector<std::size_t> index_;
    std::vector<signed char> y_;
    const Kernel &k_;
    std::vector<double> diag_;
    std::size_t capacity_;
    std::list<std::size_t> order_;
    std::unordered_map<std::size_t, std::pair<std::vector<double>, std::list<std::size_t>::iterator>> cache_;
};

struct Solution {
    std::vector<double> alpha;
    double rho = 0.0;
};

Solution solve(QMatrix &Q, const std::vector<double> &p, const std::vector<signed char> &y, std::vector<double> alpha,
               const std::vector<double> &C, double eps) {
    const std::size_t l = p.size();
    auto upper = [&](std::size_t i) { return alpha[i] >= C[i]; };
    auto lower = [&](std::size_t i) { return alpha[i] <= 0.0; };

    std::vector<double> G = p;
    for (std::size_t i = 0; i < l; ++i) {
        if (alpha[i] != 0.0) {
            const std::vector<double> &Qi = Q.row(i);
            for (std::size_t j = 0; j < l; ++j) G[j] += alpha[i] * Qi[j];
        }
    }

    const std::size_t max_iter = std::max<std::size_t>(10000000, l > SIZE_MAX / 100 ? SIZE_MAX : 100 * l);
    for (std::size_t iter = 0; iter < max_iter; ++iter) {
        // Working set selection (second-order information).
        double Gmax = -std::numeric_limits<double>::infinity(), Gmax2 = Gmax;
        std::ptrdiff_t gi = -1, gj = -1;
        for (std::size_t t = 0; t < l; ++t) {
            if (y[t] == 1) {
                if (!upper(t) && -G[t] >= Gmax) { Gmax = -G[t]; gi = static_cast<std::ptrdiff_t>(t); }
            } else {
                if (!lower(t) && G[t] >= Gmax) { Gmax = G[t]; gi = static_cast<std::ptrdiff_t>(t); }
            }
        }
        if (gi < 0) break;
        const std::size_t i = static_cast<std::size_t>(gi);
        const std::vector<double> &Qi = Q.row(i);
        double obj_min = std::numeric_limits<double>::infinity();
        for (std::size_t j = 0; j < l; ++j) {
            if (y[j] == 1) {
                if (!lower(j)) {
                    const double grad_diff = Gmax + G[j];
                    Gmax2 = std::max(Gmax2, G[j]);
                    if (grad_diff > 0) {
                        const double quad = Q.diag(i) + Q.diag(j) - 2.0 * y[i] * Qi[j];
                        const double obj = -(grad_diff * grad_diff) / (quad > 0 ? quad : kTau);
                        if (obj <= obj_min) { gj = static_cast<std::ptrdiff_t>(j); obj_min = obj; }
                    }
                }
            } else {
                if (!upper(j)) {
                    const double grad_diff = Gmax - G[j];
                    Gmax2 = std::max(Gmax2, -G[j]);
                    if (grad_diff > 0) {
                        const double quad = Q.diag(i) + Q.diag(j) + 2.0 * y[i] * Qi[j];
                        const double obj = -(grad_diff * grad_diff) / (quad > 0 ? quad : kTau);
                        if (obj <= obj_min) { gj = static_cast<std::ptrdiff_t>(j); obj_min = obj; }
                    }
                }
            }
        }
        if (Gmax + Gmax2 < eps || gj < 0) break;
        const std::size_t j = static_cast<std::size_t>(gj);

        const std::vector<double> Qi_copy = Qi;  // Q.row(j) may evict row i
        const std::vector<double> &Qj = Q.row(j);
        const double Ci = C[i], Cj = C[j];
        const double old_i = alpha[i], old_j = alpha[j];
        if (y[i] != y[j]) {
            double quad = Q.diag(i) + Q.diag(j) + 2.0 * Qi_copy[j];
            if (quad <= 0) quad = kTau;
            const double delta = (-G[i] - G[j]) / quad;
            const double diff = alpha[i] - alpha[j];
            alpha[i] += delta;
            alpha[j] += delta;
            if (diff > 0) {
                if (alpha[j] < 0) { alpha[j] = 0; alpha[i] = diff; }
            } else {
                if (alpha[i] < 0) { alpha[i] = 0; alpha[j] = -diff; }
            }
            if (diff > Ci - Cj) {
                if (alpha[i] > Ci) { alpha[i] = Ci; alpha[j] = Ci - diff; }
            } else {
                if (alpha[j] > Cj) { alpha[j] = Cj; alpha[i] = Cj + diff; }
            }
        } else {
            double quad = Q.diag(i) + Q.diag(j) - 2.0 * Qi_copy[j];
            if (quad <= 0) quad = kTau;
            const double delta = (G[i] - G[j]) / quad;
            const double sum = alpha[i] + alpha[j];
            alpha[i] -= delta;
            alpha[j] += delta;
            if (sum > Ci) {
                if (alpha[i] > Ci) { alpha[i] = Ci; alpha[j] = sum - Ci; }
            } else {
                if (alpha[j] < 0) { alpha[j] = 0; alpha[i] = sum; }
            }
            if (sum > Cj) {
                if (alpha[j] > Cj) { alpha[j] = Cj; alpha[i] = sum - Cj; }
            } else {
                if (alpha[i] < 0) { alpha[i] = 0; alpha[j] = sum; }
            }
        }
        const double di = alpha[i] - old_i, dj = alpha[j] - old_j;
        for (std::size_t k = 0; k < l; ++k) {
            G[k] += Qi_copy[k] * di + Qj[k] * dj;
        }
    }

    // rho
    double ub = std::numeric_limits<double>::infinity(), lb = -ub, sum_free = 0.0;
    std::size_t free = 0;
    for (std::size_t i = 0; i < l; ++i) {
        const double yG = y[i] * G[i];
        if (upper(i)) {
            if (y[i] == -1) ub = std::min(ub, yG); else lb = std::max(lb, yG);
        } else if (lower(i)) {
            if (y[i] == 1) ub = std::min(ub, yG); else lb = std::max(lb, yG);
        } else {
            ++free;
            sum_free += yG;
        }
    }
    Solution s;
    s.rho = free > 0 ? sum_free / static_cast<double>(free) : (ub + lb) / 2.0;
    s.alpha = std::move(alpha);
    return s;
}

// Keeps the samples with non-zero coefficients as support vectors.
SVMFunction make_function(const Matrix &X, const std::vector<std::size_t> &rows, const std::vector<double> &coef, double rho) {
    SVMFunction f;
    std::vector<std::size_t> keep;
    for (std::size_t i = 0; i < coef.size(); ++i) {
        if (std::fabs(coef[i]) > 1e-12) keep.push_back(i);
    }
    f.support = Matrix(keep.size(), X.cols);
    for (std::size_t s = 0; s < keep.size(); ++s) {
        std::copy(X.row(rows[keep[s]]), X.row(rows[keep[s]]) + X.cols, f.support.row(s));
        f.coef.push_back(coef[keep[s]]);
    }
    f.rho = rho;
    return f;
}

Kernel resolved(const Kernel &k, std::size_t features) {
    Kernel out = k;
    if (out.gamma <= 0.0) out.gamma = 1.0 / static_cast<double>(features);
    return out;
}

}  // namespace

// ---- SVC -------------------------------------------------------------------------------------

void SVC::fit(const Matrix &X, const std::vector<int> &y, std::size_t K, unsigned threads) {
    require_training_set(X, y.size());
    if (K < 2) {
        throw std::invalid_argument("SVC needs at least two distinct classes");
    }
    SVC next = *this;
    next.params.kernel = resolved(params.kernel, X.cols);
    std::vector<std::pair<std::size_t, std::size_t>> pairs;
    for (std::size_t a = 0; a < K; ++a) {
        for (std::size_t b = a + 1; b < K; ++b) pairs.emplace_back(a, b);
    }
    std::vector<SVMFunction> functions(pairs.size());
    parallel_for(pairs.size(), threads, [&](std::size_t p) {
        const auto [a, b] = pairs[p];
        std::vector<std::size_t> rows;
        std::vector<signed char> labels;
        for (std::size_t i = 0; i < X.rows; ++i) {
            if (y[i] == static_cast<int>(a) || y[i] == static_cast<int>(b)) {
                rows.push_back(i);
                labels.push_back(y[i] == static_cast<int>(a) ? 1 : -1);
            }
        }
        const std::size_t l = rows.size();
        QMatrix Q(X, rows, labels, next.params.kernel, std::max<std::size_t>(1, params.cache_mb / std::max<std::size_t>(1, threads)));
        Solution s = solve(Q, std::vector<double>(l, -1.0), labels, std::vector<double>(l, 0.0), std::vector<double>(l, params.c),
                           params.tolerance);
        std::vector<double> coef(l);
        for (std::size_t i = 0; i < l; ++i) coef[i] = s.alpha[i] * labels[i];
        functions[p] = make_function(X, rows, coef, s.rho);
    });
    next.functions = std::move(functions);
    next.classes = K;
    next.features = X.cols;
    *this = std::move(next);
}

int SVC::predict_one(const double *x) const {
    std::vector<int> votes(classes, 0);
    std::size_t p = 0;
    for (std::size_t a = 0; a < classes; ++a) {
        for (std::size_t b = a + 1; b < classes; ++b, ++p) {
            ++votes[functions[p].decision(params.kernel, x) > 0 ? a : b];
        }
    }
    return static_cast<int>(std::max_element(votes.begin(), votes.end()) - votes.begin());
}

void SVC::save(ArchiveWriter &w) const {
    params.save(w);
    w.size(classes);
    w.size(features);
    w.size(functions.size());
    for (const SVMFunction &f : functions) f.save(w);
}

void SVC::load(ArchiveReader &r) {
    params.load(r);
    classes = r.size(1u << 16);
    features = r.size(1u << 26);
    functions.assign(r.count(8), {});
    for (SVMFunction &f : functions) f.load(r, features);
    ArchiveReader::require(functions.empty() || functions.size() == classes * (classes - 1) / 2, "inconsistent SVC");
}

// ---- SVR --------------------------------------------------------------------------------------

void SVR::fit(const Matrix &X, const std::vector<double> &y) {
    require_training_set(X, y.size());
    const std::size_t n = X.rows;
    SVR next = *this;
    next.params.kernel = resolved(params.kernel, X.cols);
    std::vector<std::size_t> rows(2 * n);
    std::vector<signed char> labels(2 * n);
    std::vector<double> p(2 * n);
    for (std::size_t i = 0; i < n; ++i) {
        rows[i] = rows[i + n] = i;
        labels[i] = 1;
        labels[i + n] = -1;
        p[i] = params.epsilon - y[i];
        p[i + n] = params.epsilon + y[i];
    }
    QMatrix Q(X, rows, labels, next.params.kernel, params.cache_mb);
    Solution s = solve(Q, p, labels, std::vector<double>(2 * n, 0.0), std::vector<double>(2 * n, params.c), params.tolerance);
    std::vector<double> coef(n);
    std::vector<std::size_t> base(n);
    for (std::size_t i = 0; i < n; ++i) {
        coef[i] = s.alpha[i] - s.alpha[i + n];
        base[i] = i;
    }
    next.function = make_function(X, base, coef, s.rho);
    next.features = X.cols;
    *this = std::move(next);
}

void SVR::save(ArchiveWriter &w) const {
    params.save(w);
    w.size(features);
    function.save(w);
}

void SVR::load(ArchiveReader &r) {
    params.load(r);
    features = r.size(1u << 26);
    function.load(r, features);
}

// ---- OneClassSVM ---------------------------------------------------------------------------

void OneClassSVM::fit(const Matrix &X) {
    require_training_set(X, X.rows);
    const std::size_t n = X.rows;
    OneClassSVM next = *this;
    next.params.kernel = resolved(params.kernel, X.cols);
    std::vector<std::size_t> rows(n);
    for (std::size_t i = 0; i < n; ++i) rows[i] = i;
    const std::vector<signed char> labels(n, 1);
    // Feasible start: sum(alpha) = nu * n with 0 <= alpha <= 1.
    std::vector<double> alpha(n, 0.0);
    const double total = params.nu * static_cast<double>(n);
    const std::size_t whole = static_cast<std::size_t>(total);
    for (std::size_t i = 0; i < whole && i < n; ++i) alpha[i] = 1.0;
    if (whole < n) alpha[whole] = total - static_cast<double>(whole);
    QMatrix Q(X, rows, labels, next.params.kernel, params.cache_mb);
    Solution s = solve(Q, std::vector<double>(n, 0.0), labels, alpha, std::vector<double>(n, 1.0), params.tolerance);
    next.function = make_function(X, rows, s.alpha, s.rho);
    next.features = X.cols;
    *this = std::move(next);
}

void OneClassSVM::save(ArchiveWriter &w) const {
    params.save(w);
    w.size(features);
    function.save(w);
}

void OneClassSVM::load(ArchiveReader &r) {
    params.load(r);
    features = r.size(1u << 26);
    function.load(r, features);
}

}  // namespace mlcore
