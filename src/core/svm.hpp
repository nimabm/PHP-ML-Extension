#pragma once

#include "archive.hpp"
#include "common.hpp"

namespace mlcore {

struct Kernel {
    enum class Type : std::uint8_t { Linear, Polynomial, RBF, Sigmoid };
    Type type = Type::RBF;
    double gamma = 0.0;  // 0 = 1 / features (resolved at fit time)
    double degree = 3.0;
    double coef0 = 0.0;

    static Type parse(const std::string &name);
    static std::string name(Type t);
    double operator()(const double *a, const double *b, std::size_t n) const;
    void save(ArchiveWriter &w) const;
    void load(ArchiveReader &r);
};

// One decision function: f(x) = sum_i coef_i K(sv_i, x) - rho.
struct SVMFunction {
    Matrix support;
    std::vector<double> coef;
    double rho = 0.0;

    double decision(const Kernel &k, const double *x) const;
    void save(ArchiveWriter &w) const;
    void load(ArchiveReader &r, std::size_t features);
};

struct SVMParams {
    Kernel kernel;
    double c = 1.0;
    double epsilon = 0.1;   // SVR
    double nu = 0.1;        // one-class
    double tolerance = 1e-3;
    std::size_t cache_mb = 128;
    void save(ArchiveWriter &w) const;
    void load(ArchiveReader &r);
};

// Support vector classifier (one-vs-one for more than two classes).
class SVC {
public:
    SVMParams params;
    std::size_t classes = 0, features = 0;
    std::vector<SVMFunction> functions;  // pairs (a, b), a < b, in order

    void fit(const Matrix &X, const std::vector<int> &y, std::size_t n_classes, unsigned threads);
    int predict_one(const double *x) const;
    bool fitted() const { return !functions.empty(); }
    void save(ArchiveWriter &w) const;
    void load(ArchiveReader &r);
};

// Epsilon-insensitive support vector regression.
class SVR {
public:
    SVMParams params;
    std::size_t features = 0;
    SVMFunction function;

    void fit(const Matrix &X, const std::vector<double> &y);
    double predict(const double *x) const { return function.decision(params.kernel, x); }
    bool fitted() const { return features > 0; }
    void save(ArchiveWriter &w) const;
    void load(ArchiveReader &r);
};

// One-class SVM (Schoelkopf). score = -f(x); anomalies have score > 0.
class OneClassSVM {
public:
    SVMParams params;
    std::size_t features = 0;
    SVMFunction function;

    void fit(const Matrix &X);
    double score(const double *x) const { return -function.decision(params.kernel, x); }
    bool fitted() const { return features > 0; }
    void save(ArchiveWriter &w) const;
    void load(ArchiveReader &r);
};

}  // namespace mlcore
