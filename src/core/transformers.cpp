#include "transformers.hpp"

namespace mlcore {

// ---- MinMaxNormalizer ----------------------------------------------------------------

void MinMaxNormalizer::fit(const Matrix &X) {
    require_training_set(X, X.rows);
    std::vector<double> lo(X.cols, std::numeric_limits<double>::infinity());
    std::vector<double> hi(X.cols, -std::numeric_limits<double>::infinity());
    for (std::size_t r = 0; r < X.rows; ++r) {
        for (std::size_t c = 0; c < X.cols; ++c) {
            lo[c] = std::min(lo[c], X(r, c));
            hi[c] = std::max(hi[c], X(r, c));
        }
    }
    for (std::size_t c = 0; c < X.cols; ++c) {
        hi[c] -= lo[c];
        if (hi[c] == 0.0) {
            hi[c] = 1.0;
        }
    }
    lows = std::move(lo);
    ranges = std::move(hi);
}

void MinMaxNormalizer::transform(const double *in, double *out) const {
    for (std::size_t c = 0; c < lows.size(); ++c) {
        out[c] = min + (in[c] - lows[c]) / ranges[c] * (max - min);
    }
}

void MinMaxNormalizer::save(ArchiveWriter &w) const {
    w.f64(min);
    w.f64(max);
    w.doubles(lows);
    w.doubles(ranges);
}

void MinMaxNormalizer::load(ArchiveReader &r) {
    min = r.f64();
    max = r.f64();
    lows = r.doubles();
    ranges = r.doubles(lows.size());
    for (double v : ranges) {
        ArchiveReader::require(v != 0.0, "zero range");
    }
}

// ---- MaxAbsoluteScaler ---------------------------------------------------------------

void MaxAbsoluteScaler::fit(const Matrix &X) {
    require_training_set(X, X.rows);
    std::vector<double> s(X.cols, 0.0);
    for (std::size_t r = 0; r < X.rows; ++r) {
        for (std::size_t c = 0; c < X.cols; ++c) {
            s[c] = std::max(s[c], std::fabs(X(r, c)));
        }
    }
    for (double &v : s) {
        if (v == 0.0) {
            v = 1.0;
        }
    }
    scales = std::move(s);
}

void MaxAbsoluteScaler::transform(const double *in, double *out) const {
    for (std::size_t c = 0; c < scales.size(); ++c) {
        out[c] = in[c] / scales[c];
    }
}

void MaxAbsoluteScaler::save(ArchiveWriter &w) const { w.doubles(scales); }

void MaxAbsoluteScaler::load(ArchiveReader &r) {
    scales = r.doubles();
    for (double v : scales) {
        ArchiveReader::require(v != 0.0, "zero scale");
    }
}

// ---- RobustStandardizer ---------------------------------------------------------------

void RobustStandardizer::fit(const Matrix &X) {
    require_training_set(X, X.rows);
    std::vector<double> med(X.cols), mad(X.cols);
    std::vector<double> column(X.rows);
    for (std::size_t c = 0; c < X.cols; ++c) {
        for (std::size_t r = 0; r < X.rows; ++r) {
            column[r] = X(r, c);
        }
        med[c] = median(column);
        for (std::size_t r = 0; r < X.rows; ++r) {
            column[r] = std::fabs(X(r, c) - med[c]);
        }
        mad[c] = median(column);
        if (mad[c] == 0.0) {
            mad[c] = 1.0;
        }
    }
    medians = std::move(med);
    mads = std::move(mad);
}

void RobustStandardizer::transform(const double *in, double *out) const {
    for (std::size_t c = 0; c < medians.size(); ++c) {
        out[c] = (center ? in[c] - medians[c] : in[c]) / mads[c];
    }
}

void RobustStandardizer::save(ArchiveWriter &w) const {
    w.boolean(center);
    w.doubles(medians);
    w.doubles(mads);
}

void RobustStandardizer::load(ArchiveReader &r) {
    center = r.boolean();
    medians = r.doubles();
    mads = r.doubles(medians.size());
    for (double v : mads) {
        ArchiveReader::require(v != 0.0, "zero MAD");
    }
}

// ---- PowerTransformer -----------------------------------------------------------------

double PowerTransformer::yeo_johnson(double y, double lambda) {
    if (y >= 0.0) {
        return std::fabs(lambda) < 1e-12 ? std::log1p(y) : (std::pow(y + 1.0, lambda) - 1.0) / lambda;
    }
    return std::fabs(lambda - 2.0) < 1e-12 ? -std::log1p(-y) : -(std::pow(1.0 - y, 2.0 - lambda) - 1.0) / (2.0 - lambda);
}

namespace {

double yeo_johnson_log_likelihood(const std::vector<double> &y, double lambda) {
    const double n = static_cast<double>(y.size());
    double mean = 0.0;
    std::vector<double> t(y.size());
    for (std::size_t i = 0; i < y.size(); ++i) {
        t[i] = PowerTransformer::yeo_johnson(y[i], lambda);
        mean += t[i];
    }
    mean /= n;
    double var = 0.0, jacobian = 0.0;
    for (std::size_t i = 0; i < y.size(); ++i) {
        var += (t[i] - mean) * (t[i] - mean);
        jacobian += (y[i] >= 0 ? 1.0 : -1.0) * std::log1p(std::fabs(y[i]));
    }
    var /= n;
    if (!(var > 0.0) || !std::isfinite(var)) {
        return -std::numeric_limits<double>::infinity();
    }
    return -n / 2.0 * std::log(var) + (lambda - 1.0) * jacobian;
}

// Golden-section search for the lambda that maximizes the likelihood.
double best_lambda(const std::vector<double> &y) {
    const double phi = (std::sqrt(5.0) - 1.0) / 2.0;
    double a = -3.0, b = 3.0;
    double c = b - phi * (b - a), d = a + phi * (b - a);
    double fc = yeo_johnson_log_likelihood(y, c), fd = yeo_johnson_log_likelihood(y, d);
    for (int i = 0; i < 80; ++i) {
        if (fc > fd) {
            b = d;
            d = c;
            fd = fc;
            c = b - phi * (b - a);
            fc = yeo_johnson_log_likelihood(y, c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + phi * (b - a);
            fd = yeo_johnson_log_likelihood(y, d);
        }
    }
    return (a + b) / 2.0;
}

}  // namespace

void PowerTransformer::fit(const Matrix &X) {
    require_training_set(X, X.rows);
    if (!fixed_lambdas.empty() && fixed_lambdas.size() != X.cols) {
        throw std::invalid_argument("lambdas must contain one value per feature (" + std::to_string(X.cols) + ")");
    }
    std::vector<double> lam(X.cols), mu(X.cols, 0.0), sd(X.cols, 1.0);
    std::vector<double> column(X.rows);
    for (std::size_t c = 0; c < X.cols; ++c) {
        for (std::size_t r = 0; r < X.rows; ++r) {
            column[r] = X(r, c);
        }
        lam[c] = fixed_lambdas.empty() ? best_lambda(column) : fixed_lambdas[c];
        if (normalize) {
            double mean = 0.0, var = 0.0;
            for (double v : column) {
                mean += yeo_johnson(v, lam[c]);
            }
            mean /= static_cast<double>(X.rows);
            for (double v : column) {
                const double t = yeo_johnson(v, lam[c]) - mean;
                var += t * t;
            }
            mu[c] = mean;
            sd[c] = std::sqrt(var / static_cast<double>(X.rows));
            if (!(sd[c] > 0.0)) {
                sd[c] = 1.0;
            }
        }
    }
    lambdas = std::move(lam);
    means = std::move(mu);
    stds = std::move(sd);
}

void PowerTransformer::transform(const double *in, double *out) const {
    for (std::size_t c = 0; c < lambdas.size(); ++c) {
        out[c] = (yeo_johnson(in[c], lambdas[c]) - means[c]) / stds[c];
    }
}

void PowerTransformer::save(ArchiveWriter &w) const {
    w.doubles(fixed_lambdas);
    w.boolean(normalize);
    w.doubles(lambdas);
    w.doubles(means);
    w.doubles(stds);
}

void PowerTransformer::load(ArchiveReader &r) {
    fixed_lambdas = r.doubles();
    normalize = r.boolean();
    lambdas = r.doubles();
    means = r.doubles(lambdas.size());
    stds = r.doubles(lambdas.size());
    for (double v : stds) {
        ArchiveReader::require(v != 0.0, "zero scale");
    }
}

}  // namespace mlcore
