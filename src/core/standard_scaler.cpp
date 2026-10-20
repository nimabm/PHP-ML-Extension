#include "standard_scaler.hpp"

namespace mlcore {

void StandardScaler::fit(const Matrix &X) {
    require_training_set(X, X.rows);
    const std::size_t n = X.rows;
    const std::size_t d = X.cols;

    std::vector<double> mean(d, 0.0);
    std::vector<double> scale(d, 0.0);
    for (std::size_t r = 0; r < n; ++r) {
        const double *x = X.row(r);
        for (std::size_t j = 0; j < d; ++j) {
            mean[j] += x[j];
        }
    }
    for (double &m : mean) {
        m /= static_cast<double>(n);
    }
    for (std::size_t r = 0; r < n; ++r) {
        const double *x = X.row(r);
        for (std::size_t j = 0; j < d; ++j) {
            const double diff = x[j] - mean[j];
            scale[j] += diff * diff;
        }
    }
    for (double &s : scale) {
        s = std::sqrt(s / static_cast<double>(n));
        if (s == 0.0) {
            s = 1.0;
        }
    }

    means = std::move(mean);
    scales = std::move(scale);
}

void StandardScaler::transform(double *x) const {
    for (std::size_t j = 0; j < means.size(); ++j) {
        x[j] = (x[j] - means[j]) / scales[j];
    }
}

}  // namespace mlcore
