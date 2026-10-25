#include "distance.hpp"

#include <cstdlib>
#include <sstream>

namespace mlcore {

Distance Distance::parse(const std::string &spec) {
    Distance d;
    if (spec == "euclidean") {
        d.kind_ = Kind::Euclidean;
    } else if (spec == "squared_euclidean") {
        d.kind_ = Kind::SquaredEuclidean;
    } else if (spec == "manhattan") {
        d.kind_ = Kind::Manhattan;
    } else if (spec == "chebyshev") {
        d.kind_ = Kind::Chebyshev;
    } else if (spec == "cosine") {
        d.kind_ = Kind::Cosine;
    } else if (spec == "canberra") {
        d.kind_ = Kind::Canberra;
    } else if (spec == "hamming") {
        d.kind_ = Kind::Hamming;
    } else if (spec == "jaccard") {
        d.kind_ = Kind::Jaccard;
    } else if (spec == "safe_euclidean") {
        d.kind_ = Kind::SafeEuclidean;
    } else if (spec == "minkowski") {
        d.kind_ = Kind::Minkowski;
    } else if (spec.size() > 11 && spec.compare(0, 10, "minkowski(") == 0 && spec.back() == ')') {
        const std::string number = spec.substr(10, spec.size() - 11);
        char *end = nullptr;
        const double p = std::strtod(number.c_str(), &end);
        if (end != number.c_str() + number.size() || !(p >= 1.0) || !std::isfinite(p)) {
            throw std::invalid_argument("Minkowski p must be a number >= 1, \"" + number + "\" given");
        }
        d.kind_ = Kind::Minkowski;
        d.p_ = p;
    } else {
        throw std::invalid_argument(
            "Unknown distance metric \"" + spec +
            "\"; expected euclidean, squared_euclidean, manhattan, chebyshev, minkowski, minkowski(<p>), cosine, "
            "canberra, hamming, jaccard or safe_euclidean");
    }
    return d;
}

Distance Distance::load(ArchiveReader &r) {
    try {
        return parse(r.str());
    } catch (const std::invalid_argument &) {
        ArchiveReader::fail("unknown distance metric");
    }
}

std::string Distance::spec() const {
    switch (kind_) {
        case Kind::Euclidean: return "euclidean";
        case Kind::SquaredEuclidean: return "squared_euclidean";
        case Kind::Manhattan: return "manhattan";
        case Kind::Chebyshev: return "chebyshev";
        case Kind::Cosine: return "cosine";
        case Kind::Canberra: return "canberra";
        case Kind::Hamming: return "hamming";
        case Kind::Jaccard: return "jaccard";
        case Kind::SafeEuclidean: return "safe_euclidean";
        case Kind::Minkowski: {
            if (p_ == 3.0) {
                return "minkowski";
            }
            std::ostringstream out;
            out.precision(17);
            out << "minkowski(" << p_ << ")";
            return out.str();
        }
    }
    return "euclidean";
}

bool Distance::coordinate_bounded() const {
    return kind_ == Kind::Euclidean || kind_ == Kind::SquaredEuclidean || kind_ == Kind::Manhattan ||
           kind_ == Kind::Chebyshev || kind_ == Kind::Minkowski;
}

double Distance::coordinate_bound(double gap) const {
    return kind_ == Kind::SquaredEuclidean ? gap * gap : gap;
}

double Distance::operator()(const double *a, const double *b, std::size_t n) const {
    switch (kind_) {
        case Kind::Euclidean:
            return std::sqrt(squared_distance(a, b, n));
        case Kind::SquaredEuclidean:
            return squared_distance(a, b, n);
        case Kind::Manhattan:
            return manhattan_distance(a, b, n);
        case Kind::Chebyshev: {
            double m = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                m = std::max(m, std::fabs(a[i] - b[i]));
            }
            return m;
        }
        case Kind::Minkowski: {
            double sum = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                sum += std::pow(std::fabs(a[i] - b[i]), p_);
            }
            return std::pow(sum, 1.0 / p_);
        }
        case Kind::Cosine: {
            double ab = 0.0, aa = 0.0, bb = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                ab += a[i] * b[i];
                aa += a[i] * a[i];
                bb += b[i] * b[i];
            }
            if (aa == 0.0 && bb == 0.0) {
                return 0.0;
            }
            if (aa == 0.0 || bb == 0.0) {
                return 1.0;
            }
            return std::max(0.0, 1.0 - ab / std::sqrt(aa * bb));
        }
        case Kind::Canberra: {
            double sum = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                const double den = std::fabs(a[i]) + std::fabs(b[i]);
                if (den > 0.0) {
                    sum += std::fabs(a[i] - b[i]) / den;
                }
            }
            return sum;
        }
        case Kind::Hamming: {
            std::size_t diff = 0;
            for (std::size_t i = 0; i < n; ++i) {
                diff += a[i] != b[i];
            }
            return n == 0 ? 0.0 : static_cast<double>(diff) / static_cast<double>(n);
        }
        case Kind::Jaccard: {
            // Generalized (weighted) Jaccard distance for non-negative features.
            double mins = 0.0, maxs = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                mins += std::min(a[i], b[i]);
                maxs += std::max(a[i], b[i]);
            }
            return maxs == 0.0 ? 0.0 : 1.0 - mins / maxs;
        }
        case Kind::SafeEuclidean: {
            // Ignores NaN coordinates and scales up for the ones that were skipped.
            double sum = 0.0;
            std::size_t used = 0;
            for (std::size_t i = 0; i < n; ++i) {
                if (std::isnan(a[i]) || std::isnan(b[i])) {
                    continue;
                }
                const double d = a[i] - b[i];
                sum += d * d;
                ++used;
            }
            if (used == 0) {
                return std::numeric_limits<double>::infinity();
            }
            return std::sqrt(sum * static_cast<double>(n) / static_cast<double>(used));
        }
    }
    return 0.0;
}

}  // namespace mlcore
