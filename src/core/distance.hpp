#pragma once

#include "archive.hpp"
#include "common.hpp"

namespace mlcore {

// A distance function selected by name. Spec strings (used by the PHP API and in
// serialized models): euclidean, squared_euclidean, manhattan, chebyshev,
// minkowski or minkowski(<p>), cosine, canberra, hamming, jaccard,
// safe_euclidean.
class Distance {
public:
    enum class Kind { Euclidean, SquaredEuclidean, Manhattan, Chebyshev, Minkowski, Cosine, Canberra, Hamming, Jaccard, SafeEuclidean };

    Distance() = default;
    static Distance parse(const std::string &spec);

    double operator()(const double *a, const double *b, std::size_t n) const;

    // True for the Minkowski family, where |a_i - b_i| is a lower bound of the
    // distance. Spatial trees can only prune with these.
    bool coordinate_bounded() const;
    // Lower bound of the distance implied by a gap of `gap` in one coordinate.
    double coordinate_bound(double gap) const;

    std::string spec() const;
    Kind kind() const { return kind_; }

    void save(ArchiveWriter &w) const { w.str(spec()); }
    static Distance load(ArchiveReader &r);

private:
    Kind kind_ = Kind::Euclidean;
    double p_ = 3.0;
};

}  // namespace mlcore
