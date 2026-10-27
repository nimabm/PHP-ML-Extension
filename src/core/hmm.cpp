#include "hmm.hpp"

#include "clustering.hpp"

namespace mlcore {

namespace {

constexpr double kLog2Pi = 1.8378770664093453;
constexpr double kNegInfinity = -std::numeric_limits<double>::infinity();
// Sequences are processed in fixed blocks so that sums (and therefore results)
// do not depend on the number of threads.
constexpr std::size_t kBlock = 64;

// Normalizes counts + alpha into probabilities, row by row.
void normalize_rows(const std::vector<double> &counts, std::size_t rows, std::size_t cols, double alpha,
                    std::vector<double> &out) {
    out.resize(rows * cols);
    for (std::size_t r = 0; r < rows; ++r) {
        double total = 0.0;
        for (std::size_t c = 0; c < cols; ++c) total += counts[r * cols + c] + alpha;
        for (std::size_t c = 0; c < cols; ++c) {
            out[r * cols + c] = total > 0.0 ? (counts[r * cols + c] + alpha) / total : 1.0 / static_cast<double>(cols);
        }
    }
}

// Probabilities near uniform, jittered so that EM can break the symmetry.
void jittered(double *p, std::size_t n, Rng &rng, const double *weights = nullptr) {
    std::uniform_real_distribution<double> jitter(0.5, 1.5);
    double total = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        p[i] = jitter(rng) * (weights ? weights[i] : 1.0);
        total += p[i];
    }
    for (std::size_t i = 0; i < n; ++i) p[i] = total > 0.0 ? p[i] / total : 1.0 / static_cast<double>(n);
}

void random_chain(MarkovChain &chain, std::size_t S, Rng &rng) {
    chain.states = S;
    chain.start.resize(S);
    chain.transition.resize(S * S);
    jittered(chain.start.data(), S, rng);
    for (std::size_t s = 0; s < S; ++s) jittered(&chain.transition[s * S], S, rng);
}

void require_probabilities(const double *p, std::size_t n) {
    double total = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        ArchiveReader::require(p[i] >= 0.0 && p[i] <= 1.0, "probabilities must be between 0 and 1");
        total += p[i];
    }
    ArchiveReader::require(std::fabs(total - 1.0) < 1e-6, "probabilities must sum to 1");
}

struct Expectations {
    std::vector<double> start, transition;
    double log_likelihood = 0.0;

    explicit Expectations(std::size_t S) : start(S, 0.0), transition(S * S, 0.0) {}
    void add(const Expectations &o) {
        for (std::size_t i = 0; i < start.size(); ++i) start[i] += o.start[i];
        for (std::size_t i = 0; i < transition.size(); ++i) transition[i] += o.transition[i];
        log_likelihood += o.log_likelihood;
    }
};

// E-step over all sequences: chain expectations plus model-specific emission
// statistics (Stats::add, accumulate(stats, sequence, posteriors)).
template <typename Model, typename Stats, typename Accumulate>
Expectations expectations(const Model &m, const std::vector<typename Model::Sequence> &sequences, unsigned threads,
                          const Stats &empty, Accumulate accumulate, Stats &total) {
    const std::size_t S = m.chain.states, blocks = (sequences.size() + kBlock - 1) / kBlock;
    std::vector<Expectations> parts(blocks, Expectations(S));
    std::vector<Stats> stats(blocks, empty);
    parallel_for(blocks, threads, [&](std::size_t b) {
        Matrix posteriors;
        for (std::size_t q = b * kBlock; q < std::min(sequences.size(), (b + 1) * kBlock); ++q) {
            const Matrix log_b = m.log_emissions(sequences[q]);
            const double ll = forward_backward(m.chain, log_b, &posteriors, &parts[b].transition);
            parts[b].log_likelihood += ll;
            if (!std::isfinite(ll)) {
                continue;  // impossible under the current parameters: contributes nothing
            }
            for (std::size_t s = 0; s < S; ++s) parts[b].start[s] += posteriors(0, s);
            accumulate(stats[b], sequences[q], posteriors);
        }
    });
    Expectations sum(S);
    total = empty;
    for (std::size_t b = 0; b < blocks; ++b) {
        sum.add(parts[b]);
        total.add(stats[b]);
    }
    return sum;
}

template <typename Model>
double total_log_likelihood(const Model &m, const std::vector<typename Model::Sequence> &sequences, unsigned threads) {
    std::vector<double> ll(sequences.size());
    parallel_for(sequences.size(), threads,
                 [&](std::size_t q) { ll[q] = forward_backward(m.chain, m.log_emissions(sequences[q]), nullptr, nullptr); });
    double total = 0.0;
    for (double v : ll) total += v;
    return total;
}

// Baum-Welch: alternates expectations() and `update(stats)` (the emission M-step)
// until the log-likelihood stops improving by `tolerance`.
template <typename Model, typename Stats, typename Accumulate, typename Update>
void baum_welch(Model &m, const std::vector<typename Model::Sequence> &sequences, unsigned threads, const Stats &empty,
                Accumulate accumulate, Update update) {
    const std::size_t S = m.chain.states;
    double previous = kNegInfinity;
    m.iterations = 0;
    Stats stats = empty;
    for (long it = 0; it < m.params.max_iterations; ++it) {
        const Expectations e = expectations(m, sequences, threads, empty, accumulate, stats);
        normalize_rows(e.start, 1, S, m.params.smoothing, m.chain.start);
        normalize_rows(e.transition, S, S, m.params.smoothing, m.chain.transition);
        update(stats);
        m.iterations = it + 1;
        const bool converged = it > 0 && e.log_likelihood - previous < m.params.tolerance;
        previous = e.log_likelihood;
        if (converged) {
            break;
        }
    }
    m.log_likelihood = total_log_likelihood(m, sequences, threads);
}

// Start and transition probabilities from labeled paths.
void count_chain(MarkovChain &chain, const std::vector<std::vector<int>> &paths, std::size_t S, double alpha) {
    std::vector<double> start(S, 0.0), transition(S * S, 0.0);
    for (const auto &path : paths) {
        start[static_cast<std::size_t>(path[0])] += 1.0;
        for (std::size_t t = 1; t < path.size(); ++t) {
            transition[static_cast<std::size_t>(path[t - 1]) * S + static_cast<std::size_t>(path[t])] += 1.0;
        }
    }
    chain.states = S;
    normalize_rows(start, 1, S, alpha, chain.start);
    normalize_rows(transition, S, S, alpha, chain.transition);
}

std::size_t draw(const double *p, std::size_t n, Rng &rng) {
    const double u = std::uniform_real_distribution<double>(0.0, 1.0)(rng);
    double acc = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        acc += p[i];
        if (u < acc) return i;
    }
    return n - 1;
}

}  // namespace

// ---- shared pieces ---------------------------------------------------------------------------

void HMMParams::save(ArchiveWriter &w) const {
    w.size(states);
    w.i64(max_iterations);
    w.f64(tolerance);
    w.f64(smoothing);
    w.boolean(has_seed);
    w.u64(seed);
}

void HMMParams::load(ArchiveReader &r) {
    states = r.size();
    max_iterations = static_cast<long>(r.i64());
    tolerance = r.f64();
    smoothing = r.f64();
    has_seed = r.boolean();
    seed = r.u64();
    ArchiveReader::require(states >= 1 && max_iterations >= 1 && tolerance >= 0.0 && smoothing > 0.0,
                           "bad hidden Markov model parameters");
}

void MarkovChain::save(ArchiveWriter &w) const {
    w.size(states);
    w.doubles(start);
    w.doubles(transition);
}

void MarkovChain::load(ArchiveReader &r) {
    states = r.size(1u << 20);
    start = r.doubles();
    transition = r.doubles();
    if (states == 0) {
        ArchiveReader::require(start.empty() && transition.empty());
        return;
    }
    ArchiveReader::require(start.size() == states && transition.size() == states * states);
    require_probabilities(start.data(), states);
    for (std::size_t s = 0; s < states; ++s) require_probabilities(&transition[s * states], states);
}

double forward_backward(const MarkovChain &chain, const Matrix &log_emissions, Matrix *posteriors, std::vector<double> *xi) {
    const std::size_t T = log_emissions.rows, S = chain.states;
    // Emission likelihoods scaled per step by their maximum; the offsets go back into the log-likelihood.
    Matrix B(T, S), alpha(T, S);
    std::vector<double> scale(T);
    double log_likelihood = 0.0;
    for (std::size_t t = 0; t < T; ++t) {
        const double *lb = log_emissions.row(t);
        double offset = lb[0];
        for (std::size_t s = 1; s < S; ++s) offset = std::max(offset, lb[s]);
        if (!std::isfinite(offset)) {
            offset = 0.0;
        }
        for (std::size_t s = 0; s < S; ++s) B(t, s) = std::exp(lb[s] - offset);
        double total = 0.0;
        for (std::size_t s = 0; s < S; ++s) {
            double v;
            if (t == 0) {
                v = chain.start[s];
            } else {
                v = 0.0;
                for (std::size_t r = 0; r < S; ++r) v += alpha(t - 1, r) * chain.transition[r * S + s];
            }
            alpha(t, s) = v * B(t, s);
            total += alpha(t, s);
        }
        if (!(total > 0.0)) {
            return kNegInfinity;  // the sequence is impossible under this model
        }
        for (std::size_t s = 0; s < S; ++s) alpha(t, s) /= total;
        scale[t] = total;
        log_likelihood += std::log(total) + offset;
    }
    if (!posteriors && !xi) {
        return log_likelihood;
    }
    Matrix beta(T, S, 1.0);
    for (std::size_t t = T - 1; t-- > 0;) {
        for (std::size_t r = 0; r < S; ++r) {
            double v = 0.0;
            for (std::size_t s = 0; s < S; ++s) v += chain.transition[r * S + s] * B(t + 1, s) * beta(t + 1, s);
            beta(t, r) = v / scale[t + 1];
        }
    }
    if (posteriors) {
        *posteriors = Matrix(T, S);
        for (std::size_t t = 0; t < T; ++t) {
            double total = 0.0;
            for (std::size_t s = 0; s < S; ++s) total += alpha(t, s) * beta(t, s);
            for (std::size_t s = 0; s < S; ++s) (*posteriors)(t, s) = alpha(t, s) * beta(t, s) / total;
        }
    }
    if (xi) {
        for (std::size_t t = 0; t + 1 < T; ++t) {
            for (std::size_t r = 0; r < S; ++r) {
                const double a = alpha(t, r) / scale[t + 1];
                for (std::size_t s = 0; s < S; ++s) {
                    (*xi)[r * S + s] += a * chain.transition[r * S + s] * B(t + 1, s) * beta(t + 1, s);
                }
            }
        }
    }
    return log_likelihood;
}

std::vector<int> viterbi(const MarkovChain &chain, const Matrix &log_emissions) {
    const std::size_t T = log_emissions.rows, S = chain.states;
    std::vector<double> log_trans(S * S), delta(S), next(S);
    for (std::size_t i = 0; i < S * S; ++i) log_trans[i] = std::log(chain.transition[i]);
    std::vector<int> back(T * S, 0);
    for (std::size_t s = 0; s < S; ++s) delta[s] = std::log(chain.start[s]) + log_emissions(0, s);
    for (std::size_t t = 1; t < T; ++t) {
        for (std::size_t s = 0; s < S; ++s) {
            std::size_t best = 0;
            double best_score = delta[0] + log_trans[s];
            for (std::size_t r = 1; r < S; ++r) {
                const double score = delta[r] + log_trans[r * S + s];
                if (score > best_score) {
                    best_score = score;
                    best = r;
                }
            }
            next[s] = best_score + log_emissions(t, s);
            back[t * S + s] = static_cast<int>(best);
        }
        delta.swap(next);
    }
    std::vector<int> path(T);
    path[T - 1] = static_cast<int>(argmax(delta.data(), S));
    for (std::size_t t = T - 1; t > 0; --t) {
        path[t - 1] = back[t * S + static_cast<std::size_t>(path[t])];
    }
    return path;
}

// ---- discrete emissions ----------------------------------------------------------------------

namespace {

struct DiscreteStats {
    std::vector<double> counts;  // S x M
    void add(const DiscreteStats &o) {
        for (std::size_t i = 0; i < counts.size(); ++i) counts[i] += o.counts[i];
    }
};

}  // namespace

Matrix DiscreteHMM::log_emissions(const Sequence &sequence) const {
    const std::size_t S = chain.states;
    Matrix out(sequence.size(), S);
    for (std::size_t t = 0; t < sequence.size(); ++t) {
        if (sequence[t] < 0) {
            continue;  // unknown symbol: no information
        }
        for (std::size_t s = 0; s < S; ++s) out(t, s) = std::log(emission(s, static_cast<std::size_t>(sequence[t])));
    }
    return out;
}

void DiscreteHMM::fit(const std::vector<Sequence> &sequences, std::size_t symbol_count, unsigned threads) {
    const std::size_t S = params.states, M = symbol_count;
    DiscreteHMM next = *this;
    Rng rng = make_rng(params.has_seed, params.seed);
    random_chain(next.chain, S, rng);
    // Emissions start from the overall symbol frequencies, jittered per state.
    std::vector<double> frequency(M, 0.0);
    for (const Sequence &seq : sequences) {
        for (int o : seq) frequency[static_cast<std::size_t>(o)] += 1.0;
    }
    next.symbols = M;
    next.emission = Matrix(S, M);
    for (std::size_t s = 0; s < S; ++s) jittered(next.emission.row(s), M, rng, frequency.data());

    const double alpha = params.smoothing;
    baum_welch(
        next, sequences, threads, DiscreteStats{std::vector<double>(S * M, 0.0)},
        [S, M](DiscreteStats &st, const Sequence &seq, const Matrix &post) {
            for (std::size_t t = 0; t < seq.size(); ++t) {
                const std::size_t o = static_cast<std::size_t>(seq[t]);
                for (std::size_t s = 0; s < S; ++s) st.counts[s * M + o] += post(t, s);
            }
        },
        [&next, S, M, alpha](const DiscreteStats &st) { normalize_rows(st.counts, S, M, alpha, next.emission.data); });
    *this = std::move(next);
}

void DiscreteHMM::fit_labeled(const std::vector<Sequence> &sequences, const std::vector<std::vector<int>> &paths,
                              std::size_t state_count, std::size_t symbol_count) {
    const std::size_t S = state_count, M = symbol_count;
    DiscreteHMM next = *this;
    next.params.states = S;
    count_chain(next.chain, paths, S, params.smoothing);
    std::vector<double> counts(S * M, 0.0);
    for (std::size_t q = 0; q < sequences.size(); ++q) {
        for (std::size_t t = 0; t < sequences[q].size(); ++t) {
            counts[static_cast<std::size_t>(paths[q][t]) * M + static_cast<std::size_t>(sequences[q][t])] += 1.0;
        }
    }
    next.symbols = M;
    next.emission = Matrix(S, M);
    normalize_rows(counts, S, M, params.smoothing, next.emission.data);
    next.iterations = 0;
    next.log_likelihood = total_log_likelihood(next, sequences, 1);
    *this = std::move(next);
}

void DiscreteHMM::sample(std::size_t length, Rng &rng, std::vector<int> &observations, std::vector<int> &states) const {
    const std::size_t S = chain.states;
    observations.resize(length);
    states.resize(length);
    std::size_t s = draw(chain.start.data(), S, rng);
    for (std::size_t t = 0; t < length; ++t) {
        if (t > 0) s = draw(&chain.transition[s * S], S, rng);
        states[t] = static_cast<int>(s);
        observations[t] = static_cast<int>(draw(emission.row(s), symbols, rng));
    }
}

void DiscreteHMM::save(ArchiveWriter &w) const {
    params.save(w);
    w.size(symbols);
    chain.save(w);
    w.matrix(emission);
    w.f64(log_likelihood);
    w.i64(iterations);
}

void DiscreteHMM::load(ArchiveReader &r) {
    params.load(r);
    symbols = r.size();
    chain.load(r);
    emission = r.matrix();
    log_likelihood = r.f64();
    iterations = static_cast<long>(r.i64());
    if (fitted()) {
        ArchiveReader::require(symbols >= 1 && emission.rows == chain.states && emission.cols == symbols);
        for (std::size_t s = 0; s < chain.states; ++s) require_probabilities(emission.row(s), symbols);
    } else {
        ArchiveReader::require(symbols == 0 && emission.data.empty());
    }
}

// ---- Gaussian emissions ----------------------------------------------------------------------

namespace {

struct GaussianStats {
    std::vector<double> weight, sum, squares;  // S, S x d, S x d
    void add(const GaussianStats &o) {
        for (std::size_t i = 0; i < weight.size(); ++i) weight[i] += o.weight[i];
        for (std::size_t i = 0; i < sum.size(); ++i) {
            sum[i] += o.sum[i];
            squares[i] += o.squares[i];
        }
    }
};

// Means and variances from weighted statistics; states with no weight keep their old values.
void gaussian_update(const GaussianStats &st, std::size_t S, std::size_t d, double floor, Matrix &means, Matrix &variances) {
    for (std::size_t s = 0; s < S; ++s) {
        if (!(st.weight[s] > 1e-12)) continue;
        for (std::size_t j = 0; j < d; ++j) {
            const double mean = st.sum[s * d + j] / st.weight[s];
            means(s, j) = mean;
            variances(s, j) = std::max(st.squares[s * d + j] / st.weight[s] - mean * mean, 0.0) + floor;
        }
    }
}

}  // namespace

Matrix GaussianHMM::log_emissions(const Sequence &sequence) const {
    const std::size_t S = chain.states, d = features;
    Matrix out(sequence.rows, S);
    std::vector<double> norm(S, 0.0);
    for (std::size_t s = 0; s < S; ++s) {
        for (std::size_t j = 0; j < d; ++j) norm[s] += kLog2Pi + std::log(variances(s, j));
    }
    for (std::size_t t = 0; t < sequence.rows; ++t) {
        const double *x = sequence.row(t);
        for (std::size_t s = 0; s < S; ++s) {
            double q = 0.0;
            for (std::size_t j = 0; j < d; ++j) {
                const double diff = x[j] - means(s, j);
                q += diff * diff / variances(s, j);
            }
            out(t, s) = -0.5 * (norm[s] + q);
        }
    }
    return out;
}

void GaussianHMM::fit(const std::vector<Sequence> &sequences, unsigned threads) {
    const std::size_t S = params.states, d = sequences[0].cols;
    std::size_t total = 0;
    for (const Sequence &seq : sequences) total += seq.rows;
    if (total < S) {
        throw std::invalid_argument("Number of states (" + std::to_string(S) + ") cannot exceed the number of observations (" +
                                    std::to_string(total) + ")");
    }
    Matrix all(total, d);
    std::size_t row = 0;
    for (const Sequence &seq : sequences) {
        std::copy(seq.data.begin(), seq.data.end(), all.data.begin() + static_cast<std::ptrdiff_t>(row * d));
        row += seq.rows;
    }
    GaussianHMM next = *this;
    Rng rng = make_rng(params.has_seed, params.seed);
    random_chain(next.chain, S, rng);
    // Means start at k-means++ centers of all observations, variances at the overall variance.
    next.features = d;
    next.means = seed_centers(all, S, Seeder::PlusPlus, Matrix(), Distance(), rng);
    next.variances = Matrix(S, d);
    for (std::size_t j = 0; j < d; ++j) {
        double mean = 0.0, sq = 0.0;
        for (std::size_t t = 0; t < total; ++t) mean += all(t, j);
        mean /= static_cast<double>(total);
        for (std::size_t t = 0; t < total; ++t) sq += (all(t, j) - mean) * (all(t, j) - mean);
        const double var = sq / static_cast<double>(total) + min_variance;
        for (std::size_t s = 0; s < S; ++s) next.variances(s, j) = var;
    }

    const double floor = min_variance;
    baum_welch(
        next, sequences, threads,
        GaussianStats{std::vector<double>(S, 0.0), std::vector<double>(S * d, 0.0), std::vector<double>(S * d, 0.0)},
        [S, d](GaussianStats &st, const Sequence &seq, const Matrix &post) {
            for (std::size_t t = 0; t < seq.rows; ++t) {
                const double *x = seq.row(t);
                for (std::size_t s = 0; s < S; ++s) {
                    const double g = post(t, s);
                    st.weight[s] += g;
                    for (std::size_t j = 0; j < d; ++j) {
                        st.sum[s * d + j] += g * x[j];
                        st.squares[s * d + j] += g * x[j] * x[j];
                    }
                }
            }
        },
        [&next, S, d, floor](const GaussianStats &st) { gaussian_update(st, S, d, floor, next.means, next.variances); });
    *this = std::move(next);
}

void GaussianHMM::fit_labeled(const std::vector<Sequence> &sequences, const std::vector<std::vector<int>> &paths,
                              std::size_t state_count) {
    const std::size_t S = state_count, d = sequences[0].cols;
    GaussianHMM next = *this;
    next.params.states = S;
    count_chain(next.chain, paths, S, params.smoothing);
    GaussianStats st{std::vector<double>(S, 0.0), std::vector<double>(S * d, 0.0), std::vector<double>(S * d, 0.0)};
    for (std::size_t q = 0; q < sequences.size(); ++q) {
        for (std::size_t t = 0; t < sequences[q].rows; ++t) {
            const std::size_t s = static_cast<std::size_t>(paths[q][t]);
            const double *x = sequences[q].row(t);
            st.weight[s] += 1.0;
            for (std::size_t j = 0; j < d; ++j) {
                st.sum[s * d + j] += x[j];
                st.squares[s * d + j] += x[j] * x[j];
            }
        }
    }
    next.features = d;
    next.means = Matrix(S, d);
    next.variances = Matrix(S, d, 1.0);
    gaussian_update(st, S, d, min_variance, next.means, next.variances);
    next.iterations = 0;
    next.log_likelihood = total_log_likelihood(next, sequences, 1);
    *this = std::move(next);
}

void GaussianHMM::sample(std::size_t length, Rng &rng, Matrix &observations, std::vector<int> &states) const {
    const std::size_t S = chain.states;
    observations = Matrix(length, features);
    states.resize(length);
    std::normal_distribution<double> normal(0.0, 1.0);
    std::size_t s = draw(chain.start.data(), S, rng);
    for (std::size_t t = 0; t < length; ++t) {
        if (t > 0) s = draw(&chain.transition[s * S], S, rng);
        states[t] = static_cast<int>(s);
        for (std::size_t j = 0; j < features; ++j) {
            observations(t, j) = means(s, j) + std::sqrt(variances(s, j)) * normal(rng);
        }
    }
}

void GaussianHMM::save(ArchiveWriter &w) const {
    params.save(w);
    w.f64(min_variance);
    w.size(features);
    chain.save(w);
    w.matrix(means);
    w.matrix(variances);
    w.f64(log_likelihood);
    w.i64(iterations);
}

void GaussianHMM::load(ArchiveReader &r) {
    params.load(r);
    min_variance = r.f64();
    ArchiveReader::require(min_variance >= 0.0 && std::isfinite(min_variance), "bad minimum variance");
    features = r.size();
    chain.load(r);
    means = r.matrix();
    variances = r.matrix();
    log_likelihood = r.f64();
    iterations = static_cast<long>(r.i64());
    if (fitted()) {
        ArchiveReader::require(features >= 1 && means.rows == chain.states && means.cols == features &&
                               variances.rows == chain.states && variances.cols == features);
        for (double v : variances.data) ArchiveReader::require(v > 0.0 && std::isfinite(v), "variances must be positive");
    } else {
        ArchiveReader::require(features == 0 && means.data.empty() && variances.data.empty());
    }
}

}  // namespace mlcore
