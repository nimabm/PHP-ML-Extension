#pragma once

// Hidden Markov models with discrete or Gaussian (diagonal) emissions, trained
// with Baum-Welch (unsupervised) or by counting labeled state sequences.

#include "archive.hpp"
#include "common.hpp"

namespace mlcore {

struct HMMParams {
    std::size_t states = 2;
    long max_iterations = 100;
    double tolerance = 1e-3;  // stop once the log-likelihood improves by less
    double smoothing = 0.01;  // pseudocount for start, transition and discrete emission estimates
    bool has_seed = false;
    std::uint64_t seed = 0;

    void save(ArchiveWriter &w) const;
    void load(ArchiveReader &r);
};

// Start and transition probabilities of the hidden chain.
struct MarkovChain {
    std::size_t states = 0;
    std::vector<double> start;       // S
    std::vector<double> transition;  // S x S, row = current state

    void save(ArchiveWriter &w) const;
    void load(ArchiveReader &r);
};

// Posterior state probabilities (T x S) and log-likelihood of one sequence,
// given its emission log-likelihoods (T x S). `xi`, if given, accumulates the
// expected transition counts (S x S).
double forward_backward(const MarkovChain &chain, const Matrix &log_emissions, Matrix *posteriors,
                        std::vector<double> *xi);
// Most likely state path (Viterbi).
std::vector<int> viterbi(const MarkovChain &chain, const Matrix &log_emissions);

// Observations are symbol indices; -1 marks a symbol never seen in training,
// which carries no information (equal likelihood in every state).
class DiscreteHMM {
public:
    using Sequence = std::vector<int>;

    HMMParams params;
    std::size_t symbols = 0;
    MarkovChain chain;
    Matrix emission;  // S x M
    double log_likelihood = 0.0;
    long iterations = 0;

    void fit(const std::vector<Sequence> &sequences, std::size_t symbol_count, unsigned threads);
    void fit_labeled(const std::vector<Sequence> &sequences, const std::vector<std::vector<int>> &paths,
                     std::size_t state_count, std::size_t symbol_count);
    Matrix log_emissions(const Sequence &sequence) const;
    void sample(std::size_t length, Rng &rng, std::vector<int> &observations, std::vector<int> &states) const;
    bool fitted() const { return chain.states > 0; }
    void save(ArchiveWriter &w) const;
    void load(ArchiveReader &r);
};

// Observations are rows of a T x d matrix, modeled per state as independent normals.
class GaussianHMM {
public:
    using Sequence = Matrix;

    HMMParams params;
    double min_variance = 1e-6;
    std::size_t features = 0;
    MarkovChain chain;
    Matrix means, variances;  // S x d
    double log_likelihood = 0.0;
    long iterations = 0;

    void fit(const std::vector<Sequence> &sequences, unsigned threads);
    void fit_labeled(const std::vector<Sequence> &sequences, const std::vector<std::vector<int>> &paths,
                     std::size_t state_count);
    Matrix log_emissions(const Sequence &sequence) const;
    void sample(std::size_t length, Rng &rng, Matrix &observations, std::vector<int> &states) const;
    bool fitted() const { return chain.states > 0; }
    void save(ArchiveWriter &w) const;
    void load(ArchiveReader &r);
};

}  // namespace mlcore
