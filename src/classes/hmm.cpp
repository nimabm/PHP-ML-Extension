// Sequence models: HiddenMarkovModel (discrete symbols) and GaussianHMM
// (continuous observation vectors).

#include "binding.hpp"
#include "core/hmm.hpp"

namespace {

using namespace mlphp;
using mlcore::ArchiveReader;
using mlcore::ArchiveWriter;

// ---- reading sequences -------------------------------------------------------------------

// The first element of a non-empty array (dereferenced).
zval *first_of(HashTable *ht, const char *what) {
    zval *value;
    ZEND_HASH_FOREACH_VAL(ht, value) {
        ZVAL_DEREF(value);
        return value;
    }
    ZEND_HASH_FOREACH_END();
    throw std::invalid_argument(std::string(what) + " must not be empty");
}

std::vector<HashTable *> arrays_of(HashTable *ht, const char *what) {
    std::vector<HashTable *> out;
    zval *value;
    ZEND_HASH_FOREACH_VAL(ht, value) {
        ZVAL_DEREF(value);
        if (Z_TYPE_P(value) != IS_ARRAY) {
            throw std::invalid_argument(std::string(what) + " #" + std::to_string(out.size()) + " must be an array");
        }
        if (zend_hash_num_elements(Z_ARRVAL_P(value)) == 0) {
            throw std::invalid_argument(std::string(what) + " #" + std::to_string(out.size()) + " must not be empty");
        }
        out.push_back(Z_ARRVAL_P(value));
    }
    ZEND_HASH_FOREACH_END();
    if (out.empty()) {
        throw std::invalid_argument(std::string(what) + "s must not be empty");
    }
    return out;
}

// Learns sorted distinct labels over several label lists and encodes them.
std::vector<std::vector<int>> encode_all(const std::vector<std::vector<Label>> &lists, LabelEncoder &encoder) {
    std::vector<Label> all;
    for (const auto &list : lists) all.insert(all.end(), list.begin(), list.end());
    std::sort(all.begin(), all.end());
    all.erase(std::unique(all.begin(), all.end()), all.end());
    encoder.classes = std::move(all);
    std::vector<std::vector<int>> out(lists.size());
    for (std::size_t q = 0; q < lists.size(); ++q) {
        for (const Label &label : lists[q]) out[q].push_back(encoder.find(label));
    }
    return out;
}

std::vector<std::vector<Label>> read_label_lists(const std::vector<HashTable *> &tables, const char *what) {
    std::vector<std::vector<Label>> out;
    for (HashTable *ht : tables) out.push_back(read_labels(ht, what));
    return out;
}

void write_states(zval *out, const std::vector<int> &path, const LabelEncoder &states) {
    array_init_size(out, static_cast<uint32_t>(path.size()));
    for (int s : path) {
        zval label;
        states.write(s, &label);
        add_next_index_zval(out, &label);
    }
}

// out[row label] = [col label => value, ...] (or a plain list when `cols` is null).
void write_keyed_rows(zval *out, const mlcore::Matrix &M, const LabelEncoder &rows, const LabelEncoder *cols) {
    array_init_size(out, static_cast<uint32_t>(M.rows));
    for (std::size_t r = 0; r < M.rows; ++r) {
        zval row, key;
        if (cols) {
            array_init_size(&row, static_cast<uint32_t>(M.cols));
            for (std::size_t c = 0; c < M.cols; ++c) cols->write_keyed(&row, static_cast<int>(c), M(r, c));
        } else {
            write_doubles(&row, std::vector<double>(M.row(r), M.row(r) + M.cols));
        }
        rows.write(static_cast<int>(r), &key);
        array_set_zval_key(Z_ARRVAL_P(out), &key, &row);
        zval_ptr_dtor(&row);
        zval_ptr_dtor(&key);
    }
}

void write_posteriors(zval *out, const mlcore::Matrix &P, const LabelEncoder &states) {
    array_init_size(out, static_cast<uint32_t>(P.rows));
    for (std::size_t t = 0; t < P.rows; ++t) {
        zval row;
        array_init_size(&row, static_cast<uint32_t>(P.cols));
        for (std::size_t s = 0; s < P.cols; ++s) states.write_keyed(&row, static_cast<int>(s), P(t, s));
        add_next_index_zval(out, &row);
    }
}

// ---- models --------------------------------------------------------------------------------

struct DiscreteModel {
    static constexpr std::uint32_t kTag = mlcore::tag("HMMD");
    mlcore::DiscreteHMM core;
    LabelEncoder symbols, states;

    bool fitted() const { return core.fitted(); }

    // One sequence (a list of symbols) or several (a list of lists).
    static bool many(HashTable *ht) { return Z_TYPE_P(first_of(ht, "Sequence")) == IS_ARRAY; }

    std::vector<int> encode(HashTable *ht) const {
        std::vector<int> out;
        for (const Label &label : read_labels(ht, "Symbol")) out.push_back(symbols.find(label));
        return out;
    }

    void train(HashTable *sequences, unsigned threads) {
        const auto lists = read_label_lists(arrays_of(sequences, "Sequence"), "Symbol");
        LabelEncoder encoder;
        const auto encoded = encode_all(lists, encoder);
        mlcore::DiscreteHMM next = core;
        next.fit(encoded, encoder.size(), threads);
        core = std::move(next);
        symbols = std::move(encoder);
        states.classes.clear();
        for (std::size_t s = 0; s < core.chain.states; ++s) states.classes.emplace_back(std::in_place_index<0>, static_cast<zend_long>(s));
    }

    void train_labeled(HashTable *sequences, HashTable *paths) {
        const auto observed = read_label_lists(arrays_of(sequences, "Sequence"), "Symbol");
        const auto hidden = read_label_lists(arrays_of(paths, "State sequence"), "State");
        check_lengths(observed, hidden);
        LabelEncoder symbol_encoder, state_encoder;
        const auto encoded = encode_all(observed, symbol_encoder);
        const auto encoded_states = encode_all(hidden, state_encoder);
        mlcore::DiscreteHMM next = core;
        next.fit_labeled(encoded, encoded_states, state_encoder.size(), symbol_encoder.size());
        core = std::move(next);
        symbols = std::move(symbol_encoder);
        states = std::move(state_encoder);
    }

    static void check_lengths(const std::vector<std::vector<Label>> &observed, const std::vector<std::vector<Label>> &hidden) {
        if (observed.size() != hidden.size()) {
            throw std::invalid_argument("Number of sequences (" + std::to_string(observed.size()) +
                                        ") does not match number of state sequences (" + std::to_string(hidden.size()) + ")");
        }
        for (std::size_t q = 0; q < observed.size(); ++q) {
            if (observed[q].size() != hidden[q].size()) {
                throw std::invalid_argument("Sequence #" + std::to_string(q) + " has " + std::to_string(observed[q].size()) +
                                            " observations but " + std::to_string(hidden[q].size()) + " states");
            }
        }
    }

    void write_observations(zval *out, const std::vector<int> &observations) const {
        array_init_size(out, static_cast<uint32_t>(observations.size()));
        for (int o : observations) {
            zval label;
            symbols.write(o, &label);
            add_next_index_zval(out, &label);
        }
    }

    void save(ArchiveWriter &w) const {
        core.save(w);
        symbols.save(w);
        states.save(w);
    }
    void load(ArchiveReader &r) {
        core.load(r);
        symbols.load(r);
        states.load(r);
        ArchiveReader::require(symbols.size() == core.symbols && states.size() == core.chain.states, "label count mismatch");
    }
};

struct GaussianModel {
    static constexpr std::uint32_t kTag = mlcore::tag("HMMG");
    mlcore::GaussianHMM core;
    LabelEncoder states;

    bool fitted() const { return core.fitted(); }

    // One sequence (a list of observation vectors) or several (a list of such lists).
    static bool many(HashTable *ht) {
        zval *first = first_of(ht, "Sequence");
        if (Z_TYPE_P(first) != IS_ARRAY) {
            throw std::invalid_argument("Observations must be arrays of numbers");
        }
        return Z_TYPE_P(first_of(Z_ARRVAL_P(first), "Observation")) == IS_ARRAY;
    }

    mlcore::Matrix encode(HashTable *ht) const { return read_samples(ht, core.features); }

    static std::vector<mlcore::Matrix> read_sequences(HashTable *sequences) {
        std::vector<mlcore::Matrix> out;
        for (HashTable *ht : arrays_of(sequences, "Sequence")) {
            out.push_back(read_samples(ht, out.empty() ? 0 : out[0].cols));
        }
        return out;
    }

    void train(HashTable *sequences, unsigned threads) {
        const auto seqs = read_sequences(sequences);
        mlcore::GaussianHMM next = core;
        next.fit(seqs, threads);
        core = std::move(next);
        states.classes.clear();
        for (std::size_t s = 0; s < core.chain.states; ++s) states.classes.emplace_back(std::in_place_index<0>, static_cast<zend_long>(s));
    }

    void train_labeled(HashTable *sequences, HashTable *paths) {
        const auto seqs = read_sequences(sequences);
        const auto hidden = read_label_lists(arrays_of(paths, "State sequence"), "State");
        std::vector<std::vector<Label>> lengths(seqs.size());
        for (std::size_t q = 0; q < seqs.size(); ++q) lengths[q].resize(seqs[q].rows);
        DiscreteModel::check_lengths(lengths, hidden);
        LabelEncoder state_encoder;
        const auto encoded_states = encode_all(hidden, state_encoder);
        mlcore::GaussianHMM next = core;
        next.fit_labeled(seqs, encoded_states, state_encoder.size());
        core = std::move(next);
        states = std::move(state_encoder);
    }

    void write_observations(zval *out, const mlcore::Matrix &observations) const { write_matrix(out, observations); }

    void save(ArchiveWriter &w) const {
        core.save(w);
        states.save(w);
    }
    void load(ArchiveReader &r) {
        core.load(r);
        states.load(r);
        ArchiveReader::require(states.size() == core.chain.states, "label count mismatch");
    }
};

// ---- shared handlers ---------------------------------------------------------------------

template <typename M>
void ZEND_FASTCALL hmm_fit(INTERNAL_FUNCTION_PARAMETERS) {
    HashTable *sequences;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY_HT(sequences)
    ZEND_PARSE_PARAMETERS_END();
    if (!guard([&] {
            M &m = native<M>(ZEND_THIS);
            M next = m;
            next.train(sequences, threads());
            m = std::move(next);
        })) {
        RETURN_THROWS();
    }
    RETURN_OBJ_COPY(Z_OBJ_P(ZEND_THIS));
}

template <typename M>
void ZEND_FASTCALL hmm_fit_labeled(INTERNAL_FUNCTION_PARAMETERS) {
    HashTable *sequences, *paths;
    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_ARRAY_HT(sequences)
        Z_PARAM_ARRAY_HT(paths)
    ZEND_PARSE_PARAMETERS_END();
    if (!guard([&] {
            M &m = native<M>(ZEND_THIS);
            M next = m;
            next.train_labeled(sequences, paths);
            m = std::move(next);
        })) {
        RETURN_THROWS();
    }
    RETURN_OBJ_COPY(Z_OBJ_P(ZEND_THIS));
}

// Runs f(encoded sequence, out) for one sequence or each of several.
template <typename M, typename F>
void for_sequences(zval *self, HashTable *input, zval *return_value, F f) {
    guard([&] {
        const M &m = native<M>(self);
        require_fitted(m.fitted(), self);
        if (!M::many(input)) {
            f(m, m.encode(input), return_value);
            return;
        }
        const std::vector<HashTable *> tables = arrays_of(input, "Sequence");
        array_init_size(return_value, static_cast<uint32_t>(tables.size()));
        for (HashTable *ht : tables) {
            zval item;
            f(m, m.encode(ht), &item);
            add_next_index_zval(return_value, &item);
        }
    });
}

template <typename M>
void ZEND_FASTCALL hmm_predict(INTERNAL_FUNCTION_PARAMETERS) {
    HashTable *input;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY_HT(input)
    ZEND_PARSE_PARAMETERS_END();
    for_sequences<M>(ZEND_THIS, input, return_value, [](const M &m, const auto &seq, zval *out) {
        write_states(out, mlcore::viterbi(m.core.chain, m.core.log_emissions(seq)), m.states);
    });
}

template <typename M>
void ZEND_FASTCALL hmm_proba(INTERNAL_FUNCTION_PARAMETERS) {
    HashTable *input;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY_HT(input)
    ZEND_PARSE_PARAMETERS_END();
    for_sequences<M>(ZEND_THIS, input, return_value, [](const M &m, const auto &seq, zval *out) {
        mlcore::Matrix posteriors;
        const double ll = mlcore::forward_backward(m.core.chain, m.core.log_emissions(seq), &posteriors, nullptr);
        if (!std::isfinite(ll)) {
            throw std::runtime_error("The sequence is impossible under this model");
        }
        write_posteriors(out, posteriors, m.states);
    });
}

template <typename M>
void ZEND_FASTCALL hmm_score(INTERNAL_FUNCTION_PARAMETERS) {
    HashTable *input;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY_HT(input)
    ZEND_PARSE_PARAMETERS_END();
    for_sequences<M>(ZEND_THIS, input, return_value, [](const M &m, const auto &seq, zval *out) {
        ZVAL_DOUBLE(out, mlcore::forward_backward(m.core.chain, m.core.log_emissions(seq), nullptr, nullptr));
    });
}

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_hmm_sample, 0, 1, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO(0, length, IS_LONG, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, seed, IS_LONG, 1, "null")
ZEND_END_ARG_INFO()

template <typename M>
void ZEND_FASTCALL hmm_sample(INTERNAL_FUNCTION_PARAMETERS) {
    zend_long length, seed = 0;
    bool seed_null = true;
    ZEND_PARSE_PARAMETERS_START(1, 2)
        Z_PARAM_LONG(length)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG_OR_NULL(seed, seed_null)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        const M &m = native<M>(ZEND_THIS);
        require_fitted(m.fitted(), ZEND_THIS);
        check(length >= 1 && length <= 10000000, "length must be between 1 and 10000000");
        mlcore::Rng rng = mlcore::make_rng(!seed_null, static_cast<std::uint64_t>(seed));
        std::conditional_t<std::is_same_v<M, DiscreteModel>, std::vector<int>, mlcore::Matrix> observations;
        std::vector<int> path;
        m.core.sample(static_cast<std::size_t>(length), rng, observations, path);
        zval obs, states;
        m.write_observations(&obs, observations);
        write_states(&states, path, m.states);
        array_init(return_value);
        add_assoc_zval(return_value, "observations", &obs);
        add_assoc_zval(return_value, "states", &states);
    });
}

template <typename M>
void ZEND_FASTCALL hmm_states(INTERNAL_FUNCTION_PARAMETERS) {
    ZEND_PARSE_PARAMETERS_NONE();
    fitted_getter<M>(ZEND_THIS, return_value, [](const M &m, zval *out) { m.states.write_all(out); });
}

template <typename M>
void ZEND_FASTCALL hmm_start(INTERNAL_FUNCTION_PARAMETERS) {
    ZEND_PARSE_PARAMETERS_NONE();
    fitted_getter<M>(ZEND_THIS, return_value, [](const M &m, zval *out) {
        array_init(out);
        for (std::size_t s = 0; s < m.core.chain.states; ++s) m.states.write_keyed(out, static_cast<int>(s), m.core.chain.start[s]);
    });
}

template <typename M>
void ZEND_FASTCALL hmm_transitions(INTERNAL_FUNCTION_PARAMETERS) {
    ZEND_PARSE_PARAMETERS_NONE();
    fitted_getter<M>(ZEND_THIS, return_value, [](const M &m, zval *out) {
        const std::size_t S = m.core.chain.states;
        mlcore::Matrix T(S, S);
        T.data = m.core.chain.transition;
        write_keyed_rows(out, T, m.states, &m.states);
    });
}

template <typename M>
void ZEND_FASTCALL hmm_log_likelihood(INTERNAL_FUNCTION_PARAMETERS) {
    ZEND_PARSE_PARAMETERS_NONE();
    fitted_getter<M>(ZEND_THIS, return_value, [](const M &m, zval *out) { ZVAL_DOUBLE(out, m.core.log_likelihood); });
}

template <typename M>
void ZEND_FASTCALL hmm_iterations(INTERNAL_FUNCTION_PARAMETERS) {
    ZEND_PARSE_PARAMETERS_NONE();
    fitted_getter<M>(ZEND_THIS, return_value, [](const M &m, zval *out) { ZVAL_LONG(out, m.core.iterations); });
}

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_MASK_EX(arginfo_hmm_predict, 0, 1, MAY_BE_ARRAY)
    ZEND_ARG_TYPE_INFO(0, sequences, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_MASK_EX(arginfo_hmm_score, 0, 1, MAY_BE_ARRAY | MAY_BE_DOUBLE)
    ZEND_ARG_TYPE_INFO(0, sequences, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_MASK_EX(arginfo_hmm_fit, 0, 1, MAY_BE_STATIC)
    ZEND_ARG_TYPE_INFO(0, sequences, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_MASK_EX(arginfo_hmm_fit_labeled, 0, 2, MAY_BE_STATIC)
    ZEND_ARG_TYPE_INFO(0, sequences, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO(0, states, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

#define ML_HMM_COMMON(M)                                                            \
    ML_METHOD(fit, hmm_fit<M>, arginfo_hmm_fit)                                     \
    ML_METHOD(fitLabeled, hmm_fit_labeled<M>, arginfo_hmm_fit_labeled)              \
    ML_METHOD(predict, hmm_predict<M>, arginfo_hmm_predict)                         \
    ML_METHOD(predictProbability, hmm_proba<M>, arginfo_hmm_predict)                \
    ML_METHOD(score, hmm_score<M>, arginfo_hmm_score)                               \
    ML_METHOD(sample, hmm_sample<M>, arginfo_hmm_sample)                            \
    ML_METHOD(getStates, hmm_states<M>, arginfo_ml_get_array)                       \
    ML_METHOD(getStartProbabilities, hmm_start<M>, arginfo_ml_get_array)            \
    ML_METHOD(getTransitionMatrix, hmm_transitions<M>, arginfo_ml_get_array)        \
    ML_METHOD(getLogLikelihood, hmm_log_likelihood<M>, arginfo_ml_get_float)        \
    ML_METHOD(getIterations, hmm_iterations<M>, arginfo_ml_get_int)                 \
    ML_METHOD(__serialize, serialize_native<M>, arginfo_ml_get_array)               \
    ML_METHOD(__unserialize, unserialize_native<M>, arginfo_ml_unserialize)

void configure(mlcore::HMMParams &p, zend_long states, zend_long max_iterations, double tolerance, double smoothing,
               bool seed_null, zend_long seed) {
    p.states = positive(states, "states");
    p.max_iterations = static_cast<long>(positive(max_iterations, "maxIterations"));
    p.tolerance = non_negative(tolerance, "tolerance");
    p.smoothing = strictly_positive(smoothing, "smoothing");
    parse_seed(seed_null, seed, p.has_seed, p.seed);
}

// ---- HiddenMarkovModel -------------------------------------------------------------------

ZEND_BEGIN_ARG_INFO_EX(arginfo_discrete_construct, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, states, IS_LONG, 0, "2")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, maxIterations, IS_LONG, 0, "100")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, tolerance, IS_DOUBLE, 0, "0.001")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, smoothing, IS_DOUBLE, 0, "0.01")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, seed, IS_LONG, 1, "null")
ZEND_END_ARG_INFO()

void ZEND_FASTCALL discrete_construct(INTERNAL_FUNCTION_PARAMETERS) {
    zend_long states = 2, max_iterations = 100, seed = 0;
    double tolerance = 1e-3, smoothing = 0.01;
    bool seed_null = true;
    ZEND_PARSE_PARAMETERS_START(0, 5)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(states)
        Z_PARAM_LONG(max_iterations)
        Z_PARAM_DOUBLE(tolerance)
        Z_PARAM_DOUBLE(smoothing)
        Z_PARAM_LONG_OR_NULL(seed, seed_null)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        DiscreteModel fresh;
        configure(fresh.core.params, states, max_iterations, tolerance, smoothing, seed_null, seed);
        reset(ZEND_THIS, std::move(fresh));
    });
}

void ZEND_FASTCALL discrete_emissions(INTERNAL_FUNCTION_PARAMETERS) {
    ZEND_PARSE_PARAMETERS_NONE();
    fitted_getter<DiscreteModel>(ZEND_THIS, return_value, [](const DiscreteModel &m, zval *out) {
        write_keyed_rows(out, m.core.emission, m.states, &m.symbols);
    });
}

void ZEND_FASTCALL discrete_symbols(INTERNAL_FUNCTION_PARAMETERS) {
    ZEND_PARSE_PARAMETERS_NONE();
    fitted_getter<DiscreteModel>(ZEND_THIS, return_value, [](const DiscreteModel &m, zval *out) { m.symbols.write_all(out); });
}

const zend_function_entry discrete_methods[] = {
    ML_METHOD(__construct, discrete_construct, arginfo_discrete_construct)
    ML_HMM_COMMON(DiscreteModel)
    ML_METHOD(getEmissionProbabilities, discrete_emissions, arginfo_ml_get_array)
    ML_METHOD(getSymbols, discrete_symbols, arginfo_ml_get_array)
    PHP_FE_END};

// ---- GaussianHMM -------------------------------------------------------------------------

ZEND_BEGIN_ARG_INFO_EX(arginfo_gaussian_construct, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, states, IS_LONG, 0, "2")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, maxIterations, IS_LONG, 0, "100")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, tolerance, IS_DOUBLE, 0, "0.001")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, smoothing, IS_DOUBLE, 0, "0.01")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, minVariance, IS_DOUBLE, 0, "1.0E-6")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, seed, IS_LONG, 1, "null")
ZEND_END_ARG_INFO()

void ZEND_FASTCALL gaussian_construct(INTERNAL_FUNCTION_PARAMETERS) {
    zend_long states = 2, max_iterations = 100, seed = 0;
    double tolerance = 1e-3, smoothing = 0.01, min_variance = 1e-6;
    bool seed_null = true;
    ZEND_PARSE_PARAMETERS_START(0, 6)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(states)
        Z_PARAM_LONG(max_iterations)
        Z_PARAM_DOUBLE(tolerance)
        Z_PARAM_DOUBLE(smoothing)
        Z_PARAM_DOUBLE(min_variance)
        Z_PARAM_LONG_OR_NULL(seed, seed_null)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        GaussianModel fresh;
        configure(fresh.core.params, states, max_iterations, tolerance, smoothing, seed_null, seed);
        fresh.core.min_variance = strictly_positive(min_variance, "minVariance");
        reset(ZEND_THIS, std::move(fresh));
    });
}

void ZEND_FASTCALL gaussian_means(INTERNAL_FUNCTION_PARAMETERS) {
    ZEND_PARSE_PARAMETERS_NONE();
    fitted_getter<GaussianModel>(ZEND_THIS, return_value,
                                 [](const GaussianModel &m, zval *out) { write_keyed_rows(out, m.core.means, m.states, nullptr); });
}

void ZEND_FASTCALL gaussian_variances(INTERNAL_FUNCTION_PARAMETERS) {
    ZEND_PARSE_PARAMETERS_NONE();
    fitted_getter<GaussianModel>(ZEND_THIS, return_value, [](const GaussianModel &m, zval *out) {
        write_keyed_rows(out, m.core.variances, m.states, nullptr);
    });
}

const zend_function_entry gaussian_methods[] = {
    ML_METHOD(__construct, gaussian_construct, arginfo_gaussian_construct)
    ML_HMM_COMMON(GaussianModel)
    ML_METHOD(getMeans, gaussian_means, arginfo_ml_get_array)
    ML_METHOD(getVariances, gaussian_variances, arginfo_ml_get_array)
    PHP_FE_END};

}  // namespace

void mlphp::register_sequence() {
    register_class<DiscreteModel>("ML\\Sequence\\HiddenMarkovModel", discrete_methods);
    register_class<GaussianModel>("ML\\Sequence\\GaussianHMM", gaussian_methods);
}
