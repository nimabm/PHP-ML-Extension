// NaiveBayes (categorical features) and MultinomialNB (count features).

#include <map>

#include "binding.hpp"
#include "core/naive_bayes.hpp"

namespace {

using namespace mlphp;
using mlcore::ArchiveReader;
using mlcore::ArchiveWriter;

// Optional user priors as label => probability, resolved to class order at fit time.
using Priors = std::vector<std::pair<Label, double>>;

Priors read_priors(HashTable *ht) {
    Priors priors;
    zend_string *key;
    zend_ulong index;
    zval *value;
    ZEND_HASH_FOREACH_KEY_VAL(ht, index, key, value) {
        ZVAL_DEREF(value);
        check(Z_TYPE_P(value) == IS_LONG || Z_TYPE_P(value) == IS_DOUBLE, "priors must map labels to probabilities");
        const double p = zval_get_double(value);
        check(p > 0.0 && std::isfinite(p), "prior probabilities must be greater than 0");
        Label label = key ? Label(std::in_place_index<1>, zstr(key)) : Label(std::in_place_index<0>, static_cast<zend_long>(index));
        priors.emplace_back(std::move(label), p);
    }
    ZEND_HASH_FOREACH_END();
    check(!priors.empty(), "priors must not be empty");
    return priors;
}

// Resolves priors to the encoder's class order. Array keys turn numeric
// strings into ints, so labels are compared by their string form too.
std::vector<double> resolve_priors(const Priors &priors, const LabelEncoder &labels) {
    if (priors.empty()) {
        return {};
    }
    auto text = [](const Label &l) {
        if (const zend_long *i = std::get_if<zend_long>(&l)) {
            return std::to_string(*i);
        }
        return std::get<std::string>(l);
    };
    std::vector<double> out(labels.size(), -1.0);
    for (const auto &[label, p] : priors) {
        for (std::size_t c = 0; c < labels.size(); ++c) {
            if (text(labels.classes[c]) == text(label)) {
                out[c] = p;
            }
        }
    }
    for (double p : out) {
        check(p > 0.0, "priors must contain a probability for every class in the training labels");
    }
    return out;
}

void save_priors(ArchiveWriter &w, const Priors &priors) {
    w.size(priors.size());
    for (const auto &[label, p] : priors) {
        save_label(w, label);
        w.f64(p);
    }
}

Priors load_priors(ArchiveReader &r) {
    Priors priors(r.count(9));
    for (auto &[label, p] : priors) {
        label = load_label(r);
        p = r.f64();
        ArchiveReader::require(p > 0.0, "bad prior");
    }
    return priors;
}

// ---- NaiveBayes ------------------------------------------------------------------

struct CategoricalModel {
    static constexpr std::uint32_t kTag = mlcore::tag("CTNB");
    mlcore::CategoricalNB core;
    Priors priors;
    std::vector<std::vector<Label>> categories;  // sorted, per feature
    LabelEncoder labels;

    bool fitted() const { return core.fitted(); }
    std::size_t features() const { return core.features; }
    std::size_t classes() const { return core.classes; }

    // Encodes table cells to category indices; unknown categories become -1
    // when `learn` is false.
    std::vector<int> encode(const Table &t) const {
        std::vector<int> out(t.rows * t.cols);
        for (std::size_t r = 0; r < t.rows; ++r) {
            for (std::size_t c = 0; c < t.cols; ++c) {
                const Label label = to_label(t.at(r, c), r, c);
                const auto &cats = categories[c];
                const auto it = std::lower_bound(cats.begin(), cats.end(), label);
                out[r * t.cols + c] = it != cats.end() && *it == label ? static_cast<int>(it - cats.begin()) : -1;
            }
        }
        return out;
    }

    static Label to_label(const Cell &cell, std::size_t r, std::size_t c) {
        switch (cell.kind) {
            case Cell::Kind::Int: return Label(std::in_place_index<0>, cell.i);
            case Cell::Kind::String: return Label(std::in_place_index<1>, cell.s);
            case Cell::Kind::Bool: return Label(std::in_place_index<0>, cell.i);
            default:
                throw std::invalid_argument("Sample #" + std::to_string(r) + ", feature #" + std::to_string(c) +
                                            " must be a categorical value (int, string or bool); use "
                                            "GaussianNB or IntervalDiscretizer for continuous features");
        }
    }

    void save(ArchiveWriter &w) const {
        save_priors(w, priors);
        core.save(w);
        w.size(categories.size());
        for (const auto &cats : categories) {
            w.size(cats.size());
            for (const Label &l : cats) {
                save_label(w, l);
            }
        }
        labels.save(w);
    }
    void load(ArchiveReader &r) {
        priors = load_priors(r);
        core.load(r);
        categories.assign(r.size(core.features), {});
        ArchiveReader::require(categories.size() == core.features);
        for (std::size_t f = 0; f < categories.size(); ++f) {
            const std::size_t n = r.count(9);
            ArchiveReader::require(n == core.categories[f]);
            for (std::size_t i = 0; i < n; ++i) {
                categories[f].push_back(load_label(r));
                ArchiveReader::require(i == 0 || categories[f][i - 1] < categories[f][i], "categories must be sorted");
            }
        }
        labels.load(r);
        ArchiveReader::require(labels.size() == core.classes);
    }
};

ZEND_BEGIN_ARG_INFO_EX(arginfo_nb_construct, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, smoothing, IS_DOUBLE, 0, "1.0")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, priors, IS_ARRAY, 1, "null")
ZEND_END_ARG_INFO()

template <typename M>
void ZEND_FASTCALL nb_construct(INTERNAL_FUNCTION_PARAMETERS) {
    double smoothing = 1.0;
    HashTable *priors = nullptr;
    ZEND_PARSE_PARAMETERS_START(0, 2)
        Z_PARAM_OPTIONAL
        Z_PARAM_DOUBLE(smoothing)
        Z_PARAM_ARRAY_HT_OR_NULL(priors)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        M fresh;
        fresh.core.smoothing = strictly_positive(smoothing, "smoothing");
        if (priors) {
            fresh.priors = read_priors(priors);
        }
        reset(ZEND_THIS, std::move(fresh));
    });
}

void ZEND_FASTCALL categorical_fit(INTERNAL_FUNCTION_PARAMETERS) {
    HashTable *samples, *labels;
    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_ARRAY_HT(samples)
        Z_PARAM_ARRAY_HT(labels)
    ZEND_PARSE_PARAMETERS_END();
    if (!guard([&] {
            CategoricalModel &m = native<CategoricalModel>(ZEND_THIS);
            const Table t = read_table(samples, false);
            LabelEncoder encoder;
            const std::vector<int> y = encoder.fit(labels);
            check(y.size() == t.rows, ("Number of samples (" + std::to_string(t.rows) + ") does not match number of targets (" +
                                       std::to_string(y.size()) + ")").c_str());
            CategoricalModel next = m;
            next.categories.assign(t.cols, {});
            std::vector<std::size_t> counts(t.cols);
            for (std::size_t c = 0; c < t.cols; ++c) {
                std::vector<Label> seen;
                for (std::size_t r = 0; r < t.rows; ++r) {
                    seen.push_back(CategoricalModel::to_label(t.at(r, c), r, c));
                }
                std::sort(seen.begin(), seen.end());
                seen.erase(std::unique(seen.begin(), seen.end()), seen.end());
                counts[c] = seen.size();
                next.categories[c] = std::move(seen);
            }
            const std::vector<int> X = next.encode(t);
            next.core.fixed_priors = resolve_priors(next.priors, encoder);
            next.core.fit(X, t.rows, t.cols, counts, y, encoder.size());
            next.labels = std::move(encoder);
            m = std::move(next);
        })) {
        RETURN_THROWS();
    }
    RETURN_OBJ_COPY(Z_OBJ_P(ZEND_THIS));
}

template <bool Probabilities>
void ZEND_FASTCALL categorical_predict(INTERNAL_FUNCTION_PARAMETERS) {
    HashTable *samples;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY_HT(samples)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        const CategoricalModel &m = native<CategoricalModel>(ZEND_THIS);
        require_fitted(m.fitted(), ZEND_THIS);
        const bool batch = is_sample_list(samples);
        const Table t = read_table(samples, !batch, m.features());
        const std::vector<int> X = m.encode(t);
        const std::size_t K = m.classes();
        auto write_one = [&](std::size_t r, zval *out) {
            const int *x = &X[r * t.cols];
            if (Probabilities) {
                std::vector<double> p(K);
                m.core.probabilities(x, p.data());
                array_init_size(out, static_cast<uint32_t>(K));
                for (std::size_t k = 0; k < K; ++k) {
                    m.labels.write_keyed(out, static_cast<int>(k), p[k]);
                }
            } else {
                m.labels.write(m.core.predict_one(x), out);
            }
        };
        if (!batch) {
            write_one(0, return_value);
            return;
        }
        array_init_size(return_value, static_cast<uint32_t>(t.rows));
        for (std::size_t r = 0; r < t.rows; ++r) {
            zval v;
            write_one(r, &v);
            add_next_index_zval(return_value, &v);
        }
    });
}

const zend_function_entry naive_bayes_methods[] = {
    ML_METHOD(__construct, nb_construct<CategoricalModel>, arginfo_nb_construct)
    ML_METHOD(fit, categorical_fit, arginfo_ml_fit_supervised)
    ML_METHOD(predict, categorical_predict<false>, arginfo_ml_predict_label)
    ML_METHOD(predictProbability, categorical_predict<true>, arginfo_ml_samples_to_array)
    ML_METHOD(getClasses, get_classes<CategoricalModel>, arginfo_ml_get_array)
    ML_METHOD(__serialize, serialize_native<CategoricalModel>, arginfo_ml_get_array)
    ML_METHOD(__unserialize, unserialize_native<CategoricalModel>, arginfo_ml_unserialize)
    PHP_FE_END};

// ---- MultinomialNB -----------------------------------------------------------------

struct MultinomialModel {
    static constexpr std::uint32_t kTag = mlcore::tag("MNNB");
    mlcore::MultinomialNB core;
    Priors priors;
    LabelEncoder labels;

    bool fitted() const { return core.fitted(); }
    std::size_t features() const { return core.features; }
    std::size_t classes() const { return core.classes; }
    void fit(const mlcore::Matrix &X, const std::vector<int> &y, std::size_t K, unsigned) {
        // fit_classifier assigns `labels` after fit, so build a temporary encoder view for priors.
        LabelEncoder encoder;
        encoder.classes = pending_classes;
        core.fixed_priors = resolve_priors(priors, encoder);
        core.fit(X, y, K);
    }
    int predict_one(const double *x) const { return core.predict_one(x); }
    void probabilities(const double *x, double *out) const { core.probabilities(x, out); }

    void save(ArchiveWriter &w) const {
        save_priors(w, priors);
        core.save(w);
        labels.save(w);
    }
    void load(ArchiveReader &r) {
        priors = load_priors(r);
        core.load(r);
        labels.load(r);
        ArchiveReader::require(labels.size() == core.classes);
    }

    std::vector<Label> pending_classes;
};

void ZEND_FASTCALL multinomial_fit(INTERNAL_FUNCTION_PARAMETERS) {
    HashTable *samples, *labels;
    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_ARRAY_HT(samples)
        Z_PARAM_ARRAY_HT(labels)
    ZEND_PARSE_PARAMETERS_END();
    if (!guard([&] {
            MultinomialModel &m = native<MultinomialModel>(ZEND_THIS);
            const mlcore::Matrix X = read_samples(samples);
            LabelEncoder encoder;
            const std::vector<int> y = encoder.fit(labels);
            mlcore::require_training_set(X, y.size());
            MultinomialModel next = m;
            next.pending_classes = encoder.classes;
            next.fit(X, y, encoder.size(), 1);
            next.pending_classes.clear();
            next.labels = std::move(encoder);
            m = std::move(next);
        })) {
        RETURN_THROWS();
    }
    RETURN_OBJ_COPY(Z_OBJ_P(ZEND_THIS));
}

const zend_function_entry multinomial_methods[] = {
    ML_METHOD(__construct, nb_construct<MultinomialModel>, arginfo_nb_construct)
    ML_METHOD(fit, multinomial_fit, arginfo_ml_fit_supervised)
    ML_METHOD(predict, predict_classifier<MultinomialModel>, arginfo_ml_predict_label)
    ML_METHOD(predictProbability, predict_proba<MultinomialModel>, arginfo_ml_samples_to_array)
    ML_METHOD(getClasses, get_classes<MultinomialModel>, arginfo_ml_get_array)
    ML_METHOD(__serialize, serialize_native<MultinomialModel>, arginfo_ml_get_array)
    ML_METHOD(__unserialize, unserialize_native<MultinomialModel>, arginfo_ml_unserialize)
    PHP_FE_END};

}  // namespace

void mlphp::register_naive_bayes() {
    const auto classifier = {ce_learner, ce_classifier, ce_probabilistic};
    register_class<CategoricalModel>("ML\\Classification\\NaiveBayes", naive_bayes_methods, classifier);
    register_class<MultinomialModel>("ML\\Classification\\MultinomialNB", multinomial_methods, classifier);
}
