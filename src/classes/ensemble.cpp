// Meta estimators that wrap other estimators (any object with the right
// methods, native or user-written PHP): OneVsRest, BootstrapAggregator,
// CommitteeMachine and Pipeline. Their state lives in private properties, so
// default PHP serialization works; __clone deep-copies the wrapped objects.

#include <map>
#include <numeric>

#include "binding.hpp"
#include "metric_eval.hpp"

namespace {

using namespace mlphp;

zend_class_entry *ce_one_vs_rest, *ce_bagging, *ce_committee, *ce_pipeline;

// RAII holder for a zval we own.
struct Owned {
    zval v;
    Owned() { ZVAL_UNDEF(&v); }
    ~Owned() { zval_ptr_dtor(&v); }
    Owned(const Owned &) = delete;
    Owned &operator=(const Owned &) = delete;
    zval *operator&() { return &v; }
};

// Runs `body(list)` on the samples as a list; for a single sample, wraps it
// into a one-element list and returns the first result.
template <typename F>
void with_batch(HashTable *samples, zval *return_value, F body) {
    if (is_sample_list(samples)) {
        zval list;
        array_zval(samples, &list);
        try {
            body(&list, return_value);
        } catch (...) {
            zval_ptr_dtor(&list);
            throw;
        }
        zval_ptr_dtor(&list);
        return;
    }
    Owned list, result;
    array_init(&list);
    zval sample;
    array_zval(samples, &sample);
    add_next_index_zval(&list, &sample);
    body(&list, &result);
    check(Z_TYPE(result.v) == IS_ARRAY && zend_hash_num_elements(Z_ARRVAL(result.v)) == 1, "Estimator returned an unexpected result");
    zval *first = zend_hash_index_find(Z_ARRVAL(result.v), 0);
    check(first != nullptr, "Estimator returned an unexpected result");
    ZVAL_COPY(return_value, first);
}

void predict_into(zval *estimator, zval *samples, zval *out) {
    call_method(estimator, "predict", out, {samples});
    if (Z_TYPE_P(out) != IS_ARRAY) {
        throw std::invalid_argument("predict() of a wrapped estimator must return an array for a list of samples");
    }
}

void fit_estimator(zval *estimator, zval *samples, zval *targets) {
    Owned ignored;
    if (targets && Z_TYPE_P(targets) != IS_NULL) {
        call_method(estimator, "fit", &ignored, {samples, targets});
    } else {
        call_method(estimator, "fit", &ignored, {samples});
    }
}

// Weighted vote over lists of labels (one list per estimator); ties go to the
// label that reached the top score first.
void vote(const std::vector<zval *> &predictions, const std::vector<double> &weights, zval *out) {
    const std::size_t n = zend_hash_num_elements(Z_ARRVAL_P(predictions[0]));
    std::vector<std::vector<zval *>> values;
    for (zval *p : predictions) {
        values.push_back(values_of(Z_ARRVAL_P(p)));
        check(values.back().size() == n, "Wrapped estimators returned different numbers of predictions");
    }
    array_init_size(out, static_cast<uint32_t>(n));
    for (std::size_t i = 0; i < n; ++i) {
        std::map<Label, double> score;
        Label best_label;
        double best = -1.0;
        zval *winner = nullptr;
        for (std::size_t e = 0; e < values.size(); ++e) {
            const Label l = read_label(values[e][i], i);
            double &s = score[l];
            s += weights[e];
            if (s > best) {
                best = s;
                winner = values[e][i];
            }
        }
        Z_TRY_ADDREF_P(winner);
        add_next_index_zval(out, winner);
    }
}

void average(const std::vector<zval *> &predictions, const std::vector<double> &weights, zval *out) {
    std::vector<double> sum;
    double total = 0.0;
    for (std::size_t e = 0; e < predictions.size(); ++e) {
        const std::vector<double> p = read_numbers(Z_ARRVAL_P(predictions[e]), "Prediction");
        if (sum.empty()) {
            sum.assign(p.size(), 0.0);
        }
        check(p.size() == sum.size(), "Wrapped estimators returned different numbers of predictions");
        for (std::size_t i = 0; i < p.size(); ++i) {
            sum[i] += weights[e] * p[i];
        }
        total += weights[e];
    }
    for (double &v : sum) {
        v /= total;
    }
    write_doubles(out, sum);
}

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_MASK_EX(arginfo_fit_optional, 0, 1, MAY_BE_STATIC)
    ZEND_ARG_TYPE_INFO(0, samples, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, targets, IS_ARRAY, 1, "null")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_predict_mixed, 0, 0, 1)
    ZEND_ARG_TYPE_INFO(0, samples, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

zval *fitted_prop(zval *self, const char *name) {
    zval *v = read_prop(self, name);
    if (Z_TYPE_P(v) == IS_NULL || (Z_TYPE_P(v) == IS_ARRAY && zend_hash_num_elements(Z_ARRVAL_P(v)) == 0)) {
        require_fitted(false, self);
    }
    return v;
}

// ---- OneVsRest -------------------------------------------------------------------------

ZEND_BEGIN_ARG_INFO_EX(arginfo_ovr_construct, 0, 0, 1)
    ZEND_ARG_OBJ_INFO(0, base, ML\\Classifier, 0)
ZEND_END_ARG_INFO()

void ZEND_FASTCALL ovr_construct(INTERNAL_FUNCTION_PARAMETERS) {
    zval *base;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_OBJECT_OF_CLASS(base, ce_classifier)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        require_instance(base, ce_learner, "base");
        write_prop(ZEND_THIS, "base", base);
        zval empty;
        array_init(&empty);
        write_prop(ZEND_THIS, "classes", &empty);
        write_prop(ZEND_THIS, "estimators", &empty);
        zval_ptr_dtor(&empty);
    });
}

void ZEND_FASTCALL ovr_fit(INTERNAL_FUNCTION_PARAMETERS) {
    HashTable *samples, *labels;
    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_ARRAY_HT(samples)
        Z_PARAM_ARRAY_HT(labels)
    ZEND_PARSE_PARAMETERS_END();
    if (!guard([&] {
            zval *base = read_prop(ZEND_THIS, "base");
            LabelEncoder encoder;
            const std::vector<int> y = encoder.fit(labels);
            check(encoder.size() >= 2, "OneVsRest needs at least two distinct classes");
            check(y.size() == zend_hash_num_elements(samples), "Number of samples does not match number of labels");
            Owned estimators, classes, X;
            array_init(&estimators);
            array_zval(samples, &X);
            for (std::size_t k = 0; k < encoder.size(); ++k) {
                Owned binary, model;
                array_init_size(&binary, static_cast<uint32_t>(y.size()));
                for (int v : y) {
                    add_next_index_long(&binary, v == static_cast<int>(k) ? 1 : 0);
                }
                clone_into(base, &model);
                fit_estimator(&model, &X, &binary);
                Z_ADDREF(model.v);
                add_next_index_zval(&estimators, &model);
            }
            encoder.write_all(&classes);
            write_prop(ZEND_THIS, "estimators", &estimators);
            write_prop(ZEND_THIS, "classes", &classes);
        })) {
        RETURN_THROWS();
    }
    RETURN_OBJ_COPY(Z_OBJ_P(ZEND_THIS));
}

// Per-sample scores (one column per class) from each binary estimator.
std::vector<std::vector<double>> ovr_scores(zval *self, zval *list) {
    zval *estimators = fitted_prop(self, "estimators");
    std::vector<std::vector<double>> scores;
    zval *est;
    ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(estimators), est) {
        Owned out;
        std::vector<double> column;
        if (instance_of(est, ce_probabilistic)) {
            call_method(est, "predictProbability", &out, {list});
            check(Z_TYPE(out.v) == IS_ARRAY, "predictProbability() must return an array");
            zval *row;
            ZEND_HASH_FOREACH_VAL(Z_ARRVAL(out.v), row) {
                ZVAL_DEREF(row);
                zval *p = Z_TYPE_P(row) == IS_ARRAY ? zend_hash_index_find(Z_ARRVAL_P(row), 1) : nullptr;
                column.push_back(p ? zval_get_double(p) : 0.0);
            }
            ZEND_HASH_FOREACH_END();
        } else {
            predict_into(est, list, &out);
            column = read_numbers(Z_ARRVAL(out.v), "Prediction");
        }
        scores.push_back(std::move(column));
    }
    ZEND_HASH_FOREACH_END();
    return scores;
}

void ZEND_FASTCALL ovr_predict(INTERNAL_FUNCTION_PARAMETERS) {
    HashTable *samples;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY_HT(samples)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        with_batch(samples, return_value, [&](zval *list, zval *out) {
            const auto scores = ovr_scores(ZEND_THIS, list);
            const std::vector<zval *> classes = values_of(Z_ARRVAL_P(read_prop(ZEND_THIS, "classes")));
            const std::size_t n = scores[0].size();
            array_init_size(out, static_cast<uint32_t>(n));
            for (std::size_t i = 0; i < n; ++i) {
                std::size_t best = 0;
                for (std::size_t k = 1; k < scores.size(); ++k) {
                    if (scores[k][i] > scores[best][i]) {
                        best = k;
                    }
                }
                Z_TRY_ADDREF_P(classes[best]);
                add_next_index_zval(out, classes[best]);
            }
        });
    });
}

void ZEND_FASTCALL ovr_proba(INTERNAL_FUNCTION_PARAMETERS) {
    HashTable *samples;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY_HT(samples)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        with_batch(samples, return_value, [&](zval *list, zval *out) {
            const auto scores = ovr_scores(ZEND_THIS, list);
            const std::vector<zval *> classes = values_of(Z_ARRVAL_P(read_prop(ZEND_THIS, "classes")));
            const std::size_t n = scores[0].size(), K = scores.size();
            array_init_size(out, static_cast<uint32_t>(n));
            for (std::size_t i = 0; i < n; ++i) {
                double total = 0.0;
                for (std::size_t k = 0; k < K; ++k) {
                    total += std::max(0.0, scores[k][i]);
                }
                zval row;
                array_init_size(&row, static_cast<uint32_t>(K));
                for (std::size_t k = 0; k < K; ++k) {
                    zval p;
                    ZVAL_DOUBLE(&p, total > 0 ? std::max(0.0, scores[k][i]) / total : 1.0 / static_cast<double>(K));
                    array_set_zval_key(Z_ARRVAL(row), classes[k], &p);
                }
                add_next_index_zval(out, &row);
            }
        });
    });
}

void ZEND_FASTCALL ovr_classes(INTERNAL_FUNCTION_PARAMETERS) {
    ZEND_PARSE_PARAMETERS_NONE();
    guard([&] { ZVAL_COPY(return_value, fitted_prop(ZEND_THIS, "classes")); });
}

void ZEND_FASTCALL ovr_clone(INTERNAL_FUNCTION_PARAMETERS) {
    ZEND_PARSE_PARAMETERS_NONE();
    guard([&] {
        deep_clone_prop(ZEND_THIS, "base");
        deep_clone_prop(ZEND_THIS, "estimators");
    });
}

const zend_function_entry ovr_methods[] = {
    ML_METHOD(__construct, ovr_construct, arginfo_ovr_construct)
    ML_METHOD(fit, ovr_fit, arginfo_ml_fit_supervised)
    ML_METHOD(predict, ovr_predict, arginfo_ml_predict_label)
    ML_METHOD(predictProbability, ovr_proba, arginfo_ml_samples_to_array)
    ML_METHOD(getClasses, ovr_classes, arginfo_ml_get_array)
    ML_METHOD(__clone, ovr_clone, arginfo_ml_clone)
    PHP_FE_END};

// ---- BootstrapAggregator -------------------------------------------------------------------

ZEND_BEGIN_ARG_INFO_EX(arginfo_bagging_construct, 0, 0, 1)
    ZEND_ARG_OBJ_INFO(0, base, ML\\Learner, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, estimators, IS_LONG, 0, "10")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, ratio, IS_DOUBLE, 0, "0.5")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, seed, IS_LONG, 1, "null")
ZEND_END_ARG_INFO()

void ZEND_FASTCALL bagging_construct(INTERNAL_FUNCTION_PARAMETERS) {
    zval *base;
    zend_long estimators = 10, seed = 0;
    double ratio = 0.5;
    bool seed_null = true;
    ZEND_PARSE_PARAMETERS_START(1, 4)
        Z_PARAM_OBJECT_OF_CLASS(base, ce_learner)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(estimators)
        Z_PARAM_DOUBLE(ratio)
        Z_PARAM_LONG_OR_NULL(seed, seed_null)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        require_instance(base, ce_estimator, "base");
        positive(estimators, "estimators");
        check(ratio > 0.0 && ratio <= 1.5, "ratio must be greater than 0 and at most 1.5");
        zval v;
        write_prop(ZEND_THIS, "base", base);
        ZVAL_LONG(&v, estimators);
        write_prop(ZEND_THIS, "count", &v);
        ZVAL_DOUBLE(&v, ratio);
        write_prop(ZEND_THIS, "ratio", &v);
        if (!seed_null) {
            ZVAL_LONG(&v, seed);
            write_prop(ZEND_THIS, "seed", &v);
        }
        array_init(&v);
        write_prop(ZEND_THIS, "estimators", &v);
        zval_ptr_dtor(&v);
    });
}

void ZEND_FASTCALL bagging_fit(INTERNAL_FUNCTION_PARAMETERS) {
    HashTable *samples, *targets = nullptr;
    ZEND_PARSE_PARAMETERS_START(1, 2)
        Z_PARAM_ARRAY_HT(samples)
        Z_PARAM_OPTIONAL
        Z_PARAM_ARRAY_HT_OR_NULL(targets)
    ZEND_PARSE_PARAMETERS_END();
    if (!guard([&] {
            zval *base = read_prop(ZEND_THIS, "base");
            const zend_long count = zval_get_long(read_prop(ZEND_THIS, "count"));
            const double ratio = zval_get_double(read_prop(ZEND_THIS, "ratio"));
            zval *seed = read_prop(ZEND_THIS, "seed");
            const std::vector<zval *> X = values_of(samples);
            std::vector<zval *> Y;
            if (targets) {
                Y = values_of(targets);
                check(Y.size() == X.size(), "Number of samples does not match number of targets");
            }
            check(!X.empty(), "Samples must not be empty");
            mlcore::Rng rng = mlcore::make_rng(Z_TYPE_P(seed) == IS_LONG, static_cast<std::uint64_t>(zval_get_long(seed)));
            const std::size_t size = std::max<std::size_t>(1, static_cast<std::size_t>(std::llround(ratio * static_cast<double>(X.size()))));
            std::uniform_int_distribution<std::size_t> pick(0, X.size() - 1);
            Owned estimators;
            array_init(&estimators);
            for (zend_long e = 0; e < count; ++e) {
                std::vector<std::size_t> rows(size);
                for (std::size_t &r : rows) {
                    r = pick(rng);
                }
                Owned model, bx, by;
                clone_into(base, &model);
                mlphp::pick(X, rows, &bx);
                if (targets) {
                    mlphp::pick(Y, rows, &by);
                    fit_estimator(&model, &bx, &by);
                } else {
                    fit_estimator(&model, &bx, nullptr);
                }
                Z_ADDREF(model.v);
                add_next_index_zval(&estimators, &model);
            }
            write_prop(ZEND_THIS, "estimators", &estimators);
        })) {
        RETURN_THROWS();
    }
    RETURN_OBJ_COPY(Z_OBJ_P(ZEND_THIS));
}

// Combines the predictions of `estimators` (an array property) with `weights`.
void combine(zval *estimators, const std::vector<double> &weights, bool regression, zval *list, zval *out) {
    std::vector<Owned> predictions(zend_hash_num_elements(Z_ARRVAL_P(estimators)));
    std::vector<zval *> raw;
    std::size_t i = 0;
    zval *est;
    ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(estimators), est) {
        predict_into(est, list, &predictions[i]);
        raw.push_back(&predictions[i].v);
        ++i;
    }
    ZEND_HASH_FOREACH_END();
    if (regression) {
        average(raw, weights, out);
    } else {
        vote(raw, weights, out);
    }
}

void ZEND_FASTCALL bagging_predict(INTERNAL_FUNCTION_PARAMETERS) {
    HashTable *samples;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY_HT(samples)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        zval *estimators = fitted_prop(ZEND_THIS, "estimators");
        const bool regression = instance_of(read_prop(ZEND_THIS, "base"), ce_regressor);
        const std::vector<double> weights(zend_hash_num_elements(Z_ARRVAL_P(estimators)), 1.0);
        with_batch(samples, return_value, [&](zval *list, zval *out) { combine(estimators, weights, regression, list, out); });
    });
}

void ZEND_FASTCALL bagging_clone(INTERNAL_FUNCTION_PARAMETERS) {
    ZEND_PARSE_PARAMETERS_NONE();
    guard([&] {
        deep_clone_prop(ZEND_THIS, "base");
        deep_clone_prop(ZEND_THIS, "estimators");
    });
}

const zend_function_entry bagging_methods[] = {
    ML_METHOD(__construct, bagging_construct, arginfo_bagging_construct)
    ML_METHOD(fit, bagging_fit, arginfo_fit_optional)
    ML_METHOD(predict, bagging_predict, arginfo_predict_mixed)
    ML_METHOD(__clone, bagging_clone, arginfo_ml_clone)
    PHP_FE_END};

// ---- CommitteeMachine ------------------------------------------------------------------------

ZEND_BEGIN_ARG_INFO_EX(arginfo_committee_construct, 0, 0, 1)
    ZEND_ARG_TYPE_INFO(0, experts, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, influences, IS_ARRAY, 1, "null")
ZEND_END_ARG_INFO()

void ZEND_FASTCALL committee_construct(INTERNAL_FUNCTION_PARAMETERS) {
    HashTable *experts, *influences = nullptr;
    ZEND_PARSE_PARAMETERS_START(1, 2)
        Z_PARAM_ARRAY_HT(experts)
        Z_PARAM_OPTIONAL
        Z_PARAM_ARRAY_HT_OR_NULL(influences)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        const std::vector<zval *> list = values_of(experts);
        check(!list.empty(), "experts must not be empty");
        zend_class_entry *kinds[] = {ce_classifier, ce_regressor, ce_anomaly};
        zend_class_entry *kind = nullptr;
        for (zend_class_entry *k : kinds) {
            if (instance_of(list[0], k)) {
                kind = k;
            }
        }
        check(kind != nullptr, "experts must be classifiers, regressors or anomaly detectors");
        Owned stored;
        array_init(&stored);
        for (zval *e : list) {
            require_instance(e, ce_learner, "Each expert");
            check(instance_of(e, kind), "All experts must be of the same kind (all classifiers, all regressors or all anomaly detectors)");
            Z_ADDREF_P(e);
            add_next_index_zval(&stored, e);
        }
        std::vector<double> w(list.size(), 1.0);
        if (influences) {
            w = read_numbers(influences, "Influence");
            check(w.size() == list.size(), "influences must contain one weight per expert");
            for (double v : w) {
                check(v > 0.0 && std::isfinite(v), "influences must be positive");
            }
        }
        const double total = std::accumulate(w.begin(), w.end(), 0.0);
        for (double &v : w) {
            v /= total;
        }
        Owned weights;
        write_doubles(&weights, w);
        write_prop(ZEND_THIS, "experts", &stored);
        write_prop(ZEND_THIS, "influences", &weights);
    });
}

void ZEND_FASTCALL committee_fit(INTERNAL_FUNCTION_PARAMETERS) {
    HashTable *samples, *targets = nullptr;
    ZEND_PARSE_PARAMETERS_START(1, 2)
        Z_PARAM_ARRAY_HT(samples)
        Z_PARAM_OPTIONAL
        Z_PARAM_ARRAY_HT_OR_NULL(targets)
    ZEND_PARSE_PARAMETERS_END();
    if (!guard([&] {
            Owned X, Y;
            array_zval(samples, &X);
            if (targets) {
                array_zval(targets, &Y);
            }
            zval *expert;
            ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(read_prop(ZEND_THIS, "experts")), expert) {
                fit_estimator(expert, &X, targets ? &Y.v : nullptr);
            }
            ZEND_HASH_FOREACH_END();
            zval yes;
            ZVAL_TRUE(&yes);
            write_prop(ZEND_THIS, "trained", &yes);
        })) {
        RETURN_THROWS();
    }
    RETURN_OBJ_COPY(Z_OBJ_P(ZEND_THIS));
}

void ZEND_FASTCALL committee_predict(INTERNAL_FUNCTION_PARAMETERS) {
    HashTable *samples;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY_HT(samples)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        require_fitted(zend_is_true(read_prop(ZEND_THIS, "trained")), ZEND_THIS);
        zval *experts = read_prop(ZEND_THIS, "experts");
        const std::vector<double> weights = read_numbers(Z_ARRVAL_P(read_prop(ZEND_THIS, "influences")), "Influence");
        const bool regression = instance_of(zend_hash_index_find(Z_ARRVAL_P(experts), 0), ce_regressor);
        with_batch(samples, return_value, [&](zval *list, zval *out) { combine(experts, weights, regression, list, out); });
    });
}

void ZEND_FASTCALL committee_influences(INTERNAL_FUNCTION_PARAMETERS) {
    ZEND_PARSE_PARAMETERS_NONE();
    ZVAL_COPY(return_value, read_prop(ZEND_THIS, "influences"));
}

void ZEND_FASTCALL committee_clone(INTERNAL_FUNCTION_PARAMETERS) {
    ZEND_PARSE_PARAMETERS_NONE();
    guard([&] { deep_clone_prop(ZEND_THIS, "experts"); });
}

const zend_function_entry committee_methods[] = {
    ML_METHOD(__construct, committee_construct, arginfo_committee_construct)
    ML_METHOD(fit, committee_fit, arginfo_fit_optional)
    ML_METHOD(predict, committee_predict, arginfo_predict_mixed)
    ML_METHOD(getInfluences, committee_influences, arginfo_ml_get_array)
    ML_METHOD(__clone, committee_clone, arginfo_ml_clone)
    PHP_FE_END};

// ---- Pipeline -----------------------------------------------------------------------------------

ZEND_BEGIN_ARG_INFO_EX(arginfo_pipeline_construct, 0, 0, 2)
    ZEND_ARG_TYPE_INFO(0, transformers, IS_ARRAY, 0)
    ZEND_ARG_OBJ_INFO(0, estimator, ML\\Estimator, 0)
ZEND_END_ARG_INFO()

void ZEND_FASTCALL pipeline_construct(INTERNAL_FUNCTION_PARAMETERS) {
    HashTable *transformers;
    zval *estimator;
    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_ARRAY_HT(transformers)
        Z_PARAM_OBJECT_OF_CLASS(estimator, ce_estimator)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        Owned list;
        array_init(&list);
        for (zval *t : values_of(transformers)) {
            require_instance(t, ce_transformer, "Each transformer");
            Z_ADDREF_P(t);
            add_next_index_zval(&list, t);
        }
        write_prop(ZEND_THIS, "transformers", &list);
        write_prop(ZEND_THIS, "estimator", estimator);
    });
}

// Passes `samples` through every transformer (fitting stateful ones first when `fit`).
void run_transformers(zval *self, zval *samples, zval *targets, bool fit, zval *out) {
    ZVAL_COPY(out, samples);
    zval *transformer;
    ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(read_prop(self, "transformers")), transformer) {
        if (fit && instance_of(transformer, ce_stateful)) {
            const bool supervised = required_args(Z_OBJCE_P(transformer), "fit") >= 2;
            check(!supervised || (targets && Z_TYPE_P(targets) != IS_NULL),
                  "A supervised transformer in the pipeline needs targets");
            fit_estimator(transformer, out, supervised ? targets : nullptr);
        }
        Owned next;
        call_method(transformer, "transform", &next, {out});
        zval_ptr_dtor(out);
        ZVAL_COPY(out, &next.v);
    }
    ZEND_HASH_FOREACH_END();
}

void ZEND_FASTCALL pipeline_fit(INTERNAL_FUNCTION_PARAMETERS) {
    HashTable *samples, *targets = nullptr;
    ZEND_PARSE_PARAMETERS_START(1, 2)
        Z_PARAM_ARRAY_HT(samples)
        Z_PARAM_OPTIONAL
        Z_PARAM_ARRAY_HT_OR_NULL(targets)
    ZEND_PARSE_PARAMETERS_END();
    if (!guard([&] {
            Owned X, Y, transformed;
            array_zval(samples, &X);
            if (targets) {
                array_zval(targets, &Y);
            }
            zval *estimator = read_prop(ZEND_THIS, "estimator");
            run_transformers(ZEND_THIS, &X, targets ? &Y.v : nullptr, true, &transformed);
            if (instance_of(estimator, ce_learner)) {
                fit_estimator(estimator, &transformed, targets ? &Y.v : nullptr);
            }
            zval yes;
            ZVAL_TRUE(&yes);
            write_prop(ZEND_THIS, "trained", &yes);
        })) {
        RETURN_THROWS();
    }
    RETURN_OBJ_COPY(Z_OBJ_P(ZEND_THIS));
}

template <int Which>  // 0 predict, 1 predictProbability, 2 score, 3 transform
void ZEND_FASTCALL pipeline_call(INTERNAL_FUNCTION_PARAMETERS) {
    HashTable *samples;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY_HT(samples)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        Owned X, transformed;
        array_zval(samples, &X);
        run_transformers(ZEND_THIS, &X, nullptr, false, &transformed);
        if (Which == 3) {
            ZVAL_COPY(return_value, &transformed.v);
            return;
        }
        zval *estimator = read_prop(ZEND_THIS, "estimator");
        static const char *methods[] = {"predict", "predictProbability", "score"};
        if (Which == 1) {
            require_instance(estimator, ce_probabilistic, "The pipeline's estimator");
        } else if (Which == 2) {
            require_instance(estimator, ce_anomaly, "The pipeline's estimator");
        }
        call_method(estimator, methods[Which], return_value, {&transformed});
    });
}

void ZEND_FASTCALL pipeline_estimator(INTERNAL_FUNCTION_PARAMETERS) {
    ZEND_PARSE_PARAMETERS_NONE();
    ZVAL_COPY(return_value, read_prop(ZEND_THIS, "estimator"));
}

void ZEND_FASTCALL pipeline_transformers(INTERNAL_FUNCTION_PARAMETERS) {
    ZEND_PARSE_PARAMETERS_NONE();
    ZVAL_COPY(return_value, read_prop(ZEND_THIS, "transformers"));
}

void ZEND_FASTCALL pipeline_clone(INTERNAL_FUNCTION_PARAMETERS) {
    ZEND_PARSE_PARAMETERS_NONE();
    guard([&] {
        deep_clone_prop(ZEND_THIS, "transformers");
        deep_clone_prop(ZEND_THIS, "estimator");
    });
}

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_get_estimator, 0, 0, ML\\Estimator, 0)
ZEND_END_ARG_INFO()

const zend_function_entry pipeline_methods[] = {
    ML_METHOD(__construct, pipeline_construct, arginfo_pipeline_construct)
    ML_METHOD(fit, pipeline_fit, arginfo_fit_optional)
    ML_METHOD(predict, pipeline_call<0>, arginfo_predict_mixed)
    ML_METHOD(predictProbability, pipeline_call<1>, arginfo_ml_samples_to_array)
    ML_METHOD(score, pipeline_call<2>, arginfo_predict_mixed)
    ML_METHOD(transform, pipeline_call<3>, arginfo_ml_samples_to_array)
    ML_METHOD(getEstimator, pipeline_estimator, arginfo_get_estimator)
    ML_METHOD(getTransformers, pipeline_transformers, arginfo_ml_get_array)
    ML_METHOD(__clone, pipeline_clone, arginfo_ml_clone)
    PHP_FE_END};

void declare(zend_class_entry *ce, std::initializer_list<const char *> names) {
    for (const char *name : names) {
        zend_declare_property_null(ce, name, std::strlen(name), ZEND_ACC_PRIVATE);
    }
}

}  // namespace

zend_class_entry *mlphp::underlying_class(zval *estimator) {
    for (int depth = 0; depth < 16 && estimator && Z_TYPE_P(estimator) == IS_OBJECT; ++depth) {
        zend_class_entry *ce = Z_OBJCE_P(estimator);
        if (ce == ce_pipeline) {
            estimator = read_prop(estimator, "estimator");
        } else if (ce == ce_bagging || ce == ce_one_vs_rest) {
            estimator = read_prop(estimator, "base");
        } else if (ce == ce_committee) {
            zval *experts = read_prop(estimator, "experts");
            estimator = Z_TYPE_P(experts) == IS_ARRAY ? zend_hash_index_find(Z_ARRVAL_P(experts), 0) : nullptr;
        } else {
            return ce;
        }
    }
    return nullptr;
}

void mlphp::register_ensemble() {
    ce_one_vs_rest = register_plain_class("ML\\Ensemble\\OneVsRest", ovr_methods, {ce_learner, ce_classifier, ce_probabilistic});
    declare(ce_one_vs_rest, {"base", "classes", "estimators"});
    ce_bagging = register_plain_class("ML\\Ensemble\\BootstrapAggregator", bagging_methods, {ce_learner, ce_estimator});
    declare(ce_bagging, {"base", "count", "ratio", "seed", "estimators"});
    ce_committee = register_plain_class("ML\\Ensemble\\CommitteeMachine", committee_methods, {ce_learner, ce_estimator});
    declare(ce_committee, {"experts", "influences", "trained"});
}

void mlphp::register_pipeline() {
    ce_pipeline = register_plain_class("ML\\Pipeline", pipeline_methods, {ce_learner, ce_estimator});
    declare(ce_pipeline, {"transformers", "estimator", "trained"});
}
