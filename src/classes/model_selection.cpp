// GridSearch and BayesianSearch: hyper-parameter tuning by cross-validation.
// Both instantiate the target class with named constructor arguments, score
// each candidate with a validator, then refit the best one on all data.

#include <numeric>

#include "binding.hpp"
#include "metric_eval.hpp"

namespace {

using namespace mlphp;

struct Owned {
    zval v;
    Owned() { ZVAL_UNDEF(&v); }
    ~Owned() { zval_ptr_dtor(&v); }
    Owned(const Owned &) = delete;
    Owned &operator=(const Owned &) = delete;
};

zend_class_entry *learner_class(zend_string *name) {
    zend_class_entry *ce = zend_lookup_class(name);
    if (!ce) {
        throw std::invalid_argument("Class \"" + zstr(name) + "\" does not exist");
    }
    if (!instanceof_function(ce, ce_learner)) {
        throw std::invalid_argument("Class \"" + zstr(name) + "\" must implement ML\\Learner");
    }
    return ce;
}

void set_common(zval *self, zend_string *name, zval *validator, zval *metric) {
    zend_class_entry *ce = learner_class(name);
    Metric probe;  // validates the metric now rather than at fit time
    probe.configure(metric, ce);
    if (validator && Z_TYPE_P(validator) != IS_NULL) {
        require_instance(validator, ce_validator, "validator");
    }
    zval v;
    ZVAL_STR_COPY(&v, ce->name);
    write_prop(self, "class", &v);
    zval_ptr_dtor(&v);
    if (validator) {
        write_prop(self, "validator", validator);
    }
    if (metric) {
        write_prop(self, "metric", metric);
    }
}

// Scores one parameter set with the validator; returns the mean score.
double evaluate(zend_class_entry *ce, HashTable *params, zval *validator, HashTable *samples, HashTable *targets,
                const Metric &metric) {
    Owned model;
    instantiate(ce, params, &model.v);
    const std::vector<double> scores = cross_validate(validator, &model.v, samples, targets, metric);
    return std::accumulate(scores.begin(), scores.end(), 0.0) / static_cast<double>(scores.size());
}

void default_validator(zval *self, zval *out) {
    zval *validator = read_prop(self, "validator");
    if (Z_TYPE_P(validator) == IS_OBJECT) {
        ZVAL_COPY(out, validator);
        return;
    }
    zend_class_entry *kfold = zend_lookup_class_ex(zend_string_init_interned("ML\\CrossValidation\\KFold", sizeof("ML\\CrossValidation\\KFold") - 1, 1), nullptr, 0);
    HashTable *named = zend_new_array(1);
    zval k;
    ZVAL_LONG(&k, 3);
    zend_hash_str_update(named, "k", 1, &k);
    try {
        instantiate(kfold, named, out);
    } catch (...) {
        zend_array_destroy(named);
        throw;
    }
    zend_array_destroy(named);
}

// Records results, picks the best and refits it on all the data.
void finish(zval *self, zend_class_entry *ce, std::vector<std::pair<zval, double>> &trials, bool higher_is_better,
            HashTable *samples, HashTable *targets) {
    Owned results;
    array_init(&results.v);
    std::size_t best = 0;
    for (std::size_t i = 0; i < trials.size(); ++i) {
        const double s = trials[i].second;
        if (higher_is_better ? s > trials[best].second : s < trials[best].second) {
            best = i;
        }
    }
    for (std::size_t i = 0; i < trials.size(); ++i) {
        zval row;
        array_init(&row);
        Z_ADDREF(trials[i].first);
        add_assoc_zval(&row, "params", &trials[i].first);
        add_assoc_double(&row, "score", trials[i].second);
        add_next_index_zval(&results.v, &row);
    }
    Owned estimator, ignored, x, y;
    instantiate(ce, Z_ARRVAL(trials[best].first), &estimator.v);
    array_zval(samples, &x.v);
    array_zval(targets, &y.v);
    call_method(&estimator.v, "fit", &ignored.v, {&x.v, &y.v});

    zval best_row;
    array_init(&best_row);
    Z_ADDREF(trials[best].first);
    add_assoc_zval(&best_row, "params", &trials[best].first);
    add_assoc_double(&best_row, "score", trials[best].second);
    write_prop(self, "results", &results.v);
    write_prop(self, "best", &best_row);
    write_prop(self, "estimator", &estimator.v);
    zval_ptr_dtor(&best_row);
}

void destroy_trials(std::vector<std::pair<zval, double>> &trials) {
    for (auto &t : trials) {
        zval_ptr_dtor(&t.first);
    }
    trials.clear();
}

// ---- GridSearch ------------------------------------------------------------------------

ZEND_BEGIN_ARG_INFO_EX(arginfo_grid_construct, 0, 0, 2)
    ZEND_ARG_TYPE_INFO(0, class, IS_STRING, 0)
    ZEND_ARG_TYPE_INFO(0, grid, IS_ARRAY, 0)
    ZEND_ARG_OBJ_INFO_WITH_DEFAULT_VALUE(0, validator, ML\\CrossValidation\\Validator, 1, "null")
    ZEND_ARG_INFO_WITH_DEFAULT_VALUE(0, metric, "null")
ZEND_END_ARG_INFO()

void ZEND_FASTCALL grid_construct(INTERNAL_FUNCTION_PARAMETERS) {
    zend_string *name;
    HashTable *grid;
    zval *validator = nullptr, *metric = nullptr;
    ZEND_PARSE_PARAMETERS_START(2, 4)
        Z_PARAM_STR(name)
        Z_PARAM_ARRAY_HT(grid)
        Z_PARAM_OPTIONAL
        Z_PARAM_ZVAL(validator)
        Z_PARAM_ZVAL(metric)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        set_common(ZEND_THIS, name, validator, metric);
        Owned normalized;
        array_init(&normalized.v);
        zend_string *key;
        zval *values;
        ZEND_HASH_FOREACH_STR_KEY_VAL(grid, key, values) {
            check(key != nullptr, "grid keys must be constructor parameter names");
            ZVAL_DEREF(values);
            zval list;
            if (Z_TYPE_P(values) == IS_ARRAY) {
                check(zend_hash_num_elements(Z_ARRVAL_P(values)) > 0, "Every grid entry needs at least one value");
                ZVAL_COPY(&list, values);
            } else {
                array_init(&list);
                Z_TRY_ADDREF_P(values);
                add_next_index_zval(&list, values);
            }
            zend_hash_update(Z_ARRVAL(normalized.v), key, &list);
        }
        ZEND_HASH_FOREACH_END();
        write_prop(ZEND_THIS, "grid", &normalized.v);
    });
}

void ZEND_FASTCALL grid_fit(INTERNAL_FUNCTION_PARAMETERS) {
    HashTable *samples, *targets;
    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_ARRAY_HT(samples)
        Z_PARAM_ARRAY_HT(targets)
    ZEND_PARSE_PARAMETERS_END();
    std::vector<std::pair<zval, double>> trials;
    const bool ok = guard([&] {
        zend_class_entry *ce = learner_class(Z_STR_P(read_prop(ZEND_THIS, "class")));
        Metric metric;
        metric.configure(read_prop(ZEND_THIS, "metric"), ce);
        Owned validator;
        default_validator(ZEND_THIS, &validator.v);

        HashTable *grid = Z_ARRVAL_P(read_prop(ZEND_THIS, "grid"));
        std::vector<zend_string *> names;
        std::vector<std::vector<zval *>> values;
        zend_string *key;
        zval *list;
        ZEND_HASH_FOREACH_STR_KEY_VAL(grid, key, list) {
            names.push_back(key);
            values.push_back(values_of(Z_ARRVAL_P(list)));
        }
        ZEND_HASH_FOREACH_END();

        std::vector<std::size_t> digit(names.size(), 0);
        while (true) {
            zval params;
            array_init(&params);
            for (std::size_t i = 0; i < names.size(); ++i) {
                Z_TRY_ADDREF_P(values[i][digit[i]]);
                zend_hash_update(Z_ARRVAL(params), names[i], values[i][digit[i]]);
            }
            trials.emplace_back(params, 0.0);
            trials.back().second = evaluate(ce, Z_ARRVAL(params), &validator.v, samples, targets, metric);
            std::size_t i = 0;
            while (i < digit.size() && ++digit[i] == values[i].size()) {
                digit[i++] = 0;
            }
            if (i == digit.size()) {
                break;
            }
        }
        finish(ZEND_THIS, ce, trials, metric.higher_is_better, samples, targets);
    });
    destroy_trials(trials);
    if (!ok) {
        RETURN_THROWS();
    }
    RETURN_OBJ_COPY(Z_OBJ_P(ZEND_THIS));
}

// ---- BayesianSearch -------------------------------------------------------------------------

struct Dimension {
    zend_string *name = nullptr;
    enum Kind { Constant, Real, Integer, Choice } kind = Constant;
    double lo = 0, hi = 0;
    bool log = false;
    std::vector<zval *> choices;
    zval *constant = nullptr;
};

std::vector<Dimension> read_space(HashTable *space) {
    std::vector<Dimension> dims;
    zend_string *key;
    zval *spec;
    ZEND_HASH_FOREACH_STR_KEY_VAL(space, key, spec) {
        check(key != nullptr, "space keys must be constructor parameter names");
        ZVAL_DEREF(spec);
        Dimension d;
        d.name = key;
        d.kind = Dimension::Constant;
        if (Z_TYPE_P(spec) != IS_ARRAY) {
            d.constant = spec;
        } else if (zend_hash_str_exists(Z_ARRVAL_P(spec), "min", 3) && zend_hash_str_exists(Z_ARRVAL_P(spec), "max", 3)) {
            zval *lo = zend_hash_str_find_deref(Z_ARRVAL_P(spec), "min", 3);
            zval *hi = zend_hash_str_find_deref(Z_ARRVAL_P(spec), "max", 3);
            zval *log = zend_hash_str_find_deref(Z_ARRVAL_P(spec), "log", 3);
            check((Z_TYPE_P(lo) == IS_LONG || Z_TYPE_P(lo) == IS_DOUBLE) && (Z_TYPE_P(hi) == IS_LONG || Z_TYPE_P(hi) == IS_DOUBLE),
                  "Range bounds must be numbers");
            d.kind = Z_TYPE_P(lo) == IS_LONG && Z_TYPE_P(hi) == IS_LONG ? Dimension::Integer : Dimension::Real;
            d.lo = zval_get_double(lo);
            d.hi = zval_get_double(hi);
            d.log = log && zend_is_true(log);
            check(d.lo < d.hi, "Range min must be less than max");
            check(!d.log || d.lo > 0, "A log-scaled range needs min > 0");
        } else {
            d.kind = Dimension::Choice;
            d.choices = values_of(Z_ARRVAL_P(spec));
            check(!d.choices.empty(), "A list of choices must not be empty");
        }
        dims.push_back(d);
    }
    ZEND_HASH_FOREACH_END();
    return dims;
}

// Decodes a point in [0,1]^p into named constructor arguments.
void decode(const std::vector<Dimension> &dims, const std::vector<double> &u, zval *out) {
    array_init(out);
    std::size_t j = 0;
    for (const Dimension &d : dims) {
        zval v;
        if (d.kind == Dimension::Constant) {
            ZVAL_COPY(&v, d.constant);
        } else {
            const double t = std::min(1.0, std::max(0.0, u[j++]));
            if (d.kind == Dimension::Choice) {
                const std::size_t i = std::min(d.choices.size() - 1, static_cast<std::size_t>(t * static_cast<double>(d.choices.size())));
                ZVAL_COPY(&v, d.choices[i]);
            } else {
                double x = d.log ? std::exp(std::log(d.lo) + t * (std::log(d.hi) - std::log(d.lo))) : d.lo + t * (d.hi - d.lo);
                if (d.kind == Dimension::Integer) {
                    ZVAL_LONG(&v, static_cast<zend_long>(std::llround(x)));
                } else {
                    ZVAL_DOUBLE(&v, x);
                }
            }
        }
        zend_hash_update(Z_ARRVAL_P(out), d.name, &v);
    }
}

// Gaussian-process surrogate (RBF kernel) with Expected Improvement.
struct Surrogate {
    std::vector<std::vector<double>> X;
    std::vector<double> y;  // standardized, higher is better
    std::vector<double> L, alpha;
    double length = 0.3, noise = 1e-6;

    double kernel(const std::vector<double> &a, const std::vector<double> &b) const {
        double s = 0.0;
        for (std::size_t i = 0; i < a.size(); ++i) {
            s += (a[i] - b[i]) * (a[i] - b[i]);
        }
        return std::exp(-s / (2.0 * length * length));
    }

    void fit(const std::vector<std::vector<double>> &points, const std::vector<double> &raw) {
        X = points;
        const std::size_t n = raw.size();
        double mean = std::accumulate(raw.begin(), raw.end(), 0.0) / static_cast<double>(n), sd = 0.0;
        for (double v : raw) {
            sd += (v - mean) * (v - mean);
        }
        sd = std::sqrt(sd / static_cast<double>(n));
        y.resize(n);
        for (std::size_t i = 0; i < n; ++i) {
            y[i] = sd > 0 ? (raw[i] - mean) / sd : 0.0;
        }
        L.assign(n * n, 0.0);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j <= i; ++j) {
                double s = kernel(X[i], X[j]) + (i == j ? noise : 0.0);
                for (std::size_t k = 0; k < j; ++k) {
                    s -= L[i * n + k] * L[j * n + k];
                }
                L[i * n + j] = i == j ? std::sqrt(std::max(s, 1e-12)) : s / L[j * n + j];
            }
        }
        alpha = solve(y);
    }

    std::vector<double> forward(const std::vector<double> &b) const {
        const std::size_t n = b.size();
        std::vector<double> z(n);
        for (std::size_t i = 0; i < n; ++i) {
            double s = b[i];
            for (std::size_t k = 0; k < i; ++k) {
                s -= L[i * n + k] * z[k];
            }
            z[i] = s / L[i * n + i];
        }
        return z;
    }

    std::vector<double> solve(const std::vector<double> &b) const {
        const std::size_t n = b.size();
        std::vector<double> z = forward(b);
        for (std::size_t i = n; i-- > 0;) {
            double s = z[i];
            for (std::size_t k = i + 1; k < n; ++k) {
                s -= L[k * n + i] * z[k];
            }
            z[i] = s / L[i * n + i];
        }
        return z;
    }

    double expected_improvement(const std::vector<double> &x, double best) const {
        std::vector<double> k(X.size());
        for (std::size_t i = 0; i < X.size(); ++i) {
            k[i] = kernel(x, X[i]);
        }
        double mu = 0.0;
        for (std::size_t i = 0; i < k.size(); ++i) {
            mu += k[i] * alpha[i];
        }
        const std::vector<double> v = forward(k);
        double var = 1.0;
        for (double e : v) {
            var -= e * e;
        }
        const double sigma = std::sqrt(std::max(var, 1e-12));
        const double improvement = mu - best - 0.01;
        const double z = improvement / sigma;
        const double cdf = 0.5 * std::erfc(-z / std::sqrt(2.0));
        const double pdf = std::exp(-0.5 * z * z) / std::sqrt(2.0 * 3.14159265358979323846);
        return improvement * cdf + sigma * pdf;
    }
};

ZEND_BEGIN_ARG_INFO_EX(arginfo_bayes_construct, 0, 0, 2)
    ZEND_ARG_TYPE_INFO(0, class, IS_STRING, 0)
    ZEND_ARG_TYPE_INFO(0, space, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, trials, IS_LONG, 0, "10")
    ZEND_ARG_OBJ_INFO_WITH_DEFAULT_VALUE(0, validator, ML\\CrossValidation\\Validator, 1, "null")
    ZEND_ARG_INFO_WITH_DEFAULT_VALUE(0, metric, "null")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, seed, IS_LONG, 1, "null")
ZEND_END_ARG_INFO()

void ZEND_FASTCALL bayes_construct(INTERNAL_FUNCTION_PARAMETERS) {
    zend_string *name;
    HashTable *space;
    zend_long trials = 10, seed = 0;
    zval *validator = nullptr, *metric = nullptr;
    bool seed_null = true;
    ZEND_PARSE_PARAMETERS_START(2, 6)
        Z_PARAM_STR(name)
        Z_PARAM_ARRAY_HT(space)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(trials)
        Z_PARAM_ZVAL(validator)
        Z_PARAM_ZVAL(metric)
        Z_PARAM_LONG_OR_NULL(seed, seed_null)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        set_common(ZEND_THIS, name, validator, metric);
        read_space(space);  // validate now
        zval v;
        ZVAL_ARR(&v, zend_array_dup(space));
        write_prop(ZEND_THIS, "space", &v);
        zval_ptr_dtor(&v);
        ZVAL_LONG(&v, static_cast<zend_long>(positive(trials, "trials")));
        write_prop(ZEND_THIS, "trials", &v);
        if (!seed_null) {
            ZVAL_LONG(&v, seed);
            write_prop(ZEND_THIS, "seed", &v);
        }
    });
}

void ZEND_FASTCALL bayes_fit(INTERNAL_FUNCTION_PARAMETERS) {
    HashTable *samples, *targets;
    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_ARRAY_HT(samples)
        Z_PARAM_ARRAY_HT(targets)
    ZEND_PARSE_PARAMETERS_END();
    std::vector<std::pair<zval, double>> trials;
    const bool ok = guard([&] {
        zend_class_entry *ce = learner_class(Z_STR_P(read_prop(ZEND_THIS, "class")));
        Metric metric;
        metric.configure(read_prop(ZEND_THIS, "metric"), ce);
        Owned validator;
        default_validator(ZEND_THIS, &validator.v);
        const std::vector<Dimension> dims = read_space(Z_ARRVAL_P(read_prop(ZEND_THIS, "space")));
        const std::size_t p = static_cast<std::size_t>(std::count_if(dims.begin(), dims.end(), [](const Dimension &d) { return d.kind != Dimension::Constant; }));
        const std::size_t budget = static_cast<std::size_t>(zval_get_long(read_prop(ZEND_THIS, "trials")));
        zval *seed = read_prop(ZEND_THIS, "seed");
        mlcore::Rng rng = mlcore::make_rng(Z_TYPE_P(seed) == IS_LONG, static_cast<std::uint64_t>(zval_get_long(seed)));
        std::uniform_real_distribution<double> unit(0.0, 1.0);
        const std::size_t initial = std::min<std::size_t>(budget, std::max<std::size_t>(2, p + 1));

        std::vector<std::vector<double>> points;
        std::vector<double> oriented;  // scores where higher is always better
        for (std::size_t t = 0; t < budget; ++t) {
            std::vector<double> u(p);
            if (t < initial || p == 0) {
                for (double &x : u) {
                    x = unit(rng);
                }
            } else {
                Surrogate gp;
                gp.fit(points, oriented);
                const double best = *std::max_element(gp.y.begin(), gp.y.end());
                double best_ei = -1.0;
                std::vector<double> candidate(p);
                for (int c = 0; c < 2000; ++c) {
                    for (double &x : candidate) {
                        x = unit(rng);
                    }
                    const double ei = gp.expected_improvement(candidate, best);
                    if (ei > best_ei) {
                        best_ei = ei;
                        u = candidate;
                    }
                }
            }
            zval params;
            decode(dims, u, &params);
            trials.emplace_back(params, 0.0);
            const double score = evaluate(ce, Z_ARRVAL(params), &validator.v, samples, targets, metric);
            trials.back().second = score;
            points.push_back(u);
            oriented.push_back(metric.higher_is_better ? score : -score);
        }
        finish(ZEND_THIS, ce, trials, metric.higher_is_better, samples, targets);
    });
    destroy_trials(trials);
    if (!ok) {
        RETURN_THROWS();
    }
    RETURN_OBJ_COPY(Z_OBJ_P(ZEND_THIS));
}

// ---- shared delegation -------------------------------------------------------------------------

template <int Which>  // 0 predict, 1 predictProbability, 2 score
void ZEND_FASTCALL search_delegate(INTERNAL_FUNCTION_PARAMETERS) {
    zval *samples;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY(samples)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        zval *estimator = read_prop(ZEND_THIS, "estimator");
        require_fitted(Z_TYPE_P(estimator) == IS_OBJECT, ZEND_THIS);
        static const char *methods[] = {"predict", "predictProbability", "score"};
        if (Which == 1) {
            require_instance(estimator, ce_probabilistic, "The best estimator");
        } else if (Which == 2) {
            require_instance(estimator, ce_anomaly, "The best estimator");
        }
        call_method(estimator, methods[Which], return_value, {samples});
    });
}

template <int Which>  // 0 best, 1 results, 2 estimator
void ZEND_FASTCALL search_getter(INTERNAL_FUNCTION_PARAMETERS) {
    ZEND_PARSE_PARAMETERS_NONE();
    static const char *props[] = {"best", "results", "estimator"};
    guard([&] {
        zval *v = read_prop(ZEND_THIS, props[Which]);
        require_fitted(Z_TYPE_P(v) != IS_NULL, ZEND_THIS);
        ZVAL_COPY(return_value, v);
    });
}

void ZEND_FASTCALL search_clone(INTERNAL_FUNCTION_PARAMETERS) {
    ZEND_PARSE_PARAMETERS_NONE();
    guard([&] {
        deep_clone_prop(ZEND_THIS, "validator");
        deep_clone_prop(ZEND_THIS, "estimator");
    });
}

ZEND_BEGIN_ARG_INFO_EX(arginfo_samples, 0, 0, 1)
    ZEND_ARG_TYPE_INFO(0, samples, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_get_estimator, 0, 0, ML\\Estimator, 0)
ZEND_END_ARG_INFO()

#define ML_SEARCH_COMMON                                                       \
    ML_METHOD(predict, search_delegate<0>, arginfo_samples)                    \
    ML_METHOD(predictProbability, search_delegate<1>, arginfo_ml_samples_to_array) \
    ML_METHOD(score, search_delegate<2>, arginfo_samples)                      \
    ML_METHOD(getBest, search_getter<0>, arginfo_ml_get_array)                 \
    ML_METHOD(getResults, search_getter<1>, arginfo_ml_get_array)              \
    ML_METHOD(getEstimator, search_getter<2>, arginfo_get_estimator)           \
    ML_METHOD(__clone, search_clone, arginfo_ml_clone)

const zend_function_entry grid_methods[] = {
    ML_METHOD(__construct, grid_construct, arginfo_grid_construct)
    ML_METHOD(fit, grid_fit, arginfo_ml_fit_supervised)
    ML_SEARCH_COMMON
    PHP_FE_END};

const zend_function_entry bayes_methods[] = {
    ML_METHOD(__construct, bayes_construct, arginfo_bayes_construct)
    ML_METHOD(fit, bayes_fit, arginfo_ml_fit_supervised)
    ML_SEARCH_COMMON
    PHP_FE_END};

void declare(zend_class_entry *ce, std::initializer_list<const char *> names) {
    for (const char *name : names) {
        zend_declare_property_null(ce, name, std::strlen(name), ZEND_ACC_PRIVATE);
    }
}

}  // namespace

void mlphp::register_model_selection() {
    zend_class_entry *grid = register_plain_class("ML\\ModelSelection\\GridSearch", grid_methods, {ce_learner, ce_estimator});
    declare(grid, {"class", "grid", "validator", "metric", "results", "best", "estimator"});
    zend_class_entry *bayes = register_plain_class("ML\\ModelSelection\\BayesianSearch", bayes_methods, {ce_learner, ce_estimator});
    declare(bayes, {"class", "space", "trials", "validator", "metric", "seed", "results", "best", "estimator"});
}
