// ML\CrossValidation: Reports and the HoldOut, KFold, LeavePOut and MonteCarlo validators.

#include <numeric>

#include "binding.hpp"
#include "core/metrics.hpp"
#include "metric_eval.hpp"

namespace mlphp {
zend_class_entry *ce_validator = nullptr;
}

namespace {

using namespace mlphp;
using mlcore::ArchiveReader;
using mlcore::ArchiveWriter;

// ---- Reports ------------------------------------------------------------------------

void require_pairs(std::size_t a, std::size_t b) {
    check(a > 0, "Arrays must not be empty");
    if (a != b) {
        throw std::invalid_argument("Expected (" + std::to_string(a) + ") and predicted (" + std::to_string(b) +
                                    ") arrays must have the same size");
    }
}

void keyed_set(zval *arr, const Label &label, zval *value) {
    zval key;
    write_label(label, &key);
    array_set_zval_key(Z_ARRVAL_P(arr), &key, value);
    zval_ptr_dtor(&key);
    zval_ptr_dtor(value);
}

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_report, 0, 2, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO(0, expected, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO(0, predicted, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_contingency, 0, 2, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO(0, labels, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO(0, clusters, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

void ZEND_FASTCALL report_confusion(INTERNAL_FUNCTION_PARAMETERS) {
    HashTable *expected, *predicted;
    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_ARRAY_HT(expected)
        Z_PARAM_ARRAY_HT(predicted)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        std::vector<int> e, p;
        LabelEncoder classes;
        const std::size_t K = encode_label_pair(expected, predicted, e, p, &classes);
        require_pairs(e.size(), p.size());
        const std::vector<double> m = mlcore::confusion(e, p, K);
        array_init_size(return_value, static_cast<uint32_t>(K));
        for (std::size_t a = 0; a < K; ++a) {
            zval row;
            array_init_size(&row, static_cast<uint32_t>(K));
            for (std::size_t b = 0; b < K; ++b) {
                zval count;
                ZVAL_LONG(&count, static_cast<zend_long>(m[a * K + b]));
                keyed_set(&row, classes.classes[b], &count);
            }
            keyed_set(return_value, classes.classes[a], &row);
        }
    });
}

void ZEND_FASTCALL report_breakdown(INTERNAL_FUNCTION_PARAMETERS) {
    HashTable *expected, *predicted;
    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_ARRAY_HT(expected)
        Z_PARAM_ARRAY_HT(predicted)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        std::vector<int> e, p;
        LabelEncoder classes;
        const std::size_t K = encode_label_pair(expected, predicted, e, p, &classes);
        require_pairs(e.size(), p.size());
        const std::vector<double> m = mlcore::confusion(e, p, K);
        const std::vector<mlcore::ClassStats> stats = mlcore::class_stats(m, K);
        const double n = static_cast<double>(e.size());

        zval overall, per_class;
        array_init(&overall);
        add_assoc_double(&overall, "accuracy", mlcore::accuracy(e, p));
        add_assoc_double(&overall, "precision", mlcore::macro_precision(stats));
        add_assoc_double(&overall, "recall", mlcore::macro_recall(stats));
        add_assoc_double(&overall, "f1Score", mlcore::macro_f_beta(stats, 1.0));
        add_assoc_double(&overall, "mcc", mlcore::mcc(m, K));
        add_assoc_double(&overall, "informedness", mlcore::macro_informedness(stats));
        add_assoc_long(&overall, "cardinality", static_cast<zend_long>(e.size()));
        add_assoc_long(&overall, "classes", static_cast<zend_long>(K));

        array_init_size(&per_class, static_cast<uint32_t>(K));
        for (std::size_t k = 0; k < K; ++k) {
            const mlcore::ClassStats &s = stats[k];
            zval row;
            array_init(&row);
            add_assoc_double(&row, "precision", s.precision());
            add_assoc_double(&row, "recall", s.recall());
            add_assoc_double(&row, "specificity", s.specificity());
            add_assoc_double(&row, "f1Score", s.f_beta(1.0));
            add_assoc_double(&row, "informedness", s.recall() + s.specificity() - 1.0);
            add_assoc_long(&row, "truePositives", static_cast<zend_long>(s.tp));
            add_assoc_long(&row, "falsePositives", static_cast<zend_long>(s.fp));
            add_assoc_long(&row, "falseNegatives", static_cast<zend_long>(s.fn));
            add_assoc_long(&row, "trueNegatives", static_cast<zend_long>(s.tn));
            add_assoc_long(&row, "support", static_cast<zend_long>(s.support()));
            add_assoc_double(&row, "proportion", s.support() / n);
            keyed_set(&per_class, classes.classes[k], &row);
        }
        array_init(return_value);
        add_assoc_zval(return_value, "overall", &overall);
        add_assoc_zval(return_value, "classes", &per_class);
    });
}

void ZEND_FASTCALL report_errors(INTERNAL_FUNCTION_PARAMETERS) {
    HashTable *expected, *predicted;
    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_ARRAY_HT(expected)
        Z_PARAM_ARRAY_HT(predicted)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        const std::vector<double> e = read_numbers(expected, "Expected value");
        const std::vector<double> p = read_numbers(predicted, "Predicted value");
        require_pairs(e.size(), p.size());
        std::vector<double> errors(e.size());
        double mean = 0.0, min = std::numeric_limits<double>::infinity(), max = -min;
        for (std::size_t i = 0; i < e.size(); ++i) {
            errors[i] = e[i] - p[i];
            mean += errors[i];
            min = std::min(min, errors[i]);
            max = std::max(max, errors[i]);
        }
        mean /= static_cast<double>(e.size());
        double variance = 0.0;
        for (double err : errors) {
            variance += (err - mean) * (err - mean);
        }
        variance /= static_cast<double>(e.size());
        const double mse = mlcore::mean_squared_error(e, p);
        array_init(return_value);
        add_assoc_double(return_value, "meanAbsoluteError", mlcore::mean_absolute_error(e, p));
        add_assoc_double(return_value, "medianAbsoluteError", mlcore::median_absolute_error(e, p));
        add_assoc_double(return_value, "meanSquaredError", mse);
        add_assoc_double(return_value, "rootMeanSquaredError", std::sqrt(mse));
        add_assoc_double(return_value, "r2", mlcore::r2_score(e, p));
        add_assoc_double(return_value, "smape", mlcore::smape(e, p));
        add_assoc_double(return_value, "errorMean", mean);
        add_assoc_double(return_value, "errorVariance", variance);
        add_assoc_double(return_value, "errorStdDev", std::sqrt(variance));
        add_assoc_double(return_value, "errorMin", min);
        add_assoc_double(return_value, "errorMax", max);
        add_assoc_long(return_value, "cardinality", static_cast<zend_long>(e.size()));
    });
}

void ZEND_FASTCALL report_contingency(INTERNAL_FUNCTION_PARAMETERS) {
    HashTable *labels, *clusters;
    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_ARRAY_HT(labels)
        Z_PARAM_ARRAY_HT(clusters)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        LabelEncoder le, ce;
        const std::vector<int> l = le.fit(labels);
        const std::vector<int> c = ce.fit(clusters);
        require_pairs(l.size(), c.size());
        const std::vector<double> t = mlcore::contingency(l, le.size(), c, ce.size());
        array_init(return_value);
        for (std::size_t ci = 0; ci < ce.size(); ++ci) {
            zval row;
            array_init(&row);
            for (std::size_t li = 0; li < le.size(); ++li) {
                zval count;
                ZVAL_LONG(&count, static_cast<zend_long>(t[li * ce.size() + ci]));
                keyed_set(&row, le.classes[li], &count);
            }
            keyed_set(return_value, ce.classes[ci], &row);
        }
    });
}

#define ML_STATIC(name, handler, arginfo) ZEND_FENTRY(name, (handler), arginfo, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)

const zend_function_entry report_methods[] = {
    ML_STATIC(confusionMatrix, report_confusion, arginfo_report)
    ML_STATIC(multiclassBreakdown, report_breakdown, arginfo_report)
    ML_STATIC(errorAnalysis, report_errors, arginfo_report)
    ML_STATIC(contingencyTable, report_contingency, arginfo_contingency)
    PHP_FE_END};

// ---- validators -----------------------------------------------------------------------

enum class Kind : std::uint8_t { HoldOut, KFold, LeavePOut, MonteCarlo };

using Split = std::pair<std::vector<std::size_t>, std::vector<std::size_t>>;  // (train, test)

template <Kind TKind>
struct Validator {
    static constexpr std::uint32_t kTag = TKind == Kind::HoldOut   ? mlcore::tag("VHLD")
                                          : TKind == Kind::KFold   ? mlcore::tag("VKFD")
                                          : TKind == Kind::LeavePOut ? mlcore::tag("VLPO")
                                                                     : mlcore::tag("VMCS");
    double ratio = 0.2;
    std::size_t k = 5;            // folds (KFold), p (LeavePOut), simulations (MonteCarlo)
    bool stratify = false;
    bool has_seed = false;
    std::uint64_t seed = 0;

    void save(ArchiveWriter &w) const {
        w.f64(ratio);
        w.size(k);
        w.boolean(stratify);
        w.boolean(has_seed);
        w.u64(seed);
    }
    void load(ArchiveReader &r) {
        ratio = r.f64();
        k = r.size();
        stratify = r.boolean();
        has_seed = r.boolean();
        seed = r.u64();
        ArchiveReader::require(ratio > 0.0 && ratio < 1.0 && k >= 1);
    }

    // Indices of each stratum (class), or one stratum with everything.
    static std::vector<std::vector<std::size_t>> strata(const std::vector<int> *labels, std::size_t n, std::size_t K) {
        if (!labels) {
            std::vector<std::vector<std::size_t>> all(1, std::vector<std::size_t>(n));
            std::iota(all[0].begin(), all[0].end(), 0);
            return all;
        }
        std::vector<std::vector<std::size_t>> groups(K);
        for (std::size_t i = 0; i < n; ++i) {
            groups[static_cast<std::size_t>((*labels)[i])].push_back(i);
        }
        return groups;
    }

    std::vector<Split> splits(std::size_t n, const std::vector<int> *labels, std::size_t K) const {
        mlcore::Rng rng = mlcore::make_rng(has_seed, seed);
        std::vector<Split> out;
        auto holdout = [&](mlcore::Rng &g) {
            Split s;
            for (std::vector<std::size_t> &group : strata(labels, n, K)) {
                std::shuffle(group.begin(), group.end(), g);
                std::size_t test = static_cast<std::size_t>(std::llround(ratio * static_cast<double>(group.size())));
                test = std::min(std::max<std::size_t>(test, group.size() > 1 ? 1 : 0), group.size() - 1);
                s.second.insert(s.second.end(), group.begin(), group.begin() + static_cast<std::ptrdiff_t>(test));
                s.first.insert(s.first.end(), group.begin() + static_cast<std::ptrdiff_t>(test), group.end());
            }
            check(!s.first.empty() && !s.second.empty(), "Not enough samples to split into training and testing sets");
            return s;
        };
        switch (TKind) {
            case Kind::HoldOut:
                out.push_back(holdout(rng));
                break;
            case Kind::MonteCarlo:
                for (std::size_t i = 0; i < k; ++i) {
                    out.push_back(holdout(rng));
                }
                break;
            case Kind::KFold: {
                check(k >= 2 && k <= n, "k must be at least 2 and at most the number of samples");
                std::vector<std::vector<std::size_t>> folds(k);
                std::size_t next = 0;
                for (std::vector<std::size_t> &group : strata(labels, n, K)) {
                    std::shuffle(group.begin(), group.end(), rng);
                    for (std::size_t i : group) {
                        folds[next++ % k].push_back(i);  // round-robin keeps class proportions
                    }
                }
                for (std::size_t f = 0; f < k; ++f) {
                    Split s;
                    s.second = folds[f];
                    for (std::size_t g = 0; g < k; ++g) {
                        if (g != f) {
                            s.first.insert(s.first.end(), folds[g].begin(), folds[g].end());
                        }
                    }
                    out.push_back(std::move(s));
                }
                break;
            }
            case Kind::LeavePOut: {
                check(k < n, "p must be less than the number of samples");
                for (std::size_t start = 0; start < n; start += k) {
                    Split s;
                    for (std::size_t i = 0; i < n; ++i) {
                        (i >= start && i < start + k ? s.second : s.first).push_back(i);
                    }
                    out.push_back(std::move(s));
                }
                break;
            }
        }
        return out;
    }
};

using HoldOut = Validator<Kind::HoldOut>;
using KFold = Validator<Kind::KFold>;
using LeavePOut = Validator<Kind::LeavePOut>;
using MonteCarlo = Validator<Kind::MonteCarlo>;

ZEND_BEGIN_ARG_INFO_EX(arginfo_holdout_construct, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, ratio, IS_DOUBLE, 0, "0.2")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, stratify, _IS_BOOL, 0, "false")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, seed, IS_LONG, 1, "null")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_kfold_construct, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, k, IS_LONG, 0, "5")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, stratify, _IS_BOOL, 0, "false")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, seed, IS_LONG, 1, "null")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_lpo_construct, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, p, IS_LONG, 0, "10")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_monte_carlo_construct, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, simulations, IS_LONG, 0, "10")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, ratio, IS_DOUBLE, 0, "0.2")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, stratify, _IS_BOOL, 0, "false")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, seed, IS_LONG, 1, "null")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_test, 0, 3, IS_DOUBLE, 0)
    ZEND_ARG_TYPE_INFO(0, estimator, IS_OBJECT, 0)
    ZEND_ARG_TYPE_INFO(0, samples, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO(0, targets, IS_ARRAY, 0)
    ZEND_ARG_INFO_WITH_DEFAULT_VALUE(0, metric, "null")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_scores, 0, 3, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO(0, estimator, IS_OBJECT, 0)
    ZEND_ARG_TYPE_INFO(0, samples, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO(0, targets, IS_ARRAY, 0)
    ZEND_ARG_INFO_WITH_DEFAULT_VALUE(0, metric, "null")
ZEND_END_ARG_INFO()

void ZEND_FASTCALL holdout_construct(INTERNAL_FUNCTION_PARAMETERS) {
    double ratio = 0.2;
    bool stratify = false, seed_null = true;
    zend_long seed = 0;
    ZEND_PARSE_PARAMETERS_START(0, 3)
        Z_PARAM_OPTIONAL
        Z_PARAM_DOUBLE(ratio)
        Z_PARAM_BOOL(stratify)
        Z_PARAM_LONG_OR_NULL(seed, seed_null)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        HoldOut v;
        check(ratio > 0.0 && ratio < 1.0, "ratio must be between 0 and 1 (exclusive)");
        v.ratio = ratio;
        v.stratify = stratify;
        v.has_seed = !seed_null;
        v.seed = static_cast<std::uint64_t>(seed);
        reset(ZEND_THIS, v);
    });
}

void ZEND_FASTCALL kfold_construct(INTERNAL_FUNCTION_PARAMETERS) {
    zend_long k = 5, seed = 0;
    bool stratify = false, seed_null = true;
    ZEND_PARSE_PARAMETERS_START(0, 3)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(k)
        Z_PARAM_BOOL(stratify)
        Z_PARAM_LONG_OR_NULL(seed, seed_null)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        KFold v;
        check(k >= 2, "k must be at least 2");
        v.k = static_cast<std::size_t>(k);
        v.stratify = stratify;
        v.has_seed = !seed_null;
        v.seed = static_cast<std::uint64_t>(seed);
        reset(ZEND_THIS, v);
    });
}

void ZEND_FASTCALL lpo_construct(INTERNAL_FUNCTION_PARAMETERS) {
    zend_long p = 10;
    ZEND_PARSE_PARAMETERS_START(0, 1)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(p)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        LeavePOut v;
        v.k = positive(p, "p");
        reset(ZEND_THIS, v);
    });
}

void ZEND_FASTCALL monte_carlo_construct(INTERNAL_FUNCTION_PARAMETERS) {
    zend_long simulations = 10, seed = 0;
    double ratio = 0.2;
    bool stratify = false, seed_null = true;
    ZEND_PARSE_PARAMETERS_START(0, 4)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(simulations)
        Z_PARAM_DOUBLE(ratio)
        Z_PARAM_BOOL(stratify)
        Z_PARAM_LONG_OR_NULL(seed, seed_null)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        MonteCarlo v;
        v.k = positive(simulations, "simulations");
        check(ratio > 0.0 && ratio < 1.0, "ratio must be between 0 and 1 (exclusive)");
        v.ratio = ratio;
        v.stratify = stratify;
        v.has_seed = !seed_null;
        v.seed = static_cast<std::uint64_t>(seed);
        reset(ZEND_THIS, v);
    });
}

}  // namespace

namespace mlphp {

// Runs `validator` (any ML\CrossValidation validator object) and returns one
// score per split. Shared with GridSearch and BayesianSearch.
std::vector<double> cross_validate(zval *validator, zval *estimator, HashTable *samples, HashTable *targets, const Metric &metric);

}  // namespace mlphp

namespace {

template <typename V>
std::vector<double> run_validator(const V &v, zval *estimator, HashTable *samples, HashTable *targets, const Metric &metric) {
    require_instance(estimator, ce_learner, "estimator");
    const std::vector<zval *> X = values_of(samples);
    const std::vector<zval *> Y = values_of(targets);
    check(!X.empty(), "Samples must not be empty");
    if (X.size() != Y.size()) {
        throw std::invalid_argument("Number of samples (" + std::to_string(X.size()) + ") does not match number of targets (" +
                                    std::to_string(Y.size()) + ")");
    }
    std::vector<int> encoded;
    LabelEncoder encoder;
    if (v.stratify) {
        encoded = encoder.fit(targets);
    }
    const std::vector<Split> splits = v.splits(X.size(), v.stratify ? &encoded : nullptr, encoder.size());

    std::vector<double> scores;
    for (const Split &s : splits) {
        zval model, train_x, train_y, test_x, test_y, ignored, predicted;
        clone_into(estimator, &model);
        pick(X, s.first, &train_x);
        pick(Y, s.first, &train_y);
        pick(X, s.second, &test_x);
        pick(Y, s.second, &test_y);
        auto cleanup = [&] {
            zval_ptr_dtor(&model);
            zval_ptr_dtor(&train_x);
            zval_ptr_dtor(&train_y);
            zval_ptr_dtor(&test_x);
            zval_ptr_dtor(&test_y);
        };
        try {
            call_method(&model, "fit", &ignored, {&train_x, &train_y});
            zval_ptr_dtor(&ignored);
            call_method(&model, "predict", &predicted, {&test_x});
            try {
                scores.push_back(metric.evaluate(&test_y, &predicted));
            } catch (...) {
                zval_ptr_dtor(&predicted);
                throw;
            }
            zval_ptr_dtor(&predicted);
        } catch (...) {
            cleanup();
            throw;
        }
        cleanup();
    }
    return scores;
}

template <typename V, bool Mean>
void ZEND_FASTCALL validator_test(INTERNAL_FUNCTION_PARAMETERS) {
    zval *estimator, *metric_spec = nullptr;
    HashTable *samples, *targets;
    ZEND_PARSE_PARAMETERS_START(3, 4)
        Z_PARAM_OBJECT(estimator)
        Z_PARAM_ARRAY_HT(samples)
        Z_PARAM_ARRAY_HT(targets)
        Z_PARAM_OPTIONAL
        Z_PARAM_ZVAL(metric_spec)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        Metric metric;
        metric.configure(metric_spec, underlying_class(estimator));
        const std::vector<double> scores = run_validator(native<V>(ZEND_THIS), estimator, samples, targets, metric);
        if (Mean) {
            RETVAL_DOUBLE(std::accumulate(scores.begin(), scores.end(), 0.0) / static_cast<double>(scores.size()));
        } else {
            write_doubles(return_value, scores);
        }
    });
}

#define ML_VALIDATOR(V, ctor, arginfo)                                         \
    ML_METHOD(__construct, ctor, arginfo)                                       \
    ML_METHOD(test, (validator_test<V, true>), arginfo_test)                   \
    ML_METHOD(scores, (validator_test<V, false>), arginfo_scores)              \
    ML_METHOD(__serialize, serialize_native<V>, arginfo_ml_get_array)           \
    ML_METHOD(__unserialize, unserialize_native<V>, arginfo_ml_unserialize)

const zend_function_entry holdout_methods[] = {ML_VALIDATOR(HoldOut, holdout_construct, arginfo_holdout_construct) PHP_FE_END};
const zend_function_entry kfold_methods[] = {ML_VALIDATOR(KFold, kfold_construct, arginfo_kfold_construct) PHP_FE_END};
const zend_function_entry lpo_methods[] = {ML_VALIDATOR(LeavePOut, lpo_construct, arginfo_lpo_construct) PHP_FE_END};
const zend_function_entry monte_carlo_methods[] = {
    ML_VALIDATOR(MonteCarlo, monte_carlo_construct, arginfo_monte_carlo_construct) PHP_FE_END};

}  // namespace

namespace mlphp {

std::vector<double> cross_validate(zval *validator, zval *estimator, HashTable *samples, HashTable *targets, const Metric &metric) {
    require_instance(validator, ce_validator, "validator");
    zend_class_entry *ce = Z_OBJCE_P(validator);
    const std::string name = ZSTR_VAL(ce->name);
    if (name == "ML\\CrossValidation\\HoldOut") {
        return run_validator(native<HoldOut>(validator), estimator, samples, targets, metric);
    }
    if (name == "ML\\CrossValidation\\KFold") {
        return run_validator(native<KFold>(validator), estimator, samples, targets, metric);
    }
    if (name == "ML\\CrossValidation\\LeavePOut") {
        return run_validator(native<LeavePOut>(validator), estimator, samples, targets, metric);
    }
    return run_validator(native<MonteCarlo>(validator), estimator, samples, targets, metric);
}

}  // namespace mlphp

void mlphp::register_cross_validation() {
    zend_class_entry tmp;
    INIT_CLASS_ENTRY(tmp, "ML\\CrossValidation\\Validator", nullptr);
    ce_validator = zend_register_internal_interface(&tmp);

    INIT_CLASS_ENTRY(tmp, "ML\\CrossValidation\\Reports", report_methods);
    zend_register_internal_class(&tmp)->ce_flags |= ZEND_ACC_FINAL;

    register_class<HoldOut>("ML\\CrossValidation\\HoldOut", holdout_methods, {ce_validator});
    register_class<KFold>("ML\\CrossValidation\\KFold", kfold_methods, {ce_validator});
    register_class<LeavePOut>("ML\\CrossValidation\\LeavePOut", lpo_methods, {ce_validator});
    register_class<MonteCarlo>("ML\\CrossValidation\\MonteCarlo", monte_carlo_methods, {ce_validator});
}
