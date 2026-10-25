// ML\Metrics: static evaluation helpers, plus the shared Metric evaluator.

#include <cmath>

#include "binding.hpp"
#include "core/metrics.hpp"
#include "metric_eval.hpp"

namespace mlphp {

std::size_t encode_label_pair(HashTable *a, HashTable *b, std::vector<int> &ea, std::vector<int> &eb, LabelEncoder *encoder) {
    std::vector<Label> la = read_labels(a, "Expected label");
    std::vector<Label> lb = read_labels(b, "Predicted label");
    std::vector<Label> all = la;
    all.insert(all.end(), lb.begin(), lb.end());
    std::sort(all.begin(), all.end());
    all.erase(std::unique(all.begin(), all.end()), all.end());
    auto index = [&](const Label &l) { return static_cast<int>(std::lower_bound(all.begin(), all.end(), l) - all.begin()); };
    ea.resize(la.size());
    eb.resize(lb.size());
    for (std::size_t i = 0; i < la.size(); ++i) {
        ea[i] = index(la[i]);
    }
    for (std::size_t i = 0; i < lb.size(); ++i) {
        eb[i] = index(lb[i]);
    }
    const std::size_t K = all.size();
    if (encoder) {
        encoder->classes = std::move(all);
    }
    return K;
}

}  // namespace mlphp

namespace {

using namespace mlphp;

void require_same_size(std::size_t expected, std::size_t predicted) {
    if (expected == 0) {
        throw std::invalid_argument("Arrays must not be empty");
    }
    if (expected != predicted) {
        throw std::invalid_argument("Expected (" + std::to_string(expected) + ") and predicted (" +
                                    std::to_string(predicted) + ") arrays must have the same size");
    }
}

struct Numbers {
    std::vector<double> expected, predicted;
};

Numbers read_pairs(HashTable *expected, HashTable *predicted) {
    Numbers p{read_numbers(expected, "Expected value"), read_numbers(predicted, "Predicted value")};
    require_same_size(p.expected.size(), p.predicted.size());
    return p;
}

struct Labels {
    std::vector<int> expected, predicted;
    std::size_t K = 0;
    LabelEncoder encoder;
};

Labels read_label_pairs(HashTable *expected, HashTable *predicted) {
    Labels l;
    l.K = encode_label_pair(expected, predicted, l.expected, l.predicted, &l.encoder);
    require_same_size(l.expected.size(), l.predicted.size());
    return l;
}

std::string label_key(const Label &l) {
    if (const zend_long *i = std::get_if<zend_long>(&l)) {
        return std::to_string(*i);
    }
    return std::get<std::string>(l);
}

// Index of `positive` among the encoded classes, or throws.
int positive_index(const Labels &l, zval *positive) {
    const Label p = read_label(positive, 0);
    const int index = l.encoder.find(p);
    if (index < 0) {
        throw std::invalid_argument("positiveClass \"" + label_key(p) + "\" does not occur in the labels");
    }
    return index;
}

double classification_metric(const std::string &name, HashTable *expected, HashTable *predicted, zval *positive = nullptr,
                             double beta = 1.0) {
    const Labels l = read_label_pairs(expected, predicted);
    if (name == "accuracy") {
        return mlcore::accuracy(l.expected, l.predicted);
    }
    const std::vector<double> m = mlcore::confusion(l.expected, l.predicted, l.K);
    if (name == "mcc") {
        return mlcore::mcc(m, l.K);
    }
    const std::vector<mlcore::ClassStats> stats = mlcore::class_stats(m, l.K);
    if (positive && Z_TYPE_P(positive) != IS_NULL) {
        const mlcore::ClassStats &s = stats[static_cast<std::size_t>(positive_index(l, positive))];
        if (name == "precision") return s.precision();
        if (name == "recall") return s.recall();
        if (name == "informedness") return s.recall() + s.specificity() - 1.0;
        return s.f_beta(beta);
    }
    if (name == "precision") return mlcore::macro_precision(stats);
    if (name == "recall") return mlcore::macro_recall(stats);
    if (name == "informedness") return mlcore::macro_informedness(stats);
    return mlcore::macro_f_beta(stats, beta);
}

double regression_metric(const std::string &name, HashTable *expected, HashTable *predicted) {
    const Numbers p = read_pairs(expected, predicted);
    if (name == "r2") return mlcore::r2_score(p.expected, p.predicted);
    if (name == "mse") return mlcore::mean_squared_error(p.expected, p.predicted);
    if (name == "mae") return mlcore::mean_absolute_error(p.expected, p.predicted);
    if (name == "rmse") return std::sqrt(mlcore::mean_squared_error(p.expected, p.predicted));
    if (name == "median_absolute_error") return mlcore::median_absolute_error(p.expected, p.predicted);
    return mlcore::smape(p.expected, p.predicted);
}

bool is_classification_metric(const std::string &name) {
    return name == "accuracy" || name == "precision" || name == "recall" || name == "f1" || name == "mcc" ||
           name == "informedness";
}

bool is_regression_metric(const std::string &name) {
    return name == "r2" || name == "mse" || name == "mae" || name == "rmse" || name == "median_absolute_error" ||
           name == "smape";
}

}  // namespace

namespace mlphp {

bool is_known_metric(const std::string &name) { return is_classification_metric(name) || is_regression_metric(name); }

void Metric::configure(zval *spec, zend_class_entry *kind) {
    zval_ptr_dtor(&callable);
    ZVAL_UNDEF(&callable);
    name.clear();
    if (spec && Z_TYPE_P(spec) == IS_STRING) {
        name.assign(Z_STRVAL_P(spec), Z_STRLEN_P(spec));
        if (!is_known_metric(name)) {
            throw std::invalid_argument("Unknown metric \"" + name +
                                        "\"; expected accuracy, precision, recall, f1, mcc, informedness, r2, mse, "
                                        "mae, rmse, median_absolute_error, smape or a callable");
        }
        higher_is_better = is_classification_metric(name) || name == "r2";
        return;
    }
    if (spec && Z_TYPE_P(spec) != IS_NULL) {
        if (!zend_is_callable(spec, 0, nullptr)) {
            throw std::invalid_argument("metric must be a metric name, a callable or null");
        }
        ZVAL_COPY(&callable, spec);
        higher_is_better = true;
        return;
    }
    if (kind && instanceof_function(kind, ce_classifier)) {
        name = "accuracy";
    } else if (kind && instanceof_function(kind, ce_regressor)) {
        name = "r2";
    } else {
        throw std::invalid_argument("A metric is required for estimators that are neither classifiers nor regressors");
    }
    higher_is_better = true;
}

double Metric::evaluate(zval *expected, zval *predicted) const {
    if (Z_TYPE(callable) != IS_UNDEF) {
        zval result;
        zval args[2];
        ZVAL_COPY_VALUE(&args[0], expected);
        ZVAL_COPY_VALUE(&args[1], predicted);
        if (call_user_function(nullptr, nullptr, const_cast<zval *>(&callable), &result, 2, args) != SUCCESS || EG(exception)) {
            zval_ptr_dtor(&result);
            throw PhpError{};
        }
        const double score = zval_get_double(&result);
        zval_ptr_dtor(&result);
        return score;
    }
    if (Z_TYPE_P(predicted) != IS_ARRAY) {
        throw std::invalid_argument("predict() must return an array for a list of samples");
    }
    if (is_classification_metric(name)) {
        return classification_metric(name, Z_ARRVAL_P(expected), Z_ARRVAL_P(predicted));
    }
    return regression_metric(name, Z_ARRVAL_P(expected), Z_ARRVAL_P(predicted));
}

}  // namespace mlphp

namespace {

// Probabilities: list of [label => p]. Matching is by the label's string form
// because array keys turn numeric-string labels into ints.
struct ProbabilityRows {
    std::vector<std::vector<std::pair<std::string, double>>> rows;
};

ProbabilityRows read_probabilities(HashTable *ht) {
    ProbabilityRows p;
    zval *row;
    ZEND_HASH_FOREACH_VAL(ht, row) {
        ZVAL_DEREF(row);
        check(Z_TYPE_P(row) == IS_ARRAY, "probabilities must be a list of [label => probability] arrays");
        std::vector<std::pair<std::string, double>> entry;
        zend_string *key;
        zend_ulong index;
        zval *value;
        ZEND_HASH_FOREACH_KEY_VAL(Z_ARRVAL_P(row), index, key, value) {
            ZVAL_DEREF(value);
            check(Z_TYPE_P(value) == IS_LONG || Z_TYPE_P(value) == IS_DOUBLE, "probabilities must be numbers");
            entry.emplace_back(key ? zstr(key) : std::to_string(static_cast<zend_long>(index)), zval_get_double(value));
        }
        ZEND_HASH_FOREACH_END();
        p.rows.push_back(std::move(entry));
    }
    ZEND_HASH_FOREACH_END();
    return p;
}

#define ML_TWO_ARRAYS(a, b)              \
    HashTable *a, *b;                    \
    ZEND_PARSE_PARAMETERS_START(2, 2)    \
        Z_PARAM_ARRAY_HT(a)              \
        Z_PARAM_ARRAY_HT(b)              \
    ZEND_PARSE_PARAMETERS_END()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_metric, 0, 2, IS_DOUBLE, 0)
    ZEND_ARG_TYPE_INFO(0, expected, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO(0, predicted, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_metric_positive, 0, 2, IS_DOUBLE, 0)
    ZEND_ARG_TYPE_INFO(0, expected, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO(0, predicted, IS_ARRAY, 0)
    ZEND_ARG_TYPE_MASK(0, positiveClass, MAY_BE_LONG | MAY_BE_STRING | MAY_BE_NULL, "null")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_fbeta, 0, 2, IS_DOUBLE, 0)
    ZEND_ARG_TYPE_INFO(0, expected, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO(0, predicted, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, beta, IS_DOUBLE, 0, "1.0")
    ZEND_ARG_TYPE_MASK(0, positiveClass, MAY_BE_LONG | MAY_BE_STRING | MAY_BE_NULL, "null")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_ranking, 0, 3, IS_DOUBLE, 0)
    ZEND_ARG_TYPE_INFO(0, expected, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO(0, scores, IS_ARRAY, 0)
    ZEND_ARG_TYPE_MASK(0, positiveClass, MAY_BE_LONG | MAY_BE_STRING, NULL)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_probabilistic, 0, 2, IS_DOUBLE, 0)
    ZEND_ARG_TYPE_INFO(0, expected, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO(0, probabilities, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_top_k, 0, 2, IS_DOUBLE, 0)
    ZEND_ARG_TYPE_INFO(0, expected, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO(0, probabilities, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, k, IS_LONG, 0, "3")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_clustering, 0, 2, IS_DOUBLE, 0)
    ZEND_ARG_TYPE_INFO(0, labels, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO(0, clusters, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_v_measure, 0, 2, IS_DOUBLE, 0)
    ZEND_ARG_TYPE_INFO(0, labels, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO(0, clusters, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, beta, IS_DOUBLE, 0, "1.0")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_trustworthiness, 0, 2, IS_DOUBLE, 0)
    ZEND_ARG_TYPE_INFO(0, original, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO(0, embedded, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, k, IS_LONG, 0, "5")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_construct, 0, 0, 0)
ZEND_END_ARG_INFO()

void ZEND_FASTCALL metrics_construct(INTERNAL_FUNCTION_PARAMETERS) { ZEND_PARSE_PARAMETERS_NONE(); }

void ZEND_FASTCALL metrics_accuracy(INTERNAL_FUNCTION_PARAMETERS) {
    ML_TWO_ARRAYS(expected, predicted);
    guard([&] {
        // Strict comparison (===) of any values, as in 0.1.0.
        const std::vector<zval *> a = values_of(expected);
        const std::vector<zval *> b = values_of(predicted);
        require_same_size(a.size(), b.size());
        std::size_t correct = 0;
        for (std::size_t i = 0; i < a.size(); ++i) {
            correct += zend_is_identical(a[i], b[i]);
        }
        RETVAL_DOUBLE(static_cast<double>(correct) / static_cast<double>(a.size()));
    });
}

template <int Which>  // 0 precision, 1 recall, 2 informedness
void ZEND_FASTCALL metrics_class_stat(INTERNAL_FUNCTION_PARAMETERS) {
    HashTable *expected, *predicted;
    zval *positive = nullptr;
    ZEND_PARSE_PARAMETERS_START(2, 3)
        Z_PARAM_ARRAY_HT(expected)
        Z_PARAM_ARRAY_HT(predicted)
        Z_PARAM_OPTIONAL
        Z_PARAM_ZVAL(positive)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        const char *name = Which == 0 ? "precision" : Which == 1 ? "recall" : "informedness";
        RETVAL_DOUBLE(classification_metric(name, expected, predicted, positive));
    });
}

void ZEND_FASTCALL metrics_f_beta(INTERNAL_FUNCTION_PARAMETERS) {
    HashTable *expected, *predicted;
    double beta = 1.0;
    zval *positive = nullptr;
    ZEND_PARSE_PARAMETERS_START(2, 4)
        Z_PARAM_ARRAY_HT(expected)
        Z_PARAM_ARRAY_HT(predicted)
        Z_PARAM_OPTIONAL
        Z_PARAM_DOUBLE(beta)
        Z_PARAM_ZVAL(positive)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        strictly_positive(beta, "beta");
        RETVAL_DOUBLE(classification_metric("f1", expected, predicted, positive, beta));
    });
}

void ZEND_FASTCALL metrics_f1(INTERNAL_FUNCTION_PARAMETERS) {
    HashTable *expected, *predicted;
    zval *positive = nullptr;
    ZEND_PARSE_PARAMETERS_START(2, 3)
        Z_PARAM_ARRAY_HT(expected)
        Z_PARAM_ARRAY_HT(predicted)
        Z_PARAM_OPTIONAL
        Z_PARAM_ZVAL(positive)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] { RETVAL_DOUBLE(classification_metric("f1", expected, predicted, positive)); });
}

void ZEND_FASTCALL metrics_mcc(INTERNAL_FUNCTION_PARAMETERS) {
    ML_TWO_ARRAYS(expected, predicted);
    guard([&] { RETVAL_DOUBLE(classification_metric("mcc", expected, predicted)); });
}

template <bool Auc>
void ZEND_FASTCALL metrics_ranking(INTERNAL_FUNCTION_PARAMETERS) {
    HashTable *expected, *scores;
    zval *positive;
    ZEND_PARSE_PARAMETERS_START(3, 3)
        Z_PARAM_ARRAY_HT(expected)
        Z_PARAM_ARRAY_HT(scores)
        Z_PARAM_ZVAL(positive)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        const std::vector<Label> labels = read_labels(expected, "Expected label");
        const std::vector<double> s = read_numbers(scores, "Score");
        require_same_size(labels.size(), s.size());
        const Label p = read_label(positive, 0);
        std::vector<bool> is_positive(labels.size());
        for (std::size_t i = 0; i < labels.size(); ++i) {
            is_positive[i] = labels[i] == p;
        }
        RETVAL_DOUBLE(Auc ? mlcore::roc_auc(is_positive, s) : mlcore::average_precision(is_positive, s));
    });
}

template <int Which>  // 0 brier, 1 probabilistic accuracy
void ZEND_FASTCALL metrics_probabilistic(INTERNAL_FUNCTION_PARAMETERS) {
    ML_TWO_ARRAYS(expected, probabilities);
    guard([&] {
        const std::vector<Label> labels = read_labels(expected, "Expected label");
        const ProbabilityRows p = read_probabilities(probabilities);
        require_same_size(labels.size(), p.rows.size());
        double total = 0.0;
        for (std::size_t i = 0; i < labels.size(); ++i) {
            const std::string truth = label_key(labels[i]);
            double p_true = 0.0, sum_sq = 0.0;
            for (const auto &[label, prob] : p.rows[i]) {
                const double target = label == truth ? 1.0 : 0.0;
                if (label == truth) {
                    p_true = prob;
                }
                sum_sq += (prob - target) * (prob - target);
            }
            bool present = false;
            for (const auto &entry : p.rows[i]) {
                present = present || entry.first == truth;
            }
            if (!present) {
                sum_sq += 1.0;  // the true class had implicit probability 0
            }
            total += Which == 0 ? sum_sq : p_true;
        }
        RETVAL_DOUBLE(total / static_cast<double>(labels.size()));
    });
}

void ZEND_FASTCALL metrics_top_k(INTERNAL_FUNCTION_PARAMETERS) {
    HashTable *expected, *probabilities;
    zend_long k = 3;
    ZEND_PARSE_PARAMETERS_START(2, 3)
        Z_PARAM_ARRAY_HT(expected)
        Z_PARAM_ARRAY_HT(probabilities)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(k)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        const std::size_t top = positive(k, "k");
        const std::vector<Label> labels = read_labels(expected, "Expected label");
        ProbabilityRows p = read_probabilities(probabilities);
        require_same_size(labels.size(), p.rows.size());
        std::size_t hits = 0;
        for (std::size_t i = 0; i < labels.size(); ++i) {
            auto &row = p.rows[i];
            std::stable_sort(row.begin(), row.end(), [](const auto &a, const auto &b) { return a.second > b.second; });
            const std::string truth = label_key(labels[i]);
            for (std::size_t j = 0; j < std::min(top, row.size()); ++j) {
                if (row[j].first == truth) {
                    ++hits;
                    break;
                }
            }
        }
        RETVAL_DOUBLE(static_cast<double>(hits) / static_cast<double>(labels.size()));
    });
}

template <int Which>  // 0 mse, 1 mae, 2 r2, 3 rmse, 4 median, 5 smape
void ZEND_FASTCALL metrics_regression(INTERNAL_FUNCTION_PARAMETERS) {
    ML_TWO_ARRAYS(expected, predicted);
    guard([&] {
        static const char *names[] = {"mse", "mae", "r2", "rmse", "median_absolute_error", "smape"};
        RETVAL_DOUBLE(regression_metric(names[Which], expected, predicted));
    });
}

template <int Which>  // 0 homogeneity, 1 completeness, 2 v-measure, 3 adjusted rand, 4 purity
void ZEND_FASTCALL metrics_clustering(INTERNAL_FUNCTION_PARAMETERS) {
    HashTable *labels, *clusters;
    double beta = 1.0;
    ZEND_PARSE_PARAMETERS_START(2, 3)
        Z_PARAM_ARRAY_HT(labels)
        Z_PARAM_ARRAY_HT(clusters)
        Z_PARAM_OPTIONAL
        Z_PARAM_DOUBLE(beta)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        LabelEncoder le, ce;
        const std::vector<int> l = le.fit(labels);
        const std::vector<int> c = ce.fit(clusters);
        require_same_size(l.size(), c.size());
        const std::vector<double> t = mlcore::contingency(l, le.size(), c, ce.size());
        double v = 0.0;
        switch (Which) {
            case 0: v = mlcore::homogeneity(t, le.size(), ce.size()); break;
            case 1: v = mlcore::completeness(t, le.size(), ce.size()); break;
            case 2: v = mlcore::v_measure(t, le.size(), ce.size(), strictly_positive(beta, "beta")); break;
            case 3: v = mlcore::adjusted_rand_index(t, le.size(), ce.size()); break;
            default: v = mlcore::purity(t, le.size(), ce.size()); break;
        }
        RETVAL_DOUBLE(v);
    });
}

void ZEND_FASTCALL metrics_trustworthiness(INTERNAL_FUNCTION_PARAMETERS) {
    HashTable *original, *embedded;
    zend_long k = 5;
    ZEND_PARSE_PARAMETERS_START(2, 3)
        Z_PARAM_ARRAY_HT(original)
        Z_PARAM_ARRAY_HT(embedded)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(k)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        const mlcore::Matrix X = read_samples(original);
        const mlcore::Matrix Y = read_samples(embedded);
        RETVAL_DOUBLE(mlcore::trustworthiness(X, Y, positive(k, "k")));
    });
}

#define ML_STATIC(name, handler, arginfo) ZEND_FENTRY(name, (handler), arginfo, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)

const zend_function_entry methods[] = {
    ZEND_FENTRY(__construct, metrics_construct, arginfo_construct, ZEND_ACC_PRIVATE)
    ML_STATIC(accuracy, metrics_accuracy, arginfo_metric)
    ML_STATIC(precision, metrics_class_stat<0>, arginfo_metric_positive)
    ML_STATIC(recall, metrics_class_stat<1>, arginfo_metric_positive)
    ML_STATIC(informedness, metrics_class_stat<2>, arginfo_metric_positive)
    ML_STATIC(fBeta, metrics_f_beta, arginfo_fbeta)
    ML_STATIC(f1Score, metrics_f1, arginfo_metric_positive)
    ML_STATIC(mcc, metrics_mcc, arginfo_metric)
    ML_STATIC(rocAuc, metrics_ranking<true>, arginfo_ranking)
    ML_STATIC(averagePrecision, metrics_ranking<false>, arginfo_ranking)
    ML_STATIC(brierScore, metrics_probabilistic<0>, arginfo_probabilistic)
    ML_STATIC(probabilisticAccuracy, metrics_probabilistic<1>, arginfo_probabilistic)
    ML_STATIC(topKAccuracy, metrics_top_k, arginfo_top_k)
    ML_STATIC(meanSquaredError, metrics_regression<0>, arginfo_metric)
    ML_STATIC(meanAbsoluteError, metrics_regression<1>, arginfo_metric)
    ML_STATIC(r2Score, metrics_regression<2>, arginfo_metric)
    ML_STATIC(rootMeanSquaredError, metrics_regression<3>, arginfo_metric)
    ML_STATIC(medianAbsoluteError, metrics_regression<4>, arginfo_metric)
    ML_STATIC(smape, metrics_regression<5>, arginfo_metric)
    ML_STATIC(homogeneity, metrics_clustering<0>, arginfo_clustering)
    ML_STATIC(completeness, metrics_clustering<1>, arginfo_clustering)
    ML_STATIC(vMeasure, metrics_clustering<2>, arginfo_v_measure)
    ML_STATIC(randIndex, metrics_clustering<3>, arginfo_clustering)
    ML_STATIC(purity, metrics_clustering<4>, arginfo_clustering)
    ML_STATIC(trustworthiness, metrics_trustworthiness, arginfo_trustworthiness)
    PHP_FE_END};

}  // namespace

void mlphp::register_metrics() {
    zend_class_entry tmp;
    INIT_CLASS_ENTRY(tmp, "ML\\Metrics", methods);
    zend_class_entry *ce = zend_register_internal_class(&tmp);
    ce->ce_flags |= ZEND_ACC_FINAL;
}
