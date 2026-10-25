// AdaBoost, LogitBoost (classification), GradientBoost and ElasticNet (regression).

#include <numeric>

#include "binding.hpp"
#include "core/boosting.hpp"

namespace {

using namespace mlphp;
using mlcore::ArchiveReader;
using mlcore::ArchiveWriter;

std::vector<double> combined_importances(const std::vector<mlcore::DecisionTree> &trees, std::size_t features,
                                         const std::vector<double> *weights = nullptr) {
    std::vector<double> total(features, 0.0);
    for (std::size_t t = 0; t < trees.size(); ++t) {
        const std::vector<double> imp = trees[t].normalized_importances();
        const double w = weights ? (*weights)[t] : 1.0;
        for (std::size_t f = 0; f < features; ++f) {
            total[f] += w * imp[f];
        }
    }
    const double sum = std::accumulate(total.begin(), total.end(), 0.0);
    if (sum > 0.0) {
        for (double &v : total) {
            v /= sum;
        }
    }
    return total;
}

// ---- AdaBoost ----------------------------------------------------------------------------

struct AdaBoostModel {
    static constexpr std::uint32_t kTag = mlcore::tag("ADAB");
    mlcore::AdaBoost core;
    LabelEncoder labels;

    bool fitted() const { return core.fitted(); }
    std::size_t features() const { return core.features; }
    std::size_t classes() const { return core.classes; }
    void fit(const mlcore::Matrix &X, const std::vector<int> &y, std::size_t K, unsigned) { core.fit(X, y, K); }
    void probabilities(const double *x, double *out) const { core.probabilities(x, out); }
    int predict_one(const double *x) const {
        std::vector<double> p(core.classes);
        core.probabilities(x, p.data());
        return static_cast<int>(mlcore::argmax(p.data(), p.size()));
    }
    std::vector<double> importances() const { return combined_importances(core.trees, core.features, &core.alphas); }
    const std::vector<double> &losses() const { return core.losses; }
    void save(ArchiveWriter &w) const {
        core.save(w);
        labels.save(w);
    }
    void load(ArchiveReader &r) {
        core.load(r);
        labels.load(r);
        ArchiveReader::require(!core.fitted() || labels.size() == core.classes, "label count mismatch");
    }
};

ZEND_BEGIN_ARG_INFO_EX(arginfo_adaboost_construct, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, estimators, IS_LONG, 0, "100")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, rate, IS_DOUBLE, 0, "1.0")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, ratio, IS_DOUBLE, 0, "0.8")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, maxHeight, IS_LONG, 0, "1")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, minChange, IS_DOUBLE, 0, "1.0E-4")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, seed, IS_LONG, 1, "null")
ZEND_END_ARG_INFO()

void ZEND_FASTCALL adaboost_construct(INTERNAL_FUNCTION_PARAMETERS) {
    zend_long estimators = 100, max_height = 1, seed = 0;
    double rate = 1.0, ratio = 0.8, min_change = 1e-4;
    bool seed_null = true;
    ZEND_PARSE_PARAMETERS_START(0, 6)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(estimators)
        Z_PARAM_DOUBLE(rate)
        Z_PARAM_DOUBLE(ratio)
        Z_PARAM_LONG(max_height)
        Z_PARAM_DOUBLE(min_change)
        Z_PARAM_LONG_OR_NULL(seed, seed_null)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        AdaBoostModel fresh;
        fresh.core.estimators = positive(estimators, "estimators");
        fresh.core.rate = strictly_positive(rate, "rate");
        check(ratio > 0.0 && ratio <= 1.0, "ratio must be greater than 0 and at most 1");
        fresh.core.ratio = ratio;
        fresh.core.base.max_height = positive(max_height, "maxHeight");
        fresh.core.min_change = non_negative(min_change, "minChange");
        parse_seed(seed_null, seed, fresh.core.has_seed, fresh.core.seed);
        reset(ZEND_THIS, std::move(fresh));
    });
}

// ---- LogitBoost ------------------------------------------------------------------------------

struct LogitBoostModel {
    static constexpr std::uint32_t kTag = mlcore::tag("LGTB");
    mlcore::LogitBoost core;
    LabelEncoder labels;

    bool fitted() const { return core.fitted(); }
    std::size_t features() const { return core.features; }
    std::size_t classes() const { return 2; }
    void fit(const mlcore::Matrix &X, const std::vector<int> &y, std::size_t K, unsigned) {
        if (K != 2) {
            throw std::invalid_argument("LogitBoost is a binary classifier and needs exactly two distinct classes, " +
                                        std::to_string(K) + " given; wrap it in OneVsRest for more");
        }
        core.fit(X, y);
    }
    void probabilities(const double *x, double *out) const {
        out[1] = core.positive_probability(x);
        out[0] = 1.0 - out[1];
    }
    int predict_one(const double *x) const { return core.positive_probability(x) >= 0.5 ? 1 : 0; }
    std::vector<double> importances() const { return combined_importances(core.trees, core.features); }
    const std::vector<double> &losses() const { return core.losses; }
    void save(ArchiveWriter &w) const {
        core.save(w);
        labels.save(w);
    }
    void load(ArchiveReader &r) {
        core.load(r);
        labels.load(r);
        ArchiveReader::require(!core.fitted() || labels.size() == 2, "label count mismatch");
    }
};

ZEND_BEGIN_ARG_INFO_EX(arginfo_logitboost_construct, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, estimators, IS_LONG, 0, "100")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, rate, IS_DOUBLE, 0, "0.1")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, ratio, IS_DOUBLE, 0, "0.5")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, maxHeight, IS_LONG, 0, "3")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, minChange, IS_DOUBLE, 0, "1.0E-4")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, seed, IS_LONG, 1, "null")
ZEND_END_ARG_INFO()

void ZEND_FASTCALL logitboost_construct(INTERNAL_FUNCTION_PARAMETERS) {
    zend_long estimators = 100, max_height = 3, seed = 0;
    double rate = 0.1, ratio = 0.5, min_change = 1e-4;
    bool seed_null = true;
    ZEND_PARSE_PARAMETERS_START(0, 6)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(estimators)
        Z_PARAM_DOUBLE(rate)
        Z_PARAM_DOUBLE(ratio)
        Z_PARAM_LONG(max_height)
        Z_PARAM_DOUBLE(min_change)
        Z_PARAM_LONG_OR_NULL(seed, seed_null)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        LogitBoostModel fresh;
        fresh.core.estimators = positive(estimators, "estimators");
        fresh.core.rate = strictly_positive(rate, "rate");
        check(ratio > 0.0 && ratio <= 1.0, "ratio must be greater than 0 and at most 1");
        fresh.core.ratio = ratio;
        fresh.core.base.max_height = positive(max_height, "maxHeight");
        fresh.core.min_change = non_negative(min_change, "minChange");
        parse_seed(seed_null, seed, fresh.core.has_seed, fresh.core.seed);
        reset(ZEND_THIS, std::move(fresh));
    });
}

// ---- GradientBoost ---------------------------------------------------------------------------

struct GradientBoostModel {
    static constexpr std::uint32_t kTag = mlcore::tag("GRDB");
    mlcore::GradientBoost core;

    bool fitted() const { return core.fitted(); }
    std::size_t features() const { return core.features; }
    void fit(const mlcore::Matrix &X, const std::vector<double> &y, unsigned) { core.fit(X, y); }
    double predict_one(const double *x) const { return core.predict(x); }
    std::vector<double> importances() const { return combined_importances(core.trees, core.features); }
    const std::vector<double> &losses() const { return core.losses; }
    void save(ArchiveWriter &w) const { core.save(w); }
    void load(ArchiveReader &r) { core.load(r); }
};

ZEND_BEGIN_ARG_INFO_EX(arginfo_gradient_construct, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, estimators, IS_LONG, 0, "100")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, rate, IS_DOUBLE, 0, "0.1")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, ratio, IS_DOUBLE, 0, "0.5")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, maxHeight, IS_LONG, 0, "3")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, maxLeafSize, IS_LONG, 0, "3")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, minChange, IS_DOUBLE, 0, "1.0E-4")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, window, IS_LONG, 0, "5")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, holdOut, IS_DOUBLE, 0, "0.1")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, seed, IS_LONG, 1, "null")
ZEND_END_ARG_INFO()

void ZEND_FASTCALL gradient_construct(INTERNAL_FUNCTION_PARAMETERS) {
    zend_long estimators = 100, max_height = 3, max_leaf_size = 3, window = 5, seed = 0;
    double rate = 0.1, ratio = 0.5, min_change = 1e-4, hold_out = 0.1;
    bool seed_null = true;
    ZEND_PARSE_PARAMETERS_START(0, 9)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(estimators)
        Z_PARAM_DOUBLE(rate)
        Z_PARAM_DOUBLE(ratio)
        Z_PARAM_LONG(max_height)
        Z_PARAM_LONG(max_leaf_size)
        Z_PARAM_DOUBLE(min_change)
        Z_PARAM_LONG(window)
        Z_PARAM_DOUBLE(hold_out)
        Z_PARAM_LONG_OR_NULL(seed, seed_null)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        GradientBoostModel fresh;
        fresh.core.estimators = positive(estimators, "estimators");
        fresh.core.rate = strictly_positive(rate, "rate");
        check(ratio > 0.0 && ratio <= 1.0, "ratio must be greater than 0 and at most 1");
        fresh.core.ratio = ratio;
        fresh.core.base.max_height = positive(max_height, "maxHeight");
        fresh.core.base.max_leaf_size = positive(max_leaf_size, "maxLeafSize");
        fresh.core.min_change = non_negative(min_change, "minChange");
        fresh.core.window = positive(window, "window");
        check(hold_out >= 0.0 && hold_out < 0.5, "holdOut must be at least 0 and less than 0.5");
        fresh.core.hold_out = hold_out;
        parse_seed(seed_null, seed, fresh.core.has_seed, fresh.core.seed);
        reset(ZEND_THIS, std::move(fresh));
    });
}

void ZEND_FASTCALL gradient_validation(INTERNAL_FUNCTION_PARAMETERS) {
    ZEND_PARSE_PARAMETERS_NONE();
    guard([&] {
        const GradientBoostModel &m = native<GradientBoostModel>(ZEND_THIS);
        require_fitted(m.fitted(), ZEND_THIS);
        write_doubles(return_value, m.core.validation);
    });
}

template <typename M>
void ZEND_FASTCALL boost_losses(INTERNAL_FUNCTION_PARAMETERS) {
    ZEND_PARSE_PARAMETERS_NONE();
    guard([&] {
        const M &m = native<M>(ZEND_THIS);
        require_fitted(m.fitted(), ZEND_THIS);
        write_doubles(return_value, m.losses());
    });
}

template <typename M>
void ZEND_FASTCALL boost_count(INTERNAL_FUNCTION_PARAMETERS) {
    ZEND_PARSE_PARAMETERS_NONE();
    guard([&] {
        const M &m = native<M>(ZEND_THIS);
        require_fitted(m.fitted(), ZEND_THIS);
        RETVAL_LONG(static_cast<zend_long>(m.core.trees.size()));
    });
}

// ---- ElasticNet -------------------------------------------------------------------------------

struct ElasticNetModel {
    static constexpr std::uint32_t kTag = mlcore::tag("ELNT");
    mlcore::ElasticNet core;

    bool fitted() const { return core.fitted(); }
    std::size_t features() const { return core.coefficients.size(); }
    void fit(const mlcore::Matrix &X, const std::vector<double> &y, unsigned) { core.fit(X, y); }
    double predict_one(const double *x) const { return core.predict(x); }
    void save(ArchiveWriter &w) const { core.save(w); }
    void load(ArchiveReader &r) { core.load(r); }
};

ZEND_BEGIN_ARG_INFO_EX(arginfo_elastic_construct, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, alpha, IS_DOUBLE, 0, "1.0")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, l1Ratio, IS_DOUBLE, 0, "0.5")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, maxIterations, IS_LONG, 0, "1000")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, tolerance, IS_DOUBLE, 0, "1.0E-4")
ZEND_END_ARG_INFO()

void ZEND_FASTCALL elastic_construct(INTERNAL_FUNCTION_PARAMETERS) {
    double alpha = 1.0, l1_ratio = 0.5, tolerance = 1e-4;
    zend_long max_iterations = 1000;
    ZEND_PARSE_PARAMETERS_START(0, 4)
        Z_PARAM_OPTIONAL
        Z_PARAM_DOUBLE(alpha)
        Z_PARAM_DOUBLE(l1_ratio)
        Z_PARAM_LONG(max_iterations)
        Z_PARAM_DOUBLE(tolerance)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        ElasticNetModel fresh;
        fresh.core.alpha = non_negative(alpha, "alpha");
        fresh.core.l1_ratio = in_range(l1_ratio, 0.0, 1.0, "l1Ratio");
        fresh.core.max_iterations = static_cast<long>(positive(max_iterations, "maxIterations"));
        fresh.core.tolerance = non_negative(tolerance, "tolerance");
        reset(ZEND_THIS, std::move(fresh));
    });
}

void ZEND_FASTCALL elastic_coefficients(INTERNAL_FUNCTION_PARAMETERS) {
    ZEND_PARSE_PARAMETERS_NONE();
    guard([&] {
        const ElasticNetModel &m = native<ElasticNetModel>(ZEND_THIS);
        require_fitted(m.fitted(), ZEND_THIS);
        write_doubles(return_value, m.core.coefficients);
    });
}

void ZEND_FASTCALL elastic_intercept(INTERNAL_FUNCTION_PARAMETERS) {
    ZEND_PARSE_PARAMETERS_NONE();
    guard([&] {
        const ElasticNetModel &m = native<ElasticNetModel>(ZEND_THIS);
        require_fitted(m.fitted(), ZEND_THIS);
        RETVAL_DOUBLE(m.core.intercept);
    });
}

#define ML_BOOSTED_CLASSIFIER(M)                                                   \
    ML_METHOD(fit, fit_classifier<M>, arginfo_ml_fit_supervised)                    \
    ML_METHOD(predict, predict_classifier<M>, arginfo_ml_predict_label)             \
    ML_METHOD(predictProbability, predict_proba<M>, arginfo_ml_samples_to_array)    \
    ML_METHOD(getClasses, get_classes<M>, arginfo_ml_get_array)                     \
    ML_METHOD(getFeatureImportances, get_importances<M>, arginfo_ml_get_array)      \
    ML_METHOD(getLosses, boost_losses<M>, arginfo_ml_get_array)                     \
    ML_METHOD(getEstimatorCount, boost_count<M>, arginfo_ml_get_int)                \
    ML_METHOD(__serialize, serialize_native<M>, arginfo_ml_get_array)               \
    ML_METHOD(__unserialize, unserialize_native<M>, arginfo_ml_unserialize)

const zend_function_entry adaboost_methods[] = {
    ML_METHOD(__construct, adaboost_construct, arginfo_adaboost_construct) ML_BOOSTED_CLASSIFIER(AdaBoostModel) PHP_FE_END};
const zend_function_entry logitboost_methods[] = {
    ML_METHOD(__construct, logitboost_construct, arginfo_logitboost_construct) ML_BOOSTED_CLASSIFIER(LogitBoostModel) PHP_FE_END};

const zend_function_entry gradient_methods[] = {
    ML_METHOD(__construct, gradient_construct, arginfo_gradient_construct)
    ML_METHOD(fit, fit_regressor<GradientBoostModel>, arginfo_ml_fit_supervised)
    ML_METHOD(predict, predict_regressor<GradientBoostModel>, arginfo_ml_predict_float)
    ML_METHOD(getFeatureImportances, get_importances<GradientBoostModel>, arginfo_ml_get_array)
    ML_METHOD(getLosses, boost_losses<GradientBoostModel>, arginfo_ml_get_array)
    ML_METHOD(getValidationScores, gradient_validation, arginfo_ml_get_array)
    ML_METHOD(getEstimatorCount, boost_count<GradientBoostModel>, arginfo_ml_get_int)
    ML_METHOD(__serialize, serialize_native<GradientBoostModel>, arginfo_ml_get_array)
    ML_METHOD(__unserialize, unserialize_native<GradientBoostModel>, arginfo_ml_unserialize)
    PHP_FE_END};

const zend_function_entry elastic_methods[] = {
    ML_METHOD(__construct, elastic_construct, arginfo_elastic_construct)
    ML_METHOD(fit, fit_regressor<ElasticNetModel>, arginfo_ml_fit_supervised)
    ML_METHOD(predict, predict_regressor<ElasticNetModel>, arginfo_ml_predict_float)
    ML_METHOD(getCoefficients, elastic_coefficients, arginfo_ml_get_array)
    ML_METHOD(getIntercept, elastic_intercept, arginfo_ml_get_float)
    ML_METHOD(__serialize, serialize_native<ElasticNetModel>, arginfo_ml_get_array)
    ML_METHOD(__unserialize, unserialize_native<ElasticNetModel>, arginfo_ml_unserialize)
    PHP_FE_END};

}  // namespace

void mlphp::register_boosting() {
    const auto classifier = {ce_learner, ce_classifier, ce_probabilistic, ce_ranks_features};
    register_class<AdaBoostModel>("ML\\Classification\\AdaBoost", adaboost_methods, classifier);
    register_class<LogitBoostModel>("ML\\Classification\\LogitBoost", logitboost_methods, classifier);
    register_class<GradientBoostModel>("ML\\Regression\\GradientBoost", gradient_methods, {ce_learner, ce_regressor, ce_ranks_features});
    register_class<ElasticNetModel>("ML\\Regression\\ElasticNet", elastic_methods, {ce_learner, ce_regressor});
}
