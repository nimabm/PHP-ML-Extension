// Support vector machines: SVC (classification), SVR (regression) and
// OneClassSVM (anomaly detection).

#include "binding.hpp"
#include "core/svm.hpp"

namespace {

using namespace mlphp;
using mlcore::ArchiveReader;
using mlcore::ArchiveWriter;

void configure(mlcore::SVMParams &p, zend_string *kernel, double gamma, bool gamma_null, zend_long degree, double coef0,
               double tolerance, zend_long cache) {
    if (kernel) {
        p.kernel.type = mlcore::Kernel::parse(zstr(kernel));
    }
    p.kernel.gamma = gamma_null ? 0.0 : strictly_positive(gamma, "gamma");
    p.kernel.degree = static_cast<double>(positive(degree, "degree"));
    check(std::isfinite(coef0), "coef0 must be finite");
    p.kernel.coef0 = coef0;
    p.tolerance = strictly_positive(tolerance, "tolerance");
    p.cache_mb = positive(cache, "cacheSize");
}

// ---- SVC ----------------------------------------------------------------------------------

struct SVCModel {
    static constexpr std::uint32_t kTag = mlcore::tag("SVCX");
    mlcore::SVC core;
    LabelEncoder labels;
    bool fitted() const { return core.fitted(); }
    std::size_t features() const { return core.features; }
    std::size_t classes() const { return core.classes; }
    void fit(const mlcore::Matrix &X, const std::vector<int> &y, std::size_t K, unsigned threads) { core.fit(X, y, K, threads); }
    int predict_one(const double *x) const { return core.predict_one(x); }
    std::size_t support_vectors() const {
        std::size_t n = 0;
        for (const auto &f : core.functions) n += f.support.rows;
        return n;
    }
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

ZEND_BEGIN_ARG_INFO_EX(arginfo_svc_construct, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, c, IS_DOUBLE, 0, "1.0")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, kernel, IS_STRING, 0, "\"rbf\"")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, gamma, IS_DOUBLE, 1, "null")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, degree, IS_LONG, 0, "3")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, coef0, IS_DOUBLE, 0, "0.0")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, tolerance, IS_DOUBLE, 0, "0.001")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, cacheSize, IS_LONG, 0, "128")
ZEND_END_ARG_INFO()

void ZEND_FASTCALL svc_construct(INTERNAL_FUNCTION_PARAMETERS) {
    double c = 1.0, gamma = 0.0, coef0 = 0.0, tolerance = 1e-3;
    zend_string *kernel = nullptr;
    bool gamma_null = true;
    zend_long degree = 3, cache = 128;
    ZEND_PARSE_PARAMETERS_START(0, 7)
        Z_PARAM_OPTIONAL
        Z_PARAM_DOUBLE(c)
        Z_PARAM_STR(kernel)
        Z_PARAM_DOUBLE_OR_NULL(gamma, gamma_null)
        Z_PARAM_LONG(degree)
        Z_PARAM_DOUBLE(coef0)
        Z_PARAM_DOUBLE(tolerance)
        Z_PARAM_LONG(cache)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        SVCModel fresh;
        fresh.core.params.c = strictly_positive(c, "c");
        configure(fresh.core.params, kernel, gamma, gamma_null, degree, coef0, tolerance, cache);
        reset(ZEND_THIS, std::move(fresh));
    });
}

// ---- SVR ----------------------------------------------------------------------------------

struct SVRModel {
    static constexpr std::uint32_t kTag = mlcore::tag("SVRX");
    mlcore::SVR core;
    bool fitted() const { return core.fitted(); }
    std::size_t features() const { return core.features; }
    void fit(const mlcore::Matrix &X, const std::vector<double> &y, unsigned) { core.fit(X, y); }
    double predict_one(const double *x) const { return core.predict(x); }
    std::size_t support_vectors() const { return core.function.support.rows; }
    void save(ArchiveWriter &w) const { core.save(w); }
    void load(ArchiveReader &r) { core.load(r); }
};

ZEND_BEGIN_ARG_INFO_EX(arginfo_svr_construct, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, c, IS_DOUBLE, 0, "1.0")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, epsilon, IS_DOUBLE, 0, "0.1")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, kernel, IS_STRING, 0, "\"rbf\"")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, gamma, IS_DOUBLE, 1, "null")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, degree, IS_LONG, 0, "3")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, coef0, IS_DOUBLE, 0, "0.0")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, tolerance, IS_DOUBLE, 0, "0.001")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, cacheSize, IS_LONG, 0, "128")
ZEND_END_ARG_INFO()

void ZEND_FASTCALL svr_construct(INTERNAL_FUNCTION_PARAMETERS) {
    double c = 1.0, epsilon = 0.1, gamma = 0.0, coef0 = 0.0, tolerance = 1e-3;
    zend_string *kernel = nullptr;
    bool gamma_null = true;
    zend_long degree = 3, cache = 128;
    ZEND_PARSE_PARAMETERS_START(0, 8)
        Z_PARAM_OPTIONAL
        Z_PARAM_DOUBLE(c)
        Z_PARAM_DOUBLE(epsilon)
        Z_PARAM_STR(kernel)
        Z_PARAM_DOUBLE_OR_NULL(gamma, gamma_null)
        Z_PARAM_LONG(degree)
        Z_PARAM_DOUBLE(coef0)
        Z_PARAM_DOUBLE(tolerance)
        Z_PARAM_LONG(cache)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        SVRModel fresh;
        fresh.core.params.c = strictly_positive(c, "c");
        fresh.core.params.epsilon = non_negative(epsilon, "epsilon");
        configure(fresh.core.params, kernel, gamma, gamma_null, degree, coef0, tolerance, cache);
        reset(ZEND_THIS, std::move(fresh));
    });
}

// ---- OneClassSVM ----------------------------------------------------------------------------

struct OneClassModel {
    static constexpr std::uint32_t kTag = mlcore::tag("OCSV");
    mlcore::OneClassSVM core;
    bool fitted() const { return core.fitted(); }
    std::size_t features() const { return core.features; }
    void fit(const mlcore::Matrix &X, unsigned) { core.fit(X); }
    double score_one(const double *x) const { return core.score(x); }
    double threshold() const { return 0.0; }
    std::size_t support_vectors() const { return core.function.support.rows; }
    void save(ArchiveWriter &w) const { core.save(w); }
    void load(ArchiveReader &r) { core.load(r); }
};

ZEND_BEGIN_ARG_INFO_EX(arginfo_one_class_construct, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, nu, IS_DOUBLE, 0, "0.1")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, kernel, IS_STRING, 0, "\"rbf\"")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, gamma, IS_DOUBLE, 1, "null")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, degree, IS_LONG, 0, "3")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, coef0, IS_DOUBLE, 0, "0.0")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, tolerance, IS_DOUBLE, 0, "0.001")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, cacheSize, IS_LONG, 0, "128")
ZEND_END_ARG_INFO()

void ZEND_FASTCALL one_class_construct(INTERNAL_FUNCTION_PARAMETERS) {
    double nu = 0.1, gamma = 0.0, coef0 = 0.0, tolerance = 1e-3;
    zend_string *kernel = nullptr;
    bool gamma_null = true;
    zend_long degree = 3, cache = 128;
    ZEND_PARSE_PARAMETERS_START(0, 7)
        Z_PARAM_OPTIONAL
        Z_PARAM_DOUBLE(nu)
        Z_PARAM_STR(kernel)
        Z_PARAM_DOUBLE_OR_NULL(gamma, gamma_null)
        Z_PARAM_LONG(degree)
        Z_PARAM_DOUBLE(coef0)
        Z_PARAM_DOUBLE(tolerance)
        Z_PARAM_LONG(cache)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        OneClassModel fresh;
        check(nu > 0.0 && nu <= 1.0, "nu must be greater than 0 and at most 1");
        fresh.core.params.nu = nu;
        configure(fresh.core.params, kernel, gamma, gamma_null, degree, coef0, tolerance, cache);
        reset(ZEND_THIS, std::move(fresh));
    });
}

template <typename M>
void ZEND_FASTCALL support_vector_count(INTERNAL_FUNCTION_PARAMETERS) {
    ZEND_PARSE_PARAMETERS_NONE();
    guard([&] {
        const M &m = native<M>(ZEND_THIS);
        require_fitted(m.fitted(), ZEND_THIS);
        RETVAL_LONG(static_cast<zend_long>(m.support_vectors()));
    });
}

const zend_function_entry svc_methods[] = {
    ML_METHOD(__construct, svc_construct, arginfo_svc_construct)
    ML_METHOD(fit, fit_classifier<SVCModel>, arginfo_ml_fit_supervised)
    ML_METHOD(predict, predict_classifier<SVCModel>, arginfo_ml_predict_label)
    ML_METHOD(getClasses, get_classes<SVCModel>, arginfo_ml_get_array)
    ML_METHOD(getSupportVectorCount, support_vector_count<SVCModel>, arginfo_ml_get_int)
    ML_METHOD(__serialize, serialize_native<SVCModel>, arginfo_ml_get_array)
    ML_METHOD(__unserialize, unserialize_native<SVCModel>, arginfo_ml_unserialize)
    PHP_FE_END};

const zend_function_entry svr_methods[] = {
    ML_METHOD(__construct, svr_construct, arginfo_svr_construct)
    ML_METHOD(fit, fit_regressor<SVRModel>, arginfo_ml_fit_supervised)
    ML_METHOD(predict, predict_regressor<SVRModel>, arginfo_ml_predict_float)
    ML_METHOD(getSupportVectorCount, support_vector_count<SVRModel>, arginfo_ml_get_int)
    ML_METHOD(__serialize, serialize_native<SVRModel>, arginfo_ml_get_array)
    ML_METHOD(__unserialize, unserialize_native<SVRModel>, arginfo_ml_unserialize)
    PHP_FE_END};

const zend_function_entry one_class_methods[] = {
    ML_METHOD(__construct, one_class_construct, arginfo_one_class_construct)
    ML_METHOD(fit, fit_unsupervised<OneClassModel>, arginfo_ml_fit_unsupervised)
    ML_METHOD(predict, predict_anomaly<OneClassModel>, arginfo_ml_predict_int)
    ML_METHOD(score, score_anomaly<OneClassModel>, arginfo_ml_predict_float)
    ML_METHOD(getThreshold, get_threshold<OneClassModel>, arginfo_ml_get_float)
    ML_METHOD(getSupportVectorCount, support_vector_count<OneClassModel>, arginfo_ml_get_int)
    ML_METHOD(__serialize, serialize_native<OneClassModel>, arginfo_ml_get_array)
    ML_METHOD(__unserialize, unserialize_native<OneClassModel>, arginfo_ml_unserialize)
    PHP_FE_END};

}  // namespace

void mlphp::register_svm() {
    register_class<SVCModel>("ML\\Classification\\SVC", svc_methods, {ce_learner, ce_classifier});
    register_class<SVRModel>("ML\\Regression\\SVR", svr_methods, {ce_learner, ce_regressor});
    register_class<OneClassModel>("ML\\AnomalyDetection\\OneClassSVM", one_class_methods, {ce_learner, ce_anomaly});
}
