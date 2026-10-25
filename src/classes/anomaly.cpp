// ML\AnomalyDetection: RobustZScore, GaussianMLE, IsolationForest,
// LocalOutlierFactor and Loda. predict() returns 1 for anomalies and 0 for
// inliers; score() returns the raw anomaly score (higher = more anomalous).

#include "binding.hpp"
#include "core/anomaly.hpp"

namespace {

using namespace mlphp;
using mlcore::ArchiveReader;
using mlcore::ArchiveWriter;

void fit_core(mlcore::IsolationForest &c, const mlcore::Matrix &X, unsigned threads) { c.fit(X, threads); }
template <typename C>
void fit_core(C &c, const mlcore::Matrix &X, unsigned) { c.fit(X); }

template <typename Core, std::uint32_t Tag>
struct Detector {
    static constexpr std::uint32_t kTag = Tag;
    Core core;

    bool fitted() const { return core.features() > 0; }
    std::size_t features() const { return core.features(); }
    void fit(const mlcore::Matrix &X, unsigned threads) { fit_core(core, X, threads); }
    double score_one(const double *x) const { return core.score(x); }
    double threshold() const { return core.threshold; }
    void save(ArchiveWriter &w) const { core.save(w); }
    void load(ArchiveReader &r) { core.load(r); }
};

using RobustZ = Detector<mlcore::RobustZScore, mlcore::tag("RZSC")>;
using GaussMLE = Detector<mlcore::GaussianMLE, mlcore::tag("GMLE")>;
using IForest = Detector<mlcore::IsolationForest, mlcore::tag("IFOR")>;
using LOF = Detector<mlcore::LocalOutlierFactor, mlcore::tag("LOFA")>;
using LodaModel = Detector<mlcore::Loda, mlcore::tag("LODA")>;

double contamination_arg(double c) {
    check(c >= 0.0 && c < 0.5, "contamination must be at least 0 and less than 0.5");
    return c;
}

ZEND_BEGIN_ARG_INFO_EX(arginfo_robust_z_construct, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, threshold, IS_DOUBLE, 0, "3.5")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, beta, IS_DOUBLE, 0, "0.5")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, smoothing, IS_DOUBLE, 0, "1.0E-9")
ZEND_END_ARG_INFO()

void ZEND_FASTCALL robust_z_construct(INTERNAL_FUNCTION_PARAMETERS) {
    double threshold = 3.5, beta = 0.5, smoothing = 1e-9;
    ZEND_PARSE_PARAMETERS_START(0, 3)
        Z_PARAM_OPTIONAL
        Z_PARAM_DOUBLE(threshold)
        Z_PARAM_DOUBLE(beta)
        Z_PARAM_DOUBLE(smoothing)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        RobustZ fresh;
        fresh.core.threshold = strictly_positive(threshold, "threshold");
        fresh.core.beta = in_range(beta, 0.0, 1.0, "beta");
        fresh.core.smoothing = strictly_positive(smoothing, "smoothing");
        reset(ZEND_THIS, std::move(fresh));
    });
}

ZEND_BEGIN_ARG_INFO_EX(arginfo_gaussian_mle_construct, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, contamination, IS_DOUBLE, 0, "0.1")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, smoothing, IS_DOUBLE, 0, "1.0E-9")
ZEND_END_ARG_INFO()

void ZEND_FASTCALL gaussian_mle_construct(INTERNAL_FUNCTION_PARAMETERS) {
    double contamination = 0.1, smoothing = 1e-9;
    ZEND_PARSE_PARAMETERS_START(0, 2)
        Z_PARAM_OPTIONAL
        Z_PARAM_DOUBLE(contamination)
        Z_PARAM_DOUBLE(smoothing)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        GaussMLE fresh;
        fresh.core.contamination = contamination_arg(contamination);
        fresh.core.smoothing = non_negative(smoothing, "smoothing");
        reset(ZEND_THIS, std::move(fresh));
    });
}

ZEND_BEGIN_ARG_INFO_EX(arginfo_iforest_construct, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, estimators, IS_LONG, 0, "100")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, maxSamples, IS_LONG, 0, "256")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, contamination, IS_DOUBLE, 1, "null")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, seed, IS_LONG, 1, "null")
ZEND_END_ARG_INFO()

void ZEND_FASTCALL iforest_construct(INTERNAL_FUNCTION_PARAMETERS) {
    zend_long estimators = 100, max_samples = 256, seed = 0;
    double contamination = 0.0;
    bool contamination_null = true, seed_null = true;
    ZEND_PARSE_PARAMETERS_START(0, 4)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(estimators)
        Z_PARAM_LONG(max_samples)
        Z_PARAM_DOUBLE_OR_NULL(contamination, contamination_null)
        Z_PARAM_LONG_OR_NULL(seed, seed_null)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        IForest fresh;
        fresh.core.estimators = positive(estimators, "estimators");
        fresh.core.max_samples = positive(max_samples, "maxSamples");
        check(fresh.core.max_samples >= 2, "maxSamples must be at least 2");
        fresh.core.contamination = contamination_null ? -1.0 : contamination_arg(contamination);
        fresh.core.has_seed = !seed_null;
        fresh.core.seed = static_cast<std::uint64_t>(seed);
        reset(ZEND_THIS, std::move(fresh));
    });
}

ZEND_BEGIN_ARG_INFO_EX(arginfo_lof_construct, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, k, IS_LONG, 0, "20")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, contamination, IS_DOUBLE, 0, "0.1")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, metric, IS_STRING, 0, "\"euclidean\"")
ZEND_END_ARG_INFO()

void ZEND_FASTCALL lof_construct(INTERNAL_FUNCTION_PARAMETERS) {
    zend_long k = 20;
    double contamination = 0.1;
    zend_string *metric = nullptr;
    ZEND_PARSE_PARAMETERS_START(0, 3)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(k)
        Z_PARAM_DOUBLE(contamination)
        Z_PARAM_STR(metric)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        LOF fresh;
        fresh.core.k = positive(k, "k");
        fresh.core.contamination = contamination_arg(contamination);
        if (metric) {
            fresh.core.index.metric = mlcore::Distance::parse(zstr(metric));
        }
        reset(ZEND_THIS, std::move(fresh));
    });
}

ZEND_BEGIN_ARG_INFO_EX(arginfo_loda_construct, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, estimators, IS_LONG, 0, "100")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, bins, IS_LONG, 1, "null")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, contamination, IS_DOUBLE, 0, "0.1")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, seed, IS_LONG, 1, "null")
ZEND_END_ARG_INFO()

void ZEND_FASTCALL loda_construct(INTERNAL_FUNCTION_PARAMETERS) {
    zend_long estimators = 100, bins = 0, seed = 0;
    double contamination = 0.1;
    bool bins_null = true, seed_null = true;
    ZEND_PARSE_PARAMETERS_START(0, 4)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(estimators)
        Z_PARAM_LONG_OR_NULL(bins, bins_null)
        Z_PARAM_DOUBLE(contamination)
        Z_PARAM_LONG_OR_NULL(seed, seed_null)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        LodaModel fresh;
        fresh.core.estimators = positive(estimators, "estimators");
        fresh.core.bins = bins_null ? 0 : positive(bins, "bins");
        fresh.core.contamination = contamination_arg(contamination);
        fresh.core.has_seed = !seed_null;
        fresh.core.seed = static_cast<std::uint64_t>(seed);
        reset(ZEND_THIS, std::move(fresh));
    });
}

#define ML_DETECTOR(M, ctor, arginfo)                                          \
    ML_METHOD(__construct, ctor, arginfo)                                       \
    ML_METHOD(fit, fit_unsupervised<M>, arginfo_ml_fit_unsupervised)            \
    ML_METHOD(predict, predict_anomaly<M>, arginfo_ml_predict_int)              \
    ML_METHOD(score, score_anomaly<M>, arginfo_ml_predict_float)                \
    ML_METHOD(getThreshold, get_threshold<M>, arginfo_ml_get_float)             \
    ML_METHOD(__serialize, serialize_native<M>, arginfo_ml_get_array)           \
    ML_METHOD(__unserialize, unserialize_native<M>, arginfo_ml_unserialize)

const zend_function_entry robust_z_methods[] = {ML_DETECTOR(RobustZ, robust_z_construct, arginfo_robust_z_construct) PHP_FE_END};
const zend_function_entry gaussian_mle_methods[] = {ML_DETECTOR(GaussMLE, gaussian_mle_construct, arginfo_gaussian_mle_construct) PHP_FE_END};
const zend_function_entry iforest_methods[] = {ML_DETECTOR(IForest, iforest_construct, arginfo_iforest_construct) PHP_FE_END};
const zend_function_entry lof_methods[] = {ML_DETECTOR(LOF, lof_construct, arginfo_lof_construct) PHP_FE_END};
const zend_function_entry loda_methods[] = {ML_DETECTOR(LodaModel, loda_construct, arginfo_loda_construct) PHP_FE_END};

}  // namespace

void mlphp::register_anomaly() {
    const auto detector = {ce_learner, ce_anomaly};
    register_class<RobustZ>("ML\\AnomalyDetection\\RobustZScore", robust_z_methods, detector);
    register_class<GaussMLE>("ML\\AnomalyDetection\\GaussianMLE", gaussian_mle_methods, detector);
    register_class<IForest>("ML\\AnomalyDetection\\IsolationForest", iforest_methods, detector);
    register_class<LOF>("ML\\AnomalyDetection\\LocalOutlierFactor", lof_methods, detector);
    register_class<LodaModel>("ML\\AnomalyDetection\\Loda", loda_methods, detector);
}
