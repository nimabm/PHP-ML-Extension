// ML\Classification\GaussianNB. Serializes to the PHP array format of 0.1.0.

#include "binding.hpp"
#include "core/gaussian_nb.hpp"

namespace {

using namespace mlphp;

struct Model {
    mlcore::GaussianNB core;
    LabelEncoder labels;

    bool fitted() const { return core.fitted(); }
    std::size_t features() const { return core.features; }
    std::size_t classes() const { return core.classes; }
    void fit(const mlcore::Matrix &X, const std::vector<int> &y, std::size_t K, unsigned) { core.fit(X, y, K); }
    int predict_one(const double *x) const { return core.predict_one(x); }
    void probabilities(const double *x, double *out) const { core.probabilities(x, out); }
};

ZEND_BEGIN_ARG_INFO_EX(arginfo_construct, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, varSmoothing, IS_DOUBLE, 0, "1.0E-9")
ZEND_END_ARG_INFO()

void ZEND_FASTCALL gaussian_nb_construct(INTERNAL_FUNCTION_PARAMETERS) {
    double var_smoothing = 1e-9;
    ZEND_PARSE_PARAMETERS_START(0, 1)
        Z_PARAM_OPTIONAL
        Z_PARAM_DOUBLE(var_smoothing)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        Model fresh;
        fresh.core.var_smoothing = non_negative(var_smoothing, "varSmoothing");
        reset(ZEND_THIS, std::move(fresh));
    });
}

void ZEND_FASTCALL gaussian_nb_serialize(INTERNAL_FUNCTION_PARAMETERS) {
    ZEND_PARSE_PARAMETERS_NONE();
    const Model &m = native<Model>(ZEND_THIS);
    array_init(return_value);
    add_assoc_double(return_value, "varSmoothing", m.core.var_smoothing);
    zval classes;
    m.labels.write_all(&classes);
    add_assoc_zval(return_value, "classes", &classes);
    add_assoc_long(return_value, "features", static_cast<zend_long>(m.core.features));
    put_doubles(return_value, "means", m.core.means);
    put_doubles(return_value, "variances", m.core.variances);
    put_doubles(return_value, "logPriors", m.core.log_priors);
}

void ZEND_FASTCALL gaussian_nb_unserialize(INTERNAL_FUNCTION_PARAMETERS) {
    HashTable *data;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY_HT(data)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        Model restored;
        restored.core.var_smoothing = get_double(data, "varSmoothing");
        restored.labels.read_all(get_array(data, "classes"));
        const zend_long features = get_long(data, "features");
        std::vector<double> means = get_doubles(data, "means");
        std::vector<double> variances = get_doubles(data, "variances");
        std::vector<double> log_priors = get_doubles(data, "logPriors");

        const std::size_t K = restored.labels.size();
        if (K != 0) {
            if (features < 1 || log_priors.size() != K || !shape_matches(means.size(), K, static_cast<std::size_t>(features)) ||
                variances.size() != means.size()) {
                throw std::invalid_argument("Invalid serialized data: inconsistent GaussianNB state");
            }
            for (double v : variances) {
                if (!(v > 0.0)) {
                    throw std::invalid_argument("Invalid serialized data: variances must be positive");
                }
            }
            restored.core.classes = K;
            restored.core.features = static_cast<std::size_t>(features);
            restored.core.means = std::move(means);
            restored.core.variances = std::move(variances);
            restored.core.log_priors = std::move(log_priors);
        }
        reset(ZEND_THIS, std::move(restored));
    });
}

const zend_function_entry methods[] = {
    ML_METHOD(__construct, gaussian_nb_construct, arginfo_construct)
    ML_METHOD(fit, fit_classifier<Model>, arginfo_ml_fit_supervised)
    ML_METHOD(predict, predict_classifier<Model>, arginfo_ml_predict_label)
    ML_METHOD(predictProbability, predict_proba<Model>, arginfo_ml_samples_to_array)
    ML_METHOD(getClasses, get_classes<Model>, arginfo_ml_get_array)
    ML_METHOD(__serialize, gaussian_nb_serialize, arginfo_ml_get_array)
    ML_METHOD(__unserialize, gaussian_nb_unserialize, arginfo_ml_unserialize)
    PHP_FE_END};

}  // namespace

void mlphp::register_gaussian_nb() {
    register_class<Model>("ML\\Classification\\GaussianNB", methods, {ce_learner, ce_classifier, ce_probabilistic});
}
