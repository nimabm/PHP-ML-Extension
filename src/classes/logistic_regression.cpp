// ML\Classification\LogisticRegression. Serializes to the PHP array format of 0.1.0.

#include "binding.hpp"
#include "core/logistic_regression.hpp"

namespace {

using namespace mlphp;

struct Model {
    mlcore::LogisticRegression core;
    LabelEncoder labels;

    bool fitted() const { return core.fitted(); }
    std::size_t features() const { return core.features; }
    std::size_t classes() const { return core.classes; }
    void fit(const mlcore::Matrix &X, const std::vector<int> &y, std::size_t K, unsigned) { core.fit(X, y, K); }
    int predict_one(const double *x) const { return core.predict_one(x); }
    void probabilities(const double *x, double *out) const { core.probabilities(x, out); }
};

ZEND_BEGIN_ARG_INFO_EX(arginfo_construct, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, learningRate, IS_DOUBLE, 0, "0.1")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, maxIterations, IS_LONG, 0, "1000")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, l2, IS_DOUBLE, 0, "0.0")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, tolerance, IS_DOUBLE, 0, "1.0E-6")
ZEND_END_ARG_INFO()

void ZEND_FASTCALL logistic_construct(INTERNAL_FUNCTION_PARAMETERS) {
    double learning_rate = 0.1, l2 = 0.0, tolerance = 1e-6;
    zend_long max_iterations = 1000;
    ZEND_PARSE_PARAMETERS_START(0, 4)
        Z_PARAM_OPTIONAL
        Z_PARAM_DOUBLE(learning_rate)
        Z_PARAM_LONG(max_iterations)
        Z_PARAM_DOUBLE(l2)
        Z_PARAM_DOUBLE(tolerance)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        Model fresh;
        fresh.core.learning_rate = strictly_positive(learning_rate, "learningRate");
        fresh.core.max_iterations = static_cast<long>(positive(max_iterations, "maxIterations"));
        fresh.core.l2 = non_negative(l2, "l2");
        fresh.core.tolerance = non_negative(tolerance, "tolerance");
        reset(ZEND_THIS, std::move(fresh));
    });
}

void ZEND_FASTCALL logistic_weights(INTERNAL_FUNCTION_PARAMETERS) {
    ZEND_PARSE_PARAMETERS_NONE();
    fitted_getter<Model>(ZEND_THIS, return_value, [](const Model &m, zval *out) {
        // One row of feature weights per class, keyed by label.
        array_init_size(out, static_cast<uint32_t>(m.classes()));
        for (std::size_t k = 0; k < m.classes(); ++k) {
            zval row, label;
            const auto begin = m.core.weights.begin() + static_cast<std::ptrdiff_t>(k * m.features());
            write_doubles(&row, std::vector<double>(begin, begin + static_cast<std::ptrdiff_t>(m.features())));
            m.labels.write(static_cast<int>(k), &label);
            array_set_zval_key(Z_ARRVAL_P(out), &label, &row);
            zval_ptr_dtor(&row);
            zval_ptr_dtor(&label);
        }
    });
}

void ZEND_FASTCALL logistic_serialize(INTERNAL_FUNCTION_PARAMETERS) {
    ZEND_PARSE_PARAMETERS_NONE();
    const Model &m = native<Model>(ZEND_THIS);
    array_init(return_value);
    add_assoc_double(return_value, "learningRate", m.core.learning_rate);
    add_assoc_long(return_value, "maxIterations", m.core.max_iterations);
    add_assoc_double(return_value, "l2", m.core.l2);
    add_assoc_double(return_value, "tolerance", m.core.tolerance);
    zval classes;
    m.labels.write_all(&classes);
    add_assoc_zval(return_value, "classes", &classes);
    add_assoc_long(return_value, "features", static_cast<zend_long>(m.core.features));
    put_doubles(return_value, "weights", m.core.weights);
    put_doubles(return_value, "bias", m.core.bias);
}

void ZEND_FASTCALL logistic_unserialize(INTERNAL_FUNCTION_PARAMETERS) {
    HashTable *data;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY_HT(data)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        Model restored;
        restored.core.learning_rate = get_double(data, "learningRate");
        restored.core.max_iterations = static_cast<long>(get_long(data, "maxIterations"));
        restored.core.l2 = get_double(data, "l2");
        restored.core.tolerance = get_double(data, "tolerance");
        restored.labels.read_all(get_array(data, "classes"));
        const zend_long features = get_long(data, "features");
        std::vector<double> weights = get_doubles(data, "weights");
        std::vector<double> bias = get_doubles(data, "bias");

        const std::size_t K = restored.labels.size();
        if (K != 0) {
            if (K < 2 || features < 1 || bias.size() != K ||
                !shape_matches(weights.size(), K, static_cast<std::size_t>(features))) {
                throw std::invalid_argument("Invalid serialized data: inconsistent LogisticRegression state");
            }
            restored.core.classes = K;
            restored.core.features = static_cast<std::size_t>(features);
            restored.core.weights = std::move(weights);
            restored.core.bias = std::move(bias);
        }
        reset(ZEND_THIS, std::move(restored));
    });
}

const zend_function_entry methods[] = {
    ML_METHOD(__construct, logistic_construct, arginfo_construct)
    ML_METHOD(fit, fit_classifier<Model>, arginfo_ml_fit_supervised)
    ML_METHOD(predict, predict_classifier<Model>, arginfo_ml_predict_label)
    ML_METHOD(predictProbability, predict_proba<Model>, arginfo_ml_samples_to_array)
    ML_METHOD(getClasses, get_classes<Model>, arginfo_ml_get_array)
    ML_METHOD(getWeights, logistic_weights, arginfo_ml_get_array)
    ML_METHOD(__serialize, logistic_serialize, arginfo_ml_get_array)
    ML_METHOD(__unserialize, logistic_unserialize, arginfo_ml_unserialize)
    PHP_FE_END};

}  // namespace

void mlphp::register_logistic_regression() {
    register_class<Model>("ML\\Classification\\LogisticRegression", methods, {ce_learner, ce_classifier, ce_probabilistic});
}
