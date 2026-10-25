#include "arginfo.hpp"
#include "core/logistic_regression.hpp"

namespace {

using mlphp::guard;

struct Model {
    mlcore::LogisticRegression core;
    mlphp::LabelEncoder labels;

    bool fitted() const { return core.fitted(); }
    std::size_t features() const { return core.features; }
    std::size_t classes() const { return core.classes; }
    int predict_one(const double *x) const { return core.predict_one(x); }
    void probabilities(const double *x, double *out) const { core.probabilities(x, out); }
};

ZEND_BEGIN_ARG_INFO_EX(arginfo_construct, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, learningRate, IS_DOUBLE, 0, "0.1")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, maxIterations, IS_LONG, 0, "1000")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, l2, IS_DOUBLE, 0, "0.0")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, tolerance, IS_DOUBLE, 0, "1.0E-6")
ZEND_END_ARG_INFO()

PHP_METHOD(LogisticRegression, __construct) {
    double learning_rate = 0.1;
    zend_long max_iterations = 1000;
    double l2 = 0.0;
    double tolerance = 1e-6;
    ZEND_PARSE_PARAMETERS_START(0, 4)
        Z_PARAM_OPTIONAL
        Z_PARAM_DOUBLE(learning_rate)
        Z_PARAM_LONG(max_iterations)
        Z_PARAM_DOUBLE(l2)
        Z_PARAM_DOUBLE(tolerance)
    ZEND_PARSE_PARAMETERS_END();

    guard([&] {
        if (!(learning_rate > 0.0)) {
            throw std::invalid_argument("learningRate must be greater than 0");
        }
        if (max_iterations < 1) {
            throw std::invalid_argument("maxIterations must be at least 1");
        }
        if (!(l2 >= 0.0)) {
            throw std::invalid_argument("l2 must be greater than or equal to 0");
        }
        if (!(tolerance >= 0.0)) {
            throw std::invalid_argument("tolerance must be greater than or equal to 0");
        }
        Model fresh;
        fresh.core.learning_rate = learning_rate;
        fresh.core.max_iterations = static_cast<long>(max_iterations);
        fresh.core.l2 = l2;
        fresh.core.tolerance = tolerance;
        mlphp::native<Model>(ZEND_THIS) = std::move(fresh);
    });
}

PHP_METHOD(LogisticRegression, fit) {
    HashTable *samples, *labels;
    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_ARRAY_HT(samples)
        Z_PARAM_ARRAY_HT(labels)
    ZEND_PARSE_PARAMETERS_END();

    if (!guard([&] {
            Model &m = mlphp::native<Model>(ZEND_THIS);
            const mlcore::Matrix X = mlphp::read_samples(samples);
            mlphp::LabelEncoder encoder;
            const std::vector<int> y = encoder.fit(labels);
            m.core.fit(X, y, encoder.size());
            m.labels = std::move(encoder);
        })) {
        RETURN_THROWS();
    }
    RETURN_OBJ_COPY(Z_OBJ_P(ZEND_THIS));
}

PHP_METHOD(LogisticRegression, predict) {
    HashTable *samples;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY_HT(samples)
    ZEND_PARSE_PARAMETERS_END();

    guard([&] { mlphp::predict_labels(mlphp::native<Model>(ZEND_THIS), ZEND_THIS, samples, return_value); });
}

PHP_METHOD(LogisticRegression, predictProbability) {
    HashTable *samples;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY_HT(samples)
    ZEND_PARSE_PARAMETERS_END();

    guard([&] { mlphp::predict_probabilities(mlphp::native<Model>(ZEND_THIS), ZEND_THIS, samples, return_value); });
}

PHP_METHOD(LogisticRegression, getClasses) {
    ZEND_PARSE_PARAMETERS_NONE();
    guard([&] {
        const Model &m = mlphp::native<Model>(ZEND_THIS);
        mlphp::require_fitted(m.fitted(), ZEND_THIS);
        m.labels.write_all(return_value);
    });
}

PHP_METHOD(LogisticRegression, getWeights) {
    ZEND_PARSE_PARAMETERS_NONE();
    guard([&] {
        const Model &m = mlphp::native<Model>(ZEND_THIS);
        mlphp::require_fitted(m.fitted(), ZEND_THIS);
        // One row of feature weights per class, keyed by label.
        array_init_size(return_value, static_cast<uint32_t>(m.classes()));
        for (std::size_t k = 0; k < m.classes(); ++k) {
            zval row, label;
            const auto begin = m.core.weights.begin() + static_cast<std::ptrdiff_t>(k * m.features());
            mlphp::write_doubles(&row, std::vector<double>(begin, begin + static_cast<std::ptrdiff_t>(m.features())));
            m.labels.write(static_cast<int>(k), &label);
            array_set_zval_key(Z_ARRVAL_P(return_value), &label, &row);
            zval_ptr_dtor(&row);
            zval_ptr_dtor(&label);
        }
    });
}

PHP_METHOD(LogisticRegression, __serialize) {
    ZEND_PARSE_PARAMETERS_NONE();
    const Model &m = mlphp::native<Model>(ZEND_THIS);
    array_init(return_value);
    add_assoc_double(return_value, "learningRate", m.core.learning_rate);
    add_assoc_long(return_value, "maxIterations", m.core.max_iterations);
    add_assoc_double(return_value, "l2", m.core.l2);
    add_assoc_double(return_value, "tolerance", m.core.tolerance);
    zval classes;
    m.labels.write_all(&classes);
    add_assoc_zval(return_value, "classes", &classes);
    add_assoc_long(return_value, "features", static_cast<zend_long>(m.core.features));
    mlphp::put_doubles(return_value, "weights", m.core.weights);
    mlphp::put_doubles(return_value, "bias", m.core.bias);
}

PHP_METHOD(LogisticRegression, __unserialize) {
    HashTable *data;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY_HT(data)
    ZEND_PARSE_PARAMETERS_END();

    guard([&] {
        Model restored;
        restored.core.learning_rate = mlphp::get_double(data, "learningRate");
        restored.core.max_iterations = static_cast<long>(mlphp::get_long(data, "maxIterations"));
        restored.core.l2 = mlphp::get_double(data, "l2");
        restored.core.tolerance = mlphp::get_double(data, "tolerance");
        restored.labels.read_all(mlphp::get_array(data, "classes"));
        const zend_long features = mlphp::get_long(data, "features");
        std::vector<double> weights = mlphp::get_doubles(data, "weights");
        std::vector<double> bias = mlphp::get_doubles(data, "bias");

        const std::size_t K = restored.labels.size();
        if (K != 0) {
            if (K < 2 || features < 1 || bias.size() != K ||
                !mlphp::shape_matches(weights.size(), K, static_cast<std::size_t>(features))) {
                throw std::invalid_argument("Invalid serialized data: inconsistent LogisticRegression state");
            }
            restored.core.classes = K;
            restored.core.features = static_cast<std::size_t>(features);
            restored.core.weights = std::move(weights);
            restored.core.bias = std::move(bias);
        }
        mlphp::native<Model>(ZEND_THIS) = std::move(restored);
    });
}

const zend_function_entry methods[] = {
    PHP_ME(LogisticRegression, __construct, arginfo_construct, ZEND_ACC_PUBLIC)
    PHP_ME(LogisticRegression, fit, arginfo_ml_fit_supervised, ZEND_ACC_PUBLIC)
    PHP_ME(LogisticRegression, predict, arginfo_ml_predict_label, ZEND_ACC_PUBLIC)
    PHP_ME(LogisticRegression, predictProbability, arginfo_ml_samples_to_array, ZEND_ACC_PUBLIC)
    PHP_ME(LogisticRegression, getClasses, arginfo_ml_get_array, ZEND_ACC_PUBLIC)
    PHP_ME(LogisticRegression, getWeights, arginfo_ml_get_array, ZEND_ACC_PUBLIC)
    PHP_ME(LogisticRegression, __serialize, arginfo_ml_get_array, ZEND_ACC_PUBLIC)
    PHP_ME(LogisticRegression, __unserialize, arginfo_ml_unserialize, ZEND_ACC_PUBLIC)
    PHP_FE_END
};

}  // namespace

void mlphp::register_logistic_regression() {
    register_class<Model>("ML\\Classification\\LogisticRegression", methods, {ce_learner, ce_classifier, ce_probabilistic});
}
