#include "arginfo.hpp"
#include "core/gaussian_nb.hpp"

namespace {

using mlphp::guard;

struct Model {
    mlcore::GaussianNB core;
    mlphp::LabelEncoder labels;

    bool fitted() const { return core.fitted(); }
    std::size_t features() const { return core.features; }
    std::size_t classes() const { return core.classes; }
    int predict_one(const double *x) const { return core.predict_one(x); }
    void probabilities(const double *x, double *out) const { core.probabilities(x, out); }
};

ZEND_BEGIN_ARG_INFO_EX(arginfo_construct, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, varSmoothing, IS_DOUBLE, 0, "1.0E-9")
ZEND_END_ARG_INFO()

PHP_METHOD(GaussianNB, __construct) {
    double var_smoothing = 1e-9;
    ZEND_PARSE_PARAMETERS_START(0, 1)
        Z_PARAM_OPTIONAL
        Z_PARAM_DOUBLE(var_smoothing)
    ZEND_PARSE_PARAMETERS_END();

    guard([&] {
        if (!(var_smoothing >= 0.0)) {
            throw std::invalid_argument("varSmoothing must be greater than or equal to 0");
        }
        Model fresh;
        fresh.core.var_smoothing = var_smoothing;
        mlphp::native<Model>(ZEND_THIS) = std::move(fresh);
    });
}

PHP_METHOD(GaussianNB, fit) {
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

PHP_METHOD(GaussianNB, predict) {
    HashTable *samples;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY_HT(samples)
    ZEND_PARSE_PARAMETERS_END();

    guard([&] { mlphp::predict_labels(mlphp::native<Model>(ZEND_THIS), ZEND_THIS, samples, return_value); });
}

PHP_METHOD(GaussianNB, predictProbability) {
    HashTable *samples;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY_HT(samples)
    ZEND_PARSE_PARAMETERS_END();

    guard([&] { mlphp::predict_probabilities(mlphp::native<Model>(ZEND_THIS), ZEND_THIS, samples, return_value); });
}

PHP_METHOD(GaussianNB, getClasses) {
    ZEND_PARSE_PARAMETERS_NONE();
    guard([&] {
        const Model &m = mlphp::native<Model>(ZEND_THIS);
        mlphp::require_fitted(m.fitted(), ZEND_THIS);
        m.labels.write_all(return_value);
    });
}

PHP_METHOD(GaussianNB, __serialize) {
    ZEND_PARSE_PARAMETERS_NONE();
    const Model &m = mlphp::native<Model>(ZEND_THIS);
    array_init(return_value);
    add_assoc_double(return_value, "varSmoothing", m.core.var_smoothing);
    zval classes;
    m.labels.write_all(&classes);
    add_assoc_zval(return_value, "classes", &classes);
    add_assoc_long(return_value, "features", static_cast<zend_long>(m.core.features));
    mlphp::put_doubles(return_value, "means", m.core.means);
    mlphp::put_doubles(return_value, "variances", m.core.variances);
    mlphp::put_doubles(return_value, "logPriors", m.core.log_priors);
}

PHP_METHOD(GaussianNB, __unserialize) {
    HashTable *data;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY_HT(data)
    ZEND_PARSE_PARAMETERS_END();

    guard([&] {
        Model restored;
        restored.core.var_smoothing = mlphp::get_double(data, "varSmoothing");
        restored.labels.read_all(mlphp::get_array(data, "classes"));
        const zend_long features = mlphp::get_long(data, "features");
        std::vector<double> means = mlphp::get_doubles(data, "means");
        std::vector<double> variances = mlphp::get_doubles(data, "variances");
        std::vector<double> log_priors = mlphp::get_doubles(data, "logPriors");

        const std::size_t K = restored.labels.size();
        if (K != 0) {
            if (features < 1 || log_priors.size() != K ||
                !mlphp::shape_matches(means.size(), K, static_cast<std::size_t>(features)) ||
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
        mlphp::native<Model>(ZEND_THIS) = std::move(restored);
    });
}

const zend_function_entry methods[] = {
    PHP_ME(GaussianNB, __construct, arginfo_construct, ZEND_ACC_PUBLIC)
    PHP_ME(GaussianNB, fit, arginfo_ml_fit_supervised, ZEND_ACC_PUBLIC)
    PHP_ME(GaussianNB, predict, arginfo_ml_predict_label, ZEND_ACC_PUBLIC)
    PHP_ME(GaussianNB, predictProbability, arginfo_ml_samples_to_array, ZEND_ACC_PUBLIC)
    PHP_ME(GaussianNB, getClasses, arginfo_ml_get_array, ZEND_ACC_PUBLIC)
    PHP_ME(GaussianNB, __serialize, arginfo_ml_get_array, ZEND_ACC_PUBLIC)
    PHP_ME(GaussianNB, __unserialize, arginfo_ml_unserialize, ZEND_ACC_PUBLIC)
    PHP_FE_END
};

}  // namespace

void mlphp::register_gaussian_nb() {
    register_class<Model>("ML\\Classification\\GaussianNB", methods, {ce_learner, ce_classifier, ce_probabilistic});
}
