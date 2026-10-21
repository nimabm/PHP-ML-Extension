#include "arginfo.hpp"
#include "core/standard_scaler.hpp"

namespace {

using mlphp::guard;
using Model = mlcore::StandardScaler;

// Standardizes `samples` (a list of samples or a single one) into return_value.
void transform_into(const Model &m, HashTable *samples, zval *return_value) {
    if (mlphp::is_sample_list(samples)) {
        mlcore::Matrix X = mlphp::read_samples(samples, m.features());
        for (std::size_t r = 0; r < X.rows; ++r) {
            m.transform(X.row(r));
        }
        mlphp::write_matrix(return_value, X);
    } else {
        std::vector<double> x = mlphp::read_sample(samples, m.features());
        m.transform(x.data());
        mlphp::write_doubles(return_value, x);
    }
}

PHP_METHOD(StandardScaler, fit) {
    HashTable *samples;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY_HT(samples)
    ZEND_PARSE_PARAMETERS_END();

    if (!guard([&] { mlphp::native<Model>(ZEND_THIS).fit(mlphp::read_samples(samples)); })) {
        RETURN_THROWS();
    }
    RETURN_OBJ_COPY(Z_OBJ_P(ZEND_THIS));
}

PHP_METHOD(StandardScaler, transform) {
    HashTable *samples;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY_HT(samples)
    ZEND_PARSE_PARAMETERS_END();

    guard([&] {
        const Model &m = mlphp::native<Model>(ZEND_THIS);
        mlphp::require_fitted(m.fitted(), ZEND_THIS);
        transform_into(m, samples, return_value);
    });
}

PHP_METHOD(StandardScaler, fitTransform) {
    HashTable *samples;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY_HT(samples)
    ZEND_PARSE_PARAMETERS_END();

    guard([&] {
        Model &m = mlphp::native<Model>(ZEND_THIS);
        mlcore::Matrix X = mlphp::read_samples(samples);
        m.fit(X);
        for (std::size_t r = 0; r < X.rows; ++r) {
            m.transform(X.row(r));
        }
        mlphp::write_matrix(return_value, X);
    });
}

PHP_METHOD(StandardScaler, getMeans) {
    ZEND_PARSE_PARAMETERS_NONE();
    guard([&] {
        const Model &m = mlphp::native<Model>(ZEND_THIS);
        mlphp::require_fitted(m.fitted(), ZEND_THIS);
        mlphp::write_doubles(return_value, m.means);
    });
}

PHP_METHOD(StandardScaler, getScales) {
    ZEND_PARSE_PARAMETERS_NONE();
    guard([&] {
        const Model &m = mlphp::native<Model>(ZEND_THIS);
        mlphp::require_fitted(m.fitted(), ZEND_THIS);
        mlphp::write_doubles(return_value, m.scales);
    });
}

PHP_METHOD(StandardScaler, __serialize) {
    ZEND_PARSE_PARAMETERS_NONE();
    const Model &m = mlphp::native<Model>(ZEND_THIS);
    array_init(return_value);
    mlphp::put_doubles(return_value, "means", m.means);
    mlphp::put_doubles(return_value, "scales", m.scales);
}

PHP_METHOD(StandardScaler, __unserialize) {
    HashTable *data;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY_HT(data)
    ZEND_PARSE_PARAMETERS_END();

    guard([&] {
        Model restored;
        restored.means = mlphp::get_doubles(data, "means");
        restored.scales = mlphp::get_doubles(data, "scales");
        if (restored.means.size() != restored.scales.size()) {
            throw std::invalid_argument("Invalid serialized data: inconsistent StandardScaler state");
        }
        for (double s : restored.scales) {
            if (s == 0.0) {
                throw std::invalid_argument("Invalid serialized data: scales must be non-zero");
            }
        }
        mlphp::native<Model>(ZEND_THIS) = std::move(restored);
    });
}

const zend_function_entry methods[] = {
    PHP_ME(StandardScaler, fit, arginfo_ml_fit_unsupervised, ZEND_ACC_PUBLIC)
    PHP_ME(StandardScaler, transform, arginfo_ml_samples_to_array, ZEND_ACC_PUBLIC)
    PHP_ME(StandardScaler, fitTransform, arginfo_ml_samples_to_array, ZEND_ACC_PUBLIC)
    PHP_ME(StandardScaler, getMeans, arginfo_ml_get_array, ZEND_ACC_PUBLIC)
    PHP_ME(StandardScaler, getScales, arginfo_ml_get_array, ZEND_ACC_PUBLIC)
    PHP_ME(StandardScaler, __serialize, arginfo_ml_get_array, ZEND_ACC_PUBLIC)
    PHP_ME(StandardScaler, __unserialize, arginfo_ml_unserialize, ZEND_ACC_PUBLIC)
    PHP_FE_END
};

}  // namespace

void mlphp::register_standard_scaler() {
    register_class<Model>("ML\\Preprocessing\\StandardScaler", methods, {ce_transformer, ce_stateful});
}
