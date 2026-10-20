#include "arginfo.hpp"
#include "core/linear_regression.hpp"

namespace {

using mlphp::guard;
using Model = mlcore::LinearRegression;

ZEND_BEGIN_ARG_INFO_EX(arginfo_construct, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, l2, IS_DOUBLE, 0, "0.0")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_MASK_EX(arginfo_predict, 0, 1, MAY_BE_ARRAY | MAY_BE_DOUBLE)
    ZEND_ARG_TYPE_INFO(0, samples, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

PHP_METHOD(LinearRegression, __construct) {
    double l2 = 0.0;
    ZEND_PARSE_PARAMETERS_START(0, 1)
        Z_PARAM_OPTIONAL
        Z_PARAM_DOUBLE(l2)
    ZEND_PARSE_PARAMETERS_END();

    guard([&] {
        if (!(l2 >= 0.0)) {
            throw std::invalid_argument("l2 must be greater than or equal to 0");
        }
        Model fresh;
        fresh.l2 = l2;
        mlphp::native<Model>(ZEND_THIS) = std::move(fresh);
    });
}

PHP_METHOD(LinearRegression, fit) {
    HashTable *samples, *targets;
    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_ARRAY_HT(samples)
        Z_PARAM_ARRAY_HT(targets)
    ZEND_PARSE_PARAMETERS_END();

    if (!guard([&] {
            const mlcore::Matrix X = mlphp::read_samples(samples);
            const std::vector<double> y = mlphp::read_numbers(targets, "Target");
            mlphp::native<Model>(ZEND_THIS).fit(X, y);
        })) {
        RETURN_THROWS();
    }
    RETURN_OBJ_COPY(Z_OBJ_P(ZEND_THIS));
}

PHP_METHOD(LinearRegression, predict) {
    HashTable *samples;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY_HT(samples)
    ZEND_PARSE_PARAMETERS_END();

    guard([&] {
        const Model &m = mlphp::native<Model>(ZEND_THIS);
        mlphp::require_fitted(m.fitted(), ZEND_THIS);
        if (mlphp::is_sample_list(samples)) {
            const mlcore::Matrix X = mlphp::read_samples(samples, m.features());
            std::vector<double> out(X.rows);
            for (std::size_t r = 0; r < X.rows; ++r) {
                out[r] = m.predict_one(X.row(r));
            }
            mlphp::write_doubles(return_value, out);
        } else {
            const std::vector<double> x = mlphp::read_sample(samples, m.features());
            RETVAL_DOUBLE(m.predict_one(x.data()));
        }
    });
}

PHP_METHOD(LinearRegression, getCoefficients) {
    ZEND_PARSE_PARAMETERS_NONE();
    guard([&] {
        const Model &m = mlphp::native<Model>(ZEND_THIS);
        mlphp::require_fitted(m.fitted(), ZEND_THIS);
        mlphp::write_doubles(return_value, m.coefficients);
    });
}

PHP_METHOD(LinearRegression, getIntercept) {
    ZEND_PARSE_PARAMETERS_NONE();
    guard([&] {
        const Model &m = mlphp::native<Model>(ZEND_THIS);
        mlphp::require_fitted(m.fitted(), ZEND_THIS);
        RETVAL_DOUBLE(m.intercept);
    });
}

PHP_METHOD(LinearRegression, __serialize) {
    ZEND_PARSE_PARAMETERS_NONE();
    const Model &m = mlphp::native<Model>(ZEND_THIS);
    array_init(return_value);
    add_assoc_double(return_value, "l2", m.l2);
    mlphp::put_doubles(return_value, "coefficients", m.coefficients);
    add_assoc_double(return_value, "intercept", m.intercept);
}

PHP_METHOD(LinearRegression, __unserialize) {
    HashTable *data;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY_HT(data)
    ZEND_PARSE_PARAMETERS_END();

    guard([&] {
        Model restored;
        restored.l2 = mlphp::get_double(data, "l2");
        restored.coefficients = mlphp::get_doubles(data, "coefficients");
        restored.intercept = mlphp::get_double(data, "intercept");
        mlphp::native<Model>(ZEND_THIS) = std::move(restored);
    });
}

const zend_function_entry methods[] = {
    PHP_ME(LinearRegression, __construct, arginfo_construct, ZEND_ACC_PUBLIC)
    PHP_ME(LinearRegression, fit, arginfo_ml_fit_supervised, ZEND_ACC_PUBLIC)
    PHP_ME(LinearRegression, predict, arginfo_predict, ZEND_ACC_PUBLIC)
    PHP_ME(LinearRegression, getCoefficients, arginfo_ml_get_array, ZEND_ACC_PUBLIC)
    PHP_ME(LinearRegression, getIntercept, arginfo_ml_get_float, ZEND_ACC_PUBLIC)
    PHP_ME(LinearRegression, __serialize, arginfo_ml_get_array, ZEND_ACC_PUBLIC)
    PHP_ME(LinearRegression, __unserialize, arginfo_ml_unserialize, ZEND_ACC_PUBLIC)
    PHP_FE_END
};

}  // namespace

void mlphp::register_linear_regression() {
    register_class<Model>("ML\\Regression\\LinearRegression", methods);
}
