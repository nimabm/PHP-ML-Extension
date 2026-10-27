// ML\Regression\LinearRegression. Serializes to the PHP array format of 0.1.0.

#include "binding.hpp"
#include "core/linear_regression.hpp"

namespace {

using namespace mlphp;

struct Model {
    mlcore::LinearRegression core;
    bool fitted() const { return core.fitted(); }
    std::size_t features() const { return core.features(); }
    void fit(const mlcore::Matrix &X, const std::vector<double> &y, unsigned) { core.fit(X, y); }
    double predict_one(const double *x) const { return core.predict_one(x); }
};

ZEND_BEGIN_ARG_INFO_EX(arginfo_construct, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, l2, IS_DOUBLE, 0, "0.0")
ZEND_END_ARG_INFO()

void ZEND_FASTCALL linear_construct(INTERNAL_FUNCTION_PARAMETERS) {
    double l2 = 0.0;
    ZEND_PARSE_PARAMETERS_START(0, 1)
        Z_PARAM_OPTIONAL
        Z_PARAM_DOUBLE(l2)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        Model fresh;
        fresh.core.l2 = non_negative(l2, "l2");
        reset(ZEND_THIS, std::move(fresh));
    });
}

void ZEND_FASTCALL linear_coefficients(INTERNAL_FUNCTION_PARAMETERS) {
    ZEND_PARSE_PARAMETERS_NONE();
    fitted_getter<Model>(ZEND_THIS, return_value, [](const Model &m, zval *out) { write_doubles(out, m.core.coefficients); });
}

void ZEND_FASTCALL linear_intercept(INTERNAL_FUNCTION_PARAMETERS) {
    ZEND_PARSE_PARAMETERS_NONE();
    fitted_getter<Model>(ZEND_THIS, return_value, [](const Model &m, zval *out) { ZVAL_DOUBLE(out, m.core.intercept); });
}

void ZEND_FASTCALL linear_serialize(INTERNAL_FUNCTION_PARAMETERS) {
    ZEND_PARSE_PARAMETERS_NONE();
    const Model &m = native<Model>(ZEND_THIS);
    array_init(return_value);
    add_assoc_double(return_value, "l2", m.core.l2);
    put_doubles(return_value, "coefficients", m.core.coefficients);
    add_assoc_double(return_value, "intercept", m.core.intercept);
}

void ZEND_FASTCALL linear_unserialize(INTERNAL_FUNCTION_PARAMETERS) {
    HashTable *data;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY_HT(data)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        Model restored;
        restored.core.l2 = get_double(data, "l2");
        restored.core.coefficients = get_doubles(data, "coefficients");
        restored.core.intercept = get_double(data, "intercept");
        reset(ZEND_THIS, std::move(restored));
    });
}

const zend_function_entry methods[] = {
    ML_METHOD(__construct, linear_construct, arginfo_construct)
    ML_METHOD(fit, fit_regressor<Model>, arginfo_ml_fit_supervised)
    ML_METHOD(predict, predict_regressor<Model>, arginfo_ml_predict_float)
    ML_METHOD(getCoefficients, linear_coefficients, arginfo_ml_get_array)
    ML_METHOD(getIntercept, linear_intercept, arginfo_ml_get_float)
    ML_METHOD(__serialize, linear_serialize, arginfo_ml_get_array)
    ML_METHOD(__unserialize, linear_unserialize, arginfo_ml_unserialize)
    PHP_FE_END};

}  // namespace

void mlphp::register_linear_regression() {
    register_class<Model>("ML\\Regression\\LinearRegression", methods, {ce_learner, ce_regressor});
}
