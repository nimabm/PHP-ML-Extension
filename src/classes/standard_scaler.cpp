// ML\Preprocessing\StandardScaler. Serializes to the PHP array format of 0.1.0.

#include "binding.hpp"
#include "core/standard_scaler.hpp"

namespace {

using namespace mlphp;

struct Model {
    mlcore::StandardScaler core;
    bool fitted() const { return core.fitted(); }
    std::size_t features() const { return core.features(); }
    std::size_t output_dim() const { return core.features(); }
    void fit(const mlcore::Matrix &X, unsigned) { core.fit(X); }
    void transform_row(const double *in, double *out) const {
        std::copy(in, in + core.features(), out);
        core.transform(out);
    }
};

void ZEND_FASTCALL scaler_means(INTERNAL_FUNCTION_PARAMETERS) {
    ZEND_PARSE_PARAMETERS_NONE();
    fitted_getter<Model>(ZEND_THIS, return_value, [](const Model &m, zval *out) { write_doubles(out, m.core.means); });
}

void ZEND_FASTCALL scaler_scales(INTERNAL_FUNCTION_PARAMETERS) {
    ZEND_PARSE_PARAMETERS_NONE();
    fitted_getter<Model>(ZEND_THIS, return_value, [](const Model &m, zval *out) { write_doubles(out, m.core.scales); });
}

void ZEND_FASTCALL scaler_serialize(INTERNAL_FUNCTION_PARAMETERS) {
    ZEND_PARSE_PARAMETERS_NONE();
    const Model &m = native<Model>(ZEND_THIS);
    array_init(return_value);
    put_doubles(return_value, "means", m.core.means);
    put_doubles(return_value, "scales", m.core.scales);
}

void ZEND_FASTCALL scaler_unserialize(INTERNAL_FUNCTION_PARAMETERS) {
    HashTable *data;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY_HT(data)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        Model restored;
        restored.core.means = get_doubles(data, "means");
        restored.core.scales = get_doubles(data, "scales");
        if (restored.core.means.size() != restored.core.scales.size()) {
            throw std::invalid_argument("Invalid serialized data: inconsistent StandardScaler state");
        }
        for (double s : restored.core.scales) {
            if (s == 0.0) {
                throw std::invalid_argument("Invalid serialized data: scales must be non-zero");
            }
        }
        reset(ZEND_THIS, std::move(restored));
    });
}

const zend_function_entry methods[] = {
    ML_METHOD(fit, fit_unsupervised<Model>, arginfo_ml_fit_unsupervised)
    ML_METHOD(transform, transform<Model>, arginfo_ml_samples_to_array)
    ML_METHOD(fitTransform, fit_transform<Model>, arginfo_ml_samples_to_array)
    ML_METHOD(getMeans, scaler_means, arginfo_ml_get_array)
    ML_METHOD(getScales, scaler_scales, arginfo_ml_get_array)
    ML_METHOD(__serialize, scaler_serialize, arginfo_ml_get_array)
    ML_METHOD(__unserialize, scaler_unserialize, arginfo_ml_unserialize)
    PHP_FE_END};

}  // namespace

void mlphp::register_standard_scaler() {
    register_class<Model>("ML\\Preprocessing\\StandardScaler", methods, {ce_transformer, ce_stateful});
}
