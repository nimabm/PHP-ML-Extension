// Numeric transformers: MinMaxNormalizer, MaxAbsoluteScaler, RobustStandardizer,
// PowerTransformer (stateful) and L1Normalizer, L2Normalizer, PolynomialExpander
// (stateless).

#include "binding.hpp"
#include "core/transformers.hpp"

namespace {

using namespace mlphp;
using mlcore::ArchiveReader;
using mlcore::ArchiveWriter;

template <typename Core, std::uint32_t Tag>
struct Stateful {
    static constexpr std::uint32_t kTag = Tag;
    Core core;

    bool fitted() const { return core.features() > 0; }
    std::size_t features() const { return core.features(); }
    std::size_t output_dim() const { return core.features(); }
    void transform_row(const double *in, double *out) const { core.transform(in, out); }
    void fit(const mlcore::Matrix &X, unsigned) { core.fit(X); }
    void save(ArchiveWriter &w) const { core.save(w); }
    void load(ArchiveReader &r) { core.load(r); }
};

using MinMax = Stateful<mlcore::MinMaxNormalizer, mlcore::tag("MMNM")>;
using MaxAbs = Stateful<mlcore::MaxAbsoluteScaler, mlcore::tag("MXAB")>;
using Robust = Stateful<mlcore::RobustStandardizer, mlcore::tag("RBST")>;
using Power = Stateful<mlcore::PowerTransformer, mlcore::tag("PWRT")>;

ZEND_BEGIN_ARG_INFO_EX(arginfo_min_max_construct, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, min, IS_DOUBLE, 0, "0.0")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, max, IS_DOUBLE, 0, "1.0")
ZEND_END_ARG_INFO()

void ZEND_FASTCALL min_max_construct(INTERNAL_FUNCTION_PARAMETERS) {
    double min = 0.0, max = 1.0;
    ZEND_PARSE_PARAMETERS_START(0, 2)
        Z_PARAM_OPTIONAL
        Z_PARAM_DOUBLE(min)
        Z_PARAM_DOUBLE(max)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        check(min < max && std::isfinite(min) && std::isfinite(max), "min must be less than max");
        MinMax fresh;
        fresh.core.min = min;
        fresh.core.max = max;
        reset(ZEND_THIS, std::move(fresh));
    });
}

void ZEND_FASTCALL max_abs_construct(INTERNAL_FUNCTION_PARAMETERS) {
    ZEND_PARSE_PARAMETERS_NONE();
    reset(ZEND_THIS, MaxAbs{});
}

ZEND_BEGIN_ARG_INFO_EX(arginfo_robust_construct, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, center, _IS_BOOL, 0, "true")
ZEND_END_ARG_INFO()

void ZEND_FASTCALL robust_construct(INTERNAL_FUNCTION_PARAMETERS) {
    bool center = true;
    ZEND_PARSE_PARAMETERS_START(0, 1)
        Z_PARAM_OPTIONAL
        Z_PARAM_BOOL(center)
    ZEND_PARSE_PARAMETERS_END();
    Robust fresh;
    fresh.core.center = center;
    reset(ZEND_THIS, std::move(fresh));
}

ZEND_BEGIN_ARG_INFO_EX(arginfo_power_construct, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, lambdas, IS_ARRAY, 1, "null")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, normalize, _IS_BOOL, 0, "true")
ZEND_END_ARG_INFO()

void ZEND_FASTCALL power_construct(INTERNAL_FUNCTION_PARAMETERS) {
    HashTable *lambdas = nullptr;
    bool normalize = true;
    ZEND_PARSE_PARAMETERS_START(0, 2)
        Z_PARAM_OPTIONAL
        Z_PARAM_ARRAY_HT_OR_NULL(lambdas)
        Z_PARAM_BOOL(normalize)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        Power fresh;
        if (lambdas) {
            fresh.core.fixed_lambdas = read_numbers(lambdas, "Lambda");
            check(!fresh.core.fixed_lambdas.empty(), "lambdas must not be empty");
        }
        fresh.core.normalize = normalize;
        reset(ZEND_THIS, std::move(fresh));
    });
}

void ZEND_FASTCALL power_lambdas(INTERNAL_FUNCTION_PARAMETERS) {
    ZEND_PARSE_PARAMETERS_NONE();
    guard([&] {
        const Power &m = native<Power>(ZEND_THIS);
        require_fitted(m.fitted(), ZEND_THIS);
        write_doubles(return_value, m.core.lambdas);
    });
}

#define ML_STATEFUL(M)                                                             \
    ML_METHOD(fit, fit_unsupervised<M>, arginfo_ml_fit_unsupervised)                \
    ML_METHOD(transform, transform<M>, arginfo_ml_samples_to_array)                 \
    ML_METHOD(fitTransform, fit_transform<M>, arginfo_ml_samples_to_array)          \
    ML_METHOD(__serialize, serialize_native<M>, arginfo_ml_get_array)               \
    ML_METHOD(__unserialize, unserialize_native<M>, arginfo_ml_unserialize)

const zend_function_entry min_max_methods[] = {
    ML_METHOD(__construct, min_max_construct, arginfo_min_max_construct) ML_STATEFUL(MinMax) PHP_FE_END};
const zend_function_entry max_abs_methods[] = {
    ML_METHOD(__construct, max_abs_construct, arginfo_ml_none) ML_STATEFUL(MaxAbs) PHP_FE_END};
const zend_function_entry robust_methods[] = {
    ML_METHOD(__construct, robust_construct, arginfo_robust_construct) ML_STATEFUL(Robust) PHP_FE_END};
const zend_function_entry power_methods[] = {
    ML_METHOD(__construct, power_construct, arginfo_power_construct)
    ML_STATEFUL(Power)
    ML_METHOD(getLambdas, power_lambdas, arginfo_ml_get_array)
    PHP_FE_END};

// ---- stateless ------------------------------------------------------------------------

// Stateless transformers keep only their parameters; transform() accepts any width.
template <int Kind>  // 0 = L1, 1 = L2, 2 = polynomial
struct Stateless {
    static constexpr std::uint32_t kTag = Kind == 0 ? mlcore::tag("L1NM") : Kind == 1 ? mlcore::tag("L2NM") : mlcore::tag("POLY");
    std::size_t degree = 2;

    std::size_t output_dim(std::size_t d) const { return Kind == 2 ? d * degree : d; }
    void transform_row(const double *in, std::size_t d, double *out) const {
        if (Kind == 2) {
            for (std::size_t j = 0; j < d; ++j) {
                double power = in[j];
                for (std::size_t p = 0; p < degree; ++p) {
                    out[j * degree + p] = power;
                    power *= in[j];
                }
            }
            return;
        }
        double norm = 0.0;
        for (std::size_t j = 0; j < d; ++j) {
            norm += Kind == 0 ? std::fabs(in[j]) : in[j] * in[j];
        }
        if (Kind == 1) {
            norm = std::sqrt(norm);
        }
        for (std::size_t j = 0; j < d; ++j) {
            out[j] = norm == 0.0 ? 0.0 : in[j] / norm;
        }
    }
    void save(ArchiveWriter &w) const { w.size(degree); }
    void load(ArchiveReader &r) {
        degree = r.size();
        ArchiveReader::require(degree >= 1 && degree <= 64);
    }
};

using L1 = Stateless<0>;
using L2 = Stateless<1>;
using Polynomial = Stateless<2>;

template <typename M>
void ZEND_FASTCALL stateless_transform(INTERNAL_FUNCTION_PARAMETERS) {
    HashTable *samples;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY_HT(samples)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        const M &m = native<M>(ZEND_THIS);
        if (is_sample_list(samples)) {
            const mlcore::Matrix X = read_samples(samples);
            mlcore::Matrix Y(X.rows, m.output_dim(X.cols));
            for (std::size_t r = 0; r < X.rows; ++r) {
                m.transform_row(X.row(r), X.cols, Y.row(r));
            }
            write_matrix(return_value, Y);
        } else {
            const std::vector<double> x = read_sample(samples, zend_hash_num_elements(samples));
            std::vector<double> y(m.output_dim(x.size()));
            m.transform_row(x.data(), x.size(), y.data());
            write_doubles(return_value, y);
        }
    });
}

ZEND_BEGIN_ARG_INFO_EX(arginfo_polynomial_construct, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, degree, IS_LONG, 0, "2")
ZEND_END_ARG_INFO()

void ZEND_FASTCALL polynomial_construct(INTERNAL_FUNCTION_PARAMETERS) {
    zend_long degree = 2;
    ZEND_PARSE_PARAMETERS_START(0, 1)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(degree)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        check(degree >= 1 && degree <= 64, "degree must be between 1 and 64");
        Polynomial fresh;
        fresh.degree = static_cast<std::size_t>(degree);
        reset(ZEND_THIS, fresh);
    });
}

template <typename M>
void ZEND_FASTCALL stateless_construct(INTERNAL_FUNCTION_PARAMETERS) {
    ZEND_PARSE_PARAMETERS_NONE();
}

#define ML_STATELESS(M)                                                            \
    ML_METHOD(transform, stateless_transform<M>, arginfo_ml_samples_to_array)       \
    ML_METHOD(__serialize, serialize_native<M>, arginfo_ml_get_array)               \
    ML_METHOD(__unserialize, unserialize_native<M>, arginfo_ml_unserialize)

const zend_function_entry l1_methods[] = {
    ML_METHOD(__construct, stateless_construct<L1>, arginfo_ml_none) ML_STATELESS(L1) PHP_FE_END};
const zend_function_entry l2_methods[] = {
    ML_METHOD(__construct, stateless_construct<L2>, arginfo_ml_none) ML_STATELESS(L2) PHP_FE_END};
const zend_function_entry polynomial_methods[] = {
    ML_METHOD(__construct, polynomial_construct, arginfo_polynomial_construct) ML_STATELESS(Polynomial) PHP_FE_END};

}  // namespace

void mlphp::register_preprocessing() {
    const auto stateful = {ce_transformer, ce_stateful};
    register_class<MinMax>("ML\\Preprocessing\\MinMaxNormalizer", min_max_methods, stateful);
    register_class<MaxAbs>("ML\\Preprocessing\\MaxAbsoluteScaler", max_abs_methods, stateful);
    register_class<Robust>("ML\\Preprocessing\\RobustStandardizer", robust_methods, stateful);
    register_class<Power>("ML\\Preprocessing\\PowerTransformer", power_methods, stateful);
    register_class<L1>("ML\\Preprocessing\\L1Normalizer", l1_methods, {ce_transformer});
    register_class<L2>("ML\\Preprocessing\\L2Normalizer", l2_methods, {ce_transformer});
    register_class<Polynomial>("ML\\Preprocessing\\PolynomialExpander", polynomial_methods, {ce_transformer});
}
