// Non-linear dimensionality reduction: KernelPCA, Isomap and UMAP.

#include "binding.hpp"
#include "core/manifold.hpp"

namespace {

using namespace mlphp;
using mlcore::ArchiveReader;
using mlcore::ArchiveWriter;

// ---- KernelPCA -------------------------------------------------------------------------------

struct KernelPCAModel {
    static constexpr std::uint32_t kTag = mlcore::tag("KPCA");
    mlcore::KernelPCA core;
    bool fitted() const { return core.fitted(); }
    std::size_t features() const { return core.features(); }
    std::size_t output_dim() const { return core.alphas.cols; }
    void transform_row(const double *in, double *out) const { core.transform(in, out); }
    void fit(const mlcore::Matrix &X, unsigned threads) { core.fit(X, threads); }
    void save(ArchiveWriter &w) const { core.save(w); }
    void load(ArchiveReader &r) { core.load(r); }
};

ZEND_BEGIN_ARG_INFO_EX(arginfo_kpca_construct, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, dimensions, IS_LONG, 0, "2")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, kernel, IS_STRING, 0, "\"rbf\"")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, gamma, IS_DOUBLE, 1, "null")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, degree, IS_LONG, 0, "3")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, coef0, IS_DOUBLE, 0, "1.0")
ZEND_END_ARG_INFO()

void ZEND_FASTCALL kpca_construct(INTERNAL_FUNCTION_PARAMETERS) {
    zend_long dimensions = 2, degree = 3;
    zend_string *kernel = nullptr;
    double gamma = 0.0, coef0 = 1.0;
    bool gamma_null = true;
    ZEND_PARSE_PARAMETERS_START(0, 5)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(dimensions)
        Z_PARAM_STR(kernel)
        Z_PARAM_DOUBLE_OR_NULL(gamma, gamma_null)
        Z_PARAM_LONG(degree)
        Z_PARAM_DOUBLE(coef0)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        KernelPCAModel fresh;
        fresh.core.dimensions = positive(dimensions, "dimensions");
        if (kernel) {
            fresh.core.kernel.type = mlcore::Kernel::parse(zstr(kernel));
        }
        fresh.core.kernel.gamma = gamma_null ? 0.0 : strictly_positive(gamma, "gamma");
        fresh.core.kernel.degree = static_cast<double>(positive(degree, "degree"));
        check(std::isfinite(coef0), "coef0 must be finite");
        fresh.core.kernel.coef0 = coef0;
        reset(ZEND_THIS, std::move(fresh));
    });
}

void ZEND_FASTCALL kpca_eigenvalues(INTERNAL_FUNCTION_PARAMETERS) {
    ZEND_PARSE_PARAMETERS_NONE();
    fitted_getter<KernelPCAModel>(ZEND_THIS, return_value,
                                  [](const KernelPCAModel &m, zval *out) { write_doubles(out, m.core.eigenvalues); });
}

const zend_function_entry kpca_methods[] = {
    ML_METHOD(__construct, kpca_construct, arginfo_kpca_construct)
    ML_METHOD(fit, fit_unsupervised<KernelPCAModel>, arginfo_ml_fit_unsupervised)
    ML_METHOD(transform, transform<KernelPCAModel>, arginfo_ml_samples_to_array)
    ML_METHOD(fitTransform, fit_transform<KernelPCAModel>, arginfo_ml_samples_to_array)
    ML_METHOD(getEigenvalues, kpca_eigenvalues, arginfo_ml_get_array)
    ML_METHOD(__serialize, serialize_native<KernelPCAModel>, arginfo_ml_get_array)
    ML_METHOD(__unserialize, unserialize_native<KernelPCAModel>, arginfo_ml_unserialize)
    PHP_FE_END};

// ---- Isomap ----------------------------------------------------------------------------------

struct IsomapModel {
    static constexpr std::uint32_t kTag = mlcore::tag("ISOM");
    mlcore::Isomap core;
    bool fitted() const { return core.fitted(); }
    std::size_t features() const { return core.features(); }
    std::size_t output_dim() const { return core.vectors.cols; }
    void transform_row(const double *in, double *out) const { core.transform(in, out); }
    void fit(const mlcore::Matrix &X, unsigned threads) { core.fit(X, threads); }
    void save(ArchiveWriter &w) const { core.save(w); }
    void load(ArchiveReader &r) { core.load(r); }
};

ZEND_BEGIN_ARG_INFO_EX(arginfo_isomap_construct, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, dimensions, IS_LONG, 0, "2")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, neighbors, IS_LONG, 0, "5")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, metric, IS_STRING, 0, "\"euclidean\"")
ZEND_END_ARG_INFO()

void ZEND_FASTCALL isomap_construct(INTERNAL_FUNCTION_PARAMETERS) {
    zend_long dimensions = 2, neighbors = 5;
    zend_string *metric = nullptr;
    ZEND_PARSE_PARAMETERS_START(0, 3)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(dimensions)
        Z_PARAM_LONG(neighbors)
        Z_PARAM_STR(metric)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        IsomapModel fresh;
        fresh.core.dimensions = positive(dimensions, "dimensions");
        fresh.core.neighbors = positive(neighbors, "neighbors");
        if (metric) {
            fresh.core.metric = mlcore::Distance::parse(zstr(metric));
        }
        reset(ZEND_THIS, std::move(fresh));
    });
}

void ZEND_FASTCALL isomap_eigenvalues(INTERNAL_FUNCTION_PARAMETERS) {
    ZEND_PARSE_PARAMETERS_NONE();
    fitted_getter<IsomapModel>(ZEND_THIS, return_value,
                               [](const IsomapModel &m, zval *out) { write_doubles(out, m.core.eigenvalues); });
}

const zend_function_entry isomap_methods[] = {
    ML_METHOD(__construct, isomap_construct, arginfo_isomap_construct)
    ML_METHOD(fit, fit_unsupervised<IsomapModel>, arginfo_ml_fit_unsupervised)
    ML_METHOD(transform, transform<IsomapModel>, arginfo_ml_samples_to_array)
    ML_METHOD(fitTransform, fit_transform<IsomapModel>, arginfo_ml_samples_to_array)
    ML_METHOD(getEigenvalues, isomap_eigenvalues, arginfo_ml_get_array)
    ML_METHOD(__serialize, serialize_native<IsomapModel>, arginfo_ml_get_array)
    ML_METHOD(__unserialize, unserialize_native<IsomapModel>, arginfo_ml_unserialize)
    PHP_FE_END};

// ---- UMAP ------------------------------------------------------------------------------------

struct UMAPModel {
    static constexpr std::uint32_t kTag = mlcore::tag("UMAP");
    mlcore::UMAP core;
    bool fitted() const { return core.fitted(); }
    std::size_t features() const { return core.features(); }
    void fit(const mlcore::Matrix &X, unsigned threads) { core.fit(X, threads); }
    void save(ArchiveWriter &w) const { core.save(w); }
    void load(ArchiveReader &r) { core.load(r); }
};

ZEND_BEGIN_ARG_INFO_EX(arginfo_umap_construct, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, dimensions, IS_LONG, 0, "2")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, neighbors, IS_LONG, 0, "15")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, minDistance, IS_DOUBLE, 0, "0.1")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, spread, IS_DOUBLE, 0, "1.0")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, epochs, IS_LONG, 1, "null")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, rate, IS_DOUBLE, 0, "1.0")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, negativeSamples, IS_LONG, 0, "5")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, repulsion, IS_DOUBLE, 0, "1.0")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, init, IS_STRING, 0, "\"pca\"")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, metric, IS_STRING, 0, "\"euclidean\"")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, seed, IS_LONG, 1, "null")
ZEND_END_ARG_INFO()

void ZEND_FASTCALL umap_construct(INTERNAL_FUNCTION_PARAMETERS) {
    zend_long dimensions = 2, neighbors = 15, epochs = 0, negative = 5, seed = 0;
    double min_distance = 0.1, spread = 1.0, rate = 1.0, repulsion = 1.0;
    bool epochs_null = true, seed_null = true;
    zend_string *init = nullptr, *metric = nullptr;
    ZEND_PARSE_PARAMETERS_START(0, 11)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(dimensions)
        Z_PARAM_LONG(neighbors)
        Z_PARAM_DOUBLE(min_distance)
        Z_PARAM_DOUBLE(spread)
        Z_PARAM_LONG_OR_NULL(epochs, epochs_null)
        Z_PARAM_DOUBLE(rate)
        Z_PARAM_LONG(negative)
        Z_PARAM_DOUBLE(repulsion)
        Z_PARAM_STR(init)
        Z_PARAM_STR(metric)
        Z_PARAM_LONG_OR_NULL(seed, seed_null)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        UMAPModel fresh;
        mlcore::UMAPParams &p = fresh.core.params;
        p.dimensions = positive(dimensions, "dimensions");
        p.neighbors = positive(neighbors, "neighbors");
        p.spread = strictly_positive(spread, "spread");
        p.min_distance = in_range(min_distance, 0.0, p.spread, "minDistance");
        check(epochs_null || (epochs >= 1 && epochs <= static_cast<zend_long>(mlcore::kMaxEpochs)),
              "epochs must be between 1 and 100000");
        p.epochs = epochs_null ? 0 : static_cast<std::size_t>(epochs);
        p.rate = strictly_positive(rate, "rate");
        check(negative >= 0 && negative <= 1000, "negativeSamples must be between 0 and 1000");
        p.negative_samples = static_cast<std::size_t>(negative);
        p.repulsion = non_negative(repulsion, "repulsion");
        if (init) {
            const std::string name = zstr(init);
            check(name == "pca" || name == "random", "init must be \"pca\" or \"random\"");
            p.random_init = name == "random";
        }
        if (metric) {
            p.metric = mlcore::Distance::parse(zstr(metric));
        }
        parse_seed(seed_null, seed, p.has_seed, p.seed);
        reset(ZEND_THIS, std::move(fresh));
    });
}

void ZEND_FASTCALL umap_transform(INTERNAL_FUNCTION_PARAMETERS) {
    HashTable *samples;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY_HT(samples)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        const UMAPModel &m = native<UMAPModel>(ZEND_THIS);
        require_fitted(m.fitted(), ZEND_THIS);
        if (is_sample_list(samples)) {
            write_matrix(return_value, m.core.transform(read_samples(samples, m.features()), threads()));
        } else {
            mlcore::Matrix Q(1, m.features());
            Q.data = read_sample(samples, m.features());
            const mlcore::Matrix Y = m.core.transform(Q, 1);
            write_doubles(return_value, Y.data);
        }
    });
}

// Returns the training embedding itself (transform() of the same samples
// would place them again against it).
void ZEND_FASTCALL umap_fit_transform(INTERNAL_FUNCTION_PARAMETERS) {
    HashTable *samples;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY_HT(samples)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        UMAPModel &m = native<UMAPModel>(ZEND_THIS);
        UMAPModel next = m;
        next.fit(read_samples(samples), threads());
        m = std::move(next);
        write_matrix(return_value, m.core.embedding);
    });
}

void ZEND_FASTCALL umap_embedding(INTERNAL_FUNCTION_PARAMETERS) {
    ZEND_PARSE_PARAMETERS_NONE();
    fitted_getter<UMAPModel>(ZEND_THIS, return_value, [](const UMAPModel &m, zval *out) { write_matrix(out, m.core.embedding); });
}

const zend_function_entry umap_methods[] = {
    ML_METHOD(__construct, umap_construct, arginfo_umap_construct)
    ML_METHOD(fit, fit_unsupervised<UMAPModel>, arginfo_ml_fit_unsupervised)
    ML_METHOD(transform, umap_transform, arginfo_ml_samples_to_array)
    ML_METHOD(fitTransform, umap_fit_transform, arginfo_ml_samples_to_array)
    ML_METHOD(getEmbedding, umap_embedding, arginfo_ml_get_array)
    ML_METHOD(__serialize, serialize_native<UMAPModel>, arginfo_ml_get_array)
    ML_METHOD(__unserialize, unserialize_native<UMAPModel>, arginfo_ml_unserialize)
    PHP_FE_END};

}  // namespace

void mlphp::register_manifold() {
    const auto stateful = {ce_transformer, ce_stateful};
    register_class<KernelPCAModel>("ML\\Preprocessing\\KernelPCA", kpca_methods, stateful);
    register_class<IsomapModel>("ML\\Preprocessing\\Isomap", isomap_methods, stateful);
    register_class<UMAPModel>("ML\\Preprocessing\\UMAP", umap_methods, stateful);
}
