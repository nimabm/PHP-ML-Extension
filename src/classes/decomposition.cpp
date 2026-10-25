// Dimensionality reduction: PCA, TruncatedSVD, LinearDiscriminantAnalysis,
// GaussianRandomProjector, SparseRandomProjector and TSNE.

#include "binding.hpp"
#include "core/decomposition.hpp"

namespace {

using namespace mlphp;
using mlcore::ArchiveReader;
using mlcore::ArchiveWriter;

template <typename Core, std::uint32_t Tag>
struct Projector {
    static constexpr std::uint32_t kTag = Tag;
    Core core;

    bool fitted() const { return core.projection.W.rows > 0; }
    std::size_t features() const { return core.projection.input_dim(); }
    std::size_t output_dim() const { return core.projection.output_dim(); }
    void transform_row(const double *in, double *out) const { core.projection.transform(in, out); }
    void fit(const mlcore::Matrix &X, unsigned) { core.fit(X); }
    void save(ArchiveWriter &w) const { core.save(w); }
    void load(ArchiveReader &r) {
        core.load(r);
        ArchiveReader::require(!fitted() || output_dim() == core.dimensions, "inconsistent projection");
    }
};

using PCAModel = Projector<mlcore::PCA, mlcore::tag("PCAX")>;
using SVDModel = Projector<mlcore::TruncatedSVD, mlcore::tag("TSVD")>;
using GaussianProjector = Projector<mlcore::RandomProjector, mlcore::tag("GRPJ")>;
using SparseProjector = Projector<mlcore::RandomProjector, mlcore::tag("SRPJ")>;

struct LDAModel {
    static constexpr std::uint32_t kTag = mlcore::tag("LDAX");
    mlcore::LDA core;
    bool fitted() const { return core.projection.W.rows > 0; }
    std::size_t features() const { return core.projection.input_dim(); }
    std::size_t output_dim() const { return core.projection.output_dim(); }
    void transform_row(const double *in, double *out) const { core.projection.transform(in, out); }
    void save(ArchiveWriter &w) const { core.save(w); }
    void load(ArchiveReader &r) { core.load(r); }
};

ZEND_BEGIN_ARG_INFO_EX(arginfo_dimensions_construct, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, dimensions, IS_LONG, 0, "2")
ZEND_END_ARG_INFO()

template <typename M>
void ZEND_FASTCALL dimensions_construct(INTERNAL_FUNCTION_PARAMETERS) {
    zend_long dimensions = 2;
    ZEND_PARSE_PARAMETERS_START(0, 1)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(dimensions)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        M fresh;
        fresh.core.dimensions = positive(dimensions, "dimensions");
        reset(ZEND_THIS, std::move(fresh));
    });
}

void write_ratio(zval *out, const std::vector<double> &values, double total) {
    std::vector<double> ratio(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        ratio[i] = total > 0.0 ? values[i] / total : 0.0;
    }
    write_doubles(out, ratio);
}

void write_components(zval *out, const mlcore::Matrix &W) {
    mlcore::Matrix T(W.cols, W.rows);  // one row per component
    for (std::size_t r = 0; r < W.rows; ++r) {
        for (std::size_t c = 0; c < W.cols; ++c) {
            T(c, r) = W(r, c);
        }
    }
    write_matrix(out, T);
}

template <typename M, typename F>
void fitted_getter(zval *self, F f) {
    guard([&] {
        const M &m = native<M>(self);
        require_fitted(m.fitted(), self);
        f(m);
    });
}

void ZEND_FASTCALL pca_variance(INTERNAL_FUNCTION_PARAMETERS) {
    ZEND_PARSE_PARAMETERS_NONE();
    fitted_getter<PCAModel>(ZEND_THIS, [&](const PCAModel &m) { write_doubles(return_value, m.core.explained); });
}
void ZEND_FASTCALL pca_ratio(INTERNAL_FUNCTION_PARAMETERS) {
    ZEND_PARSE_PARAMETERS_NONE();
    fitted_getter<PCAModel>(ZEND_THIS, [&](const PCAModel &m) { write_ratio(return_value, m.core.explained, m.core.total_variance); });
}
void ZEND_FASTCALL pca_components(INTERNAL_FUNCTION_PARAMETERS) {
    ZEND_PARSE_PARAMETERS_NONE();
    fitted_getter<PCAModel>(ZEND_THIS, [&](const PCAModel &m) { write_components(return_value, m.core.projection.W); });
}
void ZEND_FASTCALL svd_values(INTERNAL_FUNCTION_PARAMETERS) {
    ZEND_PARSE_PARAMETERS_NONE();
    fitted_getter<SVDModel>(ZEND_THIS, [&](const SVDModel &m) { write_doubles(return_value, m.core.singular_values); });
}
void ZEND_FASTCALL svd_ratio(INTERNAL_FUNCTION_PARAMETERS) {
    ZEND_PARSE_PARAMETERS_NONE();
    fitted_getter<SVDModel>(ZEND_THIS, [&](const SVDModel &m) {
        std::vector<double> squares;
        for (double s : m.core.singular_values) squares.push_back(s * s);
        write_ratio(return_value, squares, m.core.total);
    });
}
void ZEND_FASTCALL svd_components(INTERNAL_FUNCTION_PARAMETERS) {
    ZEND_PARSE_PARAMETERS_NONE();
    fitted_getter<SVDModel>(ZEND_THIS, [&](const SVDModel &m) { write_components(return_value, m.core.projection.W); });
}

#define ML_PROJECTOR(M)                                                            \
    ML_METHOD(fit, fit_unsupervised<M>, arginfo_ml_fit_unsupervised)                \
    ML_METHOD(transform, transform<M>, arginfo_ml_samples_to_array)                 \
    ML_METHOD(fitTransform, fit_transform<M>, arginfo_ml_samples_to_array)          \
    ML_METHOD(__serialize, serialize_native<M>, arginfo_ml_get_array)               \
    ML_METHOD(__unserialize, unserialize_native<M>, arginfo_ml_unserialize)

const zend_function_entry pca_methods[] = {
    ML_METHOD(__construct, dimensions_construct<PCAModel>, arginfo_dimensions_construct)
    ML_PROJECTOR(PCAModel)
    ML_METHOD(getExplainedVariance, pca_variance, arginfo_ml_get_array)
    ML_METHOD(getExplainedVarianceRatio, pca_ratio, arginfo_ml_get_array)
    ML_METHOD(getComponents, pca_components, arginfo_ml_get_array)
    PHP_FE_END};

const zend_function_entry svd_methods[] = {
    ML_METHOD(__construct, dimensions_construct<SVDModel>, arginfo_dimensions_construct)
    ML_PROJECTOR(SVDModel)
    ML_METHOD(getSingularValues, svd_values, arginfo_ml_get_array)
    ML_METHOD(getExplainedVarianceRatio, svd_ratio, arginfo_ml_get_array)
    ML_METHOD(getComponents, svd_components, arginfo_ml_get_array)
    PHP_FE_END};

// ---- LDA (supervised fit) ----------------------------------------------------------------

ZEND_BEGIN_ARG_INFO_EX(arginfo_lda_construct, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, dimensions, IS_LONG, 0, "1")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_lda_fit_transform, 0, 2, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO(0, samples, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO(0, labels, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

void ZEND_FASTCALL lda_construct(INTERNAL_FUNCTION_PARAMETERS) {
    zend_long dimensions = 1;
    ZEND_PARSE_PARAMETERS_START(0, 1)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(dimensions)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        LDAModel fresh;
        fresh.core.dimensions = positive(dimensions, "dimensions");
        reset(ZEND_THIS, std::move(fresh));
    });
}

void lda_train(zval *self, HashTable *samples, HashTable *labels, mlcore::Matrix *keep) {
    LDAModel &m = native<LDAModel>(self);
    mlcore::Matrix X = read_samples(samples);
    LabelEncoder encoder;
    const std::vector<int> y = encoder.fit(labels);
    LDAModel next = m;
    next.core.fit(X, y, encoder.size());
    m = std::move(next);
    if (keep) {
        *keep = std::move(X);
    }
}

void ZEND_FASTCALL lda_fit(INTERNAL_FUNCTION_PARAMETERS) {
    HashTable *samples, *labels;
    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_ARRAY_HT(samples)
        Z_PARAM_ARRAY_HT(labels)
    ZEND_PARSE_PARAMETERS_END();
    if (!guard([&] { lda_train(ZEND_THIS, samples, labels, nullptr); })) {
        RETURN_THROWS();
    }
    RETURN_OBJ_COPY(Z_OBJ_P(ZEND_THIS));
}

void ZEND_FASTCALL lda_fit_transform(INTERNAL_FUNCTION_PARAMETERS) {
    HashTable *samples, *labels;
    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_ARRAY_HT(samples)
        Z_PARAM_ARRAY_HT(labels)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        mlcore::Matrix X;
        lda_train(ZEND_THIS, samples, labels, &X);
        const LDAModel &m = native<LDAModel>(ZEND_THIS);
        mlcore::Matrix Y(X.rows, m.output_dim());
        for (std::size_t r = 0; r < X.rows; ++r) {
            m.transform_row(X.row(r), Y.row(r));
        }
        write_matrix(return_value, Y);
    });
}

void ZEND_FASTCALL lda_variance(INTERNAL_FUNCTION_PARAMETERS) {
    ZEND_PARSE_PARAMETERS_NONE();
    fitted_getter<LDAModel>(ZEND_THIS, [&](const LDAModel &m) { write_doubles(return_value, m.core.explained); });
}

const zend_function_entry lda_methods[] = {
    ML_METHOD(__construct, lda_construct, arginfo_lda_construct)
    ML_METHOD(fit, lda_fit, arginfo_ml_fit_supervised)
    ML_METHOD(transform, transform<LDAModel>, arginfo_ml_samples_to_array)
    ML_METHOD(fitTransform, lda_fit_transform, arginfo_lda_fit_transform)
    ML_METHOD(getExplainedVariance, lda_variance, arginfo_ml_get_array)
    ML_METHOD(__serialize, serialize_native<LDAModel>, arginfo_ml_get_array)
    ML_METHOD(__unserialize, unserialize_native<LDAModel>, arginfo_ml_unserialize)
    PHP_FE_END};

// ---- random projectors ----------------------------------------------------------------------------

ZEND_BEGIN_ARG_INFO_EX(arginfo_gaussian_projector_construct, 0, 0, 1)
    ZEND_ARG_TYPE_INFO(0, dimensions, IS_LONG, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, seed, IS_LONG, 1, "null")
ZEND_END_ARG_INFO()

void ZEND_FASTCALL gaussian_projector_construct(INTERNAL_FUNCTION_PARAMETERS) {
    zend_long dimensions, seed = 0;
    bool seed_null = true;
    ZEND_PARSE_PARAMETERS_START(1, 2)
        Z_PARAM_LONG(dimensions)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG_OR_NULL(seed, seed_null)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        GaussianProjector fresh;
        fresh.core.dimensions = positive(dimensions, "dimensions");
        fresh.core.has_seed = !seed_null;
        fresh.core.seed = static_cast<std::uint64_t>(seed);
        reset(ZEND_THIS, std::move(fresh));
    });
}

ZEND_BEGIN_ARG_INFO_EX(arginfo_sparse_projector_construct, 0, 0, 1)
    ZEND_ARG_TYPE_INFO(0, dimensions, IS_LONG, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, sparsity, IS_DOUBLE, 0, "3.0")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, seed, IS_LONG, 1, "null")
ZEND_END_ARG_INFO()

void ZEND_FASTCALL sparse_projector_construct(INTERNAL_FUNCTION_PARAMETERS) {
    zend_long dimensions, seed = 0;
    double sparsity = 3.0;
    bool seed_null = true;
    ZEND_PARSE_PARAMETERS_START(1, 3)
        Z_PARAM_LONG(dimensions)
        Z_PARAM_OPTIONAL
        Z_PARAM_DOUBLE(sparsity)
        Z_PARAM_LONG_OR_NULL(seed, seed_null)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        SparseProjector fresh;
        fresh.core.dimensions = positive(dimensions, "dimensions");
        check(sparsity >= 1.0 && std::isfinite(sparsity), "sparsity must be at least 1");
        fresh.core.sparse = true;
        fresh.core.sparsity = sparsity;
        fresh.core.has_seed = !seed_null;
        fresh.core.seed = static_cast<std::uint64_t>(seed);
        reset(ZEND_THIS, std::move(fresh));
    });
}

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_min_dimensions, 0, 1, IS_LONG, 0)
    ZEND_ARG_TYPE_INFO(0, samples, IS_LONG, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, maxDistortion, IS_DOUBLE, 0, "0.5")
ZEND_END_ARG_INFO()

void ZEND_FASTCALL projector_min_dimensions(INTERNAL_FUNCTION_PARAMETERS) {
    zend_long samples;
    double distortion = 0.5;
    ZEND_PARSE_PARAMETERS_START(1, 2)
        Z_PARAM_LONG(samples)
        Z_PARAM_OPTIONAL
        Z_PARAM_DOUBLE(distortion)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        check(distortion > 0.0 && distortion < 1.0, "maxDistortion must be between 0 and 1 (exclusive)");
        RETVAL_LONG(static_cast<zend_long>(mlcore::RandomProjector::min_dimensions(positive(samples, "samples"), distortion)));
    });
}

#define ML_STATIC(name, handler, arginfo) ZEND_FENTRY(name, (handler), arginfo, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)

const zend_function_entry gaussian_projector_methods[] = {
    ML_METHOD(__construct, gaussian_projector_construct, arginfo_gaussian_projector_construct)
    ML_PROJECTOR(GaussianProjector)
    ML_STATIC(minDimensions, projector_min_dimensions, arginfo_min_dimensions)
    PHP_FE_END};

const zend_function_entry sparse_projector_methods[] = {
    ML_METHOD(__construct, sparse_projector_construct, arginfo_sparse_projector_construct)
    ML_PROJECTOR(SparseProjector)
    ML_STATIC(minDimensions, projector_min_dimensions, arginfo_min_dimensions)
    PHP_FE_END};

// ---- TSNE --------------------------------------------------------------------------------------

struct TSNEModel {
    static constexpr std::uint32_t kTag = mlcore::tag("TSNE");
    mlcore::TSNEParams params;
    std::vector<double> losses;
    void save(ArchiveWriter &w) const { params.save(w); }
    void load(ArchiveReader &r) { params.load(r); }
};

ZEND_BEGIN_ARG_INFO_EX(arginfo_tsne_construct, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, dimensions, IS_LONG, 0, "2")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, rate, IS_DOUBLE, 0, "100.0")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, perplexity, IS_DOUBLE, 0, "30.0")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, exaggeration, IS_DOUBLE, 0, "12.0")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, epochs, IS_LONG, 0, "1000")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, minGradient, IS_DOUBLE, 0, "1.0E-7")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, seed, IS_LONG, 1, "null")
ZEND_END_ARG_INFO()

void ZEND_FASTCALL tsne_construct(INTERNAL_FUNCTION_PARAMETERS) {
    zend_long dimensions = 2, epochs = 1000, seed = 0;
    double rate = 100.0, perplexity = 30.0, exaggeration = 12.0, min_gradient = 1e-7;
    bool seed_null = true;
    ZEND_PARSE_PARAMETERS_START(0, 7)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(dimensions)
        Z_PARAM_DOUBLE(rate)
        Z_PARAM_DOUBLE(perplexity)
        Z_PARAM_DOUBLE(exaggeration)
        Z_PARAM_LONG(epochs)
        Z_PARAM_DOUBLE(min_gradient)
        Z_PARAM_LONG_OR_NULL(seed, seed_null)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        TSNEModel fresh;
        fresh.params.dimensions = positive(dimensions, "dimensions");
        fresh.params.rate = strictly_positive(rate, "rate");
        fresh.params.perplexity = strictly_positive(perplexity, "perplexity");
        check(exaggeration >= 1.0, "exaggeration must be at least 1");
        fresh.params.exaggeration = exaggeration;
        fresh.params.epochs = positive(epochs, "epochs");
        fresh.params.min_gradient = non_negative(min_gradient, "minGradient");
        fresh.params.has_seed = !seed_null;
        fresh.params.seed = static_cast<std::uint64_t>(seed);
        reset(ZEND_THIS, std::move(fresh));
    });
}

void ZEND_FASTCALL tsne_embed(INTERNAL_FUNCTION_PARAMETERS) {
    HashTable *samples;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY_HT(samples)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        TSNEModel &m = native<TSNEModel>(ZEND_THIS);
        std::vector<double> losses;
        const mlcore::Matrix Y = mlcore::tsne(read_samples(samples), m.params, threads(), &losses);
        m.losses = std::move(losses);
        write_matrix(return_value, Y);
    });
}

void ZEND_FASTCALL tsne_losses(INTERNAL_FUNCTION_PARAMETERS) {
    ZEND_PARSE_PARAMETERS_NONE();
    write_doubles(return_value, native<TSNEModel>(ZEND_THIS).losses);
}

const zend_function_entry tsne_methods[] = {
    ML_METHOD(__construct, tsne_construct, arginfo_tsne_construct)
    ML_METHOD(transform, tsne_embed, arginfo_ml_samples_to_array)
    ML_METHOD(fitTransform, tsne_embed, arginfo_ml_samples_to_array)
    ML_METHOD(getLosses, tsne_losses, arginfo_ml_get_array)
    ML_METHOD(__serialize, serialize_native<TSNEModel>, arginfo_ml_get_array)
    ML_METHOD(__unserialize, unserialize_native<TSNEModel>, arginfo_ml_unserialize)
    PHP_FE_END};

}  // namespace

void mlphp::register_decomposition() {
    const auto stateful = {ce_transformer, ce_stateful};
    register_class<PCAModel>("ML\\Preprocessing\\PCA", pca_methods, stateful);
    register_class<SVDModel>("ML\\Preprocessing\\TruncatedSVD", svd_methods, stateful);
    register_class<LDAModel>("ML\\Preprocessing\\LinearDiscriminantAnalysis", lda_methods, stateful);
    register_class<GaussianProjector>("ML\\Preprocessing\\GaussianRandomProjector", gaussian_projector_methods, stateful);
    register_class<SparseProjector>("ML\\Preprocessing\\SparseRandomProjector", sparse_projector_methods, stateful);
    register_class<TSNEModel>("ML\\Preprocessing\\TSNE", tsne_methods, {ce_transformer});
}
