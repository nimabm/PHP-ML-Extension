// KMedoids, FuzzyCMeans, GaussianMixture, MeanShift (learners) and DBSCAN,
// OPTICS, Ward (cluster the samples passed to predict()).

#include "binding.hpp"
#include "core/clustering.hpp"

namespace {

using namespace mlphp;
using mlcore::ArchiveReader;
using mlcore::ArchiveWriter;

// Membership/probability rows as lists indexed by cluster number.
template <typename M>
void ZEND_FASTCALL cluster_probabilities(INTERNAL_FUNCTION_PARAMETERS) {
    HashTable *samples;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY_HT(samples)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        const M &m = native<M>(ZEND_THIS);
        require_fitted(m.fitted(), ZEND_THIS);
        const std::size_t k = m.clusters();
        if (is_sample_list(samples)) {
            const mlcore::Matrix X = read_samples(samples, m.features());
            mlcore::Matrix P(X.rows, k);
            mlcore::parallel_for(X.rows, threads(), [&](std::size_t r) { m.probabilities(X.row(r), P.row(r)); });
            write_matrix(return_value, P);
        } else {
            const std::vector<double> x = read_sample(samples, m.features());
            std::vector<double> p(k);
            m.probabilities(x.data(), p.data());
            write_doubles(return_value, p);
        }
    });
}

template <typename M, typename Getter>
void matrix_getter(zval *self, zval *return_value, Getter g) {
    guard([&] {
        const M &m = native<M>(self);
        require_fitted(m.fitted(), self);
        g(m, return_value);
    });
}

void parse_seed(bool seed_null, zend_long seed, bool &has_seed, std::uint64_t &out) {
    has_seed = !seed_null;
    out = static_cast<std::uint64_t>(seed);
}

// ---- KMedoids ------------------------------------------------------------------------

struct KMedoidsModel {
    static constexpr std::uint32_t kTag = mlcore::tag("KMED");
    mlcore::KMedoids core;
    bool fitted() const { return core.fitted(); }
    std::size_t features() const { return core.medoids.cols; }
    void fit(const mlcore::Matrix &X, unsigned) { core.fit(X); }
    int predict_one(const double *x) const { return core.predict_one(x); }
    void save(ArchiveWriter &w) const { core.save(w); }
    void load(ArchiveReader &r) { core.load(r); }
};

ZEND_BEGIN_ARG_INFO_EX(arginfo_kmedoids_construct, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, clusters, IS_LONG, 0, "8")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, maxIterations, IS_LONG, 0, "300")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, metric, IS_STRING, 0, "\"euclidean\"")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, seeder, IS_STRING, 0, "\"plusplus\"")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, seed, IS_LONG, 1, "null")
ZEND_END_ARG_INFO()

void ZEND_FASTCALL kmedoids_construct(INTERNAL_FUNCTION_PARAMETERS) {
    zend_long clusters = 8, max_iterations = 300, seed = 0;
    zend_string *metric = nullptr, *seeder = nullptr;
    bool seed_null = true;
    ZEND_PARSE_PARAMETERS_START(0, 5)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(clusters)
        Z_PARAM_LONG(max_iterations)
        Z_PARAM_STR(metric)
        Z_PARAM_STR(seeder)
        Z_PARAM_LONG_OR_NULL(seed, seed_null)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        KMedoidsModel fresh;
        fresh.core.clusters = positive(clusters, "clusters");
        fresh.core.max_iterations = static_cast<long>(positive(max_iterations, "maxIterations"));
        if (metric) {
            fresh.core.metric = mlcore::Distance::parse(zstr(metric));
        }
        if (seeder) {
            fresh.core.seeder = mlcore::parse_seeder(zstr(seeder));
            check(fresh.core.seeder != mlcore::Seeder::Preset, "KMedoids supports the plusplus, random and kmc2 seeders");
        }
        parse_seed(seed_null, seed, fresh.core.has_seed, fresh.core.seed);
        reset(ZEND_THIS, std::move(fresh));
    });
}

void ZEND_FASTCALL kmedoids_medoids(INTERNAL_FUNCTION_PARAMETERS) {
    ZEND_PARSE_PARAMETERS_NONE();
    matrix_getter<KMedoidsModel>(ZEND_THIS, return_value, [](const KMedoidsModel &m, zval *out) { write_matrix(out, m.core.medoids); });
}
void ZEND_FASTCALL kmedoids_labels(INTERNAL_FUNCTION_PARAMETERS) {
    ZEND_PARSE_PARAMETERS_NONE();
    matrix_getter<KMedoidsModel>(ZEND_THIS, return_value, [](const KMedoidsModel &m, zval *out) { write_ints(out, m.core.labels); });
}
void ZEND_FASTCALL kmedoids_inertia(INTERNAL_FUNCTION_PARAMETERS) {
    ZEND_PARSE_PARAMETERS_NONE();
    matrix_getter<KMedoidsModel>(ZEND_THIS, return_value, [](const KMedoidsModel &m, zval *out) { ZVAL_DOUBLE(out, m.core.inertia); });
}
void ZEND_FASTCALL kmedoids_iterations(INTERNAL_FUNCTION_PARAMETERS) {
    ZEND_PARSE_PARAMETERS_NONE();
    matrix_getter<KMedoidsModel>(ZEND_THIS, return_value, [](const KMedoidsModel &m, zval *out) { ZVAL_LONG(out, m.core.iterations); });
}

const zend_function_entry kmedoids_methods[] = {
    ML_METHOD(__construct, kmedoids_construct, arginfo_kmedoids_construct)
    ML_METHOD(fit, fit_unsupervised<KMedoidsModel>, arginfo_ml_fit_unsupervised)
    ML_METHOD(predict, predict_cluster<KMedoidsModel>, arginfo_ml_predict_int)
    ML_METHOD(getMedoids, kmedoids_medoids, arginfo_ml_get_array)
    ML_METHOD(getLabels, kmedoids_labels, arginfo_ml_get_array)
    ML_METHOD(getInertia, kmedoids_inertia, arginfo_ml_get_float)
    ML_METHOD(getIterations, kmedoids_iterations, arginfo_ml_get_int)
    ML_METHOD(__serialize, serialize_native<KMedoidsModel>, arginfo_ml_get_array)
    ML_METHOD(__unserialize, unserialize_native<KMedoidsModel>, arginfo_ml_unserialize)
    PHP_FE_END};

// ---- FuzzyCMeans -------------------------------------------------------------------------

struct FuzzyModel {
    static constexpr std::uint32_t kTag = mlcore::tag("FCMN");
    mlcore::FuzzyCMeans core;
    bool fitted() const { return core.fitted(); }
    std::size_t features() const { return core.centroids.cols; }
    std::size_t clusters() const { return core.centroids.rows; }
    void fit(const mlcore::Matrix &X, unsigned) { core.fit(X); }
    void probabilities(const double *x, double *out) const { core.memberships(x, out); }
    int predict_one(const double *x) const {
        std::vector<double> u(clusters());
        core.memberships(x, u.data());
        return static_cast<int>(mlcore::argmax(u.data(), u.size()));
    }
    void save(ArchiveWriter &w) const { core.save(w); }
    void load(ArchiveReader &r) { core.load(r); }
};

ZEND_BEGIN_ARG_INFO_EX(arginfo_fuzzy_construct, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, clusters, IS_LONG, 0, "8")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, fuzz, IS_DOUBLE, 0, "2.0")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, maxIterations, IS_LONG, 0, "300")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, minChange, IS_DOUBLE, 0, "1.0E-4")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, seed, IS_LONG, 1, "null")
ZEND_END_ARG_INFO()

void ZEND_FASTCALL fuzzy_construct(INTERNAL_FUNCTION_PARAMETERS) {
    zend_long clusters = 8, max_iterations = 300, seed = 0;
    double fuzz = 2.0, min_change = 1e-4;
    bool seed_null = true;
    ZEND_PARSE_PARAMETERS_START(0, 5)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(clusters)
        Z_PARAM_DOUBLE(fuzz)
        Z_PARAM_LONG(max_iterations)
        Z_PARAM_DOUBLE(min_change)
        Z_PARAM_LONG_OR_NULL(seed, seed_null)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        FuzzyModel fresh;
        fresh.core.clusters = positive(clusters, "clusters");
        check(fuzz > 1.0 && std::isfinite(fuzz), "fuzz must be greater than 1");
        fresh.core.fuzz = fuzz;
        fresh.core.max_iterations = static_cast<long>(positive(max_iterations, "maxIterations"));
        fresh.core.min_change = non_negative(min_change, "minChange");
        parse_seed(seed_null, seed, fresh.core.has_seed, fresh.core.seed);
        reset(ZEND_THIS, std::move(fresh));
    });
}

void ZEND_FASTCALL fuzzy_centroids(INTERNAL_FUNCTION_PARAMETERS) {
    ZEND_PARSE_PARAMETERS_NONE();
    matrix_getter<FuzzyModel>(ZEND_THIS, return_value, [](const FuzzyModel &m, zval *out) { write_matrix(out, m.core.centroids); });
}
void ZEND_FASTCALL fuzzy_inertia(INTERNAL_FUNCTION_PARAMETERS) {
    ZEND_PARSE_PARAMETERS_NONE();
    matrix_getter<FuzzyModel>(ZEND_THIS, return_value, [](const FuzzyModel &m, zval *out) { ZVAL_DOUBLE(out, m.core.inertia); });
}
void ZEND_FASTCALL fuzzy_iterations(INTERNAL_FUNCTION_PARAMETERS) {
    ZEND_PARSE_PARAMETERS_NONE();
    matrix_getter<FuzzyModel>(ZEND_THIS, return_value, [](const FuzzyModel &m, zval *out) { ZVAL_LONG(out, m.core.iterations); });
}

const zend_function_entry fuzzy_methods[] = {
    ML_METHOD(__construct, fuzzy_construct, arginfo_fuzzy_construct)
    ML_METHOD(fit, fit_unsupervised<FuzzyModel>, arginfo_ml_fit_unsupervised)
    ML_METHOD(predict, predict_cluster<FuzzyModel>, arginfo_ml_predict_int)
    ML_METHOD(predictProbability, cluster_probabilities<FuzzyModel>, arginfo_ml_samples_to_array)
    ML_METHOD(getCentroids, fuzzy_centroids, arginfo_ml_get_array)
    ML_METHOD(getInertia, fuzzy_inertia, arginfo_ml_get_float)
    ML_METHOD(getIterations, fuzzy_iterations, arginfo_ml_get_int)
    ML_METHOD(__serialize, serialize_native<FuzzyModel>, arginfo_ml_get_array)
    ML_METHOD(__unserialize, unserialize_native<FuzzyModel>, arginfo_ml_unserialize)
    PHP_FE_END};

// ---- GaussianMixture --------------------------------------------------------------------

struct MixtureModel {
    static constexpr std::uint32_t kTag = mlcore::tag("GMIX");
    mlcore::GaussianMixture core;
    bool fitted() const { return core.fitted(); }
    std::size_t features() const { return core.means.cols; }
    std::size_t clusters() const { return core.means.rows; }
    void fit(const mlcore::Matrix &X, unsigned) { core.fit(X); }
    void probabilities(const double *x, double *out) const { core.probabilities(x, out); }
    int predict_one(const double *x) const {
        std::vector<double> p(clusters());
        core.log_joint(x, p.data());
        return static_cast<int>(mlcore::argmax(p.data(), p.size()));
    }
    void save(ArchiveWriter &w) const { core.save(w); }
    void load(ArchiveReader &r) { core.load(r); }
};

ZEND_BEGIN_ARG_INFO_EX(arginfo_mixture_construct, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, components, IS_LONG, 0, "8")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, smoothing, IS_DOUBLE, 0, "1.0E-9")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, maxIterations, IS_LONG, 0, "100")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, minChange, IS_DOUBLE, 0, "0.001")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, seed, IS_LONG, 1, "null")
ZEND_END_ARG_INFO()

void ZEND_FASTCALL mixture_construct(INTERNAL_FUNCTION_PARAMETERS) {
    zend_long components = 8, max_iterations = 100, seed = 0;
    double smoothing = 1e-9, min_change = 1e-3;
    bool seed_null = true;
    ZEND_PARSE_PARAMETERS_START(0, 5)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(components)
        Z_PARAM_DOUBLE(smoothing)
        Z_PARAM_LONG(max_iterations)
        Z_PARAM_DOUBLE(min_change)
        Z_PARAM_LONG_OR_NULL(seed, seed_null)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        MixtureModel fresh;
        fresh.core.components = positive(components, "components");
        fresh.core.smoothing = non_negative(smoothing, "smoothing");
        fresh.core.max_iterations = static_cast<long>(positive(max_iterations, "maxIterations"));
        fresh.core.min_change = non_negative(min_change, "minChange");
        parse_seed(seed_null, seed, fresh.core.has_seed, fresh.core.seed);
        reset(ZEND_THIS, std::move(fresh));
    });
}

void ZEND_FASTCALL mixture_means(INTERNAL_FUNCTION_PARAMETERS) {
    ZEND_PARSE_PARAMETERS_NONE();
    matrix_getter<MixtureModel>(ZEND_THIS, return_value, [](const MixtureModel &m, zval *out) { write_matrix(out, m.core.means); });
}
void ZEND_FASTCALL mixture_variances(INTERNAL_FUNCTION_PARAMETERS) {
    ZEND_PARSE_PARAMETERS_NONE();
    matrix_getter<MixtureModel>(ZEND_THIS, return_value, [](const MixtureModel &m, zval *out) { write_matrix(out, m.core.variances); });
}
void ZEND_FASTCALL mixture_priors(INTERNAL_FUNCTION_PARAMETERS) {
    ZEND_PARSE_PARAMETERS_NONE();
    matrix_getter<MixtureModel>(ZEND_THIS, return_value, [](const MixtureModel &m, zval *out) {
        std::vector<double> p(m.core.log_priors.size());
        for (std::size_t i = 0; i < p.size(); ++i) {
            p[i] = std::exp(m.core.log_priors[i]);
        }
        write_doubles(out, p);
    });
}
void ZEND_FASTCALL mixture_log_likelihood(INTERNAL_FUNCTION_PARAMETERS) {
    ZEND_PARSE_PARAMETERS_NONE();
    matrix_getter<MixtureModel>(ZEND_THIS, return_value, [](const MixtureModel &m, zval *out) { ZVAL_DOUBLE(out, m.core.log_likelihood); });
}
void ZEND_FASTCALL mixture_iterations(INTERNAL_FUNCTION_PARAMETERS) {
    ZEND_PARSE_PARAMETERS_NONE();
    matrix_getter<MixtureModel>(ZEND_THIS, return_value, [](const MixtureModel &m, zval *out) { ZVAL_LONG(out, m.core.iterations); });
}

const zend_function_entry mixture_methods[] = {
    ML_METHOD(__construct, mixture_construct, arginfo_mixture_construct)
    ML_METHOD(fit, fit_unsupervised<MixtureModel>, arginfo_ml_fit_unsupervised)
    ML_METHOD(predict, predict_cluster<MixtureModel>, arginfo_ml_predict_int)
    ML_METHOD(predictProbability, cluster_probabilities<MixtureModel>, arginfo_ml_samples_to_array)
    ML_METHOD(getMeans, mixture_means, arginfo_ml_get_array)
    ML_METHOD(getVariances, mixture_variances, arginfo_ml_get_array)
    ML_METHOD(getPriors, mixture_priors, arginfo_ml_get_array)
    ML_METHOD(getLogLikelihood, mixture_log_likelihood, arginfo_ml_get_float)
    ML_METHOD(getIterations, mixture_iterations, arginfo_ml_get_int)
    ML_METHOD(__serialize, serialize_native<MixtureModel>, arginfo_ml_get_array)
    ML_METHOD(__unserialize, unserialize_native<MixtureModel>, arginfo_ml_unserialize)
    PHP_FE_END};

// ---- MeanShift ---------------------------------------------------------------------------

struct MeanShiftModel {
    static constexpr std::uint32_t kTag = mlcore::tag("MSHF");
    mlcore::MeanShift core;
    bool fitted() const { return core.fitted(); }
    std::size_t features() const { return core.centroids.cols; }
    void fit(const mlcore::Matrix &X, unsigned threads) { core.fit(X, threads); }
    int predict_one(const double *x) const { return core.predict_one(x); }
    void save(ArchiveWriter &w) const { core.save(w); }
    void load(ArchiveReader &r) { core.load(r); }
};

ZEND_BEGIN_ARG_INFO_EX(arginfo_mean_shift_construct, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, radius, IS_DOUBLE, 1, "null")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, percentile, IS_DOUBLE, 0, "30.0")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, maxIterations, IS_LONG, 0, "100")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, minShift, IS_DOUBLE, 0, "1.0E-4")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, maxSeeds, IS_LONG, 0, "1000")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, seed, IS_LONG, 1, "null")
ZEND_END_ARG_INFO()

void ZEND_FASTCALL mean_shift_construct(INTERNAL_FUNCTION_PARAMETERS) {
    double radius = 0.0, percentile = 30.0, min_shift = 1e-4;
    bool radius_null = true, seed_null = true;
    zend_long max_iterations = 100, max_seeds = 1000, seed = 0;
    ZEND_PARSE_PARAMETERS_START(0, 6)
        Z_PARAM_OPTIONAL
        Z_PARAM_DOUBLE_OR_NULL(radius, radius_null)
        Z_PARAM_DOUBLE(percentile)
        Z_PARAM_LONG(max_iterations)
        Z_PARAM_DOUBLE(min_shift)
        Z_PARAM_LONG(max_seeds)
        Z_PARAM_LONG_OR_NULL(seed, seed_null)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        MeanShiftModel fresh;
        fresh.core.radius = radius_null ? 0.0 : strictly_positive(radius, "radius");
        fresh.core.percentile = in_range(percentile, 0.0, 100.0, "percentile");
        fresh.core.max_iterations = static_cast<long>(positive(max_iterations, "maxIterations"));
        fresh.core.min_shift = non_negative(min_shift, "minShift");
        fresh.core.max_seeds = positive(max_seeds, "maxSeeds");
        parse_seed(seed_null, seed, fresh.core.has_seed, fresh.core.seed);
        reset(ZEND_THIS, std::move(fresh));
    });
}

void ZEND_FASTCALL mean_shift_centroids(INTERNAL_FUNCTION_PARAMETERS) {
    ZEND_PARSE_PARAMETERS_NONE();
    matrix_getter<MeanShiftModel>(ZEND_THIS, return_value, [](const MeanShiftModel &m, zval *out) { write_matrix(out, m.core.centroids); });
}
void ZEND_FASTCALL mean_shift_radius(INTERNAL_FUNCTION_PARAMETERS) {
    ZEND_PARSE_PARAMETERS_NONE();
    matrix_getter<MeanShiftModel>(ZEND_THIS, return_value, [](const MeanShiftModel &m, zval *out) { ZVAL_DOUBLE(out, m.core.fitted_radius); });
}

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_estimate_radius, 0, 1, IS_DOUBLE, 0)
    ZEND_ARG_TYPE_INFO(0, samples, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, percentile, IS_DOUBLE, 0, "30.0")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, seed, IS_LONG, 1, "null")
ZEND_END_ARG_INFO()

void ZEND_FASTCALL mean_shift_estimate(INTERNAL_FUNCTION_PARAMETERS) {
    HashTable *samples;
    double percentile = 30.0;
    zend_long seed = 0;
    bool seed_null = true;
    ZEND_PARSE_PARAMETERS_START(1, 3)
        Z_PARAM_ARRAY_HT(samples)
        Z_PARAM_OPTIONAL
        Z_PARAM_DOUBLE(percentile)
        Z_PARAM_LONG_OR_NULL(seed, seed_null)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        mlcore::Rng rng = mlcore::make_rng(!seed_null, static_cast<std::uint64_t>(seed));
        RETVAL_DOUBLE(mlcore::MeanShift::estimate_radius(read_samples(samples), in_range(percentile, 0, 100, "percentile"), rng));
    });
}

const zend_function_entry mean_shift_methods[] = {
    ML_METHOD(__construct, mean_shift_construct, arginfo_mean_shift_construct)
    ML_METHOD(fit, fit_unsupervised<MeanShiftModel>, arginfo_ml_fit_unsupervised)
    ML_METHOD(predict, predict_cluster<MeanShiftModel>, arginfo_ml_predict_int)
    ML_METHOD(getCentroids, mean_shift_centroids, arginfo_ml_get_array)
    ML_METHOD(getRadius, mean_shift_radius, arginfo_ml_get_float)
    ZEND_FENTRY(estimateRadius, mean_shift_estimate, arginfo_estimate_radius, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ML_METHOD(__serialize, serialize_native<MeanShiftModel>, arginfo_ml_get_array)
    ML_METHOD(__unserialize, unserialize_native<MeanShiftModel>, arginfo_ml_unserialize)
    PHP_FE_END};

// ---- DBSCAN, OPTICS, Ward -----------------------------------------------------------------

struct DensityParams {
    static constexpr std::uint32_t kTag = mlcore::tag("DENS");
    double radius = 0.5;
    double max_radius = std::numeric_limits<double>::infinity();
    std::size_t min_density = 5;
    std::size_t clusters = 2;
    mlcore::Distance metric;

    void save(ArchiveWriter &w) const {
        w.f64(radius);
        w.f64(max_radius);
        w.size(min_density);
        w.size(clusters);
        metric.save(w);
    }
    void load(ArchiveReader &r) {
        radius = r.f64();
        max_radius = r.f64();
        min_density = r.size();
        clusters = r.size();
        metric = mlcore::Distance::load(r);
        ArchiveReader::require(radius > 0 && min_density >= 1 && clusters >= 1);
    }
};

// Distinct tags per class so payloads cannot be swapped between them.
struct DbscanParams : DensityParams {
    static constexpr std::uint32_t kTag = mlcore::tag("DBSC");
};
struct OpticsParams : DensityParams {
    static constexpr std::uint32_t kTag = mlcore::tag("OPTC");
};
struct WardParams : DensityParams {
    static constexpr std::uint32_t kTag = mlcore::tag("WARD");
};

ZEND_BEGIN_ARG_INFO_EX(arginfo_dbscan_construct, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, radius, IS_DOUBLE, 0, "0.5")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, minDensity, IS_LONG, 0, "5")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, metric, IS_STRING, 0, "\"euclidean\"")
ZEND_END_ARG_INFO()

void ZEND_FASTCALL dbscan_construct(INTERNAL_FUNCTION_PARAMETERS) {
    double radius = 0.5;
    zend_long min_density = 5;
    zend_string *metric = nullptr;
    ZEND_PARSE_PARAMETERS_START(0, 3)
        Z_PARAM_OPTIONAL
        Z_PARAM_DOUBLE(radius)
        Z_PARAM_LONG(min_density)
        Z_PARAM_STR(metric)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        DbscanParams p;
        p.radius = strictly_positive(radius, "radius");
        p.min_density = positive(min_density, "minDensity");
        if (metric) {
            p.metric = mlcore::Distance::parse(zstr(metric));
        }
        reset(ZEND_THIS, p);
    });
}

void ZEND_FASTCALL dbscan_predict(INTERNAL_FUNCTION_PARAMETERS) {
    HashTable *samples;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY_HT(samples)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        const DbscanParams &p = native<DbscanParams>(ZEND_THIS);
        write_ints(return_value, mlcore::dbscan(read_samples(samples), p.radius, p.min_density, p.metric));
    });
}

ZEND_BEGIN_ARG_INFO_EX(arginfo_optics_construct, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, minDensity, IS_LONG, 0, "5")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, radius, IS_DOUBLE, 0, "0.5")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, maxRadius, IS_DOUBLE, 0, "INF")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, metric, IS_STRING, 0, "\"euclidean\"")
ZEND_END_ARG_INFO()

void ZEND_FASTCALL optics_construct(INTERNAL_FUNCTION_PARAMETERS) {
    zend_long min_density = 5;
    double radius = 0.5, max_radius = std::numeric_limits<double>::infinity();
    zend_string *metric = nullptr;
    ZEND_PARSE_PARAMETERS_START(0, 4)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(min_density)
        Z_PARAM_DOUBLE(radius)
        Z_PARAM_DOUBLE(max_radius)
        Z_PARAM_STR(metric)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        OpticsParams p;
        p.min_density = positive(min_density, "minDensity");
        p.radius = strictly_positive(radius, "radius");
        check(max_radius >= radius, "maxRadius must be at least radius");
        p.max_radius = max_radius;
        if (metric) {
            p.metric = mlcore::Distance::parse(zstr(metric));
        }
        reset(ZEND_THIS, p);
    });
}

mlcore::OpticsResult run_optics(zval *self, HashTable *samples) {
    const OpticsParams &p = native<OpticsParams>(self);
    return mlcore::optics(read_samples(samples), p.min_density, p.max_radius, p.radius, p.metric);
}

void ZEND_FASTCALL optics_predict(INTERNAL_FUNCTION_PARAMETERS) {
    HashTable *samples;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY_HT(samples)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] { write_ints(return_value, run_optics(ZEND_THIS, samples).labels); });
}

void ZEND_FASTCALL optics_reachability(INTERNAL_FUNCTION_PARAMETERS) {
    HashTable *samples;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY_HT(samples)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        const mlcore::OpticsResult r = run_optics(ZEND_THIS, samples);
        zval ordering, reach, core, labels;
        array_init_size(&ordering, static_cast<uint32_t>(r.ordering.size()));
        for (std::size_t i : r.ordering) {
            add_next_index_long(&ordering, static_cast<zend_long>(i));
        }
        write_doubles(&reach, r.reachability);
        write_doubles(&core, r.core_distance);
        write_ints(&labels, r.labels);
        array_init(return_value);
        add_assoc_zval(return_value, "ordering", &ordering);
        add_assoc_zval(return_value, "reachability", &reach);
        add_assoc_zval(return_value, "coreDistances", &core);
        add_assoc_zval(return_value, "labels", &labels);
    });
}

ZEND_BEGIN_ARG_INFO_EX(arginfo_ward_construct, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, clusters, IS_LONG, 0, "2")
ZEND_END_ARG_INFO()

void ZEND_FASTCALL ward_construct(INTERNAL_FUNCTION_PARAMETERS) {
    zend_long clusters = 2;
    ZEND_PARSE_PARAMETERS_START(0, 1)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(clusters)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        WardParams p;
        p.clusters = positive(clusters, "clusters");
        reset(ZEND_THIS, p);
    });
}

void ZEND_FASTCALL ward_predict(INTERNAL_FUNCTION_PARAMETERS) {
    HashTable *samples;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY_HT(samples)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] { write_ints(return_value, mlcore::ward(read_samples(samples), native<WardParams>(ZEND_THIS).clusters)); });
}

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_cluster_predict, 0, 1, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO(0, samples, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

const zend_function_entry dbscan_methods[] = {
    ML_METHOD(__construct, dbscan_construct, arginfo_dbscan_construct)
    ML_METHOD(predict, dbscan_predict, arginfo_cluster_predict)
    ML_METHOD(__serialize, serialize_native<DbscanParams>, arginfo_ml_get_array)
    ML_METHOD(__unserialize, unserialize_native<DbscanParams>, arginfo_ml_unserialize)
    PHP_FE_END};

const zend_function_entry optics_methods[] = {
    ML_METHOD(__construct, optics_construct, arginfo_optics_construct)
    ML_METHOD(predict, optics_predict, arginfo_cluster_predict)
    ML_METHOD(reachability, optics_reachability, arginfo_cluster_predict)
    ML_METHOD(__serialize, serialize_native<OpticsParams>, arginfo_ml_get_array)
    ML_METHOD(__unserialize, unserialize_native<OpticsParams>, arginfo_ml_unserialize)
    PHP_FE_END};

const zend_function_entry ward_methods[] = {
    ML_METHOD(__construct, ward_construct, arginfo_ward_construct)
    ML_METHOD(predict, ward_predict, arginfo_cluster_predict)
    ML_METHOD(__serialize, serialize_native<WardParams>, arginfo_ml_get_array)
    ML_METHOD(__unserialize, unserialize_native<WardParams>, arginfo_ml_unserialize)
    PHP_FE_END};

}  // namespace

void mlphp::register_clustering() {
    register_class<KMedoidsModel>("ML\\Clustering\\KMedoids", kmedoids_methods, {ce_learner, ce_clusterer});
    register_class<FuzzyModel>("ML\\Clustering\\FuzzyCMeans", fuzzy_methods, {ce_learner, ce_clusterer, ce_probabilistic});
    register_class<MixtureModel>("ML\\Clustering\\GaussianMixture", mixture_methods, {ce_learner, ce_clusterer, ce_probabilistic});
    register_class<MeanShiftModel>("ML\\Clustering\\MeanShift", mean_shift_methods, {ce_learner, ce_clusterer});
    register_class<DbscanParams>("ML\\Clustering\\DBSCAN", dbscan_methods, {ce_clusterer});
    register_class<OpticsParams>("ML\\Clustering\\OPTICS", optics_methods, {ce_clusterer});
    register_class<WardParams>("ML\\Clustering\\Ward", ward_methods, {ce_clusterer});
}
