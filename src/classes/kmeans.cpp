// ML\Clustering\KMeans. Serializes to the PHP array format of 0.1.0 (plus the
// optional "seeder" and "presetCentroids" keys added in 0.1.2).

#include "binding.hpp"
#include "core/kmeans.hpp"

namespace {

using namespace mlphp;

struct Model {
    mlcore::KMeans core;
    bool fitted() const { return core.fitted(); }
    std::size_t features() const { return core.features(); }
    std::size_t clusters() const { return core.centroids.rows; }
    void fit(const mlcore::Matrix &X, unsigned threads) { core.fit(X, threads); }
    int predict_one(const double *x) const { return core.predict_one(x); }
    void probabilities(const double *x, double *out) const { core.probabilities(x, out); }
};

ZEND_BEGIN_ARG_INFO_EX(arginfo_construct, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, clusters, IS_LONG, 0, "8")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, maxIterations, IS_LONG, 0, "300")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, tolerance, IS_DOUBLE, 0, "1.0E-4")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, seed, IS_LONG, 1, "null")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, seeder, IS_STRING, 0, "\"plusplus\"")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, centroids, IS_ARRAY, 1, "null")
ZEND_END_ARG_INFO()

void ZEND_FASTCALL kmeans_construct(INTERNAL_FUNCTION_PARAMETERS) {
    zend_long clusters = 8, max_iterations = 300, seed = 0;
    double tolerance = 1e-4;
    bool seed_null = true;
    zend_string *seeder = nullptr;
    HashTable *centroids = nullptr;
    ZEND_PARSE_PARAMETERS_START(0, 6)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(clusters)
        Z_PARAM_LONG(max_iterations)
        Z_PARAM_DOUBLE(tolerance)
        Z_PARAM_LONG_OR_NULL(seed, seed_null)
        Z_PARAM_STR(seeder)
        Z_PARAM_ARRAY_HT_OR_NULL(centroids)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        Model fresh;
        mlcore::KMeans &c = fresh.core;
        c.clusters = positive(clusters, "clusters");
        c.max_iterations = static_cast<long>(positive(max_iterations, "maxIterations"));
        c.tolerance = non_negative(tolerance, "tolerance");
        parse_seed(seed_null, seed, c.has_seed, c.seed);
        if (seeder) {
            c.seeder = mlcore::parse_seeder(zstr(seeder));
        }
        if (centroids) {
            c.preset = read_samples(centroids);
            check(c.preset.rows == c.clusters, "centroids must contain exactly one sample per cluster");
            c.seeder = mlcore::Seeder::Preset;
        } else {
            check(c.seeder != mlcore::Seeder::Preset, "The preset seeder needs the centroids parameter");
        }
        reset(ZEND_THIS, std::move(fresh));
    });
}

void ZEND_FASTCALL kmeans_centroids(INTERNAL_FUNCTION_PARAMETERS) {
    ZEND_PARSE_PARAMETERS_NONE();
    fitted_getter<Model>(ZEND_THIS, return_value, [](const Model &m, zval *out) { write_matrix(out, m.core.centroids); });
}

void ZEND_FASTCALL kmeans_labels(INTERNAL_FUNCTION_PARAMETERS) {
    ZEND_PARSE_PARAMETERS_NONE();
    fitted_getter<Model>(ZEND_THIS, return_value, [](const Model &m, zval *out) { write_ints(out, m.core.labels); });
}

void ZEND_FASTCALL kmeans_inertia(INTERNAL_FUNCTION_PARAMETERS) {
    ZEND_PARSE_PARAMETERS_NONE();
    fitted_getter<Model>(ZEND_THIS, return_value, [](const Model &m, zval *out) { ZVAL_DOUBLE(out, m.core.inertia); });
}

void ZEND_FASTCALL kmeans_iterations(INTERNAL_FUNCTION_PARAMETERS) {
    ZEND_PARSE_PARAMETERS_NONE();
    fitted_getter<Model>(ZEND_THIS, return_value, [](const Model &m, zval *out) { ZVAL_LONG(out, m.core.iterations); });
}

void ZEND_FASTCALL kmeans_serialize(INTERNAL_FUNCTION_PARAMETERS) {
    ZEND_PARSE_PARAMETERS_NONE();
    const mlcore::KMeans &m = native<Model>(ZEND_THIS).core;
    array_init(return_value);
    add_assoc_long(return_value, "clusters", static_cast<zend_long>(m.clusters));
    add_assoc_long(return_value, "maxIterations", m.max_iterations);
    add_assoc_double(return_value, "tolerance", m.tolerance);
    if (m.has_seed) {
        add_assoc_long(return_value, "seed", static_cast<zend_long>(m.seed));
    } else {
        add_assoc_null(return_value, "seed");
    }
    add_assoc_long(return_value, "features", static_cast<zend_long>(m.centroids.cols));
    put_doubles(return_value, "centroids", m.centroids.data);
    zval labels;
    write_ints(&labels, m.labels);
    add_assoc_zval(return_value, "labels", &labels);
    add_assoc_double(return_value, "inertia", m.inertia);
    add_assoc_long(return_value, "iterations", m.iterations);
    const std::string seeder = mlcore::seeder_name(m.seeder);
    add_assoc_stringl(return_value, "seeder", seeder.data(), seeder.size());
    put_doubles(return_value, "presetCentroids", m.preset.data);
}

void ZEND_FASTCALL kmeans_unserialize(INTERNAL_FUNCTION_PARAMETERS) {
    HashTable *data;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY_HT(data)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        Model fresh;
        mlcore::KMeans &restored = fresh.core;
        const zend_long clusters = get_long(data, "clusters");
        check(clusters >= 1, "Invalid serialized data: clusters must be at least 1");
        restored.clusters = static_cast<std::size_t>(clusters);
        restored.max_iterations = static_cast<long>(get_long(data, "maxIterations"));
        restored.tolerance = get_double(data, "tolerance");
        zval *seed = require_key(data, "seed");
        if (Z_TYPE_P(seed) == IS_LONG) {
            restored.has_seed = true;
            restored.seed = static_cast<std::uint64_t>(Z_LVAL_P(seed));
        } else if (Z_TYPE_P(seed) != IS_NULL) {
            throw std::invalid_argument("Invalid serialized data: \"seed\" must be an int or null");
        }

        const zend_long features = get_long(data, "features");
        std::vector<double> centroids = get_doubles(data, "centroids");
        const std::vector<double> raw_labels = read_numbers(get_array(data, "labels"), "labels");
        if (!centroids.empty()) {
            if (features < 1 || !shape_matches(centroids.size(), restored.clusters, static_cast<std::size_t>(features))) {
                throw std::invalid_argument("Invalid serialized data: inconsistent KMeans state");
            }
            restored.labels.reserve(raw_labels.size());
            for (double label : raw_labels) {
                check(label >= 0.0 && label < static_cast<double>(restored.clusters),
                      "Invalid serialized data: cluster index out of range");
                restored.labels.push_back(static_cast<int>(label));
            }
            restored.centroids.rows = restored.clusters;
            restored.centroids.cols = static_cast<std::size_t>(features);
            restored.centroids.data = std::move(centroids);
            restored.inertia = get_double(data, "inertia");
            restored.iterations = static_cast<long>(get_long(data, "iterations"));
        }
        // Keys added in 0.1.2; older payloads default to k-means++.
        if (zend_hash_str_exists(data, "seeder", sizeof("seeder") - 1)) {
            zval *seeder = require_key(data, "seeder");
            check(Z_TYPE_P(seeder) == IS_STRING, "Invalid serialized data: \"seeder\" must be a string");
            restored.seeder = mlcore::parse_seeder(std::string(Z_STRVAL_P(seeder), Z_STRLEN_P(seeder)));
            std::vector<double> preset = get_doubles(data, "presetCentroids");
            if (!preset.empty()) {
                check(shape_matches(preset.size(), restored.clusters, preset.size() / restored.clusters),
                      "Invalid serialized data: inconsistent KMeans preset");
                restored.preset = mlcore::Matrix(restored.clusters, preset.size() / restored.clusters);
                restored.preset.data = std::move(preset);
            }
            check(restored.seeder != mlcore::Seeder::Preset || restored.preset.rows > 0,
                  "Invalid serialized data: preset seeder without centroids");
        }
        reset(ZEND_THIS, std::move(fresh));
    });
}

const zend_function_entry methods[] = {
    ML_METHOD(__construct, kmeans_construct, arginfo_construct)
    ML_METHOD(fit, fit_unsupervised<Model>, arginfo_ml_fit_unsupervised)
    ML_METHOD(predict, predict_cluster<Model>, arginfo_ml_predict_int)
    ML_METHOD(predictProbability, predict_memberships<Model>, arginfo_ml_samples_to_array)
    ML_METHOD(getCentroids, kmeans_centroids, arginfo_ml_get_array)
    ML_METHOD(getLabels, kmeans_labels, arginfo_ml_get_array)
    ML_METHOD(getInertia, kmeans_inertia, arginfo_ml_get_float)
    ML_METHOD(getIterations, kmeans_iterations, arginfo_ml_get_int)
    ML_METHOD(__serialize, kmeans_serialize, arginfo_ml_get_array)
    ML_METHOD(__unserialize, kmeans_unserialize, arginfo_ml_unserialize)
    PHP_FE_END};

}  // namespace

void mlphp::register_kmeans() {
    register_class<Model>("ML\\Clustering\\KMeans", methods, {ce_learner, ce_clusterer, ce_probabilistic});
}
