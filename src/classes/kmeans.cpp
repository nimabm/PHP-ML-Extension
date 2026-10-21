#include "arginfo.hpp"
#include "core/kmeans.hpp"

namespace {

using mlphp::guard;
using Model = mlcore::KMeans;

ZEND_BEGIN_ARG_INFO_EX(arginfo_construct, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, clusters, IS_LONG, 0, "8")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, maxIterations, IS_LONG, 0, "300")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, tolerance, IS_DOUBLE, 0, "1.0E-4")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, seed, IS_LONG, 1, "null")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_MASK_EX(arginfo_predict, 0, 1, MAY_BE_ARRAY | MAY_BE_LONG)
    ZEND_ARG_TYPE_INFO(0, samples, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

PHP_METHOD(KMeans, __construct) {
    zend_long clusters = 8;
    zend_long max_iterations = 300;
    double tolerance = 1e-4;
    zend_long seed = 0;
    bool seed_is_null = true;
    ZEND_PARSE_PARAMETERS_START(0, 4)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(clusters)
        Z_PARAM_LONG(max_iterations)
        Z_PARAM_DOUBLE(tolerance)
        Z_PARAM_LONG_OR_NULL(seed, seed_is_null)
    ZEND_PARSE_PARAMETERS_END();

    guard([&] {
        if (clusters < 1) {
            throw std::invalid_argument("clusters must be at least 1");
        }
        if (max_iterations < 1) {
            throw std::invalid_argument("maxIterations must be at least 1");
        }
        if (!(tolerance >= 0.0)) {
            throw std::invalid_argument("tolerance must be greater than or equal to 0");
        }
        Model fresh;
        fresh.clusters = static_cast<std::size_t>(clusters);
        fresh.max_iterations = static_cast<long>(max_iterations);
        fresh.tolerance = tolerance;
        fresh.has_seed = !seed_is_null;
        fresh.seed = static_cast<std::uint64_t>(seed);
        mlphp::native<Model>(ZEND_THIS) = std::move(fresh);
    });
}

PHP_METHOD(KMeans, fit) {
    HashTable *samples;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY_HT(samples)
    ZEND_PARSE_PARAMETERS_END();

    if (!guard([&] { mlphp::native<Model>(ZEND_THIS).fit(mlphp::read_samples(samples)); })) {
        RETURN_THROWS();
    }
    RETURN_OBJ_COPY(Z_OBJ_P(ZEND_THIS));
}

PHP_METHOD(KMeans, predict) {
    HashTable *samples;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY_HT(samples)
    ZEND_PARSE_PARAMETERS_END();

    guard([&] {
        const Model &m = mlphp::native<Model>(ZEND_THIS);
        mlphp::require_fitted(m.fitted(), ZEND_THIS);
        if (mlphp::is_sample_list(samples)) {
            const mlcore::Matrix X = mlphp::read_samples(samples, m.features());
            std::vector<int> clusters(X.rows);
            for (std::size_t r = 0; r < X.rows; ++r) {
                clusters[r] = m.predict_one(X.row(r));
            }
            mlphp::write_ints(return_value, clusters);
        } else {
            const std::vector<double> x = mlphp::read_sample(samples, m.features());
            RETVAL_LONG(m.predict_one(x.data()));
        }
    });
}

PHP_METHOD(KMeans, getCentroids) {
    ZEND_PARSE_PARAMETERS_NONE();
    guard([&] {
        const Model &m = mlphp::native<Model>(ZEND_THIS);
        mlphp::require_fitted(m.fitted(), ZEND_THIS);
        mlphp::write_matrix(return_value, m.centroids);
    });
}

PHP_METHOD(KMeans, getLabels) {
    ZEND_PARSE_PARAMETERS_NONE();
    guard([&] {
        const Model &m = mlphp::native<Model>(ZEND_THIS);
        mlphp::require_fitted(m.fitted(), ZEND_THIS);
        mlphp::write_ints(return_value, m.labels);
    });
}

PHP_METHOD(KMeans, getInertia) {
    ZEND_PARSE_PARAMETERS_NONE();
    guard([&] {
        const Model &m = mlphp::native<Model>(ZEND_THIS);
        mlphp::require_fitted(m.fitted(), ZEND_THIS);
        RETVAL_DOUBLE(m.inertia);
    });
}

PHP_METHOD(KMeans, getIterations) {
    ZEND_PARSE_PARAMETERS_NONE();
    guard([&] {
        const Model &m = mlphp::native<Model>(ZEND_THIS);
        mlphp::require_fitted(m.fitted(), ZEND_THIS);
        RETVAL_LONG(m.iterations);
    });
}

PHP_METHOD(KMeans, __serialize) {
    ZEND_PARSE_PARAMETERS_NONE();
    const Model &m = mlphp::native<Model>(ZEND_THIS);
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
    mlphp::put_doubles(return_value, "centroids", m.centroids.data);
    zval labels;
    mlphp::write_ints(&labels, m.labels);
    add_assoc_zval(return_value, "labels", &labels);
    add_assoc_double(return_value, "inertia", m.inertia);
    add_assoc_long(return_value, "iterations", m.iterations);
}

PHP_METHOD(KMeans, __unserialize) {
    HashTable *data;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY_HT(data)
    ZEND_PARSE_PARAMETERS_END();

    guard([&] {
        Model restored;
        const zend_long clusters = mlphp::get_long(data, "clusters");
        if (clusters < 1) {
            throw std::invalid_argument("Invalid serialized data: clusters must be at least 1");
        }
        restored.clusters = static_cast<std::size_t>(clusters);
        restored.max_iterations = static_cast<long>(mlphp::get_long(data, "maxIterations"));
        restored.tolerance = mlphp::get_double(data, "tolerance");
        zval *seed = mlphp::require_key(data, "seed");
        if (Z_TYPE_P(seed) == IS_LONG) {
            restored.has_seed = true;
            restored.seed = static_cast<std::uint64_t>(Z_LVAL_P(seed));
        } else if (Z_TYPE_P(seed) != IS_NULL) {
            throw std::invalid_argument("Invalid serialized data: \"seed\" must be an int or null");
        }

        const zend_long features = mlphp::get_long(data, "features");
        std::vector<double> centroids = mlphp::get_doubles(data, "centroids");
        const std::vector<double> raw_labels = mlphp::read_numbers(mlphp::get_array(data, "labels"), "labels");
        if (!centroids.empty()) {
            if (features < 1 || !mlphp::shape_matches(centroids.size(), restored.clusters, static_cast<std::size_t>(features))) {
                throw std::invalid_argument("Invalid serialized data: inconsistent KMeans state");
            }
            restored.labels.reserve(raw_labels.size());
            for (double label : raw_labels) {
                if (!(label >= 0.0 && label < static_cast<double>(restored.clusters))) {
                    throw std::invalid_argument("Invalid serialized data: cluster index out of range");
                }
                restored.labels.push_back(static_cast<int>(label));
            }
            restored.centroids.rows = restored.clusters;
            restored.centroids.cols = static_cast<std::size_t>(features);
            restored.centroids.data = std::move(centroids);
            restored.inertia = mlphp::get_double(data, "inertia");
            restored.iterations = static_cast<long>(mlphp::get_long(data, "iterations"));
        }
        mlphp::native<Model>(ZEND_THIS) = std::move(restored);
    });
}

const zend_function_entry methods[] = {
    PHP_ME(KMeans, __construct, arginfo_construct, ZEND_ACC_PUBLIC)
    PHP_ME(KMeans, fit, arginfo_ml_fit_unsupervised, ZEND_ACC_PUBLIC)
    PHP_ME(KMeans, predict, arginfo_predict, ZEND_ACC_PUBLIC)
    PHP_ME(KMeans, getCentroids, arginfo_ml_get_array, ZEND_ACC_PUBLIC)
    PHP_ME(KMeans, getLabels, arginfo_ml_get_array, ZEND_ACC_PUBLIC)
    PHP_ME(KMeans, getInertia, arginfo_ml_get_float, ZEND_ACC_PUBLIC)
    PHP_ME(KMeans, getIterations, arginfo_ml_get_int, ZEND_ACC_PUBLIC)
    PHP_ME(KMeans, __serialize, arginfo_ml_get_array, ZEND_ACC_PUBLIC)
    PHP_ME(KMeans, __unserialize, arginfo_ml_unserialize, ZEND_ACC_PUBLIC)
    PHP_FE_END
};

}  // namespace

void mlphp::register_kmeans() {
    register_class<Model>("ML\\Clustering\\KMeans", methods);
}
