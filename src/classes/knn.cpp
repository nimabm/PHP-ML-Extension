#include "arginfo.hpp"
#include "core/knn.hpp"

namespace {

using mlphp::guard;

struct Model {
    mlcore::KNearestNeighbors core;
    mlphp::LabelEncoder labels;

    bool fitted() const { return core.fitted(); }
    std::size_t features() const { return core.features(); }
    std::size_t classes() const { return core.classes; }
    int predict_one(const double *x) const { return core.predict_one(x); }
};

const char *metric_name(mlcore::DistanceMetric metric) {
    return metric == mlcore::DistanceMetric::Euclidean ? "euclidean" : "manhattan";
}

mlcore::DistanceMetric parse_metric(const char *name, std::size_t len) {
    const std::string value(name, len);
    if (value == "euclidean") {
        return mlcore::DistanceMetric::Euclidean;
    }
    if (value == "manhattan") {
        return mlcore::DistanceMetric::Manhattan;
    }
    throw std::invalid_argument("metric must be \"euclidean\" or \"manhattan\", \"" + value + "\" given");
}

ZEND_BEGIN_ARG_INFO_EX(arginfo_construct, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, k, IS_LONG, 0, "3")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, metric, IS_STRING, 0, "\"euclidean\"")
ZEND_END_ARG_INFO()

PHP_METHOD(KNearestNeighbors, __construct) {
    zend_long k = 3;
    zend_string *metric = nullptr;
    ZEND_PARSE_PARAMETERS_START(0, 2)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(k)
        Z_PARAM_STR(metric)
    ZEND_PARSE_PARAMETERS_END();

    guard([&] {
        if (k < 1) {
            throw std::invalid_argument("k must be at least 1");
        }
        Model fresh;
        fresh.core.k = static_cast<std::size_t>(k);
        if (metric) {
            fresh.core.metric = parse_metric(ZSTR_VAL(metric), ZSTR_LEN(metric));
        }
        mlphp::native<Model>(ZEND_THIS) = std::move(fresh);
    });
}

PHP_METHOD(KNearestNeighbors, fit) {
    HashTable *samples, *labels;
    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_ARRAY_HT(samples)
        Z_PARAM_ARRAY_HT(labels)
    ZEND_PARSE_PARAMETERS_END();

    if (!guard([&] {
            Model &m = mlphp::native<Model>(ZEND_THIS);
            const mlcore::Matrix X = mlphp::read_samples(samples);
            mlphp::LabelEncoder encoder;
            const std::vector<int> y = encoder.fit(labels);
            m.core.fit(X, y, encoder.size());
            m.labels = std::move(encoder);
        })) {
        RETURN_THROWS();
    }
    RETURN_OBJ_COPY(Z_OBJ_P(ZEND_THIS));
}

PHP_METHOD(KNearestNeighbors, predict) {
    HashTable *samples;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY_HT(samples)
    ZEND_PARSE_PARAMETERS_END();

    guard([&] { mlphp::predict_labels(mlphp::native<Model>(ZEND_THIS), ZEND_THIS, samples, return_value); });
}

PHP_METHOD(KNearestNeighbors, getClasses) {
    ZEND_PARSE_PARAMETERS_NONE();
    guard([&] {
        const Model &m = mlphp::native<Model>(ZEND_THIS);
        mlphp::require_fitted(m.fitted(), ZEND_THIS);
        m.labels.write_all(return_value);
    });
}

PHP_METHOD(KNearestNeighbors, __serialize) {
    ZEND_PARSE_PARAMETERS_NONE();
    const Model &m = mlphp::native<Model>(ZEND_THIS);
    array_init(return_value);
    add_assoc_long(return_value, "k", static_cast<zend_long>(m.core.k));
    add_assoc_string(return_value, "metric", metric_name(m.core.metric));
    zval classes, labels;
    m.labels.write_all(&classes);
    add_assoc_zval(return_value, "classes", &classes);
    add_assoc_long(return_value, "features", static_cast<zend_long>(m.core.samples.cols));
    mlphp::put_doubles(return_value, "samples", m.core.samples.data);
    mlphp::write_ints(&labels, m.core.labels);
    add_assoc_zval(return_value, "labels", &labels);
}

PHP_METHOD(KNearestNeighbors, __unserialize) {
    HashTable *data;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY_HT(data)
    ZEND_PARSE_PARAMETERS_END();

    guard([&] {
        Model restored;
        const zend_long k = mlphp::get_long(data, "k");
        if (k < 1) {
            throw std::invalid_argument("Invalid serialized data: k must be at least 1");
        }
        restored.core.k = static_cast<std::size_t>(k);
        zval *metric = mlphp::require_key(data, "metric");
        if (Z_TYPE_P(metric) != IS_STRING) {
            throw std::invalid_argument("Invalid serialized data: \"metric\" must be a string");
        }
        restored.core.metric = parse_metric(Z_STRVAL_P(metric), Z_STRLEN_P(metric));
        restored.labels.read_all(mlphp::get_array(data, "classes"));

        const zend_long features = mlphp::get_long(data, "features");
        std::vector<double> flat = mlphp::get_doubles(data, "samples");
        const std::vector<double> raw_labels = mlphp::read_numbers(mlphp::get_array(data, "labels"), "labels");
        const std::size_t K = restored.labels.size();

        if (!flat.empty()) {
            if (features < 1 || K == 0 || flat.size() % static_cast<std::size_t>(features) != 0) {
                throw std::invalid_argument("Invalid serialized data: inconsistent KNearestNeighbors state");
            }
            const std::size_t rows = flat.size() / static_cast<std::size_t>(features);
            if (raw_labels.size() != rows || restored.core.k > rows) {
                throw std::invalid_argument("Invalid serialized data: inconsistent KNearestNeighbors state");
            }
            std::vector<int> y(rows);
            for (std::size_t i = 0; i < rows; ++i) {
                const double label = raw_labels[i];
                if (!(label >= 0.0 && label < static_cast<double>(K)) || label != static_cast<double>(static_cast<int>(label))) {
                    throw std::invalid_argument("Invalid serialized data: label index out of range");
                }
                y[i] = static_cast<int>(label);
            }
            restored.core.samples.rows = rows;
            restored.core.samples.cols = static_cast<std::size_t>(features);
            restored.core.samples.data = std::move(flat);
            restored.core.labels = std::move(y);
            restored.core.classes = K;
        }
        mlphp::native<Model>(ZEND_THIS) = std::move(restored);
    });
}

const zend_function_entry methods[] = {
    PHP_ME(KNearestNeighbors, __construct, arginfo_construct, ZEND_ACC_PUBLIC)
    PHP_ME(KNearestNeighbors, fit, arginfo_ml_fit_supervised, ZEND_ACC_PUBLIC)
    PHP_ME(KNearestNeighbors, predict, arginfo_ml_predict_label, ZEND_ACC_PUBLIC)
    PHP_ME(KNearestNeighbors, getClasses, arginfo_ml_get_array, ZEND_ACC_PUBLIC)
    PHP_ME(KNearestNeighbors, __serialize, arginfo_ml_get_array, ZEND_ACC_PUBLIC)
    PHP_ME(KNearestNeighbors, __unserialize, arginfo_ml_unserialize, ZEND_ACC_PUBLIC)
    PHP_FE_END
};

}  // namespace

void mlphp::register_knn() {
    register_class<Model>("ML\\Classification\\KNearestNeighbors", methods);
}
