// KNearestNeighbors, KDNeighbors, RadiusNeighbors (classifiers) and
// KNNRegressor, KDNeighborsRegressor, RadiusNeighborsRegressor.

#include "binding.hpp"
#include "core/neighbors.hpp"

namespace {

using namespace mlphp;
using mlcore::ArchiveReader;
using mlcore::ArchiveWriter;
using mlcore::Distance;
using mlcore::Neighbor;
using mlcore::NeighborIndex;

enum class Mode { KNearest, KDTree, Radius };

// Labels plus an optional "outlier" label written for class index -1.
struct OutlierLabels : LabelEncoder {
    Label outlier{std::in_place_index<1>, "?"};

    OutlierLabels &operator=(LabelEncoder &&other) {
        static_cast<LabelEncoder &>(*this) = std::move(other);
        return *this;
    }
    void write(int index, zval *out) const {
        if (index < 0) {
            write_label(outlier, out);
        } else {
            LabelEncoder::write(index, out);
        }
    }
};

template <Mode TMode>
struct NeighborClassifier {
    static constexpr std::uint32_t kTag =
        TMode == Mode::KNearest ? mlcore::tag("KNNC") : TMode == Mode::KDTree ? mlcore::tag("KDNC") : mlcore::tag("RNNC");
    std::size_t k = TMode == Mode::KNearest ? 3 : 5;
    double radius = 1.0;
    bool weighted = false;
    NeighborIndex index;
    std::vector<int> targets;
    std::size_t n_classes = 0;
    OutlierLabels labels;

    NeighborClassifier() { index.use_tree = TMode == Mode::KDTree || TMode == Mode::Radius; }
    bool fitted() const { return index.samples.rows > 0; }
    std::size_t features() const { return index.samples.cols; }
    std::size_t classes() const { return n_classes; }

    void fit(const mlcore::Matrix &X, const std::vector<int> &y, std::size_t K, unsigned) {
        if (TMode != Mode::Radius && k > X.rows) {
            throw std::invalid_argument("k (" + std::to_string(k) + ") cannot exceed the number of training samples (" +
                                        std::to_string(X.rows) + ")");
        }
        index.use_tree = TMode == Mode::KDTree || (TMode == Mode::Radius && index.metric.coordinate_bounded());
        index.build(X);
        targets = y;
        n_classes = K;
    }
    void query(const double *x, std::vector<Neighbor> &nb) const {
        if (TMode == Mode::Radius) {
            index.within(x, radius, nb);
        } else {
            index.nearest(x, k, nb);
        }
    }
    int predict_one(const double *x) const {
        std::vector<Neighbor> nb;
        query(x, nb);
        return nb.empty() ? -1 : mlcore::vote(nb, targets, n_classes, weighted);
    }
    void probabilities(const double *x, double *out) const {
        std::vector<Neighbor> nb;
        query(x, nb);
        mlcore::vote_distribution(nb, targets, n_classes, weighted, out);
    }

    void save(ArchiveWriter &w) const {
        w.size(k);
        w.f64(radius);
        w.boolean(weighted);
        save_label(w, labels.outlier);
        index.save(w);
        w.ints(targets);
        w.size(n_classes);
        labels.save(w);
    }
    void load(ArchiveReader &r) {
        k = r.size();
        radius = r.f64();
        weighted = r.boolean();
        labels.outlier = load_label(r);
        index.load(r);
        targets = r.ints();
        n_classes = r.size(1u << 20);
        for (int t : targets) {
            ArchiveReader::require(t >= 0 && static_cast<std::size_t>(t) < n_classes, "label index out of range");
        }
        labels.load(r);
        ArchiveReader::require(targets.size() == index.samples.rows && labels.size() == n_classes);
        ArchiveReader::require(TMode == Mode::Radius || index.samples.rows == 0 || k <= index.samples.rows);
    }
};

template <Mode TMode>
struct NeighborRegressor {
    static constexpr std::uint32_t kTag =
        TMode == Mode::KNearest ? mlcore::tag("KNNR") : TMode == Mode::KDTree ? mlcore::tag("KDNR") : mlcore::tag("RNNR");
    std::size_t k = 5;
    double radius = 1.0;
    bool weighted = false;
    NeighborIndex index;
    std::vector<double> targets;

    NeighborRegressor() { index.use_tree = TMode != Mode::KNearest; }
    bool fitted() const { return index.samples.rows > 0; }
    std::size_t features() const { return index.samples.cols; }

    void fit(const mlcore::Matrix &X, const std::vector<double> &y, unsigned) {
        if (TMode != Mode::Radius && k > X.rows) {
            throw std::invalid_argument("k (" + std::to_string(k) + ") cannot exceed the number of training samples (" +
                                        std::to_string(X.rows) + ")");
        }
        index.use_tree = TMode == Mode::KDTree || (TMode == Mode::Radius && index.metric.coordinate_bounded());
        index.build(X);
        targets = y;
    }
    double predict_one(const double *x) const {
        std::vector<Neighbor> nb;
        if (TMode == Mode::Radius) {
            index.within(x, radius, nb);
        } else {
            index.nearest(x, k, nb);
        }
        return mlcore::neighbor_mean(nb, targets, weighted);
    }

    void save(ArchiveWriter &w) const {
        w.size(k);
        w.f64(radius);
        w.boolean(weighted);
        index.save(w);
        w.doubles(targets);
    }
    void load(ArchiveReader &r) {
        k = r.size();
        radius = r.f64();
        weighted = r.boolean();
        index.load(r);
        targets = r.doubles(index.samples.rows);
        ArchiveReader::require(TMode == Mode::Radius || index.samples.rows == 0 || k <= index.samples.rows);
    }
};

using KNNClassifier = NeighborClassifier<Mode::KNearest>;
using KDNeighbors = NeighborClassifier<Mode::KDTree>;
using RadiusNeighbors = NeighborClassifier<Mode::Radius>;
using KNNRegressor = NeighborRegressor<Mode::KNearest>;
using KDNeighborsRegressor = NeighborRegressor<Mode::KDTree>;
using RadiusNeighborsRegressor = NeighborRegressor<Mode::Radius>;

Distance parse_metric(zend_string *metric) {
    return metric ? Distance::parse(zstr(metric)) : Distance{};
}

// ---- KNearestNeighbors keeps its 0.1.0 constructor and array serialization ------

ZEND_BEGIN_ARG_INFO_EX(arginfo_knn_construct, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, k, IS_LONG, 0, "3")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, metric, IS_STRING, 0, "\"euclidean\"")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, weighted, _IS_BOOL, 0, "false")
ZEND_END_ARG_INFO()

void ZEND_FASTCALL knn_construct(INTERNAL_FUNCTION_PARAMETERS) {
    zend_long k = 3;
    zend_string *metric = nullptr;
    bool weighted = false;
    ZEND_PARSE_PARAMETERS_START(0, 3)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(k)
        Z_PARAM_STR(metric)
        Z_PARAM_BOOL(weighted)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        KNNClassifier fresh;
        fresh.k = positive(k, "k");
        fresh.index.metric = parse_metric(metric);
        fresh.weighted = weighted;
        reset(ZEND_THIS, std::move(fresh));
    });
}

void ZEND_FASTCALL knn_serialize(INTERNAL_FUNCTION_PARAMETERS) {
    ZEND_PARSE_PARAMETERS_NONE();
    const KNNClassifier &m = native<KNNClassifier>(ZEND_THIS);
    array_init(return_value);
    add_assoc_long(return_value, "k", static_cast<zend_long>(m.k));
    const std::string metric = m.index.metric.spec();
    add_assoc_stringl(return_value, "metric", metric.data(), metric.size());
    add_assoc_bool(return_value, "weighted", m.weighted);
    zval classes, labels;
    m.labels.write_all(&classes);
    add_assoc_zval(return_value, "classes", &classes);
    add_assoc_long(return_value, "features", static_cast<zend_long>(m.index.samples.cols));
    put_doubles(return_value, "samples", m.index.samples.data);
    write_ints(&labels, m.targets);
    add_assoc_zval(return_value, "labels", &labels);
}

void ZEND_FASTCALL knn_unserialize(INTERNAL_FUNCTION_PARAMETERS) {
    HashTable *data;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY_HT(data)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        KNNClassifier restored;
        const zend_long k = get_long(data, "k");
        check(k >= 1, "Invalid serialized data: k must be at least 1");
        restored.k = static_cast<std::size_t>(k);
        zval *metric = require_key(data, "metric");
        check(Z_TYPE_P(metric) == IS_STRING, "Invalid serialized data: \"metric\" must be a string");
        restored.index.metric = Distance::parse(std::string(Z_STRVAL_P(metric), Z_STRLEN_P(metric)));
        if (zval *weighted = zend_hash_str_find_deref(data, "weighted", sizeof("weighted") - 1)) {
            restored.weighted = zend_is_true(weighted);
        }
        LabelEncoder classes;
        classes.read_all(get_array(data, "classes"));
        const zend_long features = get_long(data, "features");
        std::vector<double> flat = get_doubles(data, "samples");
        const std::vector<double> raw_labels = read_numbers(get_array(data, "labels"), "labels");
        const std::size_t K = classes.size();
        if (!flat.empty()) {
            check(features >= 1 && K > 0 && flat.size() % static_cast<std::size_t>(features) == 0,
                  "Invalid serialized data: inconsistent KNearestNeighbors state");
            const std::size_t rows = flat.size() / static_cast<std::size_t>(features);
            check(raw_labels.size() == rows && restored.k <= rows,
                  "Invalid serialized data: inconsistent KNearestNeighbors state");
            std::vector<int> y(rows);
            for (std::size_t i = 0; i < rows; ++i) {
                const double label = raw_labels[i];
                check(label >= 0.0 && label < static_cast<double>(K) && label == static_cast<double>(static_cast<int>(label)),
                      "Invalid serialized data: label index out of range");
                y[i] = static_cast<int>(label);
            }
            mlcore::Matrix X(rows, static_cast<std::size_t>(features));
            X.data = std::move(flat);
            restored.index.build(X);
            restored.targets = std::move(y);
            restored.n_classes = K;
            restored.labels = std::move(classes);
        }
        reset(ZEND_THIS, std::move(restored));
    });
}

// ---- constructors of the new classes ---------------------------------------------

ZEND_BEGIN_ARG_INFO_EX(arginfo_kd_construct, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, k, IS_LONG, 0, "5")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, weighted, _IS_BOOL, 0, "false")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, maxLeafSize, IS_LONG, 0, "30")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, metric, IS_STRING, 0, "\"euclidean\"")
ZEND_END_ARG_INFO()

template <typename M>
void ZEND_FASTCALL kd_construct(INTERNAL_FUNCTION_PARAMETERS) {
    zend_long k = 5, leaf = 30;
    bool weighted = false;
    zend_string *metric = nullptr;
    ZEND_PARSE_PARAMETERS_START(0, 4)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(k)
        Z_PARAM_BOOL(weighted)
        Z_PARAM_LONG(leaf)
        Z_PARAM_STR(metric)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        M fresh;
        fresh.k = positive(k, "k");
        fresh.weighted = weighted;
        fresh.index.leaf_size = positive(leaf, "maxLeafSize");
        fresh.index.metric = parse_metric(metric);
        check(fresh.index.metric.coordinate_bounded(),
              "K-d tree models only support euclidean, squared_euclidean, manhattan, chebyshev and minkowski distances");
        reset(ZEND_THIS, std::move(fresh));
    });
}

ZEND_BEGIN_ARG_INFO_EX(arginfo_radius_construct, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, radius, IS_DOUBLE, 0, "1.0")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, weighted, _IS_BOOL, 0, "false")
    ZEND_ARG_TYPE_MASK(0, outlierClass, MAY_BE_LONG | MAY_BE_STRING, "\"?\"")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, metric, IS_STRING, 0, "\"euclidean\"")
ZEND_END_ARG_INFO()

void ZEND_FASTCALL radius_construct(INTERNAL_FUNCTION_PARAMETERS) {
    double radius = 1.0;
    bool weighted = false;
    zval *outlier = nullptr;
    zend_string *metric = nullptr;
    ZEND_PARSE_PARAMETERS_START(0, 4)
        Z_PARAM_OPTIONAL
        Z_PARAM_DOUBLE(radius)
        Z_PARAM_BOOL(weighted)
        Z_PARAM_ZVAL(outlier)
        Z_PARAM_STR(metric)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        RadiusNeighbors fresh;
        fresh.radius = strictly_positive(radius, "radius");
        fresh.weighted = weighted;
        if (outlier) {
            fresh.labels.outlier = read_label(outlier, 0);
        }
        fresh.index.metric = parse_metric(metric);
        reset(ZEND_THIS, std::move(fresh));
    });
}

ZEND_BEGIN_ARG_INFO_EX(arginfo_knn_regressor_construct, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, k, IS_LONG, 0, "5")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, weighted, _IS_BOOL, 0, "false")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, metric, IS_STRING, 0, "\"euclidean\"")
ZEND_END_ARG_INFO()

void ZEND_FASTCALL knn_regressor_construct(INTERNAL_FUNCTION_PARAMETERS) {
    zend_long k = 5;
    bool weighted = false;
    zend_string *metric = nullptr;
    ZEND_PARSE_PARAMETERS_START(0, 3)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(k)
        Z_PARAM_BOOL(weighted)
        Z_PARAM_STR(metric)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        KNNRegressor fresh;
        fresh.k = positive(k, "k");
        fresh.weighted = weighted;
        fresh.index.metric = parse_metric(metric);
        reset(ZEND_THIS, std::move(fresh));
    });
}

ZEND_BEGIN_ARG_INFO_EX(arginfo_radius_regressor_construct, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, radius, IS_DOUBLE, 0, "1.0")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, weighted, _IS_BOOL, 0, "false")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, metric, IS_STRING, 0, "\"euclidean\"")
ZEND_END_ARG_INFO()

void ZEND_FASTCALL radius_regressor_construct(INTERNAL_FUNCTION_PARAMETERS) {
    double radius = 1.0;
    bool weighted = false;
    zend_string *metric = nullptr;
    ZEND_PARSE_PARAMETERS_START(0, 3)
        Z_PARAM_OPTIONAL
        Z_PARAM_DOUBLE(radius)
        Z_PARAM_BOOL(weighted)
        Z_PARAM_STR(metric)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        RadiusNeighborsRegressor fresh;
        fresh.radius = strictly_positive(radius, "radius");
        fresh.weighted = weighted;
        fresh.index.metric = parse_metric(metric);
        reset(ZEND_THIS, std::move(fresh));
    });
}

#define ML_NEIGHBOR_CLASSIFIER(M)                                                  \
    ML_METHOD(fit, fit_classifier<M>, arginfo_ml_fit_supervised)                    \
    ML_METHOD(predict, predict_classifier<M>, arginfo_ml_predict_label)             \
    ML_METHOD(predictProbability, predict_proba<M>, arginfo_ml_samples_to_array)    \
    ML_METHOD(getClasses, get_classes<M>, arginfo_ml_get_array)

#define ML_NEIGHBOR_REGRESSOR(M)                                                   \
    ML_METHOD(fit, fit_regressor<M>, arginfo_ml_fit_supervised)                     \
    ML_METHOD(predict, predict_regressor<M>, arginfo_ml_predict_float)              \
    ML_METHOD(__serialize, serialize_native<M>, arginfo_ml_get_array)               \
    ML_METHOD(__unserialize, unserialize_native<M>, arginfo_ml_unserialize)

const zend_function_entry knn_methods[] = {
    ML_METHOD(__construct, knn_construct, arginfo_knn_construct)
    ML_NEIGHBOR_CLASSIFIER(KNNClassifier)
    ML_METHOD(__serialize, knn_serialize, arginfo_ml_get_array)
    ML_METHOD(__unserialize, knn_unserialize, arginfo_ml_unserialize)
    PHP_FE_END};

const zend_function_entry kd_methods[] = {
    ML_METHOD(__construct, kd_construct<KDNeighbors>, arginfo_kd_construct)
    ML_NEIGHBOR_CLASSIFIER(KDNeighbors)
    ML_METHOD(__serialize, serialize_native<KDNeighbors>, arginfo_ml_get_array)
    ML_METHOD(__unserialize, unserialize_native<KDNeighbors>, arginfo_ml_unserialize)
    PHP_FE_END};

const zend_function_entry radius_methods[] = {
    ML_METHOD(__construct, radius_construct, arginfo_radius_construct)
    ML_NEIGHBOR_CLASSIFIER(RadiusNeighbors)
    ML_METHOD(__serialize, serialize_native<RadiusNeighbors>, arginfo_ml_get_array)
    ML_METHOD(__unserialize, unserialize_native<RadiusNeighbors>, arginfo_ml_unserialize)
    PHP_FE_END};

const zend_function_entry knn_regressor_methods[] = {
    ML_METHOD(__construct, knn_regressor_construct, arginfo_knn_regressor_construct)
    ML_NEIGHBOR_REGRESSOR(KNNRegressor)
    PHP_FE_END};

const zend_function_entry kd_regressor_methods[] = {
    ML_METHOD(__construct, kd_construct<KDNeighborsRegressor>, arginfo_kd_construct)
    ML_NEIGHBOR_REGRESSOR(KDNeighborsRegressor)
    PHP_FE_END};

const zend_function_entry radius_regressor_methods[] = {
    ML_METHOD(__construct, radius_regressor_construct, arginfo_radius_regressor_construct)
    ML_NEIGHBOR_REGRESSOR(RadiusNeighborsRegressor)
    PHP_FE_END};

}  // namespace

void mlphp::register_knn() {
    register_class<KNNClassifier>("ML\\Classification\\KNearestNeighbors", knn_methods,
                                  {ce_learner, ce_classifier, ce_probabilistic});
}

void mlphp::register_neighbors() {
    const auto classifier = {ce_learner, ce_classifier, ce_probabilistic};
    const auto regressor = {ce_learner, ce_regressor};
    register_class<KDNeighbors>("ML\\Classification\\KDNeighbors", kd_methods, classifier);
    register_class<RadiusNeighbors>("ML\\Classification\\RadiusNeighbors", radius_methods, classifier);
    register_class<KNNRegressor>("ML\\Regression\\KNNRegressor", knn_regressor_methods, regressor);
    register_class<KDNeighborsRegressor>("ML\\Regression\\KDNeighborsRegressor", kd_regressor_methods, regressor);
    register_class<RadiusNeighborsRegressor>("ML\\Regression\\RadiusNeighborsRegressor", radius_regressor_methods, regressor);
}
