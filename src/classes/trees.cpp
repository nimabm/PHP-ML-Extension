// ClassificationTree, ExtraTreeClassifier, RegressionTree, ExtraTreeRegressor,
// RandomForest and RandomForestRegressor.

#include <functional>
#include <numeric>

#include "binding.hpp"
#include "core/tree.hpp"

namespace {

using namespace mlphp;
using mlcore::ArchiveReader;
using mlcore::ArchiveWriter;
using mlcore::DecisionTree;
using mlcore::Forest;
using mlcore::TreeParams;

// ---- shared parameter parsing -----------------------------------------------------

TreeParams tree_params(zend_long max_height, zend_long max_leaf_size, double min_purity_increase,
                       zend_long max_features, bool max_features_null) {
    TreeParams p;
    p.max_height = positive(max_height, "maxHeight");
    p.max_leaf_size = positive(max_leaf_size, "maxLeafSize");
    p.min_purity_increase = non_negative(min_purity_increase, "minPurityIncrease");
    p.max_features = max_features_null ? 0 : positive(max_features, "maxFeatures");
    return p;
}

std::string rule_number(double v) {
    char buffer[32];
    snprintf(buffer, sizeof buffer, "%.6g", v);
    return buffer;
}

// Human-readable if/else rules of a tree.
std::string tree_rules(const DecisionTree &tree, HashTable *names, const std::function<std::string(const double *)> &leaf) {
    std::vector<std::string> feature_names(tree.features);
    for (std::size_t f = 0; f < tree.features; ++f) {
        feature_names[f] = "feature " + std::to_string(f);
    }
    if (names) {
        std::size_t f = 0;
        zval *name;
        ZEND_HASH_FOREACH_VAL(names, name) {
            if (f < tree.features) {
                zend_string *s = zval_get_string(name);
                feature_names[f] = zstr(s);
                zend_string_release(s);
            }
            ++f;
        }
        ZEND_HASH_FOREACH_END();
        if (f != tree.features) {
            throw std::invalid_argument("featureNames must contain one name per feature (" + std::to_string(tree.features) + ")");
        }
    }
    std::string out;
    std::function<void(std::size_t, std::size_t)> walk = [&](std::size_t i, std::size_t depth) {
        const DecisionTree::Node &node = tree.nodes[i];
        const std::string indent(depth * 4, ' ');
        if (node.feature < 0) {
            out += indent + "=> " + leaf(&tree.leaf_values[node.leaf * tree.width()]) + "\n";
            return;
        }
        const std::string &name = feature_names[static_cast<std::size_t>(node.feature)];
        out += indent + "if " + name + " < " + rule_number(node.threshold) + ":\n";
        walk(static_cast<std::size_t>(node.left), depth + 1);
        out += indent + "else:  # " + name + " >= " + rule_number(node.threshold) + "\n";
        walk(static_cast<std::size_t>(node.right), depth + 1);
    };
    walk(0, 0);
    return out;
}

// ---- single trees -----------------------------------------------------------------

template <bool Classification, bool Extra>
struct TreeModel {
    static constexpr std::uint32_t kTag = Classification ? (Extra ? mlcore::tag("XTRC") : mlcore::tag("CTRE"))
                                                         : (Extra ? mlcore::tag("XTRR") : mlcore::tag("RTRE"));
    TreeParams params;
    bool has_seed = false;
    std::uint64_t seed = 0;
    DecisionTree tree;
    LabelEncoder labels;

    bool fitted() const { return tree.fitted(); }
    std::size_t features() const { return tree.features; }
    std::size_t classes() const { return tree.classes; }

    void train(const mlcore::Matrix &X, const std::vector<double> &targets, std::size_t K) {
        mlcore::Rng rng = mlcore::make_rng(has_seed, seed);
        DecisionTree next;
        next.params = params;
        next.params.extra = Extra;
        next.classification = Classification;
        next.classes = Classification ? K : 1;
        std::vector<std::size_t> rows(X.rows);
        std::iota(rows.begin(), rows.end(), 0);
        next.fit(X, targets, {}, std::move(rows), rng);
        tree = std::move(next);
    }
    void fit(const mlcore::Matrix &X, const std::vector<int> &y, std::size_t K, unsigned) {
        train(X, std::vector<double>(y.begin(), y.end()), K);
    }
    void fit(const mlcore::Matrix &X, const std::vector<double> &y, unsigned) { train(X, y, 1); }

    int predict_one_class(const double *x) const {
        return static_cast<int>(mlcore::argmax(tree.leaf_for(x), tree.classes));
    }
    auto predict_one(const double *x) const {
        if constexpr (Classification) {
            return predict_one_class(x);
        } else {
            return tree.value(x);
        }
    }
    void probabilities(const double *x, double *out) const {
        const double *leaf = tree.leaf_for(x);
        std::copy(leaf, leaf + tree.classes, out);
    }
    std::vector<double> importances() const { return tree.normalized_importances(); }

    void save(ArchiveWriter &w) const {
        params.save(w);
        w.boolean(has_seed);
        w.u64(seed);
        w.boolean(fitted());
        if (fitted()) {
            tree.save(w);
            labels.save(w);
        }
    }
    void load(ArchiveReader &r) {
        params.load(r);
        has_seed = r.boolean();
        seed = r.u64();
        if (r.boolean()) {
            tree.load(r);
            labels.load(r);
            ArchiveReader::require(tree.classification == Classification);
            ArchiveReader::require(!Classification || tree.classes == labels.size(), "label count mismatch");
        }
    }
};

ZEND_BEGIN_ARG_INFO_EX(arginfo_tree_construct, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, maxHeight, IS_LONG, 0, "PHP_INT_MAX")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, maxLeafSize, IS_LONG, 0, "3")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, minPurityIncrease, IS_DOUBLE, 0, "1.0E-7")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, maxFeatures, IS_LONG, 1, "null")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, seed, IS_LONG, 1, "null")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_rules, 0, 0, IS_STRING, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, featureNames, IS_ARRAY, 1, "null")
ZEND_END_ARG_INFO()

template <typename M>
void ZEND_FASTCALL tree_construct(INTERNAL_FUNCTION_PARAMETERS) {
    zend_long max_height = ZEND_LONG_MAX, max_leaf_size = 3, max_features = 0, seed = 0;
    double min_purity_increase = 1e-7;
    bool max_features_null = true, seed_null = true;
    ZEND_PARSE_PARAMETERS_START(0, 5)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(max_height)
        Z_PARAM_LONG(max_leaf_size)
        Z_PARAM_DOUBLE(min_purity_increase)
        Z_PARAM_LONG_OR_NULL(max_features, max_features_null)
        Z_PARAM_LONG_OR_NULL(seed, seed_null)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        M fresh;
        fresh.params = tree_params(max_height, max_leaf_size, min_purity_increase, max_features, max_features_null);
        parse_seed(seed_null, seed, fresh.has_seed, fresh.seed);
        reset(ZEND_THIS, std::move(fresh));
    });
}

template <typename M>
void ZEND_FASTCALL tree_height(INTERNAL_FUNCTION_PARAMETERS) {
    ZEND_PARSE_PARAMETERS_NONE();
    guard([&] {
        const M &m = native<M>(ZEND_THIS);
        require_fitted(m.fitted(), ZEND_THIS);
        RETVAL_LONG(static_cast<zend_long>(m.tree.height));
    });
}

template <typename M>
void ZEND_FASTCALL tree_leaves(INTERNAL_FUNCTION_PARAMETERS) {
    ZEND_PARSE_PARAMETERS_NONE();
    guard([&] {
        const M &m = native<M>(ZEND_THIS);
        require_fitted(m.fitted(), ZEND_THIS);
        RETVAL_LONG(static_cast<zend_long>(m.tree.leaves()));
    });
}

std::string label_text(const LabelEncoder &labels, int index) {
    const Label &label = labels.classes[static_cast<std::size_t>(index)];
    if (const zend_long *l = std::get_if<zend_long>(&label)) {
        return std::to_string(*l);
    }
    return std::get<std::string>(label);
}

template <typename M>
void ZEND_FASTCALL tree_rules_method(INTERNAL_FUNCTION_PARAMETERS) {
    HashTable *names = nullptr;
    ZEND_PARSE_PARAMETERS_START(0, 1)
        Z_PARAM_OPTIONAL
        Z_PARAM_ARRAY_HT_OR_NULL(names)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        const M &m = native<M>(ZEND_THIS);
        require_fitted(m.fitted(), ZEND_THIS);
        std::string text;
        if (m.tree.classification) {
            text = tree_rules(m.tree, names, [&](const double *leaf) {
                const int best = static_cast<int>(mlcore::argmax(leaf, m.tree.classes));
                return label_text(m.labels, best) + " (p=" + rule_number(leaf[best]) + ")";
            });
        } else {
            text = tree_rules(m.tree, names, [](const double *leaf) { return rule_number(*leaf); });
        }
        RETVAL_STRINGL(text.data(), text.size());
    });
}

using ClassificationTree = TreeModel<true, false>;
using ExtraTreeClassifier = TreeModel<true, true>;
using RegressionTree = TreeModel<false, false>;
using ExtraTreeRegressor = TreeModel<false, true>;

#define ML_TREE_COMMON(M)                                                          \
    ML_METHOD(__construct, tree_construct<M>, arginfo_tree_construct)               \
    ML_METHOD(getFeatureImportances, get_importances<M>, arginfo_ml_get_array)      \
    ML_METHOD(getHeight, tree_height<M>, arginfo_ml_get_int)                        \
    ML_METHOD(getLeafCount, tree_leaves<M>, arginfo_ml_get_int)                     \
    ML_METHOD(rules, tree_rules_method<M>, arginfo_rules)                           \
    ML_METHOD(__serialize, serialize_native<M>, arginfo_ml_get_array)               \
    ML_METHOD(__unserialize, unserialize_native<M>, arginfo_ml_unserialize)

#define ML_CLASSIFIER_METHODS(M)                                                   \
    ML_METHOD(fit, fit_classifier<M>, arginfo_ml_fit_supervised)                    \
    ML_METHOD(predict, predict_classifier<M>, arginfo_ml_predict_label)             \
    ML_METHOD(predictProbability, predict_proba<M>, arginfo_ml_samples_to_array)    \
    ML_METHOD(getClasses, get_classes<M>, arginfo_ml_get_array)

#define ML_REGRESSOR_METHODS(M)                                                    \
    ML_METHOD(fit, fit_regressor<M>, arginfo_ml_fit_supervised)                     \
    ML_METHOD(predict, predict_regressor<M>, arginfo_ml_predict_float)

const zend_function_entry classification_tree_methods[] = {
    ML_TREE_COMMON(ClassificationTree) ML_CLASSIFIER_METHODS(ClassificationTree) PHP_FE_END};
const zend_function_entry extra_tree_classifier_methods[] = {
    ML_TREE_COMMON(ExtraTreeClassifier) ML_CLASSIFIER_METHODS(ExtraTreeClassifier) PHP_FE_END};
const zend_function_entry regression_tree_methods[] = {
    ML_TREE_COMMON(RegressionTree) ML_REGRESSOR_METHODS(RegressionTree) PHP_FE_END};
const zend_function_entry extra_tree_regressor_methods[] = {
    ML_TREE_COMMON(ExtraTreeRegressor) ML_REGRESSOR_METHODS(ExtraTreeRegressor) PHP_FE_END};

// ---- forests ------------------------------------------------------------------------

template <bool Classification>
struct ForestModel {
    static constexpr std::uint32_t kTag = Classification ? mlcore::tag("RFOR") : mlcore::tag("RFRG");
    Forest forest;
    LabelEncoder labels;

    ForestModel() { forest.classification = Classification; }
    bool fitted() const { return forest.fitted(); }
    std::size_t features() const { return forest.features; }
    std::size_t classes() const { return forest.classes; }

    void fit(const mlcore::Matrix &X, const std::vector<int> &y, std::size_t K, unsigned threads) {
        forest.fit(X, std::vector<double>(y.begin(), y.end()), K, threads);
    }
    void fit(const mlcore::Matrix &X, const std::vector<double> &y, unsigned threads) { forest.fit(X, y, 1, threads); }

    auto predict_one(const double *x) const {
        if constexpr (Classification) {
            std::vector<double> p(forest.classes);
            forest.probabilities(x, p.data());
            return static_cast<int>(mlcore::argmax(p.data(), p.size()));
        } else {
            return forest.value(x);
        }
    }
    void probabilities(const double *x, double *out) const { forest.probabilities(x, out); }
    std::vector<double> importances() const { return forest.importances(); }

    void save(ArchiveWriter &w) const {
        forest.save(w);
        labels.save(w);
    }
    void load(ArchiveReader &r) {
        forest.load(r);
        labels.load(r);
        ArchiveReader::require(forest.classification == Classification);
        ArchiveReader::require(!Classification || !forest.fitted() || forest.classes == labels.size(), "label count mismatch");
    }
};

using RandomForest = ForestModel<true>;
using RandomForestRegressor = ForestModel<false>;

ZEND_BEGIN_ARG_INFO_EX(arginfo_forest_construct, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, estimators, IS_LONG, 0, "100")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, ratio, IS_DOUBLE, 0, "1.0")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, balanced, _IS_BOOL, 0, "false")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, maxHeight, IS_LONG, 0, "PHP_INT_MAX")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, maxLeafSize, IS_LONG, 0, "3")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, minPurityIncrease, IS_DOUBLE, 0, "1.0E-7")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, maxFeatures, IS_LONG, 1, "null")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, extraTrees, _IS_BOOL, 0, "false")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, seed, IS_LONG, 1, "null")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_forest_regressor_construct, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, estimators, IS_LONG, 0, "100")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, ratio, IS_DOUBLE, 0, "1.0")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, maxHeight, IS_LONG, 0, "PHP_INT_MAX")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, maxLeafSize, IS_LONG, 0, "3")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, minPurityIncrease, IS_DOUBLE, 0, "1.0E-7")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, maxFeatures, IS_LONG, 1, "null")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, extraTrees, _IS_BOOL, 0, "false")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, seed, IS_LONG, 1, "null")
ZEND_END_ARG_INFO()

void configure_forest(Forest &f, zend_long estimators, double ratio, bool balanced, zend_long max_height,
                      zend_long max_leaf_size, double min_purity_increase, zend_long max_features,
                      bool max_features_null, bool extra, zend_long seed, bool seed_null) {
    f.estimators = positive(estimators, "estimators");
    check(ratio > 0.0 && ratio <= 1.0, "ratio must be greater than 0 and at most 1");
    f.ratio = ratio;
    f.balanced = balanced;
    f.params = tree_params(max_height, max_leaf_size, min_purity_increase, max_features, max_features_null);
    f.params.extra = extra;
    parse_seed(seed_null, seed, f.has_seed, f.seed);
}

void ZEND_FASTCALL random_forest_construct(INTERNAL_FUNCTION_PARAMETERS) {
    zend_long estimators = 100, max_height = ZEND_LONG_MAX, max_leaf_size = 3, max_features = 0, seed = 0;
    double ratio = 1.0, min_purity_increase = 1e-7;
    bool balanced = false, extra = false, max_features_null = true, seed_null = true;
    ZEND_PARSE_PARAMETERS_START(0, 9)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(estimators)
        Z_PARAM_DOUBLE(ratio)
        Z_PARAM_BOOL(balanced)
        Z_PARAM_LONG(max_height)
        Z_PARAM_LONG(max_leaf_size)
        Z_PARAM_DOUBLE(min_purity_increase)
        Z_PARAM_LONG_OR_NULL(max_features, max_features_null)
        Z_PARAM_BOOL(extra)
        Z_PARAM_LONG_OR_NULL(seed, seed_null)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        RandomForest fresh;
        configure_forest(fresh.forest, estimators, ratio, balanced, max_height, max_leaf_size, min_purity_increase,
                         max_features, max_features_null, extra, seed, seed_null);
        reset(ZEND_THIS, std::move(fresh));
    });
}

void ZEND_FASTCALL random_forest_regressor_construct(INTERNAL_FUNCTION_PARAMETERS) {
    zend_long estimators = 100, max_height = ZEND_LONG_MAX, max_leaf_size = 3, max_features = 0, seed = 0;
    double ratio = 1.0, min_purity_increase = 1e-7;
    bool extra = false, max_features_null = true, seed_null = true;
    ZEND_PARSE_PARAMETERS_START(0, 8)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(estimators)
        Z_PARAM_DOUBLE(ratio)
        Z_PARAM_LONG(max_height)
        Z_PARAM_LONG(max_leaf_size)
        Z_PARAM_DOUBLE(min_purity_increase)
        Z_PARAM_LONG_OR_NULL(max_features, max_features_null)
        Z_PARAM_BOOL(extra)
        Z_PARAM_LONG_OR_NULL(seed, seed_null)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        RandomForestRegressor fresh;
        configure_forest(fresh.forest, estimators, ratio, false, max_height, max_leaf_size, min_purity_increase,
                         max_features, max_features_null, extra, seed, seed_null);
        reset(ZEND_THIS, std::move(fresh));
    });
}

const zend_function_entry random_forest_methods[] = {
    ML_METHOD(__construct, random_forest_construct, arginfo_forest_construct)
    ML_CLASSIFIER_METHODS(RandomForest)
    ML_METHOD(getFeatureImportances, get_importances<RandomForest>, arginfo_ml_get_array)
    ML_METHOD(__serialize, serialize_native<RandomForest>, arginfo_ml_get_array)
    ML_METHOD(__unserialize, unserialize_native<RandomForest>, arginfo_ml_unserialize)
    PHP_FE_END};

const zend_function_entry random_forest_regressor_methods[] = {
    ML_METHOD(__construct, random_forest_regressor_construct, arginfo_forest_regressor_construct)
    ML_REGRESSOR_METHODS(RandomForestRegressor)
    ML_METHOD(getFeatureImportances, get_importances<RandomForestRegressor>, arginfo_ml_get_array)
    ML_METHOD(__serialize, serialize_native<RandomForestRegressor>, arginfo_ml_get_array)
    ML_METHOD(__unserialize, unserialize_native<RandomForestRegressor>, arginfo_ml_unserialize)
    PHP_FE_END};

}  // namespace

void mlphp::register_trees() {
    const auto classifier = {ce_learner, ce_classifier, ce_probabilistic, ce_ranks_features};
    const auto regressor = {ce_learner, ce_regressor, ce_ranks_features};
    register_class<ClassificationTree>("ML\\Classification\\ClassificationTree", classification_tree_methods, classifier);
    register_class<ExtraTreeClassifier>("ML\\Classification\\ExtraTreeClassifier", extra_tree_classifier_methods, classifier);
    register_class<RandomForest>("ML\\Classification\\RandomForest", random_forest_methods, classifier);
    register_class<RegressionTree>("ML\\Regression\\RegressionTree", regression_tree_methods, regressor);
    register_class<ExtraTreeRegressor>("ML\\Regression\\ExtraTreeRegressor", extra_tree_regressor_methods, regressor);
    register_class<RandomForestRegressor>("ML\\Regression\\RandomForestRegressor", random_forest_regressor_methods, regressor);
}
