#include "php_bridge.hpp"
#include "php_mlnative.h"

extern "C" {
#include "ext/standard/info.h"
}

PHP_INI_BEGIN()
    PHP_INI_ENTRY("mlnative.threads", "1", PHP_INI_ALL, nullptr)
PHP_INI_END()

PHP_MINIT_FUNCTION(mlnative) {
    REGISTER_INI_ENTRIES();

    mlphp::register_exceptions();
    mlphp::register_interfaces();

    // 0.1.0 models
    mlphp::register_linear_regression();
    mlphp::register_logistic_regression();
    mlphp::register_knn();
    mlphp::register_gaussian_nb();
    mlphp::register_kmeans();
    mlphp::register_standard_scaler();
    mlphp::register_metrics();

    // 0.1.2
    mlphp::register_distance();
    mlphp::register_datasets();
    mlphp::register_trees();
    mlphp::register_neighbors();
    mlphp::register_naive_bayes();
    mlphp::register_cross_validation();
    mlphp::register_preprocessing();
    mlphp::register_table_transformers();
    mlphp::register_clustering();
    mlphp::register_anomaly();
    mlphp::register_boosting();
    mlphp::register_ensemble();
    mlphp::register_model_selection();
    mlphp::register_pipeline();
    mlphp::register_text();
    mlphp::register_decomposition();
    mlphp::register_svm();
    mlphp::register_neuralnet();
    return SUCCESS;
}

PHP_MSHUTDOWN_FUNCTION(mlnative) {
    UNREGISTER_INI_ENTRIES();
    return SUCCESS;
}

PHP_MINFO_FUNCTION(mlnative) {
    php_info_print_table_start();
    php_info_print_table_header(2, "mlnative support", "enabled");
    php_info_print_table_row(2, "Version", PHP_MLNATIVE_VERSION);
    php_info_print_table_row(2, "Classification",
                             "LogisticRegression, KNearestNeighbors, KDNeighbors, RadiusNeighbors, GaussianNB, "
                             "NaiveBayes, MultinomialNB, ClassificationTree, ExtraTreeClassifier, RandomForest, "
                             "AdaBoost, LogitBoost, OneVsRest, SVC, MLPClassifier");
    php_info_print_table_row(2, "Regression",
                             "LinearRegression, ElasticNet, RegressionTree, ExtraTreeRegressor, "
                             "RandomForestRegressor, GradientBoost, KNNRegressor, KDNeighborsRegressor, "
                             "RadiusNeighborsRegressor, Adaline, SVR, MLPRegressor");
    php_info_print_table_row(2, "Clustering",
                             "KMeans, KMedoids, FuzzyCMeans, GaussianMixture, MeanShift, DBSCAN, OPTICS, Ward");
    php_info_print_table_row(2, "Anomaly detection",
                             "RobustZScore, GaussianMLE, IsolationForest, LocalOutlierFactor, Loda, OneClassSVM");
    php_info_print_table_end();
    DISPLAY_INI_ENTRIES();
}

static const zend_module_dep mlnative_deps[] = {
    ZEND_MOD_REQUIRED("spl")
    ZEND_MOD_END
};

zend_module_entry mlnative_module_entry = {
    STANDARD_MODULE_HEADER_EX,
    nullptr,
    mlnative_deps,
    "mlnative",
    nullptr,  // no functions: everything is a class
    PHP_MINIT(mlnative),
    PHP_MSHUTDOWN(mlnative),
    nullptr,
    nullptr,
    PHP_MINFO(mlnative),
    PHP_MLNATIVE_VERSION,
    STANDARD_MODULE_PROPERTIES
};

#ifdef COMPILE_DL_MLNATIVE
#ifdef ZTS
ZEND_TSRMLS_CACHE_DEFINE()
#endif
ZEND_GET_MODULE(mlnative)
#endif
