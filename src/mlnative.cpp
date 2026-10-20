#include "php_bridge.hpp"
#include "php_mlnative.h"

extern "C" {
#include "ext/standard/info.h"
}

PHP_MINIT_FUNCTION(mlnative) {
    mlphp::register_exceptions();
    mlphp::register_linear_regression();
    mlphp::register_logistic_regression();
    mlphp::register_knn();
    mlphp::register_gaussian_nb();
    mlphp::register_kmeans();
    mlphp::register_standard_scaler();
    mlphp::register_metrics();
    return SUCCESS;
}

PHP_MINFO_FUNCTION(mlnative) {
    php_info_print_table_start();
    php_info_print_table_header(2, "mlnative support", "enabled");
    php_info_print_table_row(2, "Version", PHP_MLNATIVE_VERSION);
    php_info_print_table_row(2, "Models",
                             "LinearRegression, LogisticRegression, KNearestNeighbors, GaussianNB, KMeans, StandardScaler");
    php_info_print_table_end();
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
    nullptr,
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
