#pragma once

// Evaluating a metric given by name or by a PHP callable, shared by
// ML\Metrics, the cross-validators, GridSearch and BayesianSearch.

#include "php_bridge.hpp"

namespace mlphp {

// Metric names accepted by validators and searches.
//   classification: accuracy, precision, recall, f1, mcc, informedness
//   regression:     r2, mse, mae, rmse, median_absolute_error, smape
struct Metric {
    std::string name;   // empty when a callable is used
    zval callable;      // IS_UNDEF unless a callable is used
    bool higher_is_better = true;

    Metric() { ZVAL_UNDEF(&callable); }
    Metric(const Metric &) = delete;
    Metric &operator=(const Metric &) = delete;
    ~Metric() { zval_ptr_dtor(&callable); }

    // From a string name, a callable, or null (default for the estimator class `kind`).
    void configure(zval *spec, zend_class_entry *kind);
    double evaluate(zval *expected, zval *predicted) const;
};

bool is_known_metric(const std::string &name);

// ML\CrossValidation\Validator interface and a way to run any validator object.
extern zend_class_entry *ce_validator;
std::vector<double> cross_validate(zval *validator, zval *estimator, HashTable *samples, HashTable *targets, const Metric &metric);
// Encodes two label lists against their union; returns the number of classes.
std::size_t encode_label_pair(HashTable *a, HashTable *b, std::vector<int> &ea, std::vector<int> &eb,
                              LabelEncoder *encoder = nullptr);

}  // namespace mlphp
