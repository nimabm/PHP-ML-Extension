#pragma once

// Argument/return type info shared by several classes. The PHP-facing
// signatures are documented in stubs/mlnative.stub.php; keep both in sync.

#include "php_bridge.hpp"

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_MASK_EX(arginfo_ml_fit_supervised, 0, 2, MAY_BE_STATIC)
    ZEND_ARG_TYPE_INFO(0, samples, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO(0, targets, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_MASK_EX(arginfo_ml_fit_unsupervised, 0, 1, MAY_BE_STATIC)
    ZEND_ARG_TYPE_INFO(0, samples, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_MASK_EX(arginfo_ml_predict_label, 0, 1, MAY_BE_ARRAY | MAY_BE_LONG | MAY_BE_STRING)
    ZEND_ARG_TYPE_INFO(0, samples, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_ml_samples_to_array, 0, 1, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO(0, samples, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_ml_get_array, 0, 0, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_ml_get_float, 0, 0, IS_DOUBLE, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_ml_get_int, 0, 0, IS_LONG, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_ml_unserialize, 0, 1, IS_VOID, 0)
    ZEND_ARG_TYPE_INFO(0, data, IS_ARRAY, 0)
ZEND_END_ARG_INFO()
