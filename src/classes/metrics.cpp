#include "arginfo.hpp"

#include <cmath>

namespace {

using mlphp::guard;

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_metric, 0, 2, IS_DOUBLE, 0)
    ZEND_ARG_TYPE_INFO(0, expected, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO(0, predicted, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_construct, 0, 0, 0)
ZEND_END_ARG_INFO()

std::vector<zval *> values_of(HashTable *ht) {
    std::vector<zval *> out;
    out.reserve(zend_hash_num_elements(ht));
    zval *value;
    ZEND_HASH_FOREACH_VAL(ht, value) {
        ZVAL_DEREF(value);
        out.push_back(value);
    }
    ZEND_HASH_FOREACH_END();
    return out;
}

void require_same_size(std::size_t expected, std::size_t predicted) {
    if (expected == 0) {
        throw std::invalid_argument("Arrays must not be empty");
    }
    if (expected != predicted) {
        throw std::invalid_argument("Expected (" + std::to_string(expected) + ") and predicted (" +
                                    std::to_string(predicted) + ") arrays must have the same size");
    }
}

struct Pairs {
    std::vector<double> expected;
    std::vector<double> predicted;
};

Pairs read_pairs(HashTable *expected, HashTable *predicted) {
    Pairs p{mlphp::read_numbers(expected, "Expected value"), mlphp::read_numbers(predicted, "Predicted value")};
    require_same_size(p.expected.size(), p.predicted.size());
    return p;
}

PHP_METHOD(Metrics, __construct) {
    ZEND_PARSE_PARAMETERS_NONE();
}

PHP_METHOD(Metrics, accuracy) {
    HashTable *expected, *predicted;
    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_ARRAY_HT(expected)
        Z_PARAM_ARRAY_HT(predicted)
    ZEND_PARSE_PARAMETERS_END();

    guard([&] {
        const std::vector<zval *> a = values_of(expected);
        const std::vector<zval *> b = values_of(predicted);
        require_same_size(a.size(), b.size());
        std::size_t correct = 0;
        for (std::size_t i = 0; i < a.size(); ++i) {
            if (zend_is_identical(a[i], b[i])) {
                ++correct;
            }
        }
        RETVAL_DOUBLE(static_cast<double>(correct) / static_cast<double>(a.size()));
    });
}

PHP_METHOD(Metrics, meanSquaredError) {
    HashTable *expected, *predicted;
    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_ARRAY_HT(expected)
        Z_PARAM_ARRAY_HT(predicted)
    ZEND_PARSE_PARAMETERS_END();

    guard([&] {
        const Pairs p = read_pairs(expected, predicted);
        double sum = 0.0;
        for (std::size_t i = 0; i < p.expected.size(); ++i) {
            const double d = p.expected[i] - p.predicted[i];
            sum += d * d;
        }
        RETVAL_DOUBLE(sum / static_cast<double>(p.expected.size()));
    });
}

PHP_METHOD(Metrics, meanAbsoluteError) {
    HashTable *expected, *predicted;
    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_ARRAY_HT(expected)
        Z_PARAM_ARRAY_HT(predicted)
    ZEND_PARSE_PARAMETERS_END();

    guard([&] {
        const Pairs p = read_pairs(expected, predicted);
        double sum = 0.0;
        for (std::size_t i = 0; i < p.expected.size(); ++i) {
            sum += std::fabs(p.expected[i] - p.predicted[i]);
        }
        RETVAL_DOUBLE(sum / static_cast<double>(p.expected.size()));
    });
}

PHP_METHOD(Metrics, r2Score) {
    HashTable *expected, *predicted;
    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_ARRAY_HT(expected)
        Z_PARAM_ARRAY_HT(predicted)
    ZEND_PARSE_PARAMETERS_END();

    guard([&] {
        const Pairs p = read_pairs(expected, predicted);
        const std::size_t n = p.expected.size();
        double mean = 0.0;
        for (double v : p.expected) {
            mean += v;
        }
        mean /= static_cast<double>(n);
        double ss_res = 0.0;
        double ss_tot = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            const double res = p.expected[i] - p.predicted[i];
            const double tot = p.expected[i] - mean;
            ss_res += res * res;
            ss_tot += tot * tot;
        }
        // Constant targets: perfect predictions score 1, anything else 0.
        if (ss_tot == 0.0) {
            RETVAL_DOUBLE(ss_res == 0.0 ? 1.0 : 0.0);
            return;
        }
        RETVAL_DOUBLE(1.0 - ss_res / ss_tot);
    });
}

const zend_function_entry methods[] = {
    PHP_ME(Metrics, __construct, arginfo_construct, ZEND_ACC_PRIVATE)
    PHP_ME(Metrics, accuracy, arginfo_metric, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    PHP_ME(Metrics, meanSquaredError, arginfo_metric, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    PHP_ME(Metrics, meanAbsoluteError, arginfo_metric, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    PHP_ME(Metrics, r2Score, arginfo_metric, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    PHP_FE_END
};

}  // namespace

void mlphp::register_metrics() {
    zend_class_entry tmp;
    INIT_CLASS_ENTRY(tmp, "ML\\Metrics", methods);
    zend_class_entry *ce = zend_register_internal_class(&tmp);
    ce->ce_flags |= ZEND_ACC_FINAL;
}
