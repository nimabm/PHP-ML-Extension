#include "php_bridge.hpp"

#include <map>

namespace mlphp {

zend_class_entry *ce_ml_exception = nullptr;
zend_class_entry *ce_invalid_argument = nullptr;
zend_class_entry *ce_runtime = nullptr;
zend_class_entry *ce_not_fitted = nullptr;

namespace {

zend_class_entry *register_exception(const char *name, zend_class_entry *parent) {
    zend_class_entry tmp;
    INIT_CLASS_ENTRY_EX(tmp, name, std::strlen(name), nullptr);
    zend_class_entry *ce = zend_register_internal_class_ex(&tmp, parent);
    zend_class_implements(ce, 1, ce_ml_exception);
    return ce;
}

std::string type_name(zval *value) {
    return zend_zval_type_name(value);
}

std::string feature_count_error(const std::string &subject, std::size_t got, std::size_t expected) {
    return subject + " has " + std::to_string(got) + " features, expected " + std::to_string(expected);
}

void read_features(HashTable *ht, const std::string &subject, double *out) {
    std::size_t j = 0;
    zval *value;
    ZEND_HASH_FOREACH_VAL(ht, value) {
        ZVAL_DEREF(value);
        switch (Z_TYPE_P(value)) {
            case IS_LONG:
                out[j] = static_cast<double>(Z_LVAL_P(value));
                break;
            case IS_DOUBLE:
                out[j] = Z_DVAL_P(value);
                break;
            default:
                throw std::invalid_argument(subject + ", feature #" + std::to_string(j) + " must be int or float, " +
                                            type_name(value) + " given");
        }
        ++j;
    }
    ZEND_HASH_FOREACH_END();
}

Label label_from(zval *value, std::size_t index) {
    ZVAL_DEREF(value);
    switch (Z_TYPE_P(value)) {
        case IS_LONG:
            return Label(std::in_place_index<0>, Z_LVAL_P(value));
        case IS_STRING:
            return Label(std::in_place_index<1>, std::string(Z_STRVAL_P(value), Z_STRLEN_P(value)));
        default:
            throw std::invalid_argument("Label #" + std::to_string(index) + " must be int or string, " +
                                        type_name(value) + " given");
    }
}

}  // namespace

void register_exceptions() {
    zend_class_entry tmp;
    INIT_CLASS_ENTRY(tmp, "ML\\Exception\\MLException", nullptr);
    ce_ml_exception = zend_register_internal_interface(&tmp);

    ce_invalid_argument = register_exception("ML\\Exception\\InvalidArgumentException", spl_ce_InvalidArgumentException);
    ce_runtime = register_exception("ML\\Exception\\RuntimeException", spl_ce_RuntimeException);
    ce_not_fitted = register_exception("ML\\Exception\\NotFittedException", spl_ce_LogicException);
}

void require_fitted(bool fitted, zval *self) {
    if (!fitted) {
        throw mlcore::NotFittedError(std::string(ZSTR_VAL(Z_OBJCE_P(self)->name)) +
                                     " is not fitted yet; call fit() first");
    }
}

bool is_sample_list(HashTable *ht) {
    zval *first;
    ZEND_HASH_FOREACH_VAL(ht, first) {
        ZVAL_DEREF(first);
        return Z_TYPE_P(first) == IS_ARRAY;
    }
    ZEND_HASH_FOREACH_END();
    throw std::invalid_argument("Samples must not be empty");
}

mlcore::Matrix read_samples(HashTable *ht, std::size_t expected_features) {
    const std::size_t n = zend_hash_num_elements(ht);
    if (n == 0) {
        throw std::invalid_argument("Samples must not be empty");
    }

    mlcore::Matrix X;
    std::size_t r = 0;
    zval *row;
    ZEND_HASH_FOREACH_VAL(ht, row) {
        ZVAL_DEREF(row);
        const std::string subject = "Sample #" + std::to_string(r);
        if (Z_TYPE_P(row) != IS_ARRAY) {
            throw std::invalid_argument(subject + " must be an array of numbers, " + type_name(row) + " given");
        }
        HashTable *features = Z_ARRVAL_P(row);
        const std::size_t d = zend_hash_num_elements(features);
        if (r == 0) {
            if (d == 0) {
                throw std::invalid_argument(subject + " has no features");
            }
            if (expected_features != 0 && d != expected_features) {
                throw std::invalid_argument(feature_count_error(subject, d, expected_features));
            }
            X = mlcore::Matrix(n, d);
        } else if (d != X.cols) {
            throw std::invalid_argument(feature_count_error(subject, d, X.cols));
        }
        read_features(features, subject, X.row(r));
        ++r;
    }
    ZEND_HASH_FOREACH_END();
    return X;
}

std::vector<double> read_sample(HashTable *ht, std::size_t expected_features) {
    const std::size_t d = zend_hash_num_elements(ht);
    if (d == 0) {
        throw std::invalid_argument("Sample must not be empty");
    }
    if (d != expected_features) {
        throw std::invalid_argument(feature_count_error("Sample", d, expected_features));
    }
    std::vector<double> x(d);
    read_features(ht, "Sample", x.data());
    return x;
}

std::vector<double> read_numbers(HashTable *ht, const char *what) {
    std::vector<double> out;
    out.reserve(zend_hash_num_elements(ht));
    zval *value;
    ZEND_HASH_FOREACH_VAL(ht, value) {
        ZVAL_DEREF(value);
        if (Z_TYPE_P(value) == IS_LONG) {
            out.push_back(static_cast<double>(Z_LVAL_P(value)));
        } else if (Z_TYPE_P(value) == IS_DOUBLE) {
            out.push_back(Z_DVAL_P(value));
        } else {
            throw std::invalid_argument(std::string(what) + " #" + std::to_string(out.size()) +
                                        " must be int or float, " + type_name(value) + " given");
        }
    }
    ZEND_HASH_FOREACH_END();
    return out;
}

void write_doubles(zval *out, const std::vector<double> &values) {
    array_init_size(out, static_cast<uint32_t>(values.size()));
    for (double v : values) {
        add_next_index_double(out, v);
    }
}

void write_ints(zval *out, const std::vector<int> &values) {
    array_init_size(out, static_cast<uint32_t>(values.size()));
    for (int v : values) {
        add_next_index_long(out, v);
    }
}

void write_matrix(zval *out, const mlcore::Matrix &m) {
    array_init_size(out, static_cast<uint32_t>(m.rows));
    for (std::size_t r = 0; r < m.rows; ++r) {
        zval row;
        array_init_size(&row, static_cast<uint32_t>(m.cols));
        for (std::size_t c = 0; c < m.cols; ++c) {
            add_next_index_double(&row, m(r, c));
        }
        add_next_index_zval(out, &row);
    }
}

std::vector<int> LabelEncoder::fit(HashTable *ht) {
    std::vector<Label> raw;
    raw.reserve(zend_hash_num_elements(ht));
    zval *value;
    ZEND_HASH_FOREACH_VAL(ht, value) {
        raw.push_back(label_from(value, raw.size()));
    }
    ZEND_HASH_FOREACH_END();
    if (raw.empty()) {
        throw std::invalid_argument("Labels must not be empty");
    }

    std::map<Label, int> index;
    for (const Label &label : raw) {
        index.emplace(label, 0);
    }
    std::vector<Label> sorted;
    sorted.reserve(index.size());
    for (auto &entry : index) {
        entry.second = static_cast<int>(sorted.size());
        sorted.push_back(entry.first);
    }

    std::vector<int> encoded;
    encoded.reserve(raw.size());
    for (const Label &label : raw) {
        encoded.push_back(index[label]);
    }
    classes = std::move(sorted);
    return encoded;
}

void LabelEncoder::write(int index, zval *out) const {
    const Label &label = classes[static_cast<std::size_t>(index)];
    if (const zend_long *l = std::get_if<zend_long>(&label)) {
        ZVAL_LONG(out, *l);
    } else {
        const std::string &s = std::get<std::string>(label);
        ZVAL_STRINGL(out, s.data(), s.size());
    }
}

void LabelEncoder::write_keyed(zval *out, int index, double value) const {
    const Label &label = classes[static_cast<std::size_t>(index)];
    if (const zend_long *l = std::get_if<zend_long>(&label)) {
        add_index_double(out, *l, value);
    } else {
        const std::string &s = std::get<std::string>(label);
        zval v;
        ZVAL_DOUBLE(&v, value);
        // symtable semantics: a label "7" becomes the integer key 7, as in PHP.
        zend_symtable_str_update(Z_ARRVAL_P(out), s.data(), s.size(), &v);
    }
}

void LabelEncoder::write_all(zval *out) const {
    array_init_size(out, static_cast<uint32_t>(classes.size()));
    for (std::size_t i = 0; i < classes.size(); ++i) {
        zval label;
        write(static_cast<int>(i), &label);
        add_next_index_zval(out, &label);
    }
}

void LabelEncoder::read_all(HashTable *ht) {
    std::vector<Label> restored;
    zval *value;
    ZEND_HASH_FOREACH_VAL(ht, value) {
        restored.push_back(label_from(value, restored.size()));
    }
    ZEND_HASH_FOREACH_END();
    classes = std::move(restored);
}

void put_doubles(zval *arr, const char *key, const std::vector<double> &values) {
    zval list;
    write_doubles(&list, values);
    add_assoc_zval(arr, key, &list);
}

zval *require_key(HashTable *ht, const char *key) {
    zval *value = zend_hash_str_find_deref(ht, key, std::strlen(key));
    if (!value) {
        throw std::invalid_argument(std::string("Invalid serialized data: missing \"") + key + "\"");
    }
    return value;
}

double get_double(HashTable *ht, const char *key) {
    zval *value = require_key(ht, key);
    if (Z_TYPE_P(value) == IS_DOUBLE) {
        return Z_DVAL_P(value);
    }
    if (Z_TYPE_P(value) == IS_LONG) {
        return static_cast<double>(Z_LVAL_P(value));
    }
    throw std::invalid_argument(std::string("Invalid serialized data: \"") + key + "\" must be a number");
}

zend_long get_long(HashTable *ht, const char *key) {
    zval *value = require_key(ht, key);
    if (Z_TYPE_P(value) != IS_LONG) {
        throw std::invalid_argument(std::string("Invalid serialized data: \"") + key + "\" must be an int");
    }
    return Z_LVAL_P(value);
}

HashTable *get_array(HashTable *ht, const char *key) {
    zval *value = require_key(ht, key);
    if (Z_TYPE_P(value) != IS_ARRAY) {
        throw std::invalid_argument(std::string("Invalid serialized data: \"") + key + "\" must be an array");
    }
    return Z_ARRVAL_P(value);
}

std::vector<double> get_doubles(HashTable *ht, const char *key) {
    return read_numbers(get_array(ht, key), key);
}

bool shape_matches(std::size_t total, std::size_t rows, std::size_t cols) {
    return rows != 0 && cols != 0 && total % rows == 0 && total / rows == cols;
}

}  // namespace mlphp
