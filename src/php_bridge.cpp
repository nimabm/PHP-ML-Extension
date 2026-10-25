#include "php_bridge.hpp"

#include <map>
#include <thread>

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

// ---- interfaces ---------------------------------------------------------------

zend_class_entry *ce_estimator = nullptr;
zend_class_entry *ce_learner = nullptr;
zend_class_entry *ce_classifier = nullptr;
zend_class_entry *ce_regressor = nullptr;
zend_class_entry *ce_clusterer = nullptr;
zend_class_entry *ce_anomaly = nullptr;
zend_class_entry *ce_probabilistic = nullptr;
zend_class_entry *ce_transformer = nullptr;
zend_class_entry *ce_stateful = nullptr;
zend_class_entry *ce_ranks_features = nullptr;
zend_class_entry *ce_tokenizer = nullptr;

namespace {
zend_class_entry *register_interface(const char *name, std::initializer_list<zend_class_entry *> parents = {}) {
    zend_class_entry tmp;
    INIT_CLASS_ENTRY_EX(tmp, name, std::strlen(name), nullptr);
    zend_class_entry *ce = zend_register_internal_interface(&tmp);
    for (zend_class_entry *parent : parents) {
        zend_class_implements(ce, 1, parent);
    }
    return ce;
}
}  // namespace

void register_interfaces() {
    ce_estimator = register_interface("ML\\Estimator");
    ce_learner = register_interface("ML\\Learner");
    ce_classifier = register_interface("ML\\Classifier", {ce_estimator});
    ce_regressor = register_interface("ML\\Regressor", {ce_estimator});
    ce_clusterer = register_interface("ML\\Clusterer", {ce_estimator});
    ce_anomaly = register_interface("ML\\AnomalyDetector", {ce_estimator});
    ce_probabilistic = register_interface("ML\\Probabilistic");
    ce_transformer = register_interface("ML\\Transformer");
    ce_stateful = register_interface("ML\\Stateful");
    ce_ranks_features = register_interface("ML\\RanksFeatures");
    ce_tokenizer = register_interface("ML\\Text\\Tokenizer");
}

zend_class_entry *register_plain_class(const char *name, const zend_function_entry *methods,
                                       std::initializer_list<zend_class_entry *> interfaces) {
    zend_class_entry tmp;
    INIT_CLASS_ENTRY_EX(tmp, name, std::strlen(name), methods);
    zend_class_entry *ce = zend_register_internal_class(&tmp);
    ce->ce_flags |= ZEND_ACC_FINAL | ZEND_ACC_NO_DYNAMIC_PROPERTIES;
    for (zend_class_entry *iface : interfaces) {
        zend_class_implements(ce, 1, iface);
    }
    return ce;
}

unsigned threads() {
    static char name[] = "mlnative.threads";
    const zend_long n = zend_ini_long(name, sizeof(name) - 1, 0);
    if (n <= 0) {
        const unsigned cores = std::thread::hardware_concurrency();
        return cores ? cores : 1;
    }
    return static_cast<unsigned>(std::min<zend_long>(n, 1024));
}

// ---- labels -------------------------------------------------------------------

Label read_label(zval *value, std::size_t index) { return label_from(value, index); }

std::vector<Label> read_labels(HashTable *ht, const char *what) {
    std::vector<Label> out;
    out.reserve(zend_hash_num_elements(ht));
    zval *value;
    ZEND_HASH_FOREACH_VAL(ht, value) {
        ZVAL_DEREF(value);
        if (Z_TYPE_P(value) == IS_LONG) {
            out.emplace_back(std::in_place_index<0>, Z_LVAL_P(value));
        } else if (Z_TYPE_P(value) == IS_STRING) {
            out.emplace_back(std::in_place_index<1>, std::string(Z_STRVAL_P(value), Z_STRLEN_P(value)));
        } else {
            throw std::invalid_argument(std::string(what) + " #" + std::to_string(out.size()) +
                                        " must be int or string, " + type_name(value) + " given");
        }
    }
    ZEND_HASH_FOREACH_END();
    return out;
}

void write_label(const Label &label, zval *out) {
    if (const zend_long *l = std::get_if<zend_long>(&label)) {
        ZVAL_LONG(out, *l);
    } else {
        const std::string &s = std::get<std::string>(label);
        ZVAL_STRINGL(out, s.data(), s.size());
    }
}

void save_label(mlcore::ArchiveWriter &w, const Label &label) {
    if (const zend_long *l = std::get_if<zend_long>(&label)) {
        w.u8(0);
        w.i64(*l);
    } else {
        w.u8(1);
        w.str(std::get<std::string>(label));
    }
}

Label load_label(mlcore::ArchiveReader &r) {
    const std::uint8_t kind = r.u8();
    if (kind == 0) {
        return Label(std::in_place_index<0>, static_cast<zend_long>(r.i64()));
    }
    if (kind == 1) {
        return Label(std::in_place_index<1>, r.str());
    }
    mlcore::ArchiveReader::fail("bad label");
}

int LabelEncoder::find(const Label &label) const {
    const auto it = std::lower_bound(classes.begin(), classes.end(), label);
    return it != classes.end() && *it == label ? static_cast<int>(it - classes.begin()) : -1;
}

void LabelEncoder::save(mlcore::ArchiveWriter &w) const {
    w.size(classes.size());
    for (const Label &label : classes) {
        save_label(w, label);
    }
}

void LabelEncoder::load(mlcore::ArchiveReader &r) {
    const std::size_t n = r.count(9);
    std::vector<Label> restored;
    restored.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        restored.push_back(load_label(r));
    }
    // find() relies on strictly increasing labels.
    for (std::size_t i = 1; i < restored.size(); ++i) {
        mlcore::ArchiveReader::require(restored[i - 1] < restored[i], "labels must be sorted and unique");
    }
    classes = std::move(restored);
}

// ---- tables -------------------------------------------------------------------

namespace {

Cell read_cell(zval *value, const std::string *placeholder) {
    ZVAL_DEREF(value);
    Cell c;
    switch (Z_TYPE_P(value)) {
        case IS_NULL:
            break;
        case IS_LONG:
            c.kind = Cell::Kind::Int;
            c.i = Z_LVAL_P(value);
            break;
        case IS_DOUBLE:
            if (!std::isnan(Z_DVAL_P(value))) {
                c.kind = Cell::Kind::Float;
                c.f = Z_DVAL_P(value);
            }
            break;
        case IS_STRING:
            if (!placeholder || placeholder->size() != Z_STRLEN_P(value) ||
                std::memcmp(placeholder->data(), Z_STRVAL_P(value), Z_STRLEN_P(value)) != 0) {
                c.kind = Cell::Kind::String;
                c.s.assign(Z_STRVAL_P(value), Z_STRLEN_P(value));
            }
            break;
        case IS_TRUE:
        case IS_FALSE:
            c.kind = Cell::Kind::Bool;
            c.i = Z_TYPE_P(value) == IS_TRUE;
            break;
        default:
            throw std::invalid_argument(std::string("Features must be int, float, string, bool or null, ") +
                                        type_name(value) + " given");
    }
    return c;
}

void read_row(HashTable *row, Table &t, std::size_t r, const std::string *placeholder) {
    std::size_t c = 0;
    zval *value;
    ZEND_HASH_FOREACH_VAL(row, value) {
        t.at(r, c++) = read_cell(value, placeholder);
    }
    ZEND_HASH_FOREACH_END();
}

void write_cell(const Cell &c, zval *out) {
    switch (c.kind) {
        case Cell::Kind::Missing: ZVAL_NULL(out); break;
        case Cell::Kind::Int: ZVAL_LONG(out, c.i); break;
        case Cell::Kind::Float: ZVAL_DOUBLE(out, c.f); break;
        case Cell::Kind::String: ZVAL_STRINGL(out, c.s.data(), c.s.size()); break;
        case Cell::Kind::Bool: ZVAL_BOOL(out, c.i != 0); break;
    }
}

}  // namespace

Table read_table(HashTable *ht, bool single, std::size_t expected_cols, const std::string *placeholder) {
    Table t;
    if (single) {
        t.rows = 1;
        t.cols = zend_hash_num_elements(ht);
        if (t.cols == 0) {
            throw std::invalid_argument("Sample must not be empty");
        }
        if (expected_cols != 0 && t.cols != expected_cols) {
            throw std::invalid_argument(feature_count_error("Sample", t.cols, expected_cols));
        }
        t.cells.resize(t.cols);
        read_row(ht, t, 0, placeholder);
        return t;
    }
    t.rows = zend_hash_num_elements(ht);
    if (t.rows == 0) {
        throw std::invalid_argument("Samples must not be empty");
    }
    std::size_t r = 0;
    zval *row;
    ZEND_HASH_FOREACH_VAL(ht, row) {
        ZVAL_DEREF(row);
        const std::string subject = "Sample #" + std::to_string(r);
        if (Z_TYPE_P(row) != IS_ARRAY) {
            throw std::invalid_argument(subject + " must be an array, " + type_name(row) + " given");
        }
        const std::size_t d = zend_hash_num_elements(Z_ARRVAL_P(row));
        if (r == 0) {
            if (d == 0) {
                throw std::invalid_argument(subject + " has no features");
            }
            if (expected_cols != 0 && d != expected_cols) {
                throw std::invalid_argument(feature_count_error(subject, d, expected_cols));
            }
            t.cols = d;
            t.cells.resize(t.rows * t.cols);
        } else if (d != t.cols) {
            throw std::invalid_argument(feature_count_error(subject, d, t.cols));
        }
        read_row(Z_ARRVAL_P(row), t, r, placeholder);
        ++r;
    }
    ZEND_HASH_FOREACH_END();
    return t;
}

void write_table(zval *out, const Table &t, bool single) {
    auto write_row = [&](zval *dst, std::size_t r) {
        array_init_size(dst, static_cast<uint32_t>(t.cols));
        for (std::size_t c = 0; c < t.cols; ++c) {
            zval v;
            write_cell(t.at(r, c), &v);
            add_next_index_zval(dst, &v);
        }
    };
    if (single) {
        write_row(out, 0);
        return;
    }
    array_init_size(out, static_cast<uint32_t>(t.rows));
    for (std::size_t r = 0; r < t.rows; ++r) {
        zval row;
        write_row(&row, r);
        add_next_index_zval(out, &row);
    }
}

// ---- calling back into PHP -------------------------------------------------------

void call_method(zval *object, const char *name, zval *retval, std::initializer_list<zval *> args) {
    zval *a = args.size() > 0 ? args.begin()[0] : nullptr;
    zval *b = args.size() > 1 ? args.begin()[1] : nullptr;
    ZVAL_UNDEF(retval);
    zend_call_method(Z_OBJ_P(object), Z_OBJCE_P(object), nullptr, name, std::strlen(name), retval,
                     static_cast<uint32_t>(args.size()), a, b);
    if (EG(exception)) {
        zval_ptr_dtor(retval);
        ZVAL_UNDEF(retval);
        throw PhpError{};
    }
}

void clone_into(zval *object, zval *out) {
    zend_object *source = Z_OBJ_P(object);
    if (!source->handlers->clone_obj) {
        throw std::invalid_argument(std::string("Objects of class ") + ZSTR_VAL(source->ce->name) + " cannot be cloned");
    }
    zend_object *copy = source->handlers->clone_obj(source);
    ZVAL_OBJ(out, copy);
    if (EG(exception)) {
        zval_ptr_dtor(out);
        ZVAL_UNDEF(out);
        throw PhpError{};
    }
}

void pick(const std::vector<zval *> &values, const std::vector<std::size_t> &indices, zval *out) {
    array_init_size(out, static_cast<uint32_t>(indices.size()));
    for (std::size_t i : indices) {
        Z_TRY_ADDREF_P(values[i]);
        add_next_index_zval(out, values[i]);
    }
}

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

void array_zval(HashTable *ht, zval *out) {
    if (GC_FLAGS(ht) & GC_IMMUTABLE) {
        ZVAL_ARR(out, ht);
        Z_TYPE_INFO_P(out) = IS_ARRAY;  // immutable arrays are never refcounted
    } else {
        GC_ADDREF(ht);
        ZVAL_ARR(out, ht);
    }
}

bool instance_of(zval *value, zend_class_entry *ce) {
    return value && Z_TYPE_P(value) == IS_OBJECT && instanceof_function(Z_OBJCE_P(value), ce);
}

void require_instance(zval *value, zend_class_entry *ce, const char *what) {
    if (!instance_of(value, ce)) {
        throw std::invalid_argument(std::string(what) + " must implement " + ZSTR_VAL(ce->name) + ", " +
                                    (value && Z_TYPE_P(value) == IS_OBJECT ? ZSTR_VAL(Z_OBJCE_P(value)->name)
                                                                            : type_name(value)) +
                                    " given");
    }
}

zval *read_prop(zval *self, const char *name) {
    zval rv;
    zval *value = zend_read_property(Z_OBJCE_P(self), Z_OBJ_P(self), name, std::strlen(name), true, &rv);
    ZVAL_DEREF(value);
    return value;
}

void write_prop(zval *self, const char *name, zval *value) {
    zend_update_property(Z_OBJCE_P(self), Z_OBJ_P(self), name, std::strlen(name), value);
}

void deep_clone_prop(zval *self, const char *name) {
    zval *value = read_prop(self, name);
    if (Z_TYPE_P(value) == IS_OBJECT) {
        zval copy;
        clone_into(value, &copy);
        write_prop(self, name, &copy);
        zval_ptr_dtor(&copy);
    } else if (Z_TYPE_P(value) == IS_ARRAY) {
        zval copy;
        array_init_size(&copy, zend_hash_num_elements(Z_ARRVAL_P(value)));
        zend_string *key;
        zend_ulong index;
        zval *item;
        ZEND_HASH_FOREACH_KEY_VAL(Z_ARRVAL_P(value), index, key, item) {
            zval element;
            if (Z_TYPE_P(item) == IS_OBJECT) {
                clone_into(item, &element);
            } else {
                ZVAL_COPY(&element, item);
            }
            if (key) {
                zend_hash_update(Z_ARRVAL(copy), key, &element);
            } else {
                zend_hash_index_update(Z_ARRVAL(copy), index, &element);
            }
        }
        ZEND_HASH_FOREACH_END();
        write_prop(self, name, &copy);
        zval_ptr_dtor(&copy);
    }
}

void instantiate(zend_class_entry *ce, HashTable *named, zval *out) {
    if (object_init_ex(out, ce) != SUCCESS || EG(exception)) {
        throw PhpError{};
    }
    zend_function *ctor = Z_OBJ_HT_P(out)->get_constructor(Z_OBJ_P(out));
    if (!ctor) {
        if (named && zend_hash_num_elements(named) > 0) {
            zval_ptr_dtor(out);
            throw std::invalid_argument(std::string(ZSTR_VAL(ce->name)) + " has no constructor parameters to set");
        }
        return;
    }
    zval retval;
    ZVAL_UNDEF(&retval);
    zend_call_known_function(ctor, Z_OBJ_P(out), Z_OBJCE_P(out), &retval, 0, nullptr, named);
    zval_ptr_dtor(&retval);
    if (EG(exception)) {
        zval_ptr_dtor(out);
        ZVAL_UNDEF(out);
        throw PhpError{};
    }
}

int required_args(zend_class_entry *ce, const char *method) {
    zend_string *lower = zend_string_init(method, std::strlen(method), false);
    zend_string *lc = zend_string_tolower(lower);
    zend_function *fn = static_cast<zend_function *>(zend_hash_find_ptr(&ce->function_table, lc));
    zend_string_release(lc);
    zend_string_release(lower);
    return fn ? static_cast<int>(fn->common.required_num_args) : -1;
}

}  // namespace mlphp
