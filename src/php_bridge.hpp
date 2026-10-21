#pragma once

// Glue between the Zend engine and the PHP-independent models in src/core:
// zval <-> Matrix conversion, label encoding, the object wrapper template and
// the C++-exception-to-PHP-exception boundary.

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

extern "C" {
#include "php.h"
#include "zend_exceptions.h"
#include "zend_interfaces.h"
#include "zend_objects.h"
#include "zend_objects_API.h"
#include "ext/spl/spl_exceptions.h"
}

#include <cstring>
#include <new>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include "core/archive.hpp"
#include "core/common.hpp"

namespace mlphp {

extern zend_class_entry *ce_ml_exception;
extern zend_class_entry *ce_invalid_argument;
extern zend_class_entry *ce_runtime;
extern zend_class_entry *ce_not_fitted;

// Marker interfaces (no methods) that describe what a class can do. Meta
// estimators (Pipeline, GridSearch, ensembles, validators) rely on them.
extern zend_class_entry *ce_estimator;       // ML\Estimator: predict()
extern zend_class_entry *ce_learner;         // ML\Learner: fit()
extern zend_class_entry *ce_classifier;      // ML\Classifier
extern zend_class_entry *ce_regressor;       // ML\Regressor
extern zend_class_entry *ce_clusterer;       // ML\Clusterer
extern zend_class_entry *ce_anomaly;         // ML\AnomalyDetector: predict() + score()
extern zend_class_entry *ce_probabilistic;   // ML\Probabilistic: predictProbability()
extern zend_class_entry *ce_transformer;     // ML\Transformer: transform()
extern zend_class_entry *ce_stateful;        // ML\Stateful: transformer with fit()
extern zend_class_entry *ce_ranks_features;  // ML\RanksFeatures: getFeatureImportances()
extern zend_class_entry *ce_tokenizer;       // ML\Text\Tokenizer: tokenize()

void register_exceptions();
void register_interfaces();
void register_linear_regression();
void register_logistic_regression();
void register_knn();
void register_gaussian_nb();
void register_kmeans();
void register_standard_scaler();
void register_metrics();
void register_trees();
void register_neighbors();
void register_naive_bayes();
void register_cross_validation();
void register_preprocessing();
void register_table_transformers();
void register_datasets();
void register_distance();
void register_clustering();
void register_anomaly();
void register_boosting();
void register_ensemble();
void register_model_selection();
void register_pipeline();
void register_text();
void register_decomposition();
void register_svm();
void register_neuralnet();

// Thrown after a call back into PHP left an exception pending; guard() then
// lets that PHP exception propagate unchanged.
struct PhpError {};

// Number of worker threads for native code (ini mlnative.threads; 0 = all cores).
unsigned threads();

// Runs `body` and turns any C++ exception into the matching PHP exception.
// Every method body that touches native code goes through here: C++
// exceptions must never unwind into the (C) engine. Returns false if a PHP
// exception is now pending.
template <typename F>
bool guard(F &&body) {
    try {
        body();
    } catch (const PhpError &) {
        // the PHP exception is already pending
    } catch (const mlcore::NotFittedError &e) {
        zend_throw_exception(ce_not_fitted, e.what(), 0);
    } catch (const std::invalid_argument &e) {
        zend_throw_exception(ce_invalid_argument, e.what(), 0);
    } catch (const std::bad_alloc &) {
        zend_throw_exception(ce_runtime, "Out of memory in native code", 0);
    } catch (const std::exception &e) {
        zend_throw_exception(ce_runtime, e.what(), 0);
    }
    return !EG(exception);
}

void require_fitted(bool fitted, zval *self);

// ---- input conversion (all throw std::invalid_argument on bad input) ------

// True when `ht` is a list of samples (its first element is an array) rather
// than a single sample.
bool is_sample_list(HashTable *ht);
// A list of samples (arrays of int|float). `expected_features` of 0 accepts
// any width, but every sample must have the same number of features.
mlcore::Matrix read_samples(HashTable *ht, std::size_t expected_features = 0);
// A single sample.
std::vector<double> read_sample(HashTable *ht, std::size_t expected_features);
// A flat list of int|float; `what` names one element in error messages.
std::vector<double> read_numbers(HashTable *ht, const char *what);

// ---- output conversion -----------------------------------------------------

void write_doubles(zval *out, const std::vector<double> &values);
void write_ints(zval *out, const std::vector<int> &values);
void write_matrix(zval *out, const mlcore::Matrix &m);

// ---- class labels ----------------------------------------------------------

using Label = std::variant<zend_long, std::string>;

// Maps arbitrary int|string labels to dense class indices (in sorted label
// order) and back.
class LabelEncoder {
public:
    std::vector<Label> classes;

    // Learns the distinct labels and returns the class index of each one.
    std::vector<int> fit(HashTable *labels);
    std::size_t size() const { return classes.size(); }
    void write(int index, zval *out) const;
    // out[label] = value
    void write_keyed(zval *out, int index, double value) const;
    void write_all(zval *out) const;
    // Inverse of write_all(), used by __unserialize().
    void read_all(HashTable *ht);
    // Class index of `label`, or -1 if it was not seen during fit().
    int find(const Label &label) const;

    void save(mlcore::ArchiveWriter &w) const;
    void load(mlcore::ArchiveReader &r);
};

// One int|string label; `index` only appears in error messages.
Label read_label(zval *value, std::size_t index);
std::vector<Label> read_labels(HashTable *ht, const char *what = "Label");
void write_label(const Label &label, zval *out);
void save_label(mlcore::ArchiveWriter &w, const Label &label);
Label load_label(mlcore::ArchiveReader &r);

// ---- mixed-type tables (numbers, strings, booleans, missing values) --------

struct Cell {
    enum class Kind : std::uint8_t { Missing, Int, Float, String, Bool };
    Kind kind = Kind::Missing;
    zend_long i = 0;
    double f = 0.0;
    std::string s;

    bool numeric() const { return kind == Kind::Int || kind == Kind::Float; }
    double number() const { return kind == Kind::Int ? static_cast<double>(i) : f; }
    static Cell make_float(double v) { Cell c; c.kind = Kind::Float; c.f = v; return c; }
    static Cell make_string(std::string v) { Cell c; c.kind = Kind::String; c.s = std::move(v); return c; }
};

struct Table {
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::vector<Cell> cells;  // row-major
    Cell &at(std::size_t r, std::size_t c) { return cells[r * cols + c]; }
    const Cell &at(std::size_t r, std::size_t c) const { return cells[r * cols + c]; }
};

// Reads a list of samples (or one sample when `single` is set) whose features
// may be int, float, string, bool or null. A null value, a NAN float or a
// string equal to `placeholder` (if given) becomes Kind::Missing.
Table read_table(HashTable *ht, bool single, std::size_t expected_cols = 0, const std::string *placeholder = nullptr);
// Writes a table back; `single` writes just the first row as a flat sample.
void write_table(zval *out, const Table &t, bool single);

// ---- calling back into PHP --------------------------------------------------

// Calls $object->name(...args) and returns the result in `retval` (which the
// caller must destroy). Throws PhpError if the call threw.
void call_method(zval *object, const char *name, zval *retval, std::initializer_list<zval *> args = {});
// Returns a clone of `object` in `out` (as `clone $object` would).
void clone_into(zval *object, zval *out);
// Builds a PHP list from the elements at `indices` of `values`.
void pick(const std::vector<zval *> &values, const std::vector<std::size_t> &indices, zval *out);
std::vector<zval *> values_of(HashTable *ht);
// Wraps `ht` in a zval that holds its own reference (safe for immutable arrays);
// the caller destroys it with zval_ptr_dtor.
void array_zval(HashTable *ht, zval *out);
bool instance_of(zval *value, zend_class_entry *ce);
// Throws invalid_argument unless `value` is an object implementing `ce`.
void require_instance(zval *value, zend_class_entry *ce, const char *what);

// ---- object properties of property-based classes ---------------------------

zval *read_prop(zval *self, const char *name);
void write_prop(zval *self, const char *name, zval *value);
// Replaces an object property (or every object inside an array property) with a clone; used by __clone.
void deep_clone_prop(zval *self, const char *name);

// `new $ce(...$named)` with named constructor arguments. Throws PhpError if the constructor throws.
void instantiate(zend_class_entry *ce, HashTable *named, zval *out);
// Number of required parameters of $object->method(), or -1 if it does not exist.
int required_args(zend_class_entry *ce, const char *method);
// The class whose kind (classifier, regressor, ...) an estimator has, looking
// through wrappers such as Pipeline, BootstrapAggregator and CommitteeMachine.
zend_class_entry *underlying_class(zval *estimator);

// ---- __serialize / __unserialize helpers ----------------------------------

void put_doubles(zval *arr, const char *key, const std::vector<double> &values);
zval *require_key(HashTable *ht, const char *key);
double get_double(HashTable *ht, const char *key);
zend_long get_long(HashTable *ht, const char *key);
std::vector<double> get_doubles(HashTable *ht, const char *key);
HashTable *get_array(HashTable *ht, const char *key);
// True if total == rows * cols without overflow.
bool shape_matches(std::size_t total, std::size_t rows, std::size_t cols);

// ---- generic classifier methods -------------------------------------------
//
// Model must provide: fitted(), features(), classes(), predict_one(x),
// probabilities(x, out) (only for predict_probabilities) and a `labels`
// LabelEncoder member.

template <typename Model>
void predict_labels(const Model &m, zval *self, HashTable *samples, zval *return_value) {
    require_fitted(m.fitted(), self);
    if (is_sample_list(samples)) {
        const mlcore::Matrix X = read_samples(samples, m.features());
        std::vector<int> predicted(X.rows);
        mlcore::parallel_for(X.rows, threads(), [&](std::size_t r) { predicted[r] = m.predict_one(X.row(r)); });
        array_init_size(return_value, static_cast<uint32_t>(X.rows));
        for (int index : predicted) {
            zval label;
            m.labels.write(index, &label);
            add_next_index_zval(return_value, &label);
        }
    } else {
        const std::vector<double> x = read_sample(samples, m.features());
        m.labels.write(m.predict_one(x.data()), return_value);
    }
}

template <typename Model>
void predict_probabilities(const Model &m, zval *self, HashTable *samples, zval *return_value) {
    require_fitted(m.fitted(), self);
    const std::size_t K = m.classes();
    const bool batch = is_sample_list(samples);
    const mlcore::Matrix X = batch ? read_samples(samples, m.features()) : [&] {
        mlcore::Matrix single(1, m.features());
        single.data = read_sample(samples, m.features());
        return single;
    }();

    std::vector<double> probs(X.rows * K);
    mlcore::parallel_for(X.rows, threads(), [&](std::size_t r) { m.probabilities(X.row(r), &probs[r * K]); });

    auto write_row = [&](zval *out, std::size_t r) {
        array_init_size(out, static_cast<uint32_t>(K));
        for (std::size_t k = 0; k < K; ++k) {
            m.labels.write_keyed(out, static_cast<int>(k), probs[r * K + k]);
        }
    };
    if (!batch) {
        write_row(return_value, 0);
        return;
    }
    array_init_size(return_value, static_cast<uint32_t>(X.rows));
    for (std::size_t r = 0; r < X.rows; ++r) {
        zval row;
        write_row(&row, r);
        add_next_index_zval(return_value, &row);
    }
}

// ---- object wrapper ---------------------------------------------------------
//
// Each PHP class owns one heap-allocated native model of type T. T must be
// default-constructible and copy-assignable (copying implements `clone`).

template <typename T>
struct Object {
    T *native;
    zend_object std;  // must stay last: properties are allocated after it
};

template <typename T>
inline Object<T> *object_from(zend_object *obj) {
    return reinterpret_cast<Object<T> *>(reinterpret_cast<char *>(obj) - XtOffsetOf(Object<T>, std));
}

template <typename T>
inline T &native(zval *self) {
    return *object_from<T>(Z_OBJ_P(self))->native;
}

template <typename T>
struct Handlers {
    static zend_object_handlers table;
};
template <typename T>
zend_object_handlers Handlers<T>::table;

template <typename T>
zend_object *create_object(zend_class_entry *ce) {
    auto *intern = static_cast<Object<T> *>(zend_object_alloc(sizeof(Object<T>), ce));
    intern->native = new (std::nothrow) T();
    if (!intern->native) {
        zend_error_noreturn(E_ERROR, "Out of memory while creating %s", ZSTR_VAL(ce->name));
    }
    zend_object_std_init(&intern->std, ce);
    object_properties_init(&intern->std, ce);
    intern->std.handlers = &Handlers<T>::table;
    return &intern->std;
}

template <typename T>
void free_object(zend_object *obj) {
    Object<T> *intern = object_from<T>(obj);
    delete intern->native;
    intern->native = nullptr;
    zend_object_std_dtor(obj);
}

template <typename T>
zend_object *clone_object(zend_object *old) {
    zend_object *copy = create_object<T>(old->ce);
    try {
        *object_from<T>(copy)->native = *object_from<T>(old)->native;
    } catch (const std::bad_alloc &) {
        zend_throw_exception(ce_runtime, "Out of memory while cloning model", 0);
    }
    zend_objects_clone_members(copy, old);
    return copy;
}

// Registers a final class without native state (state lives in declared
// properties). Used for meta estimators that hold other PHP objects.
zend_class_entry *register_plain_class(const char *name, const zend_function_entry *methods,
                                       std::initializer_list<zend_class_entry *> interfaces = {});

template <typename T>
zend_class_entry *register_class(const char *name, const zend_function_entry *methods,
                                 std::initializer_list<zend_class_entry *> interfaces = {}) {
    zend_class_entry tmp;
    INIT_CLASS_ENTRY_EX(tmp, name, std::strlen(name), methods);
    zend_class_entry *ce = zend_register_internal_class(&tmp);
    ce->ce_flags |= ZEND_ACC_FINAL | ZEND_ACC_NO_DYNAMIC_PROPERTIES;
    ce->create_object = create_object<T>;
    for (zend_class_entry *iface : interfaces) {
        zend_class_implements(ce, 1, iface);
    }

    zend_object_handlers &h = Handlers<T>::table;
    std::memcpy(&h, zend_get_std_object_handlers(), sizeof(zend_object_handlers));
    h.offset = XtOffsetOf(Object<T>, std);
    h.free_obj = free_object<T>;
    h.clone_obj = clone_object<T>;
    return ce;
}

}  // namespace mlphp
