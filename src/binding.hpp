#pragma once

// Generic PHP method handlers for native models. A class file defines a model
// wrapper type M and lists handlers such as ML_METHOD(fit, fit_classifier<M>,
// ...) in its function table, so each class only hand-writes its constructor.
//
// Wrapper requirements, by handler:
//   all:            static constexpr std::uint32_t kTag; save(ArchiveWriter&) const; load(ArchiveReader&)
//   classifiers:    fit(const Matrix&, const std::vector<int>&, size_t classes, unsigned threads);
//                   fitted(); features(); classes(); predict_one(x) -> int; LabelEncoder labels
//   probabilistic:  probabilities(x, out)
//   regressors:     fit(const Matrix&, const std::vector<double>&, unsigned threads); predict_one(x) -> double
//   unsupervised:   fit(const Matrix&, unsigned threads)
//   clusterers:     predict_one(x) -> int (a LabelEncoder `labels` of 0..k-1 enables probabilities)
//   anomaly:        score_one(x) -> double; threshold()
//   transformers:   output_dim(); transform_row(const double *in, double *out)
//   importances:    importances() -> std::vector<double>

#include "arginfo.hpp"

#define ML_METHOD(name, handler, arginfo) ZEND_FENTRY(name, (handler), arginfo, ZEND_ACC_PUBLIC)

namespace mlphp {

// ---- serialization ----------------------------------------------------------------

template <typename M>
void ZEND_FASTCALL serialize_native(INTERNAL_FUNCTION_PARAMETERS) {
    ZEND_PARSE_PARAMETERS_NONE();
    guard([&] {
        mlcore::ArchiveWriter w(M::kTag);
        native<M>(ZEND_THIS).save(w);
        array_init(return_value);
        add_assoc_long(return_value, "mlnative", mlcore::kArchiveFormat);
        add_assoc_stringl(return_value, "state", w.data().data(), w.data().size());
    });
}

template <typename M>
void ZEND_FASTCALL unserialize_native(INTERNAL_FUNCTION_PARAMETERS) {
    HashTable *data;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY_HT(data)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        zval *state = require_key(data, "state");
        if (Z_TYPE_P(state) != IS_STRING) {
            throw std::invalid_argument("Invalid serialized data: \"state\" must be a string");
        }
        M restored;
        mlcore::ArchiveReader r(Z_STRVAL_P(state), Z_STRLEN_P(state), M::kTag);
        restored.load(r);
        r.finish();
        native<M>(ZEND_THIS) = std::move(restored);
    });
}

// ---- supervised -----------------------------------------------------------------------

template <typename M>
void ZEND_FASTCALL fit_classifier(INTERNAL_FUNCTION_PARAMETERS) {
    HashTable *samples, *labels;
    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_ARRAY_HT(samples)
        Z_PARAM_ARRAY_HT(labels)
    ZEND_PARSE_PARAMETERS_END();
    if (!guard([&] {
            M &m = native<M>(ZEND_THIS);
            const mlcore::Matrix X = read_samples(samples);
            LabelEncoder encoder;
            const std::vector<int> y = encoder.fit(labels);
            mlcore::require_training_set(X, y.size());
            M next = m;  // fit into a copy so a failure keeps the old model
            next.fit(X, y, encoder.size(), threads());
            next.labels = std::move(encoder);
            m = std::move(next);
        })) {
        RETURN_THROWS();
    }
    RETURN_OBJ_COPY(Z_OBJ_P(ZEND_THIS));
}

template <typename M>
void ZEND_FASTCALL predict_classifier(INTERNAL_FUNCTION_PARAMETERS) {
    HashTable *samples;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY_HT(samples)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] { predict_labels(native<M>(ZEND_THIS), ZEND_THIS, samples, return_value); });
}

template <typename M>
void ZEND_FASTCALL predict_proba(INTERNAL_FUNCTION_PARAMETERS) {
    HashTable *samples;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY_HT(samples)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] { predict_probabilities(native<M>(ZEND_THIS), ZEND_THIS, samples, return_value); });
}

template <typename M>
void ZEND_FASTCALL get_classes(INTERNAL_FUNCTION_PARAMETERS) {
    ZEND_PARSE_PARAMETERS_NONE();
    guard([&] {
        const M &m = native<M>(ZEND_THIS);
        require_fitted(m.fitted(), ZEND_THIS);
        m.labels.write_all(return_value);
    });
}

template <typename M>
void ZEND_FASTCALL fit_regressor(INTERNAL_FUNCTION_PARAMETERS) {
    HashTable *samples, *targets;
    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_ARRAY_HT(samples)
        Z_PARAM_ARRAY_HT(targets)
    ZEND_PARSE_PARAMETERS_END();
    if (!guard([&] {
            M &m = native<M>(ZEND_THIS);
            const mlcore::Matrix X = read_samples(samples);
            const std::vector<double> y = read_numbers(targets, "Target");
            mlcore::require_training_set(X, y.size());
            M next = m;
            next.fit(X, y, threads());
            m = std::move(next);
        })) {
        RETURN_THROWS();
    }
    RETURN_OBJ_COPY(Z_OBJ_P(ZEND_THIS));
}

template <typename M>
void ZEND_FASTCALL predict_regressor(INTERNAL_FUNCTION_PARAMETERS) {
    HashTable *samples;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY_HT(samples)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        const M &m = native<M>(ZEND_THIS);
        require_fitted(m.fitted(), ZEND_THIS);
        if (is_sample_list(samples)) {
            const mlcore::Matrix X = read_samples(samples, m.features());
            std::vector<double> out(X.rows);
            mlcore::parallel_for(X.rows, threads(), [&](std::size_t r) { out[r] = m.predict_one(X.row(r)); });
            write_doubles(return_value, out);
        } else {
            const std::vector<double> x = read_sample(samples, m.features());
            RETVAL_DOUBLE(m.predict_one(x.data()));
        }
    });
}

// ---- unsupervised -----------------------------------------------------------------

template <typename M>
void ZEND_FASTCALL fit_unsupervised(INTERNAL_FUNCTION_PARAMETERS) {
    HashTable *samples;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY_HT(samples)
    ZEND_PARSE_PARAMETERS_END();
    if (!guard([&] {
            M &m = native<M>(ZEND_THIS);
            const mlcore::Matrix X = read_samples(samples);
            M next = m;
            next.fit(X, threads());
            m = std::move(next);
        })) {
        RETURN_THROWS();
    }
    RETURN_OBJ_COPY(Z_OBJ_P(ZEND_THIS));
}

template <typename M>
void ZEND_FASTCALL predict_cluster(INTERNAL_FUNCTION_PARAMETERS) {
    HashTable *samples;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY_HT(samples)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        const M &m = native<M>(ZEND_THIS);
        require_fitted(m.fitted(), ZEND_THIS);
        if (is_sample_list(samples)) {
            const mlcore::Matrix X = read_samples(samples, m.features());
            std::vector<int> out(X.rows);
            mlcore::parallel_for(X.rows, threads(), [&](std::size_t r) { out[r] = m.predict_one(X.row(r)); });
            write_ints(return_value, out);
        } else {
            const std::vector<double> x = read_sample(samples, m.features());
            RETVAL_LONG(m.predict_one(x.data()));
        }
    });
}

template <typename M>
void ZEND_FASTCALL predict_anomaly(INTERNAL_FUNCTION_PARAMETERS) {
    HashTable *samples;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY_HT(samples)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        const M &m = native<M>(ZEND_THIS);
        require_fitted(m.fitted(), ZEND_THIS);
        if (is_sample_list(samples)) {
            const mlcore::Matrix X = read_samples(samples, m.features());
            std::vector<int> out(X.rows);
            mlcore::parallel_for(X.rows, threads(),
                                 [&](std::size_t r) { out[r] = m.score_one(X.row(r)) > m.threshold() ? 1 : 0; });
            write_ints(return_value, out);
        } else {
            const std::vector<double> x = read_sample(samples, m.features());
            RETVAL_LONG(m.score_one(x.data()) > m.threshold() ? 1 : 0);
        }
    });
}

template <typename M>
void ZEND_FASTCALL score_anomaly(INTERNAL_FUNCTION_PARAMETERS) {
    HashTable *samples;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY_HT(samples)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        const M &m = native<M>(ZEND_THIS);
        require_fitted(m.fitted(), ZEND_THIS);
        if (is_sample_list(samples)) {
            const mlcore::Matrix X = read_samples(samples, m.features());
            std::vector<double> out(X.rows);
            mlcore::parallel_for(X.rows, threads(), [&](std::size_t r) { out[r] = m.score_one(X.row(r)); });
            write_doubles(return_value, out);
        } else {
            const std::vector<double> x = read_sample(samples, m.features());
            RETVAL_DOUBLE(m.score_one(x.data()));
        }
    });
}

template <typename M>
void ZEND_FASTCALL get_threshold(INTERNAL_FUNCTION_PARAMETERS) {
    ZEND_PARSE_PARAMETERS_NONE();
    guard([&] {
        const M &m = native<M>(ZEND_THIS);
        require_fitted(m.fitted(), ZEND_THIS);
        RETVAL_DOUBLE(m.threshold());
    });
}

// ---- transformers -----------------------------------------------------------------

template <typename M>
void transform_samples(const M &m, HashTable *samples, zval *return_value) {
    const std::size_t out_dim = m.output_dim();
    if (is_sample_list(samples)) {
        const mlcore::Matrix X = read_samples(samples, m.features());
        mlcore::Matrix Y(X.rows, out_dim);
        mlcore::parallel_for(X.rows, threads(), [&](std::size_t r) { m.transform_row(X.row(r), Y.row(r)); });
        write_matrix(return_value, Y);
    } else {
        const std::vector<double> x = read_sample(samples, m.features());
        std::vector<double> y(out_dim);
        m.transform_row(x.data(), y.data());
        write_doubles(return_value, y);
    }
}

template <typename M>
void ZEND_FASTCALL transform(INTERNAL_FUNCTION_PARAMETERS) {
    HashTable *samples;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY_HT(samples)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        const M &m = native<M>(ZEND_THIS);
        require_fitted(m.fitted(), ZEND_THIS);
        transform_samples(m, samples, return_value);
    });
}

template <typename M>
void ZEND_FASTCALL fit_transform(INTERNAL_FUNCTION_PARAMETERS) {
    HashTable *samples;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY_HT(samples)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        M &m = native<M>(ZEND_THIS);
        const mlcore::Matrix X = read_samples(samples);
        M next = m;
        next.fit(X, threads());
        m = std::move(next);
        mlcore::Matrix Y(X.rows, m.output_dim());
        mlcore::parallel_for(X.rows, threads(), [&](std::size_t r) { m.transform_row(X.row(r), Y.row(r)); });
        write_matrix(return_value, Y);
    });
}

template <typename M>
void ZEND_FASTCALL get_importances(INTERNAL_FUNCTION_PARAMETERS) {
    ZEND_PARSE_PARAMETERS_NONE();
    guard([&] {
        const M &m = native<M>(ZEND_THIS);
        require_fitted(m.fitted(), ZEND_THIS);
        write_doubles(return_value, m.importances());
    });
}

// ---- constructor parameter checks ---------------------------------------------

inline void check(bool ok, const char *message) {
    if (!ok) {
        throw std::invalid_argument(message);
    }
}

inline std::size_t positive(zend_long value, const char *name) {
    if (value < 1) {
        throw std::invalid_argument(std::string(name) + " must be at least 1");
    }
    return static_cast<std::size_t>(value);
}

inline std::string format_number(double v) {
    char buffer[32];
    snprintf(buffer, sizeof buffer, "%g", v);
    return buffer;
}

inline double in_range(double value, double lo, double hi, const char *name) {
    if (!(value >= lo && value <= hi)) {
        throw std::invalid_argument(std::string(name) + " must be between " + format_number(lo) + " and " + format_number(hi));
    }
    return value;
}

inline double non_negative(double value, const char *name) {
    if (!(value >= 0.0) || std::isinf(value)) {
        throw std::invalid_argument(std::string(name) + " must be greater than or equal to 0");
    }
    return value;
}

inline double strictly_positive(double value, const char *name) {
    if (!(value > 0.0) || std::isinf(value)) {
        throw std::invalid_argument(std::string(name) + " must be greater than 0");
    }
    return value;
}

inline std::string zstr(zend_string *s) { return std::string(ZSTR_VAL(s), ZSTR_LEN(s)); }

// Replaces the native model of $this with `fresh` (constructor helper).
template <typename M>
void reset(zval *self, M fresh) {
    native<M>(self) = std::move(fresh);
}

}  // namespace mlphp
