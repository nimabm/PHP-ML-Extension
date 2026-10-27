// Transformers for mixed data (numbers, strings, booleans, missing values):
// OneHotEncoder, IntervalDiscretizer, NumericStringConverter, BooleanConverter,
// MissingDataImputer, KNNImputer, HotDeckImputer and LambdaFunction.

#include <map>

#include "binding.hpp"

namespace {

using namespace mlphp;
using mlcore::ArchiveReader;
using mlcore::ArchiveWriter;

void save_cell(ArchiveWriter &w, const Cell &c) {
    w.u8(static_cast<std::uint8_t>(c.kind));
    switch (c.kind) {
        case Cell::Kind::Int:
        case Cell::Kind::Bool: w.i64(c.i); break;
        case Cell::Kind::Float: w.f64(c.f); break;
        case Cell::Kind::String: w.str(c.s); break;
        case Cell::Kind::Missing: break;
    }
}

Cell load_cell(ArchiveReader &r) {
    Cell c;
    const std::uint8_t kind = r.u8();
    ArchiveReader::require(kind <= 4, "bad cell");
    c.kind = static_cast<Cell::Kind>(kind);
    switch (c.kind) {
        case Cell::Kind::Int:
        case Cell::Kind::Bool: c.i = r.i64(); break;
        case Cell::Kind::Float: c.f = r.f64(); break;
        case Cell::Kind::String: c.s = r.str(); break;
        case Cell::Kind::Missing: break;
    }
    return c;
}

// Categories compare by kind then value; numbers and booleans as ints.
Label cell_label(const Cell &c, std::size_t r, std::size_t col) {
    switch (c.kind) {
        case Cell::Kind::String: return Label(std::in_place_index<1>, c.s);
        case Cell::Kind::Int:
        case Cell::Kind::Bool: return Label(std::in_place_index<0>, c.i);
        default:
            throw std::invalid_argument("Sample #" + std::to_string(r) + ", feature #" + std::to_string(col) +
                                        " must be a category (string, int or bool) for this column");
    }
}

std::string sample_feature(std::size_t r, std::size_t c) {
    return "Sample #" + std::to_string(r) + ", feature #" + std::to_string(c);
}

bool batch_of(HashTable *samples) { return is_sample_list(samples); }

template <typename M>
void ZEND_FASTCALL table_fit(INTERNAL_FUNCTION_PARAMETERS) {
    HashTable *samples;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY_HT(samples)
    ZEND_PARSE_PARAMETERS_END();
    if (!guard([&] {
            M &m = native<M>(ZEND_THIS);
            M next = m;
            next.fit(read_table(samples, false, 0, next.placeholder_ptr()));
            m = std::move(next);
        })) {
        RETURN_THROWS();
    }
    RETURN_OBJ_COPY(Z_OBJ_P(ZEND_THIS));
}

template <typename M>
void ZEND_FASTCALL table_transform(INTERNAL_FUNCTION_PARAMETERS) {
    HashTable *samples;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY_HT(samples)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        const M &m = native<M>(ZEND_THIS);
        require_fitted(m.fitted(), ZEND_THIS);
        const bool batch = batch_of(samples);
        Table t = read_table(samples, !batch, m.features(), m.placeholder_ptr());
        m.apply(t, return_value, !batch);
    });
}

template <typename M>
void ZEND_FASTCALL table_fit_transform(INTERNAL_FUNCTION_PARAMETERS) {
    HashTable *samples;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY_HT(samples)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        M &m = native<M>(ZEND_THIS);
        Table t = read_table(samples, false, 0, m.placeholder_ptr());
        M next = m;
        next.fit(t);
        m = std::move(next);
        m.apply(t, return_value, false);
    });
}

// ---- OneHotEncoder ------------------------------------------------------------------

struct OneHot {
    static constexpr std::uint32_t kTag = mlcore::tag("OHOT");
    std::vector<std::size_t> requested;  // explicit categorical columns; empty = auto
    bool explicit_columns = false;
    std::vector<bool> categorical;
    std::vector<std::vector<Label>> categories;

    const std::string *placeholder_ptr() const { return nullptr; }
    bool fitted() const { return !categorical.empty(); }
    std::size_t features() const { return categorical.size(); }

    void fit(const Table &t) {
        std::vector<bool> cat(t.cols, false);
        if (explicit_columns) {
            for (std::size_t c : requested) {
                check(c < t.cols, "columns contains an index beyond the number of features");
                cat[c] = true;
            }
        } else {
            for (std::size_t r = 0; r < t.rows; ++r) {
                for (std::size_t c = 0; c < t.cols; ++c) {
                    const Cell::Kind k = t.at(r, c).kind;
                    cat[c] = cat[c] || k == Cell::Kind::String || k == Cell::Kind::Bool;
                }
            }
        }
        std::vector<std::vector<Label>> cats(t.cols);
        for (std::size_t c = 0; c < t.cols; ++c) {
            if (!cat[c]) {
                continue;
            }
            for (std::size_t r = 0; r < t.rows; ++r) {
                cats[c].push_back(cell_label(t.at(r, c), r, c));
            }
            std::sort(cats[c].begin(), cats[c].end());
            cats[c].erase(std::unique(cats[c].begin(), cats[c].end()), cats[c].end());
        }
        categorical = std::move(cat);
        categories = std::move(cats);
    }

    std::size_t output_dim() const {
        std::size_t d = 0;
        for (std::size_t c = 0; c < categorical.size(); ++c) {
            d += categorical[c] ? categories[c].size() : 1;
        }
        return d;
    }

    void apply(const Table &t, zval *out, bool single) const {
        mlcore::Matrix Y(t.rows, output_dim());
        for (std::size_t r = 0; r < t.rows; ++r) {
            std::size_t j = 0;
            for (std::size_t c = 0; c < t.cols; ++c) {
                const Cell &cell = t.at(r, c);
                if (!categorical[c]) {
                    if (!cell.numeric()) {
                        throw std::invalid_argument(sample_feature(r, c) + " must be numeric (it was numeric during fit)");
                    }
                    Y(r, j++) = cell.number();
                    continue;
                }
                const auto &cats = categories[c];
                if (cell.kind != Cell::Kind::Missing && cell.kind != Cell::Kind::Float) {
                    const Label l = cell_label(cell, r, c);
                    const auto it = std::lower_bound(cats.begin(), cats.end(), l);
                    if (it != cats.end() && *it == l) {
                        Y(r, j + static_cast<std::size_t>(it - cats.begin())) = 1.0;  // unknown categories stay all-zero
                    }
                }
                j += cats.size();
            }
        }
        if (single) {
            write_doubles(out, Y.data);
        } else {
            write_matrix(out, Y);
        }
    }

    void save(ArchiveWriter &w) const {
        w.boolean(explicit_columns);
        w.sizes(requested);
        w.size(categorical.size());
        for (std::size_t c = 0; c < categorical.size(); ++c) {
            w.boolean(categorical[c]);
            w.size(categories[c].size());
            for (const Label &l : categories[c]) {
                save_label(w, l);
            }
        }
    }
    void load(ArchiveReader &r) {
        explicit_columns = r.boolean();
        requested = r.sizes();
        const std::size_t n = r.count(9);
        categorical.assign(n, false);
        categories.assign(n, {});
        for (std::size_t c = 0; c < n; ++c) {
            categorical[c] = r.boolean();
            const std::size_t k = r.count(9);
            for (std::size_t i = 0; i < k; ++i) {
                categories[c].push_back(load_label(r));
                ArchiveReader::require(i == 0 || categories[c][i - 1] < categories[c][i], "categories must be sorted");
            }
        }
    }
};

ZEND_BEGIN_ARG_INFO_EX(arginfo_one_hot_construct, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, columns, IS_ARRAY, 1, "null")
ZEND_END_ARG_INFO()

void ZEND_FASTCALL one_hot_construct(INTERNAL_FUNCTION_PARAMETERS) {
    HashTable *columns = nullptr;
    ZEND_PARSE_PARAMETERS_START(0, 1)
        Z_PARAM_OPTIONAL
        Z_PARAM_ARRAY_HT_OR_NULL(columns)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        OneHot fresh;
        if (columns) {
            fresh.explicit_columns = true;
            for (double c : read_numbers(columns, "Column")) {
                check(c >= 0 && c == std::floor(c), "columns must be non-negative column indices");
                fresh.requested.push_back(static_cast<std::size_t>(c));
            }
        }
        reset(ZEND_THIS, std::move(fresh));
    });
}

void ZEND_FASTCALL one_hot_categories(INTERNAL_FUNCTION_PARAMETERS) {
    ZEND_PARSE_PARAMETERS_NONE();
    guard([&] {
        const OneHot &m = native<OneHot>(ZEND_THIS);
        require_fitted(m.fitted(), ZEND_THIS);
        array_init(return_value);
        for (std::size_t c = 0; c < m.categorical.size(); ++c) {
            if (!m.categorical[c]) {
                continue;
            }
            zval list;
            array_init(&list);
            for (const Label &l : m.categories[c]) {
                zval v;
                write_label(l, &v);
                add_next_index_zval(&list, &v);
            }
            add_index_zval(return_value, static_cast<zend_ulong>(c), &list);
        }
    });
}

const zend_function_entry one_hot_methods[] = {
    ML_METHOD(__construct, one_hot_construct, arginfo_one_hot_construct)
    ML_METHOD(fit, table_fit<OneHot>, arginfo_ml_fit_unsupervised)
    ML_METHOD(transform, table_transform<OneHot>, arginfo_ml_samples_to_array)
    ML_METHOD(fitTransform, table_fit_transform<OneHot>, arginfo_ml_samples_to_array)
    ML_METHOD(getCategories, one_hot_categories, arginfo_ml_get_array)
    ML_METHOD(__serialize, serialize_native<OneHot>, arginfo_ml_get_array)
    ML_METHOD(__unserialize, unserialize_native<OneHot>, arginfo_ml_unserialize)
    PHP_FE_END};

// ---- IntervalDiscretizer ---------------------------------------------------------------

std::string bin_name(std::size_t index) {
    std::string name;
    do {
        name.insert(name.begin(), static_cast<char>('a' + index % 26));
        index /= 26;
    } while (index-- > 0);
    return name;
}

struct Discretizer {
    static constexpr std::uint32_t kTag = mlcore::tag("IDSC");
    std::size_t bins = 5;
    bool equi_width = true;
    std::vector<bool> numeric;
    std::vector<std::vector<double>> edges;  // inner bin boundaries per numeric column

    const std::string *placeholder_ptr() const { return nullptr; }
    bool fitted() const { return !numeric.empty(); }
    std::size_t features() const { return numeric.size(); }

    void fit(const Table &t) {
        std::vector<bool> num(t.cols, true);
        std::vector<std::vector<double>> e(t.cols);
        for (std::size_t c = 0; c < t.cols; ++c) {
            std::vector<double> values;
            for (std::size_t r = 0; r < t.rows; ++r) {
                const Cell &cell = t.at(r, c);
                if (!cell.numeric()) {
                    num[c] = false;
                    break;
                }
                values.push_back(cell.number());
            }
            if (!num[c]) {
                continue;
            }
            std::sort(values.begin(), values.end());
            for (std::size_t b = 1; b < bins; ++b) {
                const double q = static_cast<double>(b) / static_cast<double>(bins);
                e[c].push_back(equi_width ? values.front() + q * (values.back() - values.front())
                                          : mlcore::quantile(values, q));
            }
        }
        numeric = std::move(num);
        edges = std::move(e);
    }

    void apply(Table &t, zval *out, bool single) const {
        for (std::size_t r = 0; r < t.rows; ++r) {
            for (std::size_t c = 0; c < t.cols; ++c) {
                Cell &cell = t.at(r, c);
                if (!numeric[c]) {
                    continue;
                }
                if (!cell.numeric()) {
                    throw std::invalid_argument(sample_feature(r, c) + " must be numeric (it was numeric during fit)");
                }
                const auto &e = edges[c];
                const std::size_t bin = static_cast<std::size_t>(std::upper_bound(e.begin(), e.end(), cell.number()) - e.begin());
                cell = Cell::make_string(bin_name(bin));
            }
        }
        write_table(out, t, single);
    }

    void save(ArchiveWriter &w) const {
        w.size(bins);
        w.boolean(equi_width);
        w.size(numeric.size());
        for (std::size_t c = 0; c < numeric.size(); ++c) {
            w.boolean(numeric[c]);
            w.doubles(edges[c]);
        }
    }
    void load(ArchiveReader &r) {
        bins = r.size();
        equi_width = r.boolean();
        const std::size_t n = r.count(9);
        numeric.assign(n, false);
        edges.assign(n, {});
        for (std::size_t c = 0; c < n; ++c) {
            numeric[c] = r.boolean();
            edges[c] = r.doubles();
        }
    }
};

ZEND_BEGIN_ARG_INFO_EX(arginfo_discretizer_construct, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, bins, IS_LONG, 0, "5")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, equiWidth, _IS_BOOL, 0, "true")
ZEND_END_ARG_INFO()

void ZEND_FASTCALL discretizer_construct(INTERNAL_FUNCTION_PARAMETERS) {
    zend_long bins = 5;
    bool equi_width = true;
    ZEND_PARSE_PARAMETERS_START(0, 2)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(bins)
        Z_PARAM_BOOL(equi_width)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        check(bins >= 2, "bins must be at least 2");
        Discretizer fresh;
        fresh.bins = static_cast<std::size_t>(bins);
        fresh.equi_width = equi_width;
        reset(ZEND_THIS, std::move(fresh));
    });
}

const zend_function_entry discretizer_methods[] = {
    ML_METHOD(__construct, discretizer_construct, arginfo_discretizer_construct)
    ML_METHOD(fit, table_fit<Discretizer>, arginfo_ml_fit_unsupervised)
    ML_METHOD(transform, table_transform<Discretizer>, arginfo_ml_samples_to_array)
    ML_METHOD(fitTransform, table_fit_transform<Discretizer>, arginfo_ml_samples_to_array)
    ML_METHOD(__serialize, serialize_native<Discretizer>, arginfo_ml_get_array)
    ML_METHOD(__unserialize, unserialize_native<Discretizer>, arginfo_ml_unserialize)
    PHP_FE_END};

// ---- NumericStringConverter and BooleanConverter (stateless) -----------------------

struct NumericStrings {
    static constexpr std::uint32_t kTag = mlcore::tag("NSTR");
    const std::string *placeholder_ptr() const { return nullptr; }
    bool fitted() const { return true; }
    std::size_t features() const { return 0; }
    void apply(Table &t, zval *out, bool single) const {
        for (Cell &cell : t.cells) {
            if (cell.kind != Cell::Kind::String) {
                continue;
            }
            if (cell.s == "NAN" || cell.s == "NaN") {
                cell = Cell::make_float(std::numeric_limits<double>::quiet_NaN());
                continue;
            }
            zend_long l;
            double d;
            const zend_uchar type = is_numeric_string(cell.s.data(), cell.s.size(), &l, &d, false);
            if (type == IS_LONG) {
                cell.kind = Cell::Kind::Int;
                cell.i = l;
            } else if (type == IS_DOUBLE) {
                cell = Cell::make_float(d);
            }
        }
        write_table(out, t, single);
    }
    void save(ArchiveWriter &) const {}
    void load(ArchiveReader &) {}
};

struct BooleanConv {
    static constexpr std::uint32_t kTag = mlcore::tag("BOOL");
    Cell when_true = Cell::make_string("true");
    Cell when_false = Cell::make_string("false");
    const std::string *placeholder_ptr() const { return nullptr; }
    bool fitted() const { return true; }
    std::size_t features() const { return 0; }
    void apply(Table &t, zval *out, bool single) const {
        for (Cell &cell : t.cells) {
            if (cell.kind == Cell::Kind::Bool) {
                cell = cell.i ? when_true : when_false;
            }
        }
        write_table(out, t, single);
    }
    void save(ArchiveWriter &w) const {
        save_cell(w, when_true);
        save_cell(w, when_false);
    }
    void load(ArchiveReader &r) {
        when_true = load_cell(r);
        when_false = load_cell(r);
    }
};

ZEND_BEGIN_ARG_INFO_EX(arginfo_boolean_construct, 0, 0, 0)
    ZEND_ARG_TYPE_MASK(0, trueValue, MAY_BE_LONG | MAY_BE_STRING, "\"true\"")
    ZEND_ARG_TYPE_MASK(0, falseValue, MAY_BE_LONG | MAY_BE_STRING, "\"false\"")
ZEND_END_ARG_INFO()

void ZEND_FASTCALL boolean_construct(INTERNAL_FUNCTION_PARAMETERS) {
    zval *t = nullptr, *f = nullptr;
    ZEND_PARSE_PARAMETERS_START(0, 2)
        Z_PARAM_OPTIONAL
        Z_PARAM_ZVAL(t)
        Z_PARAM_ZVAL(f)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        BooleanConv fresh;
        auto to_cell = [](zval *v) {
            Cell c;
            if (Z_TYPE_P(v) == IS_LONG) {
                c.kind = Cell::Kind::Int;
                c.i = Z_LVAL_P(v);
            } else if (Z_TYPE_P(v) == IS_STRING) {
                c = Cell::make_string(std::string(Z_STRVAL_P(v), Z_STRLEN_P(v)));
            } else {
                throw std::invalid_argument("trueValue and falseValue must be int or string");
            }
            return c;
        };
        if (t) {
            fresh.when_true = to_cell(t);
        }
        if (f) {
            fresh.when_false = to_cell(f);
        }
        reset(ZEND_THIS, std::move(fresh));
    });
}

template <typename M>
void ZEND_FASTCALL stateless_table_transform(INTERNAL_FUNCTION_PARAMETERS) {
    HashTable *samples;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY_HT(samples)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        const M &m = native<M>(ZEND_THIS);
        const bool batch = batch_of(samples);
        Table t = read_table(samples, !batch, 0, nullptr);
        m.apply(t, return_value, !batch);
    });
}

void ZEND_FASTCALL noop_construct(INTERNAL_FUNCTION_PARAMETERS) { ZEND_PARSE_PARAMETERS_NONE(); }

const zend_function_entry numeric_string_methods[] = {
    ML_METHOD(__construct, noop_construct, arginfo_ml_none)
    ML_METHOD(transform, stateless_table_transform<NumericStrings>, arginfo_ml_samples_to_array)
    PHP_FE_END};

const zend_function_entry boolean_methods[] = {
    ML_METHOD(__construct, boolean_construct, arginfo_boolean_construct)
    ML_METHOD(transform, stateless_table_transform<BooleanConv>, arginfo_ml_samples_to_array)
    ML_METHOD(__serialize, serialize_native<BooleanConv>, arginfo_ml_get_array)
    ML_METHOD(__unserialize, unserialize_native<BooleanConv>, arginfo_ml_unserialize)
    PHP_FE_END};

// ---- imputers ---------------------------------------------------------------------------

// Column statistics shared by the imputers.
struct ColumnInfo {
    bool numeric = true;
    double lo = 0, hi = 0;
};

std::vector<ColumnInfo> column_info(const Table &t) {
    std::vector<ColumnInfo> info(t.cols);
    for (std::size_t c = 0; c < t.cols; ++c) {
        bool any = false;
        double lo = std::numeric_limits<double>::infinity(), hi = -lo;
        bool numeric = true;
        for (std::size_t r = 0; r < t.rows; ++r) {
            const Cell &cell = t.at(r, c);
            if (cell.kind == Cell::Kind::Missing) {
                continue;
            }
            any = true;
            if (cell.numeric()) {
                lo = std::min(lo, cell.number());
                hi = std::max(hi, cell.number());
            } else {
                numeric = false;
            }
        }
        if (!any) {
            throw std::invalid_argument("Feature #" + std::to_string(c) + " has no non-missing values to learn from");
        }
        info[c] = {numeric, numeric ? lo : 0.0, numeric ? hi : 0.0};
    }
    return info;
}

void save_table(ArchiveWriter &w, const Table &t) {
    w.size(t.rows);
    w.size(t.cols);
    for (const Cell &c : t.cells) {
        save_cell(w, c);
    }
}

Table load_table(ArchiveReader &r) {
    Table t;
    t.rows = r.size();
    t.cols = r.size();
    ArchiveReader::require(t.cols == 0 || t.rows <= (1u << 30) / std::max<std::size_t>(t.cols, 1));
    const std::size_t n = t.rows * t.cols;
    t.cells.reserve(std::min<std::size_t>(n, 1u << 20));
    for (std::size_t i = 0; i < n; ++i) {
        t.cells.push_back(load_cell(r));
    }
    return t;
}

void save_info(ArchiveWriter &w, const std::vector<ColumnInfo> &info) {
    w.size(info.size());
    for (const ColumnInfo &c : info) {
        w.boolean(c.numeric);
        w.f64(c.lo);
        w.f64(c.hi);
    }
}

std::vector<ColumnInfo> load_info(ArchiveReader &r) {
    std::vector<ColumnInfo> info(r.count(17));
    for (ColumnInfo &c : info) {
        c.numeric = r.boolean();
        c.lo = r.f64();
        c.hi = r.f64();
    }
    return info;
}

// Strategy specs: continuous mean | median | percentile(<p>) | constant(<v>) | wild_guess;
// categorical most_frequent | k_most_frequent(<k>) | prior | constant(<value>).
struct Strategy {
    std::string name;
    std::string arg;

    static Strategy parse(const std::string &spec, bool categorical) {
        Strategy s;
        const auto open = spec.find('(');
        if (open != std::string::npos) {
            check(spec.back() == ')', "Malformed imputer strategy");
            s.name = spec.substr(0, open);
            s.arg = spec.substr(open + 1, spec.size() - open - 2);
        } else {
            s.name = spec;
        }
        const bool ok = categorical ? (s.name == "most_frequent" || s.name == "k_most_frequent" || s.name == "prior" || s.name == "constant")
                                    : (s.name == "mean" || s.name == "median" || s.name == "percentile" || s.name == "constant" ||
                                       s.name == "wild_guess");
        if (!ok) {
            throw std::invalid_argument(categorical
                ? "Unknown categorical strategy \"" + spec + "\"; expected most_frequent, k_most_frequent(<k>), prior or constant(<value>)"
                : "Unknown continuous strategy \"" + spec + "\"; expected mean, median, percentile(<p>), constant(<value>) or wild_guess");
        }
        const bool needs_arg = s.name == "percentile" || s.name == "constant" || s.name == "k_most_frequent";
        check(needs_arg == !s.arg.empty() || s.name == "constant", "Malformed imputer strategy argument");
        return s;
    }
    std::string spec() const { return arg.empty() && name != "constant" ? name : name + "(" + arg + ")"; }
    double number_arg() const {
        char *end = nullptr;
        const double v = std::strtod(arg.c_str(), &end);
        check(end == arg.c_str() + arg.size() && !arg.empty(), "Strategy argument must be a number");
        return v;
    }
};

// What the imputer learned per column: a fixed value or a weighted pool to sample from.
struct Fill {
    std::vector<Cell> pool;
    std::vector<double> weights;  // cumulative; empty = always pool[0]
    double lo = 0, hi = 0;
    bool wild = false;
};

struct MissingImputer {
    static constexpr std::uint32_t kTag = mlcore::tag("MIMP");
    Strategy continuous{"mean", ""};
    Strategy categorical{"most_frequent", ""};
    std::string placeholder = "?";
    bool has_seed = false;
    std::uint64_t seed = 0;
    std::vector<Fill> fills;

    const std::string *placeholder_ptr() const { return &placeholder; }
    bool fitted() const { return !fills.empty(); }
    std::size_t features() const { return fills.size(); }

    void fit(const Table &t) {
        const std::vector<ColumnInfo> info = column_info(t);
        std::vector<Fill> out(t.cols);
        for (std::size_t c = 0; c < t.cols; ++c) {
            Fill &f = out[c];
            if (info[c].numeric) {
                std::vector<double> values;
                for (std::size_t r = 0; r < t.rows; ++r) {
                    if (t.at(r, c).numeric()) {
                        values.push_back(t.at(r, c).number());
                    }
                }
                const std::string &s = continuous.name;
                double v = 0.0;
                if (s == "mean") {
                    for (double x : values) v += x;
                    v /= static_cast<double>(values.size());
                } else if (s == "median") {
                    v = mlcore::median(values);
                } else if (s == "percentile") {
                    v = mlcore::quantile(values, in_range(continuous.number_arg(), 0, 100, "percentile") / 100.0);
                } else if (s == "constant") {
                    v = continuous.number_arg();
                } else {
                    f.wild = true;
                    f.lo = info[c].lo;
                    f.hi = info[c].hi;
                }
                f.pool.push_back(Cell::make_float(v));
                continue;
            }
            std::map<Label, double> counts;
            std::map<Label, Cell> example;
            for (std::size_t r = 0; r < t.rows; ++r) {
                const Cell &cell = t.at(r, c);
                if (cell.kind == Cell::Kind::Missing) {
                    continue;
                }
                const Label l = cell.kind == Cell::Kind::String ? Label(std::in_place_index<1>, cell.s)
                                                                 : Label(std::in_place_index<0>, cell.kind == Cell::Kind::Float ? static_cast<zend_long>(cell.f) : cell.i);
                counts[l] += 1.0;
                example.emplace(l, cell);
            }
            std::vector<std::pair<double, Label>> ranked;
            for (const auto &[l, n] : counts) {
                ranked.emplace_back(n, l);
            }
            std::stable_sort(ranked.begin(), ranked.end(), [](const auto &a, const auto &b) { return a.first > b.first; });
            const std::string &s = categorical.name;
            if (s == "constant") {
                f.pool.push_back(Cell::make_string(categorical.arg));
            } else if (s == "most_frequent") {
                f.pool.push_back(example[ranked[0].second]);
            } else {
                const std::size_t k = s == "prior" ? ranked.size() : std::min<std::size_t>(ranked.size(), static_cast<std::size_t>(std::max(1.0, categorical.number_arg())));
                double acc = 0.0;
                for (std::size_t i = 0; i < k; ++i) {
                    f.pool.push_back(example[ranked[i].second]);
                    acc += s == "prior" ? ranked[i].first : 1.0;
                    f.weights.push_back(acc);
                }
            }
        }
        fills = std::move(out);
    }

    void apply(Table &t, zval *out, bool single) const {
        mlcore::Rng rng = mlcore::make_rng(has_seed, seed);
        for (std::size_t r = 0; r < t.rows; ++r) {
            for (std::size_t c = 0; c < t.cols; ++c) {
                Cell &cell = t.at(r, c);
                if (cell.kind != Cell::Kind::Missing) {
                    continue;
                }
                const Fill &f = fills[c];
                if (f.wild) {
                    cell = Cell::make_float(std::uniform_real_distribution<double>(f.lo, f.hi > f.lo ? f.hi : f.lo + 1e-12)(rng));
                } else if (f.weights.empty()) {
                    cell = f.pool[0];
                } else {
                    const double u = std::uniform_real_distribution<double>(0.0, f.weights.back())(rng);
                    const std::size_t i = std::min<std::size_t>(f.pool.size() - 1,
                        static_cast<std::size_t>(std::upper_bound(f.weights.begin(), f.weights.end(), u) - f.weights.begin()));
                    cell = f.pool[i];
                }
            }
        }
        write_table(out, t, single);
    }

    void save(ArchiveWriter &w) const {
        w.str(continuous.spec());
        w.str(categorical.spec());
        w.str(placeholder);
        w.boolean(has_seed);
        w.u64(seed);
        w.size(fills.size());
        for (const Fill &f : fills) {
            w.size(f.pool.size());
            for (const Cell &c : f.pool) {
                save_cell(w, c);
            }
            w.doubles(f.weights);
            w.f64(f.lo);
            w.f64(f.hi);
            w.boolean(f.wild);
        }
    }
    void load(ArchiveReader &r) {
        try {
            continuous = Strategy::parse(r.str(), false);
            categorical = Strategy::parse(r.str(), true);
        } catch (const std::invalid_argument &) {
            ArchiveReader::fail("bad strategy");
        }
        placeholder = r.str();
        has_seed = r.boolean();
        seed = r.u64();
        fills.assign(r.count(1), {});
        for (Fill &f : fills) {
            const std::size_t n = r.count(1);
            for (std::size_t i = 0; i < n; ++i) {
                f.pool.push_back(load_cell(r));
            }
            f.weights = r.doubles();
            f.lo = r.f64();
            f.hi = r.f64();
            f.wild = r.boolean();
            ArchiveReader::require(!f.pool.empty() && (f.weights.empty() || f.weights.size() == f.pool.size()), "bad fill");
        }
    }
};

ZEND_BEGIN_ARG_INFO_EX(arginfo_missing_construct, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, continuous, IS_STRING, 0, "\"mean\"")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, categorical, IS_STRING, 0, "\"most_frequent\"")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, placeholder, IS_STRING, 0, "\"?\"")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, seed, IS_LONG, 1, "null")
ZEND_END_ARG_INFO()

void ZEND_FASTCALL missing_construct(INTERNAL_FUNCTION_PARAMETERS) {
    zend_string *continuous = nullptr, *categorical = nullptr, *placeholder = nullptr;
    zend_long seed = 0;
    bool seed_null = true;
    ZEND_PARSE_PARAMETERS_START(0, 4)
        Z_PARAM_OPTIONAL
        Z_PARAM_STR(continuous)
        Z_PARAM_STR(categorical)
        Z_PARAM_STR(placeholder)
        Z_PARAM_LONG_OR_NULL(seed, seed_null)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        MissingImputer fresh;
        if (continuous) {
            fresh.continuous = Strategy::parse(zstr(continuous), false);
            if (fresh.continuous.name == "percentile") {
                in_range(fresh.continuous.number_arg(), 0, 100, "percentile");
            } else if (fresh.continuous.name == "constant") {
                fresh.continuous.number_arg();
            }
        }
        if (categorical) {
            fresh.categorical = Strategy::parse(zstr(categorical), true);
            if (fresh.categorical.name == "k_most_frequent") {
                check(fresh.categorical.number_arg() >= 1, "k_most_frequent needs k >= 1");
            }
        }
        if (placeholder) {
            fresh.placeholder = zstr(placeholder);
        }
        parse_seed(seed_null, seed, fresh.has_seed, fresh.seed);
        reset(ZEND_THIS, std::move(fresh));
    });
}

const zend_function_entry missing_methods[] = {
    ML_METHOD(__construct, missing_construct, arginfo_missing_construct)
    ML_METHOD(fit, table_fit<MissingImputer>, arginfo_ml_fit_unsupervised)
    ML_METHOD(transform, table_transform<MissingImputer>, arginfo_ml_samples_to_array)
    ML_METHOD(fitTransform, table_fit_transform<MissingImputer>, arginfo_ml_samples_to_array)
    ML_METHOD(__serialize, serialize_native<MissingImputer>, arginfo_ml_get_array)
    ML_METHOD(__unserialize, unserialize_native<MissingImputer>, arginfo_ml_unserialize)
    PHP_FE_END};

// ---- KNNImputer and HotDeckImputer ------------------------------------------------------

// Gower-style distance between a row with gaps and a complete donor row.
double donor_distance(const Table &t, std::size_t r, const Table &donors, std::size_t d, const std::vector<ColumnInfo> &info) {
    double sum = 0.0;
    std::size_t used = 0;
    for (std::size_t c = 0; c < t.cols; ++c) {
        const Cell &a = t.at(r, c);
        if (a.kind == Cell::Kind::Missing) {
            continue;
        }
        const Cell &b = donors.at(d, c);
        ++used;
        if (info[c].numeric && a.numeric() && b.numeric()) {
            const double range = info[c].hi - info[c].lo;
            sum += range > 0 ? std::min(1.0, std::fabs(a.number() - b.number()) / range) : (a.number() == b.number() ? 0.0 : 1.0);
        } else {
            sum += (a.kind == b.kind && a.s == b.s && a.i == b.i && a.f == b.f) ? 0.0 : 1.0;
        }
    }
    return used == 0 ? 1.0 : sum / static_cast<double>(used);
}

template <bool HotDeck>
struct NeighborImputer {
    static constexpr std::uint32_t kTag = HotDeck ? mlcore::tag("HDIM") : mlcore::tag("KNIM");
    std::size_t k = 5;
    bool weighted = false;
    std::string placeholder = "?";
    bool has_seed = false;
    std::uint64_t seed = 0;
    Table donors;  // complete training rows
    std::vector<ColumnInfo> info;

    const std::string *placeholder_ptr() const { return &placeholder; }
    bool fitted() const { return donors.rows > 0; }
    std::size_t features() const { return donors.cols; }

    void fit(const Table &t) {
        std::vector<ColumnInfo> columns = column_info(t);
        Table complete;
        complete.cols = t.cols;
        for (std::size_t r = 0; r < t.rows; ++r) {
            bool full = true;
            for (std::size_t c = 0; c < t.cols; ++c) {
                full = full && t.at(r, c).kind != Cell::Kind::Missing;
            }
            if (full) {
                complete.cells.insert(complete.cells.end(), t.cells.begin() + static_cast<std::ptrdiff_t>(r * t.cols),
                                      t.cells.begin() + static_cast<std::ptrdiff_t>((r + 1) * t.cols));
                ++complete.rows;
            }
        }
        check(complete.rows > 0, "Need at least one sample without missing values to use as a donor");
        donors = std::move(complete);
        info = std::move(columns);
    }

    void apply(Table &t, zval *out, bool single) const {
        mlcore::Rng rng = mlcore::make_rng(has_seed, seed);
        std::vector<std::pair<double, std::size_t>> nearest;
        for (std::size_t r = 0; r < t.rows; ++r) {
            bool gaps = false;
            for (std::size_t c = 0; c < t.cols; ++c) {
                gaps = gaps || t.at(r, c).kind == Cell::Kind::Missing;
            }
            if (!gaps) {
                continue;
            }
            nearest.clear();
            for (std::size_t d = 0; d < donors.rows; ++d) {
                nearest.emplace_back(donor_distance(t, r, donors, d, info), d);
            }
            const std::size_t kk = std::min(k, nearest.size());
            std::partial_sort(nearest.begin(), nearest.begin() + static_cast<std::ptrdiff_t>(kk), nearest.end());
            nearest.resize(kk);
            std::vector<double> w(kk);
            for (std::size_t i = 0; i < kk; ++i) {
                w[i] = weighted ? 1.0 / (nearest[i].first + 1e-9) : 1.0;
            }
            for (std::size_t c = 0; c < t.cols; ++c) {
                Cell &cell = t.at(r, c);
                if (cell.kind != Cell::Kind::Missing) {
                    continue;
                }
                if (HotDeck) {
                    std::discrete_distribution<std::size_t> pick(w.begin(), w.end());
                    cell = donors.at(nearest[pick(rng)].second, c);
                } else if (info[c].numeric) {
                    double sum = 0.0, total = 0.0;
                    for (std::size_t i = 0; i < kk; ++i) {
                        sum += w[i] * donors.at(nearest[i].second, c).number();
                        total += w[i];
                    }
                    cell = Cell::make_float(sum / total);
                } else {
                    std::map<std::string, std::pair<double, std::size_t>> votes;  // key -> (weight, donor)
                    for (std::size_t i = 0; i < kk; ++i) {
                        const Cell &v = donors.at(nearest[i].second, c);
                        const std::string key = std::to_string(static_cast<int>(v.kind)) + ":" + v.s + ":" + std::to_string(v.i);
                        auto &entry = votes[key];
                        if (entry.first == 0.0) {
                            entry.second = nearest[i].second;
                        }
                        entry.first += w[i];
                    }
                    auto best = votes.begin();
                    for (auto it = votes.begin(); it != votes.end(); ++it) {
                        if (it->second.first > best->second.first) {
                            best = it;
                        }
                    }
                    cell = donors.at(best->second.second, c);
                }
            }
        }
        write_table(out, t, single);
    }

    void save(ArchiveWriter &w) const {
        w.size(k);
        w.boolean(weighted);
        w.str(placeholder);
        w.boolean(has_seed);
        w.u64(seed);
        save_table(w, donors);
        save_info(w, info);
    }
    void load(ArchiveReader &r) {
        k = r.size();
        weighted = r.boolean();
        placeholder = r.str();
        has_seed = r.boolean();
        seed = r.u64();
        donors = load_table(r);
        info = load_info(r);
        ArchiveReader::require(k >= 1 && info.size() == donors.cols);
        for (std::size_t i = 0; i < donors.cells.size(); ++i) {
            const Cell &cell = donors.cells[i];
            ArchiveReader::require(cell.kind != Cell::Kind::Missing, "donor with missing value");
            ArchiveReader::require(!info[i % donors.cols].numeric || cell.numeric(), "non-numeric donor value");
        }
    }
};

using KNNImputer = NeighborImputer<false>;
using HotDeckImputer = NeighborImputer<true>;

ZEND_BEGIN_ARG_INFO_EX(arginfo_neighbor_imputer_construct, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, k, IS_LONG, 0, "5")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, weighted, _IS_BOOL, 0, "false")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, placeholder, IS_STRING, 0, "\"?\"")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, seed, IS_LONG, 1, "null")
ZEND_END_ARG_INFO()

template <typename M>
void ZEND_FASTCALL neighbor_imputer_construct(INTERNAL_FUNCTION_PARAMETERS) {
    zend_long k = 5, seed = 0;
    bool weighted = false, seed_null = true;
    zend_string *placeholder = nullptr;
    ZEND_PARSE_PARAMETERS_START(0, 4)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(k)
        Z_PARAM_BOOL(weighted)
        Z_PARAM_STR(placeholder)
        Z_PARAM_LONG_OR_NULL(seed, seed_null)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        M fresh;
        fresh.k = positive(k, "k");
        fresh.weighted = weighted;
        if (placeholder) {
            fresh.placeholder = zstr(placeholder);
        }
        parse_seed(seed_null, seed, fresh.has_seed, fresh.seed);
        reset(ZEND_THIS, std::move(fresh));
    });
}

#define ML_IMPUTER(M)                                                              \
    ML_METHOD(__construct, neighbor_imputer_construct<M>, arginfo_neighbor_imputer_construct) \
    ML_METHOD(fit, table_fit<M>, arginfo_ml_fit_unsupervised)                       \
    ML_METHOD(transform, table_transform<M>, arginfo_ml_samples_to_array)           \
    ML_METHOD(fitTransform, table_fit_transform<M>, arginfo_ml_samples_to_array)    \
    ML_METHOD(__serialize, serialize_native<M>, arginfo_ml_get_array)               \
    ML_METHOD(__unserialize, unserialize_native<M>, arginfo_ml_unserialize)

const zend_function_entry knn_imputer_methods[] = {ML_IMPUTER(KNNImputer) PHP_FE_END};
const zend_function_entry hot_deck_methods[] = {ML_IMPUTER(HotDeckImputer) PHP_FE_END};

// ---- LambdaFunction (property-based) ----------------------------------------------------

ZEND_BEGIN_ARG_INFO_EX(arginfo_lambda_construct, 0, 0, 1)
    ZEND_ARG_TYPE_INFO(0, callback, IS_CALLABLE, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, context, IS_MIXED, 0, "null")
ZEND_END_ARG_INFO()

void ZEND_FASTCALL lambda_construct(INTERNAL_FUNCTION_PARAMETERS) {
    zval *callback, *context = nullptr;
    ZEND_PARSE_PARAMETERS_START(1, 2)
        Z_PARAM_ZVAL(callback)
        Z_PARAM_OPTIONAL
        Z_PARAM_ZVAL(context)
    ZEND_PARSE_PARAMETERS_END();
    if (!zend_is_callable(callback, 0, nullptr)) {
        zend_argument_type_error(1, "must be a valid callback");
        RETURN_THROWS();
    }
    write_prop(ZEND_THIS, "callback", callback);
    if (context) {
        write_prop(ZEND_THIS, "context", context);
    }
}

void ZEND_FASTCALL lambda_transform(INTERNAL_FUNCTION_PARAMETERS) {
    HashTable *samples;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY_HT(samples)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        zval *callback = read_prop(ZEND_THIS, "callback");
        zval *context = read_prop(ZEND_THIS, "context");
        auto call = [&](zval *sample, zend_long index, zval *result) {
            zval args[3];
            ZVAL_COPY_VALUE(&args[0], sample);
            ZVAL_LONG(&args[1], index);
            ZVAL_COPY_VALUE(&args[2], context);
            if (call_user_function(nullptr, nullptr, callback, result, 3, args) != SUCCESS || EG(exception)) {
                zval_ptr_dtor(result);
                throw PhpError{};
            }
        };
        if (!is_sample_list(samples)) {
            zval sample;
            array_zval(samples, &sample);
            try {
                call(&sample, 0, return_value);
            } catch (...) {
                zval_ptr_dtor(&sample);
                throw;
            }
            zval_ptr_dtor(&sample);
            return;
        }
        array_init_size(return_value, zend_hash_num_elements(samples));
        zend_long i = 0;
        zval *sample;
        ZEND_HASH_FOREACH_VAL(samples, sample) {
            zval result;
            call(sample, i++, &result);
            add_next_index_zval(return_value, &result);
        }
        ZEND_HASH_FOREACH_END();
    });
}

const zend_function_entry lambda_methods[] = {
    ML_METHOD(__construct, lambda_construct, arginfo_lambda_construct)
    ML_METHOD(transform, lambda_transform, arginfo_ml_samples_to_array)
    PHP_FE_END};

}  // namespace

void mlphp::register_table_transformers() {
    const auto stateful = {ce_transformer, ce_stateful};
    register_class<OneHot>("ML\\Preprocessing\\OneHotEncoder", one_hot_methods, stateful);
    register_class<Discretizer>("ML\\Preprocessing\\IntervalDiscretizer", discretizer_methods, stateful);
    register_class<NumericStrings>("ML\\Preprocessing\\NumericStringConverter", numeric_string_methods, {ce_transformer});
    register_class<BooleanConv>("ML\\Preprocessing\\BooleanConverter", boolean_methods, {ce_transformer});
    register_class<MissingImputer>("ML\\Preprocessing\\MissingDataImputer", missing_methods, stateful);
    register_class<KNNImputer>("ML\\Preprocessing\\KNNImputer", knn_imputer_methods, stateful);
    register_class<HotDeckImputer>("ML\\Preprocessing\\HotDeckImputer", hot_deck_methods, stateful);

    zend_class_entry *lambda = register_plain_class("ML\\Preprocessing\\LambdaFunction", lambda_methods, {ce_transformer});
    zend_declare_property_null(lambda, "callback", sizeof("callback") - 1, ZEND_ACC_PRIVATE);
    zend_declare_property_null(lambda, "context", sizeof("context") - 1, ZEND_ACC_PRIVATE);
}
