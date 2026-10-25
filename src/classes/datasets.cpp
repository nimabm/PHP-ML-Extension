// ML\Datasets (synthetic data generators and splitting) and ML\Distance.

#include <numeric>

#include "binding.hpp"
#include "core/distance.hpp"

namespace {

using namespace mlphp;

constexpr double kPi = 3.14159265358979323846;

mlcore::Rng rng_for(bool seed_null, zend_long seed) {
    return mlcore::make_rng(!seed_null, static_cast<std::uint64_t>(seed));
}

void put_samples_and(zval *out, const mlcore::Matrix &X, const char *key, zval *second) {
    zval samples;
    write_matrix(&samples, X);
    array_init(out);
    add_assoc_zval(out, "samples", &samples);
    add_assoc_zval(out, key, second);
}

// ---- generators ---------------------------------------------------------------------

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_blob, 0, 1, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO(0, center, IS_ARRAY, 0)
    ZEND_ARG_TYPE_MASK(0, stddev, MAY_BE_DOUBLE | MAY_BE_ARRAY, "1.0")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, n, IS_LONG, 0, "100")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, seed, IS_LONG, 1, "null")
ZEND_END_ARG_INFO()

void ZEND_FASTCALL datasets_blob(INTERNAL_FUNCTION_PARAMETERS) {
    HashTable *center;
    zval *stddev = nullptr;
    zend_long n = 100, seed = 0;
    bool seed_null = true;
    ZEND_PARSE_PARAMETERS_START(1, 4)
        Z_PARAM_ARRAY_HT(center)
        Z_PARAM_OPTIONAL
        Z_PARAM_ZVAL(stddev)
        Z_PARAM_LONG(n)
        Z_PARAM_LONG_OR_NULL(seed, seed_null)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        const std::vector<double> c = read_sample(center, zend_hash_num_elements(center));
        std::vector<double> sd(c.size(), 1.0);
        if (stddev && Z_TYPE_P(stddev) == IS_ARRAY) {
            sd = read_sample(Z_ARRVAL_P(stddev), c.size());
        } else if (stddev) {
            sd.assign(c.size(), zval_get_double(stddev));
        }
        for (double s : sd) {
            non_negative(s, "stddev");
        }
        mlcore::Rng rng = rng_for(seed_null, seed);
        std::normal_distribution<double> normal(0.0, 1.0);
        mlcore::Matrix X(positive(n, "n"), c.size());
        for (std::size_t r = 0; r < X.rows; ++r) {
            for (std::size_t j = 0; j < c.size(); ++j) {
                X(r, j) = c[j] + sd[j] * normal(rng);
            }
        }
        write_matrix(return_value, X);
    });
}

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_circle, 0, 0, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, x, IS_DOUBLE, 0, "0.0")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, y, IS_DOUBLE, 0, "0.0")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, scale, IS_DOUBLE, 0, "1.0")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, noise, IS_DOUBLE, 0, "0.1")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, n, IS_LONG, 0, "100")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, seed, IS_LONG, 1, "null")
ZEND_END_ARG_INFO()

void ZEND_FASTCALL datasets_circle(INTERNAL_FUNCTION_PARAMETERS) {
    double x = 0, y = 0, scale = 1, noise = 0.1;
    zend_long n = 100, seed = 0;
    bool seed_null = true;
    ZEND_PARSE_PARAMETERS_START(0, 6)
        Z_PARAM_OPTIONAL
        Z_PARAM_DOUBLE(x)
        Z_PARAM_DOUBLE(y)
        Z_PARAM_DOUBLE(scale)
        Z_PARAM_DOUBLE(noise)
        Z_PARAM_LONG(n)
        Z_PARAM_LONG_OR_NULL(seed, seed_null)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        non_negative(noise, "noise");
        mlcore::Rng rng = rng_for(seed_null, seed);
        std::uniform_real_distribution<double> angle(0.0, 2.0 * kPi);
        std::normal_distribution<double> normal(0.0, noise * scale);
        mlcore::Matrix X(positive(n, "n"), 2);
        for (std::size_t r = 0; r < X.rows; ++r) {
            const double a = angle(rng);
            X(r, 0) = x + scale * std::cos(a) + (noise > 0 ? normal(rng) : 0.0);
            X(r, 1) = y + scale * std::sin(a) + (noise > 0 ? normal(rng) : 0.0);
        }
        write_matrix(return_value, X);
    });
}

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_half_moon, 0, 0, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, x, IS_DOUBLE, 0, "0.0")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, y, IS_DOUBLE, 0, "0.0")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, scale, IS_DOUBLE, 0, "1.0")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, rotation, IS_DOUBLE, 0, "90.0")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, noise, IS_DOUBLE, 0, "0.1")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, n, IS_LONG, 0, "100")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, seed, IS_LONG, 1, "null")
ZEND_END_ARG_INFO()

void ZEND_FASTCALL datasets_half_moon(INTERNAL_FUNCTION_PARAMETERS) {
    double x = 0, y = 0, scale = 1, rotation = 90, noise = 0.1;
    zend_long n = 100, seed = 0;
    bool seed_null = true;
    ZEND_PARSE_PARAMETERS_START(0, 7)
        Z_PARAM_OPTIONAL
        Z_PARAM_DOUBLE(x)
        Z_PARAM_DOUBLE(y)
        Z_PARAM_DOUBLE(scale)
        Z_PARAM_DOUBLE(rotation)
        Z_PARAM_DOUBLE(noise)
        Z_PARAM_LONG(n)
        Z_PARAM_LONG_OR_NULL(seed, seed_null)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        non_negative(noise, "noise");
        mlcore::Rng rng = rng_for(seed_null, seed);
        std::uniform_real_distribution<double> angle(0.0, kPi);
        std::normal_distribution<double> normal(0.0, noise * scale);
        const double rot = rotation * kPi / 180.0;
        mlcore::Matrix X(positive(n, "n"), 2);
        for (std::size_t r = 0; r < X.rows; ++r) {
            const double a = angle(rng) + rot;
            X(r, 0) = x + scale * std::cos(a) + (noise > 0 ? normal(rng) : 0.0);
            X(r, 1) = y + scale * std::sin(a) + (noise > 0 ? normal(rng) : 0.0);
        }
        write_matrix(return_value, X);
    });
}

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_swiss_roll, 0, 0, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, x, IS_DOUBLE, 0, "0.0")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, y, IS_DOUBLE, 0, "0.0")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, z, IS_DOUBLE, 0, "0.0")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, scale, IS_DOUBLE, 0, "1.0")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, depth, IS_DOUBLE, 0, "21.0")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, noise, IS_DOUBLE, 0, "0.1")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, n, IS_LONG, 0, "100")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, seed, IS_LONG, 1, "null")
ZEND_END_ARG_INFO()

void ZEND_FASTCALL datasets_swiss_roll(INTERNAL_FUNCTION_PARAMETERS) {
    double x = 0, y = 0, z = 0, scale = 1, depth = 21, noise = 0.1;
    zend_long n = 100, seed = 0;
    bool seed_null = true;
    ZEND_PARSE_PARAMETERS_START(0, 8)
        Z_PARAM_OPTIONAL
        Z_PARAM_DOUBLE(x)
        Z_PARAM_DOUBLE(y)
        Z_PARAM_DOUBLE(z)
        Z_PARAM_DOUBLE(scale)
        Z_PARAM_DOUBLE(depth)
        Z_PARAM_DOUBLE(noise)
        Z_PARAM_LONG(n)
        Z_PARAM_LONG_OR_NULL(seed, seed_null)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        non_negative(noise, "noise");
        mlcore::Rng rng = rng_for(seed_null, seed);
        std::uniform_real_distribution<double> unit(0.0, 1.0);
        std::normal_distribution<double> normal(0.0, 1.0);
        mlcore::Matrix X(positive(n, "n"), 3);
        std::vector<double> t(X.rows);
        for (std::size_t r = 0; r < X.rows; ++r) {
            t[r] = 1.5 * kPi * (1.0 + 2.0 * unit(rng));
            X(r, 0) = x + scale * (t[r] * std::cos(t[r]) + noise * normal(rng));
            X(r, 1) = y + scale * (depth * unit(rng) + noise * normal(rng));
            X(r, 2) = z + scale * (t[r] * std::sin(t[r]) + noise * normal(rng));
        }
        zval targets;
        write_doubles(&targets, t);
        put_samples_and(return_value, X, "targets", &targets);
    });
}

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_hyperplane, 0, 1, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO(0, coefficients, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, intercept, IS_DOUBLE, 0, "0.0")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, noise, IS_DOUBLE, 0, "0.1")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, n, IS_LONG, 0, "100")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, seed, IS_LONG, 1, "null")
ZEND_END_ARG_INFO()

void ZEND_FASTCALL datasets_hyperplane(INTERNAL_FUNCTION_PARAMETERS) {
    HashTable *coefficients;
    double intercept = 0, noise = 0.1;
    zend_long n = 100, seed = 0;
    bool seed_null = true;
    ZEND_PARSE_PARAMETERS_START(1, 5)
        Z_PARAM_ARRAY_HT(coefficients)
        Z_PARAM_OPTIONAL
        Z_PARAM_DOUBLE(intercept)
        Z_PARAM_DOUBLE(noise)
        Z_PARAM_LONG(n)
        Z_PARAM_LONG_OR_NULL(seed, seed_null)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        const std::vector<double> w = read_sample(coefficients, zend_hash_num_elements(coefficients));
        non_negative(noise, "noise");
        mlcore::Rng rng = rng_for(seed_null, seed);
        std::uniform_real_distribution<double> uniform(-1.0, 1.0);
        std::normal_distribution<double> normal(0.0, 1.0);
        mlcore::Matrix X(positive(n, "n"), w.size());
        std::vector<double> y(X.rows);
        for (std::size_t r = 0; r < X.rows; ++r) {
            for (std::size_t j = 0; j < w.size(); ++j) {
                X(r, j) = uniform(rng);
            }
            y[r] = intercept + mlcore::dot(w.data(), X.row(r), w.size()) + noise * normal(rng);
        }
        zval targets;
        write_doubles(&targets, y);
        put_samples_and(return_value, X, "targets", &targets);
    });
}

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_agglomerate, 0, 1, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO(0, groups, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, shuffle, _IS_BOOL, 0, "true")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, seed, IS_LONG, 1, "null")
ZEND_END_ARG_INFO()

void ZEND_FASTCALL datasets_agglomerate(INTERNAL_FUNCTION_PARAMETERS) {
    HashTable *groups;
    bool shuffle = true, seed_null = true;
    zend_long seed = 0;
    ZEND_PARSE_PARAMETERS_START(1, 3)
        Z_PARAM_ARRAY_HT(groups)
        Z_PARAM_OPTIONAL
        Z_PARAM_BOOL(shuffle)
        Z_PARAM_LONG_OR_NULL(seed, seed_null)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        std::vector<zval *> samples;
        std::vector<zval> labels;
        zend_string *key;
        zend_ulong index;
        zval *group;
        ZEND_HASH_FOREACH_KEY_VAL(groups, index, key, group) {
            ZVAL_DEREF(group);
            check(Z_TYPE_P(group) == IS_ARRAY, "groups must map labels to lists of samples");
            zval *sample;
            ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(group), sample) {
                ZVAL_DEREF(sample);
                samples.push_back(sample);
                zval label;
                if (key) {
                    ZVAL_STR_COPY(&label, key);
                } else {
                    ZVAL_LONG(&label, static_cast<zend_long>(index));
                }
                labels.push_back(label);
            }
            ZEND_HASH_FOREACH_END();
        }
        ZEND_HASH_FOREACH_END();
        std::vector<std::size_t> order(samples.size());
        std::iota(order.begin(), order.end(), 0);
        if (shuffle) {
            mlcore::Rng rng = rng_for(seed_null, seed);
            std::shuffle(order.begin(), order.end(), rng);
        }
        zval out_samples, out_labels;
        array_init_size(&out_samples, static_cast<uint32_t>(order.size()));
        array_init_size(&out_labels, static_cast<uint32_t>(order.size()));
        for (std::size_t i : order) {
            Z_TRY_ADDREF_P(samples[i]);
            add_next_index_zval(&out_samples, samples[i]);
            Z_TRY_ADDREF(labels[i]);
            add_next_index_zval(&out_labels, &labels[i]);
        }
        for (zval &l : labels) {
            zval_ptr_dtor(&l);
        }
        array_init(return_value);
        add_assoc_zval(return_value, "samples", &out_samples);
        add_assoc_zval(return_value, "labels", &out_labels);
    });
}

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_split, 0, 2, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO(0, samples, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO(0, targets, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, testRatio, IS_DOUBLE, 0, "0.25")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, stratify, _IS_BOOL, 0, "false")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, seed, IS_LONG, 1, "null")
ZEND_END_ARG_INFO()

void ZEND_FASTCALL datasets_split(INTERNAL_FUNCTION_PARAMETERS) {
    HashTable *samples, *targets;
    double ratio = 0.25;
    bool stratify = false, seed_null = true;
    zend_long seed = 0;
    ZEND_PARSE_PARAMETERS_START(2, 5)
        Z_PARAM_ARRAY_HT(samples)
        Z_PARAM_ARRAY_HT(targets)
        Z_PARAM_OPTIONAL
        Z_PARAM_DOUBLE(ratio)
        Z_PARAM_BOOL(stratify)
        Z_PARAM_LONG_OR_NULL(seed, seed_null)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        check(ratio > 0.0 && ratio < 1.0, "testRatio must be between 0 and 1 (exclusive)");
        const std::vector<zval *> X = values_of(samples);
        const std::vector<zval *> Y = values_of(targets);
        check(X.size() >= 2, "Need at least two samples to split");
        if (X.size() != Y.size()) {
            throw std::invalid_argument("Number of samples (" + std::to_string(X.size()) +
                                        ") does not match number of targets (" + std::to_string(Y.size()) + ")");
        }
        std::vector<std::vector<std::size_t>> groups;
        if (stratify) {
            LabelEncoder encoder;
            const std::vector<int> y = encoder.fit(targets);
            groups.resize(encoder.size());
            for (std::size_t i = 0; i < y.size(); ++i) {
                groups[static_cast<std::size_t>(y[i])].push_back(i);
            }
        } else {
            groups.emplace_back(X.size());
            std::iota(groups[0].begin(), groups[0].end(), 0);
        }
        mlcore::Rng rng = rng_for(seed_null, seed);
        std::vector<std::size_t> train, test;
        for (auto &g : groups) {
            std::shuffle(g.begin(), g.end(), rng);
            const std::size_t n_test = std::min(g.size() - (g.size() > 1 ? 1 : 0),
                                                static_cast<std::size_t>(std::llround(ratio * static_cast<double>(g.size()))));
            test.insert(test.end(), g.begin(), g.begin() + static_cast<std::ptrdiff_t>(n_test));
            train.insert(train.end(), g.begin() + static_cast<std::ptrdiff_t>(n_test), g.end());
        }
        check(!train.empty() && !test.empty(), "testRatio leaves an empty training or testing set");
        zval a, b, c, d;
        pick(X, train, &a);
        pick(X, test, &b);
        pick(Y, train, &c);
        pick(Y, test, &d);
        array_init_size(return_value, 4);
        add_next_index_zval(return_value, &a);
        add_next_index_zval(return_value, &b);
        add_next_index_zval(return_value, &c);
        add_next_index_zval(return_value, &d);
    });
}

#define ML_STATIC(name, handler, arginfo) ZEND_FENTRY(name, (handler), arginfo, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)

const zend_function_entry dataset_methods[] = {
    ML_STATIC(blob, datasets_blob, arginfo_blob)
    ML_STATIC(circle, datasets_circle, arginfo_circle)
    ML_STATIC(halfMoon, datasets_half_moon, arginfo_half_moon)
    ML_STATIC(swissRoll, datasets_swiss_roll, arginfo_swiss_roll)
    ML_STATIC(hyperplane, datasets_hyperplane, arginfo_hyperplane)
    ML_STATIC(agglomerate, datasets_agglomerate, arginfo_agglomerate)
    ML_STATIC(trainTestSplit, datasets_split, arginfo_split)
    PHP_FE_END};

// ---- ML\Distance ------------------------------------------------------------------------

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_distance, 0, 3, IS_DOUBLE, 0)
    ZEND_ARG_TYPE_INFO(0, metric, IS_STRING, 0)
    ZEND_ARG_TYPE_INFO(0, a, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO(0, b, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_pairwise, 0, 2, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO(0, metric, IS_STRING, 0)
    ZEND_ARG_TYPE_INFO(0, samples, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_gower, 0, 3, IS_DOUBLE, 0)
    ZEND_ARG_TYPE_INFO(0, a, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO(0, b, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO(0, ranges, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

void ZEND_FASTCALL distance_compute(INTERNAL_FUNCTION_PARAMETERS) {
    zend_string *metric;
    HashTable *a, *b;
    ZEND_PARSE_PARAMETERS_START(3, 3)
        Z_PARAM_STR(metric)
        Z_PARAM_ARRAY_HT(a)
        Z_PARAM_ARRAY_HT(b)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        const mlcore::Distance d = mlcore::Distance::parse(zstr(metric));
        const std::vector<double> x = read_sample(a, zend_hash_num_elements(a));
        const std::vector<double> y = read_sample(b, x.size());
        RETVAL_DOUBLE(d(x.data(), y.data(), x.size()));
    });
}

void ZEND_FASTCALL distance_pairwise(INTERNAL_FUNCTION_PARAMETERS) {
    zend_string *metric;
    HashTable *samples;
    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_STR(metric)
        Z_PARAM_ARRAY_HT(samples)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        const mlcore::Distance d = mlcore::Distance::parse(zstr(metric));
        const mlcore::Matrix X = read_samples(samples);
        mlcore::Matrix D(X.rows, X.rows);
        mlcore::parallel_for(X.rows, threads(), [&](std::size_t i) {
            for (std::size_t j = 0; j < X.rows; ++j) {
                D(i, j) = d(X.row(i), X.row(j), X.cols);
            }
        });
        write_matrix(return_value, D);
    });
}

// Gower distance for mixed data: numbers contribute |a-b|/range, other values
// contribute 0 if equal and 1 otherwise; missing values (null/NAN) are skipped.
void ZEND_FASTCALL distance_gower(INTERNAL_FUNCTION_PARAMETERS) {
    HashTable *a, *b, *ranges;
    ZEND_PARSE_PARAMETERS_START(3, 3)
        Z_PARAM_ARRAY_HT(a)
        Z_PARAM_ARRAY_HT(b)
        Z_PARAM_ARRAY_HT(ranges)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        const Table x = read_table(a, true);
        const Table y = read_table(b, true, x.cols);
        const std::vector<double> r = read_sample(ranges, x.cols);
        double sum = 0.0;
        std::size_t used = 0;
        for (std::size_t j = 0; j < x.cols; ++j) {
            const Cell &u = x.at(0, j), &v = y.at(0, j);
            if (u.kind == Cell::Kind::Missing || v.kind == Cell::Kind::Missing) {
                continue;
            }
            ++used;
            if (u.numeric() && v.numeric()) {
                sum += r[j] > 0.0 ? std::min(1.0, std::fabs(u.number() - v.number()) / r[j]) : (u.number() == v.number() ? 0.0 : 1.0);
            } else {
                const bool same = u.kind == v.kind && u.s == v.s && u.i == v.i;
                sum += same ? 0.0 : 1.0;
            }
        }
        RETVAL_DOUBLE(used == 0 ? 0.0 : sum / static_cast<double>(used));
    });
}

const zend_function_entry distance_methods[] = {
    ML_STATIC(compute, distance_compute, arginfo_distance)
    ML_STATIC(pairwise, distance_pairwise, arginfo_pairwise)
    ML_STATIC(gower, distance_gower, arginfo_gower)
    PHP_FE_END};

}  // namespace

void mlphp::register_datasets() {
    zend_class_entry tmp;
    INIT_CLASS_ENTRY(tmp, "ML\\Datasets", dataset_methods);
    zend_register_internal_class(&tmp)->ce_flags |= ZEND_ACC_FINAL;
}

void mlphp::register_distance() {
    zend_class_entry tmp;
    INIT_CLASS_ENTRY(tmp, "ML\\Distance", distance_methods);
    zend_register_internal_class(&tmp)->ce_flags |= ZEND_ACC_FINAL;
}
