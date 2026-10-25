// Recommender systems: MatrixFactorization, ImplicitALS and KNNRecommender.

#include <numeric>

#include "binding.hpp"
#include "core/recommender.hpp"

namespace {

using namespace mlphp;
using mlcore::ArchiveReader;
using mlcore::ArchiveWriter;
using mlcore::kUnknown;

template <typename Core, std::uint32_t Tag>
struct RecommenderModel {
    static constexpr std::uint32_t kTag = Tag;
    Core core;
    LabelEncoder users, items;

    bool fitted() const { return core.fitted(); }

    std::size_t user_index(const Label &label) const {
        const int i = users.find(label);
        return i < 0 ? kUnknown : static_cast<std::size_t>(i);
    }
    std::size_t item_index(const Label &label) const {
        const int i = items.find(label);
        return i < 0 ? kUnknown : static_cast<std::size_t>(i);
    }

    void save(ArchiveWriter &w) const {
        core.save(w);
        users.save(w);
        items.save(w);
    }
    void load(ArchiveReader &r) {
        core.load(r);
        users.load(r);
        items.load(r);
        ArchiveReader::require(users.size() == core.data.users && items.size() == core.data.items, "label count mismatch");
    }
};

using MFModel = RecommenderModel<mlcore::MatrixFactorization, mlcore::tag("RCMF")>;
using ImplicitModel = RecommenderModel<mlcore::ImplicitALS, mlcore::tag("RCIA")>;
using KNNModel = RecommenderModel<mlcore::KNNRecommender, mlcore::tag("RCKN")>;

// Sorted distinct labels and the index of each input label.
std::vector<std::size_t> encode(const std::vector<Label> &labels, LabelEncoder &encoder) {
    std::vector<Label> distinct = labels;
    std::sort(distinct.begin(), distinct.end());
    distinct.erase(std::unique(distinct.begin(), distinct.end()), distinct.end());
    encoder.classes = std::move(distinct);
    std::vector<std::size_t> out;
    out.reserve(labels.size());
    for (const Label &label : labels) out.push_back(static_cast<std::size_t>(encoder.find(label)));
    return out;
}

// [user, item] or [user, item, value] rows.
mlcore::Interactions read_interactions(HashTable *ht, bool sum_duplicates, LabelEncoder &users, LabelEncoder &items) {
    std::vector<Label> user_labels, item_labels;
    std::vector<double> values;
    std::size_t index = 0;
    zval *row;
    ZEND_HASH_FOREACH_VAL(ht, row) {
        ZVAL_DEREF(row);
        const std::string subject = "Interaction #" + std::to_string(index);
        if (Z_TYPE_P(row) != IS_ARRAY) {
            throw std::invalid_argument(subject + " must be an array [user, item] or [user, item, value]");
        }
        const std::vector<zval *> cells = values_of(Z_ARRVAL_P(row));
        if (cells.size() != 2 && cells.size() != 3) {
            throw std::invalid_argument(subject + " must have 2 or 3 elements, " + std::to_string(cells.size()) + " given");
        }
        user_labels.push_back(read_label(cells[0], index));
        item_labels.push_back(read_label(cells[1], index));
        double value = 1.0;
        if (cells.size() == 3) {
            if (Z_TYPE_P(cells[2]) == IS_LONG) {
                value = static_cast<double>(Z_LVAL_P(cells[2]));
            } else if (Z_TYPE_P(cells[2]) == IS_DOUBLE && std::isfinite(Z_DVAL_P(cells[2]))) {
                value = Z_DVAL_P(cells[2]);
            } else {
                throw std::invalid_argument(subject + ": the value must be a finite int or float");
            }
        }
        values.push_back(value);
        ++index;
    }
    ZEND_HASH_FOREACH_END();
    if (values.empty()) {
        throw std::invalid_argument("Interactions must not be empty");
    }
    const std::vector<std::size_t> u = encode(user_labels, users), i = encode(item_labels, items);
    return mlcore::Interactions::build(u, i, values, users.size(), items.size(), sum_duplicates);
}

template <typename M, bool SumDuplicates>
void ZEND_FASTCALL recommender_fit(INTERNAL_FUNCTION_PARAMETERS) {
    HashTable *interactions;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY_HT(interactions)
    ZEND_PARSE_PARAMETERS_END();
    if (!guard([&] {
            M &m = native<M>(ZEND_THIS);
            M next = m;
            mlcore::Interactions data = read_interactions(interactions, SumDuplicates, next.users, next.items);
            next.core.fit(std::move(data), threads());
            m = std::move(next);
        })) {
        RETURN_THROWS();
    }
    RETURN_OBJ_COPY(Z_OBJ_P(ZEND_THIS));
}

template <typename M>
double predict_pair(const M &m, HashTable *pair, std::size_t index) {
    const std::vector<zval *> cells = values_of(pair);
    if (cells.size() != 2) {
        throw std::invalid_argument("Pair #" + std::to_string(index) + " must be [user, item]");
    }
    return m.core.predict(m.user_index(read_label(cells[0], index)), m.item_index(read_label(cells[1], index)));
}

template <typename M>
void ZEND_FASTCALL recommender_predict(INTERNAL_FUNCTION_PARAMETERS) {
    HashTable *pairs;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY_HT(pairs)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        const M &m = native<M>(ZEND_THIS);
        require_fitted(m.fitted(), ZEND_THIS);
        if (!is_sample_list(pairs)) {
            RETVAL_DOUBLE(predict_pair(m, pairs, 0));
            return;
        }
        std::vector<double> out;
        std::size_t index = 0;
        for (zval *pair : values_of(pairs)) {
            if (Z_TYPE_P(pair) != IS_ARRAY) {
                throw std::invalid_argument("Pair #" + std::to_string(index) + " must be [user, item]");
            }
            out.push_back(predict_pair(m, Z_ARRVAL_P(pair), index++));
        }
        write_doubles(return_value, out);
    });
}

// The best `n` items by score (ties: more popular first, then label order) as [item => score].
template <typename M>
void write_ranking(const M &m, const std::vector<double> &scores, const std::vector<char> &excluded, std::size_t n,
                   bool positive_only, zval *out) {
    std::vector<std::size_t> candidates;
    for (std::size_t i = 0; i < scores.size(); ++i) {
        if (!excluded[i] && (!positive_only || scores[i] > 0.0)) candidates.push_back(i);
    }
    const auto better = [&](std::size_t a, std::size_t b) {
        if (scores[a] != scores[b]) return scores[a] > scores[b];
        const std::size_t pa = m.core.data.popularity(a), pb = m.core.data.popularity(b);
        return pa != pb ? pa > pb : a < b;
    };
    const std::size_t keep = std::min(n, candidates.size());
    std::partial_sort(candidates.begin(), candidates.begin() + static_cast<std::ptrdiff_t>(keep), candidates.end(), better);
    array_init_size(out, static_cast<uint32_t>(keep));
    for (std::size_t k = 0; k < keep; ++k) {
        m.items.write_keyed(out, static_cast<int>(candidates[k]), scores[candidates[k]]);
    }
}

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_recommend, 0, 1, IS_ARRAY, 0)
    ZEND_ARG_TYPE_MASK(0, user, MAY_BE_LONG | MAY_BE_STRING, NULL)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, n, IS_LONG, 0, "10")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, excludeSeen, _IS_BOOL, 0, "true")
ZEND_END_ARG_INFO()

template <typename M>
void ZEND_FASTCALL recommender_recommend(INTERNAL_FUNCTION_PARAMETERS) {
    zval *user;
    zend_long n = 10;
    bool exclude_seen = true;
    ZEND_PARSE_PARAMETERS_START(1, 3)
        Z_PARAM_ZVAL(user)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(n)
        Z_PARAM_BOOL(exclude_seen)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        const M &m = native<M>(ZEND_THIS);
        require_fitted(m.fitted(), ZEND_THIS);
        const std::size_t u = m.user_index(read_label(user, 0));
        std::vector<double> scores;
        m.core.scores(u, scores);
        std::vector<char> excluded(scores.size(), 0);
        if (exclude_seen && u != kUnknown) {
            const mlcore::Interactions &d = m.core.data;
            for (std::size_t p = d.user_start[u]; p < d.user_start[u + 1]; ++p) excluded[d.user_items[p]] = 1;
        }
        write_ranking(m, scores, excluded, positive(n, "n"), false, return_value);
    });
}

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_similar_items, 0, 1, IS_ARRAY, 0)
    ZEND_ARG_TYPE_MASK(0, item, MAY_BE_LONG | MAY_BE_STRING, NULL)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, n, IS_LONG, 0, "10")
ZEND_END_ARG_INFO()

template <typename M>
void ZEND_FASTCALL recommender_similar_items(INTERNAL_FUNCTION_PARAMETERS) {
    zval *item;
    zend_long n = 10;
    ZEND_PARSE_PARAMETERS_START(1, 2)
        Z_PARAM_ZVAL(item)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(n)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        const M &m = native<M>(ZEND_THIS);
        require_fitted(m.fitted(), ZEND_THIS);
        const Label label = read_label(item, 0);
        const std::size_t i = m.item_index(label);
        if (i == kUnknown) {
            const std::string text = std::holds_alternative<zend_long>(label) ? std::to_string(std::get<zend_long>(label))
                                                                              : std::get<std::string>(label);
            throw std::invalid_argument("Unknown item \"" + text + "\"");
        }
        std::vector<double> similarities;
        m.core.item_similarities(i, similarities);
        std::vector<char> excluded(similarities.size(), 0);
        excluded[i] = 1;
        write_ranking(m, similarities, excluded, positive(n, "n"), true, return_value);
    });
}

template <typename M>
void ZEND_FASTCALL recommender_users(INTERNAL_FUNCTION_PARAMETERS) {
    ZEND_PARSE_PARAMETERS_NONE();
    fitted_getter<M>(ZEND_THIS, return_value, [](const M &m, zval *out) { m.users.write_all(out); });
}

template <typename M>
void ZEND_FASTCALL recommender_items(INTERNAL_FUNCTION_PARAMETERS) {
    ZEND_PARSE_PARAMETERS_NONE();
    fitted_getter<M>(ZEND_THIS, return_value, [](const M &m, zval *out) { m.items.write_all(out); });
}

// [label => factor vector] for each row of F.
void write_factors(zval *out, const mlcore::Matrix &F, const LabelEncoder &labels) {
    array_init_size(out, static_cast<uint32_t>(F.rows));
    for (std::size_t r = 0; r < F.rows; ++r) {
        zval row, key;
        write_doubles(&row, std::vector<double>(F.row(r), F.row(r) + F.cols));
        labels.write(static_cast<int>(r), &key);
        array_set_zval_key(Z_ARRVAL_P(out), &key, &row);
        zval_ptr_dtor(&row);
        zval_ptr_dtor(&key);
    }
}

template <typename M>
void ZEND_FASTCALL recommender_user_factors(INTERNAL_FUNCTION_PARAMETERS) {
    ZEND_PARSE_PARAMETERS_NONE();
    fitted_getter<M>(ZEND_THIS, return_value, [](const M &m, zval *out) { write_factors(out, m.core.user_factors, m.users); });
}

template <typename M>
void ZEND_FASTCALL recommender_item_factors(INTERNAL_FUNCTION_PARAMETERS) {
    ZEND_PARSE_PARAMETERS_NONE();
    fitted_getter<M>(ZEND_THIS, return_value, [](const M &m, zval *out) { write_factors(out, m.core.item_factors, m.items); });
}

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_MASK_EX(arginfo_recommender_fit, 0, 1, MAY_BE_STATIC)
    ZEND_ARG_TYPE_INFO(0, interactions, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_MASK_EX(arginfo_recommender_predict, 0, 1, MAY_BE_ARRAY | MAY_BE_DOUBLE)
    ZEND_ARG_TYPE_INFO(0, pairs, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

#define ML_RECOMMENDER(M, sum)                                                             \
    ML_METHOD(fit, (recommender_fit<M, sum>), arginfo_recommender_fit)                     \
    ML_METHOD(predict, recommender_predict<M>, arginfo_recommender_predict)                \
    ML_METHOD(recommend, recommender_recommend<M>, arginfo_recommend)                      \
    ML_METHOD(similarItems, recommender_similar_items<M>, arginfo_similar_items)           \
    ML_METHOD(getUsers, recommender_users<M>, arginfo_ml_get_array)                        \
    ML_METHOD(getItems, recommender_items<M>, arginfo_ml_get_array)                        \
    ML_METHOD(__serialize, serialize_native<M>, arginfo_ml_get_array)                      \
    ML_METHOD(__unserialize, unserialize_native<M>, arginfo_ml_unserialize)

// ---- MatrixFactorization ------------------------------------------------------------------

ZEND_BEGIN_ARG_INFO_EX(arginfo_mf_construct, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, factors, IS_LONG, 0, "10")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, l2, IS_DOUBLE, 0, "0.1")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, epochs, IS_LONG, 0, "15")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, seed, IS_LONG, 1, "null")
ZEND_END_ARG_INFO()

void ZEND_FASTCALL mf_construct(INTERNAL_FUNCTION_PARAMETERS) {
    zend_long factors = 10, epochs = 15, seed = 0;
    double l2 = 0.1;
    bool seed_null = true;
    ZEND_PARSE_PARAMETERS_START(0, 4)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(factors)
        Z_PARAM_DOUBLE(l2)
        Z_PARAM_LONG(epochs)
        Z_PARAM_LONG_OR_NULL(seed, seed_null)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        MFModel fresh;
        fresh.core.factors = positive(factors, "factors");
        fresh.core.l2 = strictly_positive(l2, "l2");
        fresh.core.epochs = positive(epochs, "epochs");
        parse_seed(seed_null, seed, fresh.core.has_seed, fresh.core.seed);
        reset(ZEND_THIS, std::move(fresh));
    });
}

void ZEND_FASTCALL mf_losses(INTERNAL_FUNCTION_PARAMETERS) {
    ZEND_PARSE_PARAMETERS_NONE();
    fitted_getter<MFModel>(ZEND_THIS, return_value, [](const MFModel &m, zval *out) { write_doubles(out, m.core.losses); });
}

const zend_function_entry mf_methods[] = {
    ML_METHOD(__construct, mf_construct, arginfo_mf_construct)
    ML_RECOMMENDER(MFModel, false)
    ML_METHOD(getUserFactors, recommender_user_factors<MFModel>, arginfo_ml_get_array)
    ML_METHOD(getItemFactors, recommender_item_factors<MFModel>, arginfo_ml_get_array)
    ML_METHOD(getLosses, mf_losses, arginfo_ml_get_array)
    PHP_FE_END};

// ---- ImplicitALS -------------------------------------------------------------------------

ZEND_BEGIN_ARG_INFO_EX(arginfo_implicit_construct, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, factors, IS_LONG, 0, "10")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, l2, IS_DOUBLE, 0, "0.01")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, alpha, IS_DOUBLE, 0, "40.0")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, epochs, IS_LONG, 0, "15")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, seed, IS_LONG, 1, "null")
ZEND_END_ARG_INFO()

void ZEND_FASTCALL implicit_construct(INTERNAL_FUNCTION_PARAMETERS) {
    zend_long factors = 10, epochs = 15, seed = 0;
    double l2 = 0.01, alpha = 40.0;
    bool seed_null = true;
    ZEND_PARSE_PARAMETERS_START(0, 5)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(factors)
        Z_PARAM_DOUBLE(l2)
        Z_PARAM_DOUBLE(alpha)
        Z_PARAM_LONG(epochs)
        Z_PARAM_LONG_OR_NULL(seed, seed_null)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        ImplicitModel fresh;
        fresh.core.factors = positive(factors, "factors");
        fresh.core.l2 = strictly_positive(l2, "l2");
        fresh.core.alpha = non_negative(alpha, "alpha");
        fresh.core.epochs = positive(epochs, "epochs");
        parse_seed(seed_null, seed, fresh.core.has_seed, fresh.core.seed);
        reset(ZEND_THIS, std::move(fresh));
    });
}

const zend_function_entry implicit_methods[] = {
    ML_METHOD(__construct, implicit_construct, arginfo_implicit_construct)
    ML_RECOMMENDER(ImplicitModel, true)
    ML_METHOD(getUserFactors, recommender_user_factors<ImplicitModel>, arginfo_ml_get_array)
    ML_METHOD(getItemFactors, recommender_item_factors<ImplicitModel>, arginfo_ml_get_array)
    PHP_FE_END};

// ---- KNNRecommender ----------------------------------------------------------------------

ZEND_BEGIN_ARG_INFO_EX(arginfo_knn_construct, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, neighbors, IS_LONG, 0, "20")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, similarity, IS_STRING, 0, "\"cosine\"")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, basis, IS_STRING, 0, "\"item\"")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, shrinkage, IS_DOUBLE, 0, "0.0")
ZEND_END_ARG_INFO()

void ZEND_FASTCALL knn_construct(INTERNAL_FUNCTION_PARAMETERS) {
    zend_long neighbors = 20;
    zend_string *similarity = nullptr, *basis = nullptr;
    double shrinkage = 0.0;
    ZEND_PARSE_PARAMETERS_START(0, 4)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(neighbors)
        Z_PARAM_STR(similarity)
        Z_PARAM_STR(basis)
        Z_PARAM_DOUBLE(shrinkage)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        KNNModel fresh;
        fresh.core.neighbors = positive(neighbors, "neighbors");
        if (similarity) {
            fresh.core.similarity = mlcore::KNNRecommender::parse(zstr(similarity));
        }
        if (basis) {
            const std::string name = zstr(basis);
            check(name == "item" || name == "user", "basis must be \"item\" or \"user\"");
            fresh.core.user_based = name == "user";
        }
        fresh.core.shrinkage = non_negative(shrinkage, "shrinkage");
        reset(ZEND_THIS, std::move(fresh));
    });
}

const zend_function_entry knn_methods[] = {
    ML_METHOD(__construct, knn_construct, arginfo_knn_construct)
    ML_RECOMMENDER(KNNModel, false)
    PHP_FE_END};

}  // namespace

void mlphp::register_recommender() {
    register_class<MFModel>("ML\\Recommender\\MatrixFactorization", mf_methods, {ce_recommender});
    register_class<ImplicitModel>("ML\\Recommender\\ImplicitALS", implicit_methods, {ce_recommender});
    register_class<KNNModel>("ML\\Recommender\\KNNRecommender", knn_methods, {ce_recommender});
}
