// ML\Text: tokenizers, the Porter stemmer, text normalizers/filters, and the
// WordCountVectorizer, TokenHashingVectorizer, TfIdfTransformer and
// BM25Transformer. Text transformers take a list of documents (strings).

#include <map>
#include <set>

#include "binding.hpp"
#include "core/text.hpp"

namespace {

using namespace mlphp;
using mlcore::ArchiveReader;
using mlcore::ArchiveWriter;
using Kind = mlcore::Tokenizer::Kind;

std::vector<std::string> read_documents(HashTable *ht) {
    std::vector<std::string> docs;
    docs.reserve(zend_hash_num_elements(ht));
    zval *v;
    ZEND_HASH_FOREACH_VAL(ht, v) {
        ZVAL_DEREF(v);
        if (Z_TYPE_P(v) != IS_STRING) {
            throw std::invalid_argument("Document #" + std::to_string(docs.size()) + " must be a string, " +
                                        zend_zval_type_name(v) + " given");
        }
        docs.emplace_back(Z_STRVAL_P(v), Z_STRLEN_P(v));
    }
    ZEND_HASH_FOREACH_END();
    return docs;
}

void write_strings(zval *out, const std::vector<std::string> &values) {
    array_init_size(out, static_cast<uint32_t>(values.size()));
    for (const std::string &s : values) {
        add_next_index_stringl(out, s.data(), s.size());
    }
}

// ---- tokenizers ------------------------------------------------------------------------

struct TokenizerModel {
    static constexpr std::uint32_t kTag = mlcore::tag("TOKN");
    mlcore::Tokenizer tok;
    void save(ArchiveWriter &w) const { tok.save(w); }
    void load(ArchiveReader &r) { tok.load(r); }
};

std::map<zend_class_entry *, Kind> tokenizer_kinds;

const mlcore::Tokenizer *native_tokenizer(zval *object) {
    if (Z_TYPE_P(object) != IS_OBJECT) return nullptr;
    const auto it = tokenizer_kinds.find(Z_OBJCE_P(object));
    return it == tokenizer_kinds.end() ? nullptr : &native<TokenizerModel>(object).tok;
}

template <Kind K>
void ZEND_FASTCALL simple_tokenizer_construct(INTERNAL_FUNCTION_PARAMETERS) {
    ZEND_PARSE_PARAMETERS_NONE();
    TokenizerModel fresh;
    fresh.tok.kind = K;
    reset(ZEND_THIS, std::move(fresh));
}

ZEND_BEGIN_ARG_INFO_EX(arginfo_whitespace_construct, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, delimiter, IS_STRING, 0, "\" \"")
ZEND_END_ARG_INFO()

void ZEND_FASTCALL whitespace_construct(INTERNAL_FUNCTION_PARAMETERS) {
    zend_string *delimiter = nullptr;
    ZEND_PARSE_PARAMETERS_START(0, 1)
        Z_PARAM_OPTIONAL
        Z_PARAM_STR(delimiter)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        TokenizerModel fresh;
        fresh.tok.kind = Kind::Whitespace;
        if (delimiter) {
            check(ZSTR_LEN(delimiter) > 0, "delimiter must not be empty");
            fresh.tok.delimiter = zstr(delimiter);
        }
        reset(ZEND_THIS, std::move(fresh));
    });
}

ZEND_BEGIN_ARG_INFO_EX(arginfo_ngram_construct, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, min, IS_LONG, 0, "2")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, max, IS_LONG, 0, "2")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_skipgram_construct, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, min, IS_LONG, 0, "2")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, max, IS_LONG, 0, "2")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, skip, IS_LONG, 0, "2")
ZEND_END_ARG_INFO()

template <Kind K>
void ZEND_FASTCALL gram_construct(INTERNAL_FUNCTION_PARAMETERS) {
    zend_long min = 2, max = 2, skip = 2;
    ZEND_PARSE_PARAMETERS_START(0, K == Kind::SkipGram ? 3 : 2)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(min)
        Z_PARAM_LONG(max)
        Z_PARAM_LONG(skip)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        check(min >= 1 && max >= min && max <= 64, "min must be at least 1 and max between min and 64");
        check(skip >= 0 && skip <= 64, "skip must be between 0 and 64");
        TokenizerModel fresh;
        fresh.tok.kind = K;
        fresh.tok.min = static_cast<std::size_t>(min);
        fresh.tok.max = static_cast<std::size_t>(max);
        fresh.tok.skip = static_cast<std::size_t>(skip);
        reset(ZEND_THIS, std::move(fresh));
    });
}

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_tokenize, 0, 1, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO(0, text, IS_STRING, 0)
ZEND_END_ARG_INFO()

void ZEND_FASTCALL tokenizer_tokenize(INTERNAL_FUNCTION_PARAMETERS) {
    zend_string *text;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_STR(text)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] { write_strings(return_value, native<TokenizerModel>(ZEND_THIS).tok.tokenize(zstr(text))); });
}

void ZEND_FASTCALL tokenizer_unserialize(INTERNAL_FUNCTION_PARAMETERS) {
    HashTable *data;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY_HT(data)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        zval *state = require_key(data, "state");
        check(Z_TYPE_P(state) == IS_STRING, "Invalid serialized data: \"state\" must be a string");
        TokenizerModel restored;
        ArchiveReader r(Z_STRVAL_P(state), Z_STRLEN_P(state), TokenizerModel::kTag);
        restored.load(r);
        r.finish();
        check(restored.tok.kind == tokenizer_kinds.at(Z_OBJCE_P(ZEND_THIS)), "Invalid serialized data: tokenizer kind mismatch");
        reset(ZEND_THIS, std::move(restored));
    });
}

#define ML_TOKENIZER(ctor, arginfo)                                            \
    ML_METHOD(__construct, ctor, arginfo)                                       \
    ML_METHOD(tokenize, tokenizer_tokenize, arginfo_tokenize)                   \
    ML_METHOD(__serialize, serialize_native<TokenizerModel>, arginfo_ml_get_array) \
    ML_METHOD(__unserialize, tokenizer_unserialize, arginfo_ml_unserialize)

const zend_function_entry word_methods[] = {ML_TOKENIZER(simple_tokenizer_construct<Kind::Word>, arginfo_ml_none) PHP_FE_END};
const zend_function_entry whitespace_methods[] = {ML_TOKENIZER(whitespace_construct, arginfo_whitespace_construct) PHP_FE_END};
const zend_function_entry sentence_methods[] = {ML_TOKENIZER(simple_tokenizer_construct<Kind::Sentence>, arginfo_ml_none) PHP_FE_END};
const zend_function_entry ngram_methods[] = {ML_TOKENIZER(gram_construct<Kind::NGram>, arginfo_ngram_construct) PHP_FE_END};
const zend_function_entry skipgram_methods[] = {ML_TOKENIZER(gram_construct<Kind::SkipGram>, arginfo_skipgram_construct) PHP_FE_END};
const zend_function_entry stemmer_methods[] = {ML_TOKENIZER(simple_tokenizer_construct<Kind::Stemmer>, arginfo_ml_none) PHP_FE_END};

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_stem, 0, 1, IS_STRING, 0)
    ZEND_ARG_TYPE_INFO(0, word, IS_STRING, 0)
ZEND_END_ARG_INFO()

void ZEND_FASTCALL porter_stem(INTERNAL_FUNCTION_PARAMETERS) {
    zend_string *word;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_STR(word)
    ZEND_PARSE_PARAMETERS_END();
    const std::string stem = mlcore::porter_stem(mlcore::to_lower_ascii(zstr(word)));
    RETVAL_STRINGL(stem.data(), stem.size());
}

const zend_function_entry porter_methods[] = {
    ZEND_FENTRY(stem, porter_stem, arginfo_stem, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    PHP_FE_END};

// Tokenizes with a native tokenizer, or calls ->tokenize() on a PHP one.
std::vector<std::string> tokenize_with(const mlcore::Tokenizer *native_tok, zval *custom, const std::string &text) {
    if (!custom || Z_TYPE_P(custom) != IS_OBJECT) {
        return native_tok->tokenize(text);
    }
    zval arg, result;
    ZVAL_STRINGL(&arg, text.data(), text.size());
    try {
        call_method(custom, "tokenize", &result, {&arg});
    } catch (...) {
        zval_ptr_dtor(&arg);
        throw;
    }
    zval_ptr_dtor(&arg);
    if (Z_TYPE(result) != IS_ARRAY) {
        zval_ptr_dtor(&result);
        throw std::invalid_argument("tokenize() must return an array of strings");
    }
    std::vector<std::string> tokens;
    zval *t;
    ZEND_HASH_FOREACH_VAL(Z_ARRVAL(result), t) {
        ZVAL_DEREF(t);
        if (Z_TYPE_P(t) == IS_STRING) {
            tokens.emplace_back(Z_STRVAL_P(t), Z_STRLEN_P(t));
        }
    }
    ZEND_HASH_FOREACH_END();
    zval_ptr_dtor(&result);
    return tokens;
}

// Accepts a tokenizer argument: native tokenizers are copied into `tok`, PHP
// tokenizers are kept in the "tokenizer" property.
void set_tokenizer(zval *self, zval *tokenizer, mlcore::Tokenizer &tok) {
    if (!tokenizer || Z_TYPE_P(tokenizer) == IS_NULL) {
        tok = mlcore::Tokenizer{};
        return;
    }
    require_instance(tokenizer, ce_tokenizer, "tokenizer");
    if (const mlcore::Tokenizer *native_tok = native_tokenizer(tokenizer)) {
        tok = *native_tok;
    } else {
        write_prop(self, "tokenizer", tokenizer);
    }
}

// ---- document transformers -------------------------------------------------------------------

struct TextNormalizerModel {
    static constexpr std::uint32_t kTag = mlcore::tag("TNRM");
    bool uppercase = false;
    bool trim = true;
    void save(ArchiveWriter &w) const {
        w.boolean(uppercase);
        w.boolean(trim);
    }
    void load(ArchiveReader &r) {
        uppercase = r.boolean();
        trim = r.boolean();
    }
};

ZEND_BEGIN_ARG_INFO_EX(arginfo_normalizer_construct, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, uppercase, _IS_BOOL, 0, "false")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, trim, _IS_BOOL, 0, "true")
ZEND_END_ARG_INFO()

void ZEND_FASTCALL normalizer_construct(INTERNAL_FUNCTION_PARAMETERS) {
    bool uppercase = false, trim = true;
    ZEND_PARSE_PARAMETERS_START(0, 2)
        Z_PARAM_OPTIONAL
        Z_PARAM_BOOL(uppercase)
        Z_PARAM_BOOL(trim)
    ZEND_PARSE_PARAMETERS_END();
    TextNormalizerModel fresh;
    fresh.uppercase = uppercase;
    fresh.trim = trim;
    reset(ZEND_THIS, fresh);
}

std::string collapse_whitespace(const std::string &s) {
    std::string out;
    bool space = false;
    for (char c : s) {
        if (std::isspace(static_cast<unsigned char>(c))) {
            space = !out.empty();
        } else {
            if (space) out += ' ';
            out += c;
            space = false;
        }
    }
    return out;
}

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_documents, 0, 1, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO(0, documents, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

void ZEND_FASTCALL normalizer_transform(INTERNAL_FUNCTION_PARAMETERS) {
    HashTable *documents;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY_HT(documents)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        const TextNormalizerModel &m = native<TextNormalizerModel>(ZEND_THIS);
        std::vector<std::string> docs = read_documents(documents);
        for (std::string &d : docs) {
            d = m.uppercase ? mlcore::to_upper_ascii(d) : mlcore::to_lower_ascii(d);
            if (m.trim) d = collapse_whitespace(d);
        }
        write_strings(return_value, docs);
    });
}

const zend_function_entry normalizer_methods[] = {
    ML_METHOD(__construct, normalizer_construct, arginfo_normalizer_construct)
    ML_METHOD(transform, normalizer_transform, arginfo_documents)
    ML_METHOD(__serialize, serialize_native<TextNormalizerModel>, arginfo_ml_get_array)
    ML_METHOD(__unserialize, unserialize_native<TextNormalizerModel>, arginfo_ml_unserialize)
    PHP_FE_END};

// Multibyte variant: delegates case conversion to mbstring.
struct MbNormalizerModel {
    static constexpr std::uint32_t kTag = mlcore::tag("MBNM");
    bool uppercase = false;
    bool trim = true;
    void save(ArchiveWriter &w) const {
        w.boolean(uppercase);
        w.boolean(trim);
    }
    void load(ArchiveReader &r) {
        uppercase = r.boolean();
        trim = r.boolean();
    }
};

void ZEND_FASTCALL mb_normalizer_construct(INTERNAL_FUNCTION_PARAMETERS) {
    bool uppercase = false, trim = true;
    ZEND_PARSE_PARAMETERS_START(0, 2)
        Z_PARAM_OPTIONAL
        Z_PARAM_BOOL(uppercase)
        Z_PARAM_BOOL(trim)
    ZEND_PARSE_PARAMETERS_END();
    MbNormalizerModel fresh;
    fresh.uppercase = uppercase;
    fresh.trim = trim;
    reset(ZEND_THIS, fresh);
}

std::string call_string_function(const char *name, const std::string &arg) {
    zval fn, param, result;
    ZVAL_STRING(&fn, name);
    ZVAL_STRINGL(&param, arg.data(), arg.size());
    const bool ok = call_user_function(nullptr, nullptr, &fn, &result, 1, &param) == SUCCESS && !EG(exception);
    zval_ptr_dtor(&fn);
    zval_ptr_dtor(&param);
    if (!ok) {
        zval_ptr_dtor(&result);
        throw PhpError{};
    }
    std::string out = Z_TYPE(result) == IS_STRING ? std::string(Z_STRVAL(result), Z_STRLEN(result)) : arg;
    zval_ptr_dtor(&result);
    return out;
}

void ZEND_FASTCALL mb_normalizer_transform(INTERNAL_FUNCTION_PARAMETERS) {
    HashTable *documents;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY_HT(documents)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        if (!zend_hash_str_exists(EG(function_table), "mb_strtolower", sizeof("mb_strtolower") - 1)) {
            throw std::runtime_error("MultibyteTextNormalizer requires the mbstring extension");
        }
        const MbNormalizerModel &m = native<MbNormalizerModel>(ZEND_THIS);
        std::vector<std::string> docs = read_documents(documents);
        for (std::string &d : docs) {
            d = call_string_function(m.uppercase ? "mb_strtoupper" : "mb_strtolower", d);
            if (m.trim) d = collapse_whitespace(d);
        }
        write_strings(return_value, docs);
    });
}

const zend_function_entry mb_normalizer_methods[] = {
    ML_METHOD(__construct, mb_normalizer_construct, arginfo_normalizer_construct)
    ML_METHOD(transform, mb_normalizer_transform, arginfo_documents)
    ML_METHOD(__serialize, serialize_native<MbNormalizerModel>, arginfo_ml_get_array)
    ML_METHOD(__unserialize, unserialize_native<MbNormalizerModel>, arginfo_ml_unserialize)
    PHP_FE_END};

// RegexFilter: removes every match of the given PCRE patterns.
struct RegexModel {
    static constexpr std::uint32_t kTag = mlcore::tag("RGXF");
    std::vector<std::string> patterns;
    void save(ArchiveWriter &w) const { w.strings(patterns); }
    void load(ArchiveReader &r) { patterns = r.strings(); }
};

ZEND_BEGIN_ARG_INFO_EX(arginfo_regex_construct, 0, 0, 1)
    ZEND_ARG_TYPE_INFO(0, patterns, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

void ZEND_FASTCALL regex_construct(INTERNAL_FUNCTION_PARAMETERS) {
    HashTable *patterns;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY_HT(patterns)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        RegexModel fresh;
        fresh.patterns = read_documents(patterns);
        check(!fresh.patterns.empty(), "patterns must not be empty");
        for (const std::string &pattern : fresh.patterns) {
            // Compile each pattern once now (silently) so mistakes surface here.
            zval fn, args[2], result;
            ZVAL_STRING(&fn, "preg_match");
            ZVAL_STRINGL(&args[0], pattern.data(), pattern.size());
            ZVAL_EMPTY_STRING(&args[1]);
            const int reporting = EG(error_reporting);
            EG(error_reporting) = 0;
            const bool called = call_user_function(nullptr, nullptr, &fn, &result, 2, args) == SUCCESS;
            EG(error_reporting) = reporting;
            const bool valid = called && !EG(exception) && Z_TYPE(result) == IS_LONG;
            zval_ptr_dtor(&fn);
            zval_ptr_dtor(&args[0]);
            zval_ptr_dtor(&args[1]);
            zval_ptr_dtor(&result);
            if (EG(exception)) {
                throw PhpError{};
            }
            check(valid, ("Invalid regular expression \"" + pattern + "\"").c_str());
        }
        reset(ZEND_THIS, std::move(fresh));
    });
}

void ZEND_FASTCALL regex_transform(INTERNAL_FUNCTION_PARAMETERS) {
    HashTable *documents;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY_HT(documents)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        const RegexModel &m = native<RegexModel>(ZEND_THIS);
        read_documents(documents);  // validates that every document is a string
        zval fn, args[3];
        ZVAL_STRING(&fn, "preg_replace");
        write_strings(&args[0], m.patterns);
        ZVAL_EMPTY_STRING(&args[1]);
        array_zval(documents, &args[2]);
        const bool ok = call_user_function(nullptr, nullptr, &fn, return_value, 3, args) == SUCCESS && !EG(exception);
        zval_ptr_dtor(&fn);
        for (zval &a : args) zval_ptr_dtor(&a);
        if (!ok || Z_TYPE_P(return_value) != IS_ARRAY) {
            zval_ptr_dtor(return_value);
            ZVAL_NULL(return_value);
            if (!EG(exception)) {
                throw std::invalid_argument("A RegexFilter pattern is not a valid regular expression");
            }
            throw PhpError{};
        }
    });
}

const zend_function_entry regex_methods[] = {
    ML_METHOD(__construct, regex_construct, arginfo_regex_construct)
    ML_METHOD(transform, regex_transform, arginfo_documents)
    ML_METHOD(__serialize, serialize_native<RegexModel>, arginfo_ml_get_array)
    ML_METHOD(__unserialize, unserialize_native<RegexModel>, arginfo_ml_unserialize)
    PHP_FE_END};

// StopWordFilter: drops whitespace-separated words found in the list (case-insensitive).
struct StopWordModel {
    static constexpr std::uint32_t kTag = mlcore::tag("STPW");
    std::set<std::string> words;
    void save(ArchiveWriter &w) const { w.strings(std::vector<std::string>(words.begin(), words.end())); }
    void load(ArchiveReader &r) {
        const std::vector<std::string> list = r.strings();
        words = std::set<std::string>(list.begin(), list.end());
    }
};

ZEND_BEGIN_ARG_INFO_EX(arginfo_stop_construct, 0, 0, 1)
    ZEND_ARG_TYPE_INFO(0, stopWords, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

void ZEND_FASTCALL stop_construct(INTERNAL_FUNCTION_PARAMETERS) {
    HashTable *words;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY_HT(words)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        StopWordModel fresh;
        for (const std::string &w : read_documents(words)) {
            fresh.words.insert(mlcore::to_lower_ascii(w));
        }
        reset(ZEND_THIS, std::move(fresh));
    });
}

std::string strip_punctuation(const std::string &w) {
    std::size_t a = 0, b = w.size();
    while (a < b && std::ispunct(static_cast<unsigned char>(w[a]))) ++a;
    while (b > a && std::ispunct(static_cast<unsigned char>(w[b - 1]))) --b;
    return w.substr(a, b - a);
}

void ZEND_FASTCALL stop_transform(INTERNAL_FUNCTION_PARAMETERS) {
    HashTable *documents;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY_HT(documents)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        const StopWordModel &m = native<StopWordModel>(ZEND_THIS);
        mlcore::Tokenizer spaces;
        spaces.kind = Kind::Whitespace;
        std::vector<std::string> docs = read_documents(documents);
        for (std::string &d : docs) {
            std::string kept;
            for (const std::string &w : spaces.tokenize(collapse_whitespace(d))) {
                if (m.words.count(mlcore::to_lower_ascii(strip_punctuation(w)))) continue;
                if (!kept.empty()) kept += ' ';
                kept += w;
            }
            d = std::move(kept);
        }
        write_strings(return_value, docs);
    });
}

const zend_function_entry stop_methods[] = {
    ML_METHOD(__construct, stop_construct, arginfo_stop_construct)
    ML_METHOD(transform, stop_transform, arginfo_documents)
    ML_METHOD(__serialize, serialize_native<StopWordModel>, arginfo_ml_get_array)
    ML_METHOD(__unserialize, unserialize_native<StopWordModel>, arginfo_ml_unserialize)
    PHP_FE_END};

// ---- vectorizers ------------------------------------------------------------------------------------

struct WordCountModel {
    static constexpr std::uint32_t kTag = mlcore::tag("WCNT");
    mlcore::WordCounter counter;
    mlcore::Tokenizer tok;
    bool fitted() const { return !counter.vocabulary.empty(); }
    void save(ArchiveWriter &w) const {
        counter.save(w);
        tok.save(w);
    }
    void load(ArchiveReader &r) {
        counter.load(r);
        tok.load(r);
    }
};

ZEND_BEGIN_ARG_INFO_EX(arginfo_word_count_construct, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, maxVocabularySize, IS_LONG, 0, "PHP_INT_MAX")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, minDocumentCount, IS_LONG, 0, "1")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, maxDocumentRatio, IS_DOUBLE, 0, "1.0")
    ZEND_ARG_OBJ_INFO_WITH_DEFAULT_VALUE(0, tokenizer, ML\\Text\\Tokenizer, 1, "null")
ZEND_END_ARG_INFO()

void ZEND_FASTCALL word_count_construct(INTERNAL_FUNCTION_PARAMETERS) {
    zend_long max_vocab = ZEND_LONG_MAX, min_count = 1;
    double max_ratio = 1.0;
    zval *tokenizer = nullptr;
    ZEND_PARSE_PARAMETERS_START(0, 4)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(max_vocab)
        Z_PARAM_LONG(min_count)
        Z_PARAM_DOUBLE(max_ratio)
        Z_PARAM_OBJECT_OF_CLASS_OR_NULL(tokenizer, ce_tokenizer)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        WordCountModel fresh;
        fresh.counter.max_vocabulary = positive(max_vocab, "maxVocabularySize");
        fresh.counter.min_document_count = positive(min_count, "minDocumentCount");
        check(max_ratio > 0.0 && max_ratio <= 1.0, "maxDocumentRatio must be greater than 0 and at most 1");
        fresh.counter.max_document_ratio = max_ratio;
        set_tokenizer(ZEND_THIS, tokenizer, fresh.tok);
        reset(ZEND_THIS, std::move(fresh));
    });
}

std::vector<std::vector<std::string>> tokenize_all(zval *self, const mlcore::Tokenizer &tok, const std::vector<std::string> &docs) {
    zval *custom = read_prop(self, "tokenizer");
    std::vector<std::vector<std::string>> out;
    out.reserve(docs.size());
    for (const std::string &d : docs) {
        out.push_back(tokenize_with(&tok, custom, d));
    }
    return out;
}

void word_count_matrix(const WordCountModel &m, const std::vector<std::vector<std::string>> &tokens, zval *out) {
    mlcore::Matrix X(tokens.size(), m.counter.vocabulary.size());
    for (std::size_t r = 0; r < tokens.size(); ++r) {
        m.counter.transform(tokens[r], X.row(r));
    }
    write_matrix(out, X);
}

template <bool Transform>
void ZEND_FASTCALL word_count_fit(INTERNAL_FUNCTION_PARAMETERS) {
    HashTable *documents;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY_HT(documents)
    ZEND_PARSE_PARAMETERS_END();
    const bool ok = guard([&] {
        WordCountModel &m = native<WordCountModel>(ZEND_THIS);
        const auto tokens = tokenize_all(ZEND_THIS, m.tok, read_documents(documents));
        WordCountModel next = m;
        next.counter.fit(tokens);
        m = std::move(next);
        if (Transform) {
            word_count_matrix(m, tokens, return_value);
        }
    });
    if (!Transform) {
        if (!ok) RETURN_THROWS();
        RETURN_OBJ_COPY(Z_OBJ_P(ZEND_THIS));
    }
}

void ZEND_FASTCALL word_count_transform(INTERNAL_FUNCTION_PARAMETERS) {
    HashTable *documents;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY_HT(documents)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        const WordCountModel &m = native<WordCountModel>(ZEND_THIS);
        require_fitted(m.fitted(), ZEND_THIS);
        word_count_matrix(m, tokenize_all(ZEND_THIS, m.tok, read_documents(documents)), return_value);
    });
}

void ZEND_FASTCALL word_count_vocabulary(INTERNAL_FUNCTION_PARAMETERS) {
    ZEND_PARSE_PARAMETERS_NONE();
    guard([&] {
        const WordCountModel &m = native<WordCountModel>(ZEND_THIS);
        require_fitted(m.fitted(), ZEND_THIS);
        write_strings(return_value, m.counter.vocabulary);
    });
}

// Serialization that also carries a user-defined (PHP) tokenizer object.
template <typename M>
void ZEND_FASTCALL vectorizer_serialize(INTERNAL_FUNCTION_PARAMETERS) {
    ZEND_PARSE_PARAMETERS_NONE();
    ArchiveWriter w(M::kTag);
    native<M>(ZEND_THIS).save(w);
    array_init(return_value);
    add_assoc_long(return_value, "mlnative", mlcore::kArchiveFormat);
    add_assoc_stringl(return_value, "state", w.data().data(), w.data().size());
    zval *custom = read_prop(ZEND_THIS, "tokenizer");
    Z_TRY_ADDREF_P(custom);
    add_assoc_zval(return_value, "tokenizer", custom);
}

template <typename M>
void ZEND_FASTCALL vectorizer_unserialize(INTERNAL_FUNCTION_PARAMETERS) {
    HashTable *data;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY_HT(data)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        zval *state = require_key(data, "state");
        check(Z_TYPE_P(state) == IS_STRING, "Invalid serialized data: \"state\" must be a string");
        M restored;
        ArchiveReader r(Z_STRVAL_P(state), Z_STRLEN_P(state), M::kTag);
        restored.load(r);
        r.finish();
        zval *custom = zend_hash_str_find_deref(data, "tokenizer", sizeof("tokenizer") - 1);
        if (custom && Z_TYPE_P(custom) != IS_NULL) {
            require_instance(custom, ce_tokenizer, "Invalid serialized data: tokenizer");
            write_prop(ZEND_THIS, "tokenizer", custom);
        }
        reset(ZEND_THIS, std::move(restored));
    });
}

const zend_function_entry word_count_methods[] = {
    ML_METHOD(__construct, word_count_construct, arginfo_word_count_construct)
    ML_METHOD(fit, word_count_fit<false>, arginfo_documents)
    ML_METHOD(transform, word_count_transform, arginfo_documents)
    ML_METHOD(fitTransform, word_count_fit<true>, arginfo_documents)
    ML_METHOD(getVocabulary, word_count_vocabulary, arginfo_ml_get_array)
    ML_METHOD(__serialize, vectorizer_serialize<WordCountModel>, arginfo_ml_get_array)
    ML_METHOD(__unserialize, vectorizer_unserialize<WordCountModel>, arginfo_ml_unserialize)
    PHP_FE_END};

struct HashingModel {
    static constexpr std::uint32_t kTag = mlcore::tag("THSH");
    std::size_t dimensions = 1024;
    std::string hash = "murmur3";
    mlcore::Tokenizer tok;
    void save(ArchiveWriter &w) const {
        w.size(dimensions);
        w.str(hash);
        tok.save(w);
    }
    void load(ArchiveReader &r) {
        dimensions = r.size();
        hash = r.str();
        tok.load(r);
        ArchiveReader::require(dimensions >= 1 && (hash == "murmur3" || hash == "crc32" || hash == "fnv1a"), "bad hashing parameters");
    }
};

ZEND_BEGIN_ARG_INFO_EX(arginfo_hashing_construct, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, dimensions, IS_LONG, 0, "1024")
    ZEND_ARG_OBJ_INFO_WITH_DEFAULT_VALUE(0, tokenizer, ML\\Text\\Tokenizer, 1, "null")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, hash, IS_STRING, 0, "\"murmur3\"")
ZEND_END_ARG_INFO()

void ZEND_FASTCALL hashing_construct(INTERNAL_FUNCTION_PARAMETERS) {
    zend_long dimensions = 1024;
    zval *tokenizer = nullptr;
    zend_string *hash = nullptr;
    ZEND_PARSE_PARAMETERS_START(0, 3)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(dimensions)
        Z_PARAM_OBJECT_OF_CLASS_OR_NULL(tokenizer, ce_tokenizer)
        Z_PARAM_STR(hash)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        HashingModel fresh;
        fresh.dimensions = positive(dimensions, "dimensions");
        check(fresh.dimensions <= (1u << 24), "dimensions must be at most 16777216");
        if (hash) {
            fresh.hash = zstr(hash);
            check(fresh.hash == "murmur3" || fresh.hash == "crc32" || fresh.hash == "fnv1a",
                  "hash must be \"murmur3\", \"crc32\" or \"fnv1a\"");
        }
        set_tokenizer(ZEND_THIS, tokenizer, fresh.tok);
        reset(ZEND_THIS, std::move(fresh));
    });
}

void ZEND_FASTCALL hashing_transform(INTERNAL_FUNCTION_PARAMETERS) {
    HashTable *documents;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY_HT(documents)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        const HashingModel &m = native<HashingModel>(ZEND_THIS);
        const auto tokens = tokenize_all(ZEND_THIS, m.tok, read_documents(documents));
        mlcore::Matrix X(tokens.size(), m.dimensions);
        for (std::size_t r = 0; r < tokens.size(); ++r) {
            for (const std::string &t : tokens[r]) {
                const std::uint32_t h = m.hash == "crc32" ? mlcore::crc32(t) : m.hash == "fnv1a" ? mlcore::fnv1a(t) : mlcore::murmur3(t);
                X(r, h % m.dimensions) += 1.0;
            }
        }
        write_matrix(return_value, X);
    });
}

const zend_function_entry hashing_methods[] = {
    ML_METHOD(__construct, hashing_construct, arginfo_hashing_construct)
    ML_METHOD(transform, hashing_transform, arginfo_documents)
    ML_METHOD(__serialize, vectorizer_serialize<HashingModel>, arginfo_ml_get_array)
    ML_METHOD(__unserialize, vectorizer_unserialize<HashingModel>, arginfo_ml_unserialize)
    PHP_FE_END};

// ---- TF-IDF and BM25 (numeric, stateful) ----------------------------------------------------

template <typename Core, std::uint32_t Tag>
struct Weighting {
    static constexpr std::uint32_t kTag = Tag;
    Core core;
    bool fitted() const { return !core.idf.empty(); }
    std::size_t features() const { return core.idf.size(); }
    std::size_t output_dim() const { return core.idf.size(); }
    void transform_row(const double *in, double *out) const { core.transform(in, out); }
    void fit(const mlcore::Matrix &X, unsigned) { core.fit(X); }
    void save(ArchiveWriter &w) const { core.save(w); }
    void load(ArchiveReader &r) { core.load(r); }
};

using TfIdfModel = Weighting<mlcore::TfIdf, mlcore::tag("TFID")>;
using BM25Model = Weighting<mlcore::BM25, mlcore::tag("BM25")>;

ZEND_BEGIN_ARG_INFO_EX(arginfo_tfidf_construct, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, smoothing, IS_DOUBLE, 0, "1.0")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, dampening, _IS_BOOL, 0, "false")
ZEND_END_ARG_INFO()

void ZEND_FASTCALL tfidf_construct(INTERNAL_FUNCTION_PARAMETERS) {
    double smoothing = 1.0;
    bool dampening = false;
    ZEND_PARSE_PARAMETERS_START(0, 2)
        Z_PARAM_OPTIONAL
        Z_PARAM_DOUBLE(smoothing)
        Z_PARAM_BOOL(dampening)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        TfIdfModel fresh;
        fresh.core.smoothing = strictly_positive(smoothing, "smoothing");
        fresh.core.dampening = dampening;
        reset(ZEND_THIS, std::move(fresh));
    });
}

ZEND_BEGIN_ARG_INFO_EX(arginfo_bm25_construct, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, dampening, IS_DOUBLE, 0, "1.2")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, normalization, IS_DOUBLE, 0, "0.75")
ZEND_END_ARG_INFO()

void ZEND_FASTCALL bm25_construct(INTERNAL_FUNCTION_PARAMETERS) {
    double dampening = 1.2, normalization = 0.75;
    ZEND_PARSE_PARAMETERS_START(0, 2)
        Z_PARAM_OPTIONAL
        Z_PARAM_DOUBLE(dampening)
        Z_PARAM_DOUBLE(normalization)
    ZEND_PARSE_PARAMETERS_END();
    guard([&] {
        BM25Model fresh;
        fresh.core.k1 = non_negative(dampening, "dampening");
        fresh.core.b = in_range(normalization, 0.0, 1.0, "normalization");
        reset(ZEND_THIS, std::move(fresh));
    });
}

#define ML_WEIGHTING(M, ctor, arginfo)                                         \
    ML_METHOD(__construct, ctor, arginfo)                                       \
    ML_METHOD(fit, fit_unsupervised<M>, arginfo_ml_fit_unsupervised)            \
    ML_METHOD(transform, transform<M>, arginfo_ml_samples_to_array)             \
    ML_METHOD(fitTransform, fit_transform<M>, arginfo_ml_samples_to_array)      \
    ML_METHOD(__serialize, serialize_native<M>, arginfo_ml_get_array)           \
    ML_METHOD(__unserialize, unserialize_native<M>, arginfo_ml_unserialize)

const zend_function_entry tfidf_methods[] = {ML_WEIGHTING(TfIdfModel, tfidf_construct, arginfo_tfidf_construct) PHP_FE_END};
const zend_function_entry bm25_methods[] = {ML_WEIGHTING(BM25Model, bm25_construct, arginfo_bm25_construct) PHP_FE_END};

zend_class_entry *tokenizer_class(const char *name, const zend_function_entry *methods, Kind kind) {
    zend_class_entry *ce = register_class<TokenizerModel>(name, methods, {ce_tokenizer});
    tokenizer_kinds[ce] = kind;
    return ce;
}

}  // namespace

void mlphp::register_text() {
    tokenizer_class("ML\\Text\\WordTokenizer", word_methods, Kind::Word);
    tokenizer_class("ML\\Text\\WhitespaceTokenizer", whitespace_methods, Kind::Whitespace);
    tokenizer_class("ML\\Text\\SentenceTokenizer", sentence_methods, Kind::Sentence);
    tokenizer_class("ML\\Text\\NGramTokenizer", ngram_methods, Kind::NGram);
    tokenizer_class("ML\\Text\\KSkipNGramTokenizer", skipgram_methods, Kind::SkipGram);
    tokenizer_class("ML\\Text\\WordStemmer", stemmer_methods, Kind::Stemmer);

    zend_class_entry tmp;
    INIT_CLASS_ENTRY(tmp, "ML\\Text\\PorterStemmer", porter_methods);
    zend_register_internal_class(&tmp)->ce_flags |= ZEND_ACC_FINAL;

    register_class<TextNormalizerModel>("ML\\Text\\TextNormalizer", normalizer_methods, {ce_transformer});
    register_class<MbNormalizerModel>("ML\\Text\\MultibyteTextNormalizer", mb_normalizer_methods, {ce_transformer});
    zend_class_entry *regex = register_class<RegexModel>("ML\\Text\\RegexFilter", regex_methods, {ce_transformer});
    const std::pair<const char *, const char *> constants[] = {
        {"URL", "/\\b(?:https?:\\/\\/|www\\.)\\S+/iu"},
        {"EMAIL", "/[\\w.+-]+@[\\w-]+\\.[\\w.-]+/u"},
        {"MENTION", "/(?<!\\w)@\\w+/u"},
        {"HASHTAG", "/(?<!\\w)#\\w+/u"},
        {"NUMBER", "/\\d+(?:[.,]\\d+)*/u"},
        {"PUNCTUATION", "/[\\p{P}\\p{S}]+/u"},
        {"EMOJI", "/[\\x{1F000}-\\x{1FAFF}\\x{2600}-\\x{27BF}\\x{FE0F}]/u"},
    };
    for (const auto &[name, pattern] : constants) {
        zend_declare_class_constant_string(regex, name, std::strlen(name), pattern);
    }
    register_class<StopWordModel>("ML\\Text\\StopWordFilter", stop_methods, {ce_transformer});

    zend_class_entry *word_count = register_class<WordCountModel>("ML\\Text\\WordCountVectorizer", word_count_methods,
                                                                 {ce_transformer, ce_stateful});
    zend_declare_property_null(word_count, "tokenizer", sizeof("tokenizer") - 1, ZEND_ACC_PRIVATE);
    zend_class_entry *hashing = register_class<HashingModel>("ML\\Text\\TokenHashingVectorizer", hashing_methods, {ce_transformer});
    zend_declare_property_null(hashing, "tokenizer", sizeof("tokenizer") - 1, ZEND_ACC_PRIVATE);
    register_class<TfIdfModel>("ML\\Text\\TfIdfTransformer", tfidf_methods, {ce_transformer, ce_stateful});
    register_class<BM25Model>("ML\\Text\\BM25Transformer", bm25_methods, {ce_transformer, ce_stateful});
}
