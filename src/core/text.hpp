#pragma once

#include <map>
#include <string>

#include "archive.hpp"
#include "common.hpp"

namespace mlcore {

// Porter (1980) English stemmer. Expects lowercase ASCII words; other input is
// returned unchanged.
std::string porter_stem(const std::string &word);

// A tokenizer configuration. Words are runs of ASCII letters/digits/underscore
// and UTF-8 bytes (so non-Latin scripts stay intact), with inner apostrophes.
struct Tokenizer {
    enum class Kind : std::uint8_t { Word, Whitespace, Sentence, NGram, SkipGram, Stemmer };
    Kind kind = Kind::Word;
    std::string delimiter = " ";
    std::size_t min = 2, max = 2, skip = 2;

    std::vector<std::string> tokenize(const std::string &text) const;
    void save(ArchiveWriter &w) const;
    void load(ArchiveReader &r);
};

std::vector<std::string> word_tokens(const std::string &text);
std::string to_lower_ascii(std::string s);
std::string to_upper_ascii(std::string s);

// 32-bit hashes for TokenHashingVectorizer.
std::uint32_t murmur3(const std::string &s, std::uint32_t seed = 0);
std::uint32_t crc32(const std::string &s);
std::uint32_t fnv1a(const std::string &s);

// Bag-of-words vocabulary.
class WordCounter {
public:
    std::size_t max_vocabulary = std::numeric_limits<std::size_t>::max();
    std::size_t min_document_count = 1;
    double max_document_ratio = 1.0;

    std::vector<std::string> vocabulary;  // sorted
    std::map<std::string, std::size_t> index;

    void fit(const std::vector<std::vector<std::string>> &documents);
    void transform(const std::vector<std::string> &tokens, double *out) const;
    void save(ArchiveWriter &w) const;
    void load(ArchiveReader &r);
};

// TF-IDF weighting of a count matrix.
struct TfIdf {
    double smoothing = 1.0;
    bool dampening = false;
    std::vector<double> idf;

    void fit(const Matrix &counts);
    void transform(const double *in, double *out) const;
    void save(ArchiveWriter &w) const;
    void load(ArchiveReader &r);
};

// Okapi BM25 weighting of a count matrix.
struct BM25 {
    double k1 = 1.2;
    double b = 0.75;
    std::vector<double> idf;
    double average_length = 0.0;

    void fit(const Matrix &counts);
    void transform(const double *in, double *out) const;
    void save(ArchiveWriter &w) const;
    void load(ArchiveReader &r);
};

}  // namespace mlcore
