#include "text.hpp"

#include <cctype>
#include <cstring>
#include <functional>
#include <set>

namespace mlcore {

// ---- Porter stemmer -------------------------------------------------------------------

namespace {

class Porter {
public:
    explicit Porter(std::string word) : b_(std::move(word)), k_(static_cast<int>(b_.size()) - 1) {}

    std::string run() {
        if (k_ <= 1) {
            return b_;
        }
        step1ab();
        if (k_ > 0) {
            step1c();
            step2();
            step3();
            step4();
            step5();
        }
        return b_.substr(0, static_cast<std::size_t>(k_ + 1));
    }

private:
    bool cons(int i) const {
        switch (b_[static_cast<std::size_t>(i)]) {
            case 'a': case 'e': case 'i': case 'o': case 'u': return false;
            case 'y': return i == 0 ? true : !cons(i - 1);
            default: return true;
        }
    }

    // Number of VC sequences in b[0..j].
    int m() const {
        int n = 0, i = 0;
        while (true) {
            if (i > j_) return n;
            if (!cons(i)) break;
            ++i;
        }
        ++i;
        while (true) {
            while (true) {
                if (i > j_) return n;
                if (cons(i)) break;
                ++i;
            }
            ++i;
            ++n;
            while (true) {
                if (i > j_) return n;
                if (!cons(i)) break;
                ++i;
            }
            ++i;
        }
    }

    bool vowel_in_stem() const {
        for (int i = 0; i <= j_; ++i) {
            if (!cons(i)) return true;
        }
        return false;
    }

    bool double_c(int j) const {
        return j >= 1 && b_[static_cast<std::size_t>(j)] == b_[static_cast<std::size_t>(j - 1)] && cons(j);
    }

    bool cvc(int i) const {
        if (i < 2 || !cons(i) || cons(i - 1) || !cons(i - 2)) return false;
        const char ch = b_[static_cast<std::size_t>(i)];
        return ch != 'w' && ch != 'x' && ch != 'y';
    }

    bool ends(const char *s) {
        const int length = static_cast<int>(std::strlen(s));
        if (length > k_ + 1) return false;
        if (b_.compare(static_cast<std::size_t>(k_ - length + 1), static_cast<std::size_t>(length), s) != 0) return false;
        j_ = k_ - length;
        return true;
    }

    void set_to(const char *s) {
        b_ = b_.substr(0, static_cast<std::size_t>(j_ + 1)) + s;
        k_ = j_ + static_cast<int>(std::strlen(s));
    }

    void r(const char *s) {
        if (m() > 0) set_to(s);
    }

    char at(int i) const { return b_[static_cast<std::size_t>(i)]; }

    void step1ab() {
        if (at(k_) == 's') {
            if (ends("sses")) {
                k_ -= 2;
            } else if (ends("ies")) {
                set_to("i");
            } else if (at(k_ - 1) != 's') {
                --k_;
            }
        }
        if (ends("eed")) {
            if (m() > 0) --k_;
        } else if ((ends("ed") || ends("ing")) && vowel_in_stem()) {
            k_ = j_;
            b_.resize(static_cast<std::size_t>(k_ + 1));
            if (ends("at")) {
                set_to("ate");
            } else if (ends("bl")) {
                set_to("ble");
            } else if (ends("iz")) {
                set_to("ize");
            } else if (double_c(k_)) {
                --k_;
                const char ch = at(k_);
                if (ch == 'l' || ch == 's' || ch == 'z') ++k_;
            } else if (m() == 1 && cvc(k_)) {
                set_to("e");
            }
        }
        b_.resize(static_cast<std::size_t>(k_ + 1));
    }

    void step1c() {
        if (ends("y") && vowel_in_stem()) {
            b_[static_cast<std::size_t>(k_)] = 'i';
        }
    }

    void step2() {
        switch (at(k_ - 1)) {
            case 'a': if (ends("ational")) { r("ate"); break; } if (ends("tional")) { r("tion"); break; } break;
            case 'c': if (ends("enci")) { r("ence"); break; } if (ends("anci")) { r("ance"); break; } break;
            case 'e': if (ends("izer")) { r("ize"); break; } break;
            case 'l':
                if (ends("bli")) { r("ble"); break; }
                if (ends("alli")) { r("al"); break; }
                if (ends("entli")) { r("ent"); break; }
                if (ends("eli")) { r("e"); break; }
                if (ends("ousli")) { r("ous"); break; }
                break;
            case 'o':
                if (ends("ization")) { r("ize"); break; }
                if (ends("ation")) { r("ate"); break; }
                if (ends("ator")) { r("ate"); break; }
                break;
            case 's':
                if (ends("alism")) { r("al"); break; }
                if (ends("iveness")) { r("ive"); break; }
                if (ends("fulness")) { r("ful"); break; }
                if (ends("ousness")) { r("ous"); break; }
                break;
            case 't':
                if (ends("aliti")) { r("al"); break; }
                if (ends("iviti")) { r("ive"); break; }
                if (ends("biliti")) { r("ble"); break; }
                break;
            case 'g': if (ends("logi")) { r("log"); break; } break;
            default: break;
        }
        b_.resize(static_cast<std::size_t>(k_ + 1));
    }

    void step3() {
        switch (at(k_)) {
            case 'e':
                if (ends("icate")) { r("ic"); break; }
                if (ends("ative")) { r(""); break; }
                if (ends("alize")) { r("al"); break; }
                break;
            case 'i': if (ends("iciti")) { r("ic"); break; } break;
            case 'l': if (ends("ical")) { r("ic"); break; } if (ends("ful")) { r(""); break; } break;
            case 's': if (ends("ness")) { r(""); break; } break;
            default: break;
        }
        b_.resize(static_cast<std::size_t>(k_ + 1));
    }

    void step4() {
        if (k_ < 1) return;
        switch (at(k_ - 1)) {
            case 'a': if (ends("al")) break; return;
            case 'c': if (ends("ance")) break; if (ends("ence")) break; return;
            case 'e': if (ends("er")) break; return;
            case 'i': if (ends("ic")) break; return;
            case 'l': if (ends("able")) break; if (ends("ible")) break; return;
            case 'n': if (ends("ant")) break; if (ends("ement")) break; if (ends("ment")) break; if (ends("ent")) break; return;
            case 'o':
                if (ends("ion") && j_ >= 0 && (at(j_) == 's' || at(j_) == 't')) break;
                if (ends("ou")) break;
                return;
            case 's': if (ends("ism")) break; return;
            case 't': if (ends("ate")) break; if (ends("iti")) break; return;
            case 'u': if (ends("ous")) break; return;
            case 'v': if (ends("ive")) break; return;
            case 'z': if (ends("ize")) break; return;
            default: return;
        }
        if (m() > 1) {
            k_ = j_;
            b_.resize(static_cast<std::size_t>(k_ + 1));
        }
    }

    void step5() {
        j_ = k_;
        if (at(k_) == 'e') {
            const int a = m();
            if (a > 1 || (a == 1 && !cvc(k_ - 1))) --k_;
        }
        if (at(k_) == 'l' && double_c(k_) && m() > 1) --k_;
        b_.resize(static_cast<std::size_t>(k_ + 1));
    }

    std::string b_;
    int k_;
    int j_ = 0;
};

bool word_byte(unsigned char c) { return std::isalnum(c) || c == '_' || c >= 0x80; }

// Unicode punctuation and spaces that separate words in non-Latin scripts.
bool unicode_separator(std::uint32_t cp) {
    return (cp >= 0x00A0 && cp <= 0x00BF) || cp == 0x00D7 || cp == 0x00F7 ||  // Latin-1 punctuation, ×, ÷
           (cp >= 0x055A && cp <= 0x055F) || cp == 0x0589 ||                  // Armenian
           cp == 0x060C || cp == 0x061B || cp == 0x061F || (cp >= 0x066A && cp <= 0x066D) || cp == 0x06D4 ||  // Arabic/Persian
           (cp >= 0x2000 && cp <= 0x206F) ||                                  // general punctuation and spaces
           (cp >= 0x3000 && cp <= 0x303F) ||                                  // CJK symbols and punctuation
           (cp >= 0xFE10 && cp <= 0xFE1F) || (cp >= 0xFE30 && cp <= 0xFE4F) ||
           (cp >= 0xFF01 && cp <= 0xFF0F) || (cp >= 0xFF1A && cp <= 0xFF20) || (cp >= 0xFF3B && cp <= 0xFF40) ||
           (cp >= 0xFF5B && cp <= 0xFF65);
}

// Decodes the UTF-8 sequence at text[i]; returns its length (1 for invalid bytes).
std::size_t utf8_sequence(const std::string &text, std::size_t i, std::uint32_t &cp) {
    const unsigned char c = static_cast<unsigned char>(text[i]);
    const std::size_t len = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 1;
    if (len == 1 || i + len > text.size()) {
        cp = c;
        return 1;
    }
    cp = c & (0xFF >> (len + 1));
    for (std::size_t k = 1; k < len; ++k) {
        const unsigned char cc = static_cast<unsigned char>(text[i + k]);
        if ((cc & 0xC0) != 0x80) {
            cp = c;
            return 1;
        }
        cp = (cp << 6) | (cc & 0x3F);
    }
    return len;
}

}  // namespace

std::string porter_stem(const std::string &word) {
    for (unsigned char c : word) {
        if (c < 'a' || c > 'z') {
            return word;  // only plain lowercase ASCII words are stemmed
        }
    }
    return Porter(word).run();
}

std::string to_lower_ascii(std::string s) {
    for (char &c : s) {
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
    }
    return s;
}

std::string to_upper_ascii(std::string s) {
    for (char &c : s) {
        if (c >= 'a' && c <= 'z') c = static_cast<char>(c - 'a' + 'A');
    }
    return s;
}

std::vector<std::string> word_tokens(const std::string &text) {
    std::vector<std::string> out;
    std::string current;
    const std::size_t n = text.size();
    for (std::size_t i = 0; i < n; ++i) {
        const unsigned char c = static_cast<unsigned char>(text[i]);
        if (c >= 0x80) {
            std::uint32_t cp;
            const std::size_t len = utf8_sequence(text, i, cp);
            if (unicode_separator(cp)) {
                if (!current.empty()) {
                    out.push_back(std::move(current));
                    current.clear();
                }
            } else {
                current.append(text, i, len);
            }
            i += len - 1;
        } else if (word_byte(c)) {
            current += static_cast<char>(c);
        } else if (c == '\'' && !current.empty() && i + 1 < n && word_byte(static_cast<unsigned char>(text[i + 1]))) {
            current += '\'';  // inner apostrophe: don't, it's
        } else if (!current.empty()) {
            out.push_back(std::move(current));
            current.clear();
        }
    }
    if (!current.empty()) {
        out.push_back(std::move(current));
    }
    return out;
}

namespace {

void skip_grams(const std::vector<std::string> &words, std::size_t n, std::size_t skip, std::vector<std::string> &out) {
    // All n-word sequences in order whose total gap is at most `skip`.
    std::vector<std::size_t> pick;
    std::function<void(std::size_t, std::size_t)> extend = [&](std::size_t next, std::size_t budget) {
        if (pick.size() == n) {
            std::string gram = words[pick[0]];
            for (std::size_t i = 1; i < pick.size(); ++i) {
                gram += ' ';
                gram += words[pick[i]];
            }
            out.push_back(std::move(gram));
            return;
        }
        for (std::size_t gap = 0; gap <= budget && next + gap < words.size(); ++gap) {
            pick.push_back(next + gap);
            extend(next + gap + 1, budget - gap);
            pick.pop_back();
        }
    };
    for (std::size_t start = 0; start < words.size(); ++start) {
        pick.assign(1, start);
        extend(start + 1, skip);
    }
}

}  // namespace

std::vector<std::string> Tokenizer::tokenize(const std::string &text) const {
    switch (kind) {
        case Kind::Word:
            return word_tokens(text);
        case Kind::Stemmer: {
            std::vector<std::string> words = word_tokens(text);
            for (std::string &w : words) {
                w = porter_stem(to_lower_ascii(w));
            }
            return words;
        }
        case Kind::Whitespace: {
            std::vector<std::string> out;
            std::size_t start = 0;
            while (start <= text.size()) {
                const std::size_t pos = text.find(delimiter, start);
                const std::size_t end = pos == std::string::npos ? text.size() : pos;
                if (end > start) {
                    out.push_back(text.substr(start, end - start));
                }
                if (pos == std::string::npos) break;
                start = pos + delimiter.size();
            }
            return out;
        }
        case Kind::Sentence: {
            std::vector<std::string> out;
            std::string current;
            auto flush = [&] {
                const std::size_t a = current.find_first_not_of(" \t\r\n");
                if (a != std::string::npos) {
                    const std::size_t b = current.find_last_not_of(" \t\r\n");
                    out.push_back(current.substr(a, b - a + 1));
                }
                current.clear();
            };
            for (std::size_t i = 0; i < text.size(); ++i) {
                current += text[i];
                const char c = text[i];
                if ((c == '.' || c == '!' || c == '?') &&
                    (i + 1 == text.size() || std::isspace(static_cast<unsigned char>(text[i + 1])))) {
                    flush();
                }
            }
            flush();
            return out;
        }
        case Kind::NGram:
        case Kind::SkipGram: {
            const std::vector<std::string> words = word_tokens(text);
            std::vector<std::string> out;
            for (std::size_t n = min; n <= max; ++n) {
                if (kind == Kind::SkipGram) {
                    skip_grams(words, n, skip, out);
                    continue;
                }
                for (std::size_t i = 0; i + n <= words.size(); ++i) {
                    std::string gram = words[i];
                    for (std::size_t j = 1; j < n; ++j) {
                        gram += ' ';
                        gram += words[i + j];
                    }
                    out.push_back(std::move(gram));
                }
            }
            return out;
        }
    }
    return {};
}

void Tokenizer::save(ArchiveWriter &w) const {
    w.u8(static_cast<std::uint8_t>(kind));
    w.str(delimiter);
    w.size(min);
    w.size(max);
    w.size(skip);
}

void Tokenizer::load(ArchiveReader &r) {
    const std::uint8_t k = r.u8();
    ArchiveReader::require(k <= 5, "bad tokenizer");
    kind = static_cast<Kind>(k);
    delimiter = r.str();
    min = r.size(64);
    max = r.size(64);
    skip = r.size(64);
    ArchiveReader::require(min >= 1 && max >= min && !delimiter.empty(), "bad tokenizer parameters");
}

// ---- hashes ---------------------------------------------------------------------------------

std::uint32_t murmur3(const std::string &s, std::uint32_t seed) {
    const auto *data = reinterpret_cast<const unsigned char *>(s.data());
    const std::size_t len = s.size(), blocks = len / 4;
    std::uint32_t h = seed;
    const std::uint32_t c1 = 0xcc9e2d51, c2 = 0x1b873593;
    auto rotl = [](std::uint32_t x, int r) { return (x << r) | (x >> (32 - r)); };
    for (std::size_t i = 0; i < blocks; ++i) {
        std::uint32_t k;
        std::memcpy(&k, data + i * 4, 4);
        k *= c1;
        k = rotl(k, 15);
        k *= c2;
        h ^= k;
        h = rotl(h, 13);
        h = h * 5 + 0xe6546b64;
    }
    std::uint32_t k = 0;
    const unsigned char *tail = data + blocks * 4;
    switch (len & 3) {
        case 3: k ^= static_cast<std::uint32_t>(tail[2]) << 16; [[fallthrough]];
        case 2: k ^= static_cast<std::uint32_t>(tail[1]) << 8; [[fallthrough]];
        case 1:
            k ^= tail[0];
            k *= c1;
            k = rotl(k, 15);
            k *= c2;
            h ^= k;
    }
    h ^= static_cast<std::uint32_t>(len);
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
}

std::uint32_t crc32(const std::string &s) {
    std::uint32_t crc = 0xFFFFFFFFu;
    for (unsigned char c : s) {
        crc ^= c;
        for (int i = 0; i < 8; ++i) {
            crc = (crc >> 1) ^ (0xEDB88320u & (0u - (crc & 1u)));
        }
    }
    return ~crc;
}

std::uint32_t fnv1a(const std::string &s) {
    std::uint32_t h = 2166136261u;
    for (unsigned char c : s) {
        h ^= c;
        h *= 16777619u;
    }
    return h;
}

// ---- WordCounter ------------------------------------------------------------------------------

void WordCounter::fit(const std::vector<std::vector<std::string>> &documents) {
    if (documents.empty()) {
        throw std::invalid_argument("Documents must not be empty");
    }
    std::map<std::string, std::size_t> df;
    for (const auto &doc : documents) {
        std::set<std::string> seen(doc.begin(), doc.end());
        for (const std::string &t : seen) {
            ++df[t];
        }
    }
    const double n = static_cast<double>(documents.size());
    std::vector<std::pair<std::size_t, std::string>> ranked;
    for (const auto &[token, count] : df) {
        if (count >= min_document_count && static_cast<double>(count) / n <= max_document_ratio) {
            ranked.emplace_back(count, token);
        }
    }
    std::stable_sort(ranked.begin(), ranked.end(), [](const auto &a, const auto &b) { return a.first > b.first; });
    if (ranked.size() > max_vocabulary) {
        ranked.resize(max_vocabulary);
    }
    std::vector<std::string> vocab;
    for (auto &entry : ranked) {
        vocab.push_back(std::move(entry.second));
    }
    std::sort(vocab.begin(), vocab.end());
    if (vocab.empty()) {
        throw std::invalid_argument("The vocabulary is empty; relax minDocumentCount or maxDocumentRatio");
    }
    std::map<std::string, std::size_t> idx;
    for (std::size_t i = 0; i < vocab.size(); ++i) {
        idx.emplace(vocab[i], i);
    }
    vocabulary = std::move(vocab);
    index = std::move(idx);
}

void WordCounter::transform(const std::vector<std::string> &tokens, double *out) const {
    std::fill(out, out + vocabulary.size(), 0.0);
    for (const std::string &t : tokens) {
        const auto it = index.find(t);
        if (it != index.end()) {
            out[it->second] += 1.0;
        }
    }
}

void WordCounter::save(ArchiveWriter &w) const {
    w.size(max_vocabulary);
    w.size(min_document_count);
    w.f64(max_document_ratio);
    w.strings(vocabulary);
}

void WordCounter::load(ArchiveReader &r) {
    max_vocabulary = r.size();
    min_document_count = r.size();
    max_document_ratio = r.f64();
    vocabulary = r.strings();
    index.clear();
    for (std::size_t i = 0; i < vocabulary.size(); ++i) {
        ArchiveReader::require(index.emplace(vocabulary[i], i).second, "duplicate vocabulary entry");
    }
}

// ---- TF-IDF and BM25 -----------------------------------------------------------------------

namespace {
std::vector<double> document_frequencies(const Matrix &counts) {
    std::vector<double> df(counts.cols, 0.0);
    for (std::size_t r = 0; r < counts.rows; ++r) {
        for (std::size_t c = 0; c < counts.cols; ++c) {
            if (counts(r, c) < 0.0) {
                throw std::invalid_argument("Term counts must be non-negative");
            }
            df[c] += counts(r, c) > 0.0;
        }
    }
    return df;
}
}  // namespace

void TfIdf::fit(const Matrix &counts) {
    require_training_set(counts, counts.rows);
    const std::vector<double> df = document_frequencies(counts);
    const double n = static_cast<double>(counts.rows);
    std::vector<double> weights(counts.cols);
    for (std::size_t c = 0; c < counts.cols; ++c) {
        weights[c] = std::log((n + smoothing) / (df[c] + smoothing)) + 1.0;
    }
    idf = std::move(weights);
}

void TfIdf::transform(const double *in, double *out) const {
    for (std::size_t c = 0; c < idf.size(); ++c) {
        const double tf = dampening && in[c] > 0.0 ? 1.0 + std::log(in[c]) : in[c];
        out[c] = tf * idf[c];
    }
}

void TfIdf::save(ArchiveWriter &w) const {
    w.f64(smoothing);
    w.boolean(dampening);
    w.doubles(idf);
}

void TfIdf::load(ArchiveReader &r) {
    smoothing = r.f64();
    dampening = r.boolean();
    idf = r.doubles();
}

void BM25::fit(const Matrix &counts) {
    require_training_set(counts, counts.rows);
    const std::vector<double> df = document_frequencies(counts);
    const double n = static_cast<double>(counts.rows);
    std::vector<double> weights(counts.cols);
    for (std::size_t c = 0; c < counts.cols; ++c) {
        weights[c] = std::log(1.0 + (n - df[c] + 0.5) / (df[c] + 0.5));
    }
    double total = 0.0;
    for (double v : counts.data) {
        total += v;
    }
    idf = std::move(weights);
    average_length = std::max(total / n, 1e-12);
}

void BM25::transform(const double *in, double *out) const {
    double length = 0.0;
    for (std::size_t c = 0; c < idf.size(); ++c) {
        length += in[c];
    }
    const double norm = k1 * (1.0 - b + b * length / average_length);
    for (std::size_t c = 0; c < idf.size(); ++c) {
        out[c] = idf[c] * in[c] * (k1 + 1.0) / (in[c] + norm);
    }
}

void BM25::save(ArchiveWriter &w) const {
    w.f64(k1);
    w.f64(b);
    w.doubles(idf);
    w.f64(average_length);
}

void BM25::load(ArchiveReader &r) {
    k1 = r.f64();
    b = r.f64();
    idf = r.doubles();
    average_length = r.f64();
    ArchiveReader::require(average_length > 0.0 || idf.empty(), "bad average length");
}

}  // namespace mlcore
