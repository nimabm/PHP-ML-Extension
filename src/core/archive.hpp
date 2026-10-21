#pragma once

// Compact binary (de)serialization for model state. Every payload starts with a
// magic number, a per-class tag and a format version, and every read is bounds
// checked, so truncated or forged data raises std::invalid_argument instead of
// reading out of bounds. Models still validate their own invariants in load().

#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

#include "common.hpp"

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "mlnative's archive format assumes a little-endian target"
#endif

namespace mlcore {

constexpr std::uint32_t kArchiveMagic = 0x4D4C4E31;  // "MLN1"
constexpr std::uint32_t kArchiveFormat = 1;

// Four-character class tag, e.g. tag("CTRE").
constexpr std::uint32_t tag(const char (&s)[5]) {
    return static_cast<std::uint32_t>(s[0]) | static_cast<std::uint32_t>(s[1]) << 8 |
           static_cast<std::uint32_t>(s[2]) << 16 | static_cast<std::uint32_t>(s[3]) << 24;
}

class ArchiveWriter {
public:
    explicit ArchiveWriter(std::uint32_t class_tag) {
        u32(kArchiveMagic);
        u32(class_tag);
        u32(kArchiveFormat);
    }

    void u8(std::uint8_t v) { raw(&v, sizeof v); }
    void u32(std::uint32_t v) { raw(&v, sizeof v); }
    void u64(std::uint64_t v) { raw(&v, sizeof v); }
    void i64(std::int64_t v) { raw(&v, sizeof v); }
    void f64(double v) { raw(&v, sizeof v); }
    void boolean(bool v) { u8(v ? 1 : 0); }
    void size(std::size_t v) { u64(static_cast<std::uint64_t>(v)); }

    void str(const std::string &s) {
        size(s.size());
        raw(s.data(), s.size());
    }

    void doubles(const std::vector<double> &v) {
        size(v.size());
        raw(v.data(), v.size() * sizeof(double));
    }

    void ints(const std::vector<int> &v) {
        size(v.size());
        raw(v.data(), v.size() * sizeof(int));
    }

    void sizes(const std::vector<std::size_t> &v) {
        size(v.size());
        for (std::size_t x : v) {
            size(x);
        }
    }

    void strings(const std::vector<std::string> &v) {
        size(v.size());
        for (const std::string &s : v) {
            str(s);
        }
    }

    void matrix(const Matrix &m) {
        size(m.rows);
        size(m.cols);
        raw(m.data.data(), m.data.size() * sizeof(double));
    }

    const std::string &data() const { return buffer_; }

private:
    void raw(const void *p, std::size_t n) { buffer_.append(static_cast<const char *>(p), n); }
    std::string buffer_;
};

class ArchiveReader {
public:
    ArchiveReader(const char *data, std::size_t length, std::uint32_t class_tag) : p_(data), end_(data + length) {
        if (u32() != kArchiveMagic) {
            fail("not an mlnative payload");
        }
        if (u32() != class_tag) {
            fail("payload belongs to a different class");
        }
        if (u32() != kArchiveFormat) {
            fail("unsupported format version");
        }
    }

    std::uint8_t u8() { return pod<std::uint8_t>(); }
    std::uint32_t u32() { return pod<std::uint32_t>(); }
    std::uint64_t u64() { return pod<std::uint64_t>(); }
    std::int64_t i64() { return pod<std::int64_t>(); }
    double f64() { return pod<double>(); }

    bool boolean() {
        const std::uint8_t v = u8();
        if (v > 1) {
            fail("bad boolean");
        }
        return v == 1;
    }

    // A count of elements that will be read next; rejects counts that cannot
    // fit in the remaining bytes so forged sizes cannot trigger huge allocations.
    std::size_t count(std::size_t element_size) {
        const std::uint64_t n = u64();
        if (element_size != 0 && n > remaining() / element_size) {
            fail("truncated");
        }
        return static_cast<std::size_t>(n);
    }

    // A size/index value; `max` bounds it (inclusive).
    std::size_t size(std::size_t max = std::numeric_limits<std::size_t>::max()) {
        const std::uint64_t n = u64();
        if (n > max) {
            fail("value out of range");
        }
        return static_cast<std::size_t>(n);
    }

    std::string str() {
        const std::size_t n = count(1);
        std::string s(p_, n);
        p_ += n;
        return s;
    }

    std::vector<double> doubles() {
        const std::size_t n = count(sizeof(double));
        std::vector<double> v(n);
        std::memcpy(v.data(), p_, n * sizeof(double));
        p_ += n * sizeof(double);
        return v;
    }

    std::vector<double> doubles(std::size_t expected) {
        std::vector<double> v = doubles();
        if (v.size() != expected) {
            fail("inconsistent array size");
        }
        return v;
    }

    std::vector<int> ints() {
        const std::size_t n = count(sizeof(int));
        std::vector<int> v(n);
        std::memcpy(v.data(), p_, n * sizeof(int));
        p_ += n * sizeof(int);
        return v;
    }

    // Integers that must all lie in [lo, hi].
    std::vector<int> ints_in(int lo, int hi) {
        std::vector<int> v = ints();
        for (int x : v) {
            if (x < lo || x > hi) {
                fail("index out of range");
            }
        }
        return v;
    }

    std::vector<std::size_t> sizes(std::size_t max = std::numeric_limits<std::size_t>::max()) {
        const std::size_t n = count(sizeof(std::uint64_t));
        std::vector<std::size_t> v(n);
        for (std::size_t &x : v) {
            x = size(max);
        }
        return v;
    }

    std::vector<std::string> strings() {
        const std::size_t n = count(sizeof(std::uint64_t));
        std::vector<std::string> v;
        v.reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
            v.push_back(str());
        }
        return v;
    }

    Matrix matrix() {
        const std::uint64_t rows = u64();
        const std::uint64_t cols = u64();
        if (cols != 0 && rows > remaining() / sizeof(double) / cols) {
            fail("truncated");
        }
        Matrix m(static_cast<std::size_t>(rows), static_cast<std::size_t>(cols));
        std::memcpy(m.data.data(), p_, m.data.size() * sizeof(double));
        p_ += m.data.size() * sizeof(double);
        return m;
    }

    Matrix matrix(std::size_t rows, std::size_t cols) {
        Matrix m = matrix();
        if (m.rows != rows || m.cols != cols) {
            fail("inconsistent matrix shape");
        }
        return m;
    }

    void finish() const {
        if (p_ != end_) {
            fail("trailing bytes");
        }
    }

    [[noreturn]] static void fail(const std::string &why) {
        throw std::invalid_argument("Invalid serialized data: " + why);
    }

    // Throws unless `condition` holds; used by models to validate invariants.
    static void require(bool condition, const char *why = "inconsistent model state") {
        if (!condition) {
            fail(why);
        }
    }

private:
    std::size_t remaining() const { return static_cast<std::size_t>(end_ - p_); }

    template <typename T>
    T pod() {
        static_assert(std::is_trivially_copyable<T>::value, "pod only");
        if (remaining() < sizeof(T)) {
            fail("truncated");
        }
        T v;
        std::memcpy(&v, p_, sizeof(T));
        p_ += sizeof(T);
        return v;
    }

    const char *p_;
    const char *end_;
};

}  // namespace mlcore
