// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "unicode.hpp"
#include "unicode-data.hpp"

#include <algorithm>
#include <cassert>
#include <codecvt>
#include <cstddef>
#include <cstdint>
#include <locale>
#include <map>
#include <regex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cv { namespace dnn {

uint32_t unicode_cpt_from_utf8(const std::string & utf8, size_t & offset) {
    assert(offset < utf8.size());
    if (!(utf8[offset + 0] & 0x80)) {
        auto result = utf8[offset + 0];
        offset += 1;
        return result;
    }
    if (!(utf8[offset + 0] & 0x40)) {
        throw std::invalid_argument("invalid character");
    }
    if (!(utf8[offset + 0] & 0x20)) {
        if (offset + 1 >= utf8.size() || ! ((utf8[offset + 1] & 0xc0) == 0x80)) {
            throw std::invalid_argument("invalid character");
        }
        auto result = ((utf8[offset + 0] & 0x1f) << 6) | (utf8[offset + 1] & 0x3f);
        offset += 2;
        return result;
    }
    if (!(utf8[offset + 0] & 0x10)) {
        if (offset + 2 >= utf8.size() || ! ((utf8[offset + 1] & 0xc0) == 0x80) || ! ((utf8[offset + 2] & 0xc0) == 0x80)) {
            throw std::invalid_argument("invalid character");
        }
        auto result = ((utf8[offset + 0] & 0x0f) << 12) | ((utf8[offset + 1] & 0x3f) << 6) | (utf8[offset + 2] & 0x3f);
        offset += 3;
        return result;
    }
    if (!(utf8[offset + 0] & 0x08)) {
        if (offset + 3 >= utf8.size() || ! ((utf8[offset + 1] & 0xc0) == 0x80) || ! ((utf8[offset + 2] & 0xc0) == 0x80) || !((utf8[offset + 3] & 0xc0) == 0x80)) {
            throw std::invalid_argument("invalid character");
        }
        auto result = ((utf8[offset + 0] & 0x07) << 18) | ((utf8[offset + 1] & 0x3f) << 12) | ((utf8[offset + 2] & 0x3f) << 6) | (utf8[offset + 3] & 0x3f);
        offset += 4;
        return result;
    }
    throw std::invalid_argument("failed to convert utf8 to codepoint");
}

static std::vector<unicode_cpt_flags> unicode_cpt_flags_array() {
    std::vector<unicode_cpt_flags> cpt_flags(MAX_CODEPOINTS, unicode_cpt_flags::UNDEFINED);

    assert (unicode_ranges_flags.begin()[0].first == 0);
    assert (unicode_ranges_flags.begin()[unicode_ranges_flags.size()-1].first == MAX_CODEPOINTS);
    for (size_t i = 1; i < unicode_ranges_flags.size(); ++i) {
        const auto range_ini = unicode_ranges_flags.begin()[i-1];  // codepoint_ini, flags
        const auto range_end = unicode_ranges_flags.begin()[i];    // codepoint_end, flags
        for (uint32_t cpt = range_ini.first; cpt < range_end.first; ++cpt) {
            cpt_flags[cpt] = range_ini.second;
        }
    }

    for (auto cpt : unicode_set_whitespace) {
        cpt_flags[cpt].is_whitespace = true;
    }

    for (auto p : unicode_map_lowercase) {
        cpt_flags[p.second].is_lowercase = true;
    }

    for (auto p : unicode_map_uppercase) {
        cpt_flags[p.second].is_uppercase = true;
    }

    for (auto &range : unicode_ranges_nfd) {  // start, last, nfd
        cpt_flags[range.nfd].is_nfd = true;
    }

    return cpt_flags;
}

static std::unordered_map<uint8_t, std::string> unicode_byte_to_utf8_map() {
    std::unordered_map<uint8_t, std::string> map;
    for (int ch = 0x21; ch <= 0x7E; ++ch) {  // u'!' to u'~'
        assert(0 <= ch && ch < 256);
        map[ch] = unicode_cpt_to_utf8(ch);
    }
    for (int ch = 0xA1; ch <= 0xAC; ++ch) {  // u'¡' to u'¬'
        assert(0 <= ch && ch < 256);
        map[ch] = unicode_cpt_to_utf8(ch);
    }
    for (int ch = 0xAE; ch <= 0xFF; ++ch) {  // u'®' to u'ÿ'
        assert(0 <= ch && ch < 256);
        map[ch] = unicode_cpt_to_utf8(ch);
    }
    auto n = 0;
    for (int ch = 0; ch < 256; ++ch) {
        if (map.find(ch) == map.end()) {
            map[ch] = unicode_cpt_to_utf8(256 + n);
            ++n;
        }
    }
    return map;
}

static std::unordered_map<std::string, uint8_t> unicode_utf8_to_byte_map() {
    std::unordered_map<std::string, uint8_t> map;
    for (int ch = 0x21; ch <= 0x7E; ++ch) {  // u'!' to u'~'
        assert(0 <= ch && ch < 256);
        map[unicode_cpt_to_utf8(ch)] = ch;
    }
    for (int ch = 0xA1; ch <= 0xAC; ++ch) {  // u'¡' to u'¬'
        assert(0 <= ch && ch < 256);
        map[unicode_cpt_to_utf8(ch)] = ch;
    }
    for (int ch = 0xAE; ch <= 0xFF; ++ch) {  // u'®' to u'ÿ'
        assert(0 <= ch && ch < 256);
        map[unicode_cpt_to_utf8(ch)] = ch;
    }
    auto n = 0;
    for (int ch = 0; ch < 256; ++ch) {
        if (map.find(unicode_cpt_to_utf8(ch)) == map.end()) {
            map[unicode_cpt_to_utf8(256 + n)] = ch;
            ++n;
        }
    }
    return map;
}

static inline std::wstring unicode_wstring_from_utf8(const std::string & s) {
#if defined(__clang__)
    // disable C++17 deprecation warning for std::codecvt_utf8
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wdeprecated-declarations"
#elif defined(__GNUC__)
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

    std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;

#if defined(__clang__)
#    pragma clang diagnostic pop
#elif defined(__GNUC__)
#    pragma GCC diagnostic pop
#endif

    return conv.from_bytes(s);
}

static std::vector<std::string> unicode_byte_encoding_process(const std::vector<std::string> & bpe_words) {
    std::vector<std::string> bpe_encoded_words;
    for (const auto & word : bpe_words) {
        std::string text_utf;
        auto utf_word =  unicode_cpts_from_utf8(word);
        for (size_t i = 0; i < utf_word.size(); ++i) {
            text_utf += unicode_cpt_to_utf8(utf_word[i]);
        }

        std::string encoded_token;
        for (char & c : text_utf) {
            encoded_token += unicode_byte_to_utf8(c);
        }
        bpe_encoded_words.emplace_back(encoded_token);
    }
    return bpe_encoded_words;
}

static const uint32_t UNICODE_CUSTOM_SPLIT_OUT_OF_RANGE = 0xFFFFFFFF;

// Free functions (no lambdas, no functors) that read/emit tokens within [offset_ini, offset_end).
// Shared by unicode_regex_split_custom_gpt2() and unicode_regex_split_custom_llama3().
static uint32_t unicode_custom_split_get_cpt(const std::vector<uint32_t> & cpts,
                                              size_t offset_ini, size_t offset_end, size_t pos) {
    return (offset_ini <= pos && pos < offset_end) ? cpts[pos] : UNICODE_CUSTOM_SPLIT_OUT_OF_RANGE;
}

static unicode_cpt_flags unicode_custom_split_get_flags(const std::vector<uint32_t> & cpts,
                                                         size_t offset_ini, size_t offset_end, size_t pos) {
    return (offset_ini <= pos && pos < offset_end) ? unicode_cpt_flags_from_cpt(cpts[pos]) : unicode_cpt_flags{};
}

static size_t unicode_custom_split_add_token(std::vector<size_t> & bpe_offsets,
                                              size_t & prev_end, size_t offset_end, size_t end) {
    assert(prev_end <= end && end <= offset_end);
    size_t len = end - prev_end;
    if (len > 0) {
        bpe_offsets.push_back(len);
    }
    prev_end = end;
    return len;
}

// GPT2 system regex:  's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
static std::vector<size_t> unicode_regex_split_custom_gpt2(const std::string & text, const std::vector<size_t> & offsets) {
    std::vector<size_t> bpe_offsets; // store the offset of each word
    bpe_offsets.reserve(offsets.size()); // Reserve memory for the approximate size

    const auto cpts = unicode_cpts_from_utf8(text);

    size_t start = 0;
    for (auto offset : offsets) {
        const size_t offset_ini = start;
        const size_t offset_end = start + offset;
        assert(offset_end <= cpts.size());
        start = offset_end;

        size_t _prev_end = offset_ini;

        for (size_t pos = offset_ini; pos < offset_end; /*pos++*/ ) {
            const uint32_t cpt = unicode_custom_split_get_cpt(cpts, offset_ini, offset_end, pos);
            const auto flags = unicode_custom_split_get_flags(cpts, offset_ini, offset_end, pos);

            // regex: 's|'t|'re|'ve|'m|'ll|'d
            if (cpt == '\'' && pos+1 < offset_end) {
                uint32_t cpt_next = unicode_custom_split_get_cpt(cpts, offset_ini, offset_end, pos+1);
                if (cpt_next == 's' || cpt_next == 't' || cpt_next == 'm' || cpt_next == 'd') {
                    pos += unicode_custom_split_add_token(bpe_offsets, _prev_end, offset_end, pos+2);
                    continue;
                }
                if (pos+2 < offset_end) {
                    uint32_t cpt_next_next = unicode_custom_split_get_cpt(cpts, offset_ini, offset_end, pos+2);
                    if ((cpt_next == 'r' && cpt_next_next == 'e') ||
                        (cpt_next == 'v' && cpt_next_next == 'e') ||
                        (cpt_next == 'l' && cpt_next_next == 'l')) {
                        pos += unicode_custom_split_add_token(bpe_offsets, _prev_end, offset_end, pos+3);
                        continue;
                    }
                }
            }

            auto flags2 = (cpt == ' ' ? unicode_custom_split_get_flags(cpts, offset_ini, offset_end, pos+1) : flags);
            // regex: <space>?\p{L}+
            if (flags2.is_letter) {
                pos += (cpt == ' ');
                while (flags2.is_letter) {
                    flags2 = unicode_custom_split_get_flags(cpts, offset_ini, offset_end, ++pos);
                }
                unicode_custom_split_add_token(bpe_offsets, _prev_end, offset_end, pos);
                continue;
            }
            // regex: <space>?\p{N}+
            if (flags2.is_number) {
                pos += (cpt == ' ');
                while (flags2.is_number) {
                    flags2 = unicode_custom_split_get_flags(cpts, offset_ini, offset_end, ++pos);
                }
                unicode_custom_split_add_token(bpe_offsets, _prev_end, offset_end, pos);
                continue;
            }
            // regex: <space>?[^\s\p{L}\p{N}]+
            if (!(flags2.is_whitespace | flags2.is_letter | flags2.is_number) && flags2.as_uint()) {
                pos += (cpt == ' ');
                while (!(flags2.is_whitespace | flags2.is_letter | flags2.is_number) && flags2.as_uint()) {
                    flags2 = unicode_custom_split_get_flags(cpts, offset_ini, offset_end, ++pos);
                }
                unicode_custom_split_add_token(bpe_offsets, _prev_end, offset_end, pos);
                continue;
            }

            size_t num_whitespaces = 0;
            while (unicode_custom_split_get_flags(cpts, offset_ini, offset_end, pos+num_whitespaces).is_whitespace) {
                num_whitespaces++;
            }

            // regex: \s+(?!\S)
            if (num_whitespaces > 1 && unicode_custom_split_get_cpt(cpts, offset_ini, offset_end, pos+num_whitespaces) != UNICODE_CUSTOM_SPLIT_OUT_OF_RANGE) {
                pos += num_whitespaces - 1;
                unicode_custom_split_add_token(bpe_offsets, _prev_end, offset_end, pos);
                continue;
            }

            // regex: \s+
            if (num_whitespaces > 0) {
                pos += num_whitespaces;
                unicode_custom_split_add_token(bpe_offsets, _prev_end, offset_end, pos);
                continue;
            }

            // no matches
            unicode_custom_split_add_token(bpe_offsets, _prev_end, offset_end, ++pos);
        }
    }

    return bpe_offsets;
}

// LLAMA3 system regex: "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
static std::vector<size_t> unicode_regex_split_custom_llama3(const std::string & text, const std::vector<size_t> & offsets) {
    std::vector<size_t> bpe_offsets; // store the offset of each word
    bpe_offsets.reserve(offsets.size()); // Reserve memory for the approximate size

    const auto cpts = unicode_cpts_from_utf8(text);

    size_t start = 0;
    for (auto offset : offsets) {
        const size_t offset_ini = start;
        const size_t offset_end = start + offset;
        assert(offset_end <= cpts.size());
        start = offset_end;

        size_t _prev_end = offset_ini;

        for (size_t pos = offset_ini; pos < offset_end; /*pos++*/ ) {
            const uint32_t cpt = unicode_custom_split_get_cpt(cpts, offset_ini, offset_end, pos);
            const auto flags = unicode_custom_split_get_flags(cpts, offset_ini, offset_end, pos);

            // regex: (?i:'s|'t|'re|'ve|'m|'ll|'d) // case insensitive
            if (cpt == '\'' && pos+1 < offset_end) {
                uint32_t cpt_next = unicode_tolower(unicode_custom_split_get_cpt(cpts, offset_ini, offset_end, pos+1));
                if (cpt_next == 's' || cpt_next == 't' || cpt_next == 'm' || cpt_next == 'd') {
                    pos += unicode_custom_split_add_token(bpe_offsets, _prev_end, offset_end, pos+2);
                    continue;
                }
                if (pos+2 < offset_end) {
                    uint32_t cpt_next_next = unicode_tolower(unicode_custom_split_get_cpt(cpts, offset_ini, offset_end, pos+2));
                    if ((cpt_next == 'r' && cpt_next_next == 'e') ||
                        (cpt_next == 'v' && cpt_next_next == 'e') ||
                        (cpt_next == 'l' && cpt_next_next == 'l')) {
                        pos += unicode_custom_split_add_token(bpe_offsets, _prev_end, offset_end, pos+3);
                        continue;
                    }
                }
            }

            // regex: [^\r\n\p{L}\p{N}]?\p{L}+
            if (!(cpt == '\r' || cpt == '\n' || flags.is_number)) {
                if (flags.is_letter || unicode_custom_split_get_flags(cpts, offset_ini, offset_end, pos+1).is_letter) {  // one or more letters
                    pos++;
                    while (unicode_custom_split_get_flags(cpts, offset_ini, offset_end, pos).is_letter) {
                        pos++;
                    }
                    unicode_custom_split_add_token(bpe_offsets, _prev_end, offset_end, pos);
                    continue;
                }
            }

            // regex: \p{N}{1,3}
            if (flags.is_number) {
                size_t ini = pos;
                while (unicode_custom_split_get_flags(cpts, offset_ini, offset_end, pos).is_number) {
                    if (++pos - ini >= 3 ) {
                        unicode_custom_split_add_token(bpe_offsets, _prev_end, offset_end, pos);
                        ini = pos;
                    }
                }
                unicode_custom_split_add_token(bpe_offsets, _prev_end, offset_end, pos);
                continue;
            }

            // regex: <space>?[^\s\p{L}\p{N}]+[\r\n]*
            auto flags2 = (cpt == ' ' ? unicode_custom_split_get_flags(cpts, offset_ini, offset_end, pos+1) : flags);
            if (!(flags2.is_whitespace | flags2.is_letter | flags2.is_number) && flags.as_uint()) {
                pos += (cpt == ' ');
                while (!(flags2.is_whitespace | flags2.is_letter | flags2.is_number) && flags2.as_uint()) {
                    flags2 = unicode_custom_split_get_flags(cpts, offset_ini, offset_end, ++pos);
                }
                uint32_t cpt2 = unicode_custom_split_get_cpt(cpts, offset_ini, offset_end, pos);
                while (cpt2 == '\r' || cpt2 == '\n') {
                    cpt2 = unicode_custom_split_get_cpt(cpts, offset_ini, offset_end, ++pos);
                }
                unicode_custom_split_add_token(bpe_offsets, _prev_end, offset_end, pos);
                continue;
            }

            size_t num_whitespaces = 0;
            size_t last_end_r_or_n = 0;
            while (unicode_custom_split_get_flags(cpts, offset_ini, offset_end, pos+num_whitespaces).is_whitespace) {
                uint32_t cpt2 = unicode_custom_split_get_cpt(cpts, offset_ini, offset_end, pos+num_whitespaces);
                if (cpt2 == '\r' || cpt2 == '\n') {
                    last_end_r_or_n = pos + num_whitespaces + 1;
                }
                num_whitespaces++;
            }

            // regex: \s*[\r\n]+
            if (last_end_r_or_n > 0) {
                pos = last_end_r_or_n;
                unicode_custom_split_add_token(bpe_offsets, _prev_end, offset_end, pos);
                continue;
            }

            // regex: \s+(?!\S)
            if (num_whitespaces > 1 && unicode_custom_split_get_cpt(cpts, offset_ini, offset_end, pos+num_whitespaces) != UNICODE_CUSTOM_SPLIT_OUT_OF_RANGE) {
                pos += num_whitespaces - 1;
                unicode_custom_split_add_token(bpe_offsets, _prev_end, offset_end, pos);
                continue;
            }

            // regex: \s+
            if (num_whitespaces > 0) {
                pos += num_whitespaces;
                unicode_custom_split_add_token(bpe_offsets, _prev_end, offset_end, pos);
                continue;
            }

            // no matches
            unicode_custom_split_add_token(bpe_offsets, _prev_end, offset_end, ++pos);
        }
    }

    return bpe_offsets;
}

// use std::wregex to split the text
static std::vector<size_t> unicode_regex_split_stl(const std::wstring & wtext, const std::wstring & regex_expr, const std::vector<size_t> & offsets) {
    std::wregex expr(regex_expr);
    std::vector<size_t> bpe_offsets; // store the offset of each word
    bpe_offsets.reserve(offsets.size()); // Reserve memory for the approximate size
    size_t start = 0;
    for (auto offset : offsets) {
        std::wcregex_iterator it(wtext.data() + start, wtext.data() + start + offset, expr);
        std::wcregex_iterator end;

        int64_t start_idx = 0;
        while (it != end) {
            std::wcmatch& match = const_cast<std::wcmatch&>(*it);
            if (match.position() > start_idx) {
                bpe_offsets.emplace_back(match.position() - start_idx);
            }
            bpe_offsets.emplace_back(match.length());
            start_idx = match.position() + match.length();
            ++it;
        }

        if (start_idx < (int64_t) offset) {
            bpe_offsets.emplace_back(offset - start_idx);
        }
        start += offset;
    }

    return bpe_offsets;
}

// use std::regex to split the text
static std::vector<size_t> unicode_regex_split_stl(const std::string & text, const std::string & regex_expr, const std::vector<size_t> & offsets) {
    std::regex expr(regex_expr);
    std::vector<size_t> bpe_offsets; // store the offset of each word
    bpe_offsets.reserve(offsets.size()); // Reserve memory for the approximate size
    size_t start = 0;
    for (auto offset : offsets) {
        std::cregex_iterator it(text.data() + start, text.data() + start + offset, expr);
        std::cregex_iterator end;

        int64_t start_idx = 0;
        while (it != end) {
            std::cmatch& match = const_cast<std::cmatch&>(*it);
            if (match.position() > start_idx) {
                bpe_offsets.emplace_back(match.position() - start_idx);
            }
            bpe_offsets.emplace_back(match.length());
            start_idx = match.position() + match.length();
            ++it;
        }

        if (start_idx < (int64_t) offset) {
            bpe_offsets.emplace_back(offset - start_idx);
        }
        start += offset;
    }

    return bpe_offsets;
}

static std::vector<size_t> unicode_regex_split_custom(const std::string & text, const std::string & regex_expr, const std::vector<size_t> & offsets) {
    std::vector<size_t> bpe_offsets;

    if (regex_expr == "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)") {
        bpe_offsets = unicode_regex_split_custom_gpt2(text, offsets);
    } else if (
            regex_expr == "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+" ||
            regex_expr == "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+") {

        bpe_offsets = unicode_regex_split_custom_llama3(text, offsets);
    }

    return bpe_offsets;
}

//
// interface
//

std::string unicode_cpt_to_utf8(uint32_t cpt) {
    std::string result;

    if (/* 0x00 <= cpt && */ cpt <= 0x7f) {
        result.push_back(cpt);
        return result;
    }
    if (0x80 <= cpt && cpt <= 0x7ff) {
        result.push_back(0xc0 | ((cpt >> 6) & 0x1f));
        result.push_back(0x80 | (cpt & 0x3f));
        return result;
    }
    if (0x800 <= cpt && cpt <= 0xffff) {
        result.push_back(0xe0 | ((cpt >> 12) & 0x0f));
        result.push_back(0x80 | ((cpt >> 6) & 0x3f));
        result.push_back(0x80 | (cpt & 0x3f));
        return result;
    }
    if (0x10000 <= cpt && cpt <= 0x10ffff) {
        result.push_back(0xf0 | ((cpt >> 18) & 0x07));
        result.push_back(0x80 | ((cpt >> 12) & 0x3f));
        result.push_back(0x80 | ((cpt >> 6) & 0x3f));
        result.push_back(0x80 | (cpt & 0x3f));
        return result;
    }

    throw std::invalid_argument("invalid codepoint");
}

std::vector<uint32_t> unicode_cpts_from_utf8(const std::string & utf8) {
    std::vector<uint32_t> result;
    result.reserve(utf8.size());
    size_t offset = 0;
    while (offset < utf8.size()) {
        try {
            result.push_back(unicode_cpt_from_utf8(utf8, offset));
        }
        catch (const std::invalid_argument & /*ex*/) {
            // Silently ignore invalid UTF-8 input to avoid leaking the exception beyond llama_tokenize
            ++offset;
            result.emplace_back(0xFFFD); // replacement character
        }
    }
    return result;
}

unicode_cpt_flags unicode_cpt_flags_from_cpt(const uint32_t cpt) {
    static const unicode_cpt_flags undef(unicode_cpt_flags::UNDEFINED);
    static const auto cpt_flags = unicode_cpt_flags_array();
    return cpt < cpt_flags.size() ? cpt_flags[cpt] : undef;
}

std::string unicode_byte_to_utf8(uint8_t byte) {
    static std::unordered_map<uint8_t, std::string> map = unicode_byte_to_utf8_map();
    return map.at(byte);
}

uint8_t unicode_utf8_to_byte(const std::string & utf8) {
    static std::unordered_map<std::string, uint8_t> map = unicode_utf8_to_byte_map();
    return map.at(utf8);
}

static bool nfd_range_first_greater(uint32_t value, const range_nfd & range) {
    return value < range.first;
}

uint32_t unicode_tolower(uint32_t cpt) {
    // binary search; std::pair::operator< compares .first first, and since .second
    // is unsigned, (a, b) < (cpt, 0) reduces exactly to a < cpt.
    auto it = std::lower_bound(unicode_map_lowercase.begin(), unicode_map_lowercase.end(),
        std::make_pair(cpt, (uint32_t) 0));
    if (it != unicode_map_lowercase.end() && it->first == cpt) {
        return it->second;
    }
    return cpt;  // Return the original code point if no lowercase mapping is found
}

uint32_t unicode_strip_accent_base(uint32_t cpt) {
    // unicode_ranges_nfd is sorted and non-overlapping by 'first'; binary search
    // for the range whose [first, last] interval contains cpt.
    auto it = std::upper_bound(unicode_ranges_nfd.begin(), unicode_ranges_nfd.end(), cpt,
        nfd_range_first_greater);
    if (it != unicode_ranges_nfd.begin()) {
        --it;
        if (cpt >= it->first && cpt <= it->last) {
            return it->nfd;
        }
    }
    return cpt;  // No accent-base mapping found, return the original code point
}

// unicode categories used by unicode_regex_split() to build a "collapsed" single-byte
// representation of the text (see collapse_codepoint() below) and of the regex patterns.
static const std::map<std::string, int> k_ucat_enum = {
    { "\\p{N}", unicode_cpt_flags::NUMBER },
    { "\\p{L}", unicode_cpt_flags::LETTER },
    { "\\p{P}", unicode_cpt_flags::PUNCTUATION },
    { "\\p{M}", unicode_cpt_flags::ACCENT_MARK },
    { "\\p{S}", unicode_cpt_flags::SYMBOL },
};

static const std::map<int, int> k_ucat_cpt = {
    { unicode_cpt_flags::NUMBER,      0xD1 },
    { unicode_cpt_flags::LETTER,      0xD2 },
    { unicode_cpt_flags::PUNCTUATION, 0xD3 },
    { unicode_cpt_flags::ACCENT_MARK, 0xD4 },
    { unicode_cpt_flags::SYMBOL,      0xD5 },
};

static const std::map<int, std::string> k_ucat_map = {
    { unicode_cpt_flags::NUMBER,      "\x30-\x39" }, // 0-9
    { unicode_cpt_flags::LETTER,      "\x41-\x5A\x61-\x7A" }, // A-Za-z
    { unicode_cpt_flags::PUNCTUATION, "\x21-\x23\x25-\x2A\x2C-\x2F\x3A-\x3B\x3F-\x40\\\x5B-\\\x5D\x5F\\\x7B\\\x7D" }, // !-#%-*,-/:-;?-@\[-\]_\{\}
    { unicode_cpt_flags::ACCENT_MARK, "" }, // no sub-128 codepoints
    { unicode_cpt_flags::SYMBOL,      "\\\x24\\\x2B\x3C-\x3E\x5E\x60\\\x7C" }, // $+<=>^`|
};

// Collapse a single codepoint into the one-byte representation used by the "collapsed" text.
static char collapse_codepoint(uint32_t cpt) {
    // keep single-byte codepoints as is
    if (cpt < 128) {
        return (char) cpt;
    }

    const auto flags = unicode_cpt_flags_from_cpt(cpt);

    if (flags.is_whitespace) {
        // std::regex \s doesn't match 0x85; use vertical tab instead.
        return (char) 0x0B;
    } else if (k_ucat_cpt.find(flags.category_flag()) != k_ucat_cpt.end()) {
        return (char) k_ucat_cpt.at(flags.category_flag());
    } else {
        return (char) 0xD0; // fallback
    }
}

// True if any of the known unicode-category placeholders (e.g. "\p{N}") appears in regex_expr.
static bool regex_uses_unicode_category(const std::string & regex_expr) {
    for (const auto & ucat : k_ucat_enum) {
        if (std::string::npos != regex_expr.find(ucat.first)) {
            return true;
        }
    }
    return false;
}

std::vector<std::string> unicode_regex_split(const std::string & text, const std::vector<std::string> & regex_exprs) {
    // compute collapsed codepoints only if needed by at least one regex
    bool need_collapse = false;
    for (const auto & regex_expr : regex_exprs) {
        if (regex_uses_unicode_category(regex_expr)) {
            need_collapse = true;
            break;
        }
    }

    const auto cpts = unicode_cpts_from_utf8(text);

    // generate a "collapsed" representation of the text, where all codepoints are replaced by a single byte
    // ref: https://github.com/ggml-org/llama.cpp/pull/6920#issuecomment-2081479935
    std::string text_collapsed;
    if (need_collapse) {
        // collapse all unicode categories
        text_collapsed.resize(cpts.size());

        std::transform(cpts.begin(), cpts.end(), text_collapsed.begin(), collapse_codepoint);
    }

    std::vector<size_t> bpe_offsets = { cpts.size() };

    for (const auto & regex_expr : regex_exprs) {
        // first, see if we have an efficient custom regex implementation
        auto tmp = unicode_regex_split_custom(text, regex_expr, bpe_offsets);

        if (!tmp.empty()) {
            bpe_offsets = std::move(tmp);
            continue;
        }

        // fallback to general-purpose std::regex / std::wregex
        try {
            // if a unicode category is used in the regex, we use the collapsed text and replace the unicode category
            // with the corresponding collapsed representation
            bool use_collapsed = regex_uses_unicode_category(regex_expr);

            if (use_collapsed) {
                // sanity-check that the original regex does not contain any non-ASCII characters
                const auto cpts_regex = unicode_cpts_from_utf8(regex_expr);
                for (uint32_t cpt : cpts_regex) {
                    if (cpt >= 128) {
                        throw std::runtime_error("Regex includes both unicode categories and non-ASCII characters - not supported");
                    }
                }

                // generate a collapsed representation of the regex
                std::string regex_expr_collapsed;

                // track if we are inside [], because nested [] are not allowed
                bool inside = false;
                for (size_t i = 0; i < regex_expr.size(); ++i) {
                    if (regex_expr[i] == '[' && (i == 0 || regex_expr[i - 1] != '\\')) {
                        regex_expr_collapsed += '[';
                        inside = true;
                        continue;
                    }

                    if (inside && regex_expr[i] == ']' && regex_expr[i - 1] != '\\') {
                        regex_expr_collapsed += ']';
                        inside = false;
                        continue;
                    }

                    if (regex_expr[i + 0] == '\\' && i + 4 < regex_expr.size() &&
                        regex_expr[i + 1] == 'p' &&
                        regex_expr[i + 2] == '{' &&
                        regex_expr[i + 4] == '}') {
                        const std::string pat = regex_expr.substr(i, 5);
                        if (k_ucat_enum.find(pat) != k_ucat_enum.end()) {
                            if (!inside) {
                                regex_expr_collapsed += '[';
                            }
                            regex_expr_collapsed += k_ucat_cpt.at(k_ucat_enum.at(pat));
                            regex_expr_collapsed += k_ucat_map.at(k_ucat_enum.at(pat));
                            if (!inside) {
                                regex_expr_collapsed += ']';
                            }
                            i += 4;
                            continue;
                        }
                    }

                    regex_expr_collapsed += regex_expr[i];
                }

                //printf("text_collapsed: %s\n", text_collapsed.c_str());
                //printf("regex_expr_collapsed: %s\n", regex_expr_collapsed.c_str());
                bpe_offsets = unicode_regex_split_stl(text_collapsed, regex_expr_collapsed, bpe_offsets);
            } else {
                // no unicode category used, we can use std::wregex directly
                const std::wstring wregex_expr = unicode_wstring_from_utf8(regex_expr);

                // std::wregex \s does not mach non-ASCII whitespaces, using 0x0B as fallback
                std::wstring wtext(cpts.begin(), cpts.end());
                for (wchar_t & c : wtext) {
                    if (c > 0x7F && unicode_cpt_flags_from_cpt(c).is_whitespace) {
                        c = (wchar_t) 0x0B;
                    }
                }

                //printf("text: %s\n", text.c_str());
                //printf("regex_expr: %s\n", regex_expr.c_str());
                bpe_offsets = unicode_regex_split_stl(wtext, wregex_expr, bpe_offsets);
            }
        } catch (std::regex_error & e) {
            fprintf(stderr, "Failed to process regex: '%s'\n", regex_expr.c_str());
            fprintf(stderr, "Regex error: %s\n", e.what());
            throw std::runtime_error("Failed to process regex");
        }
    }

    std::vector<std::string> bpe_words;
    bpe_words.reserve(bpe_offsets.size()); // reserve memory for the approximate size

    size_t start = 0;
    for (size_t & offset : bpe_offsets) {
        bpe_words.emplace_back();
        for (size_t i = start; i < start + offset; ++i) {
            bpe_words.back() += unicode_cpt_to_utf8(cpts[i]);
        }
        start += offset;
    }

    return bpe_words;

    return unicode_byte_encoding_process(bpe_words);
}
}}
