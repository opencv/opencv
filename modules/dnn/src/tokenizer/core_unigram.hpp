// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef __OPENCV_DNN_TOKENIZER_CORE_UNIGRAM_HPP__
#define __OPENCV_DNN_TOKENIZER_CORE_UNIGRAM_HPP__

#include "unicode.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

// cv::FileStorage's JSON reader has a hard per-string-token length cap
// (CV_FS_MAX_LEN, see persistence.hpp) and raises a hard parse error
// ("string is too long") the moment it meets a longer one. Real HuggingFace
// Unigram tokenizer.json files (e.g. T5) embed a SentencePiece
// "precompiled_charsmap" normalizer blob as a base64 string that is easily
// hundreds of KB, far past that cap. Since FileStorage parses eagerly on
// open, that field must never reach it: find it in the raw text first, pull
// its value out, and splice it down to an empty string before handing the
// buffer to FileStorage. The extracted payload (if any) is returned via
// out_value so the caller can still use it (e.g. build the normalizer).
static inline std::string extractAndStripLongStringField(std::string& json_text,
                                                    const std::string& field_key) {
    std::string extracted;
    const std::string needle = "\"" + field_key + "\"";
    size_t key_pos = json_text.find(needle);
    if (key_pos == std::string::npos)
        return extracted;

    size_t colon = json_text.find(':', key_pos + needle.size());
    if (colon == std::string::npos)
        return extracted;

    size_t value_start = json_text.find('"', colon + 1);
    if (value_start == std::string::npos)
        return extracted; // value isn't a string (e.g. null) -- nothing to strip

    ++value_start; // move past the opening quote

    // The base64 payload cannot contain '"' or backslash escapes, so the
    // next '"' is guaranteed to be the closing quote of this value.
    size_t value_end = json_text.find('"', value_start);
    if (value_end == std::string::npos)
        return extracted;

    extracted = json_text.substr(value_start, value_end - value_start);
    json_text.erase(value_start, value_end - value_start);
    return extracted;
}

// U+2581 LOWER ONE EIGHTH BLOCK, used by SentencePiece/HF "Metaspace" as a
// visible stand-in for the space character.
static const std::string UNIGRAM_METASPACE = "\xE2\x96\x81";

// ---------------------------------------------------------------------------
// Minimal base64 decoder, used to unpack normalizer.precompiled_charsmap.
// ---------------------------------------------------------------------------
static inline std::string unigramBase64Decode(const std::string& in) {
    static int T[256];
    static bool inited = false;
    if (!inited) {
        std::fill(std::begin(T), std::end(T), -1);
        static const char* alphabet =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for (int i = 0; i < 64; i++) T[(unsigned char)alphabet[i]] = i;
        inited = true;
    }
    std::string out;
    out.reserve(in.size() / 4 * 3 + 3);
    int val = 0, valb = -8;
    for (unsigned char c : in) {
        if (c == '=' || T[c] == -1) continue;
        val = (val << 6) + T[c];
        valb += 6;
        if (valb >= 0) {
            out.push_back(static_cast<char>((val >> valb) & 0xFF));
            valb -= 8;
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// SentencePiece "Precompiled" normalizer: a XOR-compressed compact double
// array (XCDA/darts-clone) trie mapping input byte-prefixes to replacement
// strings, followed by a blob of NUL-terminated replacement strings.
//
// Layout of the decoded blob:
//   [0..4)                        uint32 trie size in bytes -- this is the
//                                 byte length of the trie unit array that
//                                 follows, and is load-bearing: it is what
//                                 locates the start of the strings section.
//   [4..4+trieBytes)              XCDA trie units (uint32 each)
//   [4+trieBytes..end)            NUL-terminated replacement strings; leaf
//                                 "value" fields are offsets into this
//                                 section, not into the blob as a whole.
//
// Traversal algorithm below mirrors the well known darts-clone / llama.cpp
// reimplementation of SentencePiece's normalizer.
// ---------------------------------------------------------------------------
struct UnigramPrecompiledNormalizer {
    std::string blob;
    uint32_t trieSize = 0; // byte length of the trie unit array, from blob[0..4)

    bool empty() const { return blob.empty(); }

    uint32_t getNode(uint32_t index) const {
        size_t count = trieSize / 4;
        if (index >= count) return 0;
        const uint8_t* base = reinterpret_cast<const uint8_t*>(blob.data()) + 4 + (size_t)index * 4;
        uint32_t v;
        std::memcpy(&v, base, sizeof(v));
        return v;
    }
    uint32_t getBase(uint32_t index) const {
        uint32_t node = getNode(index);
        return (node >> 10) << ((node & (1u << 9)) >> 6);
    }
    uint32_t getLcheck(uint32_t index) const {
        return getNode(index) & ((1u << 31) | 0xffu);
    }
    bool getLeaf(uint32_t index) const {
        return ((getNode(index) >> 8) & 1u) != 0;
    }
    uint32_t getValue(uint32_t index) const {
        return getNode(index) & ((1u << 31) - 1);
    }

    // Finds the longest matching prefix of text[offset..] in the trie and
    // returns its replacement string; falls back to passing a single UTF-8
    // codepoint through unmodified (or U+FFFD for invalid byte sequences).
    void normalizePrefix(const std::string& text, size_t offset,
                          size_t& consumed, std::string& replacement) const {
        size_t longestLen = 0;
        uint32_t longestOff = 0;
        if (!empty() && 4 + 0u <= blob.size()) {
            // XCDA traversal accumulates the node id across steps (node ^= c,
            // then node ^= base(node) once the transition is validated) --
            // it is NOT a plain base(node) ^ c reassignment. See darts-clone /
            // llama.cpp's normalize_prefix for the reference recurrence.
            uint32_t node = getBase(0);
            size_t pos = offset;
            while (pos < text.size()) {
                unsigned char c = (unsigned char)text[pos];
                node ^= c;
                if (getLcheck(node) != c) break;
                bool leaf = getLeaf(node);
                node ^= getBase(node);
                if (leaf) {
                    longestLen = pos - offset + 1;
                    longestOff = getValue(node);
                }
                ++pos;
            }
        }
        size_t stringsStart = 4 + (size_t)trieSize;
        if (longestLen > 0 && stringsStart + (size_t)longestOff < blob.size()) {
            consumed = longestLen;
            replacement.assign(blob.data() + stringsStart + longestOff);
            return;
        }
        size_t off2 = offset;
        uint32_t cpt = unicode_cpt_from_utf8(text, off2);
        if (off2 <= offset) off2 = offset + 1;
        if (cpt == 0 && off2 == offset + 1 && (unsigned char)text[offset] >= 0x80) {
            // treat clearly-invalid lead bytes as a single replacement char
            consumed = 1;
            replacement = "\xEF\xBF\xBD";
            return;
        }
        consumed = off2 - offset;
        replacement = text.substr(offset, consumed);
    }

    std::string normalize(const std::string& text) const {
        if (empty()) return text;
        std::string out;
        out.reserve(text.size() + 16);
        size_t offset = 0;
        while (offset < text.size()) {
            size_t consumed = 0;
            std::string rep;
            normalizePrefix(text, offset, consumed, rep);
            out += rep;
            offset += consumed;
        }
        return out;
    }
};

static inline UnigramPrecompiledNormalizer buildUnigramPrecompiledNormalizer(const std::string& b64) {
    UnigramPrecompiledNormalizer n;
    if (!b64.empty()) {
        n.blob = unigramBase64Decode(b64);
        if (n.blob.size() >= 4)
            std::memcpy(&n.trieSize, n.blob.data(), sizeof(n.trieSize));
    }
    return n;
}

// ---------------------------------------------------------------------------
// SentencePiece Unigram (T5-style) core model: Viterbi segmentation over a
// vocabulary of (piece, log-probability) pairs, with a WhitespaceSplit +
// Metaspace(prepend_scheme=always) pre-tokenizer and the "Precompiled"
// charsmap normalizer above.
// ---------------------------------------------------------------------------
struct CoreUnigram {
    std::vector<std::string> idToPiece;
    std::vector<float> idToScore;
    std::unordered_map<std::string, int> pieceToId;

    std::unordered_map<std::string, int> specialToId;
    std::unordered_map<int, std::string> idToSpecial;

    int unkId = -1;
    int eosId = -1;
    float unkScore = -10.0f;
    size_t maxPieceCps = 1;

    UnigramPrecompiledNormalizer normalizer;

    void finalize() {
        float minScore = 0.0f;
        bool any = false;
        for (size_t i = 0; i < idToScore.size(); ++i) {
            if ((int)i == unkId) continue;
            if (!any || idToScore[i] < minScore) { minScore = idToScore[i]; any = true; }
        }
        unkScore = minScore - 10.0f;

        maxPieceCps = 1;
        for (const auto& kv : pieceToId) {
            size_t n = unicode_cpts_from_utf8(kv.first).size();
            if (n > maxPieceCps) maxPieceCps = n;
        }
    }

    // Viterbi-encode a single pre-tokenized chunk (already carries any
    // leading metaspace marker) and append resulting ids to `out`.
    void encodeChunk(const std::string& chunk, std::vector<int>& out) const {
        std::vector<size_t> offs;
        offs.reserve(chunk.size() + 1);
        {
            size_t p = 0;
            while (p < chunk.size()) {
                offs.push_back(p);
                size_t p2 = p;
                uint32_t cpt = unicode_cpt_from_utf8(chunk, p2);
                (void)cpt;
                if (p2 <= p) p2 = p + 1;
                p = p2;
            }
            offs.push_back(chunk.size());
        }
        size_t n = offs.size() - 1;
        if (n == 0) return;

        const float NEG_INF = -std::numeric_limits<float>::infinity();
        std::vector<float> best(n + 1, NEG_INF);
        std::vector<int> backPos(n + 1, -1);
        std::vector<int> backId(n + 1, -1);
        std::vector<char> backIsUnk(n + 1, 0);
        best[0] = 0.0f;

        for (size_t i = 0; i < n; ++i) {
            if (best[i] == NEG_INF) continue;

            // Always-available single-codepoint <unk> fallback edge.
            {
                float sc = best[i] + unkScore;
                if (sc > best[i + 1]) {
                    best[i + 1] = sc;
                    backPos[i + 1] = (int)i;
                    backIsUnk[i + 1] = 1;
                    backId[i + 1] = unkId;
                }
            }

            size_t maxLen = std::min(maxPieceCps, n - i);
            for (size_t len = 1; len <= maxLen; ++len) {
                size_t j = i + len;
                std::string sub = chunk.substr(offs[i], offs[j] - offs[i]);
                auto it = pieceToId.find(sub);
                if (it == pieceToId.end()) continue;
                int id = it->second;
                float sc = best[i] + idToScore[(size_t)id];
                if (sc > best[j]) {
                    best[j] = sc;
                    backPos[j] = (int)i;
                    backIsUnk[j] = 0;
                    backId[j] = id;
                }
            }
        }

        std::vector<std::pair<int, bool>> segs;
        size_t cur = n;
        while (cur > 0) {
            segs.emplace_back(backId[cur], backIsUnk[cur] != 0);
            cur = (size_t)backPos[cur];
        }
        std::reverse(segs.begin(), segs.end());

        for (const auto& s : segs) {
            if (s.second) {
                // fuse_unk: collapse consecutive <unk> emissions into one.
                if (!out.empty() && out.back() == unkId) continue;
                out.push_back(unkId);
            } else {
                out.push_back(s.first);
            }
        }
    }

    void pretokenizeAndEncode(const std::string& normalized, std::vector<int>& out) const {
        std::vector<size_t> offs;
        std::vector<uint32_t> cps;
        offs.reserve(normalized.size() + 1);
        cps.reserve(normalized.size());
        size_t p = 0;
        while (p < normalized.size()) {
            offs.push_back(p);
            size_t p2 = p;
            uint32_t cpt = unicode_cpt_from_utf8(normalized, p2);
            if (p2 <= p) { p2 = p + 1; cpt = 0xFFFD; }
            cps.push_back(cpt);
            p = p2;
        }
        offs.push_back(normalized.size());
        size_t n = cps.size();

        size_t i = 0;
        while (i < n) {
            while (i < n && unicode_cpt_flags_from_cpt(cps[i]).is_whitespace) ++i;
            if (i >= n) break;
            size_t start = i;
            while (i < n && !unicode_cpt_flags_from_cpt(cps[i]).is_whitespace) ++i;
            std::string word = normalized.substr(offs[start], offs[i] - offs[start]);
            std::string piece = UNIGRAM_METASPACE + word;
            encodeChunk(piece, out);
        }
    }

    std::vector<int> encode(const std::string& text, const std::unordered_set<std::string>& allowedSpecial) const {
        std::vector<int> ids;
        size_t chunkStart = 0;
        size_t pos = 0;
        while (pos < text.size()) {
            std::string matched;
            int matchedId = -1;
            if (!allowedSpecial.empty()) {
                for (const auto& kv : specialToId) {
                    const std::string& sp = kv.first;
                    if (allowedSpecial.find(sp) == allowedSpecial.end()) continue;
                    if (sp.empty()) continue;
                    if (pos + sp.size() > text.size()) continue;
                    if (text.compare(pos, sp.size(), sp) != 0) continue;
                    if (sp.size() > matched.size()) { matched = sp; matchedId = kv.second; }
                }
            }
            if (matchedId >= 0) {
                if (pos > chunkStart) {
                    std::string norm = normalizer.normalize(text.substr(chunkStart, pos - chunkStart));
                    pretokenizeAndEncode(norm, ids);
                }
                ids.push_back(matchedId);
                pos += matched.size();
                chunkStart = pos;
            } else {
                ++pos;
            }
        }
        if (chunkStart < text.size()) {
            std::string norm = normalizer.normalize(text.substr(chunkStart));
            pretokenizeAndEncode(norm, ids);
        }
        if (eosId >= 0) ids.push_back(eosId);
        return ids;
    }

    std::string decode(const std::vector<int>& tokens) const {
        std::string raw;
        for (int id : tokens) {
            if (idToSpecial.find(id) != idToSpecial.end()) continue;
            if (id >= 0 && (size_t)id < idToPiece.size())
                raw += idToPiece[(size_t)id];
        }
        std::string result;
        result.reserve(raw.size());
        size_t i = 0;
        while (i < raw.size()) {
            if (raw.compare(i, UNIGRAM_METASPACE.size(), UNIGRAM_METASPACE) == 0) {
                result += ' ';
                i += UNIGRAM_METASPACE.size();
            } else {
                result += raw[i++];
            }
        }
        if (!result.empty() && result.front() == ' ')
            result.erase(result.begin());
        return result;
    }
};

CV__DNN_INLINE_NS_END
}}

#endif // __OPENCV_DNN_TOKENIZER_CORE_UNIGRAM_HPP__
