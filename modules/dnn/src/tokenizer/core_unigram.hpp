// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef __OPENCV_DNN_TOKENIZER_CORE_UNIGRAM_HPP__
#define __OPENCV_DNN_TOKENIZER_CORE_UNIGRAM_HPP__

#include "unicode.hpp"

#include <opencv2/core.hpp>
#include <opencv2/dnn/dnn.hpp>

#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

static inline std::string extractAndStripLongStringField(std::string& jsonText,
                                                    const std::string& fieldKey)
{
    std::string extracted;
    const std::string needle = "\"" + fieldKey + "\"";
    size_t keyPos = jsonText.find(needle);
    if (keyPos == std::string::npos)
        return extracted;

    size_t colon = jsonText.find(':', keyPos + needle.size());
    if (colon == std::string::npos)
        return extracted;

    size_t valueStart = jsonText.find('"', colon + 1);
    if (valueStart == std::string::npos)
        return extracted;

    ++valueStart;

    size_t valueEnd = jsonText.find('"', valueStart);
    if (valueEnd == std::string::npos)
        return extracted;

    extracted = jsonText.substr(valueStart, valueEnd - valueStart);
    jsonText.erase(valueStart, valueEnd - valueStart);
    return extracted;
}

static const std::string UNIGRAM_METASPACE = "\xE2\x96\x81";

static inline std::string unigramBase64Decode(const std::string& in)
{
    static const std::array<int, 256> T = []{
        std::array<int, 256> t;
        t.fill(-1);
        static const char* alphabet =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for (int i = 0; i < 64; i++) t[(unsigned char)alphabet[i]] = i;
        return t;
    }();
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

/**
 * @brief SentencePiece "precompiled_charsmap" normalizer: a serialized
 * darts-clone double-array trie mapping raw text prefixes to their
 * normalized replacement strings, plus a literal-copy/replacement-char
 * fallback for unmatched prefixes.
 */
class UnigramPrecompiledNormalizer {
public:
    UnigramPrecompiledNormalizer() = default;

    /// @param blob Decoded charsmap bytes, not base64; may be empty.
    explicit UnigramPrecompiledNormalizer(std::string blob);

    bool empty() const { return blob_.empty(); }
    std::string normalize(const std::string& text) const;

private:
    uint32_t getNode(uint32_t index) const;
    uint32_t getBase(uint32_t index) const;
    uint32_t getLcheck(uint32_t index) const;
    bool getLeaf(uint32_t index) const;
    uint32_t getValue(uint32_t index) const;

    void normalizePrefix(const std::string& text, size_t offset,
                          size_t& consumed, std::string& replacement) const;

    std::string blob_;
    uint32_t trieSize_ = 0;
};

static inline UnigramPrecompiledNormalizer buildUnigramPrecompiledNormalizer(const std::string& b64)
{
    if (b64.empty())
        return UnigramPrecompiledNormalizer();
    return UnigramPrecompiledNormalizer(unigramBase64Decode(b64));
}

struct UnigramNormalizerStep
{
    enum Kind
    {
        REPLACE,
        LOWERCASE,
        STRIP_ACCENTS,
        PRECOMPILED
    };

    Kind kind = PRECOMPILED;
    std::string from;
    std::string to;
};

/**
 * @brief Core Unigram (SentencePiece-style) engine: Viterbi segmentation +
 * decode, text -> ids and ids -> text only.
 *
 * Added/special-token splitting, allowed-special filtering and eos-id
 * appending happen here (inside encode()/decode()) rather than in a separate
 * *TokenizerImpl wrapper, since the reference Unigram tokenizer folds those
 * directly into its segmentation loop.
 */
class CoreUnigram {
public:
    CoreUnigram() = default;

    /**
     * @brief Build a CoreUnigram from an already-parsed vocab and special-token table.
     *
     * @param vocab         Ordered (piece, log-probability score) pairs; the index in
     *                      this vector is the vocab id.
     * @param unkId         Vocab id of the unknown-token piece (-1 if none).
     * @param normalizer    Precompiled SentencePiece normalizer to apply before segmentation.
     * @param specialToId   Map from literal added/special-token text to vocab id.
     * @param eosId         Vocab id to append at the end of every encode() call (-1 to skip).
     * @param normSteps     Normalizer chain from tokenizer.json, in declared order.
     */
    CoreUnigram(const std::vector<std::pair<std::string, float>>& vocab,
                int unkId,
                UnigramPrecompiledNormalizer normalizer,
                const std::unordered_map<std::string, int>& specialToId,
                int eosId,
                const std::vector<UnigramNormalizerStep>& normSteps =
                    std::vector<UnigramNormalizerStep>(1));

    std::vector<int> encode(const std::string& text, const std::unordered_set<std::string>& allowedSpecial) const;
    std::string decode(const std::vector<int>& tokens) const;

private:
    // Viterbi-optimal split of one metaspace-prefixed chunk into vocab ids;
    // unkId_ fallback never repeats for adjacent no-match spans.
    void encodeChunk(const std::string& chunk, std::vector<int>& out) const;

    // Splits normalized text on whitespace; encodeChunk()s each word with a
    // leading UNIGRAM_METASPACE (SentencePiece "▁") marker.
    void pretokenizeAndEncode(const std::string& normalized, std::vector<int>& out) const;

    std::string applyNormalizer(const std::string& text) const;

    std::vector<std::string> idToPiece_;
    std::vector<float> idToScore_;
    std::unordered_map<std::string, int> pieceToId_;

    std::unordered_map<std::string, int> specialToId_;
    std::unordered_map<int, std::string> idToSpecial_;

    int unkId_ = -1;
    int eosId_ = -1;
    float unkScore_ = -10.0f;
    size_t maxPieceCps_ = 1;

    UnigramPrecompiledNormalizer normalizer_;
    std::vector<UnigramNormalizerStep> normSteps_;
};

CV__DNN_INLINE_NS_END
}}

#endif // __OPENCV_DNN_TOKENIZER_CORE_UNIGRAM_HPP__
