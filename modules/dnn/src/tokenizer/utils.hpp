// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef __OPENCV_DNN_TOKENIZER_UTILS_HPP__
#define __OPENCV_DNN_TOKENIZER_UTILS_HPP__

#include <string>
#include <unordered_map>

namespace cv { namespace dnn {

// Counts codepoints without decoding; continuation bytes are 0b10xxxxxx.
static inline size_t utf8CodepointCount(const std::string& s)
{
    size_t n = 0;
    for (char c : s)
        n += ((unsigned char)c & 0xC0) != 0x80;
    return n;
}

// Splits 'text' on the longest-matching entries of 'specialToId' that pass
// 'isAllowed', calling 'onLiteral' for each in-between run and 'onSpecialId'
// for each match, left to right. Shared by WordPiece and Unigram, whose
// special-token pre-splitting is otherwise identical.
template <typename AllowedFn, typename LiteralFn, typename SpecialFn>
static void splitOnSpecialTokens(const std::string& text,
                                  const std::unordered_map<std::string, int>& specialToId,
                                  AllowedFn isAllowed, LiteralFn onLiteral, SpecialFn onSpecialId)
{
    size_t chunkStart = 0;
    size_t pos = 0;
    while (pos < text.size()) {
        std::string matched;
        int matchedId = -1;
        const char head = text[pos];
        for (const auto& kv : specialToId) {
            const std::string& sp = kv.first;
            if (sp.empty() || sp[0] != head) continue;
            if (pos + sp.size() > text.size()) continue;
            if (sp.size() <= matched.size()) continue;
            if (!isAllowed(sp)) continue;
            if (text.compare(pos, sp.size(), sp) != 0) continue;
            matched = sp; matchedId = kv.second;
        }
        if (matchedId >= 0) {
            if (pos > chunkStart) onLiteral(text.substr(chunkStart, pos - chunkStart));
            onSpecialId(matchedId);
            pos += matched.size();
            chunkStart = pos;
        } else {
            ++pos;
        }
    }
    if (chunkStart < text.size()) onLiteral(text.substr(chunkStart));
}

// R"R50K('(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+$|\s+(?!\S)|\s)R50K"
static const std::string R50K_UTF8 = "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)";

// Extracts and erases a JSON string field FileStorage would overflow on.
static inline std::string extractAndStripBase64Field(std::string& jsonText,
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

}}
#endif
