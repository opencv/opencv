// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef __OPENCV_DNN_TOKENIZER_UTILS_HPP__
#define __OPENCV_DNN_TOKENIZER_UTILS_HPP__

#include <string>
#include <unordered_map>

namespace cv { namespace dnn {

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
        for (const auto& kv : specialToId) {
            const std::string& sp = kv.first;
            if (sp.empty() || !isAllowed(sp)) continue;
            if (pos + sp.size() > text.size()) continue;
            if (text.compare(pos, sp.size(), sp) != 0) continue;
            if (sp.size() > matched.size()) { matched = sp; matchedId = kv.second; }
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

// GPT-4’s cl100k_base split pattern
// NOTE: This pattern is adapted from the original Python regex used for GPT-4's cl100k_base BPE split.
// The original Python pattern is:
//   r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s"""
//
// This C++ version differs in the following ways:
//   1. Possessive quantifiers (`++`, `*+`, `?+`) are replaced with standard quantifiers (`+`, `*`, `?`)
//      because C++ std::regex does not support possessive quantifiers.
//   2. Inline case-insensitive group `(?i:...)` is replaced with a non-capturing group `(?:...)`
//      because C++ std::regex does not support inline flags. Case-insensitivity must be handled separately.
//   3. The `$` anchor at the end of `\s++$` is omitted because it's not needed for splitting and may cause issues.
//   4. Unicode classes (`\p{L}`, `\p{N}`) are kept because the tokenizer's implementation handles them via custom llama.cpp logic.
//
// The resulting C++ pattern is compatible with std::regex and the tokenizer's Unicode handling logic.
static const std::string CL100K_BASE = R"CL100K('(?:[sSdDmMtT]|[lL][lL]|[vV][eE]|[rR][eE])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s)CL100K";

// Qwen2.5 pre-tokenizer split pattern (from tokenizer.json)
static const std::string QWEN2_5 = R"QWEN('(?:[sSdDmMtT]|[lL][lL]|[vV][eE]|[rR][eE])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)QWEN";

}}
#endif
