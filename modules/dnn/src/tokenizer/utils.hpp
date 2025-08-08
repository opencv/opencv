/*M///////////////////////////////////////////////////////////////////////////////////////
// MIT License
//
// Copyright (c) 2018‑2019 Andrew Gallant
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////////////*/
 #pragma once

 #include "core_bpe.hpp"
 #include <unordered_map>
#include <string>
#include <vector>
#include <fstream>

namespace cv { namespace dnn { namespace tokenizer {

// GPT2 for Testing this function should be moved into registry later
// Or, to have it as a std::string:
static const std::string R50K_UTF8 = R"R50K('(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+$|\s+(?!\S)|\s)R50K";
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
static const std::string CL100K_BASE = R"CL100K('(?:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s)CL100K";

// Escape regex meta-characters in a string
static std::string escape_regex(const std::string &s) {
    static const std::string meta = R"(.^$|()[]*+?{}\")";
    std::string out;
    out.reserve(s.size() * 2);
    for (char c : s) {
        if (meta.find(c) != std::string::npos) out.push_back('\\');
        out.push_back(c);
    }
    return out;
}


inline std::vector<uint8_t> base64_decode(const std::string& in) {
    static const std::string b64_chars =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::vector<uint8_t> out;
    int val = 0, valb = -8;
    for (unsigned char c : in) {
        if (isspace(c)) continue;
        if (c == '=') break;
        int idx = b64_chars.find(c); 
        if (idx == std::string::npos) break;
        val = (val << 6) + idx;
        valb += 6;
        if (valb >= 0) {
            out.push_back(uint8_t((val >> valb) & 0xFF));
            valb -= 8;
        }
    }
    return out;
}

// Read entire file into a string
static std::string readFile(const std::string& path) {
    std::ifstream in{path, std::ios::binary};
    if (!in) throw std::runtime_error("Failed to open " + path);
    std::ostringstream buf;
    buf << in.rdbuf();
    return buf.str();
}

}}} // namespace cv namespace dnn namespace tokenizer