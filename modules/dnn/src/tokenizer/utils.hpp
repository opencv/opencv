// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef __OPENCV_DNN_TOKENIZER_UTILS_HPP__
#define __OPENCV_DNN_TOKENIZER_UTILS_HPP__

#include <string>

namespace cv { namespace dnn {

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
static const std::string CL100K_BASE = R"CL100K('(?:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s)CL100K";

}}
#endif
