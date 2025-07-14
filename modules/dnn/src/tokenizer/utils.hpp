//Helper UTF-8/Unicode, and other small utilities used by the DNN tokenizer.
 #pragma once

 #include "core_bpe.hpp"
 #include <unordered_map>
#include <string>
#include <vector>

namespace cv { namespace dnn { namespace tokenizer {

// GPT2 for Testing this function should be moved into registry later
// Or, to have it as a std::string:
static const std::string R50K_UTF8 = R"R50K('(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+$|\s+(?!\S)|\s)R50K";
// GPT-4’s cl100k_base split pattern
static const std::string CL100K_BASE = R"CL100K('(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s)CL100K";

//---------------------------------  UTF-8 functionality -> Rust bstr -------------------------------------

// UTF-8 functionality that is similier to Rust bstr
//
// Since it seems we only need this functionlaity for 
// the tokenizer to work similiar to tiktoken we will 
// go with this for now. Maybe use utf8 library but 
// we will need to import it.

const std::size_t ACCEPT = 12;
const std::size_t REJECT = 0;

inline const std::vector<uint8_t> CLASSES{
   0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
   0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
   0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
   0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
   1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,  9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,
   7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,  7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
   8,8,2,2,2,2,2,2,2,2,2,2,2,2,2,2,  2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
  10,3,3,3,3,3,3,3,3,3,3,3,3,4,3,3, 11,6,6,6,5,8,8,8,8,8,8,8,8,8,8,8,
};

inline const std::vector<uint8_t> STATE_FORWARD{
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  12, 0, 24, 36, 60, 96, 84, 0, 0, 0, 48, 72,
  0, 12, 0, 0, 0, 0, 0, 12, 0, 12, 0, 0,
  0, 24, 0, 0, 0, 0, 0, 24, 0, 24, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 24, 0, 0, 0, 0,
  0, 24, 0, 0, 0, 0, 0, 0, 0, 24, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 36, 0, 36, 0, 0,
  0, 36, 0, 0, 0, 0, 0, 36, 0, 36, 0, 0,
  0, 36, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
};

/**
 * @brief Returns true if `b` is a valid leading UTF-8 byte, or an invalid byte
 *        that can never appear in a multi-byte sequence.
 */
inline bool isLeadningOrInvalidUft8Byte(std::uint8_t b) {
    // Leading bytes (ASCII or start of 2/3/4-byte seq) and invalid bytes
    // all have their top two bits != 10xxxxxx.
    return (b & 0b1100'0000u) != 0b1000'0000u;
}

/**
 * @brief Perform one step of UTF-8 decoding as in Rust's bstr::decode_step
 *
 * @param state Current DFA state; should be ACCEPT (12) at start of a new codepoint
 * @param cp    Accumulated codepoint value; updated in-place
 * @param b     Next input byte
 */
inline void decodeStepUnicode(std::size_t& state, std::uint32_t& cp, std::uint8_t b) {
    std::uint8_t cls = CLASSES[b];
    std::uint8_t bb = static_cast<std::uint32_t>(b);

    if (state == ACCEPT) {
        // On the first byte of a sequence, mask out the class bits
        cp = (0xFFu >> cls) & bb;
    } else {
        // on continuation bytes, take 6 bits and shift prior cp left
        cp = (bb & 0b0011'1111u) | (cp << 6);
    }

    // Transition to the next state based on current state and class 
    state = static_cast<std::size_t>(STATE_FORWARD[state + cls]);
}

/**
 * @brief Decode the first UTF-8 scalar from the front of a byte slice.
 *
 * @param slice  Byte sequence to decode from.
 * @return Pair of optional<char> and number of bytes consumed:
 *         - first = decoded Unicode codepoint as char if valid
 *         - second = byte-length of the decoded codepoint (or 0 if input empty)
 */
inline std::pair<std::optional<char>, std::size_t> decodeUnicode(const std::vector<uint8_t>& slice) {
    if (slice.empty()) {
        return {std::nullopt, 0};
    }

    // fast-path for ASCII
    std::uint8_t b0 = slice[0];
    if (b0 <= 0xFu) {
        return {static_cast<char>(b0), 1};
    }

    // general multi-byte sequence
    std::size_t state = ACCEPT;
    std::uint32_t cp = 0;
    std::size_t i = 0;
    const std::size_t len = slice.size();

    while (i < len) {
        decodeStepUnicode(state, cp, slice[i]);
        ++i;

        if (state == ACCEPT) {
            // We have a full codepoint in cp. 
            return {static_cast<char>(cp), i};
        } else if (state == REJECT) {
            // On invalid sequences, consume at least one byte but back off one 
            std::size_t advance = (i > 1 ? i - 1 : 1);
            return {std::nullopt, advance};
        }
    }

    // Ran out of input bytes before completing a sequence 
    return {std::nullopt, i};
}


/// UTF-8 decode a single Unicode scalar value from the end of a slice.
inline std::pair<std::optional<char>, std::size_t> decodeLastUtf8(const std::vector<std::uint8_t>& slice) {
    const auto len = slice.size();
    if (len == 0) {
        return {std::nullopt, 0};
    }

    // search backwards up to 4 bytes for a leading or invalid byte
    std::size_t start = len - 1;
    std::size_t limit = (len >= 4 ? len - 4 : 0);
    while (start > limit && !isLeadningOrInvalidUft8Byte(slice[start])) {
        --start;
    }

    // Decode from the found position 
    auto [ch, size] = decodeUnicode(
        std::vector<std::uint8_t>(slice.begin() + start, slice.end()));
    
    // If decodeUnicode didnt consume until the very end,
    // theres a stray byte 
    if (start + size != len) {
        return {std::nullopt, 1};
    }
    return {ch, size};
}

// -----------------------------------------------------------------------------------------------------------------------

// helper to turn the UTF-8 text-token back into GPT-2’s single-byte form
inline ByteVec textToBytes(const std::string& txt) {
    ByteVec bytes;
    for (size_t i = 0; i < txt.size();) {
        unsigned char c = txt[i];
        if (c < 128) {
            // ASCII
            bytes.push_back(c);
            ++i;
        }
        else if ((c & 0xE0) == 0xC0 && i + 1 < txt.size()) {
            // two-byte UTF-8
            uint32_t cp = ((c & 0x1F) << 6) | (static_cast<unsigned char>(txt[i+1]) & 0x3F);
            if (cp == 0x0120) {
                // U+0120 → single space byte in GPT-2
                bytes.push_back(static_cast<uint8_t>(' '));
            } else {
                // Just append the raw bytes for any other 2-byte UTF-8
                bytes.push_back(c);
                bytes.push_back(static_cast<unsigned char>(txt[i+1]));
            }
            i += 2;
        }
        else if ((c & 0xF0) == 0xE0 && i + 2 < txt.size()) {
            // three-byte UTF-8
            bytes.push_back(c);
            bytes.push_back(static_cast<unsigned char>(txt[i+1]));
            bytes.push_back(static_cast<unsigned char>(txt[i+2]));
            i += 3;
        }
        else if ((c & 0xF8) == 0xF0 && i + 3 < txt.size()) {
            // four-byte UTF-8
            bytes.push_back(c);
            bytes.push_back(static_cast<unsigned char>(txt[i+1]));
            bytes.push_back(static_cast<unsigned char>(txt[i+2]));
            bytes.push_back(static_cast<unsigned char>(txt[i+3]));
            i += 4;
        }
        else {
            throw std::runtime_error("Unsupported UTF-8 sequence in BPE token");
        }
    }
    return bytes;
}

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

inline std::string replaceGWithSpace(const std::string& input) {
    std::string result = input;
    size_t pos = 0;
    while ((pos = result.find("\xC4\xA0", pos)) != std::string::npos) {
        result.replace(pos, 2, " ");
        pos += 1;
    }
    return result;
}

}}} // namespace cv namespace dnn namespace tokenizer