// Helpers for reading and interpreting GPT-2 encoder JSON, handling
//  UTF-8/Unicode, and other small utilities used by the DNN tokenizer.
 #pragma once

 #include "core_bpe.hpp"
 #include <unordered_map>
#include <string>
#include <vector>



namespace cv { namespace dnn { namespace tokenizer {
    
// ------------------------------ JSON parsing -----------------------------------
void append_utf8(uint32_t codepoint, std::string& out);

std::string unescape_json(const std::string& s);

// Return a mapping: token string (raw bytes) -> rank
std::unordered_map<std::string, int> read_encoder_json(const std::string& path);

// ---------------------------------------------------------------------------------

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
ByteVec textToBytes(const std::string& txt);

}}} // namespace cv namespace dnn namespace tokenizer