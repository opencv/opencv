#pragma once

#include <opencv2/core.hpp>

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <unordered_map>
#include <regex> // Not enough functionality as the Rust/Python version
#include <set>
#include <stdexcept>
#include <thread>
#include <unordered_set>
#include <algorithm>


#ifndef __OPENCV_DNN_SRC_TOKENIZERTOKENS_CORE_BPE_HPP__
#define __OPENCV_DNN_SRC_TOKENIZERTOKENS_CORE_BPE_HPP__

namespace cv { namespace dnn {namespace tokenizer {

    
using Rank = std::uint32_t;
using ByteVec = std::vector<std::uint8_t>;

struct ByteVecHash {
    std::size_t operator()(const ByteVec& v) const noexcept {
        std::size_t h = 0;
        for (auto b : v) h = h * 31u + static_cast<std::size_t>(b);
        return h;
    }
};

using ByteVecRankMap = std::unordered_map<ByteVec, Rank, ByteVecHash>;

// hash the OS thread ID, mod by a fixed size, and pick a pre-compiled regex 
// to avoid cross-thread contention
std::size_t hashCurrentThread();

// scan adjacent byte-pairs to find the lowest-rank merge, splice them out, 
// update neighboring ranks, and repeat until no mergeable pair remains
std::vector<std::pair<std::size_t, Rank>> bytePairMerge(const ByteVecRankMap& ranks, 
                                                        const ByteVec& piece);

// map a single-byte slice directly to its rank if present, or else call the merge loop 
// and then translate each resulting segment into its rank
std::vector<Rank> bytePairEncode(const ByteVec& piece, 
                                 const ByteVecRankMap& ranks);

// return the raw byte-sequence segments before ranking by using the same merge boundaries
CV_EXPORTS std::vector<ByteVec> bytePairSplit(const ByteVec& piece, 
                                   const ByteVecRankMap& ranks);

CV_EXPORTS std::vector<ByteVec> bytePairSplit(std::string& s,
                                   const ByteVecRankMap& ranks);

class DecoderKeyError : public std::runtime_error {
public: 
    explicit DecoderKeyError(Rank token);
    Rank token() const noexcept { return token_; }
private:
    Rank token_;
};

class DecodeError : public std::runtime_error {
public: 
    explicit DecodeError(std::string message);
};

class CV_EXPORTS CoreBPE {
public:
    
    explicit CoreBPE(ByteVecRankMap encoder,
            std::unordered_map<std::string, Rank> specialEncoder, 
            const std::string& pattern);

    template<typename EncIter, typename SpecIter>
    static inline CoreBPE create(EncIter encFirst,
                                 EncIter encLast,
                                 SpecIter specFirst,
                                 SpecIter specLast,
                                 const std::string& pat) {
        ByteVecRankMap encMap;
        for (auto it = encFirst; it != encLast; ++it)
            encMap[it->first] = it->second;

        std::unordered_map<std::string,Rank> specMap;
        for (auto it = specFirst; it != specLast; ++it)
            specMap[it->first] = it->second;

        return CoreBPE(std::move(encMap), std::move(specMap), pat);
    }

    static const std::string& patternString() {
        static const std::string pat =
            R"('([sdmt]|ll|ve|re)| ?[A-Za-z]+| ?\d+| ?[^\sA-Za-z0-9]+|\s+)";
        return pat;
    }

    static const std::regex& mainRegex() {
        static const std::regex re(
            patternString(),
            std::regex_constants::ECMAScript 
        | std::regex_constants::optimize
        );
        return re;
    }

    // Encoding 
    std::vector<Rank> encodeOrdinary(const std::string& text) const;

    std::pair<std::vector<Rank>, std::size_t> encode(const std::string& text,
                                                     const std::unordered_set<std::string>& allowedSpecial) const;

    std::vector<Rank> enocodeWithSpecialTokens(const std::string& text) const;

    std::pair<std::vector<Rank>, std::set<std::vector<Rank>>> 
    encodeUnstableNative(const std::string& text, const std::unordered_set<std::string>& allowedSpecial) const;

    // Decode
    std::optional<ByteVec> decodeBytes(const std::vector<Rank>& tokens) const;

    // Metadata
    std::set<std::string> specialTokens() const;

private:

    const std::regex& threadLocalRegex() const;
    const std::regex& threadLocalSpecialRegex() const;

    static std::string buildSpecialPattern(
        const std::unordered_map<std::string,Rank>& special
    ) {
        static const std::regex esc(R"([.^$|()\[\]{}*+?\\])");

        std::string pat;
        for (auto it = special.begin(); it != special.end(); ++it) {
            if (it != special.begin()) 
                pat += '|';
            pat += std::regex_replace(
                it->first,
                esc,
                R"(\\$&)"          
            );
        }
        return pat;
    }

    static std::regex makeSpecialRegex(
        const std::unordered_map<std::string,Rank>& special
    ) {
        return std::regex(
            buildSpecialPattern(special),
            std::regex_constants::ECMAScript 
        | std::regex_constants::optimize
        );
    }

    std::pair<std::vector<Rank>, std::size_t> increaseLastPieceTokenLen(std::vector<Rank> token,
                                                                        std::size_t lastPieceTokenLen) const;

    ByteVecRankMap encoder_;
    std::unordered_map<std::string, Rank> specialEncoder_;

    std::unordered_map<Rank, ByteVec>  decoder_;          
    std::unordered_map<Rank, ByteVec>  specialDecoder_;   

    std::vector<std::regex> regexTLS_; 
    std::vector<std::regex> specialRegexTLS_;

    std::vector<ByteVec> sortedTokenBytes_;

    static constexpr std::size_t MAX_NUM_THREADS = 128;
};


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

}}}

#endif

