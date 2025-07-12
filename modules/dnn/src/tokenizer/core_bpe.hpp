#pragma once

#include <opencv2/core.hpp>

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <unordered_map>
#include <regex> // Not enough functionality as the Rust/Python version -> change to icu 
#include <set>
#include <stdexcept>
#include <thread>
#include <unordered_set>
#include <algorithm>

#include "unicode.hpp"

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

static constexpr std::size_t MAX_NUM_THREADS = 128;

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
    
    CoreBPE(); 
    
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

    // Encoding 
    std::vector<Rank> encodeOrdinary(const std::string& text) const;

    std::pair<std::vector<Rank>, std::size_t> encode(const std::string& text,
                                                     const std::unordered_set<std::string>& allowedSpecial) const;

    std::vector<Rank> enocodeWithSpecialTokens(const std::string& text) const;

    std::pair<std::vector<Rank>, std::set<std::vector<Rank>>> 
    encodeUnstableNative(const std::string& text, const std::unordered_set<std::string>& allowedSpecial) const;

    Rank encodeSingleToken(std::vector<uint8_t>& piece) const;

    

    // Decode
    std::optional<ByteVec> decodeBytes(const std::vector<Rank>& tokens) const;

    // Metadata
    std::set<std::string> specialTokens() const;

private:

    // const icu::RegexPattern* threadLocalRegex() const;
    // const icu::RegexPattern* threadLocalSpecialRegex() const;

    static std::string makeSpecialPattern(const std::unordered_map<std::string, Rank>& special) {
        static const std::string meta = R"([.^$|()\[\]{}*+?\\])";
        std::string pat;
        pat.reserve(special.size() * 10);
        bool first = true;
        for (auto const& kv : special) {
            if (!first) pat.push_back('|');
            first = false;
            // Escape each character in the token 
            for (char c : kv.first) {
                if (meta.find(c) != std::string::npos) 
                    pat.push_back('\\');
                pat.push_back(c);
            }
        }
        return pat;
    }

    std::pair<std::vector<Rank>, std::size_t> increaseLastPieceTokenLen(std::vector<Rank> token,
                                                                        std::size_t lastPieceTokenLen) const;

    ByteVecRankMap encoder_;
    std::unordered_map<std::string, Rank> specialEncoder_;

    std::unordered_map<Rank, ByteVec>  decoder_;          
    std::unordered_map<Rank, ByteVec>  specialDecoder_;   

    // std::vector<std::regex> regexTLS_; 
    // std::vector<std::regex> specialRegexTLS_;

    std::string pattern_;
    std::string specialPattern_;
    // mutable std::vector<std::shared_ptr<icu::RegexPattern>> regexTLS_;
    // mutable std::vector<std::shared_ptr<icu::RegexPattern>> specialRegexTLS_;

    std::vector<ByteVec> sortedTokenBytes_;
};


}}}

#endif

