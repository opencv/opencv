#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <regex>
#include <set>
#include <stdexcept>
#include <thread>
#include <unordered_set>
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

std::size_t hashCurrentThread();

std::vector<std::pair<std::size_t, Rank>> bytePairMerge(const ByteVecRankMap& ranks, 
                                                        const ByteVec& piece);

std::vector<Rank> bytePairEncode(const ByteVec& piece, 
                                 const ByteVecRankMap& ranks);

std::vector<ByteVec> bytePairSplit(const ByteVec& piece, 
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

class CoreBPE {
public:
    template<typename EncIter, typename SpecIter>
    static CoreBPE create(EncIter encFist, EncIter encLast, 
                          SpecIter specFirst, SpecIter specLast,
                          const std::string& pattern);

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
    CoreBPE(ByteVecRankMap encoder,
            std::unordered_map<std::string, Rank> specialEncoder, 
            const std::string& pattern);

    const std::regex& threadLocalRegex() const;
    const std::regex& threadLocalSpecialRegex() const;

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

}}}

