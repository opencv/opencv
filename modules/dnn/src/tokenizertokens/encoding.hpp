#pragma once

#include <opencv2/core.hpp>
#include "core_bpe.hpp"

#include <string>
#include <vector>
#include <regex>
#include <unordered_map>
#include <unordered_set>

namespace cv { namespace dnn { namespace tokenizer {

class CV_EXPORTS_W Encoding {
public:
    CV_WRAP Encoding(const std::string &name, 
                     const std::string &patStr,
                     const ByteVecRankMap &mergeableRanks,
                     const std::unordered_map<std::string, Rank>& specialTokens, 
                     int explicitNvocab=-1);

    CV_WRAP std::vector<Rank> encodeOrdinary(const std::string& text) const;
    CV_WRAP std::vector<Rank> encode(const std::string& text,
                                     const std::unordered_set<std::string>& allowedSpecial={},
                                     const std::unordered_set<std::string>& isallowedSpecial={}) const;
    CV_WRAP std::string decode(const std::vector<Rank>& tokens, const std::string& errors="replace") const;
    CV_WRAP Rank encodeSingleToken(const std::vector<std::uint8_t>& bytes) const;
    CV_WRAP std::vector<std::uint8_t> decodeSingleTokenBytes(Rank token) const;
    CV_WRAP std::vector<std::vector<std::uint8_t>> decodeTokensBytes(const std::vector<Rank>& tokens) const;

    // Get the highest token ID present.
    CV_PROP Rank maxTokenValue() const;

private:
    std::string name_;
    std::string patStr_;
    std::regex patRegex_;
    
    ByteVecRankMap mergeableRanks_;
    std::unordered_map<std::string, Rank> specialTokens_;
    Rank maxTokenValue_;

    CoreBPE coreBPE_;
};

}}}