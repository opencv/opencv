#include "encoding.hpp"
#include <cassert>

namespace cv { namespace dnn { namespace tokenizer {

Encoding::Encoding(const std::string &name, 
                     const std::string &patStr,
                     const ByteVecRankMap &mergeableRanks,
                     const std::unordered_map<std::string, Rank>& specialTokens, 
                     int explicitNvocab=-1) 
    : name_(name)
    , patStr_(patStr)
    , patRegex_(patStr)
    , mergeableRanks_(mergeableRanks)
    , specialTokens_(specialTokens)
    , coreBPE_(mergeableRanks_, specialTokens_, patStr_) {
    
    // compute max token value
    Rank mrMax = 0;
    for (auto& kv : mergeableRanks_) 
        mrMax = std::max(mrMax, kv.second);
    Rank stMax = 0;
    for (auto& kv : specialTokens_) 
        stMax = std::max(stMax, kv.second);
    maxTokenValue_ = std::max(mrMax, stMax);
    if (explicitNvocab > 0) {
        assert(static_cast<int>(mergeableRanks_.size() + specialTokens_.size()) == explicitNvocab);
        assert(maxTokenValue_ == static_cast<Rank>(explicitNvocab-1));
    }
}

std::vector<Rank> Encoding::encodeOrdinary(const std::string& text) const {
    try {
        return coreBPE_.encodeOrdinary(text);
    } catch(const std::exception &e) {
        // TODO: handle UTF-16 surrogate workaround
        std::string fixed = text; // placeholder
        return coreBPE_.encodeOrdinary(fixed);
    }
}




}}}