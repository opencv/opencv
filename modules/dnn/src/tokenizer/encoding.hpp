#pragma once

#include <opencv2/core.hpp>
#include "core_bpe.hpp"
#include "unicode.hpp"

#include <string>
#include <vector>
#include <regex>
#include <vector>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <fstream>
#include <sstream>
#include <queue>
#include <memory>

namespace cv { namespace dnn { namespace tokenizer {

class CV_EXPORTS_W Encoding {
public:
    Encoding()
        : name_(""), patStr_(""), mergeableRanks_(), specialTokens_(),
          maxTokenValue_(0), coreBPE_(), merges_(), vocab_() {}

    CV_WRAP Encoding(const std::string &name, 
                     const std::string &patStr,
                     const ByteVecRankMap &mergeableRanks,
                     const std::unordered_map<std::string, Rank>& specialTokens, 
                     int explicitNvocab=-1);

    /* --------------------- Encoding ----------------------------------*/
    CV_WRAP std::vector<Rank> encode(const std::string& text,
                                     const std::unordered_set<std::string>& allowedSpecial={},
                                     const std::unordered_set<std::string>& disallowedSpecial={}) const;

    /* --------------------- Decoding ----------------------------------*/
    CV_WRAP std::vector<std::uint8_t> decodeBytes(const std::vector<Rank>& tokens) const;
    CV_EXPORTS std::string decode(const std::vector<Rank>& tokens, const std::string& errors="replace") const;

    std::vector<int> encodeUTF8(const std::string &utf8) {
        std::vector<int> out;
        out.reserve(utf8.size());
        for (unsigned char c : utf8) {
            out.push_back(static_cast<uint8_t>(c));
        }
        return out;
    }
    /* --------------------- Load/Save ----------------------------------*/
    void save();
    void load();

    // Accessors 
    const std::map<std::pair<int,int>, int>& getMerges() const { return merges_; }
    const std::map<int, std::vector<uint8_t>>& getVocab() const { return vocab_; }
    Rank maxTokenValue() const { return maxTokenValue_; }
    std::string getName() const { return name_; }

private:
    std::string name_;
    std::string patStr_;
    ByteVecRankMap mergeableRanks_;
    std::unordered_map<std::string, Rank> specialTokens_;
    Rank maxTokenValue_;
    CoreBPE coreBPE_;
    // CoreSentencePiece sentencepiece;
    std::map<std::pair<int,int>,int> merges_;
    std::map<int, std::vector<uint8_t>> vocab_;
};

CV_EXPORTS Encoding getEncodingForCl100k_base(const std::string &name, const std::string& cl00k_case_file);

CV_EXPORTS Encoding getEncodingForCl100k_baseFromJSON_FS(const std::string &name,
                                                         const std::string &json_path);


}}}