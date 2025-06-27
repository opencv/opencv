#pragma once

#include <opencv2/core.hpp>
#include "core_bpe.hpp"
#include <unicode/unistr.h>
#include <unicode/regex.h>

#include <string>
#include <vector>
#include <regex>
#include <vector>
#include <cstdint>
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

    /* --------------------- Training ----------------------------------*/
    CV_EXPORTS void train(const std::string& text, int vocabSize, bool verbose=false);

    /* --------------------- Encoding ----------------------------------*/
    CV_WRAP std::vector<Rank> encodeOrdinary(const std::string& text) const;
    CV_WRAP std::vector<Rank> encode(const std::string& text,
                                     const std::unordered_set<std::string>& allowedSpecial={},
                                     const std::unordered_set<std::string>& disallowedSpecial={}) const;
    /*---------------------------------------------------------------------------*/ 
    // Might not need these functionality but we declare them here in case we do.
    CV_WRAP std::vector<std::vector<Rank>> encodeOrdinaryBatch(const std::vector<std::string>& textsm, int numThreads=8) const; 
    CV_WRAP std::vector<std::vector<Rank>> encodeBatch(const std::vector<std::string>& texts,
                                                       int numThreads=8,
                                                       const std::unordered_set<std::string>& allowedSpecial={},
                                                       const std::unordered_set<std::string>& disallowedSpecial={}) const;
    CV_WRAP std::pair<std::vector<Rank>, std::vector<std::vector<Rank>>> encodeWithUnstable(const std::string& text, 
                                                                                            const std::unordered_set<std::string>& allowedSpecial = {},
                                                                                            const std::unordered_set<std::string>& disallowedSpecial = {}) const;
    /*---------------------------------------------------------------------------*/ 
    CV_WRAP Rank encodeSingleToken(const std::vector<std::uint8_t>& bytes) const;

    /* --------------------- Decoding ----------------------------------*/
    CV_WRAP std::vector<std::uint8_t> decodeBytes(const std::vector<Rank>& tokens) const;
    CV_WRAP std::string decode(const std::vector<Rank>& tokens, const std::string& errors="replace") const;
    CV_WRAP std::vector<std::uint8_t> decodeSingleTokenBytes(Rank token) const;
    CV_WRAP std::vector<std::vector<std::uint8_t>> decodeTokensBytes(const std::vector<Rank>& tokens) const;
    // Might used these extra functions keep for now
    CV_EXPORTS std::pair<std::string, std::vector<int>> decodeWithOffsets(const std::vector<Rank>& tokens) const;
    CV_EXPORTS std::vector<std::string> decodeBatch(const std::vector<std::vector<Rank>>& tokenBatches, 
                                                    const std::string& errors ="replace",
                                                    int numThreads=8) const;
    CV_EXPORTS std::vector<std::vector<std::uint8_t>> decodeBytesBatch(const std::vector<std::vector<Rank>>& tokenBatches) const;


    /*Declare extra functionality similar to tiktoken which we might use*/
    CV_PROP std::vector<std::vector<std::uint8_t>> tokenBytesValues() const;
    CV_PROP std::uint32_t eotToken() const;
    CV_PROP std::unordered_set<std::string> specialTokens() const;
    CV_PROP bool isSpecialToken(int token) const;
    CV_PROP int nVocab() const { return maxTokenValue_ + 1; }





    // Get the highest token ID present.
    CV_PROP Rank maxTokenValue() const;

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

    // 2) encodeUTF8 now takes a wstring, converts it, then emits ints
    std::vector<int> encodeUTF8(const std::string &uf8) {
        std::vector<int> out;
        out.reserve(utf8.size());
        for (unsigned char c : utf8) {
            out.push_back(static_cast<uint8_t>(c));
        }
        return out;
    }

private:
    std::string name_;
    std::string patStr_;
    std::unique_ptr<icu::RegexPattern> patRegex_;
    std::unique_ptr<icu::RegexPattern> compiledPattern;
    
    ByteVecRankMap mergeableRanks_;
    std::unordered_map<std::string, Rank> specialTokens_;
    Rank maxTokenValue_;

    CoreBPE coreBPE_;
    // CoreSentencePiece sentencepiece;
  
    std::map<std::pair<int,int>,int> merges_;
    std::map<int, std::vector<uint8_t>> vocab_;
    

    // Might need these functions for testing 
    std::vector<Rank> encodeSinglePiece(const std::string& text) const;
    std::vector<Rank> encodeBytes(const std::vector<std::uint8_t>& bytes) const;
    CV_WRAP std::vector<Rank> _encodeSinglePieceBytes(const std::vector<uint8_t>& bytes) const;
    CV_WRAP std::vector<Rank> _encodeOnlyNativeBpe(const std::string& text) const;
    CV_WRAP std::vector<Rank> _encodeBytesLower(const std::vector<uint8_t>& bytes) const;
};


}}}