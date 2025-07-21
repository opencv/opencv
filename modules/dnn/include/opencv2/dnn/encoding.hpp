#pragma once

#include <opencv2/core.hpp>
#include "../../../src/tokenizer/core_bpe.hpp"
#include "../../../src/tokenizer/unicode.hpp"

#include <string>
#include <vector>
#include <regex>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <fstream>
#include <sstream>
#include <queue>
#include <limits>

namespace cv { namespace dnn { namespace tokenizer {

static constexpr int MAX_TOKEN_LENGTH = std::numeric_limits<int>::max();

class CV_EXPORTS_W_SIMPLE Encoding {
public:
    Encoding()
        : name_(""), patStr_(""), mergeableRanks_(), specialTokens_(),
          maxTokenValue_(0), coreBPE_(), merges_(), vocab_() {}

    Encoding(const std::string &name, 
                     const std::string &patStr,
                     const ByteVecRankMap &mergeableRanks,
                     const std::unordered_map<std::string, uint32_t>& specialTokens, 
                     int explicitNvocab=-1);

    /* --------------------- Training ----------------------------------*/
    // New constructor for simple training
    Encoding(const std::string& text, int vocabSize, const std::string& patStr)
        : name_("trained_encoding"),
          patStr_(patStr),
          mergeableRanks_(),
          specialTokens_() {
        train_bpe(text, vocabSize, /*verbose=*/false);
        coreBPE_ = CoreBPE(mergeableRanks_, specialTokens_, patStr_);
    }

    Encoding(const std::vector<std::string>& texts, int vocabSize, const std::string& patStr, 
                        int minFreq=2, int max_token_length=2147483647, bool verbose=false)
        : name_("trained_encoding_hugface"),
           patStr_(patStr),
           mergeableRanks_(),
           specialTokens_() {
        train_bpe_v2(texts, vocabSize, minFreq, max_token_length, verbose);
    }

    CV_EXPORTS void train_bpe(const std::string& text, int vocabSize, bool verbose=false);
    CV_EXPORTS void train_bpe_v2(const std::vector<std::string>& texts, int vocabSize, int minFreq,  
                                      int max_token_length=2147483647, bool verbose=false);

    /* --------------------- Encoding ----------------------------------*/
    CV_WRAP std::vector<uint32_t> encodeOrdinary(const std::string& text) const;
    CV_WRAP std::vector<uint32_t> encode(const std::string& text) const {return encode(text, {}, {}); };
    CV_WRAP std::vector<uint32_t> encode(const std::string& text, 
                        const std::vector<std::string>& allowedSpecial) const {
        std::unordered_set<std::string> allowed(allowedSpecial.begin(), allowedSpecial.end());
        return encode(text, allowed, {}); 
    };
    std::vector<uint32_t> encode(const std::string& text,
                                     const std::unordered_set<std::string>& allowedSpecial,
                                     const std::unordered_set<std::string>& disallowedSpecial) const;
    /*---------------------------------------------------------------------------*/ 
    // Might not need these functionality but we declare them here in case we do.
    CV_WRAP std::vector<std::vector<uint32_t>> encodeOrdinaryBatch(const std::vector<std::string>& textsm, int numThreads=8) const; 
    std::vector<std::vector<uint32_t>> encodeBatch(const std::vector<std::string>& texts,
                                                       int numThreads=8,
                                                       const std::unordered_set<std::string>& allowedSpecial={},
                                                       const std::unordered_set<std::string>& disallowedSpecial={}) const;
    std::pair<std::vector<uint32_t>, std::vector<std::vector<uint32_t>>> encodeWithUnstable(const std::string& text, 
                                                                                            const std::unordered_set<std::string>& allowedSpecial={},
                                                                                            const std::unordered_set<std::string>& disallowedSpecial={}) const;
    /*---------------------------------------------------------------------------*/ 
    uint32_t encodeSingleToken(const std::vector<std::uint8_t>& bytes) const;

    /* --------------------- Decoding ----------------------------------*/
    std::vector<std::uint8_t> decodeBytes(const std::vector<uint32_t>& tokens) const;
    std::string decode(const std::vector<uint32_t>& tokens, const std::string& errors="replace") const;
    std::vector<std::uint8_t> decodeSingleTokenBytes(uint32_t token) const;
    std::vector<std::vector<std::uint8_t>> decodeTokensBytes(const std::vector<uint32_t>& tokens) const;
    // Might used these extra functions keep for now
    std::pair<std::string, std::vector<int>> decodeWithOffsets(const std::vector<uint32_t>& tokens) const;
    std::vector<std::string> decodeBatch(const std::vector<std::vector<uint32_t>>& tokenBatches, 
                                                    const std::string& errors ="replace",
                                                    int numThreads=8) const;
    std::vector<std::vector<std::uint8_t>> decodeBytesBatch(const std::vector<std::vector<uint32_t>>& tokenBatches) const;


    /*Declare extra functionality similar to tiktoken which we might use*/
    std::vector<std::vector<std::uint8_t>> tokenBytesValues() const;
    std::uint32_t eotToken() const;
    std::unordered_set<std::string> specialTokens() const;
    bool isSpecialToken(int token) const;
    int nVocab() const { return maxTokenValue_ + 1; }

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
    uint32_t maxTokenValue() const { return maxTokenValue_; }
    std::string getName() const { return name_; }

private:
    std::string name_;
    std::string patStr_;
    ByteVecRankMap mergeableRanks_;
    std::unordered_map<std::string, uint32_t> specialTokens_;
    uint32_t maxTokenValue_;
    CoreBPE coreBPE_;
    // CoreSentencePiece sentencepiece;
    std::map<std::pair<int,int>,int> merges_;
    std::map<int, std::vector<uint8_t>> vocab_;
    // Might need these functions for testing 
    std::vector<uint32_t> encodeSinglePiece(const std::string& text) const;
    std::vector<uint32_t> encodeBytes(const std::vector<std::uint8_t>& bytes) const;
    CV_WRAP std::vector<uint32_t> _encodeSinglePieceBytes(const std::vector<uint8_t>& bytes) const;
    CV_WRAP std::vector<uint32_t> _encodeOnlyNativeBpe(const std::string& text) const;
    CV_WRAP std::vector<uint32_t> _encodeBytesLower(const std::vector<uint8_t>& bytes) const;
};

std::unordered_map<std::string,int> dataGymToMergeableBpeRanks(
                                        const std::string& vocabBpePath,
                                        const std::string& encoderJsonPath);

CV_EXPORTS Encoding getEncodingForGPT2(const std::string &name);
CV_WRAP Encoding getEncodingForCl100k_base(const std::string &name);
// CV_WRAP std::vector<int> encodeCl100k_base(const std::string& name, const std::string& text); 
}}}

namespace cv { namespace dnn {

CV_WRAP std::vector<int> encodeCl100k_base(const std::string& name, const std::string& text);

}} // namespace cv::dnn