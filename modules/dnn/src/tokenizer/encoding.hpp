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

    /* --------------------- Training ----------------------------------*/
    // New constructor for simple training
    CV_EXPORTS Encoding(const std::string& text, int vocabSize, const std::string& patStr)
        : name_("trained_encoding"),
          patStr_(patStr),
          mergeableRanks_(),
          specialTokens_() {
        train_bpe(text, vocabSize, /*verbose=*/false);
        coreBPE_ = CoreBPE(mergeableRanks_, specialTokens_, patStr_);
    }

    CV_EXPORTS Encoding(const std::vector<std::string>& texts, int vocabSize, const std::string& patStr, 
                        int minFreq=2, int max_token_length=std::numeric_limits<int>::max(), bool verbose=false)
        : name_("trained_encoding_hugface"),
           patStr_(patStr),
           mergeableRanks_(),
           specialTokens_() {
        train_bpe_v2(texts, vocabSize, minFreq, max_token_length, verbose);
    }

    CV_EXPORTS void train_bpe(const std::string& text, int vocabSize, bool verbose=false);
    CV_EXPORTS void train_bpe_v2(const std::vector<std::string>& texts, int vocabSize, int minFreq,  
                                      int max_token_length=std::numeric_limits<int>::max(), bool verbose=false);

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
    CV_EXPORTS std::string decode(const std::vector<Rank>& tokens, const std::string& errors="replace") const;
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
    // Might need these functions for testing 
    std::vector<Rank> encodeSinglePiece(const std::string& text) const;
    std::vector<Rank> encodeBytes(const std::vector<std::uint8_t>& bytes) const;
    CV_WRAP std::vector<Rank> _encodeSinglePieceBytes(const std::vector<uint8_t>& bytes) const;
    CV_WRAP std::vector<Rank> _encodeOnlyNativeBpe(const std::string& text) const;
    CV_WRAP std::vector<Rank> _encodeBytesLower(const std::vector<uint8_t>& bytes) const;
};

std::unordered_map<std::string,int> dataGymToMergeableBpeRanks(
                                        const std::string& vocabBpePath
                                       /*const std::string& encoderJsonPath*/);

CV_EXPORTS Encoding getEncodingForGPT2(const std::string &name, const std::string& vocab_file);
CV_EXPORTS Encoding getEncodingForCl100k_base(const std::string &name);

CV_EXPORTS class Tokenizer {
public:
    CV_EXPORTS static Tokenizer from_pretrained(const std::string& name, const std::string& pretrained_model_path); 
    std::vector<Rank> encode(const std::string& text) { return encoder.encode(text); };
    std::string decode(const std::vector<Rank>& tokens) { return encoder.decode(tokens); };
private:
    Tokenizer() = default;
    std::string tokenizer_name;
    std::string file_path; // 
    Encoding encoder; 
};

}}}


// Tokenizer tokenizer = Tokenizer("vocab.bpe")
// tokenizer.