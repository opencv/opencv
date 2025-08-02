#pragma once

#include <opencv2/core.hpp>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <limits>


namespace cv { namespace dnn { namespace tokenizer {

class Encoding;

class CV_EXPORTS_W_SIMPLE Tokenizer {
public:

    CV_WRAP Tokenizer();
    Tokenizer(std::shared_ptr<Encoding> e,
                   std::string model_name = "");

    CV_WRAP static Tokenizer from_pretrained(const std::string& name, const std::string& pretrained_model_path); 
    CV_EXPORTS static Tokenizer train_bpe_from_corpus(const std::string& corpus,
                                   int vocab_sz,
                                   const std::string& pattern);
    CV_EXPORTS static Tokenizer train_bpe_from_corpus(const std::vector<std::string>& corpus,
                                   int vocab_sz,
                                   const std::string& pattern,
                                   int min_freq=2, 
                                   int max_token_length=std::numeric_limits<int>::max(),
                                    bool verbose=false);
    // Encoding
    CV_WRAP std::vector<int> encode(const std::string& text,
                            bool add_special_tokens=false);
    std::vector<int> encode(const std::string& text,
                                     const std::unordered_set<std::string>& allowedSpecial={},
                                     const std::unordered_set<std::string>& disallowedSpecial={}) const;
    std::vector<uint32_t> encodeOrdinary(const std::string& text) const;
    uint32_t encodeSingleToken(const std::vector<std::uint8_t>& bytes) const;
    // Decoding
    std::string decode(const std::vector<int>& tokens);
    // std::string decode(const std::vector<uint32_t>& tokens) { return enc_->decode(tokens); };
    std::vector<std::uint8_t> decodeBytes(const std::vector<uint32_t>& tokens) const;
    std::vector<std::uint8_t> decodeSingleTokenBytes(uint32_t token) const;

    // Accessors
    Encoding& encoding() {return *enc_;}

private:
    std::string tokenizer_name;
    std::string file_path; 
    std::shared_ptr<Encoding> enc_; 
};

}}}

