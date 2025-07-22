#pragma once

#include <memory>
#include <string>
#include <vector>

#include "encoding.hpp"

namespace cv { namespace dnn { namespace tokenizer {

class CV_EXPORTS_W_SIMPLE Tokenizer {
public:
    explicit Tokenizer(std::shared_ptr<Encoding> e,
                   std::string model_name = "")
    : enc_(std::move(e)),
      tokenizer_name(std::move(model_name)) {}

    CV_EXPORTS static Tokenizer from_pretrained(const std::string& name, const std::string& pretrained_model_path); 
    CV_EXPORTS static Tokenizer train_bpe_from_corpus(const std::string& corpus,
                                   int vocab_sz,
                                   const std::string& pattern) {
        auto enc = std::make_shared<Encoding>(corpus, vocab_sz, pattern);
        return Tokenizer(std::move(enc));
    }
    CV_EXPORTS static Tokenizer train_bpe_from_corpus(const std::vector<std::string>& corpus,
                                   int vocab_sz,
                                   const std::string& pattern,
                                   int min_freq=2, 
                                   int max_token_length=std::numeric_limits<int>::max(),
                                    bool verbose=false){
        auto enc = std::make_shared<Encoding>(corpus, vocab_sz, pattern, min_freq, max_token_length, verbose);
        return Tokenizer(std::move(enc));
    }
    // Encoding
    std::vector<Rank> encode(const std::string& text,
                                     const std::unordered_set<std::string>& allowedSpecial={},
                                     const std::unordered_set<std::string>& disallowedSpecial={}) const {
        return enc_->encode(text, allowedSpecial, disallowedSpecial);
    };
    std::vector<Rank> encodeOrdinary(const std::string& text) const {return enc_->encodeOrdinary(text); }
    Rank encodeSingleToken(const std::vector<std::uint8_t>& bytes) const { return enc_->encodeSingleToken(bytes); }
    // Decoding
    std::string decode(const std::vector<Rank>& tokens) { return enc_->decode(tokens); };
    std::vector<std::uint8_t> decodeBytes(const std::vector<Rank>& tokens) const { return enc_->decodeBytes(tokens); } 
    std::vector<std::uint8_t> decodeSingleTokenBytes(Rank token) const { return enc_->decodeSingleTokenBytes(token); }

    // Accessors
    Encoding& encoding() {return *enc_;}

private:
    std::string tokenizer_name;
    std::string file_path; 
    std::shared_ptr<Encoding> enc_; 
};

}}}

