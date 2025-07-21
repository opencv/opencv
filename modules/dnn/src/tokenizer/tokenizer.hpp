#pragma once

#include <memory>
#include <string>
#include <vector>

#include "encoding.hpp"

namespace cv { namespace dnn { namespace tokenizer {

class CV_EXPORTS Tokenizer {
public:
    explicit Tokenizer(std::shared_ptr<Encoding> e,
                   std::string model_name = "")
    : enc_(std::move(e)),
      tokenizer_name(std::move(model_name)) {}

    CV_EXPORTS static Tokenizer from_pretrained(const std::string& name, const std::string& pretrained_model_path); 
    static Tokenizer train_from_corpus(const std::vector<std::string>& corpus,
                                   int vocab_sz,
                                   const std::string& pattern,
                                   int min_freq = 2) {
        auto enc = std::make_shared<Encoding>(corpus, vocab_sz, pattern, min_freq);
        return Tokenizer(std::move(enc));
    }
    std::vector<Rank> encode(const std::string& text) { return enc_->encode(text); }
    std::vector<Rank> encodeOrdinary(const std::string& text) const {return enc_->encodeOrdinary(text); }
    std::string decode(const std::vector<Rank>& tokens) { return enc_->decode(tokens); };
    Encoding& encoding() {return *enc_;}

private:
    std::string tokenizer_name;
    std::string file_path; 
    std::shared_ptr<Encoding> enc_; 
};

}}}