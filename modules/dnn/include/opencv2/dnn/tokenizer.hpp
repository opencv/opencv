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
    Tokenizer(std::shared_ptr<Encoding> e);
    CV_WRAP static Tokenizer load(const std::string& model_dir); 
    // Encoding
    CV_WRAP std::vector<int> encode(const std::string& text,
                            bool add_special_tokens=false);
    std::vector<int> encode(const std::string& text,
                                     const std::unordered_set<std::string>& allowedSpecial={},
                                     const std::unordered_set<std::string>& disallowedSpecial={}) const;
    // Decoding
    std::string decode(const std::vector<int>& tokens);
    // Accessors
    Encoding& encoding() {return *enc_;}

private:
    std::shared_ptr<Encoding> enc_; 
};

}}}

