#include "../../include/opencv2/dnn/tokenizer.hpp"
#include "encoding.hpp"
#include "gpt2_tokenizer_fast.hpp"

namespace cv { namespace dnn { namespace tokenizer {

Tokenizer::Tokenizer() : enc_(nullptr), tokenizer_name("") {}

Tokenizer::Tokenizer(std::shared_ptr<Encoding> e,
                     std::string model_name)
    : enc_(std::move(e)), tokenizer_name(std::move(model_name)) {
}

Tokenizer Tokenizer::train_bpe_from_corpus(const std::string& corpus,
                                   int vocab_sz,
                                   const std::string& pattern) {
    auto enc = std::make_shared<Encoding>(corpus, vocab_sz, pattern);
    return Tokenizer(std::move(enc));
}

Tokenizer Tokenizer::train_bpe_from_corpus(const std::vector<std::string>& corpus,
                                   int vocab_sz,
                                   const std::string& pattern,
                                   int min_freq, 
                                   int max_token_length,
                                    bool verbose){
    auto enc = std::make_shared<Encoding>(corpus, vocab_sz, pattern, min_freq, max_token_length, verbose);
    return Tokenizer(std::move(enc));
}

std::vector<int> Tokenizer::encode(const std::string& text,
                            bool add_special_tokens) {
    if (!add_special_tokens) {
        return this->encode(text, {}, {});
    }
    std::unordered_set<std::string> allowedSpecial = {"__ALL__"};
    return this->encode(text, allowedSpecial, {});
}

std::vector<int> Tokenizer::encode(const std::string& text,
                                     const std::unordered_set<std::string>& allowedSpecial,
                                     const std::unordered_set<std::string>& disallowedSpecial) const {
    std::vector<uint32_t> tok = enc_->encode(text, allowedSpecial, disallowedSpecial);
    return std::vector<int>(tok.begin(), tok.end());
};

std::vector<uint32_t> Tokenizer::encodeOrdinary(const std::string& text) const {
    return enc_->encodeOrdinary(text); 
}

uint32_t Tokenizer::encodeSingleToken(const std::vector<std::uint8_t>& bytes) const { 
    return enc_->encodeSingleToken(bytes); 
}

std::string Tokenizer::decode(const std::vector<int>& tokens) { 
        std::vector<uint32_t> tokens32(tokens.begin(), tokens.end());
        return enc_->decode(tokens32); 
};

std::vector<std::uint8_t> Tokenizer::decodeBytes(const std::vector<uint32_t>& tokens) const { 
    return enc_->decodeBytes(tokens); 
} 

std::vector<std::uint8_t> Tokenizer::decodeSingleTokenBytes(uint32_t token) const { 
    return enc_->decodeSingleTokenBytes(token); 
}

Tokenizer Tokenizer::from_pretrained(const std::string& name, const std::string& pretrained_model_path) {
    // We most load files json into FileStorge
    std::shared_ptr<Encoding> enc;
    if (name == "gpt2") {
        enc = std::make_shared<Encoding>(GPT2TokenizerFast::from_pretrained(pretrained_model_path).encoding());
    } else if (name == "cl100k_base") {
      enc = std::make_shared<Encoding>(getEncodingForCl100k_base(name, pretrained_model_path));  
    } else {
        throw std::runtime_error("Unknown model name: " + name);
    }
    return Tokenizer(std::move(enc), name);
}

}}}