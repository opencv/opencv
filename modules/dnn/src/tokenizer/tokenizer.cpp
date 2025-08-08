#include "../../include/opencv2/dnn/tokenizer.hpp"
#include "encoding.hpp"
#include "gpt2_tokenizer_fast.hpp"

namespace cv { namespace dnn { namespace tokenizer {

Tokenizer::Tokenizer() : enc_(nullptr) {}

Tokenizer::Tokenizer(std::shared_ptr<Encoding> e)
    : enc_(std::move(e)) {
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

std::string Tokenizer::decode(const std::vector<int>& tokens) { 
        std::vector<uint32_t> tokens32(tokens.begin(), tokens.end());
        return enc_->decode(tokens32); 
};

Tokenizer Tokenizer::load(const std::string& model_dir) {
    const std::string cfg_path = model_dir + "config.json";
    cv::FileStorage cfg(cfg_path, cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);
    if (!cfg.isOpened()) 
        CV_Error(cv::Error::StsError, "Could not open config.json at: " + cfg_path);

    std::string model_type;
    cfg["model_type"] >> model_type;
    const std::string tok_json = model_dir + "tokenizer.json";

    std::shared_ptr<Encoding> enc;
    if (model_type == "gpt2") {
        enc = std::make_shared<Encoding>(
            GPT2TokenizerFast::from_pretrained(tok_json).encoding()
        );
    } else if (model_type == "gpt4") {
        enc = std::make_shared<Encoding>(
            getEncodingForCl100k_baseFromJSON_FS("cl100k_base", tok_json)
        );
    } else {
        CV_Error(cv::Error::StsError, "Unsupported model_type in config.json: " + model_type);
    }
    return Tokenizer(std::move(enc));
}

}}}