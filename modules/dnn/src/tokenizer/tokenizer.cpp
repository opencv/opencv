// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <opencv2/dnn/dnn.hpp>
#include "utils.hpp"
#include "unicode.hpp"
#include "core_bpe.hpp"


namespace cv { namespace dnn { 
CV__DNN_INLINE_NS_BEGIN

CoreBPE getTokenizerForGPT2FromJSON(const std::string &name, const std::string& json_path);
CoreBPE getTokenizerForCl100kBaseFromJSON_FS(const std::string &name,
                                                         const std::string &json_path);

struct Tokenizer::Impl {
    TokenizeMethod method;
    Ptr<CoreBPE> coreBPE;

    Impl(TokenizeMethod m) : method(m) {}

    std::vector<int> encode(const std::string& text) {
        switch (method) {
            case TokenizeMethod::BPE: {
                CV_Assert(coreBPE);
                std::vector<uint32_t> tok = coreBPE->encode(text, {}).first;
                return std::vector<int>(tok.begin(), tok.end());
            }
            // other cases to be added for example sentence piece
        }
        return {};
    }

    std::string decode(const std::vector<int>& tokens) {
        switch (method) {
            case TokenizeMethod::BPE: {
                CV_Assert(coreBPE);
                std::vector<uint32_t> t32(tokens.begin(), tokens.end());
                auto opt_bytes = coreBPE->decodeBytes(t32);
                if (!opt_bytes)
                    CV_Error(cv::Error::StsError, "Invalid decode.");
                const auto& bytes = *opt_bytes;
                return std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size());
            }
            // other cases to be added for example sentence piece
        }
        return {};
    }

    void loadFromConfig(const std::string& model_config) {
        // We set the full path to config.json smilair to what readNetFromCaffe(prototxt,...) does. 
        cv::FileStorage cfg(model_config, cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);
        if (!cfg.isOpened())
            CV_Error(cv::Error::StsError, "Could not open config.json: " + model_config);
        std::string model_type;
        cfg["model_type"] >> model_type;
        std::string dir = model_config;
        size_t pos = dir.find_last_of("/\\");
        dir = (pos == std::string::npos) ? std::string() : dir.substr(0, pos + 1);
        std::string tok_json = dir + "tokenizer.json";

        switch (method) {
            case TokenizeMethod::BPE: {
                if (model_type == "gpt2") {
                    coreBPE = makePtr<CoreBPE>(getTokenizerForGPT2FromJSON("gpt2", tok_json));
                } else if (model_type == "gpt4") {
                    coreBPE = makePtr<CoreBPE>(getTokenizerForCl100kBaseFromJSON_FS("cl100k_base", tok_json));
                } else {
                    CV_Error(cv::Error::StsError, "Unsupported model_type for BPE: " + model_type);
                }
                break;
            }
            // other cases to be added for example sentence piece
        }
    }
    
};

Tokenizer::Tokenizer(TokenizeMethod method)
    : impl_(makePtr<Impl>(method)) {
}

std::vector<int> Tokenizer::encode(const std::string& text) {
    if (!impl_) CV_Error(cv::Error::StsError, "Tokenizer impl null");
    return impl_->encode(text);
}

std::string Tokenizer::decode(const std::vector<int>& tokens) { 
    if (!impl_) CV_Error(cv::Error::StsError, "Tokenizer impl null");
    return impl_->decode(tokens);
};

CoreBPE getTokenizerForGPT2FromJSON(const std::string &name, const std::string& json_path) {
    cv::FileStorage fs(json_path, cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);
    if (!fs.isOpened()) {
        CV_Error(cv::Error::StsError, "Failed to open tokenizer.json: " + json_path);
    }

    cv::FileNode model = fs["model"];
    CV_CheckFalse(model.empty(), "tokenizer.json missing 'model'");

    cv::FileNode vocab = model["vocab"];
    CV_CheckFalse(vocab.empty(), "tokenizer.json missing model.vocab");

    auto token_to_bytes = [&](const std::string& token_utf8) -> std::vector<uint8_t> {
        std::vector<std::uint8_t> out;
        auto cps = unicode_cpts_from_utf8(token_utf8);  
        out.reserve(cps.size());
        for (uint32_t cp : cps) {
            const std::string one = unicode_cpt_to_utf8(cp);
            out.push_back(unicode_utf8_to_byte(one));
        }
        return out;
    };
    
    ByteVecRankMap mergeableRanks;
    mergeableRanks.reserve(vocab.size());
    int max_id = -1;

    for (cv::FileNodeIterator it = vocab.begin(); it != vocab.end(); ++it) {
        cv::FileNode valNode = *it;
        std::string key = valNode.name(); // token string
        if (key == "<|endoftext|>") continue;
        // std:: cout << key << " ";
        int id = (int)valNode;            // token id
        // std::cout << id << " ";
        mergeableRanks.emplace(token_to_bytes(key), (uint32_t)id);
        max_id = std::max(max_id, id);
    }

    // size sanity
    if ((int)mergeableRanks.size() != (int)vocab.size() - 1) {
        CV_Error(cv::Error::StsError,
                cv::format("Built %zu entries but vocab has %d",
                            mergeableRanks.size(), (int)vocab.size()));
    }

    // every byte 0..255 must be present as a 1-byte token
    for (int b = 0; b < 256; ++b) {
        std::vector<std::uint8_t> key{ (uint8_t)b };
        if (mergeableRanks.find(key) == mergeableRanks.end()) {
            std::ostringstream oss;
            oss << "Missing singleton byte token 0x" << std::hex << b;
            CV_Error(cv::Error::StsError, oss.str());
        }
    }

    std::unordered_map<std::string, uint32_t> specialTokens;
    FileNode added = fs["added_tokens"];
    if (!added.empty()) {
        for (auto it = added.begin(); it != added.end(); it++) {
            FileNode t = *it;
            bool special = false;
            t["special"] >> special;
            int id = -1;
            t["id"] >> id;
            std::string content; 
            t["content"] >> content;

            if (id >= 0 && id > max_id) max_id = id;
            if (special && id >= 0 && !content.empty()) {
                specialTokens.emplace(content, (uint32_t)id);
            }
        }
    }

    return CoreBPE(std::move(mergeableRanks), std::move(specialTokens), R50K_UTF8);
}

CoreBPE getTokenizerForCl100kBaseFromJSON_FS(const std::string &name,
                                                         const std::string &json_path)
{
    if (name != "cl100k_base")
        CV_Error(cv::Error::StsError, "Wrong model name. This model is cl100k_base");

    cv::FileStorage fs(json_path, cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);
    if (!fs.isOpened())
        CV_Error(cv::Error::StsError, "Failed to open tokenizer.json: " + json_path);

    cv::FileNode model = fs["model"];
    CV_CheckFalse(model.empty(), "tokenizer.json missing 'model'");

    cv::FileNode vocab = model["vocab"];
    CV_CheckFalse(vocab.empty(), "tokenizer.json missing model.vocab");

    ByteVecRankMap mergeableRanks;
    mergeableRanks.reserve((size_t)vocab.size());
    int max_id = -1;

    auto token_to_bytes = [&](const std::string& token_utf8) -> std::vector<uint8_t> {
        std::vector<std::uint8_t> out;
        auto cps = unicode_cpts_from_utf8(token_utf8); 
        out.reserve(cps.size());
        for (uint32_t cp : cps) {
            const std::string one = unicode_cpt_to_utf8(cp);
            out.push_back(unicode_utf8_to_byte(one));    
        }
        return out;
    };

    for (cv::FileNodeIterator it = vocab.begin(); it != vocab.end(); ++it) {
        cv::FileNode val = *it;
        std::string token = val.name();    
        int id = (int)val;
        mergeableRanks.emplace(token_to_bytes(token), (uint32_t)id);
        if (id > max_id) max_id = id;
    }

    std::unordered_map<std::string, uint32_t> specialTokens;
    cv::FileNode added = fs["added_tokens"];
    if (!added.empty()) {
        for (auto it = added.begin(); it != added.end(); ++it) {
            cv::FileNode t = *it;
            bool special = false; t["special"] >> special;
            int id = -1;          t["id"]      >> id;
            std::string content;  t["content"] >> content;
            if (special && id >= 0 && !content.empty()) {
                specialTokens.emplace(content, (uint32_t)id);
                if (id > max_id) max_id = id;
            }
        }
    }

    for (int b = 0; b < 256; ++b) {
        std::vector<std::uint8_t> key{ (uint8_t)b };
        if (mergeableRanks.find(key) == mergeableRanks.end()) {
            std::ostringstream oss; oss << "Missing singleton byte token 0x" << std::hex << b;
            CV_Error(cv::Error::StsError, oss.str());
        }
    }
    return CoreBPE(std::move(mergeableRanks), std::move(specialTokens), CL100K_BASE);
}

static Tokenizer::TokenizeMethod parseAlgorithm(const std::string& algorithm) {
    std::string alg;
    alg.reserve(algorithm.size());
    for (char c : algorithm) alg.push_back((char)std::tolower((unsigned char)c));
    if (alg == "bpe") return Tokenizer::TokenizeMethod::BPE;
    CV_Error(cv::Error::StsBadArg, "Unsupported tokenizer algorithm: " + algorithm);
}

Tokenizer Tokenizer::load(const std::string& model_config, const std::string& algorithm) {
    TokenizeMethod m = parseAlgorithm(algorithm);
    Tokenizer tok(m);
    tok.impl_->loadFromConfig(model_config);
    return tok;
}
CV__DNN_INLINE_NS_END
}}