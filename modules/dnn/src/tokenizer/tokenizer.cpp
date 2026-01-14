// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <opencv2/dnn/dnn.hpp>
#include "utils.hpp"
#include "unicode.hpp"
#include "core_bpe.hpp"

#include <functional>

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

// Registry of implementations (method -> methodImpl)
using ImplRegistry = std::function<Ptr<Tokenizer::Impl>(const FileStorage& cfg, const std::string& dir)>;

static std::unordered_map<std::string, ImplRegistry>& tokenizerRegistry() {
    static std::unordered_map<std::string, ImplRegistry> reg;
    return reg;
}

CoreBPE buildTokenizerGPT(const std::string& model_type, const std::string& json_path);

struct Tokenizer::Impl {
    virtual ~Impl() {}
    virtual std::vector<int> encode(const std::string& text) = 0;
    virtual std::string decode(const std::vector<int>& tokens) = 0;
};

struct BpeTokenizerImpl : public Tokenizer::Impl {
    Ptr<CoreBPE> coreBPE;

    explicit BpeTokenizerImpl(CoreBPE core)
        : coreBPE(makePtr<CoreBPE>(std::move(core))) {}

    std::vector<int> encode(const std::string& text) override {
        CV_Assert(coreBPE);
        std::vector<uint32_t> tok = coreBPE->encode(text, {}).first;
        return std::vector<int>(tok.begin(), tok.end());
    }

    std::string decode(const std::vector<int>& tokens) override {
        CV_Assert(coreBPE);
        std::vector<uint32_t> t32(tokens.begin(), tokens.end());
        auto opt_bytes = coreBPE->decodeBytes(t32);
        if (!opt_bytes)
            CV_Error(cv::Error::StsError, "Invalid decode.");
        const auto& bytes = *opt_bytes;
        return std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    }
};

static void registerDefaultTokenizers() {
    auto& reg = tokenizerRegistry();
    if (reg.find("BPE") == reg.end()) {
        reg["BPE"] = [](const FileStorage& cfg, const std::string& dir) -> Ptr<Tokenizer::Impl> {
            std::string model_type;
            cfg["model_type"] >> model_type;
            std::string tok_json = dir + "tokenizer.json";

            CoreBPE core;
            if (model_type == "gpt2" || model_type == "gpt4") {
                core = buildTokenizerGPT(model_type, tok_json);
            } else {
                CV_Error(cv::Error::StsError, "Unsupported model_type for BPE: " + model_type);
            }
            return makePtr<BpeTokenizerImpl>(std::move(core));
        };
    }
}

// Constructor
Tokenizer::Tokenizer(TokenizeMethod) : impl_(nullptr) {}

// The Load Function
Tokenizer Tokenizer::load(const std::string& model_config, TokenizeMethod method) {
    cv::FileStorage cfg(model_config, cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);
    if (!cfg.isOpened())
        CV_Error(cv::Error::StsError, "Could not open config.json: " + model_config);

    std::string dir = model_config;
    size_t pos = dir.find_last_of("/\\");
    dir = (pos == std::string::npos) ? std::string() : dir.substr(0, pos + 1);

    // Map Enum to internal string key
    std::string methodType;
    switch (method) {
        // FIX: Use scoped name
        case TokenizeMethod::DNN_TOKENIZER_BPE: 
            methodType = "BPE"; 
            break;
        default: 
            CV_Error(cv::Error::StsBadArg, "Unknown TokenizeMethod");
    }

    registerDefaultTokenizers();
    auto& reg = tokenizerRegistry();
    auto it = reg.find(methodType);
    if (it == reg.end())
        CV_Error(cv::Error::StsError,
            "Unsupported tokenizer method: '" + methodType + "'. Supported: BPE");

    Tokenizer tok;
    tok.impl_ = it->second(cfg, dir);
    return tok;
}

std::vector<int> Tokenizer::encode(const std::string& text) {
    if (!impl_) CV_Error(cv::Error::StsError, "Tokenizer impl null");
    return impl_->encode(text);
}

std::string Tokenizer::decode(const std::vector<int>& tokens) {
    if (!impl_) CV_Error(cv::Error::StsError, "Tokenizer impl null");
    return impl_->decode(tokens);
};

CoreBPE buildTokenizerGPT(const std::string& model_type, const std::string& json_path) {
    cv::FileStorage fs(json_path, cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);
    if (!fs.isOpened())
        CV_Error(cv::Error::StsError, "Failed to open tokenizer.json: " + json_path);

    cv::FileNode model = fs["model"];
    CV_CheckFalse(model.empty(), "tokenizer.json missing 'model'");
    cv::FileNode vocab = model["vocab"];
    CV_CheckFalse(vocab.empty(), "tokenizer.json missing model.vocab");

    std::string pattern;
    std::unordered_set<std::string> skip_tokens;
    if (model_type == "gpt2" || model_type == "r50k_base") {
        pattern = R50K_UTF8;
        skip_tokens.insert("<|endoftext|>");
    } else if (model_type == "gpt4" || model_type == "cl100k_base") {
        pattern = CL100K_BASE;
    } else {
        CV_Error(cv::Error::StsError,
            "Unsupported model_type: " + model_type + " (expected gpt2/r50k_base or gpt4/cl100k_base)");
    }

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
    mergeableRanks.reserve((size_t)vocab.size());
    int max_id = -1;

    for (cv::FileNodeIterator it = vocab.begin(); it != vocab.end(); ++it) {
        FileNode val = *it;
        std::string token = val.name();
        if (skip_tokens.find(token) != skip_tokens.end()) continue;
        int id = (int)val;
        mergeableRanks.emplace(token_to_bytes(token), (uint32_t)id);
        if (id > max_id) max_id = id;
    }

    std::unordered_map<std::string, uint32_t> specialTokens;
    FileNode added = fs["added_tokens"];
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

    return CoreBPE(std::move(mergeableRanks), std::move(specialTokens), pattern);
}

CV__DNN_INLINE_NS_END
}}