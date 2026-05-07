// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <opencv2/dnn/dnn.hpp>
#include "utils.hpp"
#include "unicode.hpp"
#include "core_bpe.hpp"
#include "core_gemma.hpp"

#include <functional>
#include <unordered_set>

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

// Registry of implementations (method -> methodImpl)
using ImplRegestry = std::function<Ptr<Tokenizer::Impl>(const FileStorage& cfg, const std::string& dir)>;

static std::unordered_map<std::string, ImplRegestry>& tokenizerRegistry() {
    static std::unordered_map<std::string, ImplRegestry> reg;
    return reg;
}

CoreBPE buildTokenizerFromJson(const std::string& model_type, const std::string& json_path,
                          std::unordered_set<std::string>* outSpecial = nullptr);

struct Tokenizer::Impl {
    virtual ~Impl() {}
    virtual std::vector<int> encode(const std::string& text) = 0;
    virtual std::string decode(const std::vector<int>& tokens) = 0;
};

struct BpeTokenizerImpl : public Tokenizer::Impl {
    Ptr<CoreBPE> coreBPE;
    std::unordered_set<std::string> allowedSpecial;

    explicit BpeTokenizerImpl(CoreBPE core,
                              std::unordered_set<std::string> special = {})
        : coreBPE(makePtr<CoreBPE>(std::move(core)))
        , allowedSpecial(std::move(special)) {}

    std::vector<int> encode(const std::string& text) override {
        CV_Assert(coreBPE);
        std::vector<uint32_t> tok = coreBPE->encode(text, allowedSpecial).first;
        return std::vector<int>(tok.begin(), tok.end());
    }

    std::string decode(const std::vector<int>& tokens) override {
        CV_Assert(coreBPE);
        std::vector<uint32_t> t32(tokens.begin(), tokens.end());
        const std::vector<std::uint8_t> opt_bytes = coreBPE->decodeBytes(t32);
        if (opt_bytes.empty())
            CV_Error(cv::Error::StsError, "Invalid decode.");
        return std::string(reinterpret_cast<const char*>(opt_bytes.data()), opt_bytes.size());
    }
};

struct GemmaBpeTokenizerImpl : public Tokenizer::Impl {
    CoreGemmaBPE model;
    std::unordered_set<std::string> allowedSpecial;

    explicit GemmaBpeTokenizerImpl(CoreGemmaBPE m,
                                   std::unordered_set<std::string> special = {})
        : model(std::move(m)), allowedSpecial(std::move(special)) {}

    std::vector<int> encode(const std::string& text) override {
        return model.encode(text, allowedSpecial);
    }

    std::string decode(const std::vector<int>& tokens) override {
        return model.decode(tokens);
    }
};

struct SentencePieceTokenizerImpl : public Tokenizer::Impl {
    CoreGemmaBPE model;
    std::unordered_set<std::string> allowedSpecial;
    int bosTokenId;

    explicit SentencePieceTokenizerImpl(CoreGemmaBPE m,
                                        std::unordered_set<std::string> special = {},
                                        int bos = -1)
        : model(std::move(m)), allowedSpecial(std::move(special)), bosTokenId(bos) {}

    std::vector<int> encode(const std::string& text) override {
        std::vector<int> ids = model.encode(text, allowedSpecial);
        // HuggingFace SentencePiece tokenizers (Gemma2) prepend <bos> automatically
        if (bosTokenId >= 0) {
            ids.insert(ids.begin(), bosTokenId);
        }
        return ids;
    }

    std::string decode(const std::vector<int>& tokens) override {
        // Skip the bos token if present at the beginning
        if (bosTokenId >= 0 && !tokens.empty() && tokens.front() == bosTokenId) {
            std::vector<int> stripped(tokens.begin() + 1, tokens.end());
            return model.decode(stripped);
        }
        return model.decode(tokens);
    }
};

static Ptr<GemmaBpeTokenizerImpl> buildGemmaFromJson(
        const std::string& json_path,
        std::unordered_set<std::string>* outSpecial = nullptr) {

    cv::FileStorage fs(json_path, cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);
    if (!fs.isOpened())
        CV_Error(cv::Error::StsError, "Failed to open tokenizer.json: " + json_path);

    cv::FileNode model_node = fs["model"];
    CV_CheckFalse(model_node.empty(), "tokenizer.json missing 'model'");

    std::string model_type;
    model_node["type"] >> model_type;
    if (model_type != "BPE")
        CV_Error(cv::Error::StsError,
            "Expected BPE model in tokenizer.json for Gemma3, got: " + model_type);

    CoreGemmaBPE gemma;

    cv::FileNode vocab_node = model_node["vocab"];
    CV_CheckFalse(vocab_node.empty(), "tokenizer.json model missing 'vocab'");

    int maxId = -1;
    for (auto it = vocab_node.begin(); it != vocab_node.end(); ++it) {
        cv::FileNode entry = *it;
        std::string piece = entry.name();
        int id = (int)entry;
        if (id > maxId) maxId = id;
        gemma.pieceToId[piece] = id;
    }

    gemma.idToPiece.resize(maxId + 1);
    for (const auto& kv : gemma.pieceToId)
        gemma.idToPiece[kv.second] = kv.first;

    cv::FileNode merges_node = model_node["merges"];
    if (!merges_node.empty()) {
        uint32_t rank = 0;
        for (auto it = merges_node.begin(); it != merges_node.end(); ++it) {
            cv::FileNode entry = *it;
            if (static_cast<int>(entry.size()) != 2) {
                ++rank;
                continue;
            }
            std::string a, b;
            entry[0] >> a;
            entry[1] >> b;
            gemma.addMerge(a, b, rank);
            ++rank;
        }
    }

    std::unordered_set<std::string> special;
    cv::FileNode added = fs["added_tokens"];
    if (!added.empty()) {
        for (auto it = added.begin(); it != added.end(); ++it) {
            cv::FileNode t = *it;
            int id = -1;         t["id"]      >> id;
            std::string content; t["content"] >> content;
            bool is_special = false; t["special"] >> is_special;
            if (id >= 0 && !content.empty()) {
                gemma.specialToId[content] = id;
                gemma.idToSpecial[id]      = content;
                // All added tokens bypass BPE (matching HuggingFace behavior),
                // not just those marked special.
                special.insert(content);
                if (outSpecial) outSpecial->insert(content);
            }
        }
    }

    return makePtr<GemmaBpeTokenizerImpl>(std::move(gemma), std::move(special));
}

static Ptr<SentencePieceTokenizerImpl> buildSentencePieceFromJson(
        const std::string& json_path,
        std::unordered_set<std::string>* outSpecial = nullptr) {

    cv::FileStorage fs(json_path, cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);
    if (!fs.isOpened())
        CV_Error(cv::Error::StsError, "Failed to open tokenizer.json: " + json_path);

    cv::FileNode model_node = fs["model"];
    CV_CheckFalse(model_node.empty(), "tokenizer.json missing 'model'");

    std::string model_type;
    model_node["type"] >> model_type;
    if (model_type != "BPE")
        CV_Error(cv::Error::StsError,
            "Expected BPE model in tokenizer.json for SentencePiece, got: " + model_type);

    CoreGemmaBPE gemma;

    cv::FileNode vocab_node = model_node["vocab"];
    CV_CheckFalse(vocab_node.empty(), "tokenizer.json model missing 'vocab'");

    int maxId = -1;
    for (auto it = vocab_node.begin(); it != vocab_node.end(); ++it) {
        cv::FileNode entry = *it;
        std::string piece = entry.name();
        int id = (int)entry;
        if (id > maxId) maxId = id;
        gemma.pieceToId[piece] = id;
    }

    gemma.idToPiece.resize(maxId + 1);
    for (const auto& kv : gemma.pieceToId)
        gemma.idToPiece[kv.second] = kv.first;

    cv::FileNode merges_node = model_node["merges"];
    if (!merges_node.empty()) {
        uint32_t rank = 0;
        for (auto it = merges_node.begin(); it != merges_node.end(); ++it) {
            cv::FileNode entry = *it;
            std::string a, b;
            if (entry.isString()) {
                // SentencePiece format: "a b" (single string with space separator)
                std::string merge_str;
                entry >> merge_str;
                size_t sp = merge_str.find(' ');
                if (sp == std::string::npos) {
                    ++rank;
                    continue;
                }
                a = merge_str.substr(0, sp);
                b = merge_str.substr(sp + 1);
            } else if (entry.size() == 2) {
                // Array format: ["a", "b"]
                entry[0] >> a;
                entry[1] >> b;
            } else {
                ++rank;
                continue;
            }
            gemma.addMerge(a, b, rank);
            ++rank;
        }
    }

    std::unordered_set<std::string> special;
    int bosId = -1;
    cv::FileNode added = fs["added_tokens"];
    if (!added.empty()) {
        for (auto it = added.begin(); it != added.end(); ++it) {
            cv::FileNode t = *it;
            int id = -1;         t["id"]      >> id;
            std::string content; t["content"] >> content;
            if (id >= 0 && !content.empty()) {
                gemma.specialToId[content] = id;
                gemma.idToSpecial[id]      = content;
                special.insert(content);
                if (outSpecial) outSpecial->insert(content);
                // Capture <bos> token id for SentencePiece auto-prepend
                if (content == "<bos>") bosId = id;
            }
        }
    }

    return makePtr<SentencePieceTokenizerImpl>(std::move(gemma), std::move(special), bosId);
}

static void registerDefaultTokenizers() {
    auto& reg = tokenizerRegistry();
    if (reg.find("BPE") == reg.end()) {
        reg["BPE"] = [](const FileStorage& cfg, const std::string& dir) -> Ptr<Tokenizer::Impl> {
            std::string model_type;
            cfg["model_type"] >> model_type;
            std::string tok_json = dir + "tokenizer.json";

            CoreBPE core;
            std::unordered_set<std::string> special;
            if (model_type == "gpt2" || model_type == "gpt4") {
                core = buildTokenizerFromJson(model_type, tok_json);
            } else if (model_type == "qwen2" || model_type == "qwen2.5") {
                core = buildTokenizerFromJson(model_type, tok_json, &special);
            } else {
                CV_Error(cv::Error::StsError, "Unsupported model_type for BPE: " + model_type);
            }
            return makePtr<BpeTokenizerImpl>(std::move(core), std::move(special));
        };
    }

    if (reg.find("Gemma") == reg.end()) {
        reg["Gemma"] = [](const FileStorage& /*cfg*/, const std::string& dir) -> Ptr<Tokenizer::Impl> {
            std::string tok_json = dir + "tokenizer.json";
            std::unordered_set<std::string> special;
            return buildGemmaFromJson(tok_json, &special);
        };
    }

    if (reg.find("SentencePiece") == reg.end()) {
        reg["SentencePiece"] = [](const FileStorage& /*cfg*/, const std::string& dir) -> Ptr<Tokenizer::Impl> {
            std::string tok_json = dir + "tokenizer.json";
            std::unordered_set<std::string> special;
            return buildSentencePieceFromJson(tok_json, &special);
        };
    }
}

Tokenizer::Tokenizer() : impl_(nullptr) {}

std::vector<int> Tokenizer::encode(const std::string& text) {
    if (!impl_) CV_Error(cv::Error::StsError, "Tokenizer impl null");
    return impl_->encode(text);
}

std::string Tokenizer::decode(const std::vector<int>& tokens) {
    if (!impl_) CV_Error(cv::Error::StsError, "Tokenizer impl null");
    return impl_->decode(tokens);
};

CoreBPE buildTokenizerFromJson(const std::string& model_type, const std::string& json_path,
                          std::unordered_set<std::string>* outSpecial) {
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
    } else if (model_type == "qwen2" || model_type == "qwen2.5") {
        pattern = QWEN2_5;
    } else {
        CV_Error(cv::Error::StsError,
            "Unsupported model_type: " + model_type + " (expected gpt2/r50k_base, gpt4/cl100k_base, or qwen2/qwen2.5)");
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
                if (outSpecial) outSpecial->insert(content);
            }
        }
    }

    return CoreBPE(std::move(mergeableRanks), std::move(specialTokens), pattern);
}

Tokenizer Tokenizer::load(const std::string& model_config) {
    cv::FileStorage cfg(model_config, cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);
    if (!cfg.isOpened())
        CV_Error(cv::Error::StsError, "Could not open config.json: " + model_config);

    std::string dir = model_config;
    size_t pos = dir.find_last_of("/\\");
    dir = (pos == std::string::npos) ? std::string() : dir.substr(0, pos + 1);

    std::string methodType = "BPE";
    if (!cfg["method"].empty())
        cfg["method"] >> methodType;

    registerDefaultTokenizers();
    auto& reg = tokenizerRegistry();
    auto it = reg.find(methodType);
    if (it == reg.end())
        CV_Error(cv::Error::StsError,
            "Unsupported tokenizer method: '" + methodType + "'. Supported: BPE, Gemma, SentencePiece");

    Tokenizer tok;
    tok.impl_ = it->second(cfg, dir);
    return tok;
}
CV__DNN_INLINE_NS_END
}}
