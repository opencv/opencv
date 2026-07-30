// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <opencv2/dnn/dnn.hpp>
#include "utils.hpp"
#include "unicode.hpp"
#include "core_bpe.hpp"
#include "core_gemma.hpp"
#include "core_unigram.hpp"

#include <cctype>
#include <fstream>
#include <functional>
#include <sstream>
#include <unordered_set>

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

// Registry of implementations (method -> methodImpl)
using ImplRegestry = std::function<Ptr<Tokenizer::Impl>(const FileStorage& cfg, const std::string& dir)>;

static std::unordered_map<std::string, ImplRegestry>& tokenizerRegistry() {
    static std::unordered_map<std::string, ImplRegestry> reg;
    return reg;
}

CoreBPE buildTokenizerFromJson(const std::string& json_path,
                          std::unordered_set<std::string>* outSpecial = nullptr);

// Opens a tokenizer.json file for FileStorage/JSON parsing while stripping
// out any oversized field values (currently just "precompiled_charsmap")
// that would otherwise overflow FileStorage's JSON parser. Pass a non-null
// out_charsmap to receive the stripped "precompiled_charsmap" value, if any.
static cv::FileStorage openTokenizerJson(const std::string& json_path,
    std::string* out_charsmap = nullptr) {
    std::ifstream in(json_path, std::ios::binary);
    if (!in.is_open())
        CV_Error(cv::Error::StsError, "Failed to open tokenizer.json: " + json_path);

    std::ostringstream ss;
    ss << in.rdbuf();
    std::string text = ss.str();

    std::string charsmap = extractAndStripLongStringField(text, "precompiled_charsmap");
    if (out_charsmap)
        *out_charsmap = std::move(charsmap);

    cv::FileStorage fs(text, cv::FileStorage::MEMORY | cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);
    if (!fs.isOpened())
        CV_Error(cv::Error::StsError, "Failed to parse tokenizer.json: " + json_path);
    return fs;
}

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

// SentencePiece Unigram (T5-style): CoreUnigram already appends the trailing
// eos (via post_processor's TemplateProcessing) inside encode() and strips
// special tokens inside decode(), so this wrapper is a thin pass-through --
// unlike SentencePieceTokenizerImpl above, it must NOT re-add or re-strip
// anything itself.
struct UnigramTokenizerImpl : public Tokenizer::Impl {
    CoreUnigram model;
    std::unordered_set<std::string> allowedSpecial;

    explicit UnigramTokenizerImpl(CoreUnigram m,
                                  std::unordered_set<std::string> special = {})
        : model(std::move(m)), allowedSpecial(std::move(special)) {}

    std::vector<int> encode(const std::string& text) override {
        return model.encode(text, allowedSpecial);
    }

    std::string decode(const std::vector<int>& tokens) override {
        return model.decode(tokens);
    }
};

static std::string expandCaseInsensitiveGroups(const std::string& in) {
    std::string out;
    out.reserve(in.size());
    size_t i = 0;
    while (i < in.size()) {
        if (in.compare(i, 4, "(?i:") == 0) {
            size_t j = i + 4;
            int depth = 1;
            while (j < in.size() && depth > 0) {
                if (in[j] == '(') depth++;
                else if (in[j] == ')') depth--;
                if (depth > 0) j++;
            }
            std::string inner = in.substr(i + 4, j - (i + 4));
            out += "(?:";
            for (char c : inner) {
                if (std::isalpha(static_cast<unsigned char>(c))) {
                    out += '[';
                    out += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
                    out += static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
                    out += ']';
                } else {
                    out += c;
                }
            }
            out += ")";
            i = (j < in.size()) ? j + 1 : j;
        } else {
            out += in[i++];
        }
    }
    return out;
}

static std::string stripPossessiveQuantifiers(const std::string& in) {
    std::string out;
    out.reserve(in.size());
    for (char c : in) {
        if (c == '+' && !out.empty()) {
            char prev = out.back();
            if (prev == '+' || prev == '*' || prev == '?') {
                continue;
            }
            if (prev == '}') {
                // '}' can either close a {m,n} repetition quantifier (in which
                // case a following '+' is a possessive quantifier that should
                // be stripped) or close a \p{...}/\P{...} Unicode property
                // escape (in which case the following '+' is a normal,
                // legitimate quantifier applied to the whole \p{...} atom and
                // must NOT be stripped). Walk back to the matching '{' to
                // tell these two cases apart.
                int depth = 1;
                size_t j = out.size() - 1;
                while (j > 0 && depth > 0) {
                    --j;
                    if (out[j] == '}') ++depth;
                    else if (out[j] == '{') --depth;
                }
                bool isPropertyEscape = (depth == 0 && out[j] == '{' &&
                    j >= 2 && out[j - 1] == 'p' && out[j - 2] == '\\');
                if (!isPropertyEscape) {
                    isPropertyEscape = (depth == 0 && out[j] == '{' &&
                        j >= 2 && out[j - 1] == 'P' && out[j - 2] == '\\');
                }
                if (depth == 0 && out[j] == '{' && !isPropertyEscape) {
                    continue;
                }
            }
        }
        out.push_back(c);
    }
    return out;
}

static std::string adaptHfPreTokenizerRegex(const std::string& raw) {
    return stripPossessiveQuantifiers(expandCaseInsensitiveGroups(raw));
}

static bool findEmbeddedSplitRegex(const cv::FileNode& preTok, std::string& outRegex) {
    if (preTok.empty()) return false;
    std::string type;
    preTok["type"] >> type;
    if (type == "Sequence") {
        cv::FileNode list = preTok["pretokenizers"];
        for (auto it = list.begin(); it != list.end(); ++it) {
            cv::FileNode child = *it;
            cv::FileNode regexNode = child["pattern"]["Regex"];
            if (!regexNode.empty() && regexNode.isString()) {
                regexNode >> outRegex;
                return true;
            }
        }
        return false;
    }
    cv::FileNode regexNode = preTok["pattern"]["Regex"];
    if (!regexNode.empty() && regexNode.isString()) {
        regexNode >> outRegex;
        return true;
    }
    return false;
}

static std::string detectSplitPattern(const cv::FileStorage& fs) {
    std::string raw;
    if (findEmbeddedSplitRegex(fs["pre_tokenizer"], raw))
        return adaptHfPreTokenizerRegex(raw);
    return R50K_UTF8;
}

static Ptr<Tokenizer::Impl> buildGemmaFamilyFromJson(
        const std::string& json_path,
        std::unordered_set<std::string>* outSpecial = nullptr) {

    cv::FileStorage fs = openTokenizerJson(json_path);

    cv::FileNode model_node = fs["model"];
    CV_CheckFalse(model_node.empty(), "tokenizer.json missing 'model'");

    std::string model_type;
    model_node["type"] >> model_type;
    if (!model_type.empty() && model_type != "BPE")
        CV_Error(cv::Error::StsError,
            "Expected a byte-fallback BPE model in tokenizer.json, got: " + model_type);

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

    bool mergesAreStringFormat = false;
    cv::FileNode merges_node = model_node["merges"];
    if (!merges_node.empty()) {
        cv::FileNode first_entry = *merges_node.begin();
        mergesAreStringFormat = first_entry.isString();

        uint32_t rank = 0;
        for (auto it = merges_node.begin(); it != merges_node.end(); ++it) {
            cv::FileNode entry = *it;
            std::string a, b;
            if (entry.isString()) {
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
                if (content == "<bos>") bosId = id;
            }
        }
    }

    int bosTokenId = (mergesAreStringFormat && bosId >= 0) ? bosId : -1;
    return makePtr<SentencePieceTokenizerImpl>(std::move(gemma), std::move(special), bosTokenId);
}

static Ptr<Tokenizer::Impl> buildUnigramTokenizerImpl(
        const std::string& json_path,
        std::unordered_set<std::string>* outSpecial = nullptr) {

    std::string charsmap_b64;
    cv::FileStorage fs = openTokenizerJson(json_path, &charsmap_b64);

    cv::FileNode model_node = fs["model"];
    CV_CheckFalse(model_node.empty(), "tokenizer.json missing 'model'");

    std::string model_type;
    model_node["type"] >> model_type;
    if (!model_type.empty() && model_type != "Unigram")
        CV_Error(cv::Error::StsError,
            "Expected a Unigram model in tokenizer.json, got: " + model_type);

    cv::FileNode vocab_node = model_node["vocab"];
    CV_CheckFalse(vocab_node.empty(), "tokenizer.json model missing 'vocab'");

    CoreUnigram unigram;
    unigram.idToPiece.reserve(vocab_node.size());
    unigram.idToScore.reserve(vocab_node.size());
    int id = 0;
    for (auto it = vocab_node.begin(); it != vocab_node.end(); ++it, ++id) {
        cv::FileNode entry = *it;
        std::string piece;
        double score = 0.0;
        if (entry.size() >= 2) {
            entry[0] >> piece;
            entry[1] >> score;
        }
        unigram.idToPiece.push_back(piece);
        unigram.idToScore.push_back((float)score);
        unigram.pieceToId[piece] = id;
    }

    int unkId = -1;
    model_node["unk_id"] >> unkId;
    unigram.unkId = unkId;

    unigram.normalizer = buildUnigramPrecompiledNormalizer(charsmap_b64);

    std::unordered_set<std::string> special;
    int eosId = -1;
    cv::FileNode added = fs["added_tokens"];
    if (!added.empty()) {
        for (auto it = added.begin(); it != added.end(); ++it) {
            cv::FileNode t = *it;
            bool isSpecial = false; t["special"] >> isSpecial;
            int tid = -1;           t["id"]      >> tid;
            std::string content;    t["content"] >> content;
            if (isSpecial && tid >= 0 && !content.empty()) {
                unigram.specialToId[content] = tid;
                unigram.idToSpecial[tid]     = content;
                special.insert(content);
                if (outSpecial) outSpecial->insert(content);
                if (content == "</s>") eosId = tid;
            }
        }
    }
    unigram.eosId = eosId;

    unigram.finalize();

    return makePtr<UnigramTokenizerImpl>(std::move(unigram), std::move(special));
}

static Ptr<Tokenizer::Impl> buildBPETokenizerImpl(const std::string& dir) {
    std::string tok_json = dir + "tokenizer.json";
    std::unordered_set<std::string> special;
    CoreBPE core = buildTokenizerFromJson(tok_json, &special);
    return makePtr<BpeTokenizerImpl>(std::move(core), std::move(special));
}

static Ptr<Tokenizer::Impl> buildFromTokenizerDir(const std::string& dir) {
    std::string tok_json = dir + "tokenizer.json";
    cv::FileStorage fs = openTokenizerJson(tok_json);

    cv::FileNode model = fs["model"];
    if (model.empty())
        CV_Error(cv::Error::StsError,
            "tokenizer.json has no 'model' field; raw rank-table tokenizers are not "
            "supported by this loader: " + tok_json);

    std::string model_type;
    model["type"] >> model_type;

    if (model_type == "Unigram")
        return buildUnigramTokenizerImpl(tok_json);

    if (!model_type.empty() && model_type != "BPE")
        CV_Error(cv::Error::StsError,
            "Unsupported tokenizer model type '" + model_type + "' in " + tok_json +
            " (only BPE-family and Unigram models are currently supported)");

    if (model["merges"].empty())
        CV_Error(cv::Error::StsError,
            "tokenizer.json model has no 'merges' table in " + tok_json +
            " (only merge-based BPE models are currently supported)");

    bool byteFallback = false;
    model["byte_fallback"] >> byteFallback;

    if (byteFallback)
        return buildGemmaFamilyFromJson(tok_json);
    return buildBPETokenizerImpl(dir);
}

static void registerDefaultTokenizers() {
    auto& reg = tokenizerRegistry();
    if (reg.find("BPE") == reg.end()) {
        reg["BPE"] = [](const FileStorage& /*cfg*/, const std::string& dir) -> Ptr<Tokenizer::Impl> {
            return buildFromTokenizerDir(dir);
        };
    }

    if (reg.find("Gemma") == reg.end()) {
        reg["Gemma"] = [](const FileStorage& /*cfg*/, const std::string& dir) -> Ptr<Tokenizer::Impl> {
            return buildFromTokenizerDir(dir);
        };
    }

    if (reg.find("SentencePiece") == reg.end()) {
        reg["SentencePiece"] = [](const FileStorage& /*cfg*/, const std::string& dir) -> Ptr<Tokenizer::Impl> {
            return buildFromTokenizerDir(dir);
        };
    }

    if (reg.find("Unigram") == reg.end()) {
        reg["Unigram"] = [](const FileStorage& /*cfg*/, const std::string& dir) -> Ptr<Tokenizer::Impl> {
            return buildFromTokenizerDir(dir);
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

CoreBPE buildTokenizerFromJson(const std::string& json_path,
                          std::unordered_set<std::string>* outSpecial) {
    cv::FileStorage fs = openTokenizerJson(json_path);

    cv::FileNode model = fs["model"];
    CV_CheckFalse(model.empty(), "tokenizer.json missing 'model'");
    cv::FileNode vocab = model["vocab"];
    CV_CheckFalse(vocab.empty(), "tokenizer.json missing model.vocab");

    std::string model_type;
    model["type"] >> model_type;
    if (!model_type.empty() && model_type != "BPE")
        CV_Error(cv::Error::StsError,
            "Expected a BPE model in tokenizer.json, got: " + model_type);

    std::string pattern = detectSplitPattern(fs);

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

    std::unordered_set<std::string> skip_tokens;
    FileNode added_peek = fs["added_tokens"];
    if (!added_peek.empty()) {
        for (auto it = added_peek.begin(); it != added_peek.end(); ++it) {
            cv::FileNode t = *it;
            bool is_special = false; t["special"]  >> is_special;
            std::string content;     t["content"]  >> content;
            if (is_special && !content.empty())
                skip_tokens.insert(content);
        }
    }

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
            "Unsupported tokenizer method: '" + methodType + "'. Supported: BPE, Gemma, SentencePiece, Unigram");

    Tokenizer tok;
    tok.impl_ = it->second(cfg, dir);
    return tok;
}

CV__DNN_INLINE_NS_END
}}
