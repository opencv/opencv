#include "../../include/opencv2/dnn/tokenizer.hpp"
#include "utils.hpp"

namespace cv { namespace dnn { namespace tokenizer {

Tokenizer::Tokenizer() : coreBPE_(nullptr) {}

Tokenizer::Tokenizer(std::shared_ptr<CoreBPE> core)
    : coreBPE_(std::move(core)) {
}

std::vector<int> Tokenizer::encode(const std::string& text) {
    // We dont allow any of the special tokens for now so we set allow
    // to empty
    std::vector<uint32_t> tok = coreBPE_->encode(text, {}).first;
    return std::vector<int>(tok.begin(), tok.end());
}

std::string Tokenizer::decode(const std::vector<int>& tokens) { 
        std::vector<uint32_t> tokens32(tokens.begin(), tokens.end());
        auto opt_bytes = coreBPE_->decodeBytes(tokens32); 
        if (!opt_bytes) throw std::runtime_error("Invalid decode.");
        const ByteVec& bytes = *opt_bytes;

        // Convert bytes to std::string (UTF-8)
        std::string result(reinterpret_cast<const char*>(bytes.data()), bytes.size());
        return result;
};

CoreBPE getEncodingForGPT2FromJSON(const std::string &name, const std::string& json_path) {
    cv::FileStorage fs(json_path, cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);
    if (!fs.isOpened()) {
        CV_Error(cv::Error::StsError, "Failed to open tokenizer.json: " + json_path);
    }

    cv::FileNode model = fs["model"];
    FileNode vocab = model["vocab"];
    if (vocab.empty()) {
        CV_Error(Error::StsError, "tokenizer.json missing model.vocab");
    }

    auto token_to_bytes = [&](const std::string& token_utf8) -> std::vector<uint8_t> {
        ByteVec out;
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
        mergeableRanks.emplace(token_to_bytes(key), (Rank)id);
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
        ByteVec key{ (uint8_t)b };
        if (mergeableRanks.find(key) == mergeableRanks.end()) {
            std::ostringstream oss;
            oss << "Missing singleton byte token 0x" << std::hex << b;
            CV_Error(cv::Error::StsError, oss.str());
        }
    }

    std::unordered_map<std::string, Rank> specialTokens;
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
                specialTokens.emplace(content, (Rank)id);
            }
        }
    }
    
    return CoreBPE(std::move(mergeableRanks), std::move(specialTokens), R50K_UTF8);
}

CoreBPE getEncodingForCl100k_baseFromJSON_FS(const std::string &name,
                                                         const std::string &json_path)
{
    if (name != "cl100k_base")
        CV_Error(cv::Error::StsError, "Wrong model name. This model is cl100k_base");

    cv::FileStorage fs(json_path, cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);
    if (!fs.isOpened())
        CV_Error(cv::Error::StsError, "Failed to open tokenizer.json: " + json_path);

    cv::FileNode model = fs["model"];
    if (model.empty()) CV_Error(cv::Error::StsError, "tokenizer.json missing 'model'");

    cv::FileNode vocab = model["vocab"];
    if (vocab.empty()) CV_Error(cv::Error::StsError, "tokenizer.json missing model.vocab");

    ByteVecRankMap mergeableRanks;
    mergeableRanks.reserve((size_t)vocab.size());
    int max_id = -1;

    auto token_to_bytes = [&](const std::string& token_utf8) -> std::vector<uint8_t> {
        ByteVec out;
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
        mergeableRanks.emplace(token_to_bytes(token), (Rank)id);
        if (id > max_id) max_id = id;
    }

    std::unordered_map<std::string, Rank> specialTokens;
    cv::FileNode added = fs["added_tokens"];
    if (!added.empty()) {
        for (auto it = added.begin(); it != added.end(); ++it) {
            cv::FileNode t = *it;
            bool special = false; t["special"] >> special;
            int id = -1;          t["id"]      >> id;
            std::string content;  t["content"] >> content;
            if (special && id >= 0 && !content.empty()) {
                specialTokens.emplace(content, (Rank)id);
                if (id > max_id) max_id = id;
            }
        }
    }

    for (int b = 0; b < 256; ++b) {
        ByteVec key{ (uint8_t)b };
        if (mergeableRanks.find(key) == mergeableRanks.end()) {
            std::ostringstream oss; oss << "Missing singleton byte token 0x" << std::hex << b;
            CV_Error(cv::Error::StsError, oss.str());
        }
    }
    return CoreBPE(std::move(mergeableRanks), std::move(specialTokens), CL100K_BASE);
}

Tokenizer Tokenizer::load(const std::string& model_dir) {
    const std::string cfg_path = model_dir + "config.json";
    cv::FileStorage cfg(cfg_path, cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);
    if (!cfg.isOpened()) 
        CV_Error(cv::Error::StsError, "Could not open config.json at: " + cfg_path);

    std::string model_type;
    cfg["model_type"] >> model_type;
    const std::string tok_json = model_dir + "tokenizer.json";
    Tokenizer tok;
    if (model_type == "gpt2") {
        tok.coreBPE_ = std::make_shared<CoreBPE>(getEncodingForGPT2FromJSON("gpt2", tok_json));
    } else if (model_type == "gpt4") {
        tok.coreBPE_ = std::make_shared<CoreBPE>(getEncodingForCl100k_baseFromJSON_FS("cl100k_base", tok_json));
    } else {
        CV_Error(cv::Error::StsError, "Unsupported model_type in config.json: " + model_type);
    }
    return tok;
}

}}}