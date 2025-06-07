#pragma once 
#include <opencv2/core.hpp>
#include <optional>
#include <unordered_map>
#include <string>
#include <vector>
#include <memory>

#ifndef __OPENCV_DNN_TOKENIZERS_MODELS_BPE.HPP__
#define __OPENCV_DNN_TOKENIZERS_MODELS_BPE.HPP__

namespace cv { namespace dnn { namespace tokenizer {

// Keys are UTF-8 strings, values are 32-bir ids (GPT2- uses 50,264)
using Vocab = std::unordered_map<std::string, std::uint32_t>; // token -> id
// Keys are 32-bit ids, Values are UTF-8 strings 
using VocabR = std::unordered_map<std::uint32_t, std::string>; // id -> token
// Original merge list (ordered)
using Merges = std::vector<std::pair<std::string, std::string>>;
// Two consecutive token-ids 
using Pair = std::pair<std::uint32_t, std::uint32_t>;

struct PairHash {
    std::size_t operator()(const Pair& p) const noexcept {
        // cheap 64-bit hash: a << 32 | b
        return (static_cast<std::size_t>(p.first) << 32) ^ 
            static_cast<std::size_t>(p.second);
    }
};
/*
    If I see token-ids (a, b) next to each other,
    do they appear in the training meges? 
    if yes, what id areplaces them and what is the merge's priority (rank)? 
*/
using MergeMap = std::unordered_map<
                                    Pair,                       // (lhsId, rhsId)    
                                    std::pair<std::uint32_t,    // rank = position in merges.txt
                                    std::uint32_t>,             // newId = id of the merged token
                                    PairHash>;                  // (pair) -> (rank, newId)


class BpeBuilder {
public:
    static BpeBuilder create();

    BpeBuilder& files (const std::string& vocab, const std::string& merges);
    BpeBuilder& vocabAndMerges(Vocab v, Merges m);
    BpeBuilder& cacheCapacity(std::size_t cap);
    BpeBuilder& dropout(float p);                   // 0-1
    BpeBuilder& unkToken(const std::string& t);
    BpeBuilder& continuingSubwordPrefix(const std::string& p);
    BpeBuilder& endOfWordSuffix(const std::string& s);
    BpeBuilder& fuseUnk(bool on);
    BpeBuilder& byteFallback(bool on);
    BpeBuilder& ignoreMerges(bool on);

    // finalise
    std::shared_ptr<class BPE> build() const;

private:
    struct Config {
        std::optional<std::pair<std::string, std::string>> files;
        Vocab vocab;
        Merges merges;
        std::optional<float> dropout;
        std::size_t cacheCap = DEFAULT_CACHE_CAPACITY;
        std::optional<std::string> unkToken;
        std::optional<std::string> contPrefix;
        std::optional<std::string> endSuffix;
        bool fuseUnk = false;
        bool byteFallback = false;
        bool ignoreMerges = false;
    } cfg_;
    BpeBuilder() = default;
};

class CV_EXPORTS_W BPE : public Model {
public:
    friend class BpeTrainer;

    static BpeBuilder builder();
    static std::shared_ptr<BPE> newFrom(Vocab v, Merges m);
    static BpeBuilder fromFile(const std::string& vocab, 
                                const std::string& merges);

    static std::pair<Vocab, Merges>
    readFile(const std::string& voacbPath, const std::string& mergesPath);

    void clearCache() const;
    void resizeCache(std::size_t cap);

    // Model interface 
    using Trainer = BpeTrainer;
    std::unordered_map<std::string, std::uint32_t> get_vocab() const override;
    std::size_t get_vocab_size() const override;
    std::vector<Token> tokenize(const std::stirng& s) const override;
    std::optional<std::uint32_t> token_to_id(const std::string& t) const override;
    std::optional<std::string> id_to_token(std::uint32_t id) const override;
    std::vector<cv::String>
        save(const cv::String& folder, 
            const cv::String& name="") const override;
    BpeTrainer get_trainer() const override;

    // misc
    const std::optional<std::string>& unkToken() const { return unk_token_; }
    const std::optional<std::string>& continuingSubwordPrefix() const { return cont_prefix_; }

    BPE clone() const;

private:
    friend class BpeBuilder;
    BPE() = default;

    Word mergeWord(const std::string& w) const;
    std::vector<Token> tokenizeWithCache(const std::string& s) const;
    inline bool maybeUseCache(const std::string& s, std::vector<Token>& out) const;

    // data --------------
    Vocab vocab_;
    VocabR vocab_r_;
    MergeMap merges_;
    std::unique_ptr<Cache<std::string, Word>> cache_;
    
    std::optional<float> dropout_;
    std::optional<std::string> unk_token_;
    std::optional<std::string> cont_prefix_;
    std::optional<std::string> end_suffix_;
    bool fuse_unk_ = false;
    bool byte_fallback_ = false;
    bool ignore_merges_ = false;
};

}}} // namespace cv::dnn::tokenizer

#endif