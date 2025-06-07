#pragma once
#include <opencv2/core.hpp>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <queue>
#include <string>
#include <cstdint>

// #include <opencv2/modules/dnn/tokenizers/src/utils/types.hpp>     // Pair, Word, AddedToken, etc.
// #include <opencv2/modules/dnn/tokenizers/src/models/model.hpp>    // abstract Trainer / Model base
#include "bpe.hpp"

#ifndef __OPENCV_DNN_TOKENIZERS_MODELS_TRAINER.HPP__
#define __OPENCV_DNN_TOKENIZERS_MODELS_TRAINER.HPP__

namespace cv { namespace dnn { namespace tokenizer {

class BpeTrainerBuilder {
public:
    static BpeTrainerBuilder create();

    BpeTrainerBuilder& minFrequency(std::uint64_t f);
    BpeTrainerBuilder& vocabSize(std::size_t s);
    BpeTrainerBuilder& showProgress(bool on);
    BpeTrainerBuilder& specialTokens(std::vector<AdddedToken> v);
    BpeTrainerBuilder& limitAlphabet(std::size_t n);
    BpeTrainerBuilder& initialAlphabet(std::unordered_set<char> s);
    BpeTrainerBuilder& continuingSubwordPrefix(const std::string& p);
    BpeTrainerBuilder& endOfWordSuffix(const std::string& s);
    BpeTrainerBuilder& maxTokenLength(std::optional<std::size_t> m);

    class BpeTrainer build() const;

private: 
    struct Config {
        std::uint64_t min_frequency  = 0;
        std::size_t vocab_size = 30'000;
        bool show_progress = true;
        std::vector<AddedToken> special_tokens;
        std::optional<std::size_t> limit_alphabet;
        std::unordered_set<char> initial_alphabet;
        std::optional<std::string> cont_prefix;
        std::optional<std::string> end_suffix;
        std::optional<std::size_t> max_token_length;
    } cfg_;
    BpeTrainerBuilder() = default();
};

// BpeTrainer 
class CV_EXPORTS_W BpeTrainer : public Trainer {
public:
    friend class BpeTrainerBuilder;

    using WordCounts = std::unordered_map<std::string, std::uint64_t>;

    static BpeTrainer createDefault();

    BpeTrainer(std::uint64_t minFreq, std::size_t vocabSize);
    static BpeTrainerBuilder builder();

    // Trainer interface 
    using Model = BPE;
    std::vector<AddedToken> train(Model& model) override;
    bool shouldShowProgress() const override;
    void feed(FeedIter&& src, FeedProcess&& fn) override;
    
    std::uint64_t minFrequency() const { return min_frequency_; }
    std::size_t vocabSize() const { return vocab_size_; }

private:
    struct Merge { 
        Pair pair;
        std::uint64_t count;
        std::unordered_set<std::size_t> pos;
    };
    struct MergeCmp {
        bool operator()(const Merge&, const Merge&) const;
    }
    
    // [TODO] fix the progress bar
    struct Progress {
        void reset(std::uint64_t = 0, const char*="") {}
        void inc(){}
        void finish(){}
    };

    Progress setUpProgress() const;
    void updateProgress(Progress&, std::size_t, const char*);
    void finalizeProgress(Progress&, std::size_t);
    void addSpecialTokens(Vocab& w2id, std::vector<std::string>& id2w) const;
    void computeAlphabet(const WordCounts&, Vocab&, std::vector<std::string>&) const;
    std::pair<std::vector<Word>, std::vector<std::uint64_t>> 
        tokenizeWords(const WordCounts&, Vocab&, std::vector<std::string>&, Progress& p) const;
    std::pair<std::unordered_map<Pair, int>, std::unordered_map<Pair, std::unordered_set<std::size_t>>> 
        countPairs(const std::vector<Word>&, const std::vector<std::uint64_t>&, Progress&) const;

    std::vector<AddedToken> doTrain(const WordCounts&, BPE&);

    std::uint64_t min_frequency_ = 0;
    std::size_t  vocab_size_ = 30'000;
    bool show_progress_ = true;
    std::vector<AddedToken>       special_tokens_;
    std::optional<std::size_t>    limit_alphabet_;
    std::unordered_set<char>      initial_alphabet_;
    std::optional<std::string>    cont_prefix_;
    std::optional<std::string>    end_suffix_;
    std::optional<std::size_t>    max_token_length_;

    WordCounts words_;
};

}}} // namespace cv::dnn::tokenizer

#endif