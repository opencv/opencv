// #include "opencv2/modules/dnn/tokenizers/src/utils/utils.hpp"   // Error
// #include "opencv2/modules/dnn/tokenizers/src/utils/word.hpp"
// #include "opencv2/modules/dnn/tokenizers/src/utils/token.hpp"

#include "trainer.hpp"

#include <algorithm>
#include <limits>
#include <tuple>

namespace cv { namespace dnn { namespace tokenizer {

BpeTrainerBuilder BpeTrainerBuilder::create() { return {}; }

#define BUILDER_SETTER(func, member, type)                        \
    BpeTrainerBuilder& BpeTrainerBuilder::func(type v) {          \
        cfg_.member = std::move(v);                               \
        return *this;                                             \
    }

BUILDER_SETTER(minFrequency , min_frequency , std::uint64_t)
BUILDER_SETTER(vocabSize, vocab_size, std::size_t)
BUILDER_SETTER(showProgress, show_progress, bool)
BUILDER_SETTER(specialTokens, special_tokens, std::vector<AddedToken>)
BUILDER_SETTER(limitAlphabet, limit_alphabet, std::size_t)
BUILDER_SETTER(initialAlphabet, initial_alphabet, std::unordered_set<char>)
BUILDER_SETTER(continuingSubwordPrefix, cont_prefix, const std::string&)
BUILDER_SETTER(endOfWordSuffix, end_suffix, const std::string&)
BUILDER_SETTER(maxTokenLength, max_token_length, std::optional<std::size_t>)
#undef BUILDER_SETTER

class BpeTrainer BpeTrainerBuilder::build() const {
    BpeTrainer t(cfg_.min_frequency, cfg_.vocab_size);
    t.show_progress_ = cfg_.show_progress;
    t.special_tokens_ = cfg_.special_tokens;
    t.limit_alphabet_ = cfg_.limit_alphabet;
    t.initial_alphabet_ = cfg_.initial_alphabet;
    t.cont_prefix_ = cfg_.cont_prefix;
    t.end_suffix_ = cfg_.end_suffix;
    t.max_token_length_ = cfg_.max_token_length;
    return t;
};

// Trainer core
BpeTrainer BpeTrainer::createDefault() {
   // [TODO]
    // return buidler().build(); 
}

BpeTrainer::BpeTrainer(std::uint64_t mf, std::size_t vs) 
    : min_frequency_(mf), vocab_size_(vs) {}

BpeTrainerBuilder BpeTrainer::builder() { return BpeTrainerBuilder::create(); }

// TODO: progress bar 
BpeTrainer::Progress BpeTrainer::setUpProgress() const {
    return Progress{};
}

void BpeTrainer::updateProgress(Progress& p, std::size_t, const char*) { p.reset(); }
void BpeTrainer::finalizeProgress(Progress& p, std::size_t) {p.finish(); }

bool BpeTrainer::MergeCmp::operator()(const Merge& a, const Merge& b) const  {
    if (a.count != b.count) return a.count < b.count;   // max-heap
    return a.pair > b.pair;                              // ascending pair
}

// core train 
std::vector<AddedToken> BpeTrainer::train(Model& model) {
    return doTrain(words_, model);
}

bool BpeTrainer::shouldShowProgress() const { return show_progress_; }

// feed : accumulate word counts
void BpeTrainer::feed(FeedIter&& src, FeedProcessor&& fn) {
    WordCounts total;
    for (auto& seq : src) {
        auto words = fn(seq.c_str());
        if (!words) return words.error();
        for (auto& w : *words)
            ++total[w];
    }
    words_ = std::move(total);
}

// algorithm slices 
// TODO
void BpeTrainer::addSpecialTokens(Vocab& w2id, std::vector<std::string>& id2w) const {
    for (auto& tok : special_tokens_)
        if (!w2id.count(tok.content)) {
            id2w.push_back(tok.content);
            w2id.emplace(id2w.back(), id2w.size() - 1);
        }
}

void BpeTrainer::computeAlphabet(const WordCounts& wc, Vocab& w2id, 
                                std::vector<std::string>& id2w) const {
                                
    std::unordered_map<char, std::size_t> freq;
    for (auto& kv : wc) 
        for (char c: kv.first) freq[c] += kv.second;

    for (char c : initial_alphabet_) 
            freq[c] = std::numeric_limits<std::size_t>::max();

    // limit alphabet
    if (limit_alphabet_ && freq.size() > *limit_alphabet_) {
        std::vector<std::pair<char, std::size_t>> tmp(freq.begin(), freq.end());
        std::nth_element(tmp.begin(), tmp.begin()+ *limit_alphabet_, tmp.end(),
                        [](auto& a, auto&b) {
                            return a.second > b.second;
                        });

        tmp.resize(*limit_alphabet_);
        freq.clear();
        for (auto& kv : tmp) freq.emplace(kv);
    }

    std::vector<char> chars;
    chars.reserve(freq.size());
    for (auto& kv : freq) chars.push_back(kv.first);
    std::sort(chars.begin(), chars.end());

    for (char c : chars) {
        std::string s(1, c);
        if (!w2id.count(s)) {
            id2w.push_back(s);
            w2id.emplace(s, id2w.size() - 1);
        }
    }
}

// tokenize words -> Word objects
auto BpeTrainer::tokenizeWords(const WordCounts& wc, Vocab& w2id, 
                                std::vector<std::string>& id2w, Progress& p) const 
-> std::pair<std::vector<Word>, std::vector<std::uint64_t>> {
    std::vector<Word> words;
    std::vector<std::uint64_t> counts;
    words.reserve(wc.size()); counts.reserve(wc.size());

    for (auto& kv : wc) {
        Word w;
        bool first = true;
        for (auto cIt = kv.first.begin(); cIt != kv.first.end(); ++cIt) {
            std::string tok(1, *cIt);
            if (!first && cont_prefix_) tok = *cont_prefix_ + tok;
            if (std::next(cIt) == kv.first.end() && end_suffix_) tok += *end_suffix_;
            if (!w2id.count(tok)) { 
                id2w.push_back(tok);
                w2id.emplace(tok, id2w.size() - 1);
            }
            w.add(w2id[tok], 1);
            first = false;
        }
        words.push_back(std::move(w));
        counts.push_back(kv.second);
        p.inc();
    }
    return {std::move(words), std::move(counts)};
}

// count pairs
auto BpeTrainer::countPairs(const std::vector<Word>& words, 
                            const std::vector<std::uint64_t>& counts, 
                            Progress& p) const 
-> std::pair<std::unordered_map<Pair, int>,
            std::unordered_map<Pair, std::unordered_set<std::size_t>>> {
    
    std::unordered_map<Pair, int> pairCounts;
    std::unordered_map<Pair, std::unordered_set<std::size_t>> where;
    for (std::size_t i = 0; i < words.size(); ++i) {
        auto& w = words[i];
        auto& vec = w.get_chars();
        for (std::size_t j = 0; j < vec.size(); ++j) {
            Pair pr(vec[j], vec[j+1]);
            pairCounts[pr] += counts[i];
            where[pr].insert(i);
        }
        p.inc();
    }
    return {std::move(pairCounts), std::move(where)};
}

// full training routine 
std::vector<AddedToken> BpeTrainer::doTrain(const WordCounts& wc, BPE& model) {
    // 1_ initial vocab
    Vocab w2id;
    std::vector<std::string> id2w;
    addSpecialTokens(w2id, id2w);
    computeAlphabet(wc, w2id, id2w);

    // 2_ tokenize words
    auto prog = setUpProgress();
    updateProgress(prog, wc.size(), "Tokenize");
    auto [words, counts] = tokenizeWords(wc, w2id, id2w, prog);
    finalizeProgress(prog, words.size());

    // 3_ count pairs
    updateProgress(prog, words.size(), "Pairs");
    auto [pairCnt, where] = countPairs(words, counts, prog);
    finalizeProgress(prog, words.size());

    // 4_ queue of merges 
    std::priority_queue<Merge, std::vector<Merge>, MergeCmp> queue;
    for (auto& kv : pairCnt) {
        queue.push(
            {kv.first, static_cast<std::uint64_t>(kv.second), where.at(kv.first)}
        );
    }

    // 5_ main loop 
    // TODO: finish and verify correctness 
    std::vector<std::pair<Pair, std::uint32_t>> merges;
    while (w2id.size() < vocab_size_ && !queue.empty()) {
        Merge top = queue.top(); queue.pop();
        if (top.count < min_frequency_) break;

        std::string a = id2w[top.pair.first], b = id2w[top.pair.second];
        if (cont_prefix_ && b.rfind(*cont_prefix_, 0) == 0)
            b.erase(0, cont_prefix_->size());
        std::string newTok = a + b;

        std::uint32_t newId;
        auto it = w2id.find(newTok);
        if (it == w2id.end()) {
            id2w.push_back(newTok);
            newId = id2w.size() - 1;
            w2id[newTok] = newId;
        } else {
            newId = it->second;
        }
        merges.emplace_back(top.pair, newId);

        // NOTE: merge the pair in words and update counts...
        // TODO: finish implementation 


        prog.inc();
    }
    finalizeProgress(prog, merges.size());

    // 6_ commit into model
    model.vocab_ = std::move(w2id);
    model.vocab_r_.clear();
    for (auto& kv : model.vocab_) {
        model.vocab_r_[kv.second] = kv.first;
    }

    model.merges_.clear();
    for (std::size_t i = 0; i < merges.size(); ++i)
        model.merges_[merges[i].first] = { static_cast<std::uint32_t>(i), merges[i].second };

    model.cont_prefix_ = cont_prefix_;
    model.end_suffix_ = end_suffix_;

    return special_tokens_;
}


}}}