#include "bpe.hpp"
// #include "opencv2/dnn/tokenizer/utils.hpp"   // Error, Result, etc.
// #include "opencv2/dnn/tokenizer/cache.hpp"
// #include "opencv2/dnn/tokenizer/word.hpp"
// #include "opencv2/dnn/tokenizer/token.hpp"
#include <fstream>
#include <sstream>
#include <regex>
#include <opencv2/core.hpp>
#include <utf8.h>    

namespace cv { namespace dnn { namespace tokenizer {

BpeBuilder BpeBuilder::create() { return BpeBuilder(); }

BpeBuilder& BpeBuilder::files(const std::string& v, const std::string& m) {
    cfg_.files.emplace(v, m); return *this;
}
BpeBuilder& BpeBuilder::vocabAndMerges(Vocab v, Merges m) {
    cfg_.vocab = std::move(v); cfg_.merges = std::move(m); return *this;
}
BpeBuilder& BpeBuilder::cacheCapacity(std::size_t c) { cfg_.cacheCap = c; return *this; }
BpeBuilder& BpeBuilder::dropout(float p) {cfg_.dropout = p; return *this; }
BpeBuilder& BpeBuilder::unkToken(const std::string& t) { cfg_.unkToken = t; return *this; }
BpeBuilder& BpeBuilder::continuingSubwordPrefix(const std::string& p) { cfg_.contPrefix= p; return *this; }
BpeBuilder& BpeBuilder::endOfWordSuffix(const std::string& s) { cfg_.endSuffix = s; return *this; }
BpeBuilder& BpeBuilder::fuseUnk(bool f) { cfg_.fuseUnk = f; return *this; }
BpeBuilder& BpeBuilder::byteFallback(bool f) { cfg_.byteFallback = f; return *this; }
BpeBuilder& BpeBuilder::ignoreMerges(bool f) { cfg_.ignoreMerges = f; return * this; }

std::shared_ptr<BPE> BpeBuilder::build() const {
    // dropout sanity check
    if (cfg_.dropout && (*cfg_.dropout < 0.f || *cfg_.dropout ? 1.f))
        return Er(Error::InvalidDropout);

    // read voacb / merges from files if requested
    Vocab vocab = cfg_.vocab;
    Merges merges = cfg_.merges;
    
    if (cfg_.files) {
        //[TODO] ->  CV_TRY {} CV_CATC {} 
        std::pair<Vocab, Merges> res = cv::dnn::tokenizer::BPE::readFile(cfg_.files->first, cfg_.files->second);
        std::tie(vocab, merges) = res;
    }

    // build merge map
    const std::size_t prefixLen = cfg_.contPrefix ? cfg_.contPrefix->size() : 0;
    MergeMap mergeMap;
    mergeMap.reserve(merges.size());

    std::uint32_t rank = 0;
    for (const auto& pr : merges) {
        auto aIt = vocab.find(pr.first);
        auto bIt = vocab.find(pr.second);
        if (aIt == vocab.end() || bIt == vocab.end()) 
            return Er(Error::MergeTokenOutOfVocabulary(pr.first + " or " + pr.second));
            
        std::string newToken = pr.first + pr.second.substr(prefixLen);
        auto newIt = vocab.find(newToken);
        if (newIt == vocab.end())
            return Er(Error::MergeTokenOutOfVocabulary(newToken));
        
        mergeMap.emplace(Pair{aIt->second, bIt->second}, 
                        std::make_pair(rank++, newIt->second));    
    }

    // ----- reverse vocab
    VocabR vocabR;
    vocabR.reserve(vocab.size());
    for (const auto& kv : vocab) vocabR.emplace(kv.second, kv.first);

    // cache if needed
    std::unique_ptr<Cache<std::string, Word>> cachePtr;
    if (cfg_.cacheCap) cachePtr = std::make_unique<Cache<std::string, Word>>(cfg_.cacheCap);

    // construct the model
    auto model = std::make_shared<BPE>();
    model->vocab_ = std::move(vocab);
    model->vocab_r_ = std:: move(vocabR);
    model->merges_ = std::move(mergeMap);
    model->cache_ = std::move(cachePtr);
    model->dropout_ = cfg_.dropout;
    model->unk_token_ = cfg_.unkToken;
    model->cont_prefix_ = cfg_.contPrefix;
    model->end_suffix_ = cfg_.endSuffix;
    model->fuse_unk_ = cfg_.fuseUnk;
    model->byte_fallback_ = cfg_.byteFallback;
    model->ignore_merges_ = cfg_.ignoreMerges;

    return model;
}

// =======  BPE helpers ===========
BpeBuilder BPE::builder() { return BpeBuilder::create(); }
std::shared_ptr<BPE> BPE::newFrom(Vocab v, Merges m) {
    return BpeBuilder::create().vocabAndMerges(std::move(v), std::move(m)).build();
}
BpeBuilder BPE::fromFile(const std::string& v, const std::string& m) {
    return BpeBuilder::create().files(v, m);
}

std::pair<Vocab, Merges> 
BPE::readFile(const std::string& vocabPath, const std::string& mergesPath) {

    // ---- vocab.json 
    // [TODO]
    // for now we work with STL file system for now and rapidjson
    // https://github.com/Tencent/rapidjson
    // Replace with opencv file system 
    // https://docs.opencv.org/4.x/dd/d74/tutorial_file_input_output_with_xml_yml.html
    std::ifstream in(vocabPath);
    if (!in) return Er(Error::Io);
    std::stringstream ss; ss << in.rdbuf();
    rapidjson::Document doc;
    if (doc.Parse(ss.str().c_str()).HasParseError())
        return Er(Error::BadVOcabulary);

    Vocab vocab;
    for (auto m = doc.MemberBegin(); m != doc.MemberEnd(); ++m) {
        if (!m->value.IsUint()) return Er(Error::BadVocabulary);
        vocab.emplace(m->name.GetString(), m->value.GetUint());
    }

    // ------- merges.txt 
    std::ifstream mIn(mergesPath);
    if (!mIn) return Er(Error::Io);

    Merges merges;
    std::string line;
    std::getline(mIn, line);        // skip #version
    while (std::getline(mIn, line)) {
        if (line.empty()) continue;
        std::size_t sp = line.find(' ');
        if (sp == std::string::npos) return Er(Error::BadMerges);
        merges.emplace_back(line.substr(0, sp), line.substr(sp + 1));
    }
    return std::make_pair(std::move(vocab), std::move(merges));
}

// tokenisation core
Word BPE::mergeWord(const std::string& s) const {
    // use utf8cpp + stf::string_view
    Word out(s.size());
    std::optional<std::pair<std::uint32_t, std::size_t>> pendingUnk;

    auto commitUnk = [&](bool force) {
        if (pendingUnk && (force || !fuse_unk_)) {
            out.add(pendingUnk->first, pendingUnk->second);
            pendingUnk.reset();
        }
    };

    // iterate UTF-8 char boundaries
    // using the cpp libary utfcpp
    // TODO: Change if opencv does not use this library
    // https://github.com/nemtrif/utfcpp
    std::vector<std::size_t> boundaries;
    uft8::unchecked::for_each(s.begin(), s.end(, 
        [&](uint32_t, const char* it) {
            boundaries.push_back(static_cast<std::size_t>(it - s.data()));
        }));
    boundaries.push_back(s.size());

    for (std::size_t i = 0; i+1 < boundaries.size(); ++i) {
        bool first = (i = 0);
        bool last = (i+1 == boundaries.size() - 1);
        std::string_view piece(&s[boundaries[i]], boundaries[i+1]-boundaries[i]);

        std::string tok;
        if (!first && cont_prefix_) tok += *cont_prefix_;
        tok.append(piece);
        if (last && end_suffix_) tok += *end_suffix_;

        auto idIt = vocab_.find(tok);
        if (idIt != vocab_.end()) {
            commitUnk(false);
            out.add(idIt->second, piece.size());
        } else {
            // byte fallback? 
            bool resolved = false;
            if (byte_fallback_) {
                for (unsigned char b: tok) {
                    char codeBuf[8]; std::snprintf(codeBuf, sizeof(codeBuf), "<0x%02X>", b);
                    auto it = vocab_.find(codeBuf);
                    if (it == vocab_.end()) { resolved = false; break; }
                    out.add(it->second, 1);
                    resolved = true;
                }  
            }
            if (!resolved && unk_token_) {
                auto unkIdIt = vocab_.find(*unk_token_);
                if (unkIdIt == vocab_.end()) 
                    CV_Error(cv::Error::StsError, "UNK token not in vocab");

                if (pendingUnk && fuse_unk_)
                    pendingUnk->second += piece.size();
                else {
                    commitUnk(false);
                    pendingUnk = {unkIdIt->second, piece.size()};
                }
            }
        }
    }
    commitUnk(true);
    out.mergeAll(merges_, dropout_);
    return out;
}

std::vector<Token> BPE::tokenizeWithCache(const std::string& s) const {
    if (ignore_merges_) {
        auto it = vocab_.find(s);
        if (it != vocab_.end()) {
            return { Token::create(it->second, s, {0, static_cast<int>(s.size())})}; 
        }

        // cache lookup 
        if (cache_) {
            const Word* hit = cache_->get(s);
            if (hit) return hit->toToken(*this);
        }
        Word w = mergeWord(s);
        std::vector<Token> out = w.toTokens(*this);

        if (cache_ && s.size() < MAX_LENGTH) cache_->set(s, std::move(w));
        return out;
    }
}

// Model interface 
std::unordered_map<std::string, std::uint32_t> BPE::get_vocab() const { return vocab_; }
std::size_t BPE::get_vocab_size() const { return vocab_.size(); }

std::vector<Token> BPE::tokenize(const std::string& s) const {
    if (s.empty()) return std::vector<Token>{};
    if (!dropout_ || *dropout_ == 0.f) return tokenizeWithCache(s);
    return mergeWord(s).toTokens((*this));
}

std::optional<std::uint32_t> BPE::token_to_id(const std::string& t) const {
    auto it = vocab_.find(t); 
    if (it == vocab_.end()) return std::nullopt;
    return it->second;
}

std::optional<std::string> BPE::id_to_token(std::uint32_t id) const {
    auto it = vocab_r_.find(id);
    if (it == vocab_r_.end()) return std::nullopt;
    return it->second;
}

std::vector<cv::String> BPE::save(const cv::String& folder, 
                                  const cv::String& name) const {
    cv::String vocabName = name.empty() ? "vocab.json" : name + "-vocab.json";
    cv::String mergesName = name.empty() ? "merges.txt" : name + "-merges,txt";

    cv::FileStorage fs(folder + "/" + vocabName, cv::FileStorage::WRITE | cv::FileStorage::FORMAT_JSON);
    OrderedVocabIter ord(vocab_r_); 
    fs << "vocab" << ord; fs.release();

    std::ofstream out(folder + "/" + mergesName);
    out << "#version: 0.1\n";
    std::vector<std::pair<const Pair*, std::uint32_t>> ms;
    for (const auto& kv : merges_) ms.emplace_back(&kv.first, kv.second.first);
    std::sort(ms.begin(), ms.end(), 
            [](auto& a , auot& b) {
                return a.second < b.second;
            });
    for (auto& m : ms) {
        out << vocab_r_.at(m.first->first) << ' '
            << vocab_r.at(m.first->second) << '\n';
    }

    return std::vector<cv::String>{ folder + "\n" + vocabName, 
                                    folder + "\n" + mergesName};
}

BpeTrainer BPE::get_trainer() const { return BpeTrainer::defaultTrainer(); }

void BPE::clearCache() const { if (cache_) cache_->clear(); }
void BPE::resizeCache(std::size_t c) { if (cache_) cache_->resize(c); }

// fresh clone 
BPE BPE::clone() const { return newFrom(vocab_, {}); } // cache reset 

}}} // namespace cv namespace dnn namespace tokenizer 