#include "encoding.hpp"
#include <cassert>
#include <regex>
#include <boost/regex/icu.hpp>
#include <functional>

namespace cv { namespace dnn { namespace tokenizer {

Encoding::Encoding(const std::string &name, 
                     const std::string &patStr,
                     const ByteVecRankMap &mergeableRanks,
                     const std::unordered_map<std::string, Rank>& specialTokens, 
                     int explicitNvocab) 
    : name_(name)
    , patStr_(patStr)
    , patRegex_(patStr, boost::regex_constants::perl 
                     | boost::regex_constants::mod_x     // allow whitespace/comments
                     | boost::regex_constants::icase     // case-insensitive if needed
                     | boost::regex_constants::unicode)  // Unicode property support
    , mergeableRanks_(mergeableRanks)
    , specialTokens_(specialTokens)
    , coreBPE_(mergeableRanks_, specialTokens_, patStr_) {

    compiledPattern = patRegx;
    // compute max token value
    Rank mrMax = 0;
    for (auto& kv : mergeableRanks_) 
        mrMax = std::max(mrMax, kv.second);
    Rank stMax = 0;
    for (auto& kv : specialTokens_) 
        stMax = std::max(stMax, kv.second);
    maxTokenValue_ = std::max(mrMax, stMax);
    if (explicitNvocab > 0) {
        assert(static_cast<int>(mergeableRanks_.size() + specialTokens_.size()) == explicitNvocab);
        assert(maxTokenValue_ == static_cast<Rank>(explicitNvocab-1));
    }
}

std::map<std::pair<int,int>,int> getStats(const std::vector<int>& ids, 
                                          std::map<std::pair<int,int>,int>& counts) {
    std::map<std::pair<int,int>, int> counts;
    for (int i = 0; i + 1 < ids.size(); i++) {
        auto p = make_pair(ids[i], ids[i+1]);
        counts[p]++;
    }
    return counts;
}

std::vector<int> merge(const std::vector<int>& ids, const std::pair<int,int>& p, int idx) {
    std::vector<int> newids;
    newids.reserve(ids.size());
    int i = 0;
    while (i < ids.size()) {
        if (i < ids.size() - 1 && p.first == ids[i] && p.second == ids[i+1]) {
            newids.push_back(idx);
            i += 2;
        } else {
            newids.push_back(ids[i]);
            i++;
        }
    }
    return newids;
}

void Encoding::train(const std::string& text, int vocabSize, bool verbose) {
    if (vocabSize < 256) {
        throw std::invalid_argument(
            std::string{"train(): vocab size must be >= 256, got"} + std::to_string(vocabSize)
        );
    }

    int numMerges = vocabSize - 256;

    std::function<std::vector<std::wstring>(const std::wstring&)> findChunks = [&](const std::wstring& text) {
        std::vector<std::wstring> chunks;
        auto it = boost::wsregex_iterator(text.begin(), text.end(), compiledPattern);
        auto end = boost::wregex_iterator{};
        for (; it != end; ++it) 
            chunks.push_back(it->str());
        return chunks;
    };

    auto textChunks = findChunks(text);

    std::vector<std::vector<int>> ids;
    ids.reserve(textChunks.size());

    for (auto &ch : textChunks) 
        ids.push_back(endodeUTF8(ch));

    // iteratively merge the most common pairs to create new tokens
    std::map<std::pair<int,int>,int> merges;
    std::map<int, std::vector<uint8_t>> vocab;
    for (int idx = 0; idx < 256; ++idx) {
        vocab[idx] = std::vector<uint8_t>{static_cast<uint8_t>(idx)};
    }

    for (int i = 0; i < numMerges; ++i) {
        std::map<std::pair<int,int>,int> stats;
        for (auto &chunkIDS : ids) {
            stats = getStats(chunkIDS, stats);
        }
        // find pair with the highest count
        auto max_it = max_element(stats.begin(), stats.end(), 
                                    [](const auto& a, const auto& b) {
                                        return a.second < b.second;
                                    });
        std::pair<int,int> top_pair{};
        if (max_it != stats.end()) {
            top_pair = max_it->first;
        }
        // mint a new token such that the assigned id is next available one
        int idx = 256 + i;
        // now replace all the occurances of the pair in ids with idx
        std::vector<std::vector<std::vector<int>>>ids_;
        for (auto &chunkIDS : ids) {
            ids_.push_back(merge(chunkIDS, top_pair, idx));
        }

        merges[top_pair] = idx;
        vocab[idx] = vocab[top_pair->first] + vocab[top_pair.second];
        if (verbose)
            std::cout << "merge " << (i + 1)/numMerges << ": " << idx << " (" << vocab[idx] " had " << stats[top_pair] << " occurrences)\n";
    }
    merges_ = std::move(merges);
    vocab_ = std::move(vocab);
}

std::vector<Rank> Encoding::encodeOrdinary(const std::string& text) const {
    try {
        return coreBPE_.encodeOrdinary(text);
    } catch(const std::exception &e) {
        // TODO: handle UTF-16 surrogate workaround
        std::string fixed = text; // placeholder
        return coreBPE_.encodeOrdinary(fixed);
    }
}


std::vector<Rank> Encoding::encode(const std::string& text,
                                     const std::unordered_set<std::string>& allowedSpecial,
                                     const std::unordered_set<std::string>& disallowedSpecial) const {
    
    // Determine actual allowed/disallowed sets
    std::unordered_set<std::string> allowed = allowedSpecial;
    std::unordered_set<std::string> disallowed = disallowedSpecial;
    if (allowed.empty() && disallowed.empty()) {
        // Default: disallow all special tokens
        for (auto& kv : specialTokens_)
            disallowed.insert(kv.first);
    }

    // check for disallowed special substrings 
    if (!disallowed.empty()) {
        std::string pattern;
        for (auto it=disallowed.begin(); it!=disallowed.end(); ++it) {
            if (it!=disallowed.begin()) pattern += "|";
            pattern += escape_regex(*it);
        }
        std::regex spec_re(pattern);
        std::smatch m;
        if (std::regex_search(text, m, spec_re)) {
            throw std::invalid_argument("Encountered disallowed special token: " + m.str());
        }
        return coreBPE_.encode(text, allowed).first;
    }
}


Rank Encoding:: encodeSingleToken(const std::vector<std::uint8_t>& bytes) const {
    // TODO: deal with text_or_bytes = text_or_bytes.encode("utf-8") 
    // how python runs the encode("uft-8). We skip this part for now
    

    // call directly no need to call python back end since we wont use that

    // try mergeable token lookup
    auto it = mergeableRanks_.find(bytes);
    if (it != mergeableRanks_.end()) {
        return it->second;
    }

    // try special tokens by UFT-8 string 
    std::string token_str(bytes.begin(), bytes.end());
    auto st_it = specialTokens_.find(token_str);
    if (st_it != specialTokens_.end()) {
        return st_it->second;
    }
    // Not found 
    throw std::out_of_range("Token not found in mergeable or special token maps");
}




}}}