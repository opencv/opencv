/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  Portions of this file are inspired by or adapted from:
//      • tiktoken (Python) implementation:
//          https://github.com/openai/tiktoken/blob/main/tiktoken/core.py
//      • minbpe by Andrej Karpathy:
//          https://github.com/karpathy/minbpe/blob/master/minbpe/regex.py
//
//  This file is part of the OpenCV DNN module for tokenization.
//
////////////////////////////////////////////////////////////////////////////////////////*/

/*M///////////////////////////////////////////////////////////////////////////////////////
// MIT License
//
// Copyright (c) 2022 OpenAI, Shantanu Jain
// Copyright (c) 2024 Andrej Karpathy
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////////////*/
#include <opencv2/core.hpp>

#include "encoding.hpp"
#include <cassert>
#include <regex>
#include <functional>
#include <iostream>
#include "unicode.hpp"
#include "utils.hpp"

namespace cv { namespace dnn { namespace tokenizer {

Encoding::Encoding(const std::string &name, 
                     const std::string &patStr,
                     const ByteVecRankMap &mergeableRanks,
                     const std::unordered_map<std::string, Rank>& specialTokens, 
                     int explicitNvocab) 
    : name_(name)
    , patStr_(patStr)
    , mergeableRanks_(mergeableRanks)
    , specialTokens_(specialTokens)
    , coreBPE_(mergeableRanks_, specialTokens_, patStr_) {
    // compute max token value
    Rank mrMax = 0;
    for (auto& kv : mergeableRanks_) 
        mrMax = std::max(mrMax, kv.second);
    Rank stMax = 0;
    for (auto& kv : specialTokens_) 
        stMax = std::max(stMax, kv.second);
    maxTokenValue_ = std::max(mrMax, stMax);
    if (explicitNvocab > 0) {
        std::cout << "mergeableRanks_.size(): " << mergeableRanks_.size() << std::endl;
        std::cout << "specialTokens_.size(): " << specialTokens_.size() << std::endl;
        std::cout << "explicitNvocab: " << explicitNvocab << std::endl;
        assert(static_cast<int>(mergeableRanks_.size() + specialTokens_.size()) == explicitNvocab);
        assert(maxTokenValue_ == static_cast<Rank>(explicitNvocab-1));
    }
}

void getStats(const std::vector<int>& ids, std::map<std::pair<int,int>,int>& counts) {
    for (int i = 0; i + 1 < ids.size(); i++) {
        auto p = std::make_pair(ids[i], ids[i+1]);
        counts[p]++;
    }
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

void Encoding::train_bpe(const std::string& text, int vocabSize, bool verbose) {
    if (vocabSize < 256) {
        throw std::invalid_argument(
            std::string{"train(): vocab size must be >= 256, got"} + std::to_string(vocabSize)
        );
    }
    int numMerges = vocabSize - 256;
    std::vector<std::string> regexes = { patStr_}; 
    std::cout << "size of text: " << text.size() << std::endl;
    // std::cout << "Before regex split new\n\n";
    /*
        Since we dont have a pre-tokenization step implemented yet 
        the training for wiki text is roughly ~540MB which is much 
        to large to process all of it at once in unicode_regex_split.
        For now a practical method is to split it into smaller chuncks 
        before processing. Later should add a pre-tokenization step for
        training like hugginface does. 
    // */
    // // split text into chucks ~10MB
    // const size_t chunck_size = 10 * 1024 * 1024; // 10MB
    // std::vector<std::string> textChunks; 
    // for (size_t i = 0; i < text.size(); i += chunck_size) {
    //     textChunks.push_back(text.substr(i, std::min(chunck_size, text.size() - i)));
    // }
    // // Tokenize chunck
    // std::vector<std::string> allTokens;
    // for (const auto& chunck : textChunks) {
    //     std::vector<std::string> tokens = unicode_regex_split(chunck, regexes);
    //     allTokens.insert(allTokens.end(), tokens.begin(), tokens.end());
    // }
    // std::vector<std::string> textChunks = unicode_regex_split(text, regexes);
    // std::cout << "After regex split\n\n";

    //[TODO] The above is still slow and wont compile 
    // DEBUG: Use only the first 1MB for testing
    // size_t sample_size = std::min<size_t>(text.size(), 1024 * 1024); // 1MB 
    size_t sample_size = std::min<size_t>(text.size(), 10 * 1024); // 10KB
    std::string sample_text = text.substr(0, sample_size);
    std::cout << "Before regex split\n\n";

    // Split sample_text into chunks (here, just one chunk for simplicity)
    std::vector<std::string> textChunks = { sample_text };

    std::vector<std::string> allTokens;
    for (const auto& chunk : textChunks) {
        std::vector<std::string> tokens = unicode_regex_split(chunk, regexes);
        allTokens.insert(allTokens.end(), tokens.begin(), tokens.end());
    }
    std::vector<std::vector<int>> ids;
    ids.reserve(allTokens.size());

    for (auto &ch : allTokens) 
        ids.push_back(encodeUTF8(ch));

    // iteratively merge the most common pairs to create new tokens
    std::map<std::pair<int,int>,int> merges;
    std::map<int, std::vector<uint8_t>> vocab;
    for (int idx = 0; idx < 256; ++idx) {
        vocab[idx] = std::vector<uint8_t>{static_cast<uint8_t>(idx)};
    }

    for (int i = 0; i < numMerges; ++i) {
        std::map<std::pair<int,int>,int> stats;
        for (auto &chunkIDS : ids) {
            getStats(chunkIDS, stats);
        }
        // find pair with the highest count
        auto max_it = max_element(stats.begin(), stats.end(), 
                                    [](const auto& a, const auto& b) {
                                        return a.second < b.second;
                                    });
        if (max_it == stats.end()) break;
        std::pair<int,int> top_pair = max_it->first;

        // mint a new token such that the assigned id is next available one
        int idx = 256 + i;
        // now replace all the occurances of the pair in ids with idx
        std::vector<std::vector<int>> ids_;
        ids_.reserve(ids.size());
        for (auto &chunkIDS : ids) {
            ids_.push_back(merge(chunkIDS, top_pair, idx));
        }
        ids.swap(ids_);

        merges[top_pair] = idx;
        std::vector<std::uint8_t> v;
        auto &A = vocab[top_pair.first];
        auto &B = vocab[top_pair.second];
        v.reserve(A.size() + B.size());
        v.insert(v.end(), A.begin(), A.end());
        v.insert(v.end(), B.begin(), B.end());
        vocab[idx] = std::move(v);

        if (verbose) {
            std::cout
                << "merge " << (i+1) << L"/" << numMerges
                << ": (" << top_pair.first << L"," << top_pair.second << L")"
                << " -> " << idx
                << " had " << max_it->second << L" occurrences\n";
        }
    }
    merges_ = std::move(merges);
    vocab_ = std::move(vocab);

    mergeableRanks_.clear();
    for (const auto& kv : vocab_) {
        // kv.first is the token id (rank), kv.second is the byte sequence
        mergeableRanks_[kv.second] = static_cast<Rank>(kv.first);
    }
}

void Encoding::train_bpe_v2(const std::vector<std::string>& texts, int vocabSize, int minFreq,  int max_token_length, bool verbose) {
    std::unordered_map<std::string, int> word_to_id;
    std::vector<std::string> id_to_word;
    int next_id = 0;
    for (const auto& kv : specialTokens_) {
        word_to_id[kv.first] = next_id++;
        id_to_word.push_back(kv.first);
    }

    // Compute initial alphabet
    std::unordered_map<std::string, int> word_counts;
    std::unordered_set<std::string> alphabet;
    for (const auto& text : texts) {
        word_counts[text]++;
        // Split text into Unicode codepoints
        std::vector<uint32_t> cps = unicode_cpts_from_utf8(text);
        for (uint32_t cp : cps) {
            std::string utf8 = unicode_cpt_to_utf8(cp);
            alphabet.insert(utf8);
        }
    }

    std::vector<std::string> sorted_alphabet(alphabet.begin(), alphabet.end());
    std::sort(sorted_alphabet.begin(), sorted_alphabet.end(), [](const std::string& a, const std::string& b) {
        uint32_t ca = unicode_cpts_from_utf8(a)[0];
        uint32_t cb = unicode_cpts_from_utf8(b)[0];
        return ca < cb;
    });
    for (const std::string& s : sorted_alphabet) {
        if (word_to_id.count(s) == 0) {
            word_to_id[s] = next_id++;
            id_to_word.push_back(s);
        }
    }
    // Tokenize words (each word as vector of token ids)
    std::vector<std::vector<int>> words;
    std::vector<int> counts;
    for (const auto& kv : word_counts) {
        std::vector<int> ids;
        std::vector<uint32_t> cps = unicode_cpts_from_utf8(kv.first);
        for (uint32_t cp : cps) {
            std::string utf8 = unicode_cpt_to_utf8(cp);
            ids.push_back(word_to_id[utf8]);
        }
        words.push_back(ids);
        counts.push_back(kv.second);
    }
    // Count pairs
    using Pair = std::pair<int, int>;
    std::map<Pair, int> pair_counts;
    std::map<Pair, std::unordered_set<int>> where_to_update;
    for (int i = 0; i < words.size(); ++i) {
        const auto& ids = words[i];
        for (int j = 0; j + 1 < ids.size(); ++j) {
            Pair p = {ids[j], ids[j+1]};
            pair_counts[p] += counts[i];
            where_to_update[p].insert(i);
        }
    }
    // Merge loop
    struct Merge {
        Pair pair;
        int count;
        int pos;
        bool operator<(const Merge& other) const { 
            if (count != other.count)
                return count < other.count; // higher count first
            return pair > other.pair; //lex smallest first (priority_queue is max-heap)
        }
    };
    std::priority_queue<Merge> queue;
    for (const auto& kv : pair_counts) {
        if (kv.second > 0) {
            queue.push(Merge{kv.first, kv.second, 0});
        }
    }

    std::vector<std::pair<Pair, int>> merges;
    while (word_to_id.size() < vocabSize && !queue.empty()) {
        Merge top = queue.top(); queue.pop();
        if (top.count < minFreq) break;
        // Build new token
        std::string new_token = id_to_word[top.pair.first] + id_to_word[top.pair.second];
        std::vector<uint32_t> cps = unicode_cpts_from_utf8(new_token);
        if (cps.size() > max_token_length) {
            continue; // skip this merge
        }
        int new_token_id = next_id++;
        id_to_word.push_back(new_token);
        word_to_id[new_token] = new_token_id;
        merges.push_back({top.pair, new_token_id});

        // Merge in all words
        for (int idx : where_to_update[top.pair]) {
            auto& ids = words[idx];
            std::vector<int> new_ids;
            for (int j = 0; j < ids.size(); ) {
                if (j + 1 < ids.size() && ids[j] == top.pair.first && ids[j+1] == top.pair.second) {
                    new_ids.push_back(new_token_id);
                    j += 2;
                } else {
                    new_ids.push_back(ids[j]);
                    j += 1;
                }
            }
            ids = std::move(new_ids);
        }
        // Re-count pairs 
        pair_counts.clear();
        where_to_update.clear();
        for (int i = 0; i < words.size(); ++i) {
            const auto& ids = words[i];
            for (int j = 0; j + 1 < ids.size(); ++j) {
                Pair p = {ids[j], ids[j+1]};
                pair_counts[p] += counts[i];
                where_to_update[p].insert(i);
            }
        }
        // Rebuild queue
        queue = std::priority_queue<Merge>();
        for (const auto& kv : pair_counts) {
            if (kv.second > 0) {
                queue.push(Merge{kv.first, kv.second, 0});
            }
        }
        if (verbose) {
            std::cout << "Merge " << (word_to_id.size() - id_to_word.size() + merges.size())
                    << "/" << (vocabSize - (int)id_to_word.size())
                    << ": (" << top.pair.first << "," << top.pair.second << ") -> " << new_token_id
                    << " freq=" << top.count << "\n";
        }
    }

    vocab_.clear();
    merges_.clear();

    // Build vocab_: map token id to its byte sequence
    for (int id = 0; id < (int)id_to_word.size(); ++id) {
        const std::string& token_str = id_to_word[id];
        std::vector<uint8_t> bytes(token_str.begin(), token_str.end());
        vocab_[id] = std::move(bytes);
    }
    // Build merges_: map pair (token id, token id) -> new token id
    for (const auto& merge : merges) {
        merges_[merge.first] = merge.second;
    }
    // mergeableRanks_: map ByteVec -> token id
    mergeableRanks_.clear();
    for (int id = 0; id < (int)id_to_word.size(); ++id) {
        ByteVec bv(id_to_word[id].begin(), id_to_word[id].end());
        mergeableRanks_[bv] = static_cast<Rank>(id);
    }
    coreBPE_ = CoreBPE(mergeableRanks_, specialTokens_, patStr_);
}

std::vector<Rank> Encoding::encodeOrdinary(const std::string& text) const {
    return coreBPE_.encodeOrdinary(text);
}

std::vector<Rank> Encoding::encode(const std::string& text,
                                     const std::unordered_set<std::string>& allowedSpecial,
                                     const std::unordered_set<std::string>& disallowedSpecial) const {
    
    // 1. Handle "all" for allowed/disallowed (simulate with a special value or overload)
    std::unordered_set<std::string> allowed = allowedSpecial;
    std::unordered_set<std::string> disallowed = disallowedSpecial;

    // If allowedSpecial is "all", allow all special tokens
    // Example: if allowedSpecial contains "__ALL__", treat as all
    if (allowed.size() == 1 && allowed.count("__ALL__")) {
        for (const auto& kv : specialTokens_)
            allowed.insert(kv.first);
        allowed.erase("__ALL__");
    }

    if (disallowed.size() == 1 && disallowed.count("__ALL__")) {
        for (const auto& kv : specialTokens_) {
            if (!allowed.count(kv.first))
                disallowed.insert(kv.first);
        }
        disallowed.erase("__ALL__");
    }

    // disallow all special tokens if both sets are empty
    if (allowed.empty() && disallowed.empty()) {
        for (const auto& kv : specialTokens_)
            disallowed.insert(kv.first);
    }

    // If disallowed is empty, allow all text (even if it matches a special token)
    if (disallowed.empty()) {
        return coreBPE_.encode(text, allowed).first;
    }

    //  Regex check for disallowed special tokens
    std::string pattern;
    for (auto it = disallowed.begin(); it != disallowed.end(); ++it) {
        if (it != disallowed.begin()) pattern += "|";
        pattern += escape_regex(*it);
    }
    std::regex spec_re(pattern);
    std::smatch m;
    if (std::regex_search(text, m, spec_re)) {
        throw std::invalid_argument("Encountered disallowed special token: " + m.str());
    }
    
    return coreBPE_.encode(text, allowed).first;
}

Rank Encoding:: encodeSingleToken(const std::vector<std::uint8_t>& bytes) const {
    return coreBPE_.encodeSingleToken(const_cast<std::vector<std::uint8_t>&>(bytes));
}

std::string Encoding::decode(const std::vector<Rank>& tokens, const std::string& errors) const {
    auto opt_bytes = coreBPE_.decodeBytes(tokens);
    if (!opt_bytes) throw std::runtime_error("Invalid decode.");
    const ByteVec& bytes = *opt_bytes;

    // Convert bytes to std::string (UTF-8)
    std::string result(reinterpret_cast<const char*>(bytes.data()), bytes.size());

    // If strict, validate UTF-8
    if (errors == "strict") {
        // Simple UTF-8 validation
        size_t i = 0;
        while (i < result.size()) {
            unsigned char c = static_cast<unsigned char>(result[i]);
            size_t len = 0;
            if (c < 0x80) len = 1;
            else if ((c & 0xE0) == 0xC0) len = 2;
            else if ((c & 0xF0) == 0xE0) len = 3;
            else if ((c & 0xF8) == 0xF0) len = 4;
            else throw std::runtime_error("Invalid UTF-8 sequence in decoded bytes");
            if (i + len > result.size())
                throw std::runtime_error("Truncated UTF-8 sequence in decoded bytes");
            // Check continuation bytes
            for (size_t j = 1; j < len; ++j)
                if ((static_cast<unsigned char>(result[i + j]) & 0xC0) != 0x80)
                    throw std::runtime_error("Invalid UTF-8 continuation byte in decoded bytes");
            i += len;
        }
    }
    // Replace U+0120 (Ġ, "\xC4\xA0") with space 
    return result;
}

std::vector<std::uint8_t> Encoding::decodeBytes(const std::vector<Rank>& tokens) const {
    auto opt_bytes = coreBPE_.decodeBytes(tokens);
    if (!opt_bytes) throw std::runtime_error("Invalid decode.");
    return *opt_bytes;
}

std::vector<std::uint8_t> Encoding::decodeSingleTokenBytes(Rank token) const {
    // Decodes a token into bytes
    return coreBPE_.decodeSingleTokenBytes(token);
}


Encoding getEncodingForCl100k_base(const std::string &name, const std::string& cl00k_case_file) {
    if (name != "cl100k_base")
        throw std::runtime_error("Wrong model name. This model is cl100k_base");
    
    auto bpeText = readFile(cl00k_case_file);
    ByteVecRankMap mergeableRanks;

    std::istringstream lines(bpeText);
    std::string line;
    while (std::getline(lines, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        std::string token_b64;
        int rank;
        if (!(iss >> token_b64 >> rank)) {
            throw std::runtime_error("Error parsing line: " + line);
        }
        // Decode base64 token 
        auto bytes = base64_decode(token_b64);
        ByteVec bv(bytes.begin(), bytes.end());
        mergeableRanks.emplace(std::move(bv), static_cast<Rank>(rank));
    }
    // Special tokens for cl100k_base
    std::unordered_map<std::string, Rank> specialTokens = {
        {"<|endoftext|>", 100257},
        {"<|fim_prefix|>", 100258},
        {"<|fim_middle|>", 100259},
        {"<|fim_suffix|>", 100260},
        {"<|endofprompt|>", 100276}
    };

    return Encoding(name, CL100K_BASE, std::move(mergeableRanks), std::move(specialTokens), -1);
}

Encoding getEncodingForCl100k_baseFromJSON_FS(const std::string &name,
                                                         const std::string &json_path)
{
    if (name != "cl100k_base")
        CV_Error(cv::Error::StsError, "Wrong model name. This model is cl100k_base");

    cv::FileStorage fs(json_path, cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);
    if (!fs.isOpened())
        CV_Error(cv::Error::StsError, "Failed to open tokenizer.json: " + json_path);

    // model.vocab (HF-style)
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

    const int explicitNVocab = -1; // We pass -1 so we dont need to check the vocab size. We only do this for gpt2.
    return Encoding(name, CL100K_BASE, std::move(mergeableRanks), std::move(specialTokens), explicitNVocab);
}


}}}