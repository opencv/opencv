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
    std::vector<std::string> textChunks = unicode_regex_split(text, regexes);

    std::vector<std::vector<int>> ids;
    ids.reserve(textChunks.size());

    for (auto &ch : textChunks) 
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

std::vector<Rank> Encoding::encodeOrdinary(const std::string& text) const {
    return coreBPE_.encodeOrdinary(text);
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


int next_utf8_codepoint(const std::string& s, size_t& i) {
    unsigned char c = s[i];
    int codepoint = 0;
    if (c < 0x80) {
        codepoint = c;
        i += 1;
    } else if ((c & 0xE0) == 0xC0) {
        codepoint = ((c & 0x1F) << 6) | (s[i+1] & 0x3F);
        i += 2;
    } else if ((c & 0xF0) == 0xE0) {
        codepoint = ((c & 0x0F) << 12) | ((s[i+1] & 0x3F) << 6) | (s[i+2] & 0x3F);
        i += 3;
    } else {
        throw std::runtime_error("Unsupported UTF-8 sequence in merge token");
    }
    return codepoint;
}

Encoding getEncodingForGPT2(const std::string& name) {

    std::ifstream bpe_f("../modules/dnn/src/tokenizer/vocab.bpe");
    if (!bpe_f) throw std::runtime_error("Failed to open vocab.bpe");

    std::string line;
    std::getline(bpe_f, line);  // skip #version
    std::vector<std::pair<std::string, std::string>> merges;
    while (std::getline(bpe_f, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        std::string lhs, rhs;
        if (iss >> lhs >> rhs) merges.emplace_back(lhs, rhs);
    }


    std::vector<uint8_t> rank_to_intbyte;
    rank_to_intbyte.reserve(256);
    std::unordered_map<std::string, uint8_t> data_gym_byte_to_byte;

    auto is_printable_py = [](int b) {
        if (b == 32 || b == 160 || b == 173) return false;
        if (33 <= b && b <= 126) return true;
        if (161 <= b && b <= 255 && b != 173) return true;
        return false;
    };


    for (int b = 0; b < 256; ++b) {
        if (is_printable_py(b)) {
            rank_to_intbyte.push_back(static_cast<uint8_t>(b));
            std::string s(1, static_cast<char>(b));
            data_gym_byte_to_byte[s] = static_cast<uint8_t>(b);
        }
    }

    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (!is_printable_py(b)) {
            rank_to_intbyte.push_back(static_cast<uint8_t>(b));
            std::string key = unicode_cpt_to_utf8(256 + n);
            data_gym_byte_to_byte[key] = static_cast<uint8_t>(b);
            ++n;
        }
    }
    assert(rank_to_intbyte.size() == 256);

    


    auto decodeDataGym = [&](const std::string& tok) -> ByteVec {
        ByteVec out;
        size_t i = 0;
        while (i < tok.size()) {
            size_t old_i = i;
            int codepoint = next_utf8_codepoint(tok, i);
            std::string key;
            if (codepoint < 256 && is_printable_py(codepoint)) {
                key = std::string(1, static_cast<char>(codepoint));
            } else {
                key = unicode_cpt_to_utf8(codepoint);
            }
            auto it = data_gym_byte_to_byte.find(key);
            if (it == data_gym_byte_to_byte.end()) {
                std::cerr << "Unknown byte in merge token: ";
                for (size_t j = 0; j < tok.size(); ++j)
                    std::cerr << std::hex << (int)(unsigned char)tok[j] << " ";
                std::cerr << " | as string: " << tok << std::endl;
                throw std::runtime_error("Unknown byte in merge token: " + tok);
            }
            out.push_back(it->second);
        }
        return out;
    };


    ByteVecRankMap mergeableRanks;
    for (size_t i = 0; i < rank_to_intbyte.size(); ++i)
        mergeableRanks[{rank_to_intbyte[i]}] = static_cast<int>(i);

    n = static_cast<int>(mergeableRanks.size());
    for (const auto& p : merges) {
        ByteVec merged = decodeDataGym(p.first);
        ByteVec second = decodeDataGym(p.second);
        merged.insert(merged.end(), second.begin(), second.end());
        mergeableRanks[std::move(merged)] = n++;
    }


    auto enc_json = read_encoder_json("../modules/dnn/src/tokenizer/encoder.json");
    ByteVecRankMap enc_json_loaded;
    for (auto &kv : enc_json)
        enc_json_loaded[decodeDataGym(kv.first)] = kv.second;

    enc_json_loaded.erase(decodeDataGym("<|endoftext|>"));
    enc_json_loaded.erase(decodeDataGym("<|startoftext|>"));

    if (enc_json_loaded != mergeableRanks) {
        std::cerr << "Sanity check failed: encoder.json does not match vocab.bpe" << std::endl;
    }

    std::unordered_map<std::string, Rank> specialTokens = {
        {"<|endoftext|>", 50256}
    };
    int explicitNvocab = 50257;

    return Encoding(name, R50K_UTF8, std::move(mergeableRanks), std::move(specialTokens), explicitNvocab);
}


std::string Encoding::decode(const std::vector<Rank>& tokens, const std::string& errors) const {
    auto opt_bytes = coreBPE_.decodeBytes(tokens);
    if (!opt_bytes) throw std::runtime_error("Invalid decode.");
    const ByteVec& bytes = *opt_bytes;

    // Convert bytes to std::string (UTF-8)
    std::string result(reinterpret_cast<const char*>(bytes.data()), bytes.size());

    // If strict, validate UTF-8 (replace with your own validator if needed)
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
    size_t pos = 0;
    while ((pos = result.find("\xC4\xA0", pos)) != std::string::npos) {
        result.replace(pos, 2, " ");
        pos += 1;
    }
    return result;
}

std::vector<std::uint8_t> Encoding::decodeBytes(const std::vector<Rank>& tokens) const {
    auto opt_bytes = coreBPE_.decodeBytes(tokens);
    if (!opt_bytes) throw std::runtime_error("Invalid decode.");
    return *opt_bytes;
}

}}}