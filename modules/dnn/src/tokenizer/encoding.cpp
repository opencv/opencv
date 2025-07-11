#include <opencv2/core.hpp>

#include "encoding.hpp"
#include <cassert>
#include <regex>
#include <functional>
#include <iostream>
#include "unicode.hpp"
#include "utils.hpp"

#include <nlohmann/json.hpp>

using json = nlohmann::json;

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


// Read entire file into a string
static std::string readFile(const std::string& path) {
    std::ifstream in{path, std::ios::binary};
    if (!in) throw std::runtime_error("Failed to open " + path);
    std::ostringstream buf;
    buf << in.rdbuf();
    return buf.str();
}

// Decode a Python‐style “data gym” string into raw bytes.
// Each codepoint <256 goes to itself if printable & != ' ',
// otherwise codepoint>=256 encodes as (codepoint−256).
static std::string decodeDataGym(const std::string& s,
    const std::unordered_map<uint32_t,uint8_t>& dg2b)
{
    std::string out;
    // decode UTF-8 codepoints one by one
    for (size_t i = 0; i < s.size();) {
        uint8_t c = static_cast<uint8_t>(s[i]);
        uint32_t cp = 0, length = 0;
        if      ((c & 0x80) == 0)       { cp = c; length = 1; }
        else if ((c & 0xE0) == 0xC0)    { cp = c & 0x1F; length = 2; }
        else if ((c & 0xF0) == 0xE0)    { cp = c & 0x0F; length = 3; }
        else if ((c & 0xF8) == 0xF0)    { cp = c & 0x07; length = 4; }
        else throw std::runtime_error("Invalid UTF-8");
        for (size_t j = 1; j < length; ++j) {
            cp = (cp << 6) | (static_cast<uint8_t>(s[i+j]) & 0x3F);
        }
        i += length;
        auto it = dg2b.find(cp);
        if (it == dg2b.end())
            throw std::runtime_error("Unknown data-gym codepoint");
        out.push_back(static_cast<char>(it->second));
    }
    return out;
}

std::unordered_map<std::string,int> dataGymToMergeableBpeRanks(
                                        const std::string& vocabBpePath,
                                        const std::string& encoderJsonPath) {
    std::vector<uint8_t> rank_to_intbyte;
    rank_to_intbyte.reserve(256);
    std::unordered_map<uint32_t,uint8_t> data_gym_byte_to_byte;
    // first the “printable” Python bytes, excluding space (32)
    for (int b = 0; b < 256; ++b) {
        bool printable =
            (b >= 33 && b <= 126) ||
            (b >= 161 && b <= 255 && b != 173);
        if (printable) {
            rank_to_intbyte.push_back(b);
            data_gym_byte_to_byte[b] = static_cast<uint8_t>(b);
        }
    }
    // then all the rest
    uint32_t nextra = 0;
    for (int b = 0; b < 256; ++b) {
        if (std::find(rank_to_intbyte.begin(),
                      rank_to_intbyte.end(),
                      static_cast<uint8_t>(b))
            == rank_to_intbyte.end())
        {
            rank_to_intbyte.push_back(b);
            // Python used codepoint = 256 + nextra
            data_gym_byte_to_byte[256 + nextra] = static_cast<uint8_t>(b);
            ++nextra;
        }
    }
    if (rank_to_intbyte.size() != 256)
        throw std::runtime_error("Bad rank_to_intbyte size");

    // read and parse merges
    auto bpeText = readFile(vocabBpePath);
    std::vector<std::pair<std::string,std::string>> merges;
    {
        std::istringstream lines(bpeText);
        std::string line;
        std::getline(lines, line); // skip #version
        while (std::getline(lines, line)) {
            if (line.empty()) continue;
            std::istringstream iss(line);
            std::string a,b;
            iss >> a >> b;
            merges.emplace_back(a,b);
        }
    }

    std::unordered_map<std::string,int> bpe_ranks;
    for (int i = 0; i < (int)rank_to_intbyte.size(); ++i) {
        bpe_ranks[ std::string(1, static_cast<char>(rank_to_intbyte[i])) ] = i;
    }

    int nextRank = bpe_ranks.size();
    for (auto &m : merges) {
        auto a = decodeDataGym(m.first,  data_gym_byte_to_byte);
        auto b = decodeDataGym(m.second, data_gym_byte_to_byte);
        bpe_ranks[a + b] = nextRank++;
    }

    // load encoder.json and sanity-check
    auto encText = readFile(encoderJsonPath);
    auto encJson = json::parse(encText);

    // decode all keys into a temp map
    std::unordered_map<std::string,int> encMap;
    for (auto& [k,v] : encJson.items()) {
        auto raw = decodeDataGym(k, data_gym_byte_to_byte);
        // drop special tokens if present
        if (raw == "<|endoftext|>" || raw == "<|startoftext|>")
            continue;
        encMap[raw] = v.get<int>();
    }
    if (encMap != bpe_ranks)
        throw std::runtime_error("BPE ranks do not match encoder.json");

    return bpe_ranks;
}

Encoding getEncodingForGPT2(const std::string& name) {

    std::unordered_map<std::string, Rank> specialTokens = {
        {"<|endoftext|>", 50256}
    };
    int explicitNvocab = 50257;

    auto bpe_ranks = dataGymToMergeableBpeRanks(
        "../modules/dnn/src/tokenizer/vocab.bpe",
        "../modules/dnn/src/tokenizer/encoder.json"
    );

    ByteVecRankMap mergeableRanks;
        mergeableRanks.reserve(bpe_ranks.size());
        for (auto& [tok, rank] : bpe_ranks) {
            // copy each std::string into a ByteVec
            ByteVec v(tok.begin(), tok.end());
            mergeableRanks.emplace(std::move(v), static_cast<Rank>(rank));
        }


    return Encoding(name, R50K_UTF8, std::move(mergeableRanks), std::move(specialTokens), explicitNvocab);
}

}}}