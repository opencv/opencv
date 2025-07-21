#include "gpt2_tokenizer_fast.hpp"
#include "utils.hpp"

namespace cv { namespace dnn { namespace tokenizer {

// Decode a Python‐style “data gym” string into raw bytes.
// Each codepoint <256 goes to itself if printable & != ' ',
// otherwise codepoint>=256 encodes as (codepoint−256).
std::string GPT2TokenizerFast::decodeDataGym(const std::string& s,
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

std::unordered_map<std::string,int> GPT2TokenizerFast::dataGymToMergeableBpeRanks(
                                        const std::string& vocabBpePath
                                        /*const std::string& encoderJsonPath*/) {
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

    // // load encoder.json and sanity-check
    // auto encText = readFile(encoderJsonPath);
    // auto encJson = json::parse(encText);

    // // decode all keys into a temp map
    // std::unordered_map<std::string,int> encMap;
    // for (auto& [k,v] : encJson.items()) {
    //     auto raw = decodeDataGym(k, data_gym_byte_to_byte);
    //     // drop special tokens if present
    //     if (raw == "<|endoftext|>" || raw == "<|startoftext|>")
    //         continue;
    //     encMap[raw] = v.get<int>();
    // }
    // if (encMap != bpe_ranks)
    //     throw std::runtime_error("BPE ranks do not match encoder.json");

    return bpe_ranks;
}

Encoding GPT2TokenizerFast::getEncodingForGPT2(const std::string& name, const std::string& vocab_file) {
    std::unordered_map<std::string, Rank> specialTokens = {
        {"<|endoftext|>", 50256}
    };
    int explicitNvocab = 50257;
    auto bpe_ranks = dataGymToMergeableBpeRanks(vocab_file);
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