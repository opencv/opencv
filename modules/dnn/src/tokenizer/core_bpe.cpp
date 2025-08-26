// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "unicode.hpp"
#include "utils.hpp"
#include "core_bpe.hpp"

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <regex>
#include <set>
#include <stdexcept>
#include <thread>
#include <cassert>
#include <optional>
#include <iostream>

namespace cv { namespace dnn { 
CV__DNN_INLINE_NS_BEGIN
static constexpr std::uint32_t RANK_MAX  = std::numeric_limits<std::uint32_t>::max();
static constexpr std::size_t SZ_MAX = std::numeric_limits<std::size_t>::max();

static std::uint32_t maybeGetRank(const ByteVecRankMap& ranks, const std::vector<std::uint8_t>& key) {
    auto it = ranks.find(key);
    return it == ranks.end() ? RANK_MAX : it->second;
}
/*
 * This function takes a sequence of bytes and a map of
 * mergeable byte pairs (with associated ranks/merge priority), and repeatedly merges the
 * lowest-ranked adjacent pairs until no further merges are possible. The result is a vector
 * describing the split points and ranks of the final token boundaries.
 *
 * This function is closely modeled after the original Rust implementation in OpenAI's tiktoken
 * library, which can be found here:
 * https://github.com/openai/tiktoken/blob/4560a8896f5fb1d35c6f8fd6eee0399f9a1a27ca/src/lib.rs#L17-L73
 *
 */
std::vector<std::pair<std::size_t, std::uint32_t>> bytePairMerge(const ByteVecRankMap& ranks, 
                                                        const std::vector<std::uint8_t>& piece) {
    std::vector<std::pair<std::size_t, std::uint32_t>> parts;
    parts.reserve(piece.size() + 1);

    std::pair<std::uint32_t, std::size_t> minRank{RANK_MAX, SZ_MAX};
    
    for (std::size_t i = 0; i+1 < piece.size(); ++i) {
        std::vector<std::uint8_t> key(piece.begin() + i, piece.begin() + i + 2);
        std::uint32_t r = maybeGetRank(ranks, key);
        if (r < minRank.first) minRank = {r, i};
        parts.emplace_back(i, r);
    } 
    parts.emplace_back(piece.size() - 1, RANK_MAX);
    parts.emplace_back(piece.size(), RANK_MAX);

    auto getRank = [&](const std::vector<std::pair<std::size_t, std::uint32_t>>& p, 
                       std::size_t idx) -> std::uint32_t {
        if (idx + 3 < p.size()) {
            std::size_t s = p[idx].first;
            std::size_t e = p[idx+3].first;
            std::vector<std::uint8_t> key(piece.begin()+s, piece.begin()+e);
            return maybeGetRank(ranks, key);
        }
        return RANK_MAX;
    };

    while (minRank.first != RANK_MAX) {
        std::size_t i = minRank.second;
        if (i > 0) parts[i-1].second = getRank(parts, i-1);
        if (i < parts.size()-2) parts[i].second = getRank(parts, i);
        parts.erase(parts.begin() + static_cast<long>(i+1));

        minRank = {RANK_MAX, SZ_MAX};
        for (std::size_t j = 0; j + 1 < parts.size(); ++j) {
            std::uint32_t r = parts[j].second;
            if (r < minRank.first) minRank = {r, j};
        }
    }
    return parts;
}

std::vector<std::uint32_t> bytePairEncode(const std::vector<std::uint8_t>& piece, 
                                 const ByteVecRankMap& ranks) {

    if (piece.size() == 1) {
        auto it = ranks.find(piece);
        return it == ranks.end() ? std::vector<std::uint32_t>{} : std::vector<std::uint32_t>{it->second};
    }
    auto merged = bytePairMerge(ranks, piece);
    std::vector<std::uint32_t> out; 
    out.reserve(merged.size()-1);

    for (std::size_t i = 0; i+1 <  merged.size(); ++i) {
        std::size_t s = merged[i].first, e = merged[i+1].first;
        out.push_back(ranks.at(std::vector<std::uint8_t>(piece.begin()+s, piece.begin()+e)));
    }
    return out;
}

std::vector<std::vector<std::uint8_t>> bytePairSplit(const std::vector<std::uint8_t>& piece, 
                                   const ByteVecRankMap& ranks){
    auto merged = bytePairMerge(ranks, piece);
    std::vector<std::vector<std::uint8_t>> out;
    out.reserve(merged.size()-1);
    for (std::size_t i = 0; i+1< merged.size(); ++i) {
        std::size_t s = merged[i].first, e = merged[i+1].first;
        out.emplace_back(piece.begin()+s, piece.begin()+e);
    }
    return out;
}

std::vector<std::vector<std::uint8_t>> bytePairSplit(std::string& s,
                                   const ByteVecRankMap& ranks)
{
    std::vector<std::uint8_t> bytes(s.begin(), s.end());
    return bytePairSplit(bytes, ranks);
}

std::string CoreBPE::makeSpecialPattern(const std::unordered_map<std::string, std::uint32_t>& special) {
    static const std::string meta = R"([.^$|()\[\]{}*+?\\])";
    std::string pat;
    pat.reserve(special.size() * 10);
    bool first = true;
    for (auto const& kv : special) {
        if (!first) pat.push_back('|');
        first = false;
        // Escape each character in the token 
        for (char c : kv.first) {
            if (meta.find(c) != std::string::npos) 
                pat.push_back('\\');
            pat.push_back(c);
        }
    }
    return pat;
}

CoreBPE::CoreBPE()
    : encoder_(),
      specialEncoder_(),
      decoder_(),
      specialDecoder_(),
      pattern_(),
      specialPattern_(),
      sortedTokenBytes_()
{}

CoreBPE::CoreBPE(ByteVecRankMap encoder,
            std::unordered_map<std::string, std::uint32_t> specialEncoder, 
            const std::string& pattern) 
    : encoder_(std::move(encoder)),  
      specialEncoder_(std::move(specialEncoder)),
      decoder_(), specialDecoder_(),
      pattern_(pattern), specialPattern_(makeSpecialPattern(specialEncoder_)),
      sortedTokenBytes_() {
        
    for (auto& kv : encoder_) 
        decoder_.emplace(kv.second, kv.first);
    for (auto& kv : specialEncoder_) 
        specialDecoder_.emplace(kv.second, std::vector<std::uint8_t>(kv.first.begin(), kv.first.end()));

    for (const auto& kv : specialDecoder_) {
        std::string str(kv.second.begin(), kv.second.end());
        specialStringDecoder_.emplace(str, kv.first);
    }

    sortedTokenBytes_.reserve(encoder_.size());
    for (auto& kv : encoder_) sortedTokenBytes_.push_back(kv.first);
    std::sort(sortedTokenBytes_.begin(), sortedTokenBytes_.end());


}

std::optional<std::vector<std::uint8_t>>
CoreBPE::decodeBytes(const std::vector<std::uint32_t>& tokens) const {
    std::vector<std::uint8_t> out;
    out.reserve(tokens.size() * 2);

    for (std::uint32_t t : tokens) {
        const std::vector<std::uint8_t>* tokenBytes = nullptr;

        auto it = decoder_.find(t);
        if (it != decoder_.end()) {
            tokenBytes = &it->second;
        } else {
            auto sit = specialDecoder_.find(t);
            if (sit != specialDecoder_.end()) {
                tokenBytes = &sit->second;
            } else {
                return std::nullopt;
            }
        }
        out.insert(out.end(), tokenBytes->begin(), tokenBytes->end());
    }

    return out;
}

std::vector<uint8_t> CoreBPE::decodeSingleTokenBytes(const std::uint32_t token) const {
    auto it = decoder_.find(token);
    if (it != decoder_.end()) {
        return it->second;
    }
    auto it_spec = specialDecoder_.find(token);
    if (it_spec != specialDecoder_.end()) {
        return it_spec->second;
    } 
    CV_Error(cv::Error::StsError, "Error in decode single token");
}

std::vector<std::uint32_t> CoreBPE::encodeOrdinary(const std::string& txt) const {

    std::vector<std::string> regexes{ pattern_ };
    auto splits = unicode_regex_split(txt, regexes);

    std::vector<std::uint32_t> tokens;
    for (auto& subUtf8 : splits) {
        // std::cout << "[" << subUtf8 << "]" << std::endl; 
        std::vector<std::uint8_t> piece(subUtf8.begin(), subUtf8.end());
        auto it = encoder_.find(piece);
        if (it != encoder_.end()) {
            tokens.push_back(it->second);
        } else {
            auto subTokens = bytePairEncode(piece, encoder_);
            tokens.insert(tokens.end(),
                          subTokens.begin(), subTokens.end());
        }
    }
    return tokens;
}
/*
 * This function tokenizes input text by handling special tokens and applying Byte Pair Encoding (BPE)
 * to ordinary text segments. It searches for allowed special tokens, processes ordinary text with BPE,
 * and emits a sequence of token IDs along with the count of tokens in the final processed segment.
 *
 * The logic and structure of this function are closely modeled after the original Rust implementation
 * in OpenAI's tiktoken library, which can be found here:
 * https://github.com/openai/tiktoken/blob/4560a8896f5fb1d35c6f8fd6eee0399f9a1a27ca/src/lib.rs#L234-L288
 *
 */
std::pair<std::vector<std::uint32_t>, std::size_t>
CoreBPE::encode(const std::string& text,
                const std::unordered_set<std::string>& allowedSpecial) const
{
    std::vector<std::uint32_t> ret;
    std::size_t last_piece_token_len = 0;
    size_t start = 0;

    // Use unicode_regex_split to find all special token matches in the text
    std::vector<std::string> special_regexes = { specialPattern_ };
    std::vector<std::pair<size_t, size_t>> special_matches;

    // Find all special token matches and their positions
    {
        std::regex special_re(specialPattern_);
        auto words_begin = std::sregex_iterator(text.begin(), text.end(), special_re);
        auto words_end = std::sregex_iterator();
        for (auto it = words_begin; it != words_end; ++it) {
            std::string match_str = it->str();
            if (allowedSpecial.count(match_str)) {
                special_matches.emplace_back(it->position(), it->position() + match_str.size());
            }
        }
    }

    size_t match_idx = 0;
    while (start < text.size()) {
        // Find the next allowed special token
        size_t next_special_start = std::string::npos;
        size_t next_special_end = std::string::npos;
        std::string matched_special;
        while (match_idx < special_matches.size()) {
            size_t s = special_matches[match_idx].first;
            size_t e = special_matches[match_idx].second;
            std::string candidate = text.substr(s, e - s);
            if (allowedSpecial.count(candidate) && s >= start) {
                next_special_start = s;
                next_special_end = e;
                matched_special = candidate;
                break;
            }
            ++match_idx;
        }

        size_t end = (next_special_start != std::string::npos) ? next_special_start : text.size();

        // Tokenize the ordinary segment [start, end)
        if (end > start) {
            std::string segment = text.substr(start, end - start);
            std::vector<std::string> regexes = { pattern_ };
            auto splits = unicode_regex_split(segment, regexes);

            for (auto& subUtf8 : splits) {
                std::vector<std::uint8_t> piece(subUtf8.begin(), subUtf8.end());
                auto it = encoder_.find(piece);
                if (it != encoder_.end()) {
                    last_piece_token_len = 1;
                    ret.push_back(it->second);
                } else {
                    auto tokens = bytePairEncode(piece, encoder_);
                    last_piece_token_len = tokens.size();
                    ret.insert(ret.end(), tokens.begin(), tokens.end());
                }
            }
        }

        // If we found a special token, add it and advance
        if (next_special_start != std::string::npos) {
            std::uint32_t token = specialEncoder_.at(matched_special);
            ret.push_back(token);
            start = next_special_end;
            last_piece_token_len = 0;
            ++match_idx;
        } else {
            break;
        }
    }

    return { ret, last_piece_token_len };
}

std::uint32_t CoreBPE::encodeSingleToken(std::vector<uint8_t>& piece) const {
    auto it = encoder_.find(piece);
    if (it != encoder_.end()) {
        return it->second;
    }

    try {
        std::string piece_str(piece.begin(), piece.end());
        auto sit = specialEncoder_.find(piece_str);
        if (sit != specialEncoder_.end()) {
            return sit->second;
        }
    } catch (...) {
        CV_Error(cv::Error::StsError, "Failed to encode single token: not found in encoder or specialEncoder");
    }
    return -1;
}
CV__DNN_INLINE_NS_END
}}
