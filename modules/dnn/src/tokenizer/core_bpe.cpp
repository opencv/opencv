/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  Portions of this file are inspired by or adapted from the tiktoken Rust
//  implementation:
//      https://github.com/openai/tiktoken/blob/main/src/lib.rs
//
//  This file is part of the OpenCV DNN module for tokenization.
//
////////////////////////////////////////////////////////////////////////////////////////*/

/*M///////////////////////////////////////////////////////////////////////////////////////
// MIT License
//
// Copyright (c) 2022 OpenAI, Shantanu Jain
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
#include "core_bpe.hpp"
#include "unicode.hpp"
#include "utils.hpp"

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

namespace cv { namespace dnn { namespace tokenizer {

static constexpr Rank RANK_MAX  = std::numeric_limits<Rank>::max();
static constexpr std::size_t SZ_MAX = std::numeric_limits<std::size_t>::max();

static Rank maybeGetRank(const ByteVecRankMap& ranks, const ByteVec& key) {
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
std::vector<std::pair<std::size_t, Rank>> bytePairMerge(const ByteVecRankMap& ranks, 
                                                        const ByteVec& piece) {
    std::vector<std::pair<std::size_t, Rank>> parts;
    parts.reserve(piece.size() + 1);

    std::pair<Rank, std::size_t> minRank{RANK_MAX, SZ_MAX};
    
    for (std::size_t i = 0; i+1 < piece.size(); ++i) {
        ByteVec key(piece.begin() + i, piece.begin() + i + 2);
        Rank r = maybeGetRank(ranks, key);
        if (r < minRank.first) minRank = {r, i};
        parts.emplace_back(i, r);
    } 
    parts.emplace_back(piece.size() - 1, RANK_MAX);
    parts.emplace_back(piece.size(), RANK_MAX);

    auto getRank = [&](const std::vector<std::pair<std::size_t, Rank>>& p, 
                       std::size_t idx) -> Rank {
        if (idx + 3 < p.size()) {
            std::size_t s = p[idx].first;
            std::size_t e = p[idx+3].first;
            ByteVec key(piece.begin()+s, piece.begin()+e);
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
            Rank r = parts[j].second;
            if (r < minRank.first) minRank = {r, j};
        }
    }
    return parts;
}

std::vector<Rank> bytePairEncode(const ByteVec& piece, 
                                 const ByteVecRankMap& ranks) {

    if (piece.size() == 1) {
        auto it = ranks.find(piece);
        return it == ranks.end() ? std::vector<Rank>{} : std::vector<Rank>{it->second};
    }
    auto merged = bytePairMerge(ranks, piece);
    std::vector<Rank> out; 
    out.reserve(merged.size()-1);

    for (std::size_t i = 0; i+1 <  merged.size(); ++i) {
        std::size_t s = merged[i].first, e = merged[i+1].first;
        out.push_back(ranks.at(ByteVec(piece.begin()+s, piece.begin()+e)));
    }
    return out;
}

std::vector<ByteVec> bytePairSplit(const ByteVec& piece, 
                                   const ByteVecRankMap& ranks){
    auto merged = bytePairMerge(ranks, piece);
    std::vector<ByteVec> out;
    out.reserve(merged.size()-1);
    for (std::size_t i = 0; i+1< merged.size(); ++i) {
        std::size_t s = merged[i].first, e = merged[i+1].first;
        out.emplace_back(piece.begin()+s, piece.begin()+e);
    }
    return out;
}

std::vector<ByteVec> bytePairSplit(std::string& s,
                                   const ByteVecRankMap& ranks)
{
    ByteVec bytes(s.begin(), s.end());
    return bytePairSplit(bytes, ranks);
}

std::string CoreBPE::makeSpecialPattern(const std::unordered_map<std::string, Rank>& special) {
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

DecoderKeyError::DecoderKeyError(Rank t)
    : std::runtime_error("Invalid token for decoding: " + std::to_string(t)), token_(t) {}

DecodeError::DecodeError(std::string msg) 
    : std::runtime_error("Could not deocde tokens: " + std::move(msg)) {}

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
            std::unordered_map<std::string, Rank> specialEncoder, 
            const std::string& pattern) 
    : encoder_(std::move(encoder)),  
      specialEncoder_(std::move(specialEncoder)),
      decoder_(), specialDecoder_(),
      pattern_(pattern), specialPattern_(makeSpecialPattern(specialEncoder_)),
      sortedTokenBytes_() {
        
    for (auto& kv : encoder_) 
        decoder_.emplace(kv.second, kv.first);
    for (auto& kv : specialEncoder_) 
        specialDecoder_.emplace(kv.second, ByteVec(kv.first.begin(), kv.first.end()));

    for (const auto& kv : specialDecoder_) {
        std::string str(kv.second.begin(), kv.second.end());
        specialStringDecoder_.emplace(str, kv.first);
    }

    sortedTokenBytes_.reserve(encoder_.size());
    for (auto& kv : encoder_) sortedTokenBytes_.push_back(kv.first);
    std::sort(sortedTokenBytes_.begin(), sortedTokenBytes_.end());


}

std::optional<ByteVec>
CoreBPE::decodeBytes(const std::vector<Rank>& tokens) const {
    ByteVec out;
    out.reserve(tokens.size() * 2);

    for (Rank t : tokens) {
        const ByteVec* tokenBytes = nullptr;

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

std::vector<uint8_t> CoreBPE::decodeSingleTokenBytes(const Rank token) const {
    auto it = decoder_.find(token);
    if (it != decoder_.end()) {
        return it->second;
    }
    auto it_spec = specialDecoder_.find(token);
    if (it_spec != specialDecoder_.end()) {
        return it_spec->second;
    } 
    throw DecodeError(std::to_string(token));
}

std::vector<Rank> CoreBPE::encodeOrdinary(const std::string& txt) const {

    std::vector<std::string> regexes{ pattern_ };
    auto splits = unicode_regex_split(txt, regexes);

    std::vector<Rank> tokens;
    for (auto& subUtf8 : splits) {
        std::cout << "[" << subUtf8 << "]" << std::endl; 
        // ByteVec piece = textToBytes(subUtf8);
        ByteVec piece(subUtf8.begin(), subUtf8.end());
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
std::pair<std::vector<Rank>, std::size_t>
CoreBPE::encode(const std::string& text,
                const std::unordered_set<std::string>& allowedSpecial) const
{
    std::vector<Rank> ret;
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
                ByteVec piece(subUtf8.begin(), subUtf8.end());
                // ByteVec piece = textToBytes(subUtf8);
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
            Rank token = specialEncoder_.at(matched_special);
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

Rank CoreBPE::encodeSingleToken(std::vector<uint8_t>& piece) const {
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
        throw std::runtime_error("Failed to encode single token: not found in encoder or specialEncoder");
    }
}

std::pair<std::vector<Rank>, std::size_t> 
CoreBPE::increaseLastPieceTokenLen(std::vector<Rank> tokens,
                                   std::size_t lastPieceTokenLen) const {
    // check if a token's byte sequence is all whitespace
    auto tokenIsAllSpace = [&](Rank token) {
        auto it = decoder_.find(token);
        if (it!=decoder_.end()) {
            const ByteVec& bytes = it->second;
            return std::all_of(bytes.rbegin(), bytes.rend(), 
                                [](uint8_t b) {
                                    return b == ' ' || b == '\n' || b == '\t'; 
                                });
                                
        }
        return false;
    };

    if (lastPieceTokenLen > 0 && tokenIsAllSpace(tokens[tokens.size() - lastPieceTokenLen])) {
        // extend to include any preceding all-space tokens 
        while (lastPieceTokenLen < tokens.size() && 
                tokenIsAllSpace(tokens[tokens.size() - lastPieceTokenLen - 1])) {
            ++lastPieceTokenLen;
        }
    }

    // sanity check: cannot exceed token count
    assert(lastPieceTokenLen <= tokens.size());

    return { std::move(tokens), lastPieceTokenLen};
}


std::pair<std::vector<Rank>, std::set<std::vector<Rank>>> 
CoreBPE::encodeUnstableNative(const std::string& text, const std::unordered_set<std::string>& allowedSpecial) const {
    // encoding + unstable length 
    auto [tokens, lastPieceTokenLen] = encode(text, allowedSpecial);
    // if the last piece was a special token (no unstable bytes)
    if (lastPieceTokenLen == 0) {
        return { tokens, {} };
    }

    // extend unstable region 
    std::tie(tokens, lastPieceTokenLen) = increaseLastPieceTokenLen(tokens, lastPieceTokenLen);

    // Decode unstable byte slice
    std::vector<uint8_t> unstableBytes;
    if (auto opt = decodeBytes(
            std::vector<Rank>(tokens.end() - lastPieceTokenLen, tokens.end()))) {
        unstableBytes = *opt;
    } else {
        unstableBytes.clear();
    }

    // remove unstable tokens from end
    tokens.resize(tokens.size() - lastPieceTokenLen);

    std::set<std::vector<Rank>> completions;
    if (unstableBytes.empty()) {
        return { tokens, completions };
    }

    // Single token completions starting with unstable bytes
    auto cmp = [&](const ByteVec& b) {
        return b < unstableBytes;
    };
    auto it = std::partition_point(
        sortedTokenBytes_.begin(), sortedTokenBytes_.end(), cmp
    );
    for (; it!=sortedTokenBytes_.end() && 
            std::equal(unstableBytes.begin(), unstableBytes.end(), it->begin()); ++it) {
        // map bytes back to token id
        Rank tok = encoder_.at(*it);
        completions.insert({ tok });
    }

    // Brute-Force splitting unstable bytes
    for (std::size_t i=1; i < unstableBytes.size(); ++i) {
        ByteVec prefix(unstableBytes.begin(), unstableBytes.begin() + i);
        ByteVec suffix(unstableBytes.begin() + i, unstableBytes.end());
        // find suffix matches
        auto cmp2 = [&](const ByteVec& b) {return b < suffix; };
        auto it2 = std::partition_point(
            sortedTokenBytes_.begin(), sortedTokenBytes_.end(), cmp2);
        for (; it2!=sortedTokenBytes_.end() &&
                std::equal(suffix.begin(), suffix.end(), it2->begin()); ++it2) {
            // combine prefix + matching token bytes 
            ByteVec candidate = prefix;
            candidate.insert(candidate.end(), it2->begin(), it2->end());
            // try to reinterpret as UTF-8
            std::vector<Rank> encoded;
            try {
                std::string s(reinterpret_cast<char*>(candidate.data()), candidate.size());
                encoded = encodeOrdinary(s);
            } catch(...) {
                // fallback to byte-pair
                encoded = bytePairEncode(candidate, encoder_);
            }
            // truncate encoded to unstable length 
            std::vector<Rank> seq;
            std::size_t lenAcc = 0;
            for (Rank id : encoded) {
                seq.push_back(id);
                lenAcc += decoder_.at(id).size();
                if (lenAcc >= unstableBytes.size()) break;
            }
            completions.insert(seq);

        }
    }

    // whitespace regex fix for last code point 
    if (unstableBytes.size() > 1) {
        // TODO: Implement the decodeLastUtf8 [DONE] -> [TESTING PROCESS]
        auto [last_char, byte_len] = decodeLastUtf8(unstableBytes);
        if (unstableBytes.size() > byte_len && std::isspace(static_cast<unsigned char>(*last_char))) {
            // re-encode in two parts
            ByteVec part1(unstableBytes.begin(), unstableBytes.end() - byte_len);
            ByteVec part2(unstableBytes.end() - byte_len, unstableBytes.end());
            auto r1 = bytePairEncode(part1, encoder_);
            auto r2 = bytePairEncode(part2, encoder_);
            std::vector<Rank> merged;
            merged.insert(merged.end(), r1.begin(), r1.end());
            merged.insert(merged.end(), r2.begin(), r2.end());
            completions.insert(merged);
        }
    }

    return { tokens, completions };
}   


}}}
