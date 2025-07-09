#include "core_bpe.hpp"
#include "unicode.hpp"

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

std::size_t hashCurrentThread() {
    return std::hash<std::thread::id>{}(std::this_thread::get_id());
}

static Rank maybeGetRank(const ByteVecRankMap& ranks, const ByteVec& key) {
    auto it = ranks.find(key);
    return it == ranks.end() ? RANK_MAX : it->second;
}

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

// Errors for debugging 
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

    // auto pat = compilePattern(pattern); 
    // std::shared_ptr<icu::RegexPattern> sharedPat{ pat.release() };
    // regexTLS_.assign(MAX_NUM_THREADS, sharedPat);

    // std::string speicalPat = makeSpecialPattern(specialEncoder_);
    // auto spPat = compilePattern(speicalPat);
    // std::shared_ptr<icu::RegexPattern> sharedSp{ spPat.release() };
    // specialRegexTLS_.assign(MAX_NUM_THREADS, sharedSp);

    sortedTokenBytes_.reserve(encoder_.size());
    for (auto& kv : encoder_) sortedTokenBytes_.push_back(kv.first);
    std::sort(sortedTokenBytes_.begin(), sortedTokenBytes_.end());
}


// const icu::RegexPattern* CoreBPE::threadLocalRegex() const {
//     return regexTLS_[hashCurrentThread()%MAX_NUM_THREADS].get();
// }

// const icu::RegexPattern* CoreBPE::threadLocalSpecialRegex() const {
//     return specialRegexTLS_[hashCurrentThread()%MAX_NUM_THREADS].get();
// }


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


std::vector<Rank> CoreBPE::encodeOrdinary(const std::string& txt) const {

    std::vector<std::string> regexes{ pattern_ };
    auto splits = unicode_regex_split(txt, regexes);

    std::vector<Rank> tokens;
    for (auto& subUtf8 : splits) {
        // Replace leading space with U+0120 (Ġ) as in OpenAI encoder.json
        // [TODO]: not sure if this should be the case for all 
        // cases of space in the json? 
        if (!subUtf8.empty() && subUtf8[0] == ' ') {
            subUtf8.replace(0, 1, "\xC4\xA0");
        }
        std::cout << "[" << subUtf8 << "]" << std::endl; 

        ByteVec piece(subUtf8.begin(), subUtf8.end());

        // for (const auto& kv : encoder_) {
        //     if (kv.first.size() > 2) {
        //         std::cout << "Token: ";
        //         for (auto b : kv.first) std::cout << std::hex << (int)b << " ";
        //         std::cout << " -> id: " << kv.second << std::endl;
        //     }
        // }

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
                if (!subUtf8.empty() && subUtf8[0] == ' ') {
                    subUtf8.replace(0, 1, "\xC4\xA0");
                }
                ByteVec piece(subUtf8.begin(), subUtf8.end());
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
