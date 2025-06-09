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
}

// Errors for debugging 
DecoderKeyError::DecoderKeyError(Rank t)
    : std::runtime_error("Invalid token for decoding: " + std::to_string(t)), token_(t) {}

DecodeError::DecodeError(std::string msg) 
    : std::runtime_error("Could not deocde tokens: " + std::move(msg)) {}

CoreBPE::CoreBPE(ByteVecRankMap encoder,
            std::unordered_map<std::string, Rank> specialEncoder, 
            const std::string& pattern) 
    : encoder_(std::move(encoder)),  
      specialEncoder_(std::move(specialEncoder)),
      decoder_(), specialDecoder_(),
      regexTLS_(), specialRegexTLS_(),
      sortedTokenBytes_() {
        
    for (auto& kv : encoder_) 
        decoder_.emplace(kv.second, kv.first);
    for (auto& kv : specialEncoder_) 
        specialDecoder_.emplace(kv.second, ByteVec(kv.first.begin(), kv.first.end()));

    std::regex mainRe(pattern, std::regex::optimize|std::regex::ECMAScript);
    std::string spPattern;
    for (auto it=specialEncoder_.begin(); it!=specialEncoder.end(); ++it) {
        if (it!=specialEncoder_.begin()) spPattern += "|";
        spPattern += std::regex_replace(it->first, std::regex(R"([.^$|()\[\]{}*+?\\])"), R"(\\$&)");
    }
    std::regex spRe(spPattern, std::regex::optimize|std::regex::ECMAScript);

    regexTLS_.assign(MAX_NUM_THREADS, mainRe);
    specialRegexTLS_.assign(MAX_NUM_THREADS, spRe);

    sortedTokenBytes_.reserve(encoder_.size());
    for (auto& kv : encoder_) sortedTokenBytes_.push_back(kv.first);
    std::sort(sortedTokenBytes_.begin(), sortedTokenBytes_.end());
}

template<typename EncIter, typename SpecIter>
CoreBPE CoreBPE::create(EncIter ef, EncIter el, SpecIter sf, SpecIter sl, const std::string& pat) {
    ByteVecRankMap enc;
    std::unordered_map<std::string, Rank> spec;
    for (auto it=ef; it!=el; ++it) enc.emplace(it->first, it->second);
    for (auto it=sf; it!=sl; ++it) spec.emplace(it->first, it->second);
    return CoreBPE(std::move(enc), std::move(spec), pat);
}

template CoreBPE CoreBPE::create(
    std::vector<std::pair<ByteVec, Rank>>::const_iterator,
    std::vector<std::pair<ByteVec, Rank>>::const_iterator,
    std::vector<std::pair<std::string, Rank>>::const_iterator,
    std::vector<std::pair<std::string, Rank>>::const_iterator,
    const std::string&
);

const std::regex& CoreBPE::threadLocalRegex() const {
    return regexTLS_[hashCurrentThread()%MAX_NUM_THREADS];
}

const std::regex& CoreBPE::threadLocalSpecialRegex() const {
    return specialRegexTLS_[hashCurrentThread()*MAX_NUM_THREADS];
}


std::optional<ByteVec>
CoreBPE::decodeBytes(const std::vector<Rank>& tokens) const {
    ByteVec out;
    out.reserve(tokens.size() * /* avg bytes per token */ 4);

    for (Rank t : tokens) {
        auto it = decoder_.find(t);
        if (it == decoder_.end()) {
            // Unknown token ID → “none”
            return std::nullopt;
        }
        // append all bytes for this token
        const ByteVec& tokenBytes = it->second;
        out.insert(out.end(), tokenBytes.begin(), tokenBytes.end());
    }

    return out;
}


std::vector<Rank> CoreBPE::encodeOrdinary(const std::string& txt) const {
    const std::regex& re = threadLocalRegex();
    std::vector<Rank> tokens;
    for (auto it=std::sregex_iterator(txt.begin(), txt.end(), re); it!=std::sregex_iterator(); ++it) {
        std::string_view sv(it->str());
        ByteVec piece(sv.begin(), sv.end());
        auto eIt = encoder_.find(piece);
        if (eIt!=encoder_.end()) tokens.push_back(eIt->second);
        else {
            auto sub = bytePairEncode(piece, encoder_);
            tokens.insert(tokens.end(), sub.begin(), sub.end());
        }
    }
}

std::pair<std::vector<Rank>, std::size_t>
CoreBPE::encode(const std::string& text,
                const std::unordered_set<std::string>& allowedSpecial) const
{
    const std::regex& specialRe = threadLocalSpecialRegex();
    const std::regex& mainRe    = threadLocalRegex();

    std::vector<Rank> ret;
    std::size_t       lastPieceTokenLen = 0;
    std::size_t       start = 0;

    while (true) {
        // Find next allowed special token
        std::cmatch m;
        std::size_t nextPos = std::string::npos;
        std::size_t nextEnd = std::string::npos;
        std::string nextText;

        std::size_t searchPos = start;
        while (searchPos < text.size()) {
            const char* begin = text.data() + searchPos;
            const char* end   = text.data() + text.size();
            if (!std::regex_search(begin, end, m, specialRe)) break;

            std::size_t matchStart = searchPos + static_cast<std::size_t>(m.position());
            std::size_t matchEnd   = matchStart + static_cast<std::size_t>(m.length());
            std::string candidate  = text.substr(matchStart, matchEnd - matchStart);

            if (allowedSpecial.count(candidate)) {
                nextPos  = matchStart;
                nextEnd  = matchEnd;
                nextText = std::move(candidate);
                break;
            }
            searchPos = matchStart + 1;  // skip past rejected match
        }

        // Encode ordinary segment [start, segmentEnd)
        std::size_t segmentEnd = (nextPos == std::string::npos) ? text.size() : nextPos;
        if (segmentEnd > start) {
            std::string segment = text.substr(start, segmentEnd - start);
            for (std::sregex_iterator it(segment.begin(), segment.end(), mainRe), e;
                 it != e; ++it) {
                std::string_view sv(it->str());
                ByteVec piece(sv.begin(), sv.end());

                auto encIt = encoder_.find(piece);
                if (encIt != encoder_.end()) {
                    lastPieceTokenLen = 1;
                    ret.push_back(encIt->second);
                } else {
                    auto toks = bytePairEncode(piece, encoder_);
                    lastPieceTokenLen = toks.size();
                    ret.insert(ret.end(), toks.begin(), toks.end());
                }
            }
        }

        // If a special token was found, append it and continue; else break
        if (nextPos != std::string::npos) {
            Rank tok = specialEncoder_.at(nextText);
            ret.push_back(tok);
            start = nextEnd;
            lastPieceTokenLen = 0;
        } else {
            break; // no more specials
        }
    }

    return { ret, lastPieceTokenLen };
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
                // fallbacj to byte-pair
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
        // TODO: Implement the decodeLastUtf8
        auto [last_char, byte_len] = decodeLastUtf8(unstableBytes);
        if (unstableBytes.size() > byte_len && std::isspace(static_cast<unsigned char>(last_char))) {
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
