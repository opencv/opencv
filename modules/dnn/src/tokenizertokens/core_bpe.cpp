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

    auto pat = compilePattern(pattern); 
    std::shared_ptr<icu::RegexPattern> sharedPat{ pat.release() };
    regexTLS_.assign(MAX_NUM_THREADS, sharedPat);

    std::string speicalPat = makeSpecialPattern(specialEncoder_);
    auto spPat = compilePattern(speicalPat);
    std::shared_ptr<icu::RegexPattern> sharedSp{ spPat.release() };
    specialRegexTLS_.assign(MAX_NUM_THREADS, sharedSp);

    sortedTokenBytes_.reserve(encoder_.size());
    for (auto& kv : encoder_) sortedTokenBytes_.push_back(kv.first);
    std::sort(sortedTokenBytes_.begin(), sortedTokenBytes_.end());
}


const icu::RegexPattern* CoreBPE::threadLocalRegex() const {
    return regexTLS_[hashCurrentThread()%MAX_NUM_THREADS].get();
}

const icu::RegexPattern* CoreBPE::threadLocalSpecialRegex() const {
    return specialRegexTLS_[hashCurrentThread()%MAX_NUM_THREADS].get();
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


std::vector<Rank> CoreBPE::encodeOrdinary(const std::string& txt) const {

    icu::UnicodeString utext = icu::UnicodeString::fromUTF8(txt);
    UErrorCode status = U_ZERO_ERROR;
    auto matcher = threadLocalRegex()->matcher(utext, status);
    if (U_FAILURE(status)) {
        CV_Error(cv::Error::StsError, 
            "ICU matcher creation failed: " + std::string(u_errorName(status)));
    }

    std::vector<Rank> tokens;
    while (matcher->find(status) && U_SUCCESS(status)) {
        // 4) Extract the match as UTF-8 bytes
        icu::UnicodeString usub = matcher->group(status);
        std::string subUtf8;
        usub.toUTF8String(subUtf8);

        // 5) Turn that into a ByteVec (vector<uint8_t>)
        ByteVec piece(subUtf8.begin(), subUtf8.end());

        // 6) Look up in the encoder map
        auto it = encoder_.find(piece);
        if (it != encoder_.end()) {
            tokens.push_back(it->second);
        } else {
            // fallback to byte-pair encode
            auto subTokens = bytePairEncode(piece, encoder_);
            tokens.insert(tokens.end(),
                          subTokens.begin(), subTokens.end());
        }
    }
    if (U_FAILURE(status)) {
        CV_Error(cv::Error::StsError,
                 "Error during ICU regex matching: " + std::string(u_errorName(status)));
    }

    return tokens;
}



std::pair<std::vector<Rank>, std::size_t>
CoreBPE::encode(const std::string& text,
                const std::unordered_set<std::string>& allowedSpecial) const
{
    icu::UnicodeString utext = icu::UnicodeString::fromUTF8(text);
    int32_t ulen = utext.length();

    const icu::RegexPattern* mainPat    = threadLocalRegex();
    const icu::RegexPattern* specialPat = threadLocalSpecialRegex();

    std::vector<Rank>      ret;
    std::size_t            lastPieceTokenLen = 0;
    int32_t                startCU = 0;  

    UErrorCode status = U_ZERO_ERROR;
    while (startCU < ulen) {
        int32_t matchStartCU = -1, matchEndCU = -1;
        std::string matchedUTF8;

        {
            auto m = specialPat->matcher(utext, status);
            if (U_FAILURE(status)) {
                CV_Error(cv::Error::StsError,
                         "ICU special matcher creation failed: " + std::string(u_errorName(status)));
            }
            m->region(startCU, ulen, status);
            while (m->find(status) && U_SUCCESS(status)) {
                int32_t s = m->start(status);
                int32_t e = m->end(status);
                icu::UnicodeString slice = m->group(status);
                slice.toUTF8String(matchedUTF8);
                if (allowedSpecial.count(matchedUTF8)) {
                    matchStartCU = s;
                    matchEndCU   = e;
                    break;
                }
                m->region(s + 1, ulen, status);
            }
        }

        int32_t segmentEndCU = (matchStartCU >= 0 ? matchStartCU : ulen);
        if (segmentEndCU > startCU) {
            icu::UnicodeString seg = utext.tempSubStringBetween(startCU, segmentEndCU);
            auto m2 = mainPat->matcher(seg, status);
            if (U_FAILURE(status)) {
                CV_Error(cv::Error::StsError,
                         "ICU main matcher creation failed: " + std::string(u_errorName(status)));
            }

            while (m2->find(status) && U_SUCCESS(status)) {
                icu::UnicodeString subUS = m2->group(status);
                std::string        sub8;
                subUS.toUTF8String(sub8);

                ByteVec piece(sub8.begin(), sub8.end());
                auto it = encoder_.find(piece);
                if (it != encoder_.end()) {
                    ret.push_back(it->second);
                    lastPieceTokenLen = 1;
                } else {
                    auto subToks = bytePairEncode(piece, encoder_);
                    lastPieceTokenLen = subToks.size();
                    ret.insert(ret.end(), subToks.begin(), subToks.end());
                }
            }
        }

        if (matchStartCU >= 0) {
            Rank tok = specialEncoder_.at(matchedUTF8);
            ret.push_back(tok);
            startCU = matchEndCU;
            lastPieceTokenLen = 0;
        }
        else {
            break;
        }
    }

    if (U_FAILURE(status)) {
        CV_Error(cv::Error::StsError,
                 "Error during ICU regex matching: " + std::string(u_errorName(status)));
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
