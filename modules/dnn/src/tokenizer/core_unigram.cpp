// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "unicode.hpp"
#include "core_unigram.hpp"

#include <algorithm>
#include <cstring>
#include <limits>

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

UnigramPrecompiledNormalizer::UnigramPrecompiledNormalizer(std::string blob)
    : blob_(std::move(blob))
{
    if (blob_.size() >= 4) {
        uint32_t rawTrieSize = 0;
        std::memcpy(&rawTrieSize, blob_.data(), sizeof(rawTrieSize));
        // Untrusted size; reject if trie doesn't fit, avoid wraparound.
        if ((size_t)rawTrieSize <= blob_.size() - 4)
            trieSize_ = rawTrieSize;
    }
}

uint32_t UnigramPrecompiledNormalizer::getNode(uint32_t index) const {
    size_t count = trieSize_ / 4;
    if (index >= count) return 0;
    const uint8_t* base = reinterpret_cast<const uint8_t*>(blob_.data()) + 4 + (size_t)index * 4;
    uint32_t v;
    std::memcpy(&v, base, sizeof(v));
    return v;
}

uint32_t UnigramPrecompiledNormalizer::getBase(uint32_t index) const {
    uint32_t node = getNode(index);
    return (node >> 10) << ((node & (1u << 9)) >> 6);
}

uint32_t UnigramPrecompiledNormalizer::getLcheck(uint32_t index) const {
    return getNode(index) & ((1u << 31) | 0xffu);
}

bool UnigramPrecompiledNormalizer::getLeaf(uint32_t index) const {
    return ((getNode(index) >> 8) & 1u) != 0;
}

uint32_t UnigramPrecompiledNormalizer::getValue(uint32_t index) const {
    return getNode(index) & ((1u << 31) - 1);
}

void UnigramPrecompiledNormalizer::normalizePrefix(const std::string& text, size_t offset,
                                                    size_t& consumed, std::string& replacement) const {
    size_t longestLen = 0;
    uint32_t longestOff = 0;
    if (!empty() && 4 + 0u <= blob_.size()) {
        uint32_t node = getBase(0);
        size_t pos = offset;
        while (pos < text.size()) {
            unsigned char c = (unsigned char)text[pos];
            node ^= c;
            if (getLcheck(node) != c) break;
            bool leaf = getLeaf(node);
            node ^= getBase(node);
            if (leaf) {
                longestLen = pos - offset + 1;
                longestOff = getValue(node);
            }
            ++pos;
        }
    }
    size_t stringsStart = 4 + (size_t)trieSize_;
    if (longestLen > 0 && stringsStart + (size_t)longestOff < blob_.size()) {
        consumed = longestLen;
        replacement.assign(blob_.data() + stringsStart + longestOff);
        return;
    }
    size_t off2 = offset;
    uint32_t cpt = unicode_cpt_from_utf8(text, off2);
    if (off2 <= offset) off2 = offset + 1;
    if (cpt == 0 && off2 == offset + 1 && (unsigned char)text[offset] >= 0x80) {
        consumed = 1;
        replacement = "\xEF\xBF\xBD";
        return;
    }
    consumed = off2 - offset;
    replacement = text.substr(offset, consumed);
}

std::string UnigramPrecompiledNormalizer::normalize(const std::string& text) const {
    if (empty()) return text;
    std::string out;
    out.reserve(text.size() + 16);
    size_t offset = 0;
    while (offset < text.size()) {
        size_t consumed = 0;
        std::string rep;
        normalizePrefix(text, offset, consumed, rep);
        out += rep;
        offset += consumed;
    }
    return out;
}

CoreUnigram::CoreUnigram(const std::vector<std::pair<std::string, float>>& vocab,
                          int unkId,
                          UnigramPrecompiledNormalizer normalizer,
                          const std::unordered_map<std::string, int>& specialToId,
                          int eosId)
    : specialToId_(specialToId),
      unkId_(unkId),
      eosId_(eosId),
      normalizer_(std::move(normalizer))
{
    idToPiece_.reserve(vocab.size());
    idToScore_.reserve(vocab.size());
    for (size_t id = 0; id < vocab.size(); ++id) {
        idToPiece_.push_back(vocab[id].first);
        idToScore_.push_back(vocab[id].second);
        pieceToId_[vocab[id].first] = (int)id;
    }
    for (const auto& kv : specialToId_)
        idToSpecial_[kv.second] = kv.first;

    float minScore = 0.0f;
    bool any = false;
    for (size_t i = 0; i < idToScore_.size(); ++i) {
        if ((int)i == unkId_) continue;
        if (!any || idToScore_[i] < minScore) { minScore = idToScore_[i]; any = true; }
    }
    unkScore_ = minScore - 10.0f;

    maxPieceCps_ = 1;
    for (const auto& kv : pieceToId_) {
        maxPieceCps_ = std::max(maxPieceCps_, unicode_cpts_from_utf8(kv.first).size());
    }
}

void CoreUnigram::encodeChunk(const std::string& chunk, std::vector<int>& out) const {
    std::vector<size_t> offs;
    offs.reserve(chunk.size() + 1);
    {
        size_t p = 0;
        while (p < chunk.size()) {
            offs.push_back(p);
            size_t p2 = p;
            uint32_t cpt = unicode_cpt_from_utf8(chunk, p2);
            (void)cpt;
            if (p2 <= p) p2 = p + 1;
            p = p2;
        }
        offs.push_back(chunk.size());
    }
    size_t n = offs.size() - 1;
    if (n == 0) return;

    const float NEG_INF = -std::numeric_limits<float>::infinity();
    std::vector<float> best(n + 1, NEG_INF);
    std::vector<int> backPos(n + 1, -1);
    std::vector<int> backId(n + 1, -1);
    std::vector<char> backIsUnk(n + 1, 0);
    best[0] = 0.0f;

    for (size_t i = 0; i < n; ++i) {
        if (best[i] == NEG_INF) continue;

        {
            float sc = best[i] + unkScore_;
            if (sc > best[i + 1]) {
                best[i + 1] = sc;
                backPos[i + 1] = (int)i;
                backIsUnk[i + 1] = 1;
                backId[i + 1] = unkId_;
            }
        }

        size_t maxLen = std::min(maxPieceCps_, n - i);
        for (size_t len = 1; len <= maxLen; ++len) {
            size_t j = i + len;
            std::string sub = chunk.substr(offs[i], offs[j] - offs[i]);
            auto it = pieceToId_.find(sub);
            if (it == pieceToId_.end()) continue;
            int id = it->second;
            float sc = best[i] + idToScore_[(size_t)id];
            if (sc > best[j]) {
                best[j] = sc;
                backPos[j] = (int)i;
                backIsUnk[j] = 0;
                backId[j] = id;
            }
        }
    }

    std::vector<std::pair<int, bool>> segs;
    size_t cur = n;
    while (cur > 0) {
        segs.emplace_back(backId[cur], backIsUnk[cur] != 0);
        cur = (size_t)backPos[cur];
    }
    std::reverse(segs.begin(), segs.end());

    for (const auto& s : segs) {
        if (s.second) {
            // unkId_<0 means drop uncovered span, not emit invalid id.
            if (unkId_ < 0) continue;
            if (!out.empty() && out.back() == unkId_) continue;
            out.push_back(unkId_);
        } else {
            out.push_back(s.first);
        }
    }
}

void CoreUnigram::pretokenizeAndEncode(const std::string& normalized, std::vector<int>& out) const {
    std::vector<size_t> offs;
    std::vector<uint32_t> cps;
    offs.reserve(normalized.size() + 1);
    cps.reserve(normalized.size());
    size_t p = 0;
    while (p < normalized.size()) {
        offs.push_back(p);
        size_t p2 = p;
        uint32_t cpt = unicode_cpt_from_utf8(normalized, p2);
        if (p2 <= p) { p2 = p + 1; cpt = 0xFFFD; }
        cps.push_back(cpt);
        p = p2;
    }
    offs.push_back(normalized.size());
    size_t n = cps.size();

    size_t i = 0;
    while (i < n) {
        while (i < n && unicode_cpt_flags_from_cpt(cps[i]).is_whitespace) ++i;
        if (i >= n) break;
        size_t start = i;
        while (i < n && !unicode_cpt_flags_from_cpt(cps[i]).is_whitespace) ++i;
        std::string word = normalized.substr(offs[start], offs[i] - offs[start]);
        std::string piece = UNIGRAM_METASPACE + word;
        encodeChunk(piece, out);
    }
}

std::vector<int> CoreUnigram::encode(const std::string& text, const std::unordered_set<std::string>& allowedSpecial) const {
    std::vector<int> ids;
    size_t chunkStart = 0;
    size_t pos = 0;
    while (pos < text.size()) {
        std::string matched;
        int matchedId = -1;
        if (!allowedSpecial.empty()) {
            for (const auto& kv : specialToId_) {
                const std::string& sp = kv.first;
                bool isCandidate = !sp.empty() &&
                    pos + sp.size() <= text.size() &&
                    allowedSpecial.find(sp) != allowedSpecial.end() &&
                    text.compare(pos, sp.size(), sp) == 0;
                if (isCandidate && sp.size() > matched.size()) { matched = sp; matchedId = kv.second; }
            }
        }
        if (matchedId >= 0) {
            if (pos > chunkStart) {
                std::string norm = normalizer_.normalize(text.substr(chunkStart, pos - chunkStart));
                pretokenizeAndEncode(norm, ids);
            }
            ids.push_back(matchedId);
            pos += matched.size();
            chunkStart = pos;
        } else {
            ++pos;
        }
    }
    if (chunkStart < text.size()) {
        std::string norm = normalizer_.normalize(text.substr(chunkStart));
        pretokenizeAndEncode(norm, ids);
    }
    if (eosId_ >= 0) ids.push_back(eosId_);
    return ids;
}

std::string CoreUnigram::decode(const std::vector<int>& tokens) const {
    std::string raw;
    for (int id : tokens) {
        if (idToSpecial_.find(id) != idToSpecial_.end()) continue;
        if (id >= 0 && (size_t)id < idToPiece_.size())
            raw += idToPiece_[(size_t)id];
    }
    std::string result;
    result.reserve(raw.size());
    size_t prev = 0;
    size_t pos;
    while ((pos = raw.find(UNIGRAM_METASPACE, prev)) != std::string::npos) {
        result.append(raw, prev, pos - prev);
        result += ' ';
        prev = pos + UNIGRAM_METASPACE.size();
    }
    result.append(raw, prev, raw.size() - prev);
    if (!result.empty() && result.front() == ' ')
        result.erase(result.begin());
    return result;
}

CV__DNN_INLINE_NS_END
}}
