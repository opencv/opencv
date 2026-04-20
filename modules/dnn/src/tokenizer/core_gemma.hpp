// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2025, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef __OPENCV_DNN_TOKENIZER_CORE_GEMMA_HPP__
#define __OPENCV_DNN_TOKENIZER_CORE_GEMMA_HPP__

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <limits>
#include <cstdio>
#include <cstring>

namespace cv { namespace dnn {

static const std::string GEMMA_METASPACE = "\xe2\x96\x81";  // UTF-8 for ▁ (U+2581)

struct CoreGemmaBPE {
    std::unordered_map<std::string, int> pieceToId;
    std::vector<std::string> idToPiece;
    std::unordered_map<std::string, uint32_t> mergeRanks;  // key = piece_a + '\0' + piece_b
    std::unordered_map<std::string, int> specialToId;
    std::unordered_map<int, std::string> idToSpecial;

    void addMerge(const std::string& a, const std::string& b, uint32_t rank) {
        mergeRanks[a + '\0' + b] = rank;
    }

    static std::string normalize(const std::string& text) {
        std::string out;
        out.reserve(text.size() + text.size() / 4);
        for (char c : text) {
            if (c == ' ')
                out += GEMMA_METASPACE;
            else
                out += c;
        }
        return out;
    }

    static std::vector<std::string> splitUtf8Chars(const std::string& text) {
        std::vector<std::string> out;
        out.reserve(text.size());
        size_t i = 0;
        while (i < text.size()) {
            unsigned char c = static_cast<unsigned char>(text[i]);
            int len;
            if      (c < 0x80) len = 1;
            else if (c < 0xE0) len = 2;
            else if (c < 0xF0) len = 3;
            else               len = 4;
            if (i + static_cast<size_t>(len) > text.size()) len = 1;
            out.push_back(text.substr(i, len));
            i += static_cast<size_t>(len);
        }
        return out;
    }

    static std::vector<std::string> byteFallback(const std::string& ch) {
        std::vector<std::string> out;
        out.reserve(ch.size());
        char buf[8];
        for (unsigned char b : ch) {
            std::snprintf(buf, sizeof(buf), "<0x%02X>", static_cast<unsigned>(b));
            out.push_back(buf);
        }
        return out;
    }

    std::vector<int> encodePiece(const std::string& text) const {
        if (text.empty()) return {};

        std::vector<std::string> pieces;
        for (const std::string& ch : splitUtf8Chars(text)) {
            if (pieceToId.count(ch)) {
                pieces.push_back(ch);
            } else {
                for (const std::string& fb : byteFallback(ch))
                    pieces.push_back(fb);
            }
        }

        static constexpr uint32_t RANK_INF = std::numeric_limits<uint32_t>::max();
        while (pieces.size() > 1) {
            uint32_t best_rank = RANK_INF;
            int best_i = -1;

            for (int i = 0; i + 1 < static_cast<int>(pieces.size()); ++i) {
                std::string key;
                key.reserve(pieces[i].size() + 1 + pieces[i + 1].size());
                key = pieces[i];
                key += '\0';
                key += pieces[i + 1];

                auto it = mergeRanks.find(key);
                if (it != mergeRanks.end() && it->second < best_rank) {
                    best_rank = it->second;
                    best_i = i;
                }
            }

            if (best_i == -1) break;

            pieces[best_i] += pieces[best_i + 1];
            pieces.erase(pieces.begin() + best_i + 1);
        }

        std::vector<int> ids;
        ids.reserve(pieces.size());
        for (const std::string& p : pieces) {
            auto it = pieceToId.find(p);
            if (it != pieceToId.end()) {
                ids.push_back(it->second);
            } else {
                for (const std::string& fb : byteFallback(p)) {
                    auto it2 = pieceToId.find(fb);
                    if (it2 != pieceToId.end())
                        ids.push_back(it2->second);
                }
            }
        }
        return ids;
    }

    std::vector<int> encode(const std::string& text,
                            const std::unordered_set<std::string>& allowedSpecial = {}) const {
        std::vector<int> result;
        size_t pos = 0;

        while (pos < text.size()) {
            bool foundSpecial = false;
            for (const auto& kv : specialToId) {
                const std::string& sp = kv.first;
                if (!allowedSpecial.count(sp)) continue;
                if (text.compare(pos, sp.size(), sp) == 0) {
                    result.push_back(kv.second);
                    pos += sp.size();
                    foundSpecial = true;
                    break;
                }
            }
            if (foundSpecial) continue;

            size_t chunkStart = pos;
            while (pos < text.size()) {
                bool atSpecial = false;
                for (const auto& kv : specialToId) {
                    if (allowedSpecial.count(kv.first) &&
                        text.compare(pos, kv.first.size(), kv.first) == 0) {
                        atSpecial = true;
                        break;
                    }
                }
                if (atSpecial) break;
                ++pos;
            }

            std::string chunk = text.substr(chunkStart, pos - chunkStart);
            std::string norm = normalize(chunk);
            auto ids = encodePiece(norm);
            result.insert(result.end(), ids.begin(), ids.end());
        }

        return result;
    }

    std::string decode(const std::vector<int>& ids) const {
        std::string raw;
        raw.reserve(ids.size() * 4);
        for (int id : ids) {
            auto sp = idToSpecial.find(id);
            if (sp != idToSpecial.end()) {
                raw += sp->second;
                continue;
            }
            if (id >= 0 && id < static_cast<int>(idToPiece.size()))
                raw += idToPiece[id];
        }

        std::string out;
        out.reserve(raw.size());
        size_t i = 0;
        while (i < raw.size()) {
            if (raw[i] == '<' && i + 5 < raw.size() &&
                raw[i + 1] == '0' && raw[i + 2] == 'x' &&
                raw[i + 5] == '>') {
                char hi = raw[i + 3], lo = raw[i + 4];
                auto hexVal = [](char c) -> int {
                    if (c >= '0' && c <= '9') return c - '0';
                    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
                    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
                    return -1;
                };
                int hv = hexVal(hi), lv = hexVal(lo);
                if (hv >= 0 && lv >= 0) {
                    out += static_cast<char>((hv << 4) | lv);
                    i += 6;
                    continue;
                }
            }
            if (i + 2 < raw.size() &&
                static_cast<unsigned char>(raw[i])     == 0xE2 &&
                static_cast<unsigned char>(raw[i + 1]) == 0x96 &&
                static_cast<unsigned char>(raw[i + 2]) == 0x81) {
                out += ' ';
                i += 3;
                continue;
            }
            out += raw[i++];
        }

        if (!out.empty() && out[0] == ' ')
            out.erase(0, 1);

        return out;
    }
};

}}  // namespace cv::dnn
#endif  // __OPENCV_DNN_TOKENIZER_CORE_GEMMA_HPP__
