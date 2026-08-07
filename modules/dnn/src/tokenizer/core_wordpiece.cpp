// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "unicode.hpp"
#include "core_wordpiece.hpp"

#include <algorithm>

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

void CoreWordPiece::addPiece(const std::string& piece, int id) {
    if (id < 0)
        CV_Error(cv::Error::StsBadArg, "WordPiece vocab entry '" + piece + "' has a negative id: " + std::to_string(id));
    pieceToId_[piece] = id;
    if ((size_t)id >= idToPiece_.size())
        idToPiece_.resize(id + 1);
    idToPiece_[id] = piece;
    maxPieceCps_ = std::max(maxPieceCps_, unicode_cpts_from_utf8(piece).size());
}

CoreWordPiece::CoreWordPiece(const std::unordered_map<std::string, int>& vocab,
                              const std::string& unkToken,
                              const std::string& continuingSubwordPrefix,
                              size_t maxInputCharsPerWord)
    : unkToken_(unkToken),
      unkId_(0),
      continuingSubwordPrefix_(continuingSubwordPrefix),
      maxInputCharsPerWord_(maxInputCharsPerWord),
      maxPieceCps_(0)
{
    for (const auto& kv : vocab)
        addPiece(kv.first, kv.second);

    auto it = pieceToId_.find(unkToken_);
    unkId_ = (it != pieceToId_.end()) ? it->second : 0;
}

void CoreWordPiece::encodeWord(const std::string& word, std::vector<int>& out) const {
    std::vector<uint32_t> cps = unicode_cpts_from_utf8(word);
    if (cps.size() > maxInputCharsPerWord_) {
        out.push_back(unkId_);
        return;
    }

    // Byte offsets avoid cp-by-cp rebuild; keeps this from O(L^3).
    std::vector<size_t> byteOffset(cps.size() + 1);
    size_t off = 0;
    for (size_t i = 0; i < cps.size(); ++i) {
        byteOffset[i] = off;
        off += unicode_cpt_to_utf8(cps[i]).size();
    }
    byteOffset[cps.size()] = off;

    std::vector<int> sub;
    size_t start = 0;
    while (start < cps.size()) {
        size_t end = maxPieceCps_ > 0 ? std::min(cps.size(), start + maxPieceCps_) : cps.size();
        int matchedId = -1;
        while (end > start) {
            std::string substr = word.substr(byteOffset[start], byteOffset[end] - byteOffset[start]);
            std::string candidate = (start > 0) ? (continuingSubwordPrefix_ + substr) : substr;
            auto it = pieceToId_.find(candidate);
            if (it != pieceToId_.end()) {
                matchedId = it->second;
                break;
            }
            --end;
        }
        if (matchedId < 0) {
            out.push_back(unkId_);
            return;
        }
        sub.push_back(matchedId);
        start = end;
    }
    out.insert(out.end(), sub.begin(), sub.end());
}

std::vector<int> CoreWordPiece::encode(const std::string& word) const {
    std::vector<int> out;
    encodeWord(word, out);
    return out;
}

std::string CoreWordPiece::decode(const std::vector<int>& tokens) const {
    std::string result;
    for (int id : tokens) {
        if (id < 0 || (size_t)id >= idToPiece_.size())
            continue;
        const std::string& piece = idToPiece_[id];
        // Empty prefix would vacuously match every piece as continuation.
        bool isCont = !continuingSubwordPrefix_.empty() &&
                      piece.compare(0, continuingSubwordPrefix_.size(), continuingSubwordPrefix_) == 0;
        if (isCont) {
            result += piece.substr(continuingSubwordPrefix_.size());
        } else {
            if (!result.empty())
                result += ' ';
            result += piece;
        }
    }
    return result;
}

bool CoreWordPiece::tryGetId(const std::string& piece, int& id) const {
    auto it = pieceToId_.find(piece);
    if (it == pieceToId_.end())
        return false;
    id = it->second;
    return true;
}

CV__DNN_INLINE_NS_END
}}
