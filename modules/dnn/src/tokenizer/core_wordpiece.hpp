// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef __OPENCV_DNN_TOKENIZER_CORE_WORDPIECE_HPP__
#define __OPENCV_DNN_TOKENIZER_CORE_WORDPIECE_HPP__

#include "unicode.hpp"

#include <opencv2/core.hpp>
#include <opencv2/dnn/dnn.hpp>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

/**
 * @brief Core WordPiece engine: vocab lookup + greedy longest-match-first split.
 *
 * Splits one pre-tokenized word into subword ids, word -> ids only. No
 * [CLS]/[SEP]/segment-id assembly here -- that's WordPieceTokenizerImpl's
 * job (tokenizer.cpp).
 */
class CoreWordPiece {
public:
    CoreWordPiece() = default;

    /**
     * @brief Build a CoreWordPiece from a vocab map.
     *
     * Populates the piece<->id tables and @c maxPieceCps_ from @p vocab (via
     * addPiece()), then resolves @p unkToken in the resulting table to set
     * the unk id (defaulting to 0 if not present).
     *
     * @param vocab                     Map from piece text to vocab id.
     * @param unkToken                  Literal text of the unknown-token piece.
     * @param continuingSubwordPrefix   Prefix marking continuation pieces (e.g. "##").
     * @param maxInputCharsPerWord      Words longer than this (in codepoints) decode to unk.
     */
    CoreWordPiece(const std::unordered_map<std::string, int>& vocab,
                  const std::string& unkToken,
                  const std::string& continuingSubwordPrefix,
                  size_t maxInputCharsPerWord);

    std::vector<int> encode(const std::string& word) const;
    std::string decode(const std::vector<int>& tokens) const;

    /**
     * @brief Look up a literal vocab piece (e.g. "[CLS]"/"[SEP]") by exact string.
     * @return true and sets @p id if found, false (leaving @p id untouched) otherwise.
     */
    bool tryGetId(const std::string& piece, int& id) const;

private:
    void addPiece(const std::string& piece, int id);

    // Greedy longest-match-first split of one pre-tokenized word into ids.
    // All-or-nothing: on any no-match position, discards partial ids and
    // emits a single unkId_ for the whole word -- matches reference
    // BertTokenizer/WordpieceTokenizer, not a bug.
    void encodeWord(const std::string& word, std::vector<int>& out) const;

    std::unordered_map<std::string, int> pieceToId_;
    std::vector<std::string> idToPiece_;

    std::string unkToken_ = "[UNK]";
    int unkId_ = 0;
    std::string continuingSubwordPrefix_ = "##";
    size_t maxInputCharsPerWord_ = 100;
    size_t maxPieceCps_ = 0;
};

CV__DNN_INLINE_NS_END
}}
#endif
