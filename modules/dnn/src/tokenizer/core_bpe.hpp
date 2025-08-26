// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

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

#ifndef __OPENCV_DNN_TOKENIZER_CORE_BPE_HPP__
#define __OPENCV_DNN_TOKENIZER_CORE_BPE_HPP__

#include <opencv2/core.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <vector>
#include <unordered_map>
#include <unordered_set>

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN  
/** 
 * @brief Hasher for byte vectors to enable use in unordered_map.
 *
 * Computes a simple rolling hash over the bytes.
 *
 * @note Intended for internal use with Byte Pair Encoding (BPE) maps.
 */
struct ByteVecHash {
    std::size_t operator()(const std::vector<std::uint8_t>& v) const noexcept {
        std::size_t h = 0;
        for (auto b : v) h = h * 31u + static_cast<std::size_t>(b);
        return h;
    }
};

/** 
 * @brief Map from raw byte-sequence tokens to their merge rank / token id.
 *
 * Keys are byte sequences (not Unicode code points). Values are the
 * token ids/ranks used by the BPE encoder/decoder.
 */
using ByteVecRankMap = std::unordered_map<std::vector<std::uint8_t>, std::uint32_t, ByteVecHash>;

/** 
 * @brief Merge-adjacent byte pairs by increasing rank until no mergeable pair remains.
 *
 * Scans adjacent byte pairs in @p piece, repeatedly splicing out the minimal-rank
 * pair (highest merge priority) and updating neighboring ranks, until no pair
 * appears in @p ranks. Returns the final segmentation as a list of split
 * boundaries and their ranks.
 *
 * @param ranks  Map of mergeable byte pairs (key = 2+ byte token, value = rank/id).
 * @param piece  Input bytes for a single text span (UTF-8 already flattened to bytes).
 * @return Vector of (start_index, rank) pairs describing token boundaries after merging.
 *         The last element is a sentinel boundary at @c piece.size().
 *
 * @note This is the low-level merge routine used by BPE; it does not translate
 *       segments into ids. For that, see bytePairEncode().
 * @see bytePairEncode, bytePairSplit
 */
CV_EXPORTS std::vector<std::pair<std::size_t, std::uint32_t>> bytePairMerge(const ByteVecRankMap& ranks, 
                                                        const std::vector<std::uint8_t>& piece);

/** 
 * @brief Encode a byte sequence into token ids using BPE merge rules.
 *
 * If @p piece is a single byte present in @p ranks, returns that id directly.
 * Otherwise, runs the merge loop (bytePairMerge) and maps each resulting segment
 * to its id via @p ranks.
 *
 * @param piece  Input bytes (one text span already split by regex).
 * @param ranks  Map from byte-sequence tokens to ids (includes all singletons 0..255).
 * @return Token ids produced by BPE for the given @p piece.
 *
 * @see bytePairMerge, bytePairSplit
 */
CV_EXPORTS std::vector<std::uint32_t> bytePairEncode(const std::vector<std::uint8_t>& piece, 
                                 const ByteVecRankMap& ranks);

/** 
 * @brief Split a byte sequence into BPE token byte-spans (no id translation).
 *
 * Applies the same merge boundaries as bytePairEncode(), but returns the raw
 * byte segments instead of ids.
 *
 * @param piece  Input bytes.
 * @param ranks  Map from byte-sequence tokens to ids (used only to test mergeability).
 * @return Vector of byte slices corresponding to final BPE tokens.
 *
 * @see bytePairEncode
 */
CV_EXPORTS std::vector<std::vector<std::uint8_t>> bytePairSplit(const std::vector<std::uint8_t>& piece, 
                                   const ByteVecRankMap& ranks);

/** 
 * @overload
 * @brief Split a UTF-8 string into BPE token byte-spans (no id translation).
 *
 * Converts @p s to bytes and calls the byte-vector overload.
 *
 * @param s      UTF-8 string (will be copied to bytes).
 * @param ranks  Map used to determine mergeability.
 * @return Vector of byte slices corresponding to final BPE tokens.
 */
CV_EXPORTS std::vector<std::vector<std::uint8_t>> bytePairSplit(std::string& s,
                                   const ByteVecRankMap& ranks);

/** 
 * @brief Core Byte Pair Encoding (BPE) engine (mergeable-ranks model).
 *
 * Encodes and decodes tokens at the byte level (UTF-8 input is split to bytes),
 * with optional support for special tokens that are matched by a separate regex.
 *
 * The implementation follows the structure of OpenAI’s tiktoken encoders.
 */
class CV_EXPORTS CoreBPE {
public:
    CoreBPE(); 
    explicit CoreBPE(ByteVecRankMap encoder,
            std::unordered_map<std::string, std::uint32_t> specialEncoder, 
            const std::string& pattern);

    /** 
     * @brief Encode text with ordinary BPE (no special tokens).
     *
     * Splits @p text using @c pattern_ and applies BPE over each split.
     *
     * @param text  UTF-8 input.
     * @return Vector of token ids.
     */
    std::vector<std::uint32_t> encodeOrdinary(const std::string& text) const;

    /** 
     * @brief Encode text with optional special tokens.
     *
     * Scans @p text for allowed special tokens, emits them as single ids,
     * and BPE-encodes the intervening ordinary segments.
     *
     * @param text            UTF-8 input.
     * @param allowedSpecial  Set of literal special-token strings that may appear and be emitted.
     * @return Pair @c (tokens, last_piece_token_len) where:
     *         - @c tokens is the full token sequence,
     *         - @c last_piece_token_len is the number of tokens produced by the final ordinary segment
     *           (0 if the text ended with a special token).
     */
    std::pair<std::vector<std::uint32_t>, std::size_t> encode(const std::string& text,
                                                     const std::unordered_set<std::string>& allowedSpecial) const;
    std::uint32_t encodeSingleToken(std::vector<uint8_t>& piece) const;
    
     /** 
      * @brief Decode a sequence of token ids into raw bytes.
     *
     * Looks up ids in either the mergeable-token or special-token decoders.
     *
     * @param tokens  Token ids.
     * @return Decoded bytes on success, or @c std::nullopt if any id is unknown.
     */
    std::optional<std::vector<std::uint8_t>> decodeBytes(const std::vector<std::uint32_t>& tokens) const;
    std::vector<uint8_t> decodeSingleTokenBytes(const std::uint32_t token) const;
    
private:
    ByteVecRankMap encoder_;
    std::unordered_map<std::string, std::uint32_t> specialEncoder_;
    std::unordered_map<std::uint32_t, std::vector<std::uint8_t>>  decoder_;          
    std::unordered_map<std::uint32_t, std::vector<std::uint8_t>>  specialDecoder_;
    std::unordered_map<std::string, std::uint32_t>  specialStringDecoder_;   
    std::string pattern_;
    std::string specialPattern_;
    std::vector<std::vector<std::uint8_t>> sortedTokenBytes_;
    std::string makeSpecialPattern(const std::unordered_map<std::string, std::uint32_t>& special);
};
CV__DNN_INLINE_NS_END
}}

#endif