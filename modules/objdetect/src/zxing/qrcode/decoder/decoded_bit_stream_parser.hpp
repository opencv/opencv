// -*- mode:c++; tab-width:2; indent-tabs-mode:nil; c-basic-offset:2 -*-

#ifndef __DECODED_BIT_STREAM_PARSER_H__
#define __DECODED_BIT_STREAM_PARSER_H__

/*
 *  DecodedBitStreamParser.hpp
 *  zxing
 *
 *  Copyright 2010 ZXing authors All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http:// www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#include "mode.hpp"
#include "../../common/bit_source.hpp"
#include "../../common/counted.hpp"
#include "../../common/array.hpp"
#include "../../common/decoder_result.hpp"
#include "../../common/character_set_eci.hpp"
#include "../../decode_hints.hpp"
#include "../../error_handler.hpp"

#include <string>
#include <sstream>
#include <map>

namespace zxing {
namespace qrcode {

class DecodedBitStreamParser {
public:
    DecodedBitStreamParser() : outputCharset("UTF-8") {}
    typedef std::map<DecodeHintType, std::string> Hashtable;
    
private:
    static char const ALPHANUMERIC_CHARS[];
    
    std::string outputCharset;
    
    char toAlphaNumericChar(size_t value, ErrorHandler & err_handler);
    
    void decodeHanziSegment(Ref<BitSource> bits, std::string &result, int count, ErrorHandler & err_handler);
    void decodeKanjiSegment(Ref<BitSource> bits, std::string &result, int count, ErrorHandler & err_handler);
    void decodeByteSegment(Ref<BitSource> bits, std::string &result, int count);
    void decodeByteSegment(Ref<BitSource> bits_,
                           std::string& result,
                           int count,
                           zxing::common::CharacterSetECI* currentCharacterSetECI,
                           ArrayRef< ArrayRef<char> >& byteSegments,
                           Hashtable const& hints,
                           ErrorHandler & err_handler);
    void decodeAlphanumericSegment(Ref<BitSource> bits, std::string &result, int count, bool fc1InEffect, ErrorHandler & err_handler);
    void decodeNumericSegment(Ref<BitSource> bits, std::string &result, int count, ErrorHandler & err_handler);
    
    void append(std::string &ost, const char *bufIn, size_t nIn,  const char* src, ErrorHandler & err_handler);
    void append(std::string &ost, std::string const& in, const char* src, ErrorHandler & err_handler);
    
public:
    Ref<DecoderResult> decode(ArrayRef<char> bytes,
                              Version *version,
                              ErrorCorrectionLevel const& ecLevel,
                              Hashtable const& hints,
                              ErrorHandler & err_handler,
                              int iVersion = -1);
};

}  // namespace qrcode
}  // namespace zxing

#endif  // QBAR_AI_QBAR_ZXING_QRCODE_DECODER_DECODEDBITSTREAMPARSER_H_
