// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").

#ifndef __ZXING_DATAMATRIX_DECODER_DECODED_BIT_STREAM_PARSER_HPP__
#define __ZXING_DATAMATRIX_DECODER_DECODED_BIT_STREAM_PARSER_HPP__

/*
 *  DecodedBitStreamParser.hpp
 *  zxing
 *
 *  Created by Luiz Silva on 09/02/2010.
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


#include "../../common/array.hpp"
#include "../../common/bit_source.hpp"
#include "../../common/counted.hpp"
#include "../../common/decoder_result.hpp"
#include "../../error_handler.hpp"

#include <string>
#include <sstream>

namespace zxing {
namespace datamatrix {

class DecodedBitStreamParser {
private:
    static const int PAD_ENCODE = 0;  // Not really an encoding
    static const int ASCII_ENCODE = 1;
    static const int C40_ENCODE = 2;
    static const int TEXT_ENCODE = 3;
    static const int ANSIX12_ENCODE = 4;
    static const int EDIFACT_ENCODE = 5;
    static const int BASE256_ENCODE = 6;
    
    /**
     * See ISO 16022:2006, Annex C Table C.1
     * The C40 Basic Character Set (*'s used for placeholders for the shift values)
     */
    static const char C40_BASIC_SET_CHARS[];
    
    static const char C40_SHIFT2_SET_CHARS[];
    /**
     * See ISO 16022:2006, Annex C Table C.2
     * The Text Basic Character Set (*'s used for placeholders for the shift values)
     */
    static const char TEXT_BASIC_SET_CHARS[];
    
    static const char TEXT_SHIFT3_SET_CHARS[];
    /**
     * See ISO 16022:2006, 5.2.3 and Annex C, Table C.2
     */
    int decodeAsciiSegment(Ref<BitSource> bits, std::ostringstream &result, std::ostringstream &resultTrailer, ErrorHandler & err_handler);
    /**
     * See ISO 16022:2006, 5.2.5 and Annex C, Table C.1
     */
    void decodeC40Segment(Ref<BitSource> bits, std::ostringstream &result, ErrorHandler & err_handler);
    /**
     * See ISO 16022:2006, 5.2.6 and Annex C, Table C.2
     */
    void decodeTextSegment(Ref<BitSource> bits, std::ostringstream &result, ErrorHandler & err_handle);
    /**
     * See ISO 16022:2006, 5.2.7
     */
    void decodeAnsiX12Segment(Ref<BitSource> bits, std::ostringstream &result, ErrorHandler & err_handler);
    /**
     * See ISO 16022:2006, 5.2.8 and Annex C Table C.3
     */
    void decodeEdifactSegment(Ref<BitSource> bits, std::ostringstream &result, ErrorHandler &err_handler);
    /**
     * See ISO 16022:2006, 5.2.9 and Annex B, B.2
     */
    void decodeBase256Segment(Ref<BitSource> bits, std::ostringstream &result, std::vector<char> byteSegments, ErrorHandler & err_handler);
    
    void parseTwoBytes(int firstByte, int secondByte, int* result);
    /**
     * See ISO 16022:2006, Annex B, B.2
     */
    char unrandomize255State(int randomizedBase256Codeword,
                             int base256CodewordPosition) {
        int pseudoRandomNumber = ((149 * base256CodewordPosition) % 255) + 1;
        int tempVariable = randomizedBase256Codeword - pseudoRandomNumber;
        return (char) (tempVariable >= 0 ? tempVariable : (tempVariable + 256));
    };
    void append(std::ostream &ost, const char *bufIn, size_t nIn, const char *src);
    
public:
    DecodedBitStreamParser() { };
    Ref<DecoderResult> decode(ArrayRef<char> bytes, ErrorHandler & err_handler);
};

}
}

#endif  // __ZXING_DATAMATRIX_DECODER_DECODED_BIT_STREAM_PARSER_HPP__
