// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").

// -*- mode:c++; tab-width:2; indent-tabs-mode:nil; c-basic-offset:2 -*-
#ifndef __ZXING_PDF417_DECODER_DECODED_BIT_STREAM_PARSER_HPP__
#define __ZXING_PDF417_DECODER_DECODED_BIT_STREAM_PARSER_HPP__

/*
 * Copyright 2010 ZXing authors All rights reserved.
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

#include "../../../bigint/big_integer.hpp"
#include "../../common/array.hpp"
#include "../../common/str.hpp"
#include "../../common/decoder_result.hpp"

namespace zxing {
namespace pdf417 {

class DecodedBitStreamParser {
protected:
    enum Mode {
        ALPHA,
        LOWER,
        MIXED,
        PUNCT,
        ALPHA_SHIFT,
        PUNCT_SHIFT
    };
    
private:
    static const int TEXT_COMPACTION_MODE_LATCH;
    static const int BYTE_COMPACTION_MODE_LATCH;
    static const int NUMERIC_COMPACTION_MODE_LATCH;
    static const int BYTE_COMPACTION_MODE_LATCH_6;
    static const int BEGIN_MACRO_PDF417_CONTROL_BLOCK;
    static const int BEGIN_MACRO_PDF417_OPTIONAL_FIELD;
    static const int MACRO_PDF417_TERMINATOR;
    static const int MODE_SHIFT_TO_BYTE_COMPACTION_MODE;
    static const int MAX_NUMERIC_CODEWORDS;
    
    static const int PL;
    static const int LL;
    static const int AS;
    static const int ML;
    static const int AL;
    static const int PS;
    static const int PAL;
    static const int EXP900_SIZE;
    
    static const char PUNCT_CHARS[];
    static const char MIXED_CHARS[];
    
    ArrayRef<BigInteger> EXP900;
    ArrayRef<BigInteger> initEXP900();
    
    int textCompaction(ArrayRef<int> codewords, int codeIndex, Ref<String> result);
    void decodeTextCompaction(ArrayRef<int> textCompactionData,
                              ArrayRef<int> byteCompactionData,
                              int length,
                              Ref<String> result);
    int byteCompaction(int mode, ArrayRef<int> codewords, int codeIndex, Ref<String> result);
    int numericCompaction(ArrayRef<int> codewords, int codeIndex, Ref<String> result, ErrorHandler& err_handler);
    Ref<String> decodeBase900toBase10(ArrayRef<int> codewords, int count, ErrorHandler & err_handler);
    
public:
    DecodedBitStreamParser();
    Ref<DecoderResult> decode(ArrayRef<int> codewords, ErrorHandler & err_handler);
};

}  // namespace pdf417
}  // namespace zxing

#endif  // __ZXING_PDF417_DECODER_DECODED_BIT_STREAM_PARSER_HPP__
