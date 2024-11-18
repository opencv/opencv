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
/*
 * Copyright 2010, 2012 ZXing authors All rights reserved.
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

#include <stdint.h>
#include "../../../bigint/big_integer_utils.hpp"
#include "../../format_exception.hpp"
#include "../../error_handler.hpp"
#include "decoded_bit_stream_parser.hpp"
#include "../../common/decoder_result.hpp"

using std::string;
using zxing::pdf417::DecodedBitStreamParser;
using zxing::ArrayRef;
using zxing::Ref;
using zxing::DecoderResult;
using zxing::String;
using zxing::FormatErrorHandler;
using zxing::ErrorHandler;

const int DecodedBitStreamParser::TEXT_COMPACTION_MODE_LATCH = 900;
const int DecodedBitStreamParser::BYTE_COMPACTION_MODE_LATCH = 901;
const int DecodedBitStreamParser::NUMERIC_COMPACTION_MODE_LATCH = 902;
const int DecodedBitStreamParser::BYTE_COMPACTION_MODE_LATCH_6 = 924;
const int DecodedBitStreamParser::BEGIN_MACRO_PDF417_CONTROL_BLOCK = 928;
const int DecodedBitStreamParser::BEGIN_MACRO_PDF417_OPTIONAL_FIELD = 923;
const int DecodedBitStreamParser::MACRO_PDF417_TERMINATOR = 922;
const int DecodedBitStreamParser::MODE_SHIFT_TO_BYTE_COMPACTION_MODE = 913;
const int DecodedBitStreamParser::MAX_NUMERIC_CODEWORDS = 15;

const int DecodedBitStreamParser::PL = 25;
const int DecodedBitStreamParser::LL = 27;
const int DecodedBitStreamParser::AS = 27;
const int DecodedBitStreamParser::ML = 28;
const int DecodedBitStreamParser::AL = 28;
const int DecodedBitStreamParser::PS = 29;
const int DecodedBitStreamParser::PAL = 29;

const int DecodedBitStreamParser::EXP900_SIZE = 16;

const char DecodedBitStreamParser::PUNCT_CHARS[] = {
    ';', '<', '>', '@', '[', '\\', '}', '_', '`', '~', '!',
    '\r', '\t', ',', ':', '\n', '-', '.', '$', '/', '"', '|', '*',
    '(', ')', '?', '{', '}', '\''};

const char DecodedBitStreamParser::MIXED_CHARS[] = {
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '&',
    '\r', '\t', ',', ':', '#', '-', '.', '$', '/', '+', '%', '*',
    '=', '^'};

ArrayRef<BigInteger> DecodedBitStreamParser::initEXP900() {
    ArrayRef<BigInteger> EXP900_ (16);
    EXP900_[0] = BigInteger(1);
    BigInteger nineHundred (900);
    EXP900_[1] = nineHundred;
    for (int i = 2; i < EXP900_->size(); i++) {
        EXP900_[i] = EXP900_[i - 1] * nineHundred;
    }
    return EXP900_;
}

DecodedBitStreamParser::DecodedBitStreamParser()
{
    EXP900 = initEXP900();
}

/**
 * PDF417 main decoder.
 **/
Ref<DecoderResult> DecodedBitStreamParser::decode(ArrayRef<int> codewords, ErrorHandler & err_handler)
{
    Ref<String> result(new String(100));
    // Get compaction mode
    int codeIndex = 1;
    int code = codewords[codeIndex++];
    while (codeIndex < codewords[0]) {
        switch (code) {
            case TEXT_COMPACTION_MODE_LATCH:
                codeIndex = textCompaction(codewords, codeIndex, result);
                break;
            case BYTE_COMPACTION_MODE_LATCH:
                codeIndex = byteCompaction(code, codewords, codeIndex, result);
                break;
            case NUMERIC_COMPACTION_MODE_LATCH:
                codeIndex = numericCompaction(codewords, codeIndex, result, err_handler);
                if (err_handler.errCode())
                    return Ref<DecoderResult>();
                break;
            case MODE_SHIFT_TO_BYTE_COMPACTION_MODE:
                codeIndex = byteCompaction(code, codewords, codeIndex, result);
                break;
            case BYTE_COMPACTION_MODE_LATCH_6:
                codeIndex = byteCompaction(code, codewords, codeIndex, result);
                break;
            default:
                // Default to text compaction. During testing numerous barcodes
                // appeared to be missing the starting mode. In these cases defaulting
                // to text compaction seems to work.
                codeIndex--;
                codeIndex = textCompaction(codewords, codeIndex, result);
                break;
        }
        if (codeIndex < codewords->size())
        {
            code = codewords[codeIndex++];
        }
        else
        {
            err_handler = FormatErrorHandler(-1);
            return Ref<DecoderResult>();
        }
    }
    return Ref<DecoderResult>(new DecoderResult(ArrayRef<char>(), result));
}

/**
 * Text Compaction mode (see 5.4.1.5) permits all printable ASCII characters to be
 * encoded, i.e. values 32 - 126 inclusive in accordance with ISO/IEC 646 (IRV), as
 * well as selected control characters.
 *
 * @param codewords The array of codewords (data + error)
 * @param codeIndex The current index into the codeword array.
 * @param result    The decoded data is appended to the result.
 * @return The next index into the codeword array.
 */
int DecodedBitStreamParser::textCompaction(ArrayRef<int> codewords,
                                           int codeIndex,
                                           Ref<String> result) {
    // 2 character per codeword
    ArrayRef<int> textCompactionData (codewords[0] << 1);
    // Used to hold the byte compaction value if there is a mode shift
    ArrayRef<int> byteCompactionData (codewords[0] << 1);
    
    int index = 0;
    bool end = false;
    while ((codeIndex < codewords[0]) && !end) {
        int code = codewords[codeIndex++];
        if (code < TEXT_COMPACTION_MODE_LATCH)
        {
            textCompactionData[index] = code / 30;
            textCompactionData[index + 1] = code % 30;
            index += 2;
        }
        else
        {
            switch (code) {
                case TEXT_COMPACTION_MODE_LATCH:
                    textCompactionData[index++] = TEXT_COMPACTION_MODE_LATCH;
                    break;
                case BYTE_COMPACTION_MODE_LATCH:
                    codeIndex--;
                    end = true;
                    break;
                case NUMERIC_COMPACTION_MODE_LATCH:
                    codeIndex--;
                    end = true;
                    break;
                case MODE_SHIFT_TO_BYTE_COMPACTION_MODE:
                    // The Mode Shift codeword 913 shall cause a temporary
                    // switch from Text Compaction mode to Byte Compaction mode.
                    // This switch shall be in effect for only the next codeword,
                    // after which the mode shall revert to the prevailing sub-mode
                    // of the Text Compaction mode. Codeword 913 is only available
                    // in Text Compaction mode; its use is described in 5.4.2.4.
                    textCompactionData[index] = MODE_SHIFT_TO_BYTE_COMPACTION_MODE;
                    code = codewords[codeIndex++];
                    byteCompactionData[index] = code;  // integer.toHexString(code);
                    index++;
                    break;
                case BYTE_COMPACTION_MODE_LATCH_6:
                    codeIndex--;
                    end = true;
                    break;
            }
        }
    }
    decodeTextCompaction(textCompactionData, byteCompactionData, index, result);
    return codeIndex;
}

/**
 * The Text Compaction mode includes all the printable ASCII characters
 * (i.e. values from 32 to 126) and three ASCII control characters: HT or tab
 * (ASCII value 9), LF or line feed (ASCII value 10), and CR or carriage
 * return (ASCII value 13). The Text Compaction mode also includes various latch
 * and shift characters which are used exclusively within the mode. The Text
 * Compaction mode encodes up to 2 characters per codeword. The compaction rules
 * for converting data into PDF417 codewords are defined in 5.4.2.2. The sub-mode
 * switches are defined in 5.4.2.3.
 *
 * @param textCompactionData The text compaction data.
 * @param byteCompactionData The byte compaction data if there
 *                           was a mode shift.
 * @param length             The size of the text compaction and byte compaction data.
 * @param result             The decoded data is appended to the result.
 */
void DecodedBitStreamParser::decodeTextCompaction(ArrayRef<int> textCompactionData,
                                                  ArrayRef<int> byteCompactionData,
                                                  int length,
                                                  Ref<String> result)
{
    // Beginning from an initial state of the Alpha sub-mode
    // The default compaction mode for PDF417 in effect at the start of each symbol shall always be Text
    // Compaction mode Alpha sub-mode (uppercase alphabetic). A latch codeword from another mode to the Text
    // Compaction mode shall always switch to the Text Compaction Alpha sub-mode.
    Mode subMode = ALPHA;
    Mode priorToShiftMode = ALPHA;
    int i = 0;
    while (i < length) {
        int subModeCh = textCompactionData[i];
        char ch = 0;
        switch (subMode) {
            case ALPHA:
                // Alpha (uppercase alphabetic)
                if (subModeCh < 26)
                {
                    // Upper case Alpha Character
                    ch = static_cast<char>('A' + subModeCh);
                }
                else
                {
                    if (subModeCh == 26)
                    {
                        ch = ' ';
                    }
                    else if (subModeCh == LL)
                    {
                        subMode = LOWER;
                    }
                    else if (subModeCh == ML)
                    {
                        subMode = MIXED;
                    }
                    else if (subModeCh == PS)
                    {
                        // Shift to punctuation
                        priorToShiftMode = subMode;
                        subMode = PUNCT_SHIFT;
                    }
                    else if (subModeCh == MODE_SHIFT_TO_BYTE_COMPACTION_MODE)
                    {
                        result->append(static_cast<char>(byteCompactionData[i]));
                    }
                    else if (subModeCh == TEXT_COMPACTION_MODE_LATCH)
                    {
                        subMode = ALPHA;
                    }
                }
                break;
                
            case LOWER:
                // Lower (lowercase alphabetic)
                if (subModeCh < 26)
                {
                    ch = static_cast<char>('a' + subModeCh);
                }
                else
                {
                    if (subModeCh == 26)
                    {
                        ch = ' ';
                    }
                    else if (subModeCh == AS)
                    {
                        // Shift to alpha
                        priorToShiftMode = subMode;
                        subMode = ALPHA_SHIFT;
                    }
                    else if (subModeCh == ML)
                    {
                        subMode = MIXED;
                    }
                    else if (subModeCh == PS)
                    {
                        // Shift to punctuation
                        priorToShiftMode = subMode;
                        subMode = PUNCT_SHIFT;
                    }
                    else if (subModeCh == MODE_SHIFT_TO_BYTE_COMPACTION_MODE)
                    {
                        result->append(static_cast<char>(byteCompactionData[i]));
                    }
                    else if (subModeCh == TEXT_COMPACTION_MODE_LATCH)
                    {
                        subMode = ALPHA;
                    }
                }
                break;
                
            case MIXED:
                // Mixed (numeric and some punctuation)
                if (subModeCh < PL) {
                    ch = MIXED_CHARS[subModeCh];
                }
                else
                {
                    if (subModeCh == PL) {
                        subMode = PUNCT;
                    }
                    else if (subModeCh == 26)
                    {
                        ch = ' ';
                    }
                    else if (subModeCh == LL)
                    {
                        subMode = LOWER;
                    }
                    else if (subModeCh == AL)
                    {
                        subMode = ALPHA;
                    }
                    else if (subModeCh == PS)
                    {
                        // Shift to punctuation
                        priorToShiftMode = subMode;
                        subMode = PUNCT_SHIFT;
                    }
                    else if (subModeCh == MODE_SHIFT_TO_BYTE_COMPACTION_MODE)
                    {
                        result->append(static_cast<char>(byteCompactionData[i]));
                    }
                    else if (subModeCh == TEXT_COMPACTION_MODE_LATCH)
                    {
                        subMode = ALPHA;
                    }
                }
                break;
                
            case PUNCT:
                // Punctuation
                if (subModeCh < PAL)
                {
                    ch = PUNCT_CHARS[subModeCh];
                }
                else
                {
                    if (subModeCh == PAL)
                    {
                        subMode = ALPHA;
                    }
                    else if (subModeCh == MODE_SHIFT_TO_BYTE_COMPACTION_MODE)
                    {
                        result->append(static_cast<char>(byteCompactionData[i]));
                    }
                    else if (subModeCh == TEXT_COMPACTION_MODE_LATCH)
                    {
                        subMode = ALPHA;
                    }
                }
                break;
                
            case ALPHA_SHIFT:
                // Restore sub-mode
                subMode = priorToShiftMode;
                if (subModeCh < 26) {
                    ch = static_cast<char>('A' + subModeCh);
                }
                else
                {
                    if (subModeCh == 26) {
                        ch = ' ';
                    }
                    else
                    {
                        if (subModeCh == 26) {
                            ch = ' ';
                        }
                        else if (subModeCh == TEXT_COMPACTION_MODE_LATCH)
                        {
                            subMode = ALPHA;
                        }
                    }
                }
                break;
                
            case PUNCT_SHIFT:
                // Restore sub-mode
                subMode = priorToShiftMode;
                if (subModeCh < PAL)
                {
                    ch = PUNCT_CHARS[subModeCh];
                }
                else
                {
                    if (subModeCh == PAL)
                    {
                        subMode = ALPHA;
                        // 2012-11-27 added from recent java code:
                    }
                    else if (subModeCh == MODE_SHIFT_TO_BYTE_COMPACTION_MODE)
                    {
                        // PS before Shift-to-Byte is used as a padding character,
                        // see 5.4.2.4 of the specification
                        result->append(static_cast<char>(byteCompactionData[i]));
                    }
                    else if (subModeCh == TEXT_COMPACTION_MODE_LATCH)
                    {
                        subMode = ALPHA;
                    }
                }
                break;
        }
        if (ch != 0)
        {
            // Append decoded character to result
            result->append(ch);
        }
        i++;
    }
}

/**
 * Byte Compaction mode (see 5.4.3) permits all 256 possible 8-bit byte values to be encoded.
 * This includes all ASCII characters value 0 to 127 inclusive and provides for international
 * character set support.
 *
 * @param mode      The byte compaction mode i.e. 901 or 924
 * @param codewords The array of codewords (data + error)
 * @param codeIndex The current index into the codeword array.
 * @param result    The decoded data is appended to the result.
 * @return The next index into the codeword array.
 */
int DecodedBitStreamParser::byteCompaction(int mode,
                                           ArrayRef<int> codewords,
                                           int codeIndex, Ref<String> result) {
    if (mode == BYTE_COMPACTION_MODE_LATCH)
    {
        // Total number of Byte Compaction characters to be encoded
        // is not a multiple of 6
        int count = 0;
        int64_t value = 0;
        ArrayRef<char> decodedData = new Array<char>(6);
        ArrayRef<int> byteCompactedCodewords = new Array<int>(6);
        bool end = false;
        int nextCode = codewords[codeIndex++];
        while ((codeIndex < codewords[0]) && !end) {
            byteCompactedCodewords[count++] = nextCode;
            // Base 900
            value = 900 * value + nextCode;
            nextCode = codewords[codeIndex++];
            // perhaps it should be ok to check only nextCode >= TEXT_COMPACTION_MODE_LATCH
            if (nextCode == TEXT_COMPACTION_MODE_LATCH ||
                nextCode == BYTE_COMPACTION_MODE_LATCH ||
                nextCode == NUMERIC_COMPACTION_MODE_LATCH ||
                nextCode == BYTE_COMPACTION_MODE_LATCH_6 ||
                nextCode == BEGIN_MACRO_PDF417_CONTROL_BLOCK ||
                nextCode == BEGIN_MACRO_PDF417_OPTIONAL_FIELD ||
                nextCode == MACRO_PDF417_TERMINATOR)
            {
                end = true;
            }
            else
            {
                if ((count%5 == 0) && (count > 0))
                {
                    // Decode every 5 codewords
                    // Convert to Base 256
                    for (int j = 0; j < 6; ++j)
                    {
                        decodedData[5 - j] = static_cast<char>(value%256);
                        value >>= 8;
                    }
                    result->append(string(&(decodedData->values()[0]), decodedData->values().size()));
                    count = 0;
                }
            }
        }
        
        // if the end of all codewords is reached the last codeword needs to be added
        if (codeIndex == codewords[0] && nextCode < TEXT_COMPACTION_MODE_LATCH)
            byteCompactedCodewords[count++] = nextCode;
        
        // If Byte Compaction mode is invoked with codeword 901,
        // the last group of codewords is interpreted directly
        // as one byte per codeword, without compaction.
        for (int i = 0; i < count; i++)
        {
            result->append(static_cast<char>(byteCompactedCodewords[i]));
        }
    }
    else if (mode == BYTE_COMPACTION_MODE_LATCH_6)
    {
        // Total number of Byte Compaction characters to be encoded
        // is an integer multiple of 6
        int count = 0;
        int64_t value = 0;
        bool end = false;
        while (codeIndex < codewords[0] && !end) {
            int code = codewords[codeIndex++];
            if (code < TEXT_COMPACTION_MODE_LATCH)
            {
                count++;
                // Base 900
                value = 900 * value + code;
            }
            else
            {
                if (code == TEXT_COMPACTION_MODE_LATCH ||
                    code == BYTE_COMPACTION_MODE_LATCH ||
                    code == NUMERIC_COMPACTION_MODE_LATCH ||
                    code == BYTE_COMPACTION_MODE_LATCH_6 ||
                    code == BEGIN_MACRO_PDF417_CONTROL_BLOCK ||
                    code == BEGIN_MACRO_PDF417_OPTIONAL_FIELD ||
                    code == MACRO_PDF417_TERMINATOR)
                {
                    codeIndex--;
                    end = true;
                }
            }
            if ((count % 5 == 0) && (count > 0))
            {
                // Decode every 5 codewords
                // Convert to Base 256
                ArrayRef<char> decodedData = new Array<char>(6);
                for (int j = 0; j < 6; ++j) {
                    decodedData[5 - j] = static_cast<char>(value & 0xFF);
                    value >>= 8;
                }
                result->append(string(&decodedData[0], 6));
                // 2012-11-27 hfn after recent java code/fix by srowen
                count = 0;
            }
        }
    }
    return codeIndex;
}

/**
 * Numeric Compaction mode (see 5.4.4) permits efficient encoding of numeric data strings.
 *
 * @param codewords The array of codewords (data + error)
 * @param codeIndex The current index into the codeword array.
 * @param result    The decoded data is appended to the result.
 * @return The next index into the codeword array.
 */
int DecodedBitStreamParser::numericCompaction(ArrayRef<int> codewords,
                                              int codeIndex,
                                              Ref<String> result,
                                              ErrorHandler & err_handler) {
    int count = 0;
    bool end = false;
    
    ArrayRef<int> numericCodewords = new Array<int>(MAX_NUMERIC_CODEWORDS);
    
    while (codeIndex < codewords[0] && !end) {
        int code = codewords[codeIndex++];
        if (codeIndex == codewords[0])
        {
            end = true;
        }
        if (code < TEXT_COMPACTION_MODE_LATCH)
        {
            numericCodewords[count] = code;
            count++;
        }
        else
        {
            if (code == TEXT_COMPACTION_MODE_LATCH ||
                code == BYTE_COMPACTION_MODE_LATCH ||
                code == BYTE_COMPACTION_MODE_LATCH_6 ||
                code == BEGIN_MACRO_PDF417_CONTROL_BLOCK ||
                code == BEGIN_MACRO_PDF417_OPTIONAL_FIELD ||
                code == MACRO_PDF417_TERMINATOR)
            {
                codeIndex--;
                end = true;
            }
        }
        if (count % MAX_NUMERIC_CODEWORDS == 0 ||
            code == NUMERIC_COMPACTION_MODE_LATCH ||
            end)
        {
            // Re-invoking Numeric Compaction mode (by using codeword 902
            // while in Numeric Compaction mode) serves  to terminate the
            // current Numeric Compaction mode grouping as described in 5.4.4.2,
            // and then to start a new one grouping.
            Ref<String> s = decodeBase900toBase10(numericCodewords, count, err_handler);
            if (err_handler.errCode() || s == NULL)
            {
                return -1;
            }
            result->append(s->getText());
            count = 0;
        }
    }
    return codeIndex;
}

/**
 * Convert a list of Numeric Compacted codewords from Base 900 to Base 10.
 *
 * @param codewords The array of codewords
 * @param count     The number of codewords
 * @return The decoded string representing the Numeric data.
 */
/*
 EXAMPLE
 Encode the fifteen digit numeric string 000213298174000
 Prefix the numeric string with a 1 and set the initial value of
 t = 1 000 213 298 174 000
 Calculate codeword 0
 d0 = 1 000 213 298 174 000 mod 900 = 200
 
 t = 1 000 213 298 174 000 div 900 = 1 111 348 109 082
 Calculate codeword 1
 d1 = 1 111 348 109 082 mod 900 = 282
 
 t = 1 111 348 109 082 div 900 = 1 234 831 232
 Calculate codeword 2
 d2 = 1 234 831 232 mod 900 = 632
 
 t = 1 234 831 232 div 900 = 1 372 034
 Calculate codeword 3
 d3 = 1 372 034 mod 900 = 434
 
 t = 1 372 034 div 900 = 1 524
 Calculate codeword 4
 d4 = 1 524 mod 900 = 624
 
 t = 1 524 div 900 = 1
 Calculate codeword 5
 d5 = 1 mod 900 = 1
 t = 1 div 900 = 0
 Codeword sequence is: 1, 624, 434, 632, 282, 200
 
 Decode the above codewords involves
 1 x 900 power of 5 + 624 x 900 power of 4 + 434 x 900 power of 3 +
 632 x 900 power of 2 + 282 x 900 power of 1 + 200 x 900 power of 0 = 1000213298174000
 
 Remove leading 1 =>  Result is 000213298174000
 */
Ref<String> DecodedBitStreamParser::decodeBase900toBase10(ArrayRef<int> codewords, int count, ErrorHandler& err_handler)
{
    BigInteger result = BigInteger(0);
    for (int i = 0; i < count; i++) {
        result = result + (EXP900[count - i - 1] * BigInteger(codewords[i]));
    }
    string resultString = bigIntegerToString(result);
    if (resultString[0] != '1')
    {
        err_handler = FormatErrorHandler("DecodedBitStreamParser::decodeBase900toBase10: String does not begin with 1");
        return Ref<String>();
    }
    string resultString2;
    resultString2.assign(resultString.begin()+1, resultString.end());
    Ref<String> res (new String(resultString2));
    return res;
}
