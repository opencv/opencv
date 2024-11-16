// -*- mode:c++; tab-width:2; indent-tabs-mode:nil; c-basic-offset:2 -*-
/*
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

#include "../zxing.hpp"
#include "code128reader.hpp"
#include "one_dresult_point.hpp"
#include "../common/array.hpp"
#include "../reader_exception.hpp"
#include "../not_found_exception.hpp"
#include "../format_exception.hpp"
#include "../checksum_exception.hpp"
#include "one_dconstant.hpp"
#include "../error_handler.hpp"

// Add charset for Code128
#include "../common/string_utils.hpp"

#include <math.h>
#include <string.h>
#include <sstream>

#include<iostream>

using std::vector;
using std::string;
using zxing::NotFoundException;
using zxing::FormatException;
using zxing::ChecksumException;
using zxing::Ref;
using zxing::Result;
using zxing::oned::Code128Reader;

using namespace zxing::oned::constant::Code128;

// Add charset for Code128
using namespace zxing::common;

// VC++
using zxing::BitArray;

const int Code128Reader::MAX_AVG_VARIANCE = static_cast<int>(PATTERN_MATCH_RESULT_SCALE_FACTOR * 250/1000);
// const int Code128Reader::MAX_INDIVIDUAL_VARIANCE = static_cast<int>(PATTERN_MATCH_RESULT_SCALE_FACTOR * 700/1000);
const int Code128Reader::MAX_INDIVIDUAL_VARIANCE = static_cast<int>(PATTERN_MATCH_RESULT_SCALE_FACTOR * 840/1000);

Code128Reader::Code128Reader(){
}

vector<int> Code128Reader::findStartPattern(Ref<BitArray> row, ONED_READER_DATA* onedReaderData){
    vector<int> counters(6, 0);
    
    int counterOffset = onedReaderData->first_is_white ? 1 : 0;
    
    // int counterPosition = 0;
    int patternStart = onedReaderData->first_is_white ? onedReaderData->all_counters[0]: 0;
    
    for (int c = counterOffset; c < onedReaderData->counter_size-5; c+=2){
        int i = patternStart;
        for (int ii = 0; ii < 6; ii++){
            counters[ii] = onedReaderData->all_counters[c+ii];
            i+=counters[ii];
        }
        
        int bestVariance = MAX_AVG_VARIANCE;
        int bestMatch = -1;
        for (int startCode = CODE_START_A; startCode <= CODE_START_C; startCode++) {
            int variance = patternMatchVariance(counters, CODE_PATTERNS[startCode], MAX_INDIVIDUAL_VARIANCE);
            if (variance < bestVariance)
            {
                bestVariance = variance;
                bestMatch = startCode;
            }
        }
        
        // Look for whitespace before start pattern, >= 50% of width of start pattern
        ErrorHandler err_handler;
        if (bestMatch >= 0 &&
            row->isRange((std::max)(0, patternStart - (i - patternStart) / 2), patternStart, false, err_handler)) {
            if (err_handler.ErrCode())   return vector<int>(0);
            vector<int> resultValue(3, 0);
            resultValue[0] = patternStart;
            resultValue[1] = i;
            resultValue[2] = bestMatch;
            
            return resultValue;
        }
        patternStart += counters[0]+counters[1];
    }
    
    return vector<int>(0);
}

int Code128Reader::decodeCode(Ref<BitArray> row, vector<int>& counters, int rowOffset, ONED_READER_DATA* onedReaderData) {
    bool rp = recordPattern(row, rowOffset, counters, onedReaderData);
    
    if (rp == false)
    {
        return -1;
    }
    
    int bestVariance = MAX_AVG_VARIANCE;  // worst variance we'll accept
    int bestMatch = -1;
    for (int d = 0; d < CODE_PATTERNS_LENGTH; d++) {
        int const* const pattern = CODE_PATTERNS[d];
        int variance = patternMatchVariance(counters, pattern, MAX_INDIVIDUAL_VARIANCE);
        if (variance < bestVariance) {
            bestVariance = variance;
            bestMatch = d;
        }
    }
    // TODO(sofiawu) We're overlooking the fact that the STOP pattern has 7 values, not 6.
    if (bestMatch >= 0)
    {
        return bestMatch;
    }
    else
    {
        return -1;
    }
}


Ref<Result> Code128Reader::decodeRow(int rowNumber, Ref<BitArray> row) {
    bool convertFNC1 = false;
    vector<int> startPatternInfo(findStartPattern(row, _onedReaderData));
    
    if (startPatternInfo.size() == 0)
    {
        return Ref<Result>(NULL);
    }
    
    int startCode = startPatternInfo[2];
    int codeSet;
    switch (startCode) {
        case CODE_START_A:
            codeSet = CODE_CODE_A;
            break;
        case CODE_START_B:
            codeSet = CODE_CODE_B;
            break;
        case CODE_START_C:
            codeSet = CODE_CODE_C;
            break;
        default:
            return Ref<Result>(NULL);
    }
    
    bool done = false;
    bool isNextShifted = false;
    
    string result;
    vector<char> rawCodes(20, 0);
    rawCodes.push_back(static_cast<char>(startCode));
    
    int lastStart = startPatternInfo[0];
    int nextStart = startPatternInfo[1];
    vector<int> counters(6, 0);
    
    int lastCode = 0;
    int code = 0;
    int checksumTotal = startCode;
    int multiplier = 0;
    bool lastCharacterWasPrintable = true;
    
    // ZXing Patch 2992
    bool upperMode = false;
    bool shiftUpperMode = false;
    
    while (!done) {
        bool unshift = isNextShifted;
        isNextShifted = false;
        
        // Save off last code
        lastCode = code;
        
        code = decodeCode(row, counters, nextStart, _onedReaderData);
        
        if (code < 0)
        {
            return Ref<Result>(NULL);
        }
        
        // Remember whether the last code was printable or not (excluding CODE_STOP)
        if (code != CODE_STOP)
        {
            lastCharacterWasPrintable = true;
        }
        
        // Add to checksum computation (if not CODE_STOP of course)
        if (code != CODE_STOP)
        {
            multiplier++;
            checksumTotal += multiplier * code;
        }
        
        // Advance to where the next code will to start
        lastStart = nextStart;
        for (int i = 0, e = counters.size(); i < e; i++) {
            nextStart += counters[i];
        }
        
        // Take care of illegal start codes
        switch (code) {
            case CODE_START_A:
            case CODE_START_B:
            case CODE_START_C:
                return Ref<Result>(NULL);
        }
        
        switch (codeSet) {
            case CODE_CODE_A:
                if (code < 64)
                {
                    // Patch 2992 : Valiantliu
                    if (shiftUpperMode == upperMode)
                    {
                        result.append(1, static_cast<char>(' ' + code));
                    }
                    else
                    {
                        result.append(1, static_cast<char>(' ' + code + 128));
                    }
                    shiftUpperMode = false;
                }
                else if (code < 96)
                {
                    if (shiftUpperMode == upperMode)
                    {
                        result.append(1, static_cast<char>(code - 64));
                    }
                    else
                    {
                        result.append(1, static_cast<char>(code + 64));
                    }
                    shiftUpperMode = false;
                }
                else
                {
                    // Don't let CODE_STOP, which always appears, affect whether whether we think the
                    // last code was printable or not.
                    if (code != CODE_STOP)
                    {
                        lastCharacterWasPrintable = false;
                    }
                    switch (code) {
                        case CODE_FNC_1:
                            if (convertFNC1)
                            {
                                if (result.length() == 0)
                                {
                                    // GS1 specification 5.4.3.7. and 5.4.6.4. If the first char after the start code
                                    // is FNC1 then this is GS1-128. We add the symbology identifier.
                                    result.append("]C1");
                                }
                                else
                                {
                                    // GS1 specification 5.4.7.5. Every subsequent FNC1 is returned as ASCII 29 (GS)
                                    result.append(1, static_cast<char>(29));
                                }
                            }
                            break;
                        case CODE_FNC_2:
                        case CODE_FNC_3:
                            break;
                            // Patch 2992 : Valiantliu
                        case CODE_FNC_4_A:
                            if (!upperMode && shiftUpperMode)
                            {
                                upperMode = true;
                                shiftUpperMode = false;
                            }
                            else if (upperMode && shiftUpperMode)
                            {
                                upperMode = false;
                                shiftUpperMode = false;
                            }
                            else
                            {
                                shiftUpperMode = true;
                            }
                            break;
                        case CODE_SHIFT:
                            isNextShifted = true;
                            codeSet = CODE_CODE_B;
                            break;
                        case CODE_CODE_B:
                            codeSet = CODE_CODE_B;
                            break;
                        case CODE_CODE_C:
                            codeSet = CODE_CODE_C;
                            break;
                        case CODE_STOP:
                            done = true;
                            break;
                    }
                }
                break;
            case CODE_CODE_B:
                if (code < 96)
                {
                    if (shiftUpperMode == upperMode)
                    {
                        result.append(1, static_cast<char>(' ' + code));
                    }
                    else
                    {
                        result.append(1, static_cast<char>(' ' + code + 128));
                    }
                    shiftUpperMode = false;
                }
                else
                {
                        if (code != CODE_STOP)
                        {
                            lastCharacterWasPrintable = false;
                        }
                        switch (code) {
                            case CODE_FNC_1:
                                if (convertFNC1)
                                {
                                    if (result.length() == 0)
                                    {
                                        // GS1 specification 5.4.3.7. and 5.4.6.4. If the first char after the start code
                                        // is FNC1 then this is GS1-128. We add the symbology identifier.
                                        result.append("]C1");
                                    }
                                    else
                                    {
                                        // GS1 specification 5.4.7.5. Every subsequent FNC1 is returned as ASCII 29 (GS)
                                        result.append(1, static_cast<char>(29));
                                    }
                                }
                                break;
                            case CODE_FNC_2:
                            case CODE_FNC_3:
                                break;
                                // Patch 2992
                            case CODE_FNC_4_B:
                                if (!upperMode && shiftUpperMode)
                                {
                                    upperMode = true;
                                    shiftUpperMode = false;
                                }
                                else if (upperMode && shiftUpperMode)
                                {
                                    upperMode = false;
                                    shiftUpperMode = false;
                                }
                                else
                                {
                                    shiftUpperMode = true;
                                }
                                break;
                            case CODE_SHIFT:
                                isNextShifted = true;
                                codeSet = CODE_CODE_A;
                                break;
                            case CODE_CODE_A:
                                codeSet = CODE_CODE_A;
                                break;
                            case CODE_CODE_C:
                                codeSet = CODE_CODE_C;
                                break;
                            case CODE_STOP:
                                done = true;
                                break;
                        }
                    }
                break;
            case CODE_CODE_C:
                if (code < 100)
                {
                    if (code < 10)
                    {
                        result.append(1, '0');
                    }
                    char ccode[20];
                    sprintf(ccode, "%d", code);
                    result.append(ccode);
                }
                else
                {
                    if (code != CODE_STOP)
                    {
                        lastCharacterWasPrintable = false;
                    }
                    switch (code) {
                        case CODE_FNC_1:
                            if (convertFNC1)
                            {
                                if (result.length() == 0)
                                {
                                    // GS1 specification 5.4.3.7. and 5.4.6.4. If the first char after the start code
                                    // is FNC1 then this is GS1-128. We add the symbology identifier.
                                    result.append("]C1");
                                }
                                else
                                {
                                    // GS1 specification 5.4.7.5. Every subsequent FNC1 is returned as ASCII 29 (GS)
                                    result.append(1, static_cast<char>(29));
                                }
                            }
                            break;
                        case CODE_CODE_A:
                            codeSet = CODE_CODE_A;
                            break;
                        case CODE_CODE_B:
                            codeSet = CODE_CODE_B;
                            break;
                        case CODE_STOP:
                            done = true;
                            break;
                    }
                }
                break;
        }
        
        // Unshift back to another code set if we were shifted
        if (unshift)
        {
            codeSet = codeSet == CODE_CODE_A ? CODE_CODE_B : CODE_CODE_A;
        }
    }
    
    int lastPatternSize = nextStart - lastStart;
    
    // Check for ample whitespace following pattern, but, to do this we first need to remember that
    // we fudged decoding CODE_STOP since it actually has 7 bars, not 6. There is a black bar left
    // to read off. Would be slightly better to properly read. Here we just skip it:
    ErrorHandler err_handler;
    nextStart = row->getNextUnset(nextStart);
    if (!row->isRange(nextStart,
                      (std::min)(row->getSize(), nextStart + (nextStart - lastStart) / 2),
                      false, err_handler)) {
        return Ref<Result>(NULL);
    }
    if (err_handler.ErrCode()) return Ref<Result>(NULL);
    
    // Pull out from sum the value of the penultimate check code
    checksumTotal -= multiplier * lastCode;
    // lastCode is the checksum then:
    if (checksumTotal % 103 != lastCode)
    {
        return Ref<Result>(NULL);
    }
    
    // Need to pull out the check digits from string
    int resultLength = result.length();
    if (resultLength == 0)
    {
        return Ref<Result>(NULL);
    }
    
    // Only bother if the result had at least one character, and if the checksum digit happened to
    // be a printable character. If it was just interpreted as a control code, nothing to remove.
    if (resultLength > 0 && lastCharacterWasPrintable)
    {
        if (codeSet == CODE_CODE_C)
        {
            result.erase(resultLength - 2, resultLength);
        }
        else
        {
            result.erase(resultLength - 1, resultLength);
        }
    }
    
    float left = static_cast<float>(startPatternInfo[1] + startPatternInfo[0]) / 2.0f;
    float right = lastStart + lastPatternSize / 2.0f;
    
    int rawCodesSize = rawCodes.size();
    ArrayRef<char> rawBytes (rawCodesSize);
    for (int i = 0; i < rawCodesSize; i++) {
        rawBytes[i] = rawCodes[i];
    }
    
    ArrayRef< Ref<ResultPoint> > resultPoints(2);
    resultPoints[0] =
    Ref<OneDResultPoint>(new OneDResultPoint(left, static_cast<float>(rowNumber)));
    resultPoints[1] =
    Ref<OneDResultPoint>(new OneDResultPoint(right, static_cast<float>(rowNumber)));
    
    return Ref<Result>(new Result(Ref<String>(new String(result)), rawBytes, resultPoints, BarcodeFormat::CODE_128));
}

Code128Reader::~Code128Reader(){}

zxing::BarcodeFormat Code128Reader::getBarcodeFormat(){
    return BarcodeFormat::CODE_128;
}
