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

#include "code39reader.hpp"
#include "one_dresult_point.hpp"
#include "../common/array.hpp"
#include "../reader_exception.hpp"
#include "../not_found_exception.hpp"
#include "../checksum_exception.hpp"
#include <math.h>
#include <limits.h>

#include<iostream>

using std::vector;
using zxing::Ref;
using zxing::Result;
using zxing::String;
using zxing::NotFoundException;
using zxing::ChecksumException;
using zxing::oned::Code39Reader;

// VC++
using zxing::BitArray;

#include "one_dconstant.hpp"
using namespace zxing;
using namespace oned;

using namespace zxing::oned::constant::Code39;

void Code39Reader::init(bool usingCheckDigit_, bool extendedMode_) {
    usingCheckDigit = usingCheckDigit_;
    extendedMode = extendedMode_;
    decodeRowResult.reserve(20);
    counters.resize(9);
}

/**
 * Creates a reader that assumes all encoded data is data, and does not treat
 * the final character as a check digit. It will not decoded "extended
 * Code 39" sequences.
 */
Code39Reader::Code39Reader() {
    init();
}

/**
 * Creates a reader that can be configured to check the last character as a
 * check digit. It will not decoded "extended Code 39" sequences.
 *
 * @param usingCheckDigit if true, treat the last data character as a check
 * digit, not data, and verify that the checksum passes.
 */
Code39Reader::Code39Reader(bool usingCheckDigit_) {
    init(usingCheckDigit_);
}

Code39Reader::Code39Reader(bool usingCheckDigit_, bool extendedMode_) {
    init(usingCheckDigit_, extendedMode_);
}

Ref<Result> Code39Reader::decodeRow(int rowNumber, Ref<BitArray> row) {
    std::vector<int>& theCounters(counters);
    
    std::string& result(decodeRowResult);
    result.clear();
    
    vector<int> start(findAsteriskPattern(row, theCounters, _onedReaderData));
    
    if (start.size() == 0)
    {
        return Ref<Result>(NULL);
    }
    
    // Read off white space
    int nextStart = row->getNextSet(start[1]);
    int end = row->getSize();
    
    ErrorHandler err_handler;
    char decodedChar;
    int lastStart;
    do {
        bool rp = recordPattern(row, nextStart, theCounters,_onedReaderData);
        
        if (rp == false) {
            return Ref<Result>(NULL);
        }
        
        int pattern = toNarrowWidePattern(theCounters);
        if (pattern < 0) {
            return Ref<Result>(NULL);
        }
        decodedChar = patternToChar(pattern, err_handler);
        if (err_handler.ErrCode())   return Ref<Result>(NULL);
        result.append(1, decodedChar);
        lastStart = nextStart;

        end=theCounters.size();
        for (int i = 0; i < end; i++) {
            nextStart += theCounters[i];
        }
        // Read off white space
        nextStart = row->getNextSet(nextStart);
    } while (decodedChar != '*');
    result.resize(decodeRowResult.length()-1);  // remove asterisk
    
    // Look for whitespace after pattern:
    int lastPatternSize = 0;
    for (int i = 0, e = theCounters.size(); i < e; i++) {
        lastPatternSize += theCounters[i];
    }
    int whiteSpaceAfterEnd = nextStart - lastStart - lastPatternSize;
    // If 50% of last pattern size, following last pattern, is not whitespace,
    // fail (but if it's whitespace to the very end of the image, that's OK)
    // if (nextStart != end && (whiteSpaceAfterEnd >> 1) < lastPatternSize) {
    // Issue #86 : Fix logic error in Code 39 that was requiring too much quiet zone
    // https:// github.com/zxing/zxing/commit/f0532a273031f71e75110276833164156c768a6f
    if (nextStart != end && (whiteSpaceAfterEnd << 1) < lastPatternSize) {
        return Ref<Result>(NULL);
    }
    
    if (usingCheckDigit) {
        int max = result.length() - 1;
        int total = 0;
        for (int i = 0; i < max; i++) {
            total += alphabet_string.find_first_of(decodeRowResult[i], 0);
        }
        if (result[max] != ALPHABET[total % 43]) {
            return Ref<Result>(NULL);
        }
        result.resize(max);
    }
    
    if (result.length() == 0) {
        // Almost false positive
        return Ref<Result>(NULL);
    }
    
    Ref<String> resultString;
    if (extendedMode) {
        resultString = decodeExtended(result, err_handler);
        if (err_handler.ErrCode())   return Ref<Result>(NULL);
    }
    else
    {
        resultString = Ref<String>(new String(result));
    }
    
    float left = static_cast<float>(start[1] + start[0]) / 2.0f;
    float right = lastStart + lastPatternSize / 2.0f;
    
    ArrayRef< Ref<ResultPoint> > resultPoints(2);
    resultPoints[0] =
    Ref<OneDResultPoint>(new OneDResultPoint(left, static_cast<float>(rowNumber)));
    resultPoints[1] =
    Ref<OneDResultPoint>(new OneDResultPoint(right, static_cast<float>(rowNumber)));
    
    return Ref<Result>(
                       new Result(resultString, ArrayRef<char>(), resultPoints, BarcodeFormat::CODE_39)
                      );
}

vector<int> Code39Reader::findAsteriskPattern(Ref<BitArray> row, vector<int>& counters, ONED_READER_DATA* onedReaderData){
    
    int counterOffset = onedReaderData->first_is_white ? 1 : 0;
    int patternLength = counters.size();
    int patternStart = onedReaderData->first_is_white ? onedReaderData->all_counters[0]: 0;
    
    for (int c = counterOffset; c < onedReaderData->counter_size-patternLength+1; c+=2){
        int i = patternStart;
        for (int ii = 0; ii < patternLength; ii++){
            counters[ii] = onedReaderData->all_counters[c+ii];
            i+=counters[ii];
        }
        
        // Look for whitespace before start pattern, >= 50% of width of
        // start pattern.
        ErrorHandler err_handler;
        if (toNarrowWidePattern(counters) == ASTERISK_ENCODING &&
            row->isRange((std::max)(0, patternStart - ((i - patternStart) >> 1)), patternStart, false, err_handler)) {
            if (err_handler.ErrCode())   return vector<int>(0);
            vector<int> resultValue (2, 0);
            resultValue[0] = patternStart;
            resultValue[1] = i;
            return resultValue;
        }
        patternStart += counters[0]+counters[1];
    }
    
    return vector<int>(0);
}

// For efficiency, returns -1 on failure. Not throwing here saved as many as
// 700 exceptions per image when using some of our blackbox images.
int Code39Reader::toNarrowWidePattern(vector<int>& counters){
    int numCounters = counters.size();
    int maxNarrowCounter = 0;
    int wideCounters;
    do {
        int minCounter = INT_MAX;
        for (int i = 0; i < numCounters; i++) {
            int counter = counters[i];
            if (counter < minCounter && counter > maxNarrowCounter) {
                minCounter = counter;
            }
        }
        maxNarrowCounter = minCounter;
        wideCounters = 0;
        int totalWideCountersWidth = 0;
        int pattern = 0;
        for (int i = 0; i < numCounters; i++) {
            int counter = counters[i];
            if (counter > maxNarrowCounter) {
                pattern |= 1 << (numCounters - 1 - i);
                wideCounters++;
                totalWideCountersWidth += counter;
            }
        }
        if (wideCounters == 3) {
            // Found 3 wide counters, but are they close enough in width?
            // We can perform a cheap, conservative check to see if any individual
            // counter is more than 1.5 times the average:
            for (int i = 0; i < numCounters && wideCounters > 0; i++) {
                int counter = counters[i];
                if (counter > maxNarrowCounter) {
                    wideCounters--;
                    // totalWideCountersWidth = 3 * average, so this checks if
                    // counter >= 3/2 * average.
                    if ((counter << 1) >= totalWideCountersWidth) {
                        return -1;
                    }
                }
            }
            return pattern;
        }
    } while (wideCounters > 3);
    return -1;
}

char Code39Reader::patternToChar(int pattern, ErrorHandler & err_handler){
    for (int i = 0; i < CHARACTER_ENCODINGS_LEN; i++) {
        if (CHARACTER_ENCODINGS[i] == pattern) {
            return ALPHABET[i];
        }
    }

    err_handler = ErrorHandler(-1);
    return 0;
}

Ref<String> Code39Reader::decodeExtended(std::string encoded, ErrorHandler & err_handler){
    std::string tmpDecoded;
    for (size_t i = 0; i < encoded.length(); i++) {
        char c = encoded[i];
        if (c == '+' || c == '$' || c == '%' || c == '/')
        {
            char next = encoded[i + 1];
            char decodedChar = '\0';
            switch (c) {
                case '+':  // +A to +Z map to a to z
                    if (next >= 'A' && next <= 'Z')
                    {
                        decodedChar = static_cast<char>(next + 32);
                    }
                    else
                    {
                        err_handler = ErrorHandler(-1);
                        return Ref<String>();
                    }
                    break;
                case '$':   // $A to $Z map to control codes SH to SB
                    if (next >= 'A' && next <= 'Z')
                    {
                        decodedChar = static_cast<char>(next - 64);
                    }
                    else
                    {
                        err_handler = ErrorHandler(-1);
                        return Ref<String>();
                    }
                    break;
                case '%':   // %A to %E map to control codes ESC to US
                    if (next >= 'A' && next <= 'E')
                    {
                        decodedChar = static_cast<char>(next - 38);
                    }
                    else if (next >= 'F' && next <= 'W')
                    {
                        decodedChar = static_cast<char>(next - 11);
                    }
                    else
                    {
                        err_handler = ErrorHandler(-1);
                        return Ref<String>();
                    }
                    break;
                case '/':   // /A to /O map to ! to , and /Z maps to :
                    if (next >= 'A' && next <= 'O') {
                        decodedChar = static_cast<char>(next - 32);
                    }
                    else if (next == 'Z')
                    {
                        decodedChar = ':';
                    }
                    else
                    {
                        err_handler = ErrorHandler(-1);
                        return Ref<String>();
                    }
                    break;
            }
            tmpDecoded.append(1, decodedChar);
            i++;   // bump up i again since we read two characters
        }
        else
        {
            tmpDecoded.append(1, c);
        }
    }
    Ref<String> decoded(new String(tmpDecoded));
    return decoded;
}
