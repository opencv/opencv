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

#include "code93reader.hpp"
#include "one_dresult_point.hpp"
#include "../common/array.hpp"
#include "../reader_exception.hpp"
#include "../format_exception.hpp"
#include "../not_found_exception.hpp"
#include "../checksum_exception.hpp"
#include <math.h>
#include <limits.h>

using std::vector;
using std::string;
using zxing::Ref;
using zxing::Result;
using zxing::String;
using zxing::NotFoundException;
using zxing::ChecksumException;
using zxing::oned::Code93Reader;

// VC++
using zxing::BitArray;

#include "one_dconstant.hpp"
using namespace zxing;
using namespace oned;
using namespace zxing::oned::constant::Code93;

Code93Reader::Code93Reader() {
    decodeRowResult.reserve(20);
    counters.resize(6);
}

Ref<Result> Code93Reader::decodeRow(int rowNumber, Ref<BitArray> row) {
    Range start(findAsteriskPattern(row));
    
    if (start.isValid() == false) {
        return Ref<Result>(NULL);
    }
    
    // Read off white space
    int nextStart = row->getNextSet(start[1]);
    int end = row->getSize();
    
    vector<int>& theCounters(counters);
    {
        int size = theCounters.size();
        theCounters.resize(0);
        theCounters.resize(size);
    }
    string& result(decodeRowResult);
    result.clear();
    
    char decodedChar;
    int lastStart;
    
    ErrorHandler err_handler;
    do {
        bool rp = recordPattern(row, nextStart, theCounters, _onedReaderData);
        
        if (rp == false) {
            return Ref<Result>(NULL);
        }
        
        int pattern = toPattern(theCounters);
        if (pattern < 0) {
            return Ref<Result>(NULL);
        }
        decodedChar = patternToChar(pattern, err_handler);
        if (err_handler.ErrCode())  return Ref<Result>(NULL);
        
        result.append(1, decodedChar);
        lastStart = nextStart;
        for (int i = 0, e=theCounters.size(); i < e; ++i) {
            nextStart += theCounters[i];
        }
        // Read off white space
        nextStart = row->getNextSet(nextStart);
    } while (decodedChar != '*');
    result.resize(result.length() - 1);  // remove asterisk
    
    // Look for whitespace after pattern:
    int lastPatternSize = 0;
    for (int i = 0, e = theCounters.size(); i < e; i++) {
        lastPatternSize += theCounters[i];
    }
    
    // Should be at least one more black module
    if (nextStart == end || !row->get(nextStart)) {
        return Ref<Result>(NULL);
    }
    
    if (result.length() < 2) {
        // false positive -- need at least 2 checksum digits
        return Ref<Result>(NULL);
    }
    
    checkChecksums(result, err_handler);
    if (err_handler.ErrCode()) return Ref<Result>(NULL);
    // Remove checksum digits
    result.resize(result.length() - 2);
    
    Ref<String> resultString = decodeExtended(result, err_handler);
    if (err_handler.ErrCode()) return Ref<Result>(NULL);
    
    float left = static_cast<float>(start[1] + start[0]) / 2.0f;
    float right = lastStart + lastPatternSize / 2.0f;
    
    ArrayRef< Ref<ResultPoint> > resultPoints(2);
    resultPoints[0] =
    Ref<OneDResultPoint>(new OneDResultPoint(left, static_cast<float>(rowNumber)));
    resultPoints[1] =
    Ref<OneDResultPoint>(new OneDResultPoint(right, static_cast<float>(rowNumber)));
    
    return Ref<Result>(new Result(
                                  resultString,
                                  ArrayRef<char>(),
                                  resultPoints,
                                  BarcodeFormat::CODE_93));
}

Code93Reader::Range Code93Reader::findAsteriskPattern(Ref<BitArray> row)  {
    (void)row;
    
    {
        int size = counters.size();
        counters.resize(0);
        counters.resize(size);
    }
    vector<int>& theCounters(counters);
    int counterOffset = _onedReaderData->first_is_white ? 1 : 0;
    int patternLength = theCounters.size();
    int patternStart = _onedReaderData->first_is_white ? _onedReaderData->all_counters[0]: 0;
    
    for (int c = counterOffset; c < _onedReaderData->counter_size-patternLength+1; c+=2){
        int i=patternStart;
        for (int ii = 0; ii < patternLength; ii++){
            theCounters[ii] = _onedReaderData->all_counters[c+ii];
            i+=theCounters[ii];
        }
        if (toPattern(theCounters) == ASTERISK_ENCODING) {
            return Range(patternStart, i);
        }
        patternStart += counters[0] + counters[1];
    }
    
    return Range(false);
}

int Code93Reader::toPattern(vector<int>& counters) {
    int max = counters.size();
    int sum = 0;
    for (int i = 0, e=counters.size(); i<e; ++i) {
        sum += counters[i];
    }
    
    if (sum==0)
        return -1;
    
    int pattern = 0;
    for (int i = 0; i < max; i++) {
        int scaledShifted = (counters[i] << INTEGER_MATH_SHIFT) * 9 / sum;
        int scaledUnshifted = scaledShifted >> INTEGER_MATH_SHIFT;
        if ((scaledShifted & 0xFF) > 0x7F) {
            scaledUnshifted++;
        }
        if (scaledUnshifted < 1 || scaledUnshifted > 4) {
            return -1;
        }
        if ((i & 0x01) == 0) {
            for (int j = 0; j < scaledUnshifted; j++) {
                pattern = (pattern << 1) | 0x01;
            }
        }
        else
        {
            pattern <<= scaledUnshifted;
        }
    }
    return pattern;
}

char Code93Reader::patternToChar(int pattern, ErrorHandler & err_handler)  {
    for (int i = 0; i < CHARACTER_ENCODINGS_LENGTH; i++) {
        if (CHARACTER_ENCODINGS[i] == pattern) {
            return ALPHABET[i];
        }
    }
    err_handler = ErrorHandler(-1);
    return 0;
}

Ref<String> Code93Reader::decodeExtended(string const& encoded, ErrorHandler & err_handler)  {
    int length = encoded.length();
    string decoded;
    for (int i = 0; i < length; i++) {
        char c = encoded[i];
        if (c >= 'a' && c <= 'd') {
            if (i >= length - 1) {
                err_handler = FormatErrorHandler(-1);
                return Ref<String>();
            }
            char next = encoded[i + 1];
            char decodedChar = '\0';
            switch (c) {
                case 'd':
                    // +A to +Z map to a to z
                    if (next >= 'A' && next <= 'Z')
                    {
                        decodedChar = static_cast<char>(next + 32);
                    }
                    else
                    {
                        err_handler = FormatErrorHandler(-1);
                        return Ref<String>();
                    }
                    break;
                case 'a':
                    // $A to $Z map to control codes SH to SB
                    if (next >= 'A' && next <= 'Z')
                    {
                        decodedChar = static_cast<char>(next - 64);
                    }
                    else
                    {
                        err_handler = FormatErrorHandler(-1);
                        return Ref<String>();
                    }
                    break;
                case 'b':
                    if (next >= 'A' && next <= 'E')
                    {
                        // %A to %E map to control codes ESC to USep
                        decodedChar = static_cast<char>(next - 38);
                    }
                    else if (next >= 'F' && next <= 'J')
                    {
                        // %F to %J map to; < = > ?
                        decodedChar = static_cast<char>(next - 11);
                    }
                    else if (next >= 'K' && next <= 'O')
                    {
                        // %K to %O map to [ \ ] ^ _
                        decodedChar = static_cast<char>(next + 16);
                    }
                    else if (next >= 'P' && next <= 'S')
                    {
                        // %P to %S map to { | } ~
                        decodedChar = static_cast<char>(next + 43);
                    }
                    else if (next >= 'T' && next <= 'Z')
                    {
                        // %T to %Z all map to DEL (127)
                        decodedChar = 127;
                    }
                    else
                    {
                        err_handler = FormatErrorHandler(-1);
                        return Ref<String>();
                    }
                    break;
                case 'c':
                    // /A to /O map to ! to , and /Z maps to :
                    if (next >= 'A' && next <= 'O')
                    {
                        decodedChar = static_cast<char>(next - 32);
                    }
                    else if (next == 'Z')
                    {
                        decodedChar = ':';
                    }
                    else
                    {
                        err_handler = FormatErrorHandler(-1);
                        return Ref<String>();
                    }
                    break;
            }
            decoded.append(1, decodedChar);
            // bump up i again since we read two characters
            i++;
        }
        else
        {
            decoded.append(1, c);
        }
    }
    return Ref<String>(new String(decoded));
}

void Code93Reader::checkChecksums(string const& result, ErrorHandler & err_handler) {
    int length = result.length();
    checkOneChecksum(result, length - 2, 20, err_handler);
    if (err_handler.ErrCode())   return;
    checkOneChecksum(result, length - 1, 15, err_handler);
    if (err_handler.ErrCode())   return;
}

void Code93Reader::checkOneChecksum(string const& result,
                                    int checkPosition,
                                    int weightMax,
                                    ErrorHandler & err_handler) {
    int weight = 1;
    int total = 0;
    for (int i = checkPosition - 1; i >= 0; i--) {
        total += weight * ALPHABET_STRING.find_first_of(result[i]);
        if (++weight > weightMax) {
            weight = 1;
        }
    }
    if (result[checkPosition] != ALPHABET[total % 47]) {
        err_handler = CheckSumErrorHandler("checkOneChecksum");
        return;
    }
}
