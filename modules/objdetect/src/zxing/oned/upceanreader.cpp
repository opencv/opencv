// -*- mode:c++; tab-width:2; indent-tabs-mode:nil; c-basic-offset:2 -*-
/*
 *  UPCEANReader.cpp
 *  ZXing
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

#include "../zxing.hpp"
#include "upceanreader.hpp"
#include "one_dresult_point.hpp"
#include "../reader_exception.hpp"
#include "../format_exception.hpp"
#include "../not_found_exception.hpp"
#include "../checksum_exception.hpp"

#include "one_dconstant.hpp"

using std::vector;
using std::string;

using zxing::Ref;
using zxing::Result;
using zxing::NotFoundException;
using zxing::ChecksumException;
using zxing::FormatException;
using zxing::oned::UPCEANReader;
using zxing::ErrorHandler;

// VC++
using zxing::BitArray;
using zxing::String;

using namespace zxing::oned::constant::UPCEAN;

const int UPCEANReader::MAX_AVG_VARIANCE = static_cast<int>(PATTERN_MATCH_RESULT_SCALE_FACTOR * 0.48f);
const int UPCEANReader::MAX_INDIVIDUAL_VARIANCE = static_cast<int>(PATTERN_MATCH_RESULT_SCALE_FACTOR * 0.7f);

#define VECTOR_INIT(v) v, v + sizeof(v)/sizeof(v[0])

const vector<int>
UPCEANReader::START_END_PATTERN (VECTOR_INIT(START_END_PATTERN_));

const vector<int>
UPCEANReader::MIDDLE_PATTERN (VECTOR_INIT(MIDDLE_PATTERN_));
const vector<int const*>
UPCEANReader::L_PATTERNS (VECTOR_INIT(L_PATTERNS_));
const vector<int const*>
UPCEANReader::G_PATTERNS (VECTOR_INIT(G_PATTERNS_));
const vector<int const*>
UPCEANReader::L_AND_G_PATTERNS (VECTOR_INIT(L_AND_G_PATTERNS_));


UPCEANReader::UPCEANReader() {
    //  -- To remember decodeDigit result for L_PATTERNS & L_AND_G_PATTERNS
}

Ref<Result> UPCEANReader::decodeRow(int rowNumber, Ref<BitArray> row) {
    ErrorHandler err_handler;

    Ref<Result> rst = decodeRow(rowNumber, row, findStartGuardPattern(row, _onedReaderData, err_handler));
    if (err_handler.ErrCode())   return Ref<Result>();
    return rst;
}

#ifdef USE_PRE_BESTMATCH
int UPCEANReader::getCounterOffset(vector<int> counters)
{
    int counterOffset = 0;
    
    for (int i = 0, e = counters.size(); i < e; i++) {
        counterOffset += counters[i];
    }
    
    return counterOffset;
}
#endif

#ifdef USE_PRE_BESTMATCH
int UPCEANReader::initbestMatchDigit(Ref<BitArray> row, ONED_READER_DATA* onedReaderData)
{
    //  -- To remember decodeDigit result for L_PATTERNS & L_AND_G_PATTERNS
    size_t rowLength = row->getSize();
    if (rowLength > onedReaderData->digitResultCache.size())
    {
        onedReaderData->digitResultCache.resize(rowLength);
    }
    
    for (size_t i = 0; i < onedReaderData->digitResultCache.size(); i++)
    {
        onedReaderData->digitResultCache[i].bestMatch[0] = -2;
        onedReaderData->digitResultCache[i].bestMatch[1] = -2;
        onedReaderData->digitResultCache[i].counterOffset = -1;
    }
    
    return 0;
}
#endif

Ref<Result> UPCEANReader::decodeRow(int rowNumber,
                                    Ref<BitArray> row,
                                    Range const& startGuardRange) {
    
    string& result = decodeRowStringBuffer;
    result.clear();
    int endStart = decodeMiddle(row, startGuardRange, result);
    
    if (endStart < 0) {
        return Ref<Result>(NULL);
    }
    
    Range endRange = decodeEnd(row, endStart);
    
    if (endRange.isValid() == false)
    {
        return Ref<Result>(NULL);
    }
    
    // Make sure there is a quiet zone at least as big as the end pattern after the barcode.
    // The spec might want more whitespace, but in practice this is the maximum we can count on.
    
    int end = endRange[1];
    int quietEnd = end + (end - endRange[0]);
    ErrorHandler err_handler;
    if (quietEnd >= row->getSize() || !row->isRange(end, quietEnd, false, err_handler)) {
        return Ref<Result>(NULL);
    }
    if (err_handler.ErrCode()) return Ref<Result>(NULL);
    
    // https:// code.google.com/p/zxing/issues/detail?id=1736
    // UPC/EAN should never be less than 8 chars anyway
    if (result.length() < 8) {
        return Ref<Result>(NULL);
    }
    
    Ref<String> resultString (new String(result));
    if (!checkChecksum(resultString)) {
        return Ref<Result>(NULL);
    }
    
    float left = static_cast<float>(startGuardRange[1] + startGuardRange[0]) / 2.0f;
    float right = static_cast<float>(endRange[1] + endRange[0]) / 2.0f;
    BarcodeFormat format = getBarcodeFormat();
    ArrayRef< Ref<ResultPoint> > resultPoints(2);
    resultPoints[0] = Ref<ResultPoint>(new OneDResultPoint(left, static_cast<float>(rowNumber)));
    resultPoints[1] = Ref<ResultPoint>(new OneDResultPoint(right, static_cast<float>(rowNumber)));
    Ref<Result> decodeResult(new Result(resultString, ArrayRef<char>(), resultPoints, format));
    // Java extension and man stuff
    return decodeResult;
}

UPCEANReader::Range UPCEANReader::findStartGuardPattern(Ref<BitArray> row, ONED_READER_DATA* onedReaderData, ErrorHandler &err_handler) {
    bool foundStart = false;
    Range startRange;
    int nextStart = 0;
    vector<int> counters(START_END_PATTERN.size(), 0);
    
    while (!foundStart) {
        for (size_t i = 0; i < START_END_PATTERN.size(); ++i) {
            counters[i] = 0;
        }
        startRange = findGuardPattern(row, nextStart, false, START_END_PATTERN, counters, onedReaderData);
        if (startRange.isValid() == false) {
            return startRange;
        }
        
        int start = startRange[0];
        nextStart = startRange[1];
        // Make sure there is a quiet zone at least as big as the start pattern before the barcode.
        // If this check would run off the left edge of the image, do not accept this barcode,
        // as it is very likely to be a false positive.
        int quietStart = start - (nextStart - start);
        if (quietStart >= 0) {
            foundStart = row->isRange(quietStart, start, false, err_handler);
            if (err_handler.ErrCode()) return startRange;
        }
    }
    return startRange;
}

UPCEANReader::Range UPCEANReader::findGuardPattern(Ref<BitArray> row,
                                                   int rowOffset,
                                                   bool whiteFirst,
                                                   vector<int> const& pattern,
                                                   ONED_READER_DATA* onedReaderData) {
    vector<int> counters (pattern.size(), 0);
    return findGuardPattern(row, rowOffset, whiteFirst, pattern, counters, onedReaderData);
}

UPCEANReader::Range UPCEANReader::findGuardPattern(Ref<BitArray> row,
                                                   int rowOffset,
                                                   bool whiteFirst,
                                                   vector<int> const& pattern,
                                                   vector<int>& counters,
                                                   ONED_READER_DATA* onedReaderData) {
    int patternLength = pattern.size();
    int patternStart = whiteFirst ? row->getNextUnset(rowOffset) : row->getNextSet(rowOffset);
    if (patternStart == row->getSize())
    {
        return UPCEANReader::Range(false);
    }
    int counterOffset = 0;
    int patternOffset = 0;
    
    while (counterOffset<onedReaderData->counter_size-1&&patternOffset<patternStart){
        counterOffset++;
        patternOffset = onedReaderData->all_counters_offsets[counterOffset];
    }
    
    for (int c = counterOffset; c < onedReaderData->counter_size - patternLength + 1; c += 2){
        int x = patternStart;
        if (c == counterOffset){
            counters[0] = onedReaderData->all_counters[c] - (patternStart - patternOffset);
            x += counters[0];
            for (int ii = 1; ii < patternLength; ii++) {
                counters[ii] = onedReaderData->all_counters[c+ii];
                x += counters[ii];
            }
        }
        else
        {
            for (int ii = 0; ii < patternLength; ii++){
                counters[ii] = onedReaderData->all_counters[c + ii];
                x += counters[ii];
            }
        }
        
        if (patternMatchVariance(counters, pattern, MAX_INDIVIDUAL_VARIANCE) < MAX_AVG_VARIANCE) {
            return Range(patternStart, x);
        }
        patternStart += counters[0] + counters[1];
    }
    
    return UPCEANReader::Range(false);
}

UPCEANReader::Range UPCEANReader::decodeEnd(Ref<BitArray> row, int endStart) {
    return findGuardPattern(row, endStart, false, START_END_PATTERN, _onedReaderData);
}

#ifdef USE_PRE_BESTMATCH
UPCEANReader::DigitResult UPCEANReader::decodeDigit(Ref<BitArray> row,
                                                    vector<int> & counters,
                                                    int rowOffset,
                                                    vector<int const*> const& patterns,
                                                    ONED_READER_DATA* onedReaderData) {
    
    DigitResult digitResult;
    
    //  -- Added by Valiantliu
    //  -- Speed up if we already have the result
    int idxLG = -1;
    if (patterns.size() == L_PATTERNS.size())
    {
        idxLG = 0;
    }
    else if (patterns.size() == L_AND_G_PATTERNS.size())
    {
        idxLG = 1;
    }
    else
    {
        std::cout<<"******************** ERROR HERE!!"<<std::endl;
        digitResult.bestMatch = -100;
        digitResult.counterOffset = -100;
        return digitResult;
    }
    
    int preBestMatch = onedReaderData->digitResultCache[rowOffset].bestMatch[idxLG];
    int preCounterOffset = onedReaderData->digitResultCache[rowOffset].counterOffset;
    
    //  -- Already has result
    if (preBestMatch > -2 && preCounterOffset > -1)
    {
        digitResult.bestMatch = preBestMatch;
        digitResult.counterOffset = preCounterOffset;
        
        return digitResult;
    }
    
    bool rp = recordPattern(row, rowOffset, counters, onedReaderData);
    
    if (rp == false) {
        digitResult.bestMatch = -1;
        digitResult.counterOffset = preCounterOffset;
        
        return digitResult;
    }
    
    int counterOffset = getCounterOffset(counters);
    
    int bestVariance = MAX_AVG_VARIANCE;  // worst variance we'll accept
    int bestMatch = -1;
    
    int max = patterns.size();
    
    for (int i = 0; i < max; i++) {
        int const* pattern(patterns[i]);
        int variance = patternMatchVariance(counters, pattern, MAX_INDIVIDUAL_VARIANCE);
        if (variance < bestVariance) {
            bestVariance = variance;
            bestMatch = i;
        }
    }
    onedReaderData->digitResultCache[rowOffset].bestMatch[idxLG] = bestMatch;
    
    //  -- Added by Valiantliu
    onedReaderData->digitResultCache[rowOffset].counterOffset = counterOffset;
    
    digitResult.bestMatch = bestMatch;
    digitResult.counterOffset = counterOffset;
    
    return digitResult;
}
#else
int UPCEANReader::decodeDigit(Ref<BitArray> row,
                              vector<int> & counters,
                              int rowOffset,
                              vector<int const*> const& patterns,
                              ErrorHandler & err_handler) {
    recordPattern(row, rowOffset, counters);
    int bestVariance = MAX_AVG_VARIANCE;  // worst variance we'll accept
    int bestMatch = -1;
    int max = patterns.size();
    for (int i = 0; i < max; i++) {
        int const* pattern(patterns[i]);
        int variance = patternMatchVariance(counters, pattern, MAX_INDIVIDUAL_VARIANCE);
        if (variance < bestVariance) {
            bestVariance = variance;
            bestMatch = i;
        }
    }
    
    if (bestMatch >= 0)
    {
        return bestMatch;
    }
    else
    {
        err_handler = NotFoundErrorHandler(-1);
        return -1;
    }
}
#endif

/**
 * @return {@link #checkStandardUPCEANChecksum(String)}
 */
bool UPCEANReader::checkChecksum(Ref<String> const& s) {
    return checkStandardUPCEANChecksum(s);
}

/**
 * Computes the UPC/EAN checksum on a string of digits, and reports
 * whether the checksum is correct or not.
 *
 * @param s string of digits to check
 * @return true iff string of digits passes the UPC/EAN checksum algorithm
 */
bool UPCEANReader::checkStandardUPCEANChecksum(Ref<String> const& s_) {
    std::string const& s (s_->getText());
    int length = s.length();
    if (length == 0) {
        return false;
    }
    
    int sum = 0;
    for (int i = length - 2; i >= 0; i -= 2) {
        int digit = static_cast<int>(s[i]) - static_cast<int>('0');
        if (digit < 0 || digit > 9) {
            return false;
        }
        sum += digit;
    }
    sum *= 3;
    for (int i = length - 1; i >= 0; i -= 2) {
        int digit = static_cast<int>(s[i]) - static_cast<int>('0');
        if (digit < 0 || digit > 9) {
            return false;
        }
        sum += digit;
    }
    return sum % 10 == 0;
}

/**
 * @return {@link #getChecksum(String)}
 */
int UPCEANReader::getChecksum(Ref<String> const& s) {
    return getStandardUPCEANChecksum(s);
}

int UPCEANReader::getStandardUPCEANChecksum(Ref<String> const& s_)
{
    std::string const& s (s_->getText());
    int length = s.length();
    if (length == 0) {
        return false;
    }
    
    int sum = 0;
    for (int i = length - 1; i >= 0; i -= 2) {
        int digit = static_cast<int>(s[i]) - static_cast<int>('0');
        if (digit < 0 || digit > 9) {
            return -1;
        }
        sum += digit;
    }
    sum *= 3;
    for (int i = length - 2; i >= 0; i -= 2) {
        int digit = static_cast<int>(s[i]) - static_cast<int>('0');
        if (digit < 0 || digit > 9) {
            return -1;
        }
        sum += digit;
    }
    return (10 - sum % 10);
}

UPCEANReader::~UPCEANReader() {
}
