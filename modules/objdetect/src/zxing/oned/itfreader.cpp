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
#include "itfreader.hpp"
#include "one_dresult_point.hpp"
#include "../common/array.hpp"
#include "../reader_exception.hpp"
#include "../format_exception.hpp"
#include "../not_found_exception.hpp"
#include <math.h>

using std::vector;
using zxing::Ref;
using zxing::ArrayRef;
using zxing::Array;
using zxing::Result;
using zxing::FormatException;
using zxing::NotFoundException;
using zxing::oned::ITFReader;

// VC++
using zxing::BitArray;

#include "one_dconstant.hpp"
using namespace zxing;
using namespace oned;

using namespace zxing::oned::constant::ITF;

ITFReader::ITFReader() : narrowLineWidth(-1) {
}


Ref<Result> ITFReader::decodeRow(int rowNumber, Ref<BitArray> row) {
    // Find out where the Middle section (payload) starts & ends
    
    Range startRange = decodeStart(row);
    
    if (startRange.isValid() == false) {
        return Ref<Result>(NULL);
    }
    
    Range endRange = decodeEnd(row);
    
    if (endRange.isValid() == false) {
        return Ref<Result>(NULL);
    }
    
    
    // To speed up get next sets function
    row->initAllNextSets();
    
    std::string result;
    int endStart = decodeMiddle(row, startRange[1], endRange[0], result);
    
    if (endStart < 0) {
        return Ref<Result>(NULL);
    }
    
    Ref<String> resultString(new String(result));
    
    ArrayRef<int> allowedLengths;
    // Java hints stuff missing
    if (!allowedLengths) {
        allowedLengths = DEFAULT_ALLOWED_LENGTHS;
    }
    
    // To avoid false positives with 2D barcodes (and other patterns), make
    // an assumption that the decoded string must be 6, 10 or 14 digits.
    int length = resultString->size();
    bool lengthOK = false;
    for (int i = 0, e = allowedLengths->size(); i < e; i++) {
        if (length == allowedLengths[i]) {
            lengthOK = true;
            break;
        }
    }
    
    if (!lengthOK) {
        return Ref<Result>(NULL);
    }
    
    ArrayRef< Ref<ResultPoint> > resultPoints(2);
    resultPoints[0] =
    Ref<OneDResultPoint>(new OneDResultPoint(static_cast<float>(startRange[1]), static_cast<float>(rowNumber)));
    resultPoints[1] =
    Ref<OneDResultPoint>(new OneDResultPoint(static_cast<float>(endRange[0]), static_cast<float>(rowNumber)));
    return Ref<Result>(new Result(resultString, ArrayRef<char>(), resultPoints, BarcodeFormat::ITF));
}

/**
 * @param row          row of black/white values to search
 * @param payloadStart offset of start pattern
 * @param resultString {@link StringBuffer} to append decoded chars to
 * @throws ReaderException if decoding could not complete successfully
 */
int ITFReader::decodeMiddle(Ref<BitArray> row,
                            int payloadStart,
                            int payloadEnd,
                            std::string& resultString) {
    // Digits are interleaved in pairs - 5 black lines for one digit, and the
    // 5
    // interleaved white lines for the second digit.
    // Therefore, need to scan 10 lines and then
    // split these into two arrays
    vector<int> counterDigitPair(10, 0);
    vector<int> counterBlack(5, 0);
    vector<int> counterWhite(5, 0);
    
    while (payloadStart < payloadEnd) {
        
        // Get 10 runs of black/white.
        bool rp = recordPattern(row, payloadStart, counterDigitPair, _onedReaderData);
        
        if (rp == false) {
            return -1;
        }
        
        // Split them into each array
        for (int k = 0; k < 5; k++) {
            int twoK = k << 1;
            counterBlack[k] = counterDigitPair[twoK];
            counterWhite[k] = counterDigitPair[twoK + 1];
        }
        
        int bestMatch = decodeDigit(counterBlack);
        
        
        // To decrease throw times, use this instead
        if (bestMatch < 0) {
            return -1;
        }
        
        resultString.append(1, static_cast<char>('0' + bestMatch));
        bestMatch = decodeDigit(counterWhite);
        
        // To decrease throw times, use this instead
        if (bestMatch < 0) {
            return -1;
        }
        
        resultString.append(1, static_cast<char>('0' + bestMatch));
        
        for (int i = 0, e = counterDigitPair.size(); i < e; i++) {
            payloadStart += counterDigitPair[i];
        }
    }
    
    return 1;
}

/**
 * Identify where the start of the middle / payload section starts.
 *
 * @param row row of black/white values to search
 * @return Array, containing index of start of 'start block' and end of
 *         'start block'
 * @throws ReaderException
 */
ITFReader::Range ITFReader::decodeStart(Ref<BitArray> row) {
    int endStart = skipWhiteSpace(row);
    
    if (endStart < 0) {
        return Range(false);
    }
    
    Range startPattern = findGuardPattern(row, endStart, START_PATTERN_VECTOR, _onedReaderData);
    
    if (startPattern.isValid() == false) {
        return startPattern;
    }
    
    // Determine the width of a narrow line in pixels. We can do this by
    // getting the width of the start pattern and dividing by 4 because its
    // made up of 4 narrow lines.
    narrowLineWidth = (startPattern[1] - startPattern[0]) >> 2;
    
    bool isValidQZ = validateQuietZone(row, startPattern[0]);
    
    if (isValidQZ==false) {
        return Range(false);
    }
    return startPattern;
}

/**
 * Identify where the end of the middle / payload section ends.
 *
 * @param row row of black/white values to search
 * @return Array, containing index of start of 'end block' and end of 'end
 *         block'
 * @throws ReaderException
 */

ITFReader::Range ITFReader::decodeEnd(Ref<BitArray> row) {
    // For convenience, reverse the row and then
    // search from 'the start' for the end block
    BitArray::Reverse r(row);
    
    
    reverseAllPattern(_onedReaderData);
    
    
    // To speed up get next sets function
    row->initAllNextSets();

    int endStart = skipWhiteSpace(row);
    Range endPattern = findGuardPattern(row, endStart, END_PATTERN_REVERSED, _onedReaderData);
    
    if (endPattern.isValid() == false) {
        return endPattern;
    }
    
    
    // The start & end patterns must be pre/post fixed by a quiet zone. This
    // zone must be at least 10 times the width of a narrow line.
    // ref: http:// www.barcode-1.net/i25code.html
    bool isValidQZ = validateQuietZone(row, endPattern[0]);
    
    if (isValidQZ==false) {
        return Range(false);
    }
    
    // Now recalculate the indices of where the 'endblock' starts & stops to
    // accommodate
    // the reversed nature of the search
    int temp = endPattern[0];
    endPattern[0] = row->getSize() - endPattern[1];
    endPattern[1] = row->getSize() - temp;
    
    reverseAllPattern(_onedReaderData);
    return endPattern;
}

/**
 * The start & end patterns must be pre/post fixed by a quiet zone. This
 * zone must be at least 10 times the width of a narrow line.  Scan back until
 * we either get to the start of the barcode or match the necessary number of
 * quiet zone pixels.
 *
 * Note: Its assumed the row is reversed when using this method to find
 * quiet zone after the end pattern.
 *
 * ref: http:// www.barcode-1.net/i25code.html
 *
 * @param row bit array representing the scanned barcode.
 * @param startPattern index into row of the start or end pattern.
 * @throws ReaderException if the quiet zone cannot be found, a ReaderException is thrown.
 */
bool ITFReader::validateQuietZone(Ref<BitArray> row, int startPattern) {
    int quietCount = this->narrowLineWidth * 10;  // expect to find this many pixels of quiet zone
    
    for (int i = startPattern - 1; quietCount > 0 && i >= 0; i--) {
        if (row->get(i)) {
            break;
        }
        quietCount--;
    }
    if (quietCount != 0) {
        // Unable to find the necessary number of quiet zone pixels.
        return false;
    }
    
    return true;
}

/**
 * Skip all whitespace until we get to the first black line.
 *
 * @param row row of black/white values to search
 * @return index of the first black line.
 * @throws ReaderException Throws exception if no black lines are found in the row
 */
int ITFReader::skipWhiteSpace(Ref<BitArray> row) {
    int width = row->getSize();
    int endStart = row->getNextSet(0);
    if (endStart == width) {
        return -1;
    }
    return endStart;
}

/**
 * @param row       row of black/white values to search
 * @param rowOffset position to start search
 * @param pattern   pattern of counts of number of black and white pixels that are
 *                  being searched for as a pattern
 * @return start/end horizontal offset of guard pattern, as an array of two
 *         ints
 * @throws ReaderException if pattern is not found
 */
ITFReader::Range ITFReader::findGuardPattern(Ref<BitArray> row,
                                             int rowOffset,
                                             vector<int> const& pattern,
                                             ONED_READER_DATA* onedReaderData) {
    if (rowOffset == row->getSize())
    {
        return Range(false);
    }
    
    int patternLength = pattern.size();
    vector<int> counters(patternLength);
    int counterOffset=0;
    int patternOffset=0;
    bool isWhite= onedReaderData->first_is_white;
    
    
    while (counterOffset<onedReaderData->counter_size-1&&patternOffset<rowOffset){
        counterOffset++;
        patternOffset = onedReaderData->all_counters_offsets[counterOffset];
        isWhite = !isWhite;
    }
    
    if (isWhite){
        counters[0] = 0;
    }
    else
    {
        counters[0]= onedReaderData->all_counters[counterOffset]-(rowOffset-patternOffset);
    }
    int x=rowOffset+counters[0];
    
    for (int c = counterOffset; c < onedReaderData->counter_size-patternLength+1; c+=2){
        if (c==counterOffset){
            for (int ii=1; ii < patternLength; ii++){
                counters[ii] = onedReaderData->all_counters[c+ii];
                x+=counters[ii];
            }
        }
        else
        {
            for (int ii = 0; ii < patternLength; ii++){
                counters[ii] = onedReaderData->all_counters[c+ii];
                x+=counters[ii];
            }
        }
        
        if (patternMatchVariance(counters, &pattern[0], MAX_INDIVIDUAL_VARIANCE) < MAX_AVG_VARIANCE) {
            return Range(rowOffset, x);
        }
        
        rowOffset += counters[0]+counters[1];
    }
    return Range(false);
}

/**
 * Attempts to decode a sequence of ITF black/white lines into single
 * digit.
 *
 * @param counters the counts of runs of observed black/white/black/... values
 * @return The decoded digit
 * @throws ReaderException if digit cannot be decoded
 */
int ITFReader::decodeDigit(vector<int>& counters){
    
    int bestVariance = MAX_AVG_VARIANCE;  // worst variance we'll accept
    int bestMatch = -1;
    int max = sizeof(PATTERNS)/sizeof(PATTERNS[0]);
    for (int i = 0; i < max; i++) {
        int const* pattern = PATTERNS[i];
        int variance = patternMatchVariance(counters, pattern, MAX_INDIVIDUAL_VARIANCE);
        if (variance < bestVariance)
        {
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
        return -1;
    }
}

ITFReader::~ITFReader(){}

