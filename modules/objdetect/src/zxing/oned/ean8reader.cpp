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

#include "ean8reader.hpp"
#include "../reader_exception.hpp"

using std::vector;
using zxing::oned::EAN8Reader;

// VC++
using zxing::Ref;
using zxing::BitArray;

EAN8Reader::EAN8Reader() : decodeMiddleCounters(4, 0) {}

int EAN8Reader::decodeMiddle(Ref<BitArray> row,
                             Range const& startRange,
                             std::string& result){
    vector<int>& counters (decodeMiddleCounters);
    counters[0] = 0;
    counters[1] = 0;
    counters[2] = 0;
    counters[3] = 0;
    
    int end = row->getSize();
    int rowOffset = startRange[1];
    
    for (int x = 0; x < 4 && rowOffset < end; x++) {
#ifdef USE_PRE_BESTMATCH
        DigitResult digitResult = decodeDigit(row, counters, rowOffset, L_PATTERNS, _onedReaderData);
        int bestMatch = digitResult.bestMatch;
        int counterOffset = digitResult.counterOffset;
#else
        int bestMatch = decodeDigit(row, counters, rowOffset, L_PATTERNS);
#endif
        
        // To decrease throw times, use this instead
        if (bestMatch < 0) {
            return -1;
        }
        
        result.append(1, static_cast<char>('0' + bestMatch));
#ifdef USE_PRE_ROWOFFSET
        rowOffset += counterOffset;
#else
        for (int i = 0, end = counters.size(); i < end; i++) {
            rowOffset += counters[i];
        }
#endif
    }
    
    Range middleRange = findGuardPattern(row, rowOffset, true, MIDDLE_PATTERN, _onedReaderData);
    
    if (middleRange.isValid() == false) {
        return -1;
    }
    rowOffset = middleRange[1];
    for (int x = 0; x < 4 && rowOffset < end; x++) {
#ifdef USE_PRE_BESTMATCH
        DigitResult digitResult = decodeDigit(row, counters, rowOffset, L_PATTERNS, _onedReaderData);
        int bestMatch = digitResult.bestMatch;
        int counterOffset = digitResult.counterOffset;
#else
        int bestMatch = decodeDigit(row, counters, rowOffset, L_PATTERNS);
#endif
        
        // To decrease throw times, use this instead
        if (bestMatch < 0) {
            return -1;
        }
        
        result.append(1, static_cast<char>('0' + bestMatch));
#ifdef USE_PRE_ROWOFFSET
        rowOffset += counterOffset;
#else
        for (int i = 0, end = counters.size(); i < end; i++) {
            rowOffset += counters[i];
        }
#endif
    }
    return rowOffset;
}

zxing::BarcodeFormat EAN8Reader::getBarcodeFormat(){
    return BarcodeFormat::EAN_8;
}
