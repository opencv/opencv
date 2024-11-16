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

#include "ean13reader.hpp"
#include "../not_found_exception.hpp"

#include "one_dconstant.hpp"

using namespace zxing;
using namespace oned;
using namespace zxing::oned::constant::UPCEAN;
using namespace zxing::oned::constant::EAN13;

using std::vector;
using zxing::Ref;
using zxing::BitArray;
using zxing::oned::EAN13Reader;

EAN13Reader::EAN13Reader() : decodeMiddleCounters(4, 0) { }

int EAN13Reader::decodeMiddle(Ref<BitArray> row,
                              Range const& startRange,
                              std::string& resultString) {
    vector<int>& counters (decodeMiddleCounters);
    counters.clear();
    counters.resize(4);
    int end = row->getSize();
    int rowOffset = startRange[1];
    
    int lgPatternFound = 0;
    
    for (int x = 0; x < 6 && rowOffset < end; x++) {
#ifdef USE_PRE_BESTMATCH
        DigitResult digitResult = decodeDigit(row, counters, rowOffset, L_AND_G_PATTERNS, _onedReaderData);
        int bestMatch = digitResult.bestMatch;
        int counterOffset = digitResult.counterOffset;
#else
        int bestMatch = decodeDigit(row, counters, rowOffset, L_AND_G_PATTERNS);
        
#endif
        
        // To decrease throw times, use this instead
        if (bestMatch < 0) {
            return -1;
        }
        
        resultString.append(1, static_cast<char>('0' + bestMatch % 10));
#ifdef USE_PRE_BESTMATCH
        rowOffset += counterOffset;
#else
        for (int i = 0, end = counters.size(); i <end; i++) {
            rowOffset += counters[i];
        }
#endif
        if (bestMatch >= 10) {
            lgPatternFound |= 1 << (5 - x);
        }
    }
    
    _onedReaderData->ean13_decode_middle_middle_offset = rowOffset;
    _onedReaderData->ean13_decode_middle_middle_string = resultString;
    
    ErrorHandler err_handler;
    determineFirstDigit(resultString, lgPatternFound, err_handler);
    if (err_handler.ErrCode()) return -1;
    
    Range middleRange = findGuardPattern(row, rowOffset, true, MIDDLE_PATTERN, _onedReaderData);
    if (middleRange.isValid() == false) {
        return -1;
    }
    rowOffset = middleRange[1];
    
    for (int x = 0; x < 6 && rowOffset < end; x++) {
#ifdef USE_PRE_BESTMATCH
        DigitResult digitResult = decodeDigit(row, counters, rowOffset, L_PATTERNS, _onedReaderData);
        int bestMatch = digitResult.bestMatch;
        int counterOffset = digitResult.counterOffset;
#else
        int bestMatch =
        decodeDigit(row, counters, rowOffset, L_PATTERNS);
#endif
        
        // To decrease throw times, use this instead
        if (bestMatch < 0) {
            return -1;
        }
        
        resultString.append(1, static_cast<char>('0' + bestMatch));
#ifdef USE_PRE_BESTMATCH
        rowOffset += counterOffset;
#else
        for (int i = 0, end = counters.size(); i < end; i++) {
            rowOffset += counters[i];
        }
#endif
    }
    
    _onedReaderData->ean13_checked =true;
    _onedReaderData->ean13_lg_pattern_found = lgPatternFound;
    _onedReaderData->ean13_decode_middle_final_offset = rowOffset;
    _onedReaderData->ean13_decode_middle_final_string = resultString;
    
    return rowOffset;
}

void EAN13Reader::determineFirstDigit(std::string& resultString, int lgPatternFound, ErrorHandler & err_handler) {
    for (int d = 0; d < 10; d++) {
        if (lgPatternFound == FIRST_DIGIT_ENCODINGS[d]) {
            resultString.insert((std::string::size_type)0, (std::string::size_type)1, static_cast<char>('0' + d));
            return;
        }
    }

    err_handler = NotFoundErrorHandler(-1);
}

zxing::BarcodeFormat EAN13Reader::getBarcodeFormat(){
    return BarcodeFormat::EAN_13;
}
