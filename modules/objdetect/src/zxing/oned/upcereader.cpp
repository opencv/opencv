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
#include "upcereader.hpp"
#include "../reader_exception.hpp"

using std::string;
using std::vector;
using zxing::Ref;
using zxing::String;
using zxing::oned::UPCEReader;

// VC++
using zxing::BitArray;

#define VECTOR_INIT(v) v, v + sizeof(v)/sizeof(v[0])

namespace {
/**
 * The pattern that marks the middle, and end, of a UPC-E pattern.
 * There is no "second half" to a UPC-E barcode.
 */
const int MIDDLE_END_PATTERN_[6] = {1, 1, 1, 1, 1, 1};
const vector<int> MIDDLE_END_PATTERN (VECTOR_INIT(MIDDLE_END_PATTERN_));


/**
 * See {@link #L_AND_G_PATTERNS}; these values similarly represent patterns of
 * even-odd parity encodings of digits that imply both the number system (0 or 1)
 * used, and the check digit.
 */
const int NUMSYS_AND_CHECK_DIGIT_PATTERNS[2][10] = {
    {0x38, 0x34, 0x32, 0x31, 0x2C, 0x26, 0x23, 0x2A, 0x29, 0x25},
    {0x07, 0x0B, 0x0D, 0x0E, 0x13, 0x19, 0x1C, 0x15, 0x16, 0x1A}
};
}  // namespace

UPCEReader::UPCEReader() {
}

int UPCEReader::decodeMiddle(Ref<BitArray> row, Range const& startRange, string& result) {
    
    if (_onedReaderData->ean13_checked){
        result = _onedReaderData->ean13_decode_middle_middle_string;
        int rowOffset = _onedReaderData->ean13_decode_middle_middle_offset;
        int lgPatternFound = _onedReaderData->ean13_lg_pattern_found;
        determineNumSysAndCheckDigit(result, lgPatternFound);
        return rowOffset;
    }
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
        
        result.append(1, static_cast<char>('0' + bestMatch % 10));
        
#ifdef USE_PRE_BESTMATCH
        rowOffset += counterOffset;
#else
        for (int i = 0, e = counters.size(); i < e; i++) {
            rowOffset += counters[i];
        }
#endif
        if (bestMatch >= 10) {
            lgPatternFound |= 1 << (5 - x);
        }
    }
    
    determineNumSysAndCheckDigit(result, lgPatternFound);
    
    return rowOffset;
}

UPCEReader::Range UPCEReader::decodeEnd(Ref<BitArray> row, int endStart) {
    return findGuardPattern(row, endStart, true, MIDDLE_END_PATTERN, _onedReaderData);
}

bool UPCEReader::checkChecksum(Ref<String> const& s){
    // Special check for UPC-E
    // System code must be 0 or 1
    // http:// www.barcodeisland.com/upce.phtml
    // By Skylook
    
    string const& upce(s->getText());
    char firstChar = upce[0];
    
    // Additionally, UPC-E may only be used if the number system is 0 or 1.
    if (firstChar == '0' || firstChar == '1')
    {
        Ref<String> sa = convertUPCEtoUPCA(s);
        return UPCEANReader::checkChecksum(sa);
    }
    else
    {
        return false;
    }
}


bool UPCEReader::determineNumSysAndCheckDigit(std::string& resultString, int lgPatternFound) {
    for (int numSys = 0; numSys <= 1; numSys++) {
        for (int d = 0; d < 10; d++) {
            if (lgPatternFound == NUMSYS_AND_CHECK_DIGIT_PATTERNS[numSys][d]) {
                resultString.insert((string::size_type)0, (string::size_type)1, static_cast<char>('0' + numSys));
                resultString.append(1, static_cast<char>('0' + d));
                return true;
            }
        }
    }
    return false;
}

/**
 * Expands a UPC-E value back into its full, equivalent UPC-A code value.
 *
 * @param upce UPC-E code as string of digits
 * @return equivalent UPC-A code as string of digits
 */
Ref<String> UPCEReader::convertUPCEtoUPCA(Ref<String> const& upce_) {
    
    string const& upce(upce_->getText());
    string result;
    result.append(1, upce[0]);
    char lastChar = upce[6];
    switch (lastChar) {
        case '0':
        case '1':
        case '2':
            result.append(upce.substr(1, 2));
            result.append(1, lastChar);
            result.append("0000");
            result.append(upce.substr(3, 3));
            break;
        case '3':
            result.append(upce.substr(1, 3));
            result.append("00000");
            result.append(upce.substr(4, 2));
            break;
        case '4':
            result.append(upce.substr(1, 4));
            result.append("00000");
            result.append(1, upce[5]);
            break;
        default:
            result.append(upce.substr(1, 5));
            result.append("0000");
            result.append(1, lastChar);
            break;
    }
    result.append(1, upce[7]);
    return Ref<String>(new String(result));
}

zxing::BarcodeFormat UPCEReader::getBarcodeFormat() {
    return BarcodeFormat::UPC_E;
}
