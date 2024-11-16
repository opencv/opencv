/*
 *  FormatInformation.cpp
 *  zxing
 *
 *  Created by Christian Brunschen on 18/05/2008.
 *  Copyright 2008 ZXing authors All rights reserved.
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

#include "format_information.hpp"
#include <limits>

using zxing::ErrorHandler;

namespace zxing {
namespace qrcode {

int FormatInformation::FORMAT_INFO_MASK_QR = 0x5412;
int FormatInformation::FORMAT_INFO_DECODE_LOOKUP[][2] = { { 0x5412, 0x00 }, { 0x5125, 0x01 }, { 0x5E7C, 0x02 }, {
    0x5B4B, 0x03 }, { 0x45F9, 0x04 }, { 0x40CE, 0x05 }, { 0x4F97, 0x06 }, { 0x4AA0, 0x07 }, { 0x77C4, 0x08 }, {
        0x72F3, 0x09 }, { 0x7DAA, 0x0A }, { 0x789D, 0x0B }, { 0x662F, 0x0C }, { 0x6318, 0x0D }, { 0x6C41, 0x0E }, {
            0x6976, 0x0F }, { 0x1689, 0x10 }, { 0x13BE, 0x11 }, { 0x1CE7, 0x12 }, { 0x19D0, 0x13 }, { 0x0762, 0x14 }, {
                0x0255, 0x15 }, { 0x0D0C, 0x16 }, { 0x083B, 0x17 }, { 0x355F, 0x18 }, { 0x3068, 0x19 }, { 0x3F31, 0x1A }, {
                    0x3A06, 0x1B }, { 0x24B4, 0x1C }, { 0x2183, 0x1D }, { 0x2EDA, 0x1E }, { 0x2BED, 0x1F },
};
int FormatInformation::N_FORMAT_INFO_DECODE_LOOKUPS = 32;

int FormatInformation::BITS_SET_IN_HALF_BYTE[] = { 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4 };

FormatInformation::FormatInformation(int formatInfo, float possiableFix, ErrorHandler & err_handler) :
errorCorrectionLevel_(ErrorCorrectionLevel::forBits((formatInfo >> 3) & 0x03, err_handler)), dataMask_(static_cast<char>(formatInfo & 0x07)) {
    possiableFix_ = possiableFix;
    if (err_handler.ErrCode())
        return;
}

ErrorCorrectionLevel& FormatInformation::getErrorCorrectionLevel() {
    return errorCorrectionLevel_;
}

char FormatInformation::getDataMask() {
    return dataMask_;
}

float FormatInformation::getPossiableFix() {
    return possiableFix_;
}

int FormatInformation::numBitsDiffering(int a, int b) {
    a ^= b;
    return BITS_SET_IN_HALF_BYTE[a & 0x0F] + BITS_SET_IN_HALF_BYTE[(a >> 4 & 0x0F)] + BITS_SET_IN_HALF_BYTE[(a >> 8
                                                                                                             & 0x0F)] + BITS_SET_IN_HALF_BYTE[(a >> 12 & 0x0F)] + BITS_SET_IN_HALF_BYTE[(a >> 16 & 0x0F)]
    + BITS_SET_IN_HALF_BYTE[(a >> 20 & 0x0F)] + BITS_SET_IN_HALF_BYTE[(a >> 24 & 0x0F)]
    + BITS_SET_IN_HALF_BYTE[(a >> 28 & 0x0F)];
}

Ref<FormatInformation> FormatInformation::decodeFormatInformation(int maskedFormatInfo1, int maskedFormatInfo2) {
    Ref<FormatInformation> result(doDecodeFormatInformation(maskedFormatInfo1, maskedFormatInfo2));
    if (result != 0) {
        return result;
    }
    // Should return null, but, some QR codes apparently
    // do not mask this info. Try again by actually masking the pattern
    // first
    return doDecodeFormatInformation(maskedFormatInfo1 ^ FORMAT_INFO_MASK_QR,
                                     maskedFormatInfo2  ^ FORMAT_INFO_MASK_QR);
}
Ref<FormatInformation> FormatInformation::doDecodeFormatInformation(int maskedFormatInfo1, int maskedFormatInfo2)
{
    ErrorHandler err_handler;
    
    int distance = numBitsDiffering(maskedFormatInfo1, maskedFormatInfo2);
    float possiableFix_ = (16.0 - (distance > 16 ? 16 : distance)) / 16.0;
    
    // Find the int in FORMAT_INFO_DECODE_LOOKUP with fewest bits differing
    int bestDifference = std::numeric_limits<int>::max();
    int bestFormatInfo = 0;
    for (int i = 0; i < N_FORMAT_INFO_DECODE_LOOKUPS; i++) {
        int* decodeInfo = FORMAT_INFO_DECODE_LOOKUP[i];
        int targetInfo = decodeInfo[0];
        if (targetInfo == maskedFormatInfo1 || targetInfo == maskedFormatInfo2) {
            // Found an exact match
            Ref<FormatInformation> result(new FormatInformation(decodeInfo[1], possiableFix_, err_handler));
            if (err_handler.ErrCode())
                return Ref<FormatInformation>();
            return result;
        }
        int bitsDifference = numBitsDiffering(maskedFormatInfo1, targetInfo);
        if (bitsDifference < bestDifference) {
            bestFormatInfo = decodeInfo[1];
            bestDifference = bitsDifference;
        }
        if (maskedFormatInfo1 != maskedFormatInfo2) {
            // also try the other option
            bitsDifference = numBitsDiffering(maskedFormatInfo2, targetInfo);
            if (bitsDifference < bestDifference) {
                bestFormatInfo = decodeInfo[1];
                bestDifference = bitsDifference;
            }
        }
    }
    if (bestDifference <= 3) {
        Ref<FormatInformation> result(new FormatInformation(bestFormatInfo, possiableFix_, err_handler));
        if (err_handler.ErrCode())
            return Ref<FormatInformation>();
        return result;
    }
    Ref<FormatInformation> result;
    return result;
}

bool operator==(const FormatInformation &a, const FormatInformation &b) {
    return &(a.errorCorrectionLevel_) == &(b.errorCorrectionLevel_) && a.dataMask_ == b.dataMask_;
}

std::ostream& operator<<(std::ostream& out, const FormatInformation& fi) {
    const FormatInformation *fip = &fi;
    out << "FormatInformation @ " << fip;
    return out;
}

}  // namespace qrcode
}  // namespace zxing
