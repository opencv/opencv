// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").

/*
 *  Version.cpp
 *  zxing
 *
 *  Created by Christian Brunschen on 14/05/2008.
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

#include "version.hpp"
#include "format_information.hpp"
#include "../format_exception.hpp"
#include <limits>
#include <iostream>
#include <cstdarg>

using std::vector;
using std::numeric_limits;
using zxing::ErrorHandler;

namespace zxing {
namespace qrcode {

ECB::ECB(int count, int dataCodewords) :
count_(count), dataCodewords_(dataCodewords) {
}

int ECB::getCount() {
    return count_;
}

int ECB::getDataCodewords() {
    return dataCodewords_;
}

ECBlocks::ECBlocks(int ecCodewords, ECB *ecBlocks) :
ecCodewords_(ecCodewords), ecBlocks_(1, ecBlocks) {
}

ECBlocks::ECBlocks(int ecCodewords, ECB *ecBlocks1, ECB *ecBlocks2) :
ecCodewords_(ecCodewords), ecBlocks_(1, ecBlocks1) {
    ecBlocks_.push_back(ecBlocks2);
}

int ECBlocks::getECCodewords() {
    return ecCodewords_;
}

std::vector<ECB*>& ECBlocks::getECBlocks() {
    return ecBlocks_;
}

ECBlocks::~ECBlocks() {
    for (size_t i = 0; i < ecBlocks_.size(); i++) {
        delete ecBlocks_[i];
    }
}

unsigned int Version::VERSION_DECODE_INFO[] = { 0x07C94, 0x085BC, 0x09A99, 0x0A4D3, 0x0BBF6, 0x0C762, 0x0D847, 0x0E60D,
    0x0F928, 0x10B78, 0x1145D, 0x12A17, 0x13532, 0x149A6, 0x15683, 0x168C9, 0x177EC, 0x18EC4, 0x191E1, 0x1AFAB,
    0x1B08E, 0x1CC1A, 0x1D33F, 0x1ED75, 0x1F250, 0x209D5, 0x216F0, 0x228BA, 0x2379F, 0x24B0B, 0x2542E, 0x26A64,
    0x27541, 0x28C69
};
int Version::N_VERSION_DECODE_INFOS = 34;


vector<Ref<Version> > Version::VERSIONS;
static int N_VERSIONS = Version::buildVersions();

int Version::getVersionNumber() {
    return versionNumber_;
}

vector<int> &Version::getAlignmentPatternCenters() {
    return alignmentPatternCenters_;
}

int Version::getTotalCodewords() {
    return totalCodewords_;
}

int Version::getDimensionForVersion(ErrorHandler & err_handler) {
    
    // lincoln add
    if (versionNumber_ == 0) return MICRO_DEMITION;
    
    if (versionNumber_ < 0 || versionNumber_ > N_VERSIONS){
        err_handler = zxing::ReaderErrorHandler("versionNumber must be between 1 and 40");
        return -1;
    }
    return 17 + 4 * versionNumber_;
}

ECBlocks& Version::getECBlocksForLevel(ErrorCorrectionLevel &ecLevel) {
    return *ecBlocks_[ecLevel.ordinal()];
}

bool Version::checkECBlocksForLevel(ErrorCorrectionLevel &ecLevel) {
    if (ecBlocks_.size() <= 0) {
        return false;
    }
    if (ecLevel.ordinal() >= static_cast<int>(ecBlocks_.size())) return false;
    return true;
}


static vector<int> *intArray(size_t n...) {
    va_list ap;
    va_start(ap, n);
    vector<int> *result = new vector<int>(n);
    for (size_t i = 0; i < n; i++) {
        (*result)[i] = va_arg(ap, int);
    }
    va_end(ap);
    return result;
}

Version *Version::getProvisionalVersionForDimension(int dimension, ErrorHandler & err_handler) {
    
    // lincoln add
    if (dimension == MICRO_DEMITION)
    {
        return MICRO_VERSION;
    }
    
    if (dimension % 4 != 1) {
        err_handler = zxing::FormatErrorHandler("dimension % 4 != 1");
        return NULL;
    }
    
    Version * version = Version::getVersionForNumber((dimension - 17) >> 2, err_handler);
    if (err_handler.errCode())
    {
        err_handler = zxing::FormatErrorHandler("err format");
        return NULL;
    }
    return version;
    
}

Version *Version::getVersionForNumber(int versionNumber, ErrorHandler &err_handler) {
    // lincoln add
    if (versionNumber == 0)
    {
        return MICRO_VERSION;
    }
    if (versionNumber < 1 || versionNumber > N_VERSIONS) {
        err_handler = zxing::ReaderErrorHandler("versionNumber must be between 1 and 40");
        return NULL;
    }
    return VERSIONS[versionNumber - 1];
}

Version::Version(int versionNumber, vector<int> *alignmentPatternCenters, ECBlocks *ecBlocks1, ECBlocks *ecBlocks2,
                 ECBlocks *ecBlocks3, ECBlocks *ecBlocks4) :
versionNumber_(versionNumber), alignmentPatternCenters_(*alignmentPatternCenters), ecBlocks_(4), totalCodewords_(0) {
    ecBlocks_[0] = ecBlocks1;
    ecBlocks_[1] = ecBlocks2;
    ecBlocks_[2] = ecBlocks3;
    ecBlocks_[3] = ecBlocks4;
    
    int total = 0;
    int ecCodewords = ecBlocks1->getECCodewords();
    vector<ECB*> &ecbArray = ecBlocks1->getECBlocks();
    for (size_t i = 0; i < ecbArray.size(); i++) {
        ECB *ecBlock = ecbArray[i];
        total += ecBlock->getCount() * (ecBlock->getDataCodewords() + ecCodewords);
    }
    totalCodewords_ = total;
}

Version::~Version() {
    delete &alignmentPatternCenters_;
    for (size_t i = 0; i < ecBlocks_.size(); i++) {
        delete ecBlocks_[i];
    }
}

Version *Version::decodeVersionInformation(unsigned int versionBits) {
    int bestDifference = numeric_limits<int>::max();
    size_t bestVersion = 0;
    ErrorHandler err_handler;
    for (int i = 0; i < N_VERSION_DECODE_INFOS; i++) {
        unsigned targetVersion = VERSION_DECODE_INFO[i];
        // Do the version info bits match exactly? done.
        if (targetVersion == versionBits) {
            Version* version = getVersionForNumber(i + 7 , err_handler);
            if (err_handler.errCode())   return 0;
            return version;
        }
        // Otherwise see if this is the closest to a real version info bit
        // string we have seen so far
        int bitsDifference = FormatInformation::numBitsDiffering(versionBits, targetVersion);
        if (bitsDifference < bestDifference) {
            bestVersion = i + 7;
            bestDifference = bitsDifference;
        }
    }
    // We can tolerate up to 3 bits of error since no two version info codewords will
    // differ in less than 4 bits.
    if (bestDifference <= 3) {
        Version * version = getVersionForNumber(bestVersion , err_handler);
        if (err_handler.errCode())   return 0;
        return version;
    }
    // If we didn't find a close enough match, fail
    return 0;
}

Ref<BitMatrix> Version::buildFixedPatternValue(ErrorHandler & err_handler)
{
    int dimension = getDimensionForVersion(err_handler);
    if (err_handler.errCode())   return Ref<BitMatrix>();
    
    Ref<BitMatrix> fixedInfo(new BitMatrix(dimension, err_handler));
    if (err_handler.errCode())   return Ref<BitMatrix>();
    
    // first timming patterns
    for (int i = 0; i < dimension; i += 2) fixedInfo->set(i, 6);
    for (int i = 0; i < dimension; i += 2) fixedInfo->set(6, i);
    
    // fP top left
    fixedInfo->setRegion(0, 0, 8, 8, err_handler);
    fixedInfo->flipRegion(0, 0, 8, 8, err_handler);
    fixedInfo->flipRegion(0, 0, 7, 7, err_handler);
    fixedInfo->flipRegion(1, 1, 5, 5, err_handler);
    fixedInfo->flipRegion(2, 2, 3, 3, err_handler);
    
    // fP top right
    fixedInfo->setRegion(dimension - 8, 0, 8, 8, err_handler);
    fixedInfo->flipRegion(dimension - 8, 0, 8, 8, err_handler);
    fixedInfo->flipRegion(dimension - 7, 0, 7, 7, err_handler);
    fixedInfo->flipRegion(dimension - 6, 1, 5, 5, err_handler);
    fixedInfo->flipRegion(dimension - 5, 2, 3, 3, err_handler);
    
    // fP bottom left
    fixedInfo->setRegion(0, dimension - 8, 8, 8, err_handler);
    fixedInfo->flipRegion(0, dimension - 8, 8, 8, err_handler);
    fixedInfo->flipRegion(0, dimension - 7, 7, 7, err_handler);
    fixedInfo->flipRegion(1, dimension - 6, 5, 5, err_handler);
    fixedInfo->flipRegion(2, dimension - 5, 3, 3, err_handler);
    if (err_handler.errCode())   return Ref<BitMatrix>();
    
    // alignment patterns
    size_t max = alignmentPatternCenters_.size();
    for (size_t x = 0; x < max; x++) {
        int i = alignmentPatternCenters_[x] - 2;
        for (size_t y = 0; y < max; y++) {
            if ((x == 0 && (y == 0 || y == max - 1)) || (x == max - 1 && y == 0)) {
                // No alignment patterns near the three finder patterns
                continue;
            }
            fixedInfo->setRegion(alignmentPatternCenters_[y] - 2, i, 5, 5, err_handler);
            fixedInfo->flipRegion(alignmentPatternCenters_[y] - 1, i + 1, 3, 3, err_handler);
            fixedInfo->flipRegion(alignmentPatternCenters_[y], i + 2, 1, 1, err_handler);
            if (err_handler.errCode())   return Ref<BitMatrix>();
        }
    }
    return fixedInfo;
}

Ref<BitMatrix> Version::buildFixedPatternTemplate(ErrorHandler & err_handler) {
    int dimension = getDimensionForVersion(err_handler);
    Ref<BitMatrix> functionPattern(new BitMatrix(dimension, err_handler));
    if (err_handler.errCode())   return Ref<BitMatrix>();
    
    
    // Top left finder pattern + separator + format
    functionPattern->setRegion(0, 0, 8, 8, err_handler);
    // Top right finder pattern + separator + format
    functionPattern->setRegion(dimension - 8, 0, 8, 8, err_handler);
    // Bottom left finder pattern + separator + format
    functionPattern->setRegion(0, dimension - 8, 8, 8, err_handler);
    if (err_handler.errCode())   return Ref<BitMatrix>();
    
    // alignment patterns
    size_t max = alignmentPatternCenters_.size();
    for (size_t x = 0; x < max; x++) {
        int i = alignmentPatternCenters_[x] - 2;
        for (size_t y = 0; y < max; y++) {
            if ((x == 0 && (y == 0 || y == max - 1)) || (x == max - 1 && y == 0)) {
                // No alignment patterns near the three finder patterns
                continue;
            }
            functionPattern->setRegion(alignmentPatternCenters_[y] - 2, i, 5, 5, err_handler);
        }
    }
    
    // Vertical timing pattern
    functionPattern->setRegion(6, 8, 1, dimension - 16, err_handler);
    // Horizontal timing pattern
    functionPattern->setRegion(8, 6, dimension - 16, 1, err_handler);
    if (err_handler.errCode())   return Ref<BitMatrix>();
    
    return functionPattern;
}

Ref<BitMatrix> Version::buildFunctionPattern(ErrorHandler & err_handler) {
    int dimension = getDimensionForVersion(err_handler);
    Ref<BitMatrix> functionPattern(new BitMatrix(dimension, err_handler));
    if (err_handler.errCode()) return Ref<BitMatrix>();
    
    // Top left finder pattern + separator + format
    functionPattern->setRegion(0, 0, 9, 9, err_handler);
    // Top right finder pattern + separator + format
    functionPattern->setRegion(dimension - 8, 0, 8, 9, err_handler);
    // Bottom left finder pattern + separator + format
    functionPattern->setRegion(0, dimension - 8, 9, 8, err_handler);
    
    
    // Alignment patterns
    size_t max = alignmentPatternCenters_.size();
    for (size_t x = 0; x < max; x++) {
        int i = alignmentPatternCenters_[x] - 2;
        for (size_t y = 0; y < max; y++) {
            if ((x == 0 && (y == 0 || y == max - 1)) || (x == max - 1 && y == 0)) {
                // No alignment patterns near the three finder patterns
                continue;
            }
            functionPattern->setRegion(alignmentPatternCenters_[y] - 2, i, 5, 5, err_handler);
        }
    }
    
    // Vertical timing pattern
    functionPattern->setRegion(6, 9, 1, dimension - 17, err_handler);
    // Horizontal timing pattern
    functionPattern->setRegion(9, 6, dimension - 17, 1, err_handler);
    if (err_handler.errCode()) return Ref<BitMatrix>();
    
    if (versionNumber_ > 6) {
        // Version info, top right
        functionPattern->setRegion(dimension - 11, 0, 3, 6, err_handler);
        // Version info, bottom left
        functionPattern->setRegion(0, dimension - 11, 6, 3, err_handler);
        if (err_handler.errCode())   return Ref<BitMatrix>();
    }
    
    return functionPattern;
}

Ref<Version> Version::MICRO_VERSION = Ref<Version>(new Version(0, intArray(0),
                                                               new ECBlocks(5, new ECB(1, 11)),
                                                               new ECBlocks(6, new ECB(1, 10)),
                                                               new ECBlocks(7, new ECB(1, 9)),
                                                               new ECBlocks(8, new ECB(1, 8))));

int Version::buildVersions() {
    VERSIONS.push_back(Ref<Version>(new Version(1, intArray(0),
                                                new ECBlocks(7, new ECB(1, 19)),
                                                new ECBlocks(10, new ECB(1, 16)),
                                                new ECBlocks(13, new ECB(1, 13)),
                                                new ECBlocks(17, new ECB(1, 9)))));
    VERSIONS.push_back(Ref<Version>(new Version(2, intArray(2, 6, 18),
                                                new ECBlocks(10, new ECB(1, 34)),
                                                new ECBlocks(16, new ECB(1, 28)),
                                                new ECBlocks(22, new ECB(1, 22)),
                                                new ECBlocks(28, new ECB(1, 16)))));
    VERSIONS.push_back(Ref<Version>(new Version(3, intArray(2, 6, 22),
                                                new ECBlocks(15, new ECB(1, 55)),
                                                new ECBlocks(26, new ECB(1, 44)),
                                                new ECBlocks(18, new ECB(2, 17)),
                                                new ECBlocks(22, new ECB(2, 13)))));
    VERSIONS.push_back(Ref<Version>(new Version(4, intArray(2, 6, 26),
                                                new ECBlocks(20, new ECB(1, 80)),
                                                new ECBlocks(18, new ECB(2, 32)),
                                                new ECBlocks(26, new ECB(2, 24)),
                                                new ECBlocks(16, new ECB(4, 9)))));
    VERSIONS.push_back(Ref<Version>(new Version(5, intArray(2, 6, 30),
                                                new ECBlocks(26, new ECB(1, 108)),
                                                new ECBlocks(24, new ECB(2, 43)),
                                                new ECBlocks(18, new ECB(2, 15),
                                                             new ECB(2, 16)),
                                                new ECBlocks(22, new ECB(2, 11),
                                                             new ECB(2, 12)))));
    VERSIONS.push_back(Ref<Version>(new Version(6, intArray(2, 6, 34),
                                                new ECBlocks(18, new ECB(2, 68)),
                                                new ECBlocks(16, new ECB(4, 27)),
                                                new ECBlocks(24, new ECB(4, 19)),
                                                new ECBlocks(28, new ECB(4, 15)))));
    VERSIONS.push_back(Ref<Version>(new Version(7, intArray(3, 6, 22, 38),
                                                new ECBlocks(20, new ECB(2, 78)),
                                                new ECBlocks(18, new ECB(4, 31)),
                                                new ECBlocks(18, new ECB(2, 14),
                                                             new ECB(4, 15)),
                                                new ECBlocks(26, new ECB(4, 13),
                                                             new ECB(1, 14)))));
    VERSIONS.push_back(Ref<Version>(new Version(8, intArray(3, 6, 24, 42),
                                                new ECBlocks(24, new ECB(2, 97)),
                                                new ECBlocks(22, new ECB(2, 38),
                                                             new ECB(2, 39)),
                                                new ECBlocks(22, new ECB(4, 18),
                                                             new ECB(2, 19)),
                                                new ECBlocks(26, new ECB(4, 14),
                                                             new ECB(2, 15)))));
    VERSIONS.push_back(Ref<Version>(new Version(9, intArray(3, 6, 26, 46),
                                                new ECBlocks(30, new ECB(2, 116)),
                                                new ECBlocks(22, new ECB(3, 36),
                                                             new ECB(2, 37)),
                                                new ECBlocks(20, new ECB(4, 16),
                                                             new ECB(4, 17)),
                                                new ECBlocks(24, new ECB(4, 12),
                                                             new ECB(4, 13)))));
    VERSIONS.push_back(Ref<Version>(new Version(10, intArray(3, 6, 28, 50),
                                                new ECBlocks(18, new ECB(2, 68),
                                                             new ECB(2, 69)),
                                                new ECBlocks(26, new ECB(4, 43),
                                                             new ECB(1, 44)),
                                                new ECBlocks(24, new ECB(6, 19),
                                                             new ECB(2, 20)),
                                                new ECBlocks(28, new ECB(6, 15),
                                                             new ECB(2, 16)))));
    VERSIONS.push_back(Ref<Version>(new Version(11, intArray(3, 6, 30, 54),
                                                new ECBlocks(20, new ECB(4, 81)),
                                                new ECBlocks(30, new ECB(1, 50),
                                                             new ECB(4, 51)),
                                                new ECBlocks(28, new ECB(4, 22),
                                                             new ECB(4, 23)),
                                                new ECBlocks(24, new ECB(3, 12),
                                                             new ECB(8, 13)))));
    VERSIONS.push_back(Ref<Version>(new Version(12, intArray(3, 6, 32, 58),
                                                new ECBlocks(24, new ECB(2, 92),
                                                             new ECB(2, 93)),
                                                new ECBlocks(22, new ECB(6, 36),
                                                             new ECB(2, 37)),
                                                new ECBlocks(26, new ECB(4, 20),
                                                             new ECB(6, 21)),
                                                new ECBlocks(28, new ECB(7, 14),
                                                             new ECB(4, 15)))));
    VERSIONS.push_back(Ref<Version>(new Version(13, intArray(3, 6, 34, 62),
                                                new ECBlocks(26, new ECB(4, 107)),
                                                new ECBlocks(22, new ECB(8, 37),
                                                             new ECB(1, 38)),
                                                new ECBlocks(24, new ECB(8, 20),
                                                             new ECB(4, 21)),
                                                new ECBlocks(22, new ECB(12, 11),
                                                             new ECB(4, 12)))));
    VERSIONS.push_back(Ref<Version>(new Version(14, intArray(4, 6, 26, 46, 66),
                                                new ECBlocks(30, new ECB(3, 115),
                                                             new ECB(1, 116)),
                                                new ECBlocks(24, new ECB(4, 40),
                                                             new ECB(5, 41)),
                                                new ECBlocks(20, new ECB(11, 16),
                                                             new ECB(5, 17)),
                                                new ECBlocks(24, new ECB(11, 12),
                                                             new ECB(5, 13)))));
    VERSIONS.push_back(Ref<Version>(new Version(15, intArray(4, 6, 26, 48, 70),
                                                new ECBlocks(22, new ECB(5, 87),
                                                             new ECB(1, 88)),
                                                new ECBlocks(24, new ECB(5, 41),
                                                             new ECB(5, 42)),
                                                new ECBlocks(30, new ECB(5, 24),
                                                             new ECB(7, 25)),
                                                new ECBlocks(24, new ECB(11, 12),
                                                             new ECB(7, 13)))));
    VERSIONS.push_back(Ref<Version>(new Version(16, intArray(4, 6, 26, 50, 74),
                                                new ECBlocks(24, new ECB(5, 98),
                                                             new ECB(1, 99)),
                                                new ECBlocks(28, new ECB(7, 45),
                                                             new ECB(3, 46)),
                                                new ECBlocks(24, new ECB(15, 19),
                                                             new ECB(2, 20)),
                                                new ECBlocks(30, new ECB(3, 15),
                                                             new ECB(13, 16)))));
    VERSIONS.push_back(Ref<Version>(new Version(17, intArray(4, 6, 30, 54, 78),
                                                new ECBlocks(28, new ECB(1, 107),
                                                             new ECB(5, 108)),
                                                new ECBlocks(28, new ECB(10, 46),
                                                             new ECB(1, 47)),
                                                new ECBlocks(28, new ECB(1, 22),
                                                             new ECB(15, 23)),
                                                new ECBlocks(28, new ECB(2, 14),
                                                             new ECB(17, 15)))));
    VERSIONS.push_back(Ref<Version>(new Version(18, intArray(4, 6, 30, 56, 82),
                                                new ECBlocks(30, new ECB(5, 120),
                                                             new ECB(1, 121)),
                                                new ECBlocks(26, new ECB(9, 43),
                                                             new ECB(4, 44)),
                                                new ECBlocks(28, new ECB(17, 22),
                                                             new ECB(1, 23)),
                                                new ECBlocks(28, new ECB(2, 14),
                                                             new ECB(19, 15)))));
    VERSIONS.push_back(Ref<Version>(new Version(19, intArray(4, 6, 30, 58, 86),
                                                new ECBlocks(28, new ECB(3, 113),
                                                             new ECB(4, 114)),
                                                new ECBlocks(26, new ECB(3, 44),
                                                             new ECB(11, 45)),
                                                new ECBlocks(26, new ECB(17, 21),
                                                             new ECB(4, 22)),
                                                new ECBlocks(26, new ECB(9, 13),
                                                             new ECB(16, 14)))));
    VERSIONS.push_back(Ref<Version>(new Version(20, intArray(4, 6, 34, 62, 90),
                                                new ECBlocks(28, new ECB(3, 107),
                                                             new ECB(5, 108)),
                                                new ECBlocks(26, new ECB(3, 41),
                                                             new ECB(13, 42)),
                                                new ECBlocks(30, new ECB(15, 24),
                                                             new ECB(5, 25)),
                                                new ECBlocks(28, new ECB(15, 15),
                                                             new ECB(10, 16)))));
    VERSIONS.push_back(Ref<Version>(new Version(21, intArray(5, 6, 28, 50, 72, 94),
                                                new ECBlocks(28, new ECB(4, 116),
                                                             new ECB(4, 117)),
                                                new ECBlocks(26, new ECB(17, 42)),
                                                new ECBlocks(28, new ECB(17, 22),
                                                             new ECB(6, 23)),
                                                new ECBlocks(30, new ECB(19, 16),
                                                             new ECB(6, 17)))));
    VERSIONS.push_back(Ref<Version>(new Version(22, intArray(5, 6, 26, 50, 74, 98),
                                                new ECBlocks(28, new ECB(2, 111),
                                                             new ECB(7, 112)),
                                                new ECBlocks(28, new ECB(17, 46)),
                                                new ECBlocks(30, new ECB(7, 24),
                                                             new ECB(16, 25)),
                                                new ECBlocks(24, new ECB(34, 13)))));
    VERSIONS.push_back(Ref<Version>(new Version(23, intArray(5, 6, 30, 54, 78, 102),
                                                new ECBlocks(30, new ECB(4, 121),
                                                             new ECB(5, 122)),
                                                new ECBlocks(28, new ECB(4, 47),
                                                             new ECB(14, 48)),
                                                new ECBlocks(30, new ECB(11, 24),
                                                             new ECB(14, 25)),
                                                new ECBlocks(30, new ECB(16, 15),
                                                             new ECB(14, 16)))));
    VERSIONS.push_back(Ref<Version>(new Version(24, intArray(5, 6, 28, 54, 80, 106),
                                                new ECBlocks(30, new ECB(6, 117),
                                                             new ECB(4, 118)),
                                                new ECBlocks(28, new ECB(6, 45),
                                                             new ECB(14, 46)),
                                                new ECBlocks(30, new ECB(11, 24),
                                                             new ECB(16, 25)),
                                                new ECBlocks(30, new ECB(30, 16),
                                                             new ECB(2, 17)))));
    VERSIONS.push_back(Ref<Version>(new Version(25, intArray(5, 6, 32, 58, 84, 110),
                                                new ECBlocks(26, new ECB(8, 106),
                                                             new ECB(4, 107)),
                                                new ECBlocks(28, new ECB(8, 47),
                                                             new ECB(13, 48)),
                                                new ECBlocks(30, new ECB(7, 24),
                                                             new ECB(22, 25)),
                                                new ECBlocks(30, new ECB(22, 15),
                                                             new ECB(13, 16)))));
    VERSIONS.push_back(Ref<Version>(new Version(26, intArray(5, 6, 30, 58, 86, 114),
                                                new ECBlocks(28, new ECB(10, 114),
                                                             new ECB(2, 115)),
                                                new ECBlocks(28, new ECB(19, 46),
                                                             new ECB(4, 47)),
                                                new ECBlocks(28, new ECB(28, 22),
                                                             new ECB(6, 23)),
                                                new ECBlocks(30, new ECB(33, 16),
                                                             new ECB(4, 17)))));
    VERSIONS.push_back(Ref<Version>(new Version(27, intArray(5, 6, 34, 62, 90, 118),
                                                new ECBlocks(30, new ECB(8, 122),
                                                             new ECB(4, 123)),
                                                new ECBlocks(28, new ECB(22, 45),
                                                             new ECB(3, 46)),
                                                new ECBlocks(30, new ECB(8, 23),
                                                             new ECB(26, 24)),
                                                new ECBlocks(30, new ECB(12, 15),
                                                             new ECB(28, 16)))));
    VERSIONS.push_back(Ref<Version>(new Version(28, intArray(6, 6, 26, 50, 74, 98, 122),
                                                new ECBlocks(30, new ECB(3, 117),
                                                             new ECB(10, 118)),
                                                new ECBlocks(28, new ECB(3, 45),
                                                             new ECB(23, 46)),
                                                new ECBlocks(30, new ECB(4, 24),
                                                             new ECB(31, 25)),
                                                new ECBlocks(30, new ECB(11, 15),
                                                             new ECB(31, 16)))));
    VERSIONS.push_back(Ref<Version>(new Version(29, intArray(6, 6, 30, 54, 78, 102, 126),
                                                new ECBlocks(30, new ECB(7, 116),
                                                             new ECB(7, 117)),
                                                new ECBlocks(28, new ECB(21, 45),
                                                             new ECB(7, 46)),
                                                new ECBlocks(30, new ECB(1, 23),
                                                             new ECB(37, 24)),
                                                new ECBlocks(30, new ECB(19, 15),
                                                             new ECB(26, 16)))));
    VERSIONS.push_back(Ref<Version>(new Version(30, intArray(6, 6, 26, 52, 78, 104, 130),
                                                new ECBlocks(30, new ECB(5, 115),
                                                             new ECB(10, 116)),
                                                new ECBlocks(28, new ECB(19, 47),
                                                             new ECB(10, 48)),
                                                new ECBlocks(30, new ECB(15, 24),
                                                             new ECB(25, 25)),
                                                new ECBlocks(30, new ECB(23, 15),
                                                             new ECB(25, 16)))));
    VERSIONS.push_back(Ref<Version>(new Version(31, intArray(6, 6, 30, 56, 82, 108, 134),
                                                new ECBlocks(30, new ECB(13, 115),
                                                             new ECB(3, 116)),
                                                new ECBlocks(28, new ECB(2, 46),
                                                             new ECB(29, 47)),
                                                new ECBlocks(30, new ECB(42, 24),
                                                             new ECB(1, 25)),
                                                new ECBlocks(30, new ECB(23, 15),
                                                             new ECB(28, 16)))));
    VERSIONS.push_back(Ref<Version>(new Version(32, intArray(6, 6, 34, 60, 86, 112, 138),
                                                new ECBlocks(30, new ECB(17, 115)),
                                                new ECBlocks(28, new ECB(10, 46),
                                                             new ECB(23, 47)),
                                                new ECBlocks(30, new ECB(10, 24),
                                                             new ECB(35, 25)),
                                                new ECBlocks(30, new ECB(19, 15),
                                                             new ECB(35, 16)))));
    VERSIONS.push_back(Ref<Version>(new Version(33, intArray(6, 6, 30, 58, 86, 114, 142),
                                                new ECBlocks(30, new ECB(17, 115),
                                                             new ECB(1, 116)),
                                                new ECBlocks(28, new ECB(14, 46),
                                                             new ECB(21, 47)),
                                                new ECBlocks(30, new ECB(29, 24),
                                                             new ECB(19, 25)),
                                                new ECBlocks(30, new ECB(11, 15),
                                                             new ECB(46, 16)))));
    VERSIONS.push_back(Ref<Version>(new Version(34, intArray(6, 6, 34, 62, 90, 118, 146),
                                                new ECBlocks(30, new ECB(13, 115),
                                                             new ECB(6, 116)),
                                                new ECBlocks(28, new ECB(14, 46),
                                                             new ECB(23, 47)),
                                                new ECBlocks(30, new ECB(44, 24),
                                                             new ECB(7, 25)),
                                                new ECBlocks(30, new ECB(59, 16),
                                                             new ECB(1, 17)))));
    VERSIONS.push_back(Ref<Version>(new Version(35, intArray(7, 6, 30, 54, 78,
                                                             102, 126, 150),
                                                new ECBlocks(30, new ECB(12, 121),
                                                             new ECB(7, 122)),
                                                new ECBlocks(28, new ECB(12, 47),
                                                             new ECB(26, 48)),
                                                new ECBlocks(30, new ECB(39, 24),
                                                             new ECB(14, 25)),
                                                new ECBlocks(30, new ECB(22, 15),
                                                             new ECB(41, 16)))));
    VERSIONS.push_back(Ref<Version>(new Version(36, intArray(7, 6, 24, 50, 76,
                                                             102, 128, 154),
                                                new ECBlocks(30, new ECB(6, 121),
                                                             new ECB(14, 122)),
                                                new ECBlocks(28, new ECB(6, 47),
                                                             new ECB(34, 48)),
                                                new ECBlocks(30, new ECB(46, 24),
                                                             new ECB(10, 25)),
                                                new ECBlocks(30, new ECB(2, 15),
                                                             new ECB(64, 16)))));
    VERSIONS.push_back(Ref<Version>(new Version(37, intArray(7, 6, 28, 54, 80,
                                                             106, 132, 158),
                                                new ECBlocks(30, new ECB(17, 122),
                                                             new ECB(4, 123)),
                                                new ECBlocks(28, new ECB(29, 46),
                                                             new ECB(14, 47)),
                                                new ECBlocks(30, new ECB(49, 24),
                                                             new ECB(10, 25)),
                                                new ECBlocks(30, new ECB(24, 15),
                                                             new ECB(46, 16)))));
    VERSIONS.push_back(Ref<Version>(new Version(38, intArray(7, 6, 32, 58, 84,
                                                             110, 136, 162),
                                                new ECBlocks(30, new ECB(4, 122),
                                                             new ECB(18, 123)),
                                                new ECBlocks(28, new ECB(13, 46),
                                                             new ECB(32, 47)),
                                                new ECBlocks(30, new ECB(48, 24),
                                                             new ECB(14, 25)),
                                                new ECBlocks(30, new ECB(42, 15),
                                                             new ECB(32, 16)))));
    VERSIONS.push_back(Ref<Version>(new Version(39, intArray(7, 6, 26, 54, 82,
                                                             110, 138, 166),
                                                new ECBlocks(30, new ECB(20, 117),
                                                             new ECB(4, 118)),
                                                new ECBlocks(28, new ECB(40, 47),
                                                             new ECB(7, 48)),
                                                new ECBlocks(30, new ECB(43, 24),
                                                             new ECB(22, 25)),
                                                new ECBlocks(30, new ECB(10, 15),
                                                             new ECB(67, 16)))));
    VERSIONS.push_back(Ref<Version>(new Version(40, intArray(7, 6, 30, 58, 86,
                                                             114, 142, 170),
                                                new ECBlocks(30, new ECB(19, 118),
                                                             new ECB(6, 119)),
                                                new ECBlocks(28, new ECB(18, 47),
                                                             new ECB(31, 48)),
                                                new ECBlocks(30, new ECB(34, 24),
                                                             new ECB(34, 25)),
                                                new ECBlocks(30, new ECB(20, 15),
                                                             new ECB(61, 16)))));
    return VERSIONS.size();
}

}  // namespace qrcode
}  // namespace zxing
