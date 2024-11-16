/*
 *  Version.cpp
 *  zxing
 *
 *  Created by Luiz Silva on 09/02/2010.
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

#include "version.hpp"
#include <limits>
#include <iostream>

namespace zxing {
namespace datamatrix {

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

std::vector<Ref<Version> > Version::VERSIONS;
static int N_VERSIONS = Version::buildVersions();

Version::Version(int versionNumber, int symbolSizeRows, int symbolSizeColumns, int dataRegionSizeRows,
                 int dataRegionSizeColumns, ECBlocks* ecBlocks) : versionNumber_(versionNumber),
symbolSizeRows_(symbolSizeRows), symbolSizeColumns_(symbolSizeColumns),
dataRegionSizeRows_(dataRegionSizeRows), dataRegionSizeColumns_(dataRegionSizeColumns),
ecBlocks_(ecBlocks), totalCodewords_(0) {
    // Calculate the total number of codewords
    int total = 0;
    int ecCodewords = ecBlocks_->getECCodewords();
    std::vector<ECB*> &ecbArray = ecBlocks_->getECBlocks();
    for (unsigned int i = 0; i < ecbArray.size(); i++) {
        ECB *ecBlock = ecbArray[i];
        total += ecBlock->getCount() * (ecBlock->getDataCodewords() + ecCodewords);
    }
    totalCodewords_ = total;
}

Version::~Version() {
    delete ecBlocks_;
}

int Version::getVersionNumber() {
    return versionNumber_;
}

int Version::getSymbolSizeRows() {
    return symbolSizeRows_;
}

int Version::getSymbolSizeColumns() {
    return symbolSizeColumns_;
}

int Version::getDataRegionSizeRows() {
    return dataRegionSizeRows_;
}

int Version::getDataRegionSizeColumns() {
    return dataRegionSizeColumns_;
}

int Version::getTotalCodewords() {
    return totalCodewords_;
}

ECBlocks* Version::getECBlocks() {
    return ecBlocks_;
}

Version* Version::getVersionForDimensions(int numRows, int numColumns, ErrorHandler & err_handler) {
    if ((numRows & 0x01) != 0 || (numColumns & 0x01) != 0)
    {
        err_handler = ReaderErrorHandler("Number of rows and columns must be even");
        return NULL;
    }
    
    // TODO(bbrown): This is doing a linear search through the array of versions.
    // If we interleave the rectangular versions with the square versions we could
    // do a binary search.
    for (int i = 0; i < N_VERSIONS; ++i){
        Version * version = VERSIONS[i];
        if (version->getSymbolSizeRows() == numRows && version->getSymbolSizeColumns() == numColumns)
        {
            return version;
        }
    }
    
    err_handler = ReaderErrorHandler("Error version not found");
    return NULL;
}

/**
 * See ISO 16022:2006 5.5.1 Table 7
 */
int Version::buildVersions() {
    VERSIONS.push_back(Ref<Version>(new Version(1, 10, 10, 8, 8,
                                                new ECBlocks(5, new ECB(1, 3)))));
    VERSIONS.push_back(Ref<Version>(new Version(2, 12, 12, 10, 10,
                                                new ECBlocks(7, new ECB(1, 5)))));
    VERSIONS.push_back(Ref<Version>(new Version(3, 14, 14, 12, 12,
                                                new ECBlocks(10, new ECB(1, 8)))));
    VERSIONS.push_back(Ref<Version>(new Version(4, 16, 16, 14, 14,
                                                new ECBlocks(12, new ECB(1, 12)))));
    VERSIONS.push_back(Ref<Version>(new Version(5, 18, 18, 16, 16,
                                                new ECBlocks(14, new ECB(1, 18)))));
    VERSIONS.push_back(Ref<Version>(new Version(6, 20, 20, 18, 18,
                                                new ECBlocks(18, new ECB(1, 22)))));
    VERSIONS.push_back(Ref<Version>(new Version(7, 22, 22, 20, 20,
                                                new ECBlocks(20, new ECB(1, 30)))));
    VERSIONS.push_back(Ref<Version>(new Version(8, 24, 24, 22, 22,
                                                new ECBlocks(24, new ECB(1, 36)))));
    VERSIONS.push_back(Ref<Version>(new Version(9, 26, 26, 24, 24,
                                                new ECBlocks(28, new ECB(1, 44)))));
    VERSIONS.push_back(Ref<Version>(new Version(10, 32, 32, 14, 14,
                                                new ECBlocks(36, new ECB(1, 62)))));
    VERSIONS.push_back(Ref<Version>(new Version(11, 36, 36, 16, 16,
                                                new ECBlocks(42, new ECB(1, 86)))));
    VERSIONS.push_back(Ref<Version>(new Version(12, 40, 40, 18, 18,
                                                new ECBlocks(48, new ECB(1, 114)))));
    VERSIONS.push_back(Ref<Version>(new Version(13, 44, 44, 20, 20,
                                                new ECBlocks(56, new ECB(1, 144)))));
    VERSIONS.push_back(Ref<Version>(new Version(14, 48, 48, 22, 22,
                                                new ECBlocks(68, new ECB(1, 174)))));
    VERSIONS.push_back(Ref<Version>(new Version(15, 52, 52, 24, 24,
                                                new ECBlocks(42, new ECB(2, 102)))));
    VERSIONS.push_back(Ref<Version>(new Version(16, 64, 64, 14, 14,
                                                new ECBlocks(56, new ECB(2, 140)))));
    VERSIONS.push_back(Ref<Version>(new Version(17, 72, 72, 16, 16,
                                                new ECBlocks(36, new ECB(4, 92)))));
    VERSIONS.push_back(Ref<Version>(new  Version(18, 80, 80, 18, 18,
                                                 new ECBlocks(48, new ECB(4, 114)))));
    VERSIONS.push_back(Ref<Version>(new Version(19, 88, 88, 20, 20,
                                                new ECBlocks(56, new ECB(4, 144)))));
    VERSIONS.push_back(Ref<Version>(new Version(20, 96, 96, 22, 22,
                                                new ECBlocks(68, new ECB(4, 174)))));
    VERSIONS.push_back(Ref<Version>(new Version(21, 104, 104, 24, 24,
                                                new ECBlocks(56, new ECB(6, 136)))));
    VERSIONS.push_back(Ref<Version>(new Version(22, 120, 120, 18, 18,
                                                new ECBlocks(68, new ECB(6, 175)))));
    VERSIONS.push_back(Ref<Version>(new Version(23, 132, 132, 20, 20,
                                                new ECBlocks(62, new ECB(8, 163)))));
    VERSIONS.push_back(Ref<Version>(new Version(24, 144, 144, 22, 22,
                                                new ECBlocks(62, new ECB(8, 156), new ECB(2, 155)))));
    VERSIONS.push_back(Ref<Version>(new Version(25, 8, 18, 6, 16,
                                                new ECBlocks(7, new ECB(1, 5)))));
    VERSIONS.push_back(Ref<Version>(new Version(26, 8, 32, 6, 14,
                                                new ECBlocks(11, new ECB(1, 10)))));
    VERSIONS.push_back(Ref<Version>(new Version(27, 12, 26, 10, 24,
                                                new ECBlocks(14, new ECB(1, 16)))));
    VERSIONS.push_back(Ref<Version>(new Version(28, 12, 36, 10, 16,
                                                new ECBlocks(18, new ECB(1, 22)))));
    VERSIONS.push_back(Ref<Version>(new Version(29, 16, 36, 14, 16,
                                                new ECBlocks(24, new ECB(1, 32)))));
    VERSIONS.push_back(Ref<Version>(new Version(30, 16, 48, 14, 22,
                                                new ECBlocks(28, new ECB(1, 49)))));
    return VERSIONS.size();
}
}  // namespace datamatrix
}  // namespace zxing

