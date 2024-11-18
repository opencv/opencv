// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").

#ifndef __ZXING_QRCODE_VERSION_HPP__
#define __ZXING_QRCODE_VERSION_HPP__

/*
 *  Version.hpp
 *  zxing
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

#include "../common/counted.hpp"
#include "error_correction_level.hpp"
#include "../reader_exception.hpp"
#include "../common/bit_matrix.hpp"
#include "../common/counted.hpp"
#include "../error_handler.hpp"
#include <vector>

namespace zxing {
namespace qrcode {

// Encapsualtes the parameters for one error-correction block in one symbol version.
// This includes the number of data codewords, and the number of times a block with these
// parameters is used consecutively in the QR code version's format.
class ECB {
private:
    int count_;
    int dataCodewords_;

public:
    ECB(int count, int dataCodewords);
    int getCount();
    int getDataCodewords();
};

// Encapsulates a set of error-correction blocks in one symbol version. Most versions will
// use blocks of differing sizes within one version, so, this encapsulates the parameters for
// each set of blocks. It also holds the number of error-correction codewords per block since it
// will be the same across all blocks within one version.</p>
class ECBlocks {
private:
    int ecCodewords_;
    std::vector<ECB*> ecBlocks_;

public:
    ECBlocks(int ecCodewords, ECB *ecBlocks);
    ECBlocks(int ecCodewords, ECB *ecBlocks1, ECB *ecBlocks2);
    int getECCodewords();
    std::vector<ECB*>& getECBlocks();
    ~ECBlocks();
};

class Version : public Counted {
private:
    int versionNumber_;
    std::vector<int> &alignmentPatternCenters_;
    std::vector<ECBlocks*> ecBlocks_;
    int totalCodewords_;
    Version(int versionNumber, std::vector<int> *alignmentPatternCenters, ECBlocks *ecBlocks1, ECBlocks *ecBlocks2,
            ECBlocks *ecBlocks3, ECBlocks *ecBlocks4);
    
public:
    static unsigned int VERSION_DECODE_INFO[];
    static int N_VERSION_DECODE_INFOS;
    static std::vector<Ref<Version> > VERSIONS;
    
    // lincoln add
    static  Ref<Version> MICRO_VERSION;
    
    static const int MICRO_DEMITION = 19;
    
    ~Version();
    int getVersionNumber();
    std::vector<int> &getAlignmentPatternCenters();
    int getTotalCodewords();
    int getDimensionForVersion(ErrorHandler & err_handler);
    ECBlocks &getECBlocksForLevel(ErrorCorrectionLevel &ecLevel);
    bool checkECBlocksForLevel(ErrorCorrectionLevel &ecLevel);
    static Version *getProvisionalVersionForDimension(int dimension, ErrorHandler & err_handler);
    static Version *getVersionForNumber(int versionNumber , ErrorHandler & err_handler);
    static Version *decodeVersionInformation(unsigned int versionBits);
    Ref<BitMatrix> buildFunctionPattern(ErrorHandler & err_handler);
    Ref<BitMatrix> buildFixedPatternValue(ErrorHandler & err_handler);
    Ref<BitMatrix> buildFixedPatternTemplate(ErrorHandler & err_handler);
    static int buildVersions();
};
}  // namespace qrcode
}  // namespace zxing

#endif  // __ZXING_QRCODE_VERSION_HPP__
