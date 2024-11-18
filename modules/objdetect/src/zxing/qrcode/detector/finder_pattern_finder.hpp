// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").

// -*- mode:c++; tab-width:2; indent-tabs-mode:nil; c-basic-offset:2 -*-
#ifndef __ZXING_QRCODE_DETECTOR_FINDER_PATTERN_FINDER_HP__
#define __ZXING_QRCODE_DETECTOR_FINDER_PATTERN_FINDER_HP__

/*
 *  FinderPatternFinder.hpp
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

#include "finder_pattern.hpp"
#include "finder_pattern_info.hpp"
#include "../../common/counted.hpp"
#include "../../common/bit_matrix.hpp"
#include "../../result_point_callback.hpp"
#include "../../common/unicom_block.hpp"
#include "../../error_handler.hpp"
#include <vector>

using zxing::ReaderErrorHandler;
using zxing::ErrorHandler;

namespace zxing {

class DecodeHints;

namespace qrcode {

class FinderPatternFinder {

public:
	enum  CrossCheckState{ 
		NORMAL = 0,
		LEFT_SPILL = 1,
		RIHGT_SPILL = 2,
		LEFT_RIGHT_SPILL = 3,
		NOT_PATTERN = 4,
	};

private:
    static int CENTER_QUORUM;
    static int INTEGER_MATH_SHIFT;
    CrossCheckState CURRENT_CHECK_STATE;
    int compared_finder_counts;

    bool tryHarder_;

    struct HorizontalCheckedResult
    {
        size_t centerI;
        float centerJ;
    };

    std::vector<std::vector<HorizontalCheckedResult> > _horizontalCheckedResult;

    // iNI CONFIG
    float FPS_MS_VAL;
    int FP_IS_SELECT_BEST;
    int FP_IS_SELECT_FILE_BEST;
    int FP_INPUT_MAX_NUM;
    int FP_FILTER_SIZE;
    float FP_COUNT_MIN;
    float FP_MS_MIN;
    int FPS_CLUSTER_MAX;
    int FPS_RESULT_MAX;
    int K_FACTOR;
    float FP_RIGHT_ANGLE;
    float FP_SMALL_ANGLE1;
    float FP_SMALL_ANGLE2;
      
    float QR_MIN_FP_AREA_ERR;
    float QR_MIN_FP_MS_ERR;
    int QR_MIN_FP_ACCEPT;

protected:
    static int MIN_SKIP;
    static int MAX_MODULES;

    Ref<BitMatrix> image_;
    std::vector<Ref<FinderPattern> > possibleCenters_;
    std::vector<Ref<FinderPattern> > possiblePureCenters_;
    bool hasSkipped_;

    static int _minModuleSize;

    Ref<ResultPointCallback> callback_;
    Ref<UnicomBlock> block_;

    /** stateCount must be int[5] */
    float centerFromEnd(int* stateCount, int end);
    bool foundPatternCross(int* stateCount);
      
    // Used only for checkDiagonal
    bool foundPatternCrossLoose(int* stateCount);
    bool finderConditionLoose_;
    
    bool crossCheckDiagonal(int startI, int centerJ, int maxCount, int originalStateCountTotal);
    float crossCheckVertical(size_t startI, size_t centerJ, int maxCount, int originalStateCountTota, float& estimatedVerticalModuleSize);
    float crossCheckHorizontal(size_t startJ, size_t centerI, int maxCount, int originalStateCountTotal, float& estimatedHorizontalModuleSize);

    float hasHorizontalCheckedResult(size_t startJ, size_t centerI);
    int addHorizontalCheckedResult(size_t startJ, size_t centerI, float centerJ);
    int getMinModuleSize();

    bool isEqualResult(Ref<FinderPatternInfo> src, Ref<FinderPatternInfo> dst);
   
    /** stateCount must be int[5] */
    bool tryToPushToCenters(float posX, float posY, float estimatedModuleSize,
                          CrossCheckState horizontalState = FinderPatternFinder::NORMAL,
                          CrossCheckState verticalState = FinderPatternFinder::NORMAL);
    bool handlePossibleCenter(int* stateCount, size_t i, size_t j);
    int getMaxMinModuleSize(float& minModuleSize, float& maxModuleSize);
    int findRowSkip();
    std::vector< Ref<FinderPattern> > selectBestPatterns(ErrorHandler & err_handler);
    std::vector< Ref<FinderPattern> > selectFileBestPatterns(ErrorHandler & err_handler);
    std::vector<Ref<FinderPattern> > orderBestPatterns(std::vector<Ref<FinderPattern> > patterns);

    std::vector<Ref<FinderPatternInfo> > getPatternInfos(ErrorHandler & err_handler);
    std::vector<Ref<FinderPatternInfo> > getPatternInfosFileMode(DecodeHints const& hints, ErrorHandler & err_handler);

    bool isPossibleFindPatterInfo(Ref<FinderPattern> a, Ref<FinderPattern> b, Ref<FinderPattern> c);
    void pushToResult( Ref<FinderPattern> a, Ref<FinderPattern> b, Ref<FinderPattern> c, std::vector<Ref<FinderPatternInfo> > & patternInfos);

    Ref<BitMatrix> getImage();
    std::vector<Ref<FinderPattern> >& getPossibleCenters();

public:
    void initConfig();
    float distance(Ref<ResultPoint> p1, Ref<ResultPoint> p2);
    FinderPatternFinder(Ref<BitMatrix> image,  Ref<UnicomBlock> block, Ref<ResultPointCallback>const&);
    
    std::vector<Ref<FinderPatternInfo> > find(DecodeHints const& hints, ErrorHandler & err_handler);
    void setFinderLoose(bool loose){finderConditionLoose_=loose;}
    bool getFinderLoose(){return finderConditionLoose_;}
    
    bool checkIsoscelesRightTriangle(Ref<FinderPattern> centerA, Ref<FinderPattern> centerB, Ref<FinderPattern> centerC, float & longSide);
};
}  // namespace qrcode
}  // namespace zxing

#endif  // __ZXING_QRCODE_DETECTOR_FINDER_PATTERN_FINDER_HP__
