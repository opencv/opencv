// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").

#ifndef __ZXING_QRCODE_DETECTOR_POSSIBLE_DETECTOR_RESULT_HPP__
#define __ZXING_QRCODE_DETECTOR_POSSIBLE_DETECTOR_RESULT_HPP__

/*
 *  DetectorResult.hpp
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

#include "../../common/counted.hpp"
#include "../../common/array.hpp"
#include "../../common/bit_matrix.hpp"
#include "../../result_point.hpp"
#include "alignment_pattern.hpp"
#include "finder_pattern_info.hpp"
#include "finder_pattern.hpp"
#include "../../common/detector_result.hpp"

#include <vector>
#include "opencv2/core.hpp"

namespace zxing {
namespace qrcode {
class PatternResult : public Counted
{
public:
    Ref<FinderPatternInfo> finderPatternInfo;
    std::vector<Ref<AlignmentPattern> > possibleAlignmentPatterns;
    
    std::vector<std::vector<cv::Point2f>> moreAlignmentPoints;
    
    std::vector<cv::Point2f> topLeftPoints;
    std::vector<cv::Point2f> topRightPoints;
    std::vector<cv::Point2f> bottomLeftPoints;
    
    Ref<AlignmentPattern> confirmedAlignmentPattern;
    int possibleDimension;
    std::vector<int> possibleDimensions;
    unsigned int possibleVersion;
    
    float possibleFix;
    float possibleModuleSize;
    
    PatternResult(Ref<FinderPatternInfo> info);
    void setConfirmedAlignmentPattern(int index);
    int getPossibleAlignmentCount(){return possibleAlignmentPatterns.size();}
    
public:
    unsigned int getPossibleVersion(){return possibleVersion;}
    float getPossibleFix(){return possibleFix;}
    float getPossibleModuleSize(){return possibleModuleSize;}
    int getDimension(){return possibleDimension;}
};
}  // namespace qrcode
}  // namespace zxing

#endif  // __ZXING_QRCODE_DETECTOR_POSSIBLE_DETECTOR_RESULT_HPP__
