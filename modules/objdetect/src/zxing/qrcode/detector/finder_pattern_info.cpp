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
 *  FinderPatternInfo.cpp
 *  zxing
 *
 *  Created by Christian Brunschen on 13/05/2008.
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

#include "finder_pattern_info.hpp"
#include <stdio.h>

namespace zxing {
namespace qrcode {
FinderPatternInfo::FinderPatternInfo(std::vector<Ref<FinderPattern> > patternCenters):
bottomLeft_(patternCenters[0]), topLeft_(patternCenters[1]), topRight_(patternCenters[2]), possibleFix_(0) {
    estimateFinderPatternInfo();
}

Ref<FinderPattern> FinderPatternInfo::getBottomLeft() {
    return bottomLeft_;
}

Ref<FinderPattern> FinderPatternInfo::getTopLeft() {
    return topLeft_;
}

Ref<FinderPattern> FinderPatternInfo::getTopRight() {
    return topRight_;
}

void FinderPatternInfo::showDetail()
{
    std::vector<Ref<FinderPattern> > findPatterns;
    findPatterns.push_back(topLeft_); findPatterns.push_back(topRight_); findPatterns.push_back(bottomLeft_);
    
    for (size_t i = 0; i < findPatterns.size(); i++)
    {
        printf("count %d ms %.4f hstate %d vstate %d anglefix %.4f\n",
               findPatterns[i]->getCount(), findPatterns[i]->getEstimatedModuleSize(),
               findPatterns[i]->getHorizontalCheckState(), findPatterns[i]->getVerticalCheckState(), anglePossibleFix_);
    }
}

float FinderPatternInfo::getPossibleFix()
{
    return possibleFix_;
}

float FinderPatternInfo::getAnglePossibleFix()
{
    return anglePossibleFix_;
}

// bottomLeft_ => centerA
void FinderPatternInfo::calculateSides(Ref<FinderPattern> centerA, Ref<FinderPattern> centerB,
                                       Ref<FinderPattern> centerC, float &longSide,
                                       float &shortSide1, float &shortSide2) {
    float a_m_b_x = centerA->getX() - centerB->getX();
    float a_m_b_y = centerA->getY() - centerB->getY();
    float ab_s = a_m_b_x * a_m_b_x + a_m_b_y * a_m_b_y;
    float a_m_c_x = centerA->getX() - centerC->getX();
    float a_m_c_y = centerA->getY() - centerC->getY();
    float ac_s = a_m_c_x * a_m_c_x + a_m_c_y * a_m_c_y;
    float b_m_c_x = centerB->getX() - centerC->getX();
    float b_m_c_y = centerB->getY() - centerC->getY();
    float bc_s = b_m_c_x * b_m_c_x + b_m_c_y * b_m_c_y;
    
    if (ab_s > bc_s && ab_s > ac_s) {
        longSide = ab_s;
        shortSide1 = ac_s;
        shortSide2 = bc_s;
    }
    else if (bc_s > ab_s && bc_s > ac_s)
    {
        longSide = bc_s;
        shortSide1 = ab_s;
        shortSide2 = ac_s;
    }
    else
    {
        longSide = ac_s;
        shortSide1 = ab_s;
        shortSide2 = bc_s;
    }
}

void FinderPatternInfo::estimateFinderPatternInfo()
{
    float longSide, shortSide1, shortSide2;
    calculateSides(bottomLeft_, topLeft_, topRight_, longSide, shortSide1, shortSide2);
    
    float CosLong = (shortSide1 + shortSide2 - longSide) / (2 * sqrt(shortSide1)*sqrt(shortSide2));
    float CosShort1 = (longSide + shortSide1 - shortSide2) / (2 * sqrt(longSide)*sqrt(shortSide1));
    float CosShort2 = (longSide + shortSide2 - shortSide1) / (2 * sqrt(longSide)*sqrt(shortSide2));
    
    float fAngleLong = acos( CosLong) * 180 / acos(-1.0);
    float fAngleShort1 = acos( CosShort1) * 180 / acos(-1.0);
    float fAngleShort2 = acos( CosShort2) * 180 / acos(-1.0);
    if (fAngleShort1 < fAngleShort2) std::swap(fAngleShort1, fAngleShort2);
    
    float fLongDiff = fabs(fAngleLong - 90);
    float fLongScore = 100.0 - fLongDiff;
    
    // The maximum error of the small angle.
    float fShortDiff = (std::max)(fabs(fAngleShort1 - 45), fabs(fAngleShort2 - 45));
    float fShortScore = 100.0 - 2 * fShortDiff;
    
    float fFinalScore = (std::min)(fShortScore, fLongScore);
    
    anglePossibleFix_ = fFinalScore / 100.0;
    
    int totalCount = (bottomLeft_->getCount() + topLeft_->getCount() + topRight_->getCount());
    
    float fCountScore = ((std::max)(3, (std::min)(totalCount, 10)) - 3) / (10.0 - 3.0);
    
    possibleFix_ = anglePossibleFix_ * 0.5 + fCountScore * 0.5;
}
}  // namespace qrcode
}  // namespace zxing
