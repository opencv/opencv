// -*- mode:c++; tab-width:2; indent-tabs-mode:nil; c-basic-offset:2 -*-
#ifndef __FINDER_PATTERN_H__
#define __FINDER_PATTERN_H__

/*
 *  FinderPattern.hpp
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

#include "../../result_point.hpp"
#include <cmath>
#include "../../common/bit_matrix.hpp"

namespace zxing {
namespace qrcode {

class FinderPattern : public ResultPoint {
public:
    enum  CheckState{
        HORIZONTAL_STATE_NORMAL = 0,
        HORIZONTAL_STATE_LEFT_SPILL = 1,
        HORIZONTAL_STATE_RIGHT_SPILL = 2,
        VERTICAL_STATE_NORMAL = 3,
        VERTICAL_STATE_UP_SPILL = 4,
        VERTICAL_STATE_DOWN_SPILL = 5
    };
    
private:
    float estimatedModuleSize_;
    int count_;
    
    FinderPattern(float posX, float posY, float estimatedModuleSize, int count);
    
public:
    FinderPattern(float posX, float posY, float estimatedModuleSize);
    int getCount() const;
    float getEstimatedModuleSize() const;
    void incrementCount();
    bool aboutEquals(float moduleSize, float i, float j) const;
    Ref<FinderPattern> combineEstimate(float i, float j, float newModuleSize) const;
    
    void setHorizontalCheckState(int state);
    void setVerticalCheckState(int state);
    void setPureBarcode(bool pureBarcode){pureBarcode_ = pureBarcode;}
    
    int getHorizontalCheckState(){return horizontalState_;}
    int getVerticalCheckState(){return verticalState_;}
    bool getPureBarcode(){return pureBarcode_;}
    
private:
    float fix_;
    float moduleSize_;
    CheckState horizontalState_;
    CheckState verticalState_;
    bool pureBarcode_;
    
#ifdef USXING_WX
public:
    FinderPattern(float posX, float posY, const int *horizontalState, const int *verticalState);
    bool aboutEquals(float i, float j, float verticalModuleSize, float horizontalModuleSize) const;
    void combineEstimate(float i, float j, const int *verticalState, const int *horizontalState);
    float getFix() const;
    void setFix(float fix);
    bool hadMatchedPatter();
    const float *getHorizontalState() const;
    const float *getVerticalState() const;
    static float getEstimatedModuleSize(const int *state);
    static float getEstimatedModuleSize(const float *state);
    float getEstimatedHorizontalModuleSize() const;
    float getEstimatedVerticalModuleSize() const;
    bool hadMatchedPattern();
    float matchPattern(BitMatrix &matrix, bool store);
    void getHorizontalPatternState(float *state) const;
    void getVerticalPatternState(float *state) const;
#endif
};
}  // namespace qrcode
}  // namespace zxing

#endif  // QBAR_AI_QBAR_ZXING_QRCODE_DETECTOR_FINDERPATTERN_H_
