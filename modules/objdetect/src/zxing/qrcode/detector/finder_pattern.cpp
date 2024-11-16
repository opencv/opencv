// -*- mode:c++; tab-width:2; indent-tabs-mode:nil; c-basic-offset:2 -*-
/*
 *  FinderPattern.cpp
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

#include "finder_pattern.hpp"

using std::abs;
using zxing::Ref;

namespace zxing
{
namespace qrcode
{

FinderPattern::FinderPattern(float posX, float posY, float estimatedModuleSize)
: ResultPoint(posX, posY), estimatedModuleSize_(estimatedModuleSize), count_(1),
horizontalState_(FinderPattern::HORIZONTAL_STATE_NORMAL),
verticalState_(FinderPattern::VERTICAL_STATE_NORMAL)
{
    fix_ = -1.0f;
    pureBarcode_ = false;
    moduleSize_ = -1;
}

FinderPattern::FinderPattern(float posX, float posY, float estimatedModuleSize, int count)
: ResultPoint(posX, posY), estimatedModuleSize_(estimatedModuleSize), count_(count),
horizontalState_(FinderPattern::HORIZONTAL_STATE_NORMAL),
verticalState_(FinderPattern::VERTICAL_STATE_NORMAL)
{
    fix_ = -1.0f;
    pureBarcode_ = false;
    moduleSize_ = -1;
}

int FinderPattern::getCount() const {
    return count_;
}

void FinderPattern::incrementCount() {
    count_++;
}

bool FinderPattern::aboutEquals(float moduleSize, float i, float j) const
{
    if (abs(i - getY()) <= moduleSize && abs(j - getX()) <= moduleSize)
    {
        float moduleSizeDiff = abs(moduleSize - estimatedModuleSize_);
        return moduleSizeDiff <= 1.0f || moduleSizeDiff <= estimatedModuleSize_;
    }
    return false;
}

float FinderPattern::getEstimatedModuleSize() const {
    return estimatedModuleSize_;
}

Ref<FinderPattern> FinderPattern::combineEstimate(float i, float j, float newModuleSize) const
{
    int combinedCount = count_ + 1;
    float combinedX = getX();
    float combinedY = getY();
    float combinedModuleSize = getEstimatedModuleSize();
    if (combinedCount <= 3)
    {
        combinedX = (count_ * getX() + j) / combinedCount;
        combinedY = (count_ * getY() + i) / combinedCount;
        combinedModuleSize = (count_ * getEstimatedModuleSize() + newModuleSize) / combinedCount;
    }
    return Ref<FinderPattern>(new FinderPattern(combinedX, combinedY, combinedModuleSize, combinedCount));
}


void FinderPattern::setHorizontalCheckState(int state) {
    switch (state) {
        case 0:
            horizontalState_ = FinderPattern::HORIZONTAL_STATE_NORMAL;
            break;
        case 1:
            horizontalState_ = FinderPattern::HORIZONTAL_STATE_LEFT_SPILL;
            break;
        case 2:
            horizontalState_ = FinderPattern::HORIZONTAL_STATE_RIGHT_SPILL;
            break;
    }
    return;
}
void FinderPattern::setVerticalCheckState(int state) {
    switch (state) {
        case 0:
            verticalState_ = FinderPattern::VERTICAL_STATE_NORMAL;
            break;
        case 1:
            verticalState_ = FinderPattern::VERTICAL_STATE_UP_SPILL;
            break;
        case 2:
            verticalState_ = FinderPattern::VERTICAL_STATE_DOWN_SPILL;
            break;
    }
    return;
}

#ifdef USING_WX
FinderPattern::FinderPattern(float posX, float posY, const int *horizontalState, const int *verticalState) :
ResultPoint(posX, posY) {
    count_ = 1;
    fix_ = -1.0f;
    for (int i = 0; i < 5; ++i) {
        horizontalState_[i] = horizontalState[i];
        verticalState_[i] = verticalState[i];
    }
    horizontalModuleSize_ = getEstimatedModuleSize(horizontalState_);
    verticalModuleSize_ = getEstimatedModuleSize(verticalState_);
    moduleSize_ = (horizontalModuleSize_ + verticalModuleSize_) / 2.0;
}

bool FinderPattern::aboutEquals(float i, float j, float verticalModuleSize, float horizontalModuleSize) const {
    
    if (abs(i - getY()) <= verticalModuleSize_ && abs(j - getX()) <= horizontalModuleSize_)
    {
        float verticalModuleSizeDiff = abs(verticalModuleSize_ - verticalModuleSize);
        float horizontalModuleSizeDiff = abs(horizontalModuleSize_ - horizontalModuleSize);
        bool result;
        result = (verticalModuleSizeDiff <= 1.0f || verticalModuleSizeDiff <= verticalModuleSize_) &&
        (horizontalModuleSizeDiff <= 1.0f || horizontalModuleSizeDiff <= horizontalModuleSize_);
        return result;
    }
    return false;
}


bool FinderPattern::hadMatchedPatter() {
    return fix_ >= 0.0;
}

float FinderPattern::getFix()const {
    return fix_;
}

void FinderPattern::setFix(float fix) {
    fix_ = fix;
}

const float *FinderPattern::getHorizontalState() const {
    return horizontalState_;
}

const float *FinderPattern::getVerticalState() const {
    return verticalState_;
}

float FinderPattern::getEstimatedModuleSize(const int *state) {
    float total = state[1] + state[2] + state[3];
    float count = 5.0f;
    float unsuitable = 1.5f;
    float moduleSize = total / count;
    
    if (state[0] / moduleSize < unsuitable)
    {
        total += state[0];
        ++count;
    }
    
    if (state[4] / moduleSize < unsuitable)
    {
        total += state[4];
        ++count;
    }
    return total / count;
}

float FinderPattern::getEstimatedModuleSize(const float *state) {
    float total = state[1] + state[2] + state[3];
    float count = 5.0f;
    float unsuitable = 1.5f;
    float moduleSize = total / count;
    
    if (state[0] / moduleSize < unsuitable)
    {
        total += state[0];
        ++count;
    }
    
    if (state[4] / moduleSize < unsuitable)
    {
        total += state[4];
        ++count;
    }
    return total / count;
}

void FinderPattern::combineEstimate(float i, float j, const int *verticalState, const int *horizontalState) {
    for (int i = 0; i < 5; ++i) {
        verticalState_[i] = (verticalState_[i] * count_ + verticalState[i]) / (count_ + 1);
        horizontalState_[i] = (horizontalState_[i] * count_ + horizontalState[i]) / (count_ + 1);
    }
    ++count_;
    verticalModuleSize_ = getEstimatedModuleSize(verticalState_);
    horizontalModuleSize_ = getEstimatedModuleSize(horizontalState_);
    moduleSize_ = (verticalModuleSize_ + horizontalModuleSize_) / 2.0f;
}

float FinderPattern::getEstimatedHorizontalModuleSize() const {
    return horizontalModuleSize_;
}

float FinderPattern::getEstimatedVerticalModuleSize() const {
    return verticalModuleSize_;
}

float FinderPattern::matchPattern(BitMatrix &matrix, bool store) {
    float horizontalStateReal[5];
    float verticalStateReal[5];
    int horizontalState[5];
    int verticalState[5];
    int verticalTotalCount = 0;
    int horizontalTotalCount = 0;
    getHorizontalPatternState(horizontalStateReal);
    getVerticalPatternState(verticalStateReal);
    for (int i = 0; i < 5; ++i) {
        verticalState[i] = static_cast<int>(verticalStateReal[i] + 0.5);
        horizontalState[i] = static_cast<int>(horizontalStateReal[i] + 0.5);
        verticalTotalCount += verticalState[i];
        horizontalTotalCount += horizontalState[i];
    }
    int right = static_cast<int>(getX() + (horizontalState[1] + horizontalState[2] + horizontalState[3]) / 2.0 + horizontalState[4] + 0.5f);
    int left = right - horizontalTotalCount;
    int bottom = static_cast<int>(getY() + (verticalState[1] + verticalState[2] + verticalState[3]) / 2.0 + verticalState[4] + 0.5f);
    int top = bottom - verticalTotalCount;
    
    if (left == -1)
        left = 0;
    if (right == static_cast<int>(matrix.getWidth()) + 1)
        right = static_cast<int>(matrix.getWidth());
    if (top == -1)
        top = 0;
    if (bottom == static_cast<int>(matrix.getHeight()) + 1)
        bottom = matrix.getHeight();
    
    horizontalTotalCount = right - left;
    verticalTotalCount = bottom - top;
    
    int totalCount = verticalTotalCount * horizontalTotalCount;
    int count = 0;
    
    if (left < 0 || right > static_cast<int>(matrix.getWidth()) ||
        top < 0 || bottom > static_cast<int>(matrix.getHeight())) {
        return 0.0;
    }
    int y = top;
    
    // all black
    for (int i = 0; i < verticalState[0]; ++i, ++y) {
        for (int x = left; x < right; ++x)
            if (matrix.get(x, y))
                ++count;
    }
    
    // black white black, 1:5:1
    for (int i = 0; i < verticalState[1]; ++i, ++y) {
        int x = left;
        for (int j = 0; j < horizontalState[0]; ++j, ++x)
            if (matrix.get(x, y))
                ++count;
        int tmp = horizontalState[1] + horizontalState[2] + horizontalState[3];
        for (int j = 0; j < tmp; ++j, ++x)
            if (!matrix.get(x, y))
                ++count;
        for (; x < right; ++x)
            if (matrix.get(x, y))
                ++count;
    }
    
    // black white black white black, 1:1:3:1:1
    for (int i = 0; i < verticalState[2]; ++i, ++y) {
        int x = left;
        for (int j = 0; j < horizontalState[0]; ++j, ++x)
            if (matrix.get(x, y))
                ++count;
        for (int j = 0; j < horizontalState[1]; ++j, ++x)
            if (!matrix.get(x, y))
                ++count;
        for (int j = 0; j < horizontalState[2]; ++j, ++x)
            if (matrix.get(x, y))
                ++count;
        for (int j = 0; j < horizontalState[3]; ++j, ++x)
            if (!matrix.get(x, y))
                ++count;
        for (; x < right; ++x)
            if (matrix.get(x, y))
                ++count;
    }
    
    // black white black, 1:5:1
    for (int i = 0; i < verticalState[3]; ++i, ++y) {
        int x = left;
        for (int j = 0; j < horizontalState[0]; ++j, ++x)
            if (matrix.get(x, y))
                ++count;
        int tmp = horizontalState[1] + horizontalState[2] + horizontalState[3];
        for (int j = 0; j < tmp; ++j, ++x)
            if (!matrix.get(x, y))
                ++count;
        for (; x < right; ++x)
            if (matrix.get(x, y))
                ++count;
    }
    
    // all black
    for (; y < bottom; ++y)
        for (int x = left; x < right; ++x)
            if (matrix.get(x, y))
                ++count;
    
    float fix = count * 100.0 / totalCount;
    if (store)
        fix_ = fix;
    return fix;
}

bool FinderPattern::hadMatchedPattern() {
    return fix_ >= 0.0;
}

void FinderPattern::getHorizontalPatternState(float *state) const {
    float unsuitable = 1.0f;
    state[1] = horizontalState_[1];
    state[2] = horizontalState_[2];
    state[3] = horizontalState_[3];
    state[0] = min(horizontalModuleSize_ * unsuitable, horizontalState_[0]);
    state[4] = min(horizontalModuleSize_ * unsuitable, horizontalState_[4]);
}

void FinderPattern::getVerticalPatternState(float *state) const {
    float unsuitable = 1.0f;
    state[1] = verticalState_[1];
    state[2] = verticalState_[2];
    state[3] = verticalState_[3];
    state[0] = min(verticalModuleSize_ * unsuitable, verticalState_[0]);
    state[4] = min(verticalModuleSize_ * unsuitable, verticalState_[4]);
}

#endif
}  // namespace qrcode
}  // namespace zxing
