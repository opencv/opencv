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
/*
 *  FinderPatternFinder.cpp
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

#include "finder_pattern_finder.hpp"
#include "../../reader_exception.hpp"
#include "../../error_handler.hpp"
#include "../../decode_hints.hpp"
#include "../../common/kmeans.hpp"
#include "../../common/util/inireader.hpp"
#include "../../common/detector/math_utils.hpp"

#include <climits>
#include <cmath>

using std::sort;
using std::max;
using std::abs;
using std::vector;
using zxing::Ref;
using zxing::qrcode::FinderPatternFinder;
using zxing::qrcode::FinderPattern;
using zxing::qrcode::FinderPatternInfo;

// VC++

using zxing::BitMatrix;
using zxing::ResultPointCallback;
using zxing::ResultPoint;
using zxing::DecodeHints;

using namespace zxing;
using namespace qrcode;

#define CHECK_MORE_THAN_ONE_CENTER 1

namespace {
class FinderPatternComination: public Counted
{
public:
    int point[3];
};

class FurthestFromAverageComparator
{
private:
    const float averageModuleSize_;
public:
    FurthestFromAverageComparator(float averageModuleSize) :
    averageModuleSize_(averageModuleSize) {}
    
    bool operator()(Ref<FinderPattern> a, Ref<FinderPattern> b)
    {
        float dA = abs(a->getEstimatedModuleSize() - averageModuleSize_);
        float dB = abs(b->getEstimatedModuleSize() - averageModuleSize_);
        return dA > dB;
    }
};

// Orders by furthes from average
class CenterComparator
{
    const float averageModuleSize_;
public:
    CenterComparator(float averageModuleSize) :
    averageModuleSize_(averageModuleSize) {}
    
    bool operator()(Ref<FinderPattern> a, Ref<FinderPattern> b)
    {
        // N.B.: we want the result in descending order ...
        if (a->getCount() != b->getCount())
        {
            return a->getCount() > b->getCount();
        }
        else
        {
            float dA = abs(a->getEstimatedModuleSize() - averageModuleSize_);
            float dB = abs(b->getEstimatedModuleSize() - averageModuleSize_);
            return dA < dB;
        }
    }
};

class CountComparator
{
public:
    bool operator()(Ref<FinderPattern> a, Ref<FinderPattern> b)
    {
        return a->getCount() > b->getCount();
    }
};

class ModuleSizeComparator
{
public:
    bool operator()(Ref<FinderPattern> a, Ref<FinderPattern> b)
    {
        return a->getEstimatedModuleSize() > b->getEstimatedModuleSize();
    }
};

class BestComparator
{
public:
    bool operator()(Ref<FinderPattern> a, Ref<FinderPattern> b)
    {
        if (a->getCount() != b->getCount())
        {
            return a->getCount() > b->getCount();
        }
        else
        {
            return a->getEstimatedModuleSize() > b->getEstimatedModuleSize();
        }
    }
};
class BestComparator2
{
public:
    bool operator()(Ref<FinderPattern> a, Ref<FinderPattern> b)
    {
        if (a->getCount() != b->getCount())
        {
            return a->getCount() > b->getCount();
        }
        else
        {
            int aErr = 0, bErr = 0;
            if (a->getHorizontalCheckState() != FinderPattern::HORIZONTAL_STATE_NORMAL)  aErr++;
            if (a->getVerticalCheckState() != FinderPattern::VERTICAL_STATE_NORMAL)  aErr++;
            if (b->getHorizontalCheckState() != FinderPattern::HORIZONTAL_STATE_NORMAL)  bErr++;
            if (b->getVerticalCheckState() != FinderPattern::VERTICAL_STATE_NORMAL)  bErr++;
            
            if (aErr != bErr)
            {
                return aErr < bErr;
            }
            else
            {
                return a->getEstimatedModuleSize() > b->getEstimatedModuleSize();
            }
        }
    }
};


class XComparator
{
public:
    bool operator()(Ref<FinderPattern> a, Ref<FinderPattern> b)
    {
        return a->getX() < b->getX();
    }
};

class YComparator
{
public:
    bool operator()(Ref<FinderPattern> a, Ref<FinderPattern> b)
    {
        return a->getY() < b->getY();
    }
};

class ModuleSizeBucket {
public:
    ModuleSizeBucket()
    {
        count = 0;
        centers.clear();
    }
    
public:
    int count;
    
    vector<Ref<FinderPattern> > centers;
};

}  // namespace

int FinderPatternFinder::CENTER_QUORUM = 2;
int FinderPatternFinder::MIN_SKIP = 1;  // 1 pixel/module times MIN_SKIP modules/center
int FinderPatternFinder::MAX_MODULES = 177;  // support up to version 40  which has 177 modules
int FinderPatternFinder::_minModuleSize = 1;  // 1 pixel/module times MIN_SKIP modules/center
int FinderPatternFinder::INTEGER_MATH_SHIFT = 8;

void FinderPatternFinder::initConfig()
{
    FPS_MS_VAL = GetIniParser()->getReal("FP_SELECT", "FPS_MS_VAL", 1.0);
    FP_IS_SELECT_BEST = GetIniParser()->getInteger("FP_SELECT", "FP_IS_SELECT_BEST", 1);
    FP_IS_SELECT_FILE_BEST = GetIniParser()->getInteger("FP_SELECT", "FP_IS_SELECT_FILE_BEST", 1);
    FP_INPUT_MAX_NUM = GetIniParser()->getInteger("FP_SELECT", "FP_INPUT_MAX_NUM", 100);
    FP_FILTER_SIZE = GetIniParser()->getReal("FP_SELECT", "FP_FILTER_SIZE", 100);
    FP_COUNT_MIN = GetIniParser()->getReal("FP_SELECT", "FP_COUNT_MIN", 2.0);
    FP_MS_MIN = GetIniParser()->getReal("FP_SELECT", "FP_MS_MIN", 1.0);
    FPS_CLUSTER_MAX = GetIniParser()->getInteger("FP_SELECT", "FPS_CLUSTER_MAX", 4);
    FPS_RESULT_MAX = GetIniParser()->getInteger("FP_SELECT", "FPS_RESULT_MAX", 12);
    K_FACTOR = GetIniParser()->getInteger("FP_SELECT", "K_FACTOR", 2);
    FP_RIGHT_ANGLE = GetIniParser()->getReal("FP_SELECT", "FP_RIGHT_ANGLE", 0.342);
    FP_SMALL_ANGLE1 = GetIniParser()->getReal("FP_SELECT", "FP_SMALL_ANGLE1", 0.8191);
    FP_SMALL_ANGLE2 = GetIniParser()->getReal("FP_SELECT", "FP_SMALL_ANGLE2", 0.5736);
    
    QR_MIN_FP_AREA_ERR = GetIniParser()->getReal("FP_SELECT", "BLOCK_AREA_ERR", 3);
    QR_MIN_FP_MS_ERR = GetIniParser()->getReal("FP_SELECT", "BLOCK_MS_ERR", 1);
    QR_MIN_FP_ACCEPT = GetIniParser()->getReal("FP_SELECT", "BLOCK_ACCEPT", 4);
}

std::vector<Ref<FinderPatternInfo> > FinderPatternFinder::find(DecodeHints const& hints , ErrorHandler & err_handler)
{
    bool tryHarder = hints.getTryHarder();
    tryHarder_ = tryHarder;
    
    size_t maxI = image_->getHeight();
    size_t maxJ = image_->getWidth();
    
    // Init pre check result
    _horizontalCheckedResult.clear();
    _horizontalCheckedResult.resize(maxJ);
    
    // As this is used often, we use an integer array instead of vector
    int stateCount[5];
    
    // Let's assume that the maximum version QR Code we support
    // (Version 40, 177modules, and finder pattern start at: 0~7) takes up 1/4
    // the height of the image, and then account for the center being 3
    // modules in size. This gives the smallest number of pixels the center
    // could be, so skip this often. When trying harder, look for all
    // QR versions regardless of how dense they are.
    
    int iSkip = (3 * maxI) / (4 * MAX_MODULES);
    if (iSkip < MIN_SKIP || tryHarder)
    {
        iSkip = MIN_SKIP;
    }
    
    int start_i = iSkip - 1;
    int end_i = maxI;
    int start_j = 0;
    int end_j = maxJ;
    if (hints.qbar_points.size() > 0)
    {
        float min_x = INT_MAX;
        float max_x = 0;
        float min_y = INT_MAX;
        float max_y = 0;
        for (size_t kk = 0; kk < hints.qbar_points.size(); kk++) {
            if (hints.qbar_points[kk].x > max_x) max_x = hints.qbar_points[kk].x;
            if (hints.qbar_points[kk].y > max_y) max_y = hints.qbar_points[kk].y;
            if (hints.qbar_points[kk].x < min_x) min_x = hints.qbar_points[kk].x;
            if (hints.qbar_points[kk].y < min_y) min_y = hints.qbar_points[kk].y;
        }
        if (start_j < min_x) start_j = static_cast<int>(min_x);
        if (end_j > max_x) end_j = static_cast<int>(max_x);
        if (start_i < min_y) start_i = static_cast<int>(min_y);
        if (end_i > max_y) end_i = static_cast<int>(max_y);
    }
    
    // This is slightly faster than using the Ref. Efficiency is important here
    BitMatrix& matrix = *image_;
    
    // If we need to use getRowRecords or getRowCounterOffsetEnd, we should call initRowCounters first : Valiantliu
    matrix.initRowCounters();
    
    // for (size_t i = iSkip - 1; i < maxI; i += iSkip)
    for (int i = start_i; i < end_i; i += iSkip)
    {
        COUNTER_TYPE* irow_states = matrix.getRowRecords(i);
        COUNTER_TYPE* irow_offsets = matrix.getRowRecordsOffset(i);
        
        size_t rj = matrix.getRowFirstIsWhite(i)? 1 : 0;
        COUNTER_TYPE row_counter_width = matrix.getRowCounterOffsetEnd(i);
        
        for (; (rj + 4) < size_t(row_counter_width) && (rj + 4) < maxJ; rj += 2)
        {
            stateCount[0] = irow_states[rj];
            stateCount[1] = irow_states[rj + 1];
            stateCount[2] = irow_states[rj + 2];
            stateCount[3] = irow_states[rj + 3];
            stateCount[4] = irow_states[rj + 4];
            
            int j = irow_offsets[rj + 4] + stateCount[4];
            if (j > static_cast<int>(maxJ) || j < start_j || j > end_j)
            {
                rj = row_counter_width - 1;
                continue;
            }
            
            if (foundPatternCross(stateCount))
            {
                if (j == static_cast<int>(maxJ))
                {
                    bool confirmed = handlePossibleCenter(stateCount, i, maxJ);
                    if (confirmed)
                    {
                        iSkip = static_cast<int>(possibleCenters_.back()->getEstimatedModuleSize());
                        if (iSkip < 1) iSkip = 1;
                    }
                    rj = row_counter_width - 1;
                    continue;
                }
                else
                {
                    bool confirmed = handlePossibleCenter(stateCount, i, j);
                    if (confirmed)
                    {
                        // Start examining every other line. Checking each line turned out to be too
                        // expensive and didn't improve performance.
                        iSkip = 2;
                        if (!hasSkipped_)
                        {
                            int rowSkip = findRowSkip();
                            if (rowSkip > stateCount[2])
                            {
                                // Skip rows between row of lower confirmed center and top of presumed third confirmed center
                                // but back up a bit to get a full chance of detecting it, entire width of center of finder pattern
                                // Skip by rowSkip, but back off by stateCount[2] (size of last center of pattern we saw) to
                                // be conservative, and also back off by iSkip which is about to be re-added
                                i += rowSkip - stateCount[2] - iSkip;
                                rj = row_counter_width - 1;
                                j = maxJ - 1;
                            }
                        }
                    }
                    else
                    {
                        continue;
                    }
                    rj += 4;
                }
            }
        }
    }
    
    if (hints.isUseAI() && !hints.getTryVideo())
    {
        for (size_t i = iSkip - 1; i < maxI; i += iSkip)
        {
            COUNTER_TYPE* irow_states = matrix.getRowRecords(i);
            COUNTER_TYPE* irow_offsets = matrix.getRowRecordsOffset(i);
            COUNTER_TYPE row_counter_width = matrix.getRowCounterOffsetEnd(i);
            
            for (size_t rj = matrix.getRowFirstIsWhite(i) ? 1 : 0; (rj + 4) < size_t(row_counter_width); rj += 2)
            {
                if (block_->getUnicomBlockIndex(i, irow_offsets[rj]) == block_->getUnicomBlockIndex(i, irow_offsets[rj + 4]) &&
                    block_->getUnicomBlockIndex(i, irow_offsets[rj + 1]) == block_->getUnicomBlockIndex(i, irow_offsets[rj + 3]) &&
                    block_->getUnicomBlockIndex(i, irow_offsets[rj]) != block_->getUnicomBlockIndex(i, irow_offsets[rj + 2]))
                {
                    const int iBlackCir = block_->getUnicomBlockSize(i, irow_offsets[rj]);
                    const int iWhiteCir = block_->getUnicomBlockSize(i, irow_offsets[rj + 1]);
                    const int iBlackPnt = block_->getUnicomBlockSize(i, irow_offsets[rj + 2]);
                    
                    // we can pass : iBlackCir == iBlackCir
                    // so optimizing
                    if (-1 == iBlackCir || -1 == iWhiteCir) continue;
                    
                    const float fBlackCir = sqrt(iBlackCir / 24.0);
                    const float fWhiteCir = sqrt(iWhiteCir / 16.0);
                    const float fBlackPnt = sqrt(iBlackPnt / 9.0);
                    
                    // a plan for MS
                    const float fRealMS = sqrt((iWhiteCir + iBlackPnt) / 25.0);
                    
                    // b plan for MS
                    int iTotalCount = 0;
                    for (int j = 1; j < 4; ++j) iTotalCount += irow_states[rj + j];
                    const float fEstRowMS = iTotalCount / 5.0;
                    
                    if (fabs(fBlackCir - fWhiteCir) <= QR_MIN_FP_AREA_ERR &&
                        fabs(fWhiteCir - fBlackPnt) <= QR_MIN_FP_AREA_ERR &&
                        fabs(fRealMS - fEstRowMS) < QR_MIN_FP_MS_ERR)
                    {
                        int centerI = 0;
                        int centerJ = 0;
                        if (fRealMS < QR_MIN_FP_ACCEPT)
                        {
                            centerI = i;
                            centerJ = irow_offsets[rj + 2] + irow_states[rj + 2] / 2;
                        }
                        else
                        {
                            int iMinX = 0, iMinY = 0, iMaxX = 0, iMaxY = 0;
                            block_->getMinPoint(i, irow_offsets[rj + 1], iMinY, iMinX);
                            block_->getMaxPoint(i, irow_offsets[rj + 3], iMaxY, iMaxX);
                            centerI = (iMaxY + iMinY) / 2.0;  // y
                            centerJ = (iMaxX + iMinX) / 2.0;  // x
                        }
                        tryToPushToCenters(centerI, centerJ, fRealMS);
                        int rowSkip = findRowSkip();
                        if (rowSkip > irow_states[rj + 2])
                        {
                            // Skip rows between row of lower confirmed center and top of presumed third confirmed center
                            // but back up a bit to get a full chance of detecting it, entire width of center of finder pattern
                            // Skip by rowSkip, but back off by stateCount[2] (size of last center of pattern we saw) to
                            // be conservative, and also back off by iSkip which is about to be re-added
                            i += rowSkip - irow_states[rj + 2] - iSkip;
                            rj = row_counter_width - 1;
                        }
                        rj += 4;
                    }
                }
            }
        }
    }
    
    // filter and sort
    std::vector<Ref<FinderPatternInfo> > patternInfos;
    patternInfos = getPatternInfosFileMode(hints, err_handler);
    if (err_handler.errCode())
    {
        return std::vector<Ref<FinderPatternInfo> >();
    }
    // sort with score
    sort(patternInfos.begin(), patternInfos.end(),
         [](Ref<FinderPatternInfo> a, Ref<FinderPatternInfo> b) {
        return a->getPossibleFix() > b->getPossibleFix();
    });
    
    return patternInfos;
}

int FinderPatternFinder::getMaxMinModuleSize(float& minModuleSize, float& maxModuleSize)
{
    minModuleSize = 1000000000000.0f;
    maxModuleSize = 0.0f;
    
    for (size_t i = 0; i < possibleCenters_.size(); i++)
    {
        float moduleSize = possibleCenters_[i]->getEstimatedModuleSize();
        
        if (moduleSize < minModuleSize)
        {
            minModuleSize = moduleSize;
        }
        if (moduleSize > maxModuleSize)
        {
            maxModuleSize = moduleSize;
        }
    }
    
    return 1;
}

bool FinderPatternFinder::isEqualResult(Ref<FinderPatternInfo> src, Ref<FinderPatternInfo> dst)
{
    if (src == NULL)
    {
        return false;
    }
    
    if (dst == NULL)
    {
        return true;
    }
    
    Ref<FinderPattern> topLeft = src->getTopLeft();
    Ref<FinderPattern> bottomLeft = src->getBottomLeft();
    Ref<FinderPattern> topRight = src->getTopRight();
    
    bool res = true;
    
    res = topLeft->aboutEquals(1.0, dst->getTopLeft()->getY(), dst->getTopLeft()->getX());
    if (res == false)
    {
        return false;
    }
    
    res = bottomLeft->aboutEquals(1.0, dst->getBottomLeft()->getY(), dst->getBottomLeft()->getX());
    if (res == false)
    {
        return false;
    }
    
    res = topRight->aboutEquals(1.0, dst->getTopRight()->getY(), dst->getTopRight()->getX());
    if (res == false)
    {
        return false;
    }
    
    return true;
}

// Added by Valiantliu : try all possible patterns
vector<Ref<FinderPatternInfo> > FinderPatternFinder::getPatternInfos(ErrorHandler & err_handler)
{
    size_t startSize = possibleCenters_.size();
    
    if (startSize < 3)
    {
        // Couldn't find enough finder patterns
        err_handler = ReaderErrorHandler("Could not find three finder patterns");
        return vector< Ref<FinderPatternInfo> >();
    }
    
    std::vector<Ref<FinderPatternInfo> > patternInfos;
    vector<Ref<FinderPattern> > result(3);
    
    if (startSize == 3)
    {
        result.resize(3);
        result[0] = possibleCenters_[0];
        result[1] = possibleCenters_[1];
        result[2] = possibleCenters_[2];
        
        vector<Ref<FinderPattern> > finderPattern = orderBestPatterns(result);
        
        Ref<FinderPatternInfo> patternInfo(new FinderPatternInfo(finderPattern));
        
        patternInfos.push_back(patternInfo);
        return patternInfos;
    }
    
    vector<Ref<FinderPattern> > finderPatterns;
    Ref<FinderPatternInfo> resultBest;
    
    // Handle possible centers for both pure and normal qrcode
    finderPatterns = selectBestPatterns(err_handler);
    if (err_handler.errCode() == 0)
    {
        finderPatterns = orderBestPatterns(finderPatterns);
        resultBest = new FinderPatternInfo(finderPatterns);
        patternInfos.push_back(resultBest);
    }
    
    // Also try other best patterns
    int pSize = static_cast<int>(possibleCenters_.size());
    
    float minModuleSize;
    float maxModuleSize;
    
    getMaxMinModuleSize(minModuleSize, maxModuleSize);
    float stepModuleSize = (maxModuleSize - minModuleSize) / static_cast<float>(pSize);
    
    // Get match from module size bucket
    vector<ModuleSizeBucket> moduleSizeBucket(pSize);
    
    int maxCount = 0;
    // int maxIdx = 0;
    
    for (int i = 0; i < pSize; i++)
    {
        float moduleSize = possibleCenters_[i]->getEstimatedModuleSize();
        int moduleSizeIdx = (moduleSize - minModuleSize) / stepModuleSize;
        if (moduleSizeIdx < 0)
        {
            moduleSizeIdx = 0;
        }
        else if (moduleSizeIdx >= pSize)
        {
            moduleSizeIdx = pSize - 1;
        }
        
        moduleSizeBucket[moduleSizeIdx].count++;
        moduleSizeBucket[moduleSizeIdx].centers.push_back(possibleCenters_[i]);
        
        if (moduleSizeBucket[moduleSizeIdx].count > maxCount)
        {
            maxCount = moduleSizeBucket[moduleSizeIdx].count;
            // maxIdx = moduleSizeIdx;
        }
    }
    
    int iListMaxCheck = 9;
    for (int i = pSize - 1; i >= 0; --i)
    {
        int iListSize = moduleSizeBucket[i].count;
        if (iListSize > iListMaxCheck) iListSize = iListMaxCheck;
        for (int x = 0; x < iListSize; ++x)
        {
            result[0] = moduleSizeBucket[i].centers[x];
            for (int y = x + 1; y < iListSize; ++y)
            {
                result[1] =  moduleSizeBucket[i].centers[y];
                for (int z = y + 1; z < iListSize; ++z)
                {
                    result[2] = moduleSizeBucket[i].centers[z];
                    float longSize = 0;
                    if (checkIsoscelesRightTriangle(moduleSizeBucket[i].centers[x], moduleSizeBucket[i].centers[y], moduleSizeBucket[i].centers[z], longSize))
                    {
                        Ref<FinderPatternInfo> patternInfo(new FinderPatternInfo(result));
                        
                        if (isEqualResult(resultBest, patternInfo) == false)
                        {
                            patternInfos.push_back(patternInfo);
                        }
                    }
                }
            }
        }
    }
    return patternInfos;
}

bool FinderPatternFinder::isPossibleFindPatterInfo( Ref<FinderPattern> a, Ref<FinderPattern> b, Ref<FinderPattern> c)
{
    // check fangcha
    float aMs = a->getEstimatedModuleSize();
    float bMs = b->getEstimatedModuleSize();
    float cMs = c->getEstimatedModuleSize();
    
    float avg = (aMs + bMs + cMs) / 3.0;
    float val = sqrt((aMs - avg) * (aMs - avg) + (bMs - avg) * (bMs - avg) + (cMs - avg) * (cMs - avg));
    
    if (val >= FPS_MS_VAL)
        return false;
    
    float longSize = 0.0;
    if (!checkIsoscelesRightTriangle(a, b, c, longSize))
    {
        return false;
    }
    
    return true;
}

void FinderPatternFinder::pushToResult( Ref<FinderPattern> a, Ref<FinderPattern> b, Ref<FinderPattern> c, vector<Ref<FinderPatternInfo> > & patternInfos)
{
    vector< Ref<FinderPattern> > finderPatterns;
    finderPatterns.push_back(a); finderPatterns.push_back(b); finderPatterns.push_back(c);
    vector<Ref<FinderPattern> > finderPattern = orderBestPatterns(finderPatterns);
    
    Ref<FinderPatternInfo> patternInfo(new FinderPatternInfo(finderPattern));
    
    for (size_t j = 0; j < patternInfos.size(); j++)
    {
        if (isEqualResult(patternInfos[j], patternInfo))
        {
            return;
        }
    }
    patternInfos.push_back(patternInfo);
}

vector<Ref<FinderPatternInfo> > FinderPatternFinder::getPatternInfosFileMode(DecodeHints const& hints, ErrorHandler& err_handler)
{
    size_t startSize = possibleCenters_.size();
    
    if (startSize < 3)
    {
        // Couldn't find enough finder patterns
        err_handler = ReaderErrorHandler("Could not find three finder patterns");
        return vector< Ref<FinderPatternInfo> >();
    }
    
    // filter - add by sofiawu
    for (size_t i = 0; i < possibleCenters_.size(); i++)
    {
        if (!(possibleCenters_[i]->getEstimatedModuleSize() >= FP_MS_MIN &&
              possibleCenters_[i]->getCount() >= FP_COUNT_MIN))
        {
            possibleCenters_.erase(possibleCenters_.begin() + i);
            i--;
        }
    }
    startSize = possibleCenters_.size();
    if (startSize < 3)
    {
        err_handler = ReaderErrorHandler("Could not find three finder patterns");
        return vector< Ref<FinderPatternInfo> >();
    }
    // end add
    
    std::vector<Ref<FinderPatternInfo> > patternInfos;
    
    if (startSize == 3)
    {
        pushToResult(possibleCenters_[0], possibleCenters_[1] , possibleCenters_[2], patternInfos);
        return patternInfos;
    }
    
    vector<Ref<FinderPattern> > finderPatterns;
    Ref<FinderPatternInfo> resultBest;
    
    if (hints.qbar_points.size() > 0)
    {
        std::vector<float> possible_dists;
        std::vector<int> possible_idxs;
        
        for (size_t i = 0; i < hints.qbar_points.size(); i++) {
            float min_dist = INT_MAX;
            int min_idx = -1;
            for (size_t j = 0; j < possibleCenters_.size(); j++) {
                float dist = fabs(possibleCenters_[j]->getX() - hints.qbar_points[i].x) + fabs(possibleCenters_[j]->getY() - hints.qbar_points[i].y);
                if (dist < min_dist)
                {
                    min_dist = dist;
                    min_idx = j;
                }
            }
            possible_dists.push_back(min_dist);
            possible_idxs.push_back(min_idx);
        }
        int max_idx = -1;
        float max_dist = -1;
        for (size_t i = 0; i < possible_idxs.size(); i++) {
            if (possible_dists[i] > max_dist)
            {
                max_dist = possible_dists[i];
                max_idx = i;
            }
        }
        possible_idxs.erase(possible_idxs.begin() + max_idx);
        if (possible_idxs.size() == 3)
        {
            pushToResult(possibleCenters_[possible_idxs[0]], possibleCenters_[possible_idxs[1]], possibleCenters_[possible_idxs[2]], patternInfos);
        }
    }
    
    // select best
    if (FP_IS_SELECT_BEST)
    {
        finderPatterns = selectBestPatterns(err_handler);
        if (err_handler.errCode() == 0)
            pushToResult(finderPatterns[0], finderPatterns[1], finderPatterns[2], patternInfos);
    }
    if (FP_IS_SELECT_FILE_BEST)
    {
        finderPatterns = selectFileBestPatterns(err_handler);
        if (err_handler.errCode() == 0)
            pushToResult(finderPatterns[0], finderPatterns[1], finderPatterns[2], patternInfos);
    }
    
    // kmean
    /*const int maxepoches = 100;
     const int minchanged = 0;
     
     // sort and filter
     sort(possibleCenters_.begin(), possibleCenters_.end(), ModuleSizeComparator());
     std::vector<Ref<FinderPattern> > standardCenters;
     
     int const FP_INPUT_CNN_MAX_NUM = 10;
     for (size_t i = 0; i < possibleCenters_.size(); i++)
     {
     if (possibleCenters_[i]->getEstimatedModuleSize() >= FP_MS_MIN &&
     possibleCenters_[i]->getCount() >= FP_COUNT_MIN)
     {
     standardCenters.push_back(possibleCenters_[i]);
     if (standardCenters.size() >= size_t(FP_INPUT_MAX_NUM)) break;
     if (hints.isUseAI() && standardCenters.size() >= FP_INPUT_CNN_MAX_NUM) break;
     }
     }
     
     if (standardCenters.size() < 3)
     {
     err_handler = ReaderErrorHandler("Could not find three finder patterns");
     return vector< Ref<FinderPatternInfo> >();
     }
     
     if (standardCenters.size() <= FP_INPUT_CNN_MAX_NUM)
     {
     for (uint x = 0; x < standardCenters.size(); x++)
     {
     for (uint y = x + 1; y < standardCenters.size(); y++)
     {
     for (uint z = y + 1; z < standardCenters.size(); z++)
     {
     bool check_result = isPossibleFindPatterInfo(
     standardCenters[x], standardCenters[y], standardCenters[z]);
     if (check_result)
     {
     pushToResult(standardCenters[x], standardCenters[y], standardCenters[z],
     patternInfos);
     }
     }
     }
     }
     return patternInfos;
     }
     
     // calculate K
     int k = log(static_cast<float>(standardCenters.size())) * K_FACTOR -1;
     if (k < 1) k = 1;
     
     vector<vector<double> > trainX;
     for (size_t i = 0; i < standardCenters.size(); i++)
     {
     vector<double> tmp;
     tmp.push_back(standardCenters[i]->getCount());
     tmp.push_back(standardCenters[i]->getEstimatedModuleSize());
     trainX.push_back(tmp);
     }
     
     vector<Cluster> clusters_out = k_means(trainX, k, maxepoches, minchanged);
     
     for (uint i = 0; i < clusters_out.size(); i++)
     {
     int cluster_select = 0;
     
     if (clusters_out[i].samples.size() < 3)
     {
     if (i< clusters_out.size()-1 && clusters_out[i + 1].samples.size() < 3)
     {
     for (size_t j = 0; j < clusters_out[i].samples.size(); j++)
     clusters_out[i + 1].samples.push_back(clusters_out[i].samples[j]);
     }
     continue;
     }
     
     vector<Ref<FinderPattern> > clusterPatterns;
     for (size_t j = 0; j < clusters_out[i].samples.size(); j++)
     {
     clusterPatterns.push_back(standardCenters[clusters_out[i].samples[j]]);
     }
     
     sort(clusterPatterns.begin(), clusterPatterns.end(), BestComparator2());
     
     for (uint x = 0; x < clusters_out[i].samples.size() && cluster_select <= FPS_CLUSTER_MAX && patternInfos.size() <= size_t(FPS_RESULT_MAX); x++)
     {
     for (uint y = x + 1; y < clusters_out[i].samples.size() && cluster_select <= FPS_CLUSTER_MAX && patternInfos.size() <= size_t(FPS_RESULT_MAX); y++)
     {
     for (uint z = y + 1; z < clusters_out[i].samples.size() && cluster_select <= FPS_CLUSTER_MAX && patternInfos.size() <= size_t(FPS_RESULT_MAX); z++)
     {
     bool check_result = isPossibleFindPatterInfo(clusterPatterns[x], clusterPatterns[y], clusterPatterns[z]);
     if (check_result)
     {
     pushToResult(clusterPatterns[x], clusterPatterns[y], clusterPatterns[z], patternInfos);
     cluster_select++;
     }
     }
     }
     }
     }*/
    return patternInfos;
}

// Given a count of black/white/black/white/black pixels just seen and an end position,
// figures the location of the center of this run.
float FinderPatternFinder::centerFromEnd(int* stateCount, int end) {
    // calculate the center by pattern 1:3:1 is better than pattern 3
    // because the finder pattern is irregular in some case
    return static_cast<float>(end - stateCount[4]) - (stateCount[3] + stateCount[2] + stateCount[1]) / 2.0f;
}

// return if the proportions of the counts is close enough to 1/1/3/1/1 ratios
// used by finder patterns to be considered a match
bool FinderPatternFinder::foundPatternCross(int* stateCount)
{
    int totalModuleSize = 0;
    
    int stateCountINT[5];
    
    for (int i = 0; i < 5; i++)
    {
        if (stateCount[i] <= 0)
        {
            return false;
        }
        stateCountINT[i] = stateCount[i] << INTEGER_MATH_SHIFT;
        totalModuleSize += stateCount[i];
    }
    
    if (totalModuleSize < 7)
    {
        return false;
    }
    
    int minModuleSizeINT = 3;
    minModuleSizeINT <<= INTEGER_MATH_SHIFT;
    
    CURRENT_CHECK_STATE = FinderPatternFinder::NOT_PATTERN;
    
    totalModuleSize = totalModuleSize << INTEGER_MATH_SHIFT;
    
    // Newer version to check 1 time, use 3 points
    
    int moduleSize = ((totalModuleSize - stateCountINT[0] - stateCountINT[4])) / 5;
    
    int maxVariance = moduleSize;
    
    if (!getFinderLoose() || moduleSize > minModuleSizeINT)
        maxVariance = moduleSize / 2;
    
    int startCountINT = stateCountINT[0];
    int endCountINT = stateCountINT[4];
    
    bool leftFit = (abs(moduleSize - startCountINT) <= maxVariance);
    bool rightFit = (abs(moduleSize - endCountINT) <= maxVariance);
    
    if (leftFit)
    {
        if (rightFit)
        {
            moduleSize = totalModuleSize / 7;
            CURRENT_CHECK_STATE = FinderPatternFinder::NORMAL;
        }
        else
        {
            moduleSize = (totalModuleSize - stateCountINT[4]) / 6;
            CURRENT_CHECK_STATE = FinderPatternFinder::RIHGT_SPILL;
        }
    }
    else
    {
        if (rightFit)
        {
            moduleSize = (totalModuleSize - stateCountINT[0]) / 6;
            CURRENT_CHECK_STATE = FinderPatternFinder::LEFT_SPILL;
        }
        else
        {
            // return false;
            CURRENT_CHECK_STATE = FinderPatternFinder::LEFT_RIGHT_SPILL;
        }
    }
    
    // 1:1:3:1:1 || n:1:3:1:1 || 1:1:3:1:n
    if (	abs(moduleSize - stateCountINT[1]) <= maxVariance &&
       abs(3 * moduleSize - stateCountINT[2]) <= 2 * maxVariance &&
       abs(moduleSize - stateCountINT[3]) <= maxVariance)
    {
        return true;
    }
    
    return false;
}


int FinderPatternFinder::getMinModuleSize()
{
    int minModuleSize = (3 * (std::min)(image_->getWidth(), image_->getHeight())) / (4 * MAX_MODULES);
    
    if (minModuleSize < MIN_SKIP)
    {
        minModuleSize = MIN_SKIP;
    }
    
    return minModuleSize;
}

// return if the proportions of the counts is close enough to 1/1/3/1/1 ratios
// used by finder patterns to be considered a match
bool FinderPatternFinder::foundPatternCrossLoose(int* stateCount)
{
    int totalModuleSize = 0;
    
    int stateCountINT[5];
    
    int minModuleSizeINT = 3;
    minModuleSizeINT <<= INTEGER_MATH_SHIFT;
    
    for (int i = 0; i < 5; i++)
    {
        if (stateCount[i] <= 0)
        {
            return false;
        }
        stateCountINT[i] = stateCount[i] << INTEGER_MATH_SHIFT;
        totalModuleSize += stateCount[i];
    }
    if (totalModuleSize < 7)
    {
        return false;
    }
    
    CURRENT_CHECK_STATE = FinderPatternFinder::NOT_PATTERN;
    
    totalModuleSize = totalModuleSize << INTEGER_MATH_SHIFT;
    
    // Older version to check 3 times, use 4 or 5 points
    int moduleSize = totalModuleSize / 7;
    
    int maxVariance = moduleSize;
    
    if (!getFinderLoose() || moduleSize > minModuleSizeINT)
        maxVariance = moduleSize / 2;
    
    // 1:1:3:1:1
    if (abs(moduleSize-stateCountINT[0]) < maxVariance &&
       abs(moduleSize - stateCountINT[1]) < maxVariance &&
       abs(3 * moduleSize - stateCountINT[2]) < 1.5 * maxVariance &&
       abs(moduleSize - stateCountINT[3]) < maxVariance &&
       abs(moduleSize-stateCount[4]) < maxVariance)
    {
        CURRENT_CHECK_STATE = FinderPatternFinder::NORMAL;
        return true;
    }
    
    // n:1:3:1:1 (but n > 1, by Valiantliu)
    moduleSize = (totalModuleSize - stateCountINT[0]) / 6;
    maxVariance = moduleSize;
    
    if (!getFinderLoose() || moduleSize > minModuleSizeINT)
        maxVariance= moduleSize / 2;
    
    if (	abs(moduleSize - stateCountINT[1]) < maxVariance &&
       abs(3 * moduleSize - stateCountINT[2]) <  1.5 * maxVariance &&
       abs(moduleSize-stateCountINT[3]) <maxVariance &&
       abs(moduleSize - stateCountINT[4]) < maxVariance)
    {
        CURRENT_CHECK_STATE = FinderPatternFinder::LEFT_SPILL;
        return true;
    }
    
    // 1:1:3:1:n (but n > 1, by Valiantliu)
    moduleSize = (totalModuleSize-stateCountINT[4]) / 6;
    maxVariance = moduleSize;
    
    if (!getFinderLoose() || moduleSize > minModuleSizeINT)
        maxVariance= moduleSize / 2;
    
    if (	abs(moduleSize - stateCountINT[0]) < maxVariance &&
       abs(moduleSize - stateCountINT[1]) < maxVariance &&
       abs(3 * moduleSize - stateCountINT[2]) <  1.5 * maxVariance &&
       abs(moduleSize - stateCountINT[3]) < maxVariance)
    {
        CURRENT_CHECK_STATE = FinderPatternFinder::RIHGT_SPILL;
        return true;
    }
    
    // n:1:3:1:n (but n > 1, by Valiantliu)
    moduleSize = (totalModuleSize - stateCountINT[4] - stateCountINT[0]) / 5;
    maxVariance = moduleSize;
    
    if (!getFinderLoose() || moduleSize > minModuleSizeINT)
        maxVariance= moduleSize / 2;
    
    if (	abs(moduleSize - stateCountINT[1]) < maxVariance &&
       abs(3 * moduleSize - stateCountINT[2]) <  1.5 * maxVariance &&
       abs(moduleSize - stateCountINT[3]) < maxVariance)
    {
        if ((stateCountINT[4] - moduleSize) > -maxVariance && (stateCountINT[0] - moduleSize) > -maxVariance)
        {
            CURRENT_CHECK_STATE = FinderPatternFinder::LEFT_RIGHT_SPILL;
            return true;
        }
    }
    return false;
}

/**
 * After a vertical and horizontal scan finds a potential finder pattern, this method
 * "cross-cross-cross-checks" by scanning down diagonally through the center of the possible
 * finder pattern to see if the same proportion is detected.
 *
 * @param startI row where a finder pattern was detected
 * @param centerJ center of the section that appears to cross a finder pattern
 * @param maxCount maximum reasonable number of modules that should be
 *  observed in any reading state, based on the results of the horizontal scan
 * @param originalStateCountTotal The original state count total.
 * @return true if proportions are withing expected limits
 */
bool FinderPatternFinder::crossCheckDiagonal(int startI, int centerJ, int maxCount, int originalStateCountTotal)
{
    int maxI = image_->getHeight();
    int maxJ = image_->getWidth();
    
    // Fix possible crash 20140418
    if ((startI < 0) || (startI > maxI - 1) || (centerJ < 0) || (centerJ > maxJ - 1))
    {
        return false;
    }
    
    int stateCount[5];
    stateCount[0] = 0;
    stateCount[1] = 0;
    stateCount[2] = 0;
    stateCount[3] = 0;
    stateCount[4] = 0;
    
    
    if (getFinderLoose())
    {
        if (!image_->get(centerJ, startI))
        {
            if (static_cast<int>(startI) + 1 < maxI  && image_->get(centerJ, startI + 1))
                startI = startI + 1;
            else if (0 < static_cast<int>(startI) - 1 && image_->get(centerJ, startI - 1))
                startI = startI - 1;
            else
                return false;
        }
    }
    
    // This is slightly faster than using the Ref. Efficiency is important here
    BitMatrix& matrix = *image_;
    
    // Start counting up, left from center finding black center mass
    int i = 0;
    // Fix possible crash 20140418
    while ((startI - i >= 0) && (centerJ - i >= 0) && matrix.get(centerJ - i, startI - i))
    {
        stateCount[2]++;
        i++;
    }
    
    if ((startI - i < 0) || (centerJ - i < 0))
    {
        return false;
    }
    
    // Continue up, left finding white space
    while ((startI - i >= 0) && (centerJ - i >= 0) && !matrix.get(centerJ - i, startI - i) && stateCount[1] <= maxCount)
    {
        stateCount[1]++;
        i++;
    }
    
    // If already too many modules in this state or ran off the edge:
    if ((startI - i < 0) || (centerJ - i < 0) || stateCount[1] > maxCount)
    {
        return false;
    }
    
    CrossCheckState tmpCheckState = FinderPatternFinder::NORMAL;
    
    // Continue up, left finding black border
    while ((startI - i >= 0) && (centerJ - i >= 0) && matrix.get(centerJ - i, startI - i) && stateCount[0] <= maxCount)
    {
        stateCount[0]++;
        i++;
    }
    
    if (stateCount[0] >= maxCount)
    {
        tmpCheckState = FinderPatternFinder::LEFT_SPILL;
    }
    
    // Now also count down, right from center
    i = 1;
    while ((startI + i < maxI) && (centerJ + i < maxJ) && matrix.get(centerJ + i, startI + i))
    {
        stateCount[2]++;
        i++;
    }
    
    // Ran off the edge?
    if ((startI + i >= maxI) || (centerJ + i >= maxJ))
    {
        return false;
    }
    
    while ((startI + i < maxI) && (centerJ + i < maxJ) && !matrix.get(centerJ + i, startI + i) && stateCount[3] < maxCount)
    {
        stateCount[3]++;
        i++;
    }
    
    if ((startI + i >= maxI) || (centerJ + i >= maxJ) || stateCount[3] >= maxCount)
    {
        return false;
    }
    
    while ((startI + i < maxI) && (centerJ + i < maxJ) && matrix.get(centerJ + i, startI + i) && stateCount[4] < maxCount)
    {
        stateCount[4]++;
        i++;
    }
    
    if (tmpCheckState==FinderPatternFinder::LEFT_SPILL)
    {
        if (stateCount[4] >= maxCount)
        {
            tmpCheckState = FinderPatternFinder::LEFT_RIGHT_SPILL;
        }
    }
    else
    {
        if (stateCount[4] >= maxCount)
        {
            tmpCheckState = FinderPatternFinder::RIHGT_SPILL;
        }
    }
    
    // If we found a finder-pattern-like section, but its size is more than 100% different than
    // the original, assume it's a false positive
    bool diagonal_check = foundPatternCross(stateCount);
    if (!diagonal_check)
        return false;
    
    if ( CURRENT_CHECK_STATE == FinderPatternFinder::LEFT_SPILL)
    {
        if (tmpCheckState == FinderPatternFinder::RIHGT_SPILL)
            return false;
    }
    else if ( CURRENT_CHECK_STATE == FinderPatternFinder::RIHGT_SPILL)
    {
        if (tmpCheckState == FinderPatternFinder::LEFT_SPILL)
            return false;
    }
    
    int stateCountTotal = stateCount[1] + stateCount[2] + stateCount[3];
    
    if (CURRENT_CHECK_STATE == FinderPatternFinder::NORMAL)
    {
        stateCountTotal = stateCountTotal + stateCount[0] + stateCount[4];
    }
    else if (CURRENT_CHECK_STATE == FinderPatternFinder::LEFT_SPILL)
    {
        stateCountTotal = stateCountTotal + stateCount[1] + stateCount[4];
    }
    else if (CURRENT_CHECK_STATE == FinderPatternFinder::RIHGT_SPILL)
    {
        stateCountTotal = stateCountTotal + stateCount[0] + stateCount[3];
    }
    else if (CURRENT_CHECK_STATE == FinderPatternFinder::LEFT_RIGHT_SPILL)
    {
        stateCountTotal = stateCountTotal + stateCount[1] + stateCount[3];
    }
    
    if (abs(stateCountTotal - originalStateCountTotal) < 2 * originalStateCountTotal)
    {
        return true;
    }
    else
    {
        return false;
    }
}


// After a horizontal scan finds a potential finder pattern, this method "cross-checks"
// by scanning down vertically through the center of the possible finder pattern to see
// if the same proportion is detected.
float FinderPatternFinder::crossCheckVertical(size_t startI, size_t centerJ, 
                                              int maxCount, int originalStateCountTotal,
                                              float& estimatedVerticalModuleSize)
{
    int maxI = image_->getHeight();
    
    int stateCount[5];
    for (int i = 0; i < 5; i++)
        stateCount[i] = 0;
    
    if (getFinderLoose())
    {
        if (!image_->get(centerJ, startI))
        {
            if (static_cast<int>(startI) + 1 < maxI  && image_->get(centerJ, startI + 1))
                startI = startI + 1;
            else if (0 < static_cast<int>(startI) - 1 && image_->get(centerJ, startI - 1))
                startI = startI - 1;
            else
                return nan();
        }
    }
    
    // This is slightly faster than using the Ref. Efficiency is important here
    BitMatrix& matrix = *image_;
    
    bool* imageRow0 = matrix.getRowBoolPtr(0);
    bool* p = imageRow0;
    int imgWidth = matrix.getWidth();
    
    // Start counting up from center
    int ii = startI;
    
    p = imageRow0 + ii*imgWidth + centerJ;
    
    while (ii >= 0 && *p)
    {
        stateCount[2]++;
        ii--;
        p -= imgWidth;
    }
    if (ii < 0)
    {
        return nan();
    }
    
    while (ii >= 0 && !*p && stateCount[1] <= maxCount)
    {
        stateCount[1]++;
        ii--;
        p -= imgWidth;
    }
    // If already too many modules in this state or ran off the edge:
    if (ii < 0 || stateCount[1] > maxCount)
    {
        return nan();
    }
    
    CrossCheckState tmpCheckState=FinderPatternFinder::NORMAL;
    
    while (ii >= 0 && *p)
    {
        stateCount[0]++;
        ii--;
        p -= imgWidth;
    }
    if (stateCount[0] >= maxCount)
    {
        tmpCheckState = FinderPatternFinder::LEFT_SPILL;
    }
    
    // Now also count down from center
    ii = startI + 1;
    
    p = imageRow0 + ii * imgWidth + centerJ;
    
    while (ii < maxI && *p)
    {
        stateCount[2]++;
        ii++;
        
        p += imgWidth;
    }
    if (ii == maxI)
    {
        return nan();
    }
    
    while (ii < maxI && !*p && stateCount[3] < maxCount)
    {
        stateCount[3]++;
        ii++;
        
        p += imgWidth;
    }
    if (ii == maxI || stateCount[3] >= maxCount)
    {
        return nan();
    }
    
    if (tmpCheckState == FinderPatternFinder::LEFT_SPILL)
    {
        while (ii < maxI && *p && stateCount[4] < maxCount)
        {
            stateCount[4]++;
            ii++;
            
            p += imgWidth;
        }
        if (stateCount[4] >= maxCount)
        {
            return nan();
        }
    }
    else
    {
        while (ii < maxI && *p)
        {
            stateCount[4]++;
            ii++;
            
            p += imgWidth;
        }
        if (stateCount[4] >= maxCount)
        {
            tmpCheckState = FinderPatternFinder::RIHGT_SPILL;
        }
    }
    
    // If we found a finder-pattern-like section, but its size is more than 40% different than
    // the original, assume it's a false positive
    
    bool vertical_check = foundPatternCross(stateCount);
    if (!vertical_check)
        return nan();
    
    // Cannot be a LEFT-RIGHT center
    if ( CURRENT_CHECK_STATE == FinderPatternFinder::LEFT_SPILL)
    {
        if (tmpCheckState == FinderPatternFinder::RIHGT_SPILL)
            return nan();
    }
    else if ( CURRENT_CHECK_STATE == FinderPatternFinder::RIHGT_SPILL)
    {
        if (tmpCheckState == FinderPatternFinder::LEFT_SPILL)
            return nan();
    }
    
    int stateCountTotal = stateCount[1] + stateCount[2] + stateCount[3];
    
    if (CURRENT_CHECK_STATE == FinderPatternFinder::NORMAL)
    {
        stateCountTotal = stateCountTotal + stateCount[0] + stateCount[4];
    }
    else if (CURRENT_CHECK_STATE == FinderPatternFinder::LEFT_SPILL)
    {
        stateCountTotal = stateCountTotal + stateCount[1] + stateCount[4];
    }
    else if (CURRENT_CHECK_STATE == FinderPatternFinder::RIHGT_SPILL)
    {
        stateCountTotal = stateCountTotal + stateCount[0] + stateCount[3];
    }
    
    if (5 * abs(stateCountTotal - originalStateCountTotal) >= 2 * originalStateCountTotal)
    {
        return nan();
    }
    
    estimatedVerticalModuleSize = static_cast<float>(stateCountTotal) / 7.0f;
    
    return  centerFromEnd(stateCount, ii);
}


// Like #crossCheckVertical(), and in fact is basically identical,
// except it reads horizontally instead of vertically. This is used to cross-cross check
// a vertical cross check and locate the real center of the alignment pattern.
float FinderPatternFinder::crossCheckHorizontal(size_t startJ, size_t centerI, int maxCount,
                                                int originalStateCountTotal, float& estimatedHorizontalModuleSize)
{
    int maxJ = image_->getWidth();
    
    int stateCount[5];
    for (int i = 0; i < 5; i++)
        stateCount[i] = 0;
    
    if (getFinderLoose())
    {
        if (!image_->get(startJ, centerI))
        {
            if (static_cast<int>(startJ) + 1 < maxJ  && image_->get(startJ + 1, centerI))
                startJ = startJ + 1;
            else if (0 < static_cast<int>(startJ) - 1 && image_->get(startJ - 1, centerI))
                startJ = startJ - 1;
            else
                return nan();
        }
    }
    
    // This is slightly faster than using the Ref. Efficiency is important here
    BitMatrix& matrix = *image_;
    // int bitsize = matrix.getRowBitsSize();
    int j = startJ;
    
    bool* centerIrow = NULL;
    
    
    centerIrow = matrix.getRowBoolPtr(centerI);
    
    while (j >= 0 && centerIrow[j])
    {
        stateCount[2]++;
        j--;
    }
    if (j < 0)
    {
        return nan();
    }
    
    while (j >= 0 && !centerIrow[j] && stateCount[1] <= maxCount)
    {
        stateCount[1]++;
        j--;
    }
    if (j < 0 || stateCount[1] > maxCount)
    {
        return nan();
    }
    
    CrossCheckState tmpCheckState = FinderPatternFinder::NORMAL;
    
    while (j >= 0 &&centerIrow[j])
    {
        stateCount[0]++;
        j--;
    }
    if (stateCount[0] >= maxCount)
    {
        tmpCheckState = FinderPatternFinder::LEFT_SPILL;
    }
    
    j = startJ + 1;
    while (j < maxJ && centerIrow[j])
    {
        stateCount[2]++;
        j++;
    }
    if (j == maxJ)
    {
        return nan();
    }
    
    while (j < maxJ && !centerIrow[j] && stateCount[3] < maxCount)
    {
        stateCount[3]++;
        j++;
    }
    if (j == maxJ || stateCount[3] >= maxCount)
    {
        return nan();
    }
    
    if (tmpCheckState == LEFT_SPILL)
    {
        while (j < maxJ && centerIrow[j] && stateCount[4] <= maxCount)
        {
            stateCount[4]++;
            j++;
        }
        if (stateCount[4] >= maxCount)
        {
            return nan();
        }
    }
    else
    {
        while (j < maxJ && centerIrow[j])
        {
            stateCount[4]++;
            j++;
        }
        if (stateCount[4] >= maxCount)
        {
            tmpCheckState = RIHGT_SPILL;
        }
    }
    
    while (j < maxJ && centerIrow[j])
    {
        stateCount[4]++;
        j++;
    }
    
    // If we found a finder-pattern-like section, but its size is significantly different than
    // the original, assume it's a false positive
    // int stateCountTotal = stateCount[0] + stateCount[1] + stateCount[2] + stateCount[3] + stateCount[4];
    bool horizontal_check = foundPatternCross(stateCount);
    if (!horizontal_check)
        return nan();
    
    // Cannot be a LEFT-RIGHT center
    if ( CURRENT_CHECK_STATE == FinderPatternFinder::LEFT_SPILL)
    {
        if (tmpCheckState == FinderPatternFinder::RIHGT_SPILL)
            return nan();
    }
    else if ( CURRENT_CHECK_STATE == FinderPatternFinder::RIHGT_SPILL)
    {
        if (tmpCheckState == FinderPatternFinder::LEFT_SPILL)
            return nan();
    }
    
    int stateCountTotal = stateCount[1] + stateCount[2] + stateCount[3];
    
    if (CURRENT_CHECK_STATE == FinderPatternFinder::NORMAL)
    {
        stateCountTotal = stateCountTotal + stateCount[0] + stateCount[4];
    }
    else if (CURRENT_CHECK_STATE == FinderPatternFinder::LEFT_SPILL)
    {
        stateCountTotal = stateCountTotal + stateCount[1] + stateCount[4];
    }
    else if (CURRENT_CHECK_STATE == FinderPatternFinder::RIHGT_SPILL)
    {
        stateCountTotal = stateCountTotal + stateCount[0] + stateCount[3];
    }
    
    if (5 * abs(stateCountTotal - originalStateCountTotal) >= originalStateCountTotal)
    {
        return nan();
    }
    
    estimatedHorizontalModuleSize = static_cast<float>(stateCountTotal) / 7.0f;
    return centerFromEnd(stateCount, j);
}


float FinderPatternFinder::hasHorizontalCheckedResult(size_t startJ, size_t centerI)
{
    for (size_t i = 0; i < _horizontalCheckedResult[startJ].size(); i++)
    {
        if (_horizontalCheckedResult[startJ][i].centerI == centerI)
        {
            return _horizontalCheckedResult[startJ][i].centerJ;
        }
    }
    
    return -1.0;
}


int FinderPatternFinder::addHorizontalCheckedResult(size_t startJ, size_t centerI, float centerJ)
{
    HorizontalCheckedResult result;
    result.centerI = centerI;
    result.centerJ = centerJ;
    
    _horizontalCheckedResult[startJ].push_back(result);
    
    return 1;
}

#define CENTER_CHECK_TIME 3

// This is called when a horizontal scan finds a possible alignment pattern. 
// It will cross check with a vertical scan, and if successful, will,  cross-cross-check with 
// another horizontal scan. This is needed primarily to locate the real horizontal center
// of the pattern in cases of extreme skew.
// If that succeeds the finder pattern location is added to a list that tracks the number
// of times each location has been nearly-mathced as a finder pattern. Each additional find
// is more eviednce that the location is in fact a finder pattern center.
// stateCount: reading state module counts from horizontal scan
// i: row where finder pattern may be found
// j: end of possible finder pattern in row

/**
 * <p>This is called when a horizontal scan finds a possible alignment pattern. It will
 * cross check with a vertical scan, and if successful, will, ah, cross-cross-check
 * with another horizontal scan. This is needed primarily to locate the real horizontal
 * center of the pattern in cases of extreme skew.
 * And then we cross-cross-cross check with another diagonal scan.</p>
 *
 * <p>If that succeeds the finder pattern location is added to a list that tracks
 * the number of times each location has been nearly-matched as a finder pattern.
 * Each additional find is more evidence that the location is in fact a finder
 * pattern center
 *
 * @param stateCount reading state module counts from horizontal scan
 * @param i row where finder pattern may be found
 * @param j end of possible finder pattern in row
 * @return true if a finder pattern candidate was found this time
 */
bool FinderPatternFinder::handlePossibleCenter(int* stateCount, size_t i, size_t j)
{
    CrossCheckState tmpHorizontalState = CURRENT_CHECK_STATE;
    float centerJ = centerFromEnd(stateCount, j);
    int stateCountTotal = stateCount[1] + stateCount[2] + stateCount[3];
    if (tmpHorizontalState == FinderPatternFinder::NORMAL)
    {
        stateCountTotal = stateCountTotal + stateCount[0] + stateCount[4];
    }
    else if (tmpHorizontalState == FinderPatternFinder::LEFT_SPILL)
    {
        stateCountTotal = stateCountTotal + stateCount[1] + stateCount[4];
    }
    else if (tmpHorizontalState == FinderPatternFinder::RIHGT_SPILL)
    {
        stateCountTotal = stateCountTotal + stateCount[0] + stateCount[3];
    }
    float estimatedHorizontalModuleSize = static_cast<float>(stateCountTotal) / 7.0f;
    
    float estimatedVerticalModuleSize;
    float tolerateModuleSize = estimatedHorizontalModuleSize > 4.0 ? estimatedHorizontalModuleSize / 2.0f : 1.0f;
    float possbileCenterJs[7] =
    {
        centerJ,
        centerJ - tolerateModuleSize,
        centerJ + tolerateModuleSize,
        centerJ - 2 * tolerateModuleSize,
        centerJ + 2 * tolerateModuleSize,
        centerJ - 3 * tolerateModuleSize,
        centerJ + 3 * tolerateModuleSize
    };
    
    int image_height = image_->getHeight();
    int image_width = image_->getWidth();
    for (int k = 0; k < CENTER_CHECK_TIME; k++)
    {
        float possibleCenterJ = possbileCenterJs[k];
        if (possibleCenterJ < 0 || possibleCenterJ >= image_width)
            continue;
        
        float centerI = crossCheckVertical(i,(size_t)possibleCenterJ, stateCount[2],
                                           stateCountTotal, estimatedVerticalModuleSize);
        
        if (!isnan(centerI) && centerI >= 0.0)
        {
            CrossCheckState tmpVerticalState = CURRENT_CHECK_STATE;
            
            float moduleSizeDiff = abs(estimatedHorizontalModuleSize - estimatedVerticalModuleSize);
            
            if (moduleSizeDiff > estimatedHorizontalModuleSize ||
               moduleSizeDiff > estimatedVerticalModuleSize)
                return false;
            
            tolerateModuleSize = estimatedVerticalModuleSize > 4.0 ? estimatedVerticalModuleSize / 2.0f : 1.0f;
            
            float possbileCenterIs[7] =
            {
                centerI,
                centerI - tolerateModuleSize,
                centerI + tolerateModuleSize,
                centerI - 2 * tolerateModuleSize,
                centerI + 2 * tolerateModuleSize,
                centerI - 3 * tolerateModuleSize,
                centerI + 3 * tolerateModuleSize
                
            };
            
            for (int l = 0; l < CENTER_CHECK_TIME; l++)
            {
                float possibleCenterI = possbileCenterIs[l];
                if (possibleCenterI < 0 || possibleCenterI >= image_height)
                    continue;
                
                // Re-cross check
                float reEstimatedHorizontalModuleSize;
                
                float cJ = hasHorizontalCheckedResult(centerJ, possibleCenterI);
                
                if (!isnan(cJ) && cJ >= 0.0)
                {
                    centerJ = cJ;
                }
                else
                {
                    cJ = centerJ;
                    
                    float ccj = crossCheckHorizontal((size_t)cJ, (size_t)possibleCenterI, stateCount[2],
                                                     stateCountTotal, reEstimatedHorizontalModuleSize);
                    
                    if (!isnan(ccj))
                    {
                        centerJ = ccj;
                        addHorizontalCheckedResult(cJ, possibleCenterI, ccj);
                    }
                }
                
                if ( !isnan(centerJ))
                {
                    tryToPushToCenters(centerI, centerJ,
                                       (estimatedHorizontalModuleSize + estimatedVerticalModuleSize) / 2.0,
                                       tmpHorizontalState, tmpVerticalState);
                    return true;
                }
            }
        }
    }
    
    return false;
}

// return the number of rows we could safely skip during scanning, based on the first two 
// finder patterns that have been located. In some cases their position will allow us to 
// infer that the third pattern must lie below a certain point farther down the image. 
int FinderPatternFinder::findRowSkip()
{
    int max = possibleCenters_.size();
    if (max <= 1)
    {
        return 0;
    }
    
    if (max <= compared_finder_counts)
        return 0;
    
    Ref<FinderPattern> firstConfirmedCenter, secondConfirmedCenter;
    
    for (int i = 0; i < max - 1; i++)
    {
        firstConfirmedCenter = possibleCenters_[i];
        if (firstConfirmedCenter->getCount() >= CENTER_QUORUM)
        {
            float firstModuleSize = firstConfirmedCenter->getEstimatedModuleSize();
            int j_start = (i < compared_finder_counts)? compared_finder_counts : i + 1;
            for (int j = j_start; j < max; j++)
            {
                secondConfirmedCenter = possibleCenters_[j];
                if (secondConfirmedCenter->getCount() >= CENTER_QUORUM)
                {
                    float secondModuleSize = secondConfirmedCenter->getEstimatedModuleSize();
                    float moduleSizeDiff = abs(firstModuleSize-secondModuleSize);
                    if (moduleSizeDiff < 1.0f)
                    {
                        hasSkipped_ = true;
                        return static_cast<int>(abs(firstConfirmedCenter->getX() - secondConfirmedCenter->getX())
                                     - abs(firstConfirmedCenter->getY()- secondConfirmedCenter->getY())) /2;
                    }
                }
            }
        }
    }
    
    compared_finder_counts = max;
    
    return 0;
}

// return the 3 finder patterns from our list of candidates. The "best" are those that have 
// been detected at least #CENTER_QUORUM times, and whose module size differs from 
// the average among those patterns the least.
vector< Ref<FinderPattern> > FinderPatternFinder::selectBestPatterns(ErrorHandler & err_handler) 
{
    size_t startSize = possibleCenters_.size();
    
    if (startSize < 3)
    {
        // Couldn't find enough finder patterns
        err_handler = ReaderErrorHandler("Could not find three finder patterns");
        return vector< Ref<FinderPattern> >();
    }
    
    vector<Ref<FinderPattern> > result(3);
    
    if (startSize == 3)
    {
        result[0] = possibleCenters_[0];
        result[1] = possibleCenters_[1];
        result[2] = possibleCenters_[2];
        
        return result;
    }
    
    sort(possibleCenters_.begin(), possibleCenters_.end(), CountComparator());
    if ((possibleCenters_[2]->getCount() - possibleCenters_[3]->getCount()) > 1 && possibleCenters_[2]->getCount() > 1)
    {
        result[0] = possibleCenters_[0];
        result[1] = possibleCenters_[1];
        result[2] = possibleCenters_[2];
        
        return result;
        
    }
    else if (possibleCenters_[3]->getCount() > 1)
    {
        float totalModuleSize = 0.0f;
        for (int i = 0; i < 4; i++)
        {
            totalModuleSize += possibleCenters_[i]->getEstimatedModuleSize();
        }
        
        float everageModuleSize = totalModuleSize / 4.0f;
        float maxDiffModuleSize = 0.0f;
        int maxID = 0;
        for (int i = 0; i < 4; i++)
        {
            float diff = abs(possibleCenters_[i]->getEstimatedModuleSize() - everageModuleSize);
            if (diff > maxDiffModuleSize)
            {
                maxDiffModuleSize = diff;
                maxID = i;
            }
        }
        
        switch (maxID)
        {
            case 0:
                result[0] = possibleCenters_[1];
                result[1] = possibleCenters_[2];
                result[2] = possibleCenters_[3];
                break;
            case 1:
                result[0] = possibleCenters_[0];
                result[1] = possibleCenters_[2];
                result[2] = possibleCenters_[3];
                break;
            case 2:
                result[0] = possibleCenters_[0];
                result[1] = possibleCenters_[1];
                result[2] = possibleCenters_[3];
                break;
            default:
                result[0] = possibleCenters_[0];
                result[1] = possibleCenters_[1];
                result[2] = possibleCenters_[2];
                break;
        }
        
        return result;
    }
    else if (possibleCenters_[1]->getCount() > 1 && possibleCenters_[2]->getCount() == 1)
    {
        vector<Ref<FinderPattern> > possibleThirdCenter;
        float possibleModuleSize = (possibleCenters_[0]->getEstimatedModuleSize()
                                    + possibleCenters_[1]->getEstimatedModuleSize()) / 2.0f;
        for (size_t i = 2; i < startSize; i++)
        {
            if (abs(possibleCenters_[i]->getEstimatedModuleSize() - possibleModuleSize) < 0.5 * possibleModuleSize)
                possibleThirdCenter.push_back(possibleCenters_[i]);
        }
        
        float longestSide = 0.0f;
        size_t longestId = 0;
        for (size_t i = 0; i < possibleThirdCenter.size(); i++)
        {
            float tmpLongSide = 0.0f;
            if (checkIsoscelesRightTriangle(possibleCenters_[0], possibleCenters_[1], possibleThirdCenter[i],
                                           tmpLongSide))
            {
                if (tmpLongSide >= longestSide)
                {
                    longestSide = tmpLongSide;
                    longestId = i;
                }
            }
        }
        result[0] = possibleCenters_[0];
        result[1] = possibleCenters_[1];
        
        // Error with decoding : valiantliu -- 20140304
        if (longestId >= possibleThirdCenter.size())
        {
            err_handler = ReaderErrorHandler("Not find any available possibleThirdCenter");
            return vector< Ref<FinderPattern> >();
        }
        else
        {
            result[2] = possibleThirdCenter[longestId];
        }
        
        return result;
    }
    
    // Filter outlier possibilities whose module size is too different
    if (startSize > 3)
    {
        // But we can only afford to do so if we have at least 4 possibilities to choose from
        float totalModuleSize = 0.0f;
        float square = 0.0f;
        for (size_t i = 0; i < startSize; i++)
        {
            float size = possibleCenters_[i]->getEstimatedModuleSize();
            totalModuleSize += size;
            square += size * size;
        }
        float average = totalModuleSize / static_cast<float>(startSize);
        float stdDev = static_cast<float>(sqrt(square / startSize - average * average));
        
        sort(possibleCenters_.begin(), possibleCenters_.end(), FurthestFromAverageComparator(average));
        
        // float limit = max(0.2f * average, stdDev);
        float limit = max(0.5f * average, stdDev);
        
        for (size_t i = 0; i < possibleCenters_.size() && possibleCenters_.size() > 3; i++)
        {
            if (abs(possibleCenters_[i]->getEstimatedModuleSize() - average) > limit)
            {
                possibleCenters_.erase(possibleCenters_.begin()+i);
                i--;
            }
        }
    }
    
    size_t tryHardPossibleCenterSize = 15;
    size_t possibleCenterSize = 12;
    
    // add by sofiawu
    float totalModuleSize = 0.0f;
    for (size_t i = 0; i < possibleCenters_.size(); i++)
    {
        float size = possibleCenters_[i]->getEstimatedModuleSize();
        totalModuleSize += size;
    }
    float average = totalModuleSize / static_cast<float>(possibleCenters_.size());
    sort(possibleCenters_.begin(), possibleCenters_.end(), CenterComparator(average));
    // end add
    
    if (possibleCenters_.size() > tryHardPossibleCenterSize && getFinderLoose())
    {
        possibleCenters_.erase(possibleCenters_.begin() + tryHardPossibleCenterSize, possibleCenters_.end());
    }
    else if (possibleCenters_.size() > possibleCenterSize && getFinderLoose())
    {
        possibleCenters_.erase(possibleCenters_.begin() + possibleCenterSize, possibleCenters_.end());
    }
    
    if (possibleCenters_.size() >= 6 && getFinderLoose())
    {
        possibleCenters_.erase(possibleCenters_.begin() + 6, possibleCenters_.end());
    }
    else if (possibleCenters_.size() > 3)
    {
        // Throw away all but those first size candidate points we found.
        possibleCenters_.erase(possibleCenters_.begin() + 3, possibleCenters_.end());
    }
    
    result[0] = possibleCenters_[0];
    result[1] = possibleCenters_[1];
    result[2] = possibleCenters_[2];
    
    return result;
}


vector< Ref<FinderPattern> > FinderPatternFinder::selectFileBestPatterns(ErrorHandler & err_handler)
{
    size_t startSize = possibleCenters_.size();
    
    if (startSize < 3)
    {
        // Couldn't find enough finder patterns
        err_handler = ReaderErrorHandler("Could not find three finder patterns");
        return vector< Ref<FinderPattern> >();
    }
    
    vector<Ref<FinderPattern> > result(3);
    
    if (startSize == 3)
    {
        result[0] = possibleCenters_[0];
        result[1] = possibleCenters_[1];
        result[2] = possibleCenters_[2];
        return result;
    }
    
    sort(possibleCenters_.begin(), possibleCenters_.end(), BestComparator());
    
    result[0] = possibleCenters_[0];
    result[1] = possibleCenters_[1];
    result[2] = possibleCenters_[2];
    
    for (size_t i = 0; i < possibleCenters_.size() - 2; ++i)
    {
        float tmpLongSide = 0;
        
        int iCountDiff = 0;
        float fModuleSizeDiff = 0;
        for (size_t j = 0; j < 3; ++j)
        {
            iCountDiff += abs(possibleCenters_[i + j]->getCount() - possibleCenters_[i + ((j + 1) % 3)]->getCount());
            if (iCountDiff > 2) break;
            fModuleSizeDiff += fabs(possibleCenters_[i + j]->getEstimatedModuleSize() - possibleCenters_[i + ((j + 1) % 3)]->getEstimatedModuleSize());
            if (fModuleSizeDiff > 5) break;
        }
        
        if (iCountDiff > 2) continue;
        if (fModuleSizeDiff > 5) continue;
        
        if (checkIsoscelesRightTriangle(possibleCenters_[i], possibleCenters_[i + 1], possibleCenters_[i + 2],
                                        tmpLongSide))
        {
            result[0] = possibleCenters_[i];
            result[1] = possibleCenters_[i + 1];
            result[2] = possibleCenters_[i + 2];
            
            break;
        }
    }
    
    return result;
}

// Orders an array of three patterns in an order [A, B, C] such that
// AB<AC and BC<AC and the angle between BC and BA is less than 180 degrees.
vector<Ref<FinderPattern> > FinderPatternFinder::orderBestPatterns(vector<Ref<FinderPattern> > patterns)
{
    // Find distances between pattern centers
    float abDistance = distance(patterns[0], patterns[1]);
    float bcDistance = distance(patterns[1], patterns[2]);
    float acDistance = distance(patterns[0], patterns[2]);
    
    Ref<FinderPattern> topLeft;
    Ref<FinderPattern> topRight;
    Ref<FinderPattern> bottomLeft;
    
    // Assume one closest to other two is top left;
    // topRight and bottomLeft will just be guesses below at first
    if (bcDistance >= abDistance && bcDistance >= acDistance)
    {
        topLeft = patterns[0];
        topRight = patterns[1];
        bottomLeft = patterns[2];
    }
    else if (acDistance >= bcDistance && acDistance >= abDistance)
    {
        topLeft = patterns[1];
        topRight = patterns[0];
        bottomLeft = patterns[2];
    }
    else
    {
        topLeft = patterns[2];
        topRight = patterns[0];
        bottomLeft = patterns[1];
    }
    
    // Use cross product to figure out which of other1/2 is the bottom left
    // pattern. The vector "top_left -> bottom-_eft" x "top_left -> top_right"
    // should yield a vector with positive z component
    if ((bottomLeft->getY() - topLeft->getY()) * (topRight->getX() - topLeft->getX()) < (bottomLeft->getX()
                                                                                         - topLeft->getX()) * (topRight->getY() - topLeft->getY()))
    {
        Ref<FinderPattern> temp = topRight;
        topRight = bottomLeft;
        bottomLeft = temp;
    }
    
    vector<Ref<FinderPattern> > results(3);
    results[0] = bottomLeft;
    results[1] = topLeft;
    results[2] = topRight;
    
    return results;
}

// add by chicodai
bool FinderPatternFinder::tryToPushToCenters(float centerI, float centerJ,
                                             float estimatedModuleSize,
                                             CrossCheckState horizontalState,
                                             CrossCheckState verticalState)
{
    for (size_t index = 0; index < possibleCenters_.size(); index++)
    {
        Ref<FinderPattern> center = possibleCenters_[index];
        // Look for about the same center and module size:
        if (center->aboutEquals(estimatedModuleSize, centerI, centerJ))
        {
            possibleCenters_[index] =
            center->combineEstimate(centerI, centerJ, estimatedModuleSize);
            possibleCenters_[index]->setHorizontalCheckState(
                                                             horizontalState == FinderPatternFinder::NORMAL ? center->getHorizontalCheckState()
                                                             : horizontalState);
            possibleCenters_[index]->setVerticalCheckState(
                                                           verticalState == FinderPatternFinder::NORMAL ? center->getVerticalCheckState()
                                                           : verticalState);
            return false;
        }
    }
    
    Ref<FinderPattern> newPattern(new FinderPattern(centerJ, centerI, estimatedModuleSize));
    newPattern->setHorizontalCheckState(horizontalState);
    newPattern->setVerticalCheckState(verticalState);
    possibleCenters_.push_back(newPattern);
    return true;
}

bool FinderPatternFinder::checkIsoscelesRightTriangle(Ref<FinderPattern> centerA,
                                                      Ref<FinderPattern> centerB,
                                                      Ref<FinderPattern> centerC, float& longSide)
{
    float shortSide1, shortSide2;
    FinderPatternInfo::calculateSides(centerA, centerB, centerC, longSide, shortSide1, shortSide2);
    
    auto minAmongThree = [](float a, float b, float c) { return (std::min)((std::min)(a, b), c); };
    auto maxAmongThree = [](float a, float b, float c) { return (std::max)((std::max)(a, b), c); };
    
    float shortSideSqrt1 = sqrt(shortSide1);
    float shortSideSqrt2 = sqrt(shortSide2);
    float longSideSqrt = sqrt(longSide);
    auto minSide = minAmongThree(shortSideSqrt1, shortSideSqrt2, longSideSqrt);
    auto maxModuleSize =
    maxAmongThree(centerA->getEstimatedModuleSize(), centerB->getEstimatedModuleSize(),
                  centerC->getEstimatedModuleSize());
    
    // if edge length smaller than 21 * module size add by chicodai
    if (minSide <= maxModuleSize * 21) return false;
    
    float CosLong = (shortSide1 + shortSide2 - longSide) / (2 * shortSideSqrt1 * shortSideSqrt2);
    float CosShort1 = (longSide + shortSide1 - shortSide2) / (2 * longSideSqrt * shortSideSqrt1);
    float CosShort2 = (longSide + shortSide2 - shortSide1) / (2 * longSideSqrt * shortSideSqrt2);
    
    if (abs(CosLong) > FP_RIGHT_ANGLE ||
        (CosShort1 < FP_SMALL_ANGLE2 || CosShort1 > FP_SMALL_ANGLE1) ||
        (CosShort2 < FP_SMALL_ANGLE2 || CosShort2 > FP_SMALL_ANGLE1))
    {
        return false;
    }
    
    return true;
}

// return distance between two points
float FinderPatternFinder::distance(Ref<ResultPoint> p1, Ref<ResultPoint> p2)
{
    float dx = p1->getX() - p2->getX();
    float dy = p1->getY() - p2->getY();
    return static_cast<float>(sqrt(dx * dx + dy * dy));
}

FinderPatternFinder::FinderPatternFinder(Ref<BitMatrix> image,
                                         Ref<UnicomBlock> block,
                                         Ref<ResultPointCallback>const& callback) : image_(image), possibleCenters_(), hasSkipped_(false), block_(block) , callback_(callback),
compared_finder_counts(0) {
    CURRENT_CHECK_STATE = FinderPatternFinder::NORMAL;
    initConfig();
}

Ref<BitMatrix> FinderPatternFinder::getImage() {
    return image_;
}

vector<Ref<FinderPattern> >& FinderPatternFinder::getPossibleCenters() {
    return possibleCenters_;
}
