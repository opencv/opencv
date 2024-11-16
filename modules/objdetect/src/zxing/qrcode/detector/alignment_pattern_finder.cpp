// -*- mode:c++; tab-width:2; indent-tabs-mode:nil; c-basic-offset:2 -*-
/*
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

#include "alignment_pattern_finder.hpp"
#include "../../common/bit_array.hpp"
#include <vector>
#include <cmath>
#include <cstdlib>
#include <map>

using std::abs;
using std::vector;
using std::map;
using zxing::Ref;
using zxing::qrcode::AlignmentPatternFinder;
using zxing::qrcode::AlignmentPattern;
using zxing::qrcode::FinderPattern;
using zxing::ErrorHandler;
using zxing::ReaderErrorHandler;

// VC++
// This class attempts to find alignment patterns in a QR Code. Alignment patterns look like finder
// patterns but are smaller and appear at regular intervals throughout the image.
// At the moment this only looks for the bottom-right alignment pattern.
// This is mostly a simplified copy of {@link FinderPatternFinder}. It is copied,
// pasted and stripped down here for maximum performance but does unfortunately duplicat some code.
// This class is thread-safe but not reentrant. Each thread must allocate its own object.
using zxing::BitMatrix;
using zxing::ResultPointCallback;

// Creates a finder that will look in a portion of the whole image.
AlignmentPatternFinder::AlignmentPatternFinder(Ref<BitMatrix> image, int startX, int startY, int width,
                                               int height, float moduleSize,
                                               Ref<ResultPointCallback>const& callback) :
image_(image), possibleCenters_(new vector<AlignmentPattern *> ()), startX_(startX), startY_(startY),
width_(width), height_(height), moduleSize_(moduleSize), callback_(callback)
#ifdef FIND_WHITE_ALIGNMENTPATTERN
, foundAlignmentPattern(false), foundWhiteAlignmentPattern(false),
possibleWhiteCenters_(new vector<AlignmentPattern *> ()), lastTry(false)
#endif
{
}

AlignmentPatternFinder::AlignmentPatternFinder(Ref<BitMatrix> image, 
                                               float moduleSize, Ref<ResultPointCallback>const& callback):
image_(image), moduleSize_(moduleSize), callback_(callback)
#ifdef FIND_WHITE_ALIGNMENTPATTERN
, foundAlignmentPattern(false), foundWhiteAlignmentPattern(false), lastTry(false)
#endif
{
    
}

// This method attempts to find the bottom-right alignment pattern in the image. It is a bit messy since
// it's pretty performance-critical and so is written to be fast foremost.
#ifdef FIND_WHITE_ALIGNMENTPATTERN
std::vector<Ref<AlignmentPattern> > AlignmentPatternFinder::find(ErrorHandler & err_handler) {
    int maxJ = startX_ + width_;
    int middleI = startY_ + (height_ >> 1);
    // We are looking for black/white/black modules in 1:1:1 ratio;
    // this tracks the number of black/white/black modules seen so far
    vector<int> stateCount(3, 0);
    for (int iGen = 0; iGen < height_; iGen++) {
        // Search from middle outwards
        int i = middleI + ((iGen & 0x01) == 0 ? ((iGen + 1) >> 1) : -((iGen + 1) >> 1));
        stateCount[0] = 0;
        stateCount[1] = 0;
        stateCount[2] = 0;
        int j = startX_;
        // Burn off leading white pixels before anything else; if we start in the middle of a white run,
        // it doesn't make sense to count its length, since we don't know if the white run continued to the left of the start point
        while (j < maxJ && !image_->get(j, i)) {
            j++;
        }
        int currentState = 0;
        while (j < maxJ) {
            if (image_->get(j, i))
            {
                // Black pixel
                if (currentState == 1)
                {  // Counting black pixels
                    stateCount[currentState]++;
                }
                else
                {  // Counting white pixels
                    if (currentState == 2)
                    {  // A winner?
                        foundPatternCross(stateCount);
                        if (foundAlignmentPattern || foundWhiteAlignmentPattern)
                        {
                            handlePossibleCenter(stateCount, i, j);
                            vector<Ref<AlignmentPattern> > results;
                            if (confirmedAlignmentPattern != 0 && confirmedWhiteAlignmentPattern != 0)
                            {
                                results.push_back(confirmedAlignmentPattern);
                                results.push_back(confirmedWhiteAlignmentPattern);
                                return results;
                            }
                            else if (confirmedAlignmentPattern != 0)
                            {
                                results.push_back(confirmedAlignmentPattern);
                                return results;
                            }
                            else if (lastTry && confirmedWhiteAlignmentPattern != 0)
                            {
                                results.push_back(confirmedWhiteAlignmentPattern);
                                return results;
                            }
                        }
                        stateCount[0] = stateCount[2];
                        stateCount[1] = 1;
                        stateCount[2] = 0;
                        currentState = 1;
                    }
                    else
                    {
                        stateCount[++currentState]++;
                    }
                }
            }
            else
            {  // White pixel
                if (currentState == 1)
                {  // Counting black pixels
                    currentState++;
                }
                stateCount[currentState]++;
            }
            j++;
        }
        
        foundPatternCross(stateCount);
        if (foundAlignmentPattern || foundWhiteAlignmentPattern)
        {
            vector<Ref<AlignmentPattern> > results;
            handlePossibleCenter(stateCount, i, j);
            if (confirmedAlignmentPattern != 0 && confirmedWhiteAlignmentPattern != 0)
            {
                results.push_back(confirmedAlignmentPattern);
                results.push_back(confirmedWhiteAlignmentPattern);
                return results;
            }
            else if (confirmedAlignmentPattern != 0)
            {
                results.push_back(confirmedAlignmentPattern);
                return results;
            }
            else if (lastTry && confirmedWhiteAlignmentPattern != 0)
            {
                results.push_back(confirmedWhiteAlignmentPattern);
                return results;
            }
        }
    }
    // Nothing we saw was observed and confirmed twice. If we had any guess at all, return it.
    vector<Ref<AlignmentPattern> > results;
    if (possibleCenters_->size() > 0 && possibleWhiteCenters_->size() > 0)
    {
        Ref<AlignmentPattern> center((*possibleCenters_)[0]);
        Ref<AlignmentPattern> white_center((*possibleWhiteCenters_)[0]);
        results.push_back(center);
        results.push_back(white_center);
        return results;
    }
    else if (possibleCenters_->size() > 0)
    {
        Ref<AlignmentPattern> center((*possibleCenters_)[0]);
        results.push_back(center);
        return results;
    }
    else if (lastTry && possibleWhiteCenters_->size() > 0)
    {
        Ref<AlignmentPattern> white_center((*possibleWhiteCenters_)[0]);
        results.push_back(white_center);
        return results;
    }
    err_handler = ReaderErrorHandler("Could not find alignment pattern");
    return std::vector<Ref<AlignmentPattern> >();
}

#else

Ref<AlignmentPattern> AlignmentPatternFinder::find(ErrorHandler &err_handler) {
    int maxJ = startX_ + width_;
    int middleI = startY_ + (height_ >> 1);
    // We are looking for black/white/black modules in 1:1:1 ratio;
    // this tracks the number of black/white/black modules seen so far
    vector<int> stateCount(3, 0);
    for (int iGen = 0; iGen < height_; iGen++) {
        // Search from middle outwards
        int i = middleI + ((iGen & 0x01) == 0 ? ((iGen + 1) >> 1) : -((iGen + 1) >> 1));
        stateCount[0] = 0;
        stateCount[1] = 0;
        stateCount[2] = 0;
        int j = startX_;
        // Burn off leading white pixels before anything else; if we start in the middle of a white run,
        // it doesn't make sense to count its length, since we don't know if the white run continued to the left of the start point
        while (j < maxJ && !image_->get(j, i)) {
            j++;
        }
        int currentState = 0;
        while (j < maxJ) {
            if (image_->get(j, i))
            {
                // Black pixel
                if (currentState == 1)
                {  // Counting black pixels
                    stateCount[currentState]++;
                }
                else
                {  // Counting white pixels
                    if (currentState == 2)
                    {  // A winner?
                        if (foundPatternCross(stateCount))
                        {  // Yes
                            Ref<AlignmentPattern> confirmed(handlePossibleCenter(stateCount, i, j));
                            if (confirmed != 0)
                            {
                                return confirmed;
                            }
                        }
                        stateCount[0] = stateCount[2];
                        stateCount[1] = 1;
                        stateCount[2] = 0;
                        currentState = 1;
                    }
                    else
                    {
                        stateCount[++currentState]++;
                    }
                }
            }
            else
            {  // White pixel
                if (currentState == 1)
                {  // Counting black pixels
                    currentState++;
                }
                stateCount[currentState]++;
            }
            j++;
        }
        if (foundPatternCross(stateCount))
        {
            Ref<AlignmentPattern> confirmed(handlePossibleCenter(stateCount, i, maxJ));
            if (confirmed != 0)
            {
                return confirmed;
            }
        }
    }
    // Nothing we saw was observed and confirmed twice. If we had any guess at all, return it.
    if (possibleCenters_->size() > 0)
    {
        Ref<AlignmentPattern> center((*possibleCenters_)[0]);
        return center;
    }
    err_handler = ReaderErrorHandler("Could not find alignment pattern");
    return Ref<AlignmentPattern>();
}
#endif

// Given a count of black/white/black pixels just seen and an end position,
// figures the location of the center of this black/white/black run.
float AlignmentPatternFinder::centerFromEnd(vector<int>& stateCount, int end) {
    return static_cast<float>(end - stateCount[2]) - stateCount[1] / 2.0f;
}

#ifdef FIND_WHITE_ALIGNMENTPATTERN
float AlignmentPatternFinder::centerFromEndWhite(vector<int>& stateCount, int end) {
    return static_cast<float>(end - stateCount[2]-stateCount[1]) - stateCount[0] / 2.0f;
}
#endif

#ifdef FIND_WHITE_ALIGNMENTPATTERN
void AlignmentPatternFinder::foundPatternCross(vector<int> &stateCount) {
    foundAlignmentPattern = true;
    float maxVariance = moduleSize_ / 2.0f;
    for (int i = 0; i < 3; i++) {
        if (abs(moduleSize_ - stateCount[i]) >= maxVariance)
        {
            foundAlignmentPattern = false;
            break;
        }
    }
    foundWhiteAlignmentPattern = false;
    if (abs(3.0*moduleSize_ - stateCount[0])<=moduleSize_)
        foundWhiteAlignmentPattern =  true;
    
    return;
}
#else
bool AlignmentPatternFinder::foundPatternCross(vector<int> &stateCount) {
    float maxVariance = moduleSize_ / 2.0f;
    for (int i = 0; i < 3; i++) {
        if (abs(moduleSize_ - stateCount[i]) >= maxVariance)
        {
            return false;
        }
    }
    return true;
}
#endif

// After a horizontal scan finds a potential alignment pattern, this method "cross-checks" by scanning 
// down vertically through the center of the possible alignment pattern to see if the same proportion is detected.
// return vertical center of alignment pattern, or nan() if not found
// startI: row where an alignment pattern was detected
// centerJ: center of the section that appears to cross an alignment pattern
// maxCount: maximum reasonable number of modules that should be observed in any reading state,
// based on the results of the horizontal scan
float AlignmentPatternFinder::crossCheckVertical(int startI, int centerJ, int maxCount, int originalStateCountTotal) {
    // This is slightly faster than using the Ref. Efficiency is important here
    BitMatrix& matrix = *image_;
    
    int maxI = matrix.getHeight();
    vector<int> stateCount(3, 0);
    // Start counting up from center
    int i = startI;
    while (i >= 0 && matrix.get(centerJ, i) && stateCount[1] <= maxCount) {
        stateCount[1]++;
        i--;
    }
    // If already too many modules in this state or ran off the edge:
    if (i < 0 || stateCount[1] > maxCount)
    {
        return nan();
    }
    while (i >= 0 && !matrix.get(centerJ, i) && stateCount[0] <= maxCount) {
        stateCount[0]++;
        i--;
    }
    if (stateCount[0] > maxCount)
    {
        return nan();
    }
    
    // Now also count down from center
    i = startI + 1;
    while (i < maxI && matrix.get(centerJ, i) && stateCount[1] <= maxCount) {
        stateCount[1]++;
        i++;
    }
    if (i == maxI || stateCount[1] > maxCount)
    {
        return nan();
    }
    while (i < maxI && !matrix.get(centerJ, i) && stateCount[2] <= maxCount) {
        stateCount[2]++;
        i++;
    }
    if (stateCount[2] > maxCount) {
        return nan();
    }
    
    int stateCountTotal = stateCount[0] + stateCount[1] + stateCount[2];
    if (5 * abs(stateCountTotal - originalStateCountTotal) >= 2 * originalStateCountTotal)
    {
        return nan();
    }
#ifdef FIND_WHITE_ALIGNMENTPATTERN
    foundPatternCross(stateCount);
    return foundAlignmentPattern ? centerFromEnd(stateCount, i):nan();
#else 
    return foundPatternCross(stateCount) ? centerFromEnd(stateCount, i) : nan();
#endif
}

#ifdef FIND_WHITE_ALIGNMENTPATTERN
float AlignmentPatternFinder::crossCheckVerticalWhite(int startI, int centerJ, int maxCount, int originalStateCountTotal) {
    // This is slightly faster than using the Ref. Efficiency is important here
    BitMatrix& matrix = *image_;
    int maxI = matrix.getHeight();
    vector<int> stateCount(3, 0);
    // Start counting up from center
    int i = startI;
    while (i >= 0 && !matrix.get(centerJ, i) && stateCount[0] <= maxCount) {
        stateCount[0]++;
        i--;
    }
    if (i < 0 || stateCount[0]>maxCount)
    {
        return nan();
    }
    // Now also count down from center
    i = startI + 1;
    while (i < maxI && !matrix.get(centerJ, i) && stateCount[0] <= maxCount) {
        stateCount[0]++;
        i++;
    }
    if (i == maxI || stateCount[0] > maxCount)
    {
        return nan();
    }
    
}
#endif

// This is called when a horizontal scan finds a possible alignment pattern. It will cross check 
// with a vertical scan, and if successful, will see if this pattern had been found on a previous horizontal scan. 
// If so, we consider it confirmed and conclude we have found the alignment pattern.
// return {@link AlignmentPattern} if we have found the same pattern twice, or null if not
// i: row where alignment pattern may be found
// j: end of possible alignment pattern in row
#ifdef FIND_WHITE_ALIGNMENTPATTERN
void AlignmentPatternFinder::handlePossibleCenter(vector<int> &stateCount, int i, int j) {
    float centerJ, centerI, estimatedModuleSize;
    if (foundAlignmentPattern){
        int stateCountTotal = stateCount[0] + stateCount[1] + stateCount[2];
        centerJ = centerFromEnd(stateCount, j);
        centerI = crossCheckVertical(i, static_cast<int>(centerJ), 2 * stateCount[1], stateCountTotal);
        if (!isnan(centerI))
        {
            estimatedModuleSize = static_cast<float>(stateCount[0] + stateCount[1] + stateCount[2]) / 3.0f;
            int max = possibleCenters_->size();
            for (int index = 0; index < max; index++) {
                Ref<AlignmentPattern> center((*possibleCenters_)[index]);
                // Look for about the same center and module size:
                if (center->aboutEquals(estimatedModuleSize, centerI, centerJ))
                {
                    confirmedAlignmentPattern = center->combineEstimate(centerI, centerJ, estimatedModuleSize);
                }
            }
            // Hadn't found this before; save it
            AlignmentPattern *tmp = new AlignmentPattern(centerJ, centerI, estimatedModuleSize);
            tmp->retain();
            possibleCenters_->push_back(tmp);
            if (callback_ != 0) {
                callback_->foundPossibleResultPoint(*tmp);
            }
        }
    }
    if (foundWhiteAlignmentPattern){
        centerJ = centerFromEndWhite(stateCount, j);
        centerI = crossCheckVerticalWhite(i, static_cast<int>(centerJ), stateCount[0]*3, stateCount[0]*3);
        if (!isnan(centerI))
        {
            estimatedModuleSize = static_cast<float>(stateCount[0]) / 3.0f;
            int max = possibleWhiteCenters_->size();
            for (int index = 0; index < max; index++) {
                Ref<AlignmentPattern> center((*possibleWhiteCenters_)[index]);
                // Look for about the same center and module size:
                if (center->aboutEquals(estimatedModuleSize, centerI, centerJ))
                {
                    //  return center->combineEstimate(centerI, centerJ, estimatedModuleSize);
                    confirmedWhiteAlignmentPattern = center->combineEstimate(centerI, centerJ, estimatedModuleSize);
                }
            }
            AlignmentPattern *tmp = new AlignmentPattern(centerJ, centerI, estimatedModuleSize);
            tmp->retain();
            possibleWhiteCenters_->push_back(tmp);
            if (callback_ != 0)
            {
                callback_->foundPossibleResultPoint(*tmp);
            }
        }
    }
    return;
}
#else
Ref<AlignmentPattern> AlignmentPatternFinder::handlePossibleCenter(vector<int> &stateCount, int i, int j) {
    int stateCountTotal = stateCount[0] + stateCount[1] + stateCount[2];
    float centerJ = centerFromEnd(stateCount, j);
    float centerI = crossCheckVertical(i, static_cast<int>(centerJ), 2 * stateCount[1], stateCountTotal);
    if (!isnan(centerI))
    {
        float estimatedModuleSize = static_cast<float>(stateCount[0] + stateCount[1] + stateCount[2]) / 3.0f;
        int max = possibleCenters_->size();
        for (int index = 0; index < max; index++) {
            Ref<AlignmentPattern> center((*possibleCenters_)[index]);
            // Look for about the same center and module size:
            if (center->aboutEquals(estimatedModuleSize, centerI, centerJ))
            {
                return center->combineEstimate(centerI, centerJ, estimatedModuleSize);
            }
        }
        // Hadn't found this before; save it
        AlignmentPattern *tmp = new AlignmentPattern(centerJ, centerI, estimatedModuleSize);
        tmp->retain();
        possibleCenters_->push_back(tmp);
        if (callback_ != 0)
        {
            callback_->foundPossibleResultPoint(*tmp);
        }
    }
    Ref<AlignmentPattern> result;
    return result;
}
#endif


AlignmentPatternFinder::~AlignmentPatternFinder() {
    for (size_t i = 0; i < possibleCenters_->size(); i++) {
        (*possibleCenters_)[i]->release();
        (*possibleCenters_)[i] = 0;
    }
    delete possibleCenters_;
}

Ref<AlignmentPattern> AlignmentPatternFinder::findByPattern(Ref<AlignmentPattern>const &estimateCenter, Ref<FinderPattern>const &topLeft,
                                                            Ref<FinderPattern>const &topRight, Ref<FinderPattern>const &bottomLeft){
    (void)estimateCenter;
    (void)topLeft;
    (void)topRight;
    (void)bottomLeft;
    
    Ref<AlignmentPattern> result;
    return result;
}

#ifdef USING_WX

float AlignmentPatternFinder::centerFromEnd(int* stateCount, int end) {
    return static_cast<float>(end - stateCount[4]) - (stateCount[3] + stateCount[2] + stateCount[1]) / 2.0f;
}

bool AlignmentPatternFinder::foundPatternCross(int* stateCount) {
    float unsuitable = 1.5f;
    float maxVariance = 2.0f;
    float maxAntiOutOfFocusVariance1 = 4.0f;
    float maxAntiOutOfFocusVariance2 = 2.5f;
    int antiOutOfFocusCount = 0;
    float moduleSize = (stateCount[1] + stateCount[2] + stateCount[3]) / 3.0f;
    if (moduleSize / moduleSize_ > unsuitable || moduleSize / moduleSize_ < 1.0f / unsuitable)
        return false;
    for (int i = 0; i < 5; i++)
        if (stateCount[i] == 0)
            return false;
    for (int i = 1; i < 4; i++) {
        float variance = stateCount[i] / moduleSize;
        if (variance > maxVariance)
            return false;
        if (variance < 1.0f / maxVariance)
        {
            if (stateCount[i] == 1 && variance > 1.0f / maxAntiOutOfFocusVariance1)
                ++antiOutOfFocusCount;
            else if (stateCount[i] == 2 && variance < 1.0f / maxAntiOutOfFocusVariance2)
                ++antiOutOfFocusCount;
            else
                return false;
        }
        if (antiOutOfFocusCount > 2)
            return false;
    }
    return true;
}

void AlignmentPatternFinder::handlePossibleCenter(int *stateCount, size_t i, size_t j) {
    int verticalState[5];
    int horizontalState[5];
    int stateCountTotal = stateCount[1] + stateCount[2] + stateCount[3];
    float centerJ = centerFromEnd(stateCount, j);
    float centerI = crossCheckVertical(i, (size_t)centerJ, stateCountTotal, stateCountTotal, verticalState);
    if (!isnan(centerI))
    {
        // Re-cross check
        stateCountTotal = verticalState[1] + verticalState[2] + verticalState[3];
        centerJ = crossCheckHorizontal((size_t)centerJ, (size_t)centerI, stateCountTotal, stateCountTotal, horizontalState);
        if (!isnan(centerJ))
        {
            int max = possibleCenters_.size();
            float verticalModuleSize = AlignmentPattern::getEstimatedModuleSize(verticalState);
            float horizontalModuleSize = AlignmentPattern::getEstimatedModuleSize(horizontalState);
            for (int index = 0; index < max; index++) {
                Ref<AlignmentPattern> center(possibleCenters_[index]);
                // Look for about the same center and module size:
                if (center->aboutEquals(centerI, centerJ, verticalModuleSize, horizontalModuleSize))
                {
                    center->combineEstimate(centerI, centerJ, verticalState, horizontalState);
                    return;
                }
            }
            Ref<AlignmentPattern> newPattern(new AlignmentPattern(centerJ, centerI, horizontalState, verticalState));
            possibleCenters_.push_back(newPattern);
            if (callback_ != 0)
            {
                callback_->foundPossibleResultPoint(*newPattern);
            }
        }
    }
}

AlignmentPatternFinder::AlignmentPatternFinder(Ref<BitMatrix> image, float moduleSize, Ref<ResultPointCallback>const& callback)
: image_(image), moduleSize_(moduleSize), callback_(callback) {
}

bool AlignmentPatternFinder::findPatternLine(Ref<FinderPattern>const &from, 
                                             Ref<FinderPattern>const &to, double &aValue, double &pValue){
    vector<pair<int, int> > sample;
    if (!findPatternLineSample(from, to, sample))
    {
        return false;
    }
    
    vector<map<int, pair<int, double> > > houghMatrix(180);
    
    for (size_t i = 0; i < sample.size(); ++i) {
        for (int a = 0; a < 180; ++a)
        {
            double p = sample[i].first * cosTable_[a] + sample[i].second * sinTable_[a];
            int pBin = static_cast<int>(p + 0.5f);
            map<int, pair<int, double> >::iterator iter = houghMatrix[a].find(pBin);
            if (iter !=  houghMatrix[a].end())
            {
                iter->second.first += 1;
                iter->second.second += p;
            }
            else
            {
                houghMatrix[a][pBin] = make_pair(1, p);
            }
        }
    }
    
    int maxCount = 0;
    int maxA = 0;
    int maxP = 0;
    for (int a = 0; a < 180; ++a) {
        for (map<int, pair<int, double> >::iterator iter = houghMatrix[a].begin();
             iter != houghMatrix[a].end(); ++iter) {
            if (iter->second.first > maxCount)
            {
                maxCount = iter->second.first;
                maxA = a;
                maxP = iter->first;
            }
        }
    }
    aValue = static_cast<double>(maxA);
    pValue = houghMatrix[maxA][maxP].second / houghMatrix[maxA][maxP].first;
    return true;
}

bool AlignmentPatternFinder::findPatternLineSample(Ref<FinderPattern>const &from, Ref<FinderPattern>const &to,
                                                   vector<pair<int, int> > &sample){
    int startX, startY;
    bool steep;
    int xstep;
    if (!findPatternLineStartPoint(from, to, startX, startY, steep, xstep))
        return false;
    
    const float *horizontalState, *verticalState;
    float deltaX;
    int beginY, endY;
    if (steep)
    {
        horizontalState = from->getVerticalState();
        verticalState = from->getHorizontalState();
        if (xstep == -1)
            deltaX = horizontalState[0] / 2;
        else
            deltaX = horizontalState[4] / 2;
        beginY = static_cast<int>(static_cast<float>(startY) - verticalState[2] / 2.0f - verticalState[3]);
        endY = static_cast<int>(static_cast<float>(startY) + verticalState[2] / 2.0f + verticalState[3]);
    }
    else
    {
        horizontalState = from->getHorizontalState();
        verticalState = from->getVerticalState();
        if (xstep == -1)
            deltaX = horizontalState[0] / 2;
        else
            deltaX = horizontalState[4] / 2;
        beginY = static_cast<int>(static_cast<float>(startY) - verticalState[2] / 2.0f - verticalState[3]);
        endY = static_cast<int>(static_cast<float>(startY) + verticalState[2] / 2.0f + verticalState[3]);
    }
    
    sample.clear();
    
    findPatternLineSampleOneSide(startX, startY, beginY + 1, static_cast<int>(deltaX + 0.5f), steep, -1, sample);
    findPatternLineSampleOneSide(startX, startY + 1, endY + 1, static_cast<int>(deltaX + 0.5f), steep, 1, sample);
    
    int suitable = static_cast<int>(0.7f * (endY - beginY + 1));
    if (static_cast<int>(sample.size()) < suitable)
    {
        return false;
    }
    
    return true;
}


void AlignmentPatternFinder::findPatternLineSampleOneSide(int startX, int startY, int endY, int deltaX, int steep, int side,
                                                          std::vector<std::pair<int, int> > &sample){
    int variance = deltaX == 0 ? 1 : deltaX;
    
    int lastX = startX;
    int lastY = startY - 1;
    for (int y = startY; y != endY; y += side) {
        int estX = findPatternLineCenter(lastX, y, deltaX, steep);
        if (abs(estX - lastX) > abs(y - lastY) * variance)
        {
            continue;
        }
        lastX = estX;
        lastY = y;
        if (steep)
            sample.push_back(make_pair(lastY, lastX));
        else
            sample.push_back(make_pair(lastX, lastY));
    }
}

int AlignmentPatternFinder::findPatternLineCenter(int startX, int startY, int deltaX, bool steep){
    int beginX = findPatternLineCenterOneSide(startX, startY, deltaX, steep, -1);
    int endX = findPatternLineCenterOneSide(startX, startY, deltaX, steep, 1);
    return (beginX + endX) / 2;
}

bool AlignmentPatternFinder::findPatternLineStartPoint(Ref<FinderPattern>const &from, Ref<FinderPattern>const &to,
                                                       int &startX, int &startY, bool &steep, int &xstep) {
    // Mild variant of Bresenham's algorithm;
    // see http:// en.wikipedia.org/wiki/Bresenham's_line_algorithm
    
    float fromX, fromY, toX, toY;
    steep = fabs(to->getY() - from->getY()) > fabs(to->getX() - from->getX());
    
    if (steep)
    {
        fromX = from->getY();
        fromY = from->getX();
        toX = to->getY();
        toY = to->getX();
    }
    else
    {
        fromX = from->getX();
        fromY = from->getY();
        toX = to->getX();
        toY = to->getY();
    }
    
    int dx = static_cast<int>(fabs(toX - fromX));
    int dy = static_cast<int>(fabs(toY - fromY));
    int error = -dx >> 1;
    xstep = fromX < toX ? 1 : -1;
    int ystep = fromY < toY ? 1 : -1;
    
    int beginX, beginY;
    
    // In black pixels, looking for white, first or second time.
    int state = 0;
    // Loop up until x == toX, but not beyond
    int xLimit = static_cast<int>(toX + xstep);
    for (int x = static_cast<int>(fromX), y = static_cast<int>(fromY); x != xLimit; x += xstep) {
        int realX, realY;
        if (steep)
        {
            realX = y;
            realY = x;
        }
        else
        {
            realX = x;
            realY = y;
        }
        
        switch (state) {
            case 0:
                if (!image_->get(realX, realY))
                    ++state;
                break;
            case 1:
                if (image_->get(realX, realY))
                {
                    beginX = x;
                    beginY = y;
                    ++state;
                }
                break;
            case 2:
                if (!image_->get(realX, realY))
                {
                    startX = (beginX + x) / 2;
                    startY = (beginY + y) / 2;
                    return true;
                }
                break;
        }
        
        error += dy;
        if (error > 0)
        {
            if (y == toY)
            {
                break;
            }
            y += ystep;
            error -= dx;
        }
    }
    return false;
}

int AlignmentPatternFinder::findPatternLineCenterOneSide(int startX, int startY, int deltaX, bool steep, int side) {
    int x, y, i;
    int range = static_cast<int>(1.5f * deltaX);
    for (i = 0; i <= range; ++i) {
        if (steep)
        {
            x = startY;
            y = startX + i * side;
        }
        else
        {
            x = startX + i * side;
            y = startY;
        }
        if (x < 0 || (size_t)x >= image_->getWidth() || y < 0 || (size_t)y >= image_->getHeight())
        {
            if (i > 0)
                --i;
            break;
        }
        if (!image_->get(x, y))
        {
            if (i > 0)
                --i;
            break;
        }
    }
    if (i < 0.8f * deltaX)
    {
        int beginI = static_cast<int>(1.2f * deltaX);
        int beforeI = i;
        for (i = beginI; i > 0; --i) {
            if (steep)
            {
                x = startY;
                y = startX + i * side;
            }
            else
            {
                x = startX + i * side;
                y = startY;
            }
            if (x < 0 || (size_t)x >= image_->getWidth() || y < 0 || (size_t)y >= image_->getHeight())
            {
                if (i == beginI)
                    i = beforeI;
                else if (i > 0)
                    --i;
                break;
            }
            if (image_->get(x, y))
                break;
        }
    }
    return startX + i * side;
}

float AlignmentPatternFinder::crossCheckVertical(size_t startI, size_t centerJ, int maxCount,
                                                 int originalStateCountTotal, int *state) {
    int maxI = image_->getHeight();
    for (int i = 0; i < 5; i++)
        state[i] = 0;
    
    if (!image_->get(centerJ, startI))
    {
        if (static_cast<int>(startI) + 1 < maxI  && image_->get(centerJ, startI + 1))
            startI = startI + 1;
        else if (0 < static_cast<int>(startI) - 1 && image_->get(centerJ, startI - 1))
            startI = startI - 1;
        else
            return nan();
    }
    
    // Start counting up from center
    int i = startI;
    while (i >= 0 && image_->get(centerJ, i)) {
        state[2]++;
        i--;
    }
    if (i < 0)
    {
        return  nan();
    }
    while (i >= 0 && !image_->get(centerJ, i) && state[1] <= maxCount) {
        state[1]++;
        i--;
    }
    // If already too many modules in this state or ran off the edge:
    if (i < 0 || state[1] > maxCount)
    {
        return  nan();
    }
    while (i >= 0 && image_->get(centerJ, i) && state[0] <= maxCount) {
        state[0]++;
        i--;
    }
    
    // Now also count down from center
    i = startI + 1;
    while (i < maxI && image_->get(centerJ, i)) {
        state[2]++;
        i++;
    }
    if (i == maxI)
    {
        return  nan();
    }
    while (i < maxI && !image_->get(centerJ, i) && state[3] < maxCount) {
        state[3]++;
        i++;
    }
    if (i == maxI || state[3] >= maxCount)
    {
        return  nan();
    }
    while (i < maxI && image_->get(centerJ, i) && state[4] < maxCount) {
        state[4]++;
        i++;
    }
    
    int stateTotal = state[1] + state[2] + state[3];
    
    float maxVariance = 2.0;
    float variance = stateTotal * 1.0 / originalStateCountTotal;
    
    if (variance > maxVariance || variance < 1.0 / maxVariance)
        return  nan();
    
    return foundPatternCross(state) ? centerFromEnd(state, i) :  nan();
}

float AlignmentPatternFinder::crossCheckHorizontal(size_t startJ, size_t centerI, int maxCount,
                                                   int originalStateCountTotal, int *state) {
    int maxJ = image_->getWidth();
    for (int i = 0; i < 5; i++)
        state[i] = 0;
    
    if (!image_->get(startJ, centerI))
    {
        if (static_cast<int>(startJ) + 1 < maxJ  && image_->get(startJ + 1, centerI))
            startJ = startJ + 1;
        else if (0 < static_cast<int>(startJ) - 1 && image_->get(startJ - 1, centerI))
            startJ = startJ - 1;
        else
            return  nan();
    }
    
    int j = startJ;
    while (j >= 0 && image_->get(j, centerI)) {
        state[2]++;
        j--;
    }
    if (j < 0)
    {
        return  nan();
    }
    while (j >= 0 && !image_->get(j, centerI) && state[1] <= maxCount) {
        state[1]++;
        j--;
    }
    if (j < 0 || state[1] > maxCount)
    {
        return  nan();
    }
    while (j >= 0 && image_->get(j, centerI) && state[0] <= maxCount) {
        state[0]++;
        j--;
    }
    
    j = startJ + 1;
    while (j < maxJ && image_->get(j, centerI)) {
        state[2]++;
        j++;
    }
    if (j == maxJ)
    {
        return  nan();
    }
    while (j < maxJ && !image_->get(j, centerI) && state[3] < maxCount) {
        state[3]++;
        j++;
    }
    if (j == maxJ || state[3] >= maxCount)
    {
        return  nan();
    }
    while (j < maxJ && image_->get(j, centerI) && state[4] < maxCount) {
        state[4]++;
        j++;
    }
    
    int stateTotal = state[1] + state[2] + state[3];
    float maxVariance = 2.0;
    float variance = stateTotal * 1.0 / originalStateCountTotal;
    if (variance > maxVariance || variance < 1.0 / maxVariance)
        return  nan();
    
    return foundPatternCross(state) ? centerFromEnd(state, j) : nan();
} 


Ref<AlignmentPattern> AlignmentPatternFinder::findInRange(Ref<AlignmentPattern>const &estimateCenter, 
                                                          size_t startX, size_t startY, size_t width, size_t height, ErrorHandler & err_handler) {
    size_t maxJ = startX + width;
    size_t middleI = startY + (height >> 1);
    
    int stateCount[5];
    for (size_t iGen = 0; iGen < height; iGen++) {
        // Search from middle outwards
        size_t i = middleI + ((iGen & 0x01) == 0 ? ((iGen + 1) >> 1) : -((iGen + 1) >> 1));
        
        stateCount[0] = 0;
        stateCount[1] = 0;
        stateCount[2] = 0;
        stateCount[3] = 0;
        stateCount[4] = 0;
        size_t j = startX;
        
        int currentState = 0;
        for (; j < maxJ; ++j) {
            switch (currentState) {
                case 0:
                case 2:
                    if (image_->get(j, i))
                        ++stateCount[currentState];
                    else
                        ++stateCount[++currentState];
                    break;
                case 1:
                case 3:
                    if (image_->get(j, i))
                        ++stateCount[++currentState];
                    else
                        ++stateCount[currentState];
                    break;
                case 4:
                    if (image_->get(j, i))
                    {
                        ++stateCount[currentState];
                    }
                    else
                    {
                        if (foundPatternCross(stateCount)){
                            handlePossibleCenter(stateCount, i, j);
                        }
                        
                        stateCount[0] = stateCount[2];
                        stateCount[1] = stateCount[3];
                        stateCount[2] = stateCount[4];
                        stateCount[3] = 1;
                        stateCount[4] = 0;
                        currentState = 3;
                    }
                    break;
            }
        }
        if (foundPatternCross(stateCount))
            handlePossibleCenter(stateCount, i, maxJ);
    }
    
    if (possibleCenters_.size() > 0)
    {
        int maxFixIndex = 0;
        float maxFix = 0.0;
        for (size_t i = 0; i < possibleCenters_.size(); ++i) {
            float fix = possibleCenters_[i]->matchPattern(*image_);
            if (fix >= 90.0f)
            {
                maxFix = fix;
                maxFixIndex = i;
                break;
            }
            else if (possibleCenters_[i]->aboutEquals(estimateCenter->getY(), estimateCenter->getX(),
                                                      estimateCenter->getEstimatedVerticalModuleSize(),
                                                      estimateCenter->getEstimatedHorizontalModuleSize()) && fix >= 70.0f)
            {
                maxFix = fix;
                maxFixIndex = i;
                break;
            }
            if (fix > maxFix)
            {
                maxFix = fix;
                maxFixIndex = i;
            }
        }
        
        return Ref<AlignmentPattern>(new
                                     AlignmentPattern((possibleCenters_[maxFixIndex])->getX(),  (possibleCenters_[maxFixIndex])->getY(), (possibleCenters_[maxFixIndex])->getEstimatedModuleSize()));
    }
    
    err_handler = ReaderErrorHandler("Could not find alignment pattern");
    return std::vector<Ref<AlignmentPattern> >();
}

#endif
