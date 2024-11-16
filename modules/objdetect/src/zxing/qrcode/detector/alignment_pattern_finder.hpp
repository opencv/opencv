#ifndef __ALIGNMENT_PATTERN_FINDER_H__
#define __ALIGNMENT_PATTERN_FINDER_H__

/*
 *  AlignmentPatternFinder.hpp
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

#include "alignment_pattern.hpp"
#include "../../common/counted.hpp"
#include "../../common/bit_matrix.hpp"
#include "../../result_point_callback.hpp"
#include "alignment_pattern.hpp"
#include "finder_pattern.hpp"
#include "../../error_handler.hpp"

#include <vector>

namespace zxing {
namespace qrcode {

class AlignmentPatternFinder : public Counted {
private:
    static int CENTER_QUORUM;
    static int MIN_SKIP;
    static int MAX_MODULES;
    
    Ref<BitMatrix> image_;
    std::vector<AlignmentPattern *> *possibleCenters_;
    
    int startX_;
    int startY_;
    int width_;
    int height_;
    float moduleSize_;
    
#ifdef FIND_WHITE_ALIGNMENTPATTERN
    std::vector<AlignmentPattern *> *possibleWhiteCenters_;
    Ref<AlignmentPattern> confirmedWhiteAlignmentPattern;
    Ref<AlignmentPattern> confirmedAlignmentPattern;
    bool foundAlignmentPattern;
    bool foundWhiteAlignmentPattern;
    bool lastTry;
#endif
    
    static float centerFromEnd(std::vector<int> &stateCount, int end);
    float crossCheckVertical(int startI, int centerJ, int maxCount, int originalStateCountTotal);
#ifdef FIND_WHITE_ALIGNMENTPATTERN
    static float centerFromEndWhite(std::vector<int> &stateCount, int end);
    float crossCheckVerticalWhite(int startI, int centerJ, int maxCount, int originalStateCountTotal);
#endif
    
public:
    AlignmentPatternFinder(Ref<BitMatrix> image, int startX, int startY, int width, int height,
                           float moduleSize, Ref<ResultPointCallback>const& callback);
    AlignmentPatternFinder(Ref<BitMatrix> image, float moduleSize, Ref<ResultPointCallback>const& callback);
    ~AlignmentPatternFinder();
    Ref<AlignmentPattern> findByPattern(Ref<AlignmentPattern>const &estimateCenter, Ref<FinderPattern>const &topLeft,
                                        Ref<FinderPattern>const &topRight, Ref<FinderPattern>const &bottomLeft);
#ifdef FIND_WHITE_ALIGNMENTPATTERN
    std::vector<Ref<AlignmentPattern> > find(ErrorHandler & err_handler);
    void setLastTry(bool toSet){lastTry = toSet;}
    void foundPatternCross(std::vector<int> &stateCount);
    void  handlePossibleCenter(std::vector<int> &stateCount, int i, int j);
#else 
    Ref<AlignmentPattern> find(ErrorHandler & err_handler);
    bool foundPatternCross(std::vector<int> &stateCount);
    Ref<AlignmentPattern> handlePossibleCenter(std::vector<int> &stateCount, int i, int j);
#endif
    
private:
    AlignmentPatternFinder(const AlignmentPatternFinder&);
    AlignmentPatternFinder& operator =(const AlignmentPatternFinder&);
    Ref<ResultPointCallback> callback_;
    
#ifdef   USING_WX
private:
    std::vector<float> sinTable_;
    std::vector<float> cosTable_;
    float crossCheckVertical(size_t startI, size_t centerJ, int maxCount, int originalStateCountTotal, int *state);
    float crossCheckHorizontal(size_t startI, size_t centerJ, int maxCount, int originalStateCountTotal, int *state);
    static float centerFromEnd(int* stateCount, int end);
    bool foundPatternCross(int* stateCount);
    
public:
    AlignmentPatternFinder(Ref<BitMatrix> image, float moduleSize, Ref<ResultPointCallback>const& callback);
    
    void initAngleTable();
    bool findPatternLine(Ref<FinderPattern>const &from, Ref<FinderPattern>const &to, double &aValue, double &pValue);
    bool findPatternLineSample(Ref<FinderPattern>const &from, Ref<FinderPattern>const &to,
                               std::vector<std::pair<int, int> > &sample);
    int findPatternLineCenterOneSide(int startX, int startY, int deltaX, bool steep, int side);
    int findPatternLineCenter(int startX, int startY, int deltaX, bool steep);
    bool findPatternLineStartPoint(Ref<FinderPattern>const &from, Ref<FinderPattern>const &to,
                                   int &startX, int &startY, bool &steep, int &xstep);
    void findPatternLineSampleOneSide(int startX, int startY, int endY, int deltaX, int steep, int side,
                                      std::vector<std::pair<int, int> > &sample);
    Ref<AlignmentPattern> findInRange(Ref<AlignmentPattern>const &estimateCenter,
                                      size_t startX, size_t startY, size_t width, size_t height, ErrorHandler & err_handler);
    void handlePossibleCenter(int *stateCount, size_t i, size_t j);
#endif
};
}  // namespace qrcode
}  // namespace zxing

#endif  // __ALIGNMENT_PATTERN_FINDER_H__

