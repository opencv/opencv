// -*- mode:c++; tab-width:2; indent-tabs-mode:nil; c-basic-offset:2 -*-
#ifndef __DETECTOR_H__
#define __DETECTOR_H__

/*
 *  Detector.hpp
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
#include "../../common/detector_result.hpp"
#include "../../common/bit_matrix.hpp"
#include "alignment_pattern.hpp"
#include "../../common/perspective_transform.hpp"
#include "../../result_point_callback.hpp"
#include "finder_pattern_info.hpp"
#include "finder_pattern.hpp"
#include "pattern_result.hpp"
#include "../../common/unicom_block.hpp"
#include "../../error_handler.hpp"
#include <vector>

#include "opencv2/calib3d.hpp"

namespace zxing
{

class DecodeHints;

namespace qrcode {

class Detector : public Counted
{
public:
    enum  DetectorState
    {
        START = 10,
        FINDFINDERPATTERN = 11,
        FINDALIGNPATTERN = 12,
    };
    
    // Fix module size error when LEFT_SPILL or RIGHT_SPILL
    // By Valiantliu
    enum FinderPatternMode
    {
        NORMAL = 0,
        LEFT_SPILL = 1,
        RIGHT_SPILL = 2,
        UP_SPILL = 3,
        DOWN_SPILL = 4,
    };
    
    typedef struct Rect_
    {
        int x;
        int y;
        int width;
        int height;
    } Rect;
    
private:
    Ref<BitMatrix> image_;
    Ref<UnicomBlock> block_;
    
    Ref<ResultPointCallback> callback_;
    
    std::vector<Ref<PatternResult> > possiblePatternResults_;
    
    DetectorState detectorState_;
    bool finderConditionLoose_;
    
protected:
    Ref<BitMatrix> getImage() const;
    Ref<ResultPointCallback> getResultPointCallback() const;
    static int computeDimension(Ref<ResultPoint> topLeft, Ref<ResultPoint> topRight, Ref<ResultPoint> bottomLeft,
                                float moduleSizeX, float moduleSizeY);
    float refineModuleSize(Ref<ResultPoint> topLeft, Ref<ResultPoint> topRight, Ref<ResultPoint> bottomLeft, int dimension);
    float calculateModuleSize(Ref<ResultPoint> topLeft, Ref<ResultPoint> topRight, Ref<ResultPoint> bottomLeft);
    float calculateModuleSizeOneWay(Ref<ResultPoint> pattern, Ref<ResultPoint> otherPattern, int patternState, int otherPatternState);
    float sizeOfBlackWhiteBlackRunBothWays(int fromX, int fromY, int toX, int toY, int patternState, bool isReverse);
    float sizeOfBlackWhiteBlackRunBothWays(int fromX, int fromY, int toX, int toY);
    float sizeOfBlackWhiteBlackRun(int fromX, int fromY, int toX, int toY);
    Ref<AlignmentPattern> findAlignmentInRegion(float overallEstModuleSize, int estAlignmentX, int estAlignmentY, float allowanceFactor, ErrorHandler & err_handler);
    Ref<AlignmentPattern> findAlignmentWithFitLine(Ref<ResultPoint> topLeft, Ref<ResultPoint> topRight, Ref<ResultPoint> bottomLeft, float moduleSize, ErrorHandler & err_handler);
    int fitLine(std::vector<Ref<ResultPoint> > &oldPoints, float& k, float& b, int& a);
    bool checkTolerance(Ref<ResultPoint> &topLeft, Ref<ResultPoint> &topRight, Rect& topRightRect,
                        double modelSize, Ref<ResultPoint>& p, int flag);
    void findPointsForLine(Ref<ResultPoint> &topLeft, Ref<ResultPoint> &topRight, Ref<ResultPoint> &bottomLeft,
                           Rect topRightRect, Rect bottomLeftRect, std::vector<Ref<ResultPoint> > &topRightPoints,
                           std::vector<Ref<ResultPoint> > &bottomLeftPoints, float modelSize);
    bool checkConvexQuadrilateral(Ref<ResultPoint> topLeft, Ref<ResultPoint> topRight, Ref<ResultPoint> bottomLeft,
                                  Ref<ResultPoint>bottomRight);
    
public:
    virtual Ref<PerspectiveTransform> createTransform(Ref<ResultPoint> topLeft, Ref<ResultPoint> topRight, Ref <
                                                      ResultPoint > bottomLeft, Ref<ResultPoint> alignmentPattern, int dimension);
    Ref<PerspectiveTransform> createInvertedTransform(Ref<ResultPoint> topLeft, Ref<ResultPoint> topRight, Ref <
                                                      ResultPoint > bottomLeft, Ref<ResultPoint> alignmentPattern, int dimension);
    
    Ref<PerspectiveTransform> createTransform( Ref<FinderPatternInfo>finderPatternInfo,
                                              Ref<ResultPoint> alignmentPattern, int dimension);
    
    static Ref<BitMatrix> sampleGrid(Ref<BitMatrix> image, int dimension, Ref<PerspectiveTransform>, ErrorHandler &err_handler);
    static Ref<BitMatrix> sampleGrid(Ref<BitMatrix> image, int dimension, cv::Mat&, ErrorHandler &err_handler);
    static Ref<BitMatrix> sampleGrid(Ref<BitMatrix> image, int dimension,
                                     float ax, float bx, float cx, float dx, float ex, float fx,
                                     float ay, float by, float cy, float dy, float ey, float fy,
                                     ErrorHandler &err_handler);
    
    Detector(Ref<BitMatrix> image, Ref<UnicomBlock> block);
    
    void detect(DecodeHints const& hints,ErrorHandler & err_handler);
    Ref<DetectorResult> getResultViaAlignment(size_t patternIdx, size_t alignmentIndex, int possibleDimension, ErrorHandler & err_handler);
    Ref<DetectorResult> getResultViaAlignmentMore(DecodeHints const& hints, size_t patternIdx, size_t alignmentIndex, ArrayRef< Ref<ResultPoint> > points, int dimension, float module_size, int mode, ErrorHandler & err_handler);
    Ref<DetectorResult> getResultViaPoints(DecodeHints const& hints, int dimension,  std::vector<cv::Point2f>& pts_src, std::vector<cv::Point2f>& pts_dst, bool is_homo, ErrorHandler & err_handler);
    
    int getPossibleAlignmentPointsMore(size_t patternIdx, size_t alignmentIdx, int dimension, float module_size, int center_nums);
    int getFixedAlignmentPoints(size_t patternIdx, int dimension, float module_size, std::vector<cv::Point2f>& pts_src, std::vector<cv::Point2f>& pts_dst);
    int getFlexibleAlignmentPoints(size_t patternIdx, size_t alignmentIdx, int dimension, float module_size, std::vector<cv::Point2f>& pts_src, std::vector<cv::Point2f>& pts_dst);
    int getPatternPoints(size_t patternIdx, size_t alignmentIdx, int dimension, std::vector<cv::Point2f>& pts_src, std::vector<cv::Point2f>& pts_dst);
    int getCornerPoints(size_t patternIdx, int dimension, float module_size, std::vector<cv::Point2f>& pts_src, std::vector<cv::Point2f>& pts_dst, bool pure_code = false);
    int getPossibleAlignmentPatterRect(float x, float y);
    // Added by Valiantliu
    int getPossiblePatternCount(){return possiblePatternResults_.size();}
    int getPossibleAlignmentCount(size_t idx, DecodeHints& hints);
    Ref<AlignmentPattern> findPossibleAlignment(int dimension, float moduleSize, float x, float y);
    
    Ref<AlignmentPattern> getNearestAlignmentPattern(int tryFindRange, float moduleSize, int estAlignmentX, int estAlignmentY);
    bool hasSameResult(std::vector<Ref<AlignmentPattern> > possibleAlignmentPatterns, Ref<AlignmentPattern> alignmentPattern);
    void fixAlignmentPattern(float &alignmentX, float &alignmentY,
                             Ref<ResultPoint> topLeft, Ref<ResultPoint> topRight, Ref<ResultPoint> bottomLeft,
                             float moduleSize);
    
    Ref<PatternResult> processFinderPatternInfo(DecodeHints& hints, Ref<FinderPatternInfo> info, ErrorHandler & err_handler);
    
    int getPossiblePatternRect(size_t idx, float module_size, int ori);
    int locatePatternRect(float x, float y, std::vector<cv::Point2f>& points, float module_size, int ori);
    
public:
    Ref<FinderPatternInfo>getFinderPatternInfo(int idx){return possiblePatternResults_[idx]->finderPatternInfo;}
    Ref<AlignmentPattern> getAlignmentPattern(int patternIdx, int alignmentIdx){return possiblePatternResults_[patternIdx]->possibleAlignmentPatterns[alignmentIdx];}
    
    DetectorState getState(){return detectorState_;}
    void setFinderLoose(bool loose){finderConditionLoose_=loose;}
    bool getFinderLoose(){return finderConditionLoose_;}
    
    unsigned int getPossibleVersion(int idx){return possiblePatternResults_[idx]->possibleVersion;}
    float getPossibleFix(int idx){return possiblePatternResults_[idx]->possibleFix;}
    float getPossibleModuleSize(int idx){return possiblePatternResults_[idx]->possibleModuleSize;}
    int getDimension(int idx){return possiblePatternResults_[idx]->possibleDimension;}
};

}  // namespace qrcode
}  // namespace zxing

#endif  // __DETECTOR_H__
