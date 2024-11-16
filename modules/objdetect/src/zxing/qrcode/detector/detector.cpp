// -*- mode:c++; tab-width:2; indent-tabs-mode:nil; c-basic-offset:2 -*-
/*
 *  Detector.cpp
 *  zxing
 *
 *  Created by Christian Brunschen on 14/05/2008.
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

#include "detector.hpp"
#include "finder_pattern_finder.hpp"
#include "finder_pattern.hpp"
#include "alignment_pattern.hpp"
#include "alignment_pattern_finder.hpp"
#include "../version.hpp"
#include "../../common/grid_sampler.hpp"
#include "../../decode_hints.hpp"
#include "../../common/detector/math_utils.hpp"

#include <sstream>
#include <cstdlib>

#include<iostream>
#include <queue>

using std::ostringstream;
using std::abs;
using std::min;
using std::max;
using zxing::qrcode::Detector;
using zxing::Ref;
using zxing::BitMatrix;
using zxing::ResultPointCallback;
using zxing::DetectorResult;
using zxing::PerspectiveTransform;
using zxing::qrcode::AlignmentPattern;
using zxing::common::detector::MathUtils;
using zxing::qrcode::FinderPattern;
using zxing::ErrorHandler;

// VC++
using zxing::DecodeHints;
using zxing::qrcode::FinderPatternFinder;
using zxing::qrcode::FinderPatternInfo;
using zxing::ResultPoint;
using zxing::qrcode::PatternResult;
using zxing::UnicomBlock;

#define FIX_ALIGNMENT_CENTER 1

// Encapsulates logic that can detect a QR Code in an image, 
// even if the QR Code is rotated or skewed, or partially obscured.
Detector::Detector(Ref<BitMatrix> image, Ref<UnicomBlock> block) :
image_(image), block_(block)
{
    detectorState_ = START;
    finderConditionLoose_ = false;
    possiblePatternResults_.clear();
}

Ref<BitMatrix> Detector::getImage() const {
    return image_;
}

Ref<ResultPointCallback> Detector::getResultPointCallback() const {
    return callback_;
}

// Detects a QR Code in an image
void Detector::detect(DecodeHints const& hints, ErrorHandler & err_handler)
{
    callback_ = hints.getResultPointCallback();
    FinderPatternFinder finder(image_, block_, hints.getResultPointCallback());
    finder.setFinderLoose(finderConditionLoose_);
    
    std::vector<Ref<FinderPatternInfo> > finderInfos = finder.find(hints, err_handler);
    if (err_handler.ErrCode())
        return;
    
    // Get all possible results
    possiblePatternResults_.clear();
    for (size_t i = 0; i < finderInfos.size(); i++)
    {
        Ref<PatternResult> result(new PatternResult(finderInfos[i]));
        result->possibleVersion = 0;
        result->possibleFix = 0.0f;
        result->possibleModuleSize = 0.0f;
        
        possiblePatternResults_.push_back(result);
    }
    detectorState_ = FINDFINDERPATTERN;
}

int Detector::locatePatternRect(float x, float y, std::vector<cv::Point2f>& points, float module_size, int ori) {
    int width = image_->getWidth();
    int height = image_->getHeight();
    
    float offset = fmin(10 * module_size,  width / 3.0);
    float min_x = fmax(0, x - offset);
    float max_x = fmin(x + offset, width - 1);
    float min_y = fmax(0, y - offset);
    float max_y = fmin(y + offset, height - 1);
    
    cv::Mat flag_mat(height, width, CV_8UC1, cv::Scalar(0));
    
    std::vector<cv::Point2f> cur_point_list, next_point_list;
    cur_point_list.push_back(cv::Point2f(x, y));
    flag_mat.at<uchar>(y, x) = 255;
    
    int DIR[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    
    while (!cur_point_list.empty()) {
        cv::Point2f pt = cur_point_list.back();
        cur_point_list.pop_back();
        
        int x = static_cast<int>(pt.x), y = static_cast<int>(pt.y);
        for (int i = 0; i < 4; i++) {
            int grow_x = x + DIR[i][0], grow_y = y + DIR[i][1];
            if (grow_x < min_x || grow_y < min_y || grow_x > max_x || grow_y > max_y)
                return -1;
            
            if (image_->get(grow_x, grow_y))
            {
                if (flag_mat.at<uchar>(grow_y, grow_x) == 0)
                {
                    flag_mat.at<uchar>(grow_y, grow_x) = 255;
                    cur_point_list.push_back(cv::Point2f(grow_x, grow_y));
                }
            }
            else
            {
                if (flag_mat.at<uchar>(grow_y, grow_x) == 0)
                {
                    flag_mat.at<uchar>(grow_y, grow_x) = 255;
                    next_point_list.push_back(cv::Point2f(grow_x, grow_y));
                }
            }
        }
    }
    
    if (next_point_list.empty()) return -1;
    
    while (!next_point_list.empty()) {
        cv::Point2f pt = next_point_list.back();
        next_point_list.pop_back();
        
        int x = static_cast<int>(pt.x), y = static_cast<int>(pt.y);
        for (int i = 0; i < 4; i++) {
            int grow_x = x + DIR[i][0], grow_y = y + DIR[i][1];
            if (grow_x < min_x || grow_y < min_y || grow_x > max_x || grow_y > max_y){
                return -1;
            }
            
            if (image_->get(grow_x, grow_y) == 0)
            {
                if (flag_mat.at<uchar>(grow_y, grow_x) == 0)
                {
                    flag_mat.at<uchar>(grow_y, grow_x) = 255;
                    next_point_list.push_back(cv::Point2f(grow_x, grow_y));
                }
            }
            else
            {
                if (flag_mat.at<uchar>(grow_y, grow_x) == 0)
                {
                    flag_mat.at<uchar>(grow_y, grow_x) = 255;
                    cur_point_list.push_back(cv::Point2f(grow_x, grow_y));
                }
            }
        }
    }
    
    if (cur_point_list.empty()) return -1;
    
    while (!cur_point_list.empty()) {
        cv::Point2f pt = cur_point_list.back();
        cur_point_list.pop_back();
        
        int x = static_cast<int>(pt.x), y = static_cast<int>(pt.y);
        bool has_add = false;
        for (int i = 0; i < 4; i++) {
            int grow_x = x + DIR[i][0], grow_y = y + DIR[i][1];
            
            if (grow_x < min_x || grow_y < min_y || grow_x > max_x || grow_y > max_y)
            {
                if (!has_add){
                    next_point_list.push_back(cv::Point2f(x, y));
                    has_add = true;
                }
                continue;
            }
            
            if (image_->get(grow_x, grow_y))
            {
                if (flag_mat.at<uchar>(grow_y, grow_x) == 0)
                {
                    flag_mat.at<uchar>(grow_y, grow_x) = 255;
                    cur_point_list.push_back(cv::Point2f(grow_x, grow_y));
                }
            }
            else
            {
                if (flag_mat.at<uchar>(grow_y, grow_x) == 0)
                {
                    flag_mat.at<uchar>(grow_y, grow_x) = 255;
                    next_point_list.push_back(cv::Point2f(grow_x, grow_y));
                }
            }
        }
    }
    
    if (next_point_list.size() < 10) return -1;
    
    int min_sum_idx = -1, max_sum_idx = -1, min_diff_idx = -1, max_diff_idx = -1;
    float min_sum = INT_MAX, max_sum = 0, min_diff = INT_MAX, max_diff = -INT_MAX;
    for (size_t i = 0; i < next_point_list.size(); i++) {
        float sum = next_point_list[i].x + next_point_list[i].y;
        float diff = next_point_list[i].x - next_point_list[i].y;
        if (sum  < min_sum)
        {
            min_sum_idx = i;
            min_sum = sum;
        }
        if (sum > max_sum)
        {
            max_sum_idx = i;
            max_sum = sum;
        }
        if (diff < min_diff)
        {
            min_diff_idx = i;
            min_diff = diff;
        }
        if (diff > max_diff)
        {
            max_diff_idx = i;
            max_diff = diff;
        }
    }
    cv::Point2f pt1 = next_point_list[min_sum_idx];
    cv::Point2f pt2 = next_point_list[max_sum_idx];
    cv::Point2f pt3 = next_point_list[min_diff_idx];
    cv::Point2f pt4 = next_point_list[max_diff_idx];

    if (fabs(pt2.x - pt1.x) > 10 * module_size) return -1;
    if (fabs(pt2.y - pt1.y) > 10 * module_size) return -1;
    
    points.clear();
    if (ori == 1)
    {
        points.push_back(pt3);
        points.push_back(pt4);
        points.push_back(pt2);
        points.push_back(pt1);
    }
    else if (ori == 2)
    {
        points.push_back(pt2);
        points.push_back(pt1);
        points.push_back(pt4);
        points.push_back(pt3);
    }
    else if (ori == 3)
    {
        points.push_back(pt4);
        points.push_back(pt3);
        points.push_back(pt1);
        points.push_back(pt2);
    }
    else
    {
        points.push_back(pt1);  // top left
        points.push_back(pt2);  // bottom right
        points.push_back(pt3);  // bottom left
        points.push_back(pt4);  // top right
    }
    
    return 0;
}

int Detector::getFixedAlignmentPoints(size_t patternIdx, int dimension, float module_size,  std::vector<cv::Point2f>& pts_src, std::vector<cv::Point2f>& pts_dst) {
    ErrorHandler err_handler;
    Version *provisionalVersion = Version::getProvisionalVersionForDimension(dimension, err_handler);
    if (err_handler.ErrCode() != 0)
        return -1;
    
    if (patternIdx >= possiblePatternResults_.size()) return -1;
    
    pts_src.clear();
    pts_dst.clear();
    
    Ref<FinderPattern> topLeft(possiblePatternResults_[patternIdx]->finderPatternInfo->getTopLeft());
    Ref<FinderPattern> topRight(possiblePatternResults_[patternIdx]->finderPatternInfo->getTopRight());
    Ref<FinderPattern> bottomLeft(possiblePatternResults_[patternIdx]->finderPatternInfo->getBottomLeft());
    
    std::vector<int> alignmentPatternCenters = provisionalVersion->getAlignmentPatternCenters();
    if (alignmentPatternCenters.size() == 3)
    {
        float v1 = alignmentPatternCenters[0] + 0.5;
        float v2 = alignmentPatternCenters[1] + 0.5;
        // float v3 = alignmentPatternCenters[2] + 0.5;
        
        // 0
        float x = (topLeft->getX() + bottomLeft->getX()) / 2.0;
        float y = (topLeft->getY() + bottomLeft->getY()) / 2.0;
        
        Ref<AlignmentPattern> alignment = findPossibleAlignment(dimension, module_size, x, y);
        if (alignment != NULL)
        {
            pts_src.push_back(cv::Point2f(v1, v2));
            pts_dst.push_back(cv::Point2f(alignment->getX(), alignment->getY()));
        }
        
        // 1
        x = (topLeft->getX() + topRight->getX()) / 2.0;
        y = (topLeft->getY() + topRight->getY()) / 2.0;
        
        alignment = findPossibleAlignment(dimension, module_size, x, y);
        if (alignment != NULL)
        {
            pts_src.push_back(cv::Point2f(v2, v1));
            pts_dst.push_back(cv::Point2f(alignment->getX(), alignment->getY()));
        }
        
        // 2
        x = (bottomLeft->getX() + topRight->getX()) / 2.0;
        y = (bottomLeft->getY() + topRight->getY()) / 2.0;
        
        alignment = findPossibleAlignment(dimension, module_size, x, y);
        if (alignment != NULL)
        {
            pts_src.push_back(cv::Point2f(v2, v2));
            pts_dst.push_back(cv::Point2f(alignment->getX(), alignment->getY()));
        }
    }
    else if (alignmentPatternCenters.size() == 4)
    {
        float v1 = alignmentPatternCenters[0] + 0.5;
        float v2 = alignmentPatternCenters[1] + 0.5;
        float v3 = alignmentPatternCenters[2] + 0.5;
        // float v4 = alignmentPatternCenters[3] + 0.5;
        
        // 0
        float x = (2 * topLeft->getX() + bottomLeft->getX()) / 3.0;
        float y = (2 * topLeft->getY() + bottomLeft->getY()) / 3.0;
        
        Ref<AlignmentPattern> alignment = findPossibleAlignment(dimension, module_size, x, y);
        if (alignment != NULL)
        {
            pts_src.push_back(cv::Point2f(v1, v2));
            pts_dst.push_back(cv::Point2f(alignment->getX(), alignment->getY()));
        }
        
        // 1
        x = (topLeft->getX() + 2 * bottomLeft->getX()) / 3.0;
        y = (topLeft->getY() + 2 * bottomLeft->getY()) / 3.0;
        
        alignment = findPossibleAlignment(dimension, module_size, x, y);
        if (alignment != NULL)
        {
            pts_src.push_back(cv::Point2f(v1, v3));
            pts_dst.push_back(cv::Point2f(alignment->getX(), alignment->getY()));
        }
        
        // 2
        x = (2 * topLeft->getX() + topRight->getX()) / 3.0;
        y = (2 * topLeft->getY() + topRight->getY()) / 3.0;
        
        alignment = findPossibleAlignment(dimension, module_size, x, y);
        if (alignment != NULL)
        {
            pts_src.push_back(cv::Point2f(v2, v1));
            pts_dst.push_back(cv::Point2f(alignment->getX(), alignment->getY()));
        }
        
        // 3
        x = (topLeft->getX() + 2 * topRight->getX()) / 3.0;
        y = (topLeft->getY() + 2 * topRight->getY()) / 3.0;
        
        alignment = findPossibleAlignment(dimension, module_size, x, y);
        if (alignment != NULL)
        {
            pts_src.push_back(cv::Point2f(v3, v1));
            pts_dst.push_back(cv::Point2f(alignment->getX(), alignment->getY()));
        }
        
        // 6
        x = (2 * topRight->getX() + bottomLeft->getX()) / 3.0;
        y = (2 * topRight->getY() + bottomLeft->getY()) / 3.0;
        
        alignment = findPossibleAlignment(dimension, module_size, x, y);
        if (alignment != NULL)
        {
            pts_src.push_back(cv::Point2f(v3, v2));
            pts_dst.push_back(cv::Point2f(alignment->getX(), alignment->getY()));
        }
        
        // 7
        x = (topRight->getX() + 2 * bottomLeft->getX()) / 3.0;
        y = (topRight->getY() + 2 * bottomLeft->getY()) / 3.0;
        
        alignment = findPossibleAlignment(dimension, module_size, x, y);
        if (alignment != NULL)
        {
            pts_src.push_back(cv::Point2f(v2, v3));
            pts_dst.push_back(cv::Point2f(alignment->getX(), alignment->getY()));
        }
    }
    else
    {
        return -1;
    }
    
    return 0;
}

int Detector::getFlexibleAlignmentPoints(size_t patternIdx, size_t alignmentIdx, int dimension, float module_size, std::vector<cv::Point2f>& pts_src, std::vector<cv::Point2f>& pts_dst) {
    ErrorHandler err_handler;
    Version *provisionalVersion = Version::getProvisionalVersionForDimension(dimension, err_handler);
    if (err_handler.ErrCode() != 0)
        return -1;
    
    if (patternIdx >= possiblePatternResults_.size()) return -1;
    if (alignmentIdx >= possiblePatternResults_[patternIdx]->possibleAlignmentPatterns.size()) return -1;
    
    float alignment_x = possiblePatternResults_[patternIdx]->possibleAlignmentPatterns[alignmentIdx]->getX();
    float alignment_y = possiblePatternResults_[patternIdx]->possibleAlignmentPatterns[alignmentIdx]->getY();
    if (alignment_x == 0 && alignment_y == 0) return -1;
    
    pts_src.clear();
    pts_dst.clear();
    
    Ref<FinderPattern> topLeft(possiblePatternResults_[patternIdx]->finderPatternInfo->getTopLeft());
    Ref<FinderPattern> topRight(possiblePatternResults_[patternIdx]->finderPatternInfo->getTopRight());
    Ref<FinderPattern> bottomLeft(possiblePatternResults_[patternIdx]->finderPatternInfo->getBottomLeft());
    
    std::vector<int> alignmentPatternCenters = provisionalVersion->getAlignmentPatternCenters();
    if (alignmentPatternCenters.size() == 3)
    {
        // float v1 = alignmentPatternCenters[0] + 0.5;
        float v2 = alignmentPatternCenters[1] + 0.5;
        float v3 = alignmentPatternCenters[2] + 0.5;
        
        // 3
        float x = (bottomLeft->getX() + alignment_x) / 2.0;
        float y = (bottomLeft->getY() + alignment_y) / 2.0;
        
        Ref<AlignmentPattern> alignment = findPossibleAlignment(dimension, module_size, x, y);
        if (alignment != NULL)
        {
            pts_src.push_back(cv::Point2f(v2, v3));
            pts_dst.push_back(cv::Point2f(alignment->getX(), alignment->getY()));
        }
        
        // 4
        x = (topRight->getX() + alignment_x) / 2.0;
        y = (topRight->getY() + alignment_y) / 2.0;
        
        alignment = findPossibleAlignment(dimension, module_size, x, y);
        if (alignment != NULL)
        {
            pts_src.push_back(cv::Point2f(v3, v2));
            pts_dst.push_back(cv::Point2f(alignment->getX(), alignment->getY()));
        }
    }
    else if (alignmentPatternCenters.size() == 4)
    {
        // float v1 = alignmentPatternCenters[0] + 0.5;
        float v2 = alignmentPatternCenters[1] + 0.5;
        float v3 = alignmentPatternCenters[2] + 0.5;
        float v4 = alignmentPatternCenters[3] + 0.5;
        
        // 4
        float x = (2 * topLeft->getX() + alignment_x) / 3.0;
        float y = (2 * topLeft->getY() + alignment_y) / 3.0;
        
        Ref<AlignmentPattern> alignment = findPossibleAlignment(dimension, module_size, x, y);
        if (alignment != NULL)
        {
            pts_src.push_back(cv::Point2f(v2, v2));
            pts_dst.push_back(cv::Point2f(alignment->getX(), alignment->getY()));
        }
        
        // 5
        x = (topLeft->getX() + 2 * alignment_x) / 3.0;
        y = (topLeft->getY() + 2 * alignment_y) / 3.0;
        
        alignment = findPossibleAlignment(dimension, module_size, x, y);
        if (alignment != NULL)
        {
            pts_src.push_back(cv::Point2f(v3, v3));
            pts_dst.push_back(cv::Point2f(alignment->getX(), alignment->getY()));
        }
        
        // 8
        x = (2 * bottomLeft->getX() + alignment_x) / 3.0;
        y = (2 * bottomLeft->getY() + alignment_y) / 3.0;
        
        alignment = findPossibleAlignment(dimension, module_size, x, y);
        if (alignment != NULL)
        {
            pts_src.push_back(cv::Point2f(v2, v4));
            pts_dst.push_back(cv::Point2f(alignment->getX(), alignment->getY()));
        }
        
        // 9
        x = (bottomLeft->getX() + 2 * alignment_x) / 3.0;
        y = (bottomLeft->getY() + 2 * alignment_y) / 3.0;
        
        alignment = findPossibleAlignment(dimension, module_size, x, y);
        if (alignment != NULL)
        {
            pts_src.push_back(cv::Point2f(v3, v4));
            pts_dst.push_back(cv::Point2f(alignment->getX(), alignment->getY()));
        }
        
        // 10
        x = (2 * topRight->getX() + alignment_x) / 3.0;
        y = (2 * topRight->getY() + alignment_y) / 3.0;
        
        alignment = findPossibleAlignment(dimension, module_size, x, y);
        if (alignment != NULL)
        {
            pts_src.push_back(cv::Point2f(v4, v2));
            pts_dst.push_back(cv::Point2f(alignment->getX(), alignment->getY()));
        }
        
        // 11
        x = (topRight->getX() + 2 * alignment_x) / 3.0;
        y = (topRight->getY() + 2 * alignment_y) / 3.0;
        
        alignment = findPossibleAlignment(dimension, module_size, x, y);
        if (alignment != NULL)
        {
            pts_src.push_back(cv::Point2f(v4, v3));
            pts_dst.push_back(cv::Point2f(alignment->getX(), alignment->getY()));
        }
    }
    else
    {
        return -1;
    }
    return 0;
}

int Detector::getPatternPoints(size_t patternIdx, size_t alignmentIdx, int dimension, std::vector<cv::Point2f>& pts_src, std::vector<cv::Point2f>& pts_dst){
    if (patternIdx >= possiblePatternResults_.size()) return -1;
    
    pts_src.clear();
    pts_dst.clear();
    
    Ref<FinderPattern> topLeft(possiblePatternResults_[patternIdx]->finderPatternInfo->getTopLeft());
    Ref<FinderPattern> topRight(possiblePatternResults_[patternIdx]->finderPatternInfo->getTopRight());
    Ref<FinderPattern> bottomLeft(possiblePatternResults_[patternIdx]->finderPatternInfo->getBottomLeft());
    
    float alignment_x = 0, alignment_y = 0;
    if (alignmentIdx < possiblePatternResults_[patternIdx]->possibleAlignmentPatterns.size()){
        alignment_x = possiblePatternResults_[patternIdx]->possibleAlignmentPatterns[alignmentIdx]->getX();
        alignment_y = possiblePatternResults_[patternIdx]->possibleAlignmentPatterns[alignmentIdx]->getY();
    }
    
    // top left
    pts_src.push_back(cv::Point2f(3.5, 3.5));
    pts_dst.push_back(cv::Point2f(topLeft->getX(), topLeft->getY()));
    
    // top right
    pts_src.push_back(cv::Point2f(dimension - 3.5, 3.5));
    pts_dst.push_back(cv::Point2f(topRight->getX(), topRight->getY()));
    
    // bottom left
    pts_src.push_back(cv::Point2f(3.5, dimension - 3.5));
    pts_dst.push_back(cv::Point2f(bottomLeft->getX(), bottomLeft->getY()));
    
    // alignment
    if (alignment_x != 0 && alignment_y != 0)
    {
        pts_src.push_back(cv::Point2f(dimension - 6.5, dimension - 6.5));
        pts_dst.push_back(cv::Point2f(alignment_x, alignment_y));
    }
    
    return 0;
}

int Detector::getCornerPoints(size_t patternIdx, int dimension, float module_size, std::vector<cv::Point2f>& pts_src, std::vector<cv::Point2f>& pts_dst, bool pure_code) {
    if (patternIdx >= possiblePatternResults_.size()) return -1;
    
    pts_src.clear();
    pts_dst.clear();
    
    Ref<FinderPattern> topLeft(possiblePatternResults_[patternIdx]->finderPatternInfo->getTopLeft());
    Ref<FinderPattern> topRight(possiblePatternResults_[patternIdx]->finderPatternInfo->getTopRight());
    Ref<FinderPattern> bottomLeft(possiblePatternResults_[patternIdx]->finderPatternInfo->getBottomLeft());
    
    // calc oritation
    float k = (topRight->getY() - bottomLeft->getY()) / (topRight->getX() - bottomLeft->getX());
    int ori = 0;
    if (k < 0 && topRight->getX() > bottomLeft->getX() && topRight->getY() < bottomLeft->getY())
    {
        ori = 0;
    }
    else if (k > 0 && topRight->getX() < bottomLeft->getX() && topRight->getY() < bottomLeft->getY())
    {
        ori = 1;
    }
    else if (k < 0 && topRight->getX() < bottomLeft->getX() && topRight->getY() > bottomLeft->getY())
    {
        ori = 2;
    }
    else if (k > 0 && topRight->getX() >bottomLeft->getX() && topRight->getY() > bottomLeft->getY())
    {
        ori = 3;
    }
    else
    {
        ori = 0;
    }
    
    float offset = 0.1;
    float offset1 = 7 - offset;
    float temp_offset = 0.15;
    float temp_offset1 = 7 - temp_offset;
    
    std::vector<float> offset_list = {0.1, 0.15, 0.05};
    std::vector<float> offset1_list = {6.9, 6.85, 6.95};
    
    std::vector<cv::Point2f> temp_pts_src;
    std::vector<std::vector<cv::Point2f>> pts_src_list;
    pts_src_list.resize(3);
    
    // top left
    std::vector<cv::Point2f> top_left_points;
    if (locatePatternRect(topLeft->getX(), topLeft->getY(), top_left_points, module_size, ori) == 0)
    {
        if (top_left_points.size() == 4)
        {
            // 0
            pts_src.push_back(cv::Point2f(offset, offset));
            pts_dst.push_back(cv::Point2f(top_left_points[0].x, top_left_points[0].y));
            temp_pts_src.push_back(cv::Point2f(temp_offset, temp_offset));
            for (size_t i = 0; i < offset_list.size(); i++) {
                pts_src_list[i].push_back(cv::Point2f(offset_list[i], offset_list[i]));
            }
            
            // 1
            pts_src.push_back(cv::Point2f(offset1, offset));
            pts_dst.push_back(cv::Point2f(top_left_points[3].x, top_left_points[3].y));
            temp_pts_src.push_back(cv::Point2f(temp_offset1, temp_offset));
            for (size_t i = 0; i < offset_list.size(); i++) {
                pts_src_list[i].push_back(cv::Point2f(offset1_list[i], offset_list[i]));
            }
            
            // 2
            pts_src.push_back(cv::Point2f(offset, offset1));
            pts_dst.push_back(cv::Point2f(top_left_points[2].x, top_left_points[2].y));
            temp_pts_src.push_back(cv::Point2f(temp_offset, temp_offset1));
            for (size_t i = 0; i < offset_list.size(); i++) {
                pts_src_list[i].push_back(cv::Point2f(offset_list[i], offset1_list[i]));
            }
            
            // 3
            pts_src.push_back(cv::Point2f(offset1, offset1));
            pts_dst.push_back(cv::Point2f(top_left_points[1].x, top_left_points[1].y));
            temp_pts_src.push_back(cv::Point2f(temp_offset1, temp_offset1));
            for (size_t i = 0; i < offset_list.size(); i++) {
                pts_src_list[i].push_back(cv::Point2f(offset1_list[i], offset1_list[i]));
            }
        }
    }
    else if (pure_code)
    {
        if (ori == 0)
        {
            ArrayRef<int> top_left_black = image_->getTopLeftOnBitNew();
            pts_src.push_back(cv::Point2f(0.01, 0.01));
            pts_dst.push_back(cv::Point2f(top_left_black[0], top_left_black[1]));
        }
    }
    
    // top right
    std::vector<cv::Point2f> top_right_points;
    if (locatePatternRect(topRight->getX(), topRight->getY(), top_right_points, module_size, ori) == 0)
    {
        if (top_right_points.size() == 4)
        {
            // 0
            pts_src.push_back(cv::Point2f(dimension - offset1, offset));
            pts_dst.push_back(cv::Point2f(top_right_points[0].x, top_right_points[0].y));
            temp_pts_src.push_back(cv::Point2f(dimension - temp_offset1, temp_offset));
            for (size_t i = 0; i < offset_list.size(); i++) {
                pts_src_list[i].push_back(cv::Point2f(dimension - offset1_list[i], offset_list[i]));
            }
            
            // 1
            pts_src.push_back(cv::Point2f(dimension - offset, offset1));
            pts_dst.push_back(cv::Point2f(top_right_points[1].x, top_right_points[1].y));
            temp_pts_src.push_back(cv::Point2f(dimension - temp_offset, temp_offset1));
            for (size_t i = 0; i < offset_list.size(); i++) {
                pts_src_list[i].push_back(cv::Point2f(dimension - offset_list[i], offset1_list[i]));
            }
            
            // 2
            pts_src.push_back(cv::Point2f(dimension - offset, offset));
            pts_dst.push_back(cv::Point2f(top_right_points[3].x, top_right_points[3].y));
            temp_pts_src.push_back(cv::Point2f(dimension - temp_offset, temp_offset));
            for (size_t i = 0; i < offset_list.size(); i++) {
                pts_src_list[i].push_back(cv::Point2f(dimension - offset_list[i], offset_list[i]));
            }
            
            // 3
            pts_src.push_back(cv::Point2f(dimension - offset1, offset1));
            pts_dst.push_back(cv::Point2f(top_right_points[2].x, top_right_points[2].y));
            temp_pts_src.push_back(cv::Point2f(dimension - temp_offset1, temp_offset1));
            for (size_t i = 0; i < offset_list.size(); i++) {
                pts_src_list[i].push_back(cv::Point2f(dimension - offset1_list[i], offset1_list[i]));
            }
        }
    }
    else if (pure_code)
    {
        if (ori == 0)
        {
            ArrayRef<int> top_right_black = image_->getTopRightOnBitNew();
            pts_src.push_back(cv::Point2f(dimension - 0.01, 0.01));
            pts_dst.push_back(cv::Point2f(top_right_black[0], top_right_black[1]));
        }
    }
    
    // bottom left
    std::vector<cv::Point2f> bottom_left_points;
    if (locatePatternRect(bottomLeft->getX(), bottomLeft->getY(), bottom_left_points, module_size, ori) == 0)
    {
        if (bottom_left_points.size() == 4)
        {
            // 0
            pts_src.push_back(cv::Point2f(offset, dimension-offset1));
            pts_dst.push_back(cv::Point2f(bottom_left_points[0].x, bottom_left_points[0].y));
            temp_pts_src.push_back(cv::Point2f(temp_offset, dimension-temp_offset1));
            for (size_t i = 0; i < offset_list.size(); i++) {
                pts_src_list[i].push_back(cv::Point2f(offset_list[i], dimension - offset1_list[i]));
            }
            
            // 1
            pts_src.push_back(cv::Point2f(offset1, dimension-offset));
            pts_dst.push_back(cv::Point2f(bottom_left_points[1].x, bottom_left_points[1].y));
            temp_pts_src.push_back(cv::Point2f(temp_offset1, dimension-temp_offset));
            for (size_t i = 0; i < offset_list.size(); i++) {
                pts_src_list[i].push_back(cv::Point2f(offset1_list[i], dimension - offset_list[i]));
            }
            
            // 2
            pts_src.push_back(cv::Point2f(offset, dimension-offset));
            pts_dst.push_back(cv::Point2f(bottom_left_points[2].x, bottom_left_points[2].y));
            temp_pts_src.push_back(cv::Point2f(temp_offset, dimension-temp_offset));
            for (size_t i = 0; i < offset_list.size(); i++) {
                pts_src_list[i].push_back(cv::Point2f(offset_list[i], dimension - offset_list[i]));
            }
            
            // 3
            pts_src.push_back(cv::Point2f(offset1, dimension-offset1));
            pts_dst.push_back(cv::Point2f(bottom_left_points[3].x, bottom_left_points[3].y));
            temp_pts_src.push_back(cv::Point2f(temp_offset1, dimension-temp_offset1));
            for (size_t i = 0; i < offset_list.size(); i++) {
                pts_src_list[i].push_back(cv::Point2f(offset1_list[i], dimension - offset1_list[i]));
            }
        }
    }
    else if (pure_code)
    {
        if (ori == 0)
        {
            ArrayRef<int> bottom_left_black = image_->getBottomLeftOnBitNew();
            pts_src.push_back(cv::Point2f(0.01, dimension-0.01));
            pts_dst.push_back(cv::Point2f(bottom_left_black[0], bottom_left_black[1]));
        }
    }
    
    pts_src.clear();
    for (size_t i = 0; i < pts_src_list.size(); i++) {
        pts_src.insert(pts_src.end(), pts_src_list[i].begin(), pts_src_list[i].end());
    }
    
    return 0;
}

int Detector::getPossibleAlignmentPointsMore(size_t patternIdx, size_t alignmentIdx, int dimension, float module_size, int center_nums)
{
    if (patternIdx >= possiblePatternResults_.size()) return -1;
    if (alignmentIdx >= possiblePatternResults_[patternIdx]->possibleAlignmentPatterns.size()) return -1;
    if (possiblePatternResults_[patternIdx]->moreAlignmentPoints.size() !=  possiblePatternResults_[patternIdx]->possibleAlignmentPatterns.size()) {
        possiblePatternResults_[patternIdx]->moreAlignmentPoints.resize(possiblePatternResults_[patternIdx]->possibleAlignmentPatterns.size());
    }
    
    Ref<FinderPattern> topLeft(possiblePatternResults_[patternIdx]->finderPatternInfo->getTopLeft());
    Ref<FinderPattern> topRight(possiblePatternResults_[patternIdx]->finderPatternInfo->getTopRight());
    Ref<FinderPattern> bottomLeft(possiblePatternResults_[patternIdx]->finderPatternInfo->getBottomLeft());
    
    float alignment_x = possiblePatternResults_[patternIdx]->possibleAlignmentPatterns[alignmentIdx]->getX();
    float alignment_y = possiblePatternResults_[patternIdx]->possibleAlignmentPatterns[alignmentIdx]->getY();
    
    if (center_nums == 3)
    {
        if (possiblePatternResults_[patternIdx]->moreAlignmentPoints[alignmentIdx].size() == 0)
            possiblePatternResults_[patternIdx]->moreAlignmentPoints[alignmentIdx].resize(5, cv::Point2f(-1, -1));
        else if (possiblePatternResults_[patternIdx]->moreAlignmentPoints[alignmentIdx].size() != 5)
            possiblePatternResults_[patternIdx]->moreAlignmentPoints[alignmentIdx].resize(5, cv::Point2f(-1, -1));
        
        for (size_t i = 0; i < possiblePatternResults_[patternIdx]->moreAlignmentPoints[alignmentIdx].size(); i++) {
            if (possiblePatternResults_[patternIdx]->moreAlignmentPoints[alignmentIdx][i].x == -1 &&
                possiblePatternResults_[patternIdx]->moreAlignmentPoints[alignmentIdx][i].y == -1) {
                float x = -1 , y = -1;
                if (i == 0)
                {
                    x = (topLeft->getX() + bottomLeft->getX()) / 2.0;
                    y = (topLeft->getY() + bottomLeft->getY()) / 2.0;
                }
                else if (i == 1)
                {
                    x = (topLeft->getX() + topRight->getX()) / 2.0;
                    y = (topLeft->getY() + topRight->getY()) / 2.0;
                }
                else if (i == 2)
                {
                    x = (bottomLeft->getX() + topRight->getX()) / 2.0;
                    y = (bottomLeft->getY() + topRight->getY()) / 2.0;
                }
                else if (i == 3)
                {
                    if (!(alignment_x == 0 && alignment_y == 0))
                    {
                        x = (bottomLeft->getX() + alignment_x) / 2.0;
                        y = (bottomLeft->getY() + alignment_y) / 2.0;
                    }
                }
                else
                {
                    if (!(alignment_x == 0 && alignment_y == 0))
                    {
                        x = (topRight->getX() + alignment_x) / 2.0;
                        y = (topRight->getY() + alignment_y) / 2.0;
                    }
                }
                if (x != -1 && y != -1)
                {
                    Ref<AlignmentPattern> alignment = findPossibleAlignment(dimension, module_size, x, y);
                    if (alignment != NULL)
                    {
                        possiblePatternResults_[patternIdx]->moreAlignmentPoints[alignmentIdx][i].x = alignment->getX();
                        possiblePatternResults_[patternIdx]->moreAlignmentPoints[alignmentIdx][i].y = alignment->getY();
                    }
                }
            }
        }
    }
    else if (center_nums == 4)
    {
        if (possiblePatternResults_[patternIdx]->moreAlignmentPoints[alignmentIdx].size() == 0)
            possiblePatternResults_[patternIdx]->moreAlignmentPoints[alignmentIdx].resize(12, cv::Point2f(-1, -1));
        else if (possiblePatternResults_[patternIdx]->moreAlignmentPoints[alignmentIdx].size() != 12)
            possiblePatternResults_[patternIdx]->moreAlignmentPoints[alignmentIdx].resize(12, cv::Point2f(-1, -1));
        
        for (size_t i = 0; i < possiblePatternResults_[patternIdx]->moreAlignmentPoints[alignmentIdx].size(); i++) {
            if (possiblePatternResults_[patternIdx]->moreAlignmentPoints[alignmentIdx][i].x == -1 &&
                possiblePatternResults_[patternIdx]->moreAlignmentPoints[alignmentIdx][i].y == -1)
            {
                float x = -1 , y = -1;
                if (i == 0)
                {
                    x = (2 * topLeft->getX() + bottomLeft->getX()) / 3.0;
                    y = (2 * topLeft->getY() + bottomLeft->getY()) / 3.0;
                }
                else if (i == 1)
                {
                    x = (topLeft->getX() + 2 * bottomLeft->getX()) / 3.0;
                    y = (topLeft->getY() + 2 * bottomLeft->getY()) / 3.0;
                }
                else if (i == 2)
                {
                    x = (2 * topLeft->getX() + topRight->getX()) / 3.0;
                    y = (2 * topLeft->getY() + topRight->getY()) / 3.0;
                }
                else if (i == 3)
                {
                    x = (topLeft->getX() + 2 * topRight->getX()) / 3.0;
                    y = (topLeft->getY() + 2 * topRight->getY()) / 3.0;
                }
                else if (i == 4)
                {
                    if (alignment_x != 0 || alignment_y != 0)
                    {
                        x = (2 * topLeft->getX() + alignment_x) / 3.0;
                        y = (2 * topLeft->getY() + alignment_y) / 3.0;
                    }
                }
                else if (i == 5)
                {
                    if (alignment_x != 0 || alignment_y != 0)
                    {
                        x = (topLeft->getX() + 2 * alignment_x) / 3.0;
                        y = (topLeft->getY() + 2 * alignment_y) / 3.0;
                    }
                }
                else if (i == 6)
                {
                    x = (2 * topRight->getX() + bottomLeft->getX()) / 3.0;
                    y = (2 * topRight->getY() + bottomLeft->getY()) / 3.0;
                }
                else if (i == 7)
                {
                    x = (topRight->getX() + 2 * bottomLeft->getX()) / 3.0;
                    y = (topRight->getY() + 2 * bottomLeft->getY()) / 3.0;
                }
                else if (i == 8)
                {
                    if (alignment_x != 0 || alignment_y != 0)
                    {
                        x = (2 * bottomLeft->getX() + alignment_x) / 3.0;
                        y = (2 * bottomLeft->getY() + alignment_y) / 3.0;
                    }
                }
                else if (i == 9)
                {
                    if (alignment_x != 0 || alignment_y != 0)
                    {
                        x = (bottomLeft->getX() + 2 * alignment_x) / 3.0;
                        y = (bottomLeft->getY() + 2 * alignment_y) / 3.0;
                    }
                }
                else if (i == 10)
                {
                    if (alignment_x != 0 || alignment_y != 0)
                    {
                        x = (2 * topRight->getX() + alignment_x) / 3.0;
                        y = (2 * topRight->getY() + alignment_y) / 3.0;
                    }
                }
                else
                {
                    if (!(alignment_x == 0 && alignment_y == 0))
                    {
                        x = (topRight->getX() + 2 * alignment_x) / 3.0;
                        y = (topRight->getY() + 2 * alignment_y) / 3.0;
                    }
                }
                if (x != -1 && y != -1)
                {
                    Ref<AlignmentPattern> alignment = findPossibleAlignment(dimension, module_size, x, y);
                    if (alignment != NULL)
                    {
                        possiblePatternResults_[patternIdx]->moreAlignmentPoints[alignmentIdx][i].x = alignment->getX();
                        possiblePatternResults_[patternIdx]->moreAlignmentPoints[alignmentIdx][i].y = alignment->getY();
                    }
                }
            }
        }
    }
    else
        possiblePatternResults_[patternIdx]->moreAlignmentPoints[alignmentIdx].clear();
    
    return 0;
}

int Detector::getPossibleAlignmentPatterRect(float x, float y) {
    int width = image_->getWidth();
    int height = image_->getHeight();
    
    float min_x = fmax(0, x - width / 10.0);
    float max_x = fmin(x + width / 10.0, width - 1);
    float min_y = fmax(0, y - height / 10.0);
    float max_y = fmin(y + height / 10.0, height - 1);
    
    std::queue<cv::Point2f> point_list, point_list_1;
    point_list.push(cv::Point2f(x, y));
    
    cv::Mat flag_mat(height, width, CV_8UC1, cv::Scalar(0));
    
    while (!point_list.empty()) {
        cv::Point2f pt = point_list.front();
        point_list.pop();
        
        int x_ = static_cast<int>(pt.x), y_ = static_cast<int>(pt.y);
        if (flag_mat.at<uchar>(y_, x_) == 255) continue;
        flag_mat.at<uchar>(y_, x_) = 255;
        
        bool flag1 = false, flag2 = false, flag3 = false, flag4 = false;
        if ((x_ - 1) >= min_x && image_->get(x_ - 1, y_))
        {
            if (flag_mat.at<uchar>(y_, x_ - 1) == 0)
                point_list.push(cv::Point2f(pt.x - 1, pt.y));
            flag1 = true;
        }
        if ((x_ + 1) <= max_x && image_->get(x_ + 1, y_))
        {
            if (flag_mat.at<uchar>(y_, x_ + 1) == 0)
                point_list.push(cv::Point2f(pt.x + 1, pt.y));
            flag2 = true;
        }
        if ((y_ - 1) >= min_y && image_->get(x_, y_ - 1))
        {
            if (flag_mat.at<uchar>(y_ - 1, x_) == 0)
            point_list.push(cv::Point2f(pt.x, pt.y - 1));
        flag3 = true;
        }
        if ((y_ + 1) <= max_y && image_->get(x_, y_ + 1))
        {
            if (flag_mat.at<uchar>(y_ + 1, x_) == 0)
                point_list.push(cv::Point2f(pt.x, pt.y + 1));
            flag4 = true;
        }
        if (!(flag1 && flag2 && flag3 && flag4))
        {
            point_list_1.push(cv::Point2f(pt.x, pt.y));
        }
    }
    
    if (point_list_1.empty()) return -1;
    
    while (!point_list_1.empty()) {
        cv::Point2f pt = point_list_1.front();
        point_list_1.pop();
        
        int x_ = static_cast<int>(pt.x), y_ = static_cast<int>(pt.y);
        if ((x_ - 1) >= min_x && !image_->get(x_ - 1, y_))
        {
            if (flag_mat.at<uchar>(y_, x_ - 1) == 0)
                point_list.push(cv::Point2f(pt.x - 1, pt.y));
        }
        if ((x_ + 1) <= max_x && !image_->get(x_ + 1, y_))
        {
            if (flag_mat.at<uchar>(y_, x_ + 1) == 0)
                point_list.push(cv::Point2f(pt.x + 1, pt.y));
        }
        if ((y_ - 1) >= min_y && !image_->get(x_, y_ - 1))
        {
            if (flag_mat.at<uchar>(y_ - 1, x_) == 0)
                point_list.push(cv::Point2f(pt.x, pt.y - 1));
        }
        if ((y_ + 1) <= max_y && !image_->get(x_, y_ + 1))
        {
            if (flag_mat.at<uchar>(y_ + 1, x_) == 0)
                point_list.push(cv::Point2f(pt.x, pt.y + 1));
        }
    }
    
    if (point_list.empty()) return -1;
    
    std::vector<cv::Point2f> corner_list;
    while (!point_list.empty()) {
        cv::Point2f pt = point_list.front();
        point_list.pop();
        
        int x_ = static_cast<int>(pt.x), y_ = static_cast<int>(pt.y);
        if (flag_mat.at<uchar>(y_, x_) == 255) continue;
        flag_mat.at<uchar>(y_, x_) = 255;
        
        bool flag1 = false, flag2 = false, flag3 = false, flag4 = false;
        if ((x_ - 1) >= min_x && !image_->get(x_ - 1, y_))
        {
            if (flag_mat.at<uchar>(y_, x_ - 1) == 0)
                point_list.push(cv::Point2f(pt.x - 1, pt.y));
            flag1 = true;
        }
        if ((x_ + 1) <= max_x && !image_->get(x_ + 1, y_))
        {
            if (flag_mat.at<uchar>(y_, x_ + 1) == 0)
                point_list.push(cv::Point2f(pt.x + 1, pt.y));
            flag2 = true;
        }
        if ((y_ - 1) >= min_y && !image_->get(x_, y_ - 1))
        {
            if (flag_mat.at<uchar>(y_ - 1, x_) == 0)
                point_list.push(cv::Point2f(pt.x, pt.y - 1));
            flag3 = true;
        }
        if ((y_ + 1) <= max_y && !image_->get(x_, y_ + 1))
        {
            if (flag_mat.at<uchar>(y_ + 1, x_) == 0)
                point_list.push(cv::Point2f(pt.x, pt.y + 1));
            flag4 = true;
        }
        if (!(flag1 && flag2 && flag3 && flag4))
        {
            corner_list.push_back(cv::Point2f(pt.x, pt.y));
        }
    }
    
    if (corner_list.size() == 0) return -1;
    
    // int min_sum_idx = -1, max_sum_idx = -1, min_diff_idx = -1, max_diff_idx = -1;
    float min_sum = INT_MAX, max_sum = 0, min_diff = INT_MAX, max_diff = -INT_MAX;
    for (size_t i = 0; i < corner_list.size(); i++) {
        float sum = corner_list[i].x + corner_list[i].y;
        float diff = corner_list[i].x - corner_list[i].y;
        if (sum  < min_sum)
        {
            // min_sum_idx = i;
            min_sum = sum;
        }
        if (sum > max_sum)
        {
            // max_sum_idx = i;
            max_sum = sum;
        }
        if (diff < min_diff)
        {
            // min_diff_idx = i;
            min_diff = diff;
        }
        if (diff > max_diff)
        {
            // max_diff_idx = i;
            max_diff = diff;
        }
    }
    // cv::Point2f pt1 = corner_list[min_sum_idx];
    // cv::Point2f pt2 = corner_list[max_sum_idx];
    // cv::Point2f pt3 = corner_list[min_diff_idx];
    // cv::Point2f pt4 = corner_list[max_diff_idx];
    
    return 0;
}

int Detector::getPossiblePatternRect(size_t idx, float module_size, int ori) {
    if (idx >= possiblePatternResults_.size()) return -1;
    
    int ret = 0;
    Ref<FinderPatternInfo> patterInfo = possiblePatternResults_[idx]->finderPatternInfo;
    
    // If it is first time to get, process it now
    if (possiblePatternResults_[idx]->topLeftPoints.size() == 0)
    {
        ret = locatePatternRect(patterInfo->getTopLeft()->getX(), patterInfo->getTopLeft()->getY(), possiblePatternResults_[idx]->topLeftPoints, module_size, ori);
        if (ret != 0) possiblePatternResults_[idx]->topLeftPoints.clear();
    }
    if (possiblePatternResults_[idx]->topRightPoints.size() == 0)
    {
        ret = locatePatternRect(patterInfo->getTopRight()->getX(), patterInfo->getTopRight()->getY(), possiblePatternResults_[idx]->topRightPoints, module_size, ori);
        if (ret != 0) possiblePatternResults_[idx]->topRightPoints.clear();
    }
    if (possiblePatternResults_[idx]->bottomLeftPoints.size() == 0)
    {
        ret = locatePatternRect(patterInfo->getBottomLeft()->getX(), patterInfo->getBottomLeft()->getY(), possiblePatternResults_[idx]->bottomLeftPoints, module_size, ori);
        if (ret != 0) possiblePatternResults_[idx]->bottomLeftPoints.clear();
    }
    
    return 0;
}

int Detector::getPossibleAlignmentCount(size_t idx, DecodeHints& hints)
{
    if (idx >= possiblePatternResults_.size()) return -1;
    
    ErrorHandler err_handler;
    // If it is first time to get, process it now
    if (possiblePatternResults_[idx]->possibleAlignmentPatterns.size() == 0)
    {
        Ref<PatternResult> result = processFinderPatternInfo(hints, possiblePatternResults_[idx]->finderPatternInfo, err_handler);
        if (err_handler.ErrCode())
            return -1;
        
        possiblePatternResults_[idx] = result;
    }
    
    return possiblePatternResults_[idx]->possibleAlignmentPatterns.size();
}

Ref<AlignmentPattern> Detector::findPossibleAlignment(int dimension, float moduleSize, float x, float y) {
    int tryFindRange =  dimension / 4;
    
    Ref<AlignmentPattern> estAP = getNearestAlignmentPattern(tryFindRange, moduleSize, x, y);
    
    return estAP;
}

Ref<DetectorResult> Detector::getResultViaAlignmentMore(DecodeHints const& hints, size_t patternIdx, size_t alignmentIndex, ArrayRef< Ref<ResultPoint> > points, int dimension, float module_size, int mode, ErrorHandler & err_handler)
{
    (void)hints;
    (void)points;

    if (patternIdx >= possiblePatternResults_.size() || patternIdx < 0)
    {
        return Ref<DetectorResult>(NULL);
    }
    
    // Default is the dimension
    if (dimension <= 0)
    {
        dimension = possiblePatternResults_[patternIdx]->getDimension();
    }
    
    Version *provisionalVersion = Version::getProvisionalVersionForDimension(dimension, err_handler);
    if (err_handler.ErrCode() != 0)
        return Ref<DetectorResult>(NULL);
    
    Ref<FinderPattern> topLeft(possiblePatternResults_[patternIdx]->finderPatternInfo->getTopLeft());
    Ref<FinderPattern> topRight(possiblePatternResults_[patternIdx]->finderPatternInfo->getTopRight());
    Ref<FinderPattern> bottomLeft(possiblePatternResults_[patternIdx]->finderPatternInfo->getBottomLeft());
    
    std::vector<cv::Point2f> pts_src, pts_dst;
    
    if (alignmentIndex < possiblePatternResults_[patternIdx]->possibleAlignmentPatterns.size() && alignmentIndex >= 0)
    {
        if (possiblePatternResults_[patternIdx]->possibleAlignmentPatterns[alignmentIndex]->getX() != 0 ||
            possiblePatternResults_[patternIdx]->possibleAlignmentPatterns[alignmentIndex]->getY() != 0) {
            pts_src.push_back(cv::Point2f(dimension - 6.5, dimension - 6.5));
            pts_dst.push_back(cv::Point2f(possiblePatternResults_[patternIdx]->possibleAlignmentPatterns[alignmentIndex]->getX(), possiblePatternResults_[patternIdx]->possibleAlignmentPatterns[alignmentIndex]->getY()));
            
            // getPossibleAlignmentPatterRect(possiblePatternResults_[patternIdx]->possibleAlignmentPatterns[alignmentIndex]->getX(), possiblePatternResults_[patternIdx]->possibleAlignmentPatterns[alignmentIndex]->getY());
        }
    }
    
    // find corner points
    
    // decide the oritation
    float k = (topRight->getY() - bottomLeft->getY()) / (topRight->getX() - bottomLeft->getX());
    int ori = 0;
    if (k < 0 && topRight->getX() > bottomLeft->getX() && topRight->getY() < bottomLeft->getY())
    {
        ori = 0;
    }
    else if (k > 0 && topRight->getX() < bottomLeft->getX() && topRight->getY() < bottomLeft->getY())
    {
        ori = 1;
    }
    else if (k < 0 && topRight->getX() < bottomLeft->getX() && topRight->getY() > bottomLeft->getY())
    {
        ori = 2;
    }
    else if (k > 0 && topRight->getX() >bottomLeft->getX() && topRight->getY() > bottomLeft->getY())
    {
        ori = 3;
    }
    else
    {
        ori = 0;
    }
    
    int ret = getPossiblePatternRect(patternIdx, module_size, ori);
    if ((mode == 0 || mode == 1) && ret != 0) return Ref<DetectorResult>(NULL);
    if (mode == 0 || mode == 1 || mode == 4 || mode == 5)
    {
        bool flag = false;
        float offset = 0.1;
        float offset1 = 7 - offset;
        if (possiblePatternResults_[patternIdx]->topLeftPoints.size() == 4)
        {
            pts_src.push_back(cv::Point2f(3.5, 3.5));
            pts_dst.push_back(cv::Point2f(topLeft->getX(), topLeft->getY()));
            
            pts_src.push_back(cv::Point2f(offset, offset));
            pts_dst.push_back(cv::Point2f(possiblePatternResults_[patternIdx]->topLeftPoints[0].x, possiblePatternResults_[patternIdx]->topLeftPoints[0].y));
            
            pts_src.push_back(cv::Point2f(offset1, offset));
            pts_dst.push_back(cv::Point2f(possiblePatternResults_[patternIdx]->topLeftPoints[3].x, possiblePatternResults_[patternIdx]->topLeftPoints[3].y));
            
            pts_src.push_back(cv::Point2f(offset, offset1));
            pts_dst.push_back(cv::Point2f(possiblePatternResults_[patternIdx]->topLeftPoints[2].x, possiblePatternResults_[patternIdx]->topLeftPoints[2].y));
            
            pts_src.push_back(cv::Point2f(offset1, offset1));
            pts_dst.push_back(cv::Point2f(possiblePatternResults_[patternIdx]->topLeftPoints[1].x, possiblePatternResults_[patternIdx]->topLeftPoints[1].y));
            
            flag = true;
        }
        if (possiblePatternResults_[patternIdx]->topRightPoints.size() == 4)
        {
            pts_src.push_back(cv::Point2f(dimension - 3.5, 3.5));
            pts_dst.push_back(cv::Point2f(topRight->getX(), topRight->getY()));
            
            pts_src.push_back(cv::Point2f(dimension - offset1, offset));
            pts_dst.push_back(cv::Point2f(possiblePatternResults_[patternIdx]->topRightPoints[0].x, possiblePatternResults_[patternIdx]->topRightPoints[0].y));
            
            pts_src.push_back(cv::Point2f(dimension - offset, offset1));
            pts_dst.push_back(cv::Point2f(possiblePatternResults_[patternIdx]->topRightPoints[1].x, possiblePatternResults_[patternIdx]->topRightPoints[1].y));
            
            pts_src.push_back(cv::Point2f(dimension - offset, offset));
            pts_dst.push_back(cv::Point2f(possiblePatternResults_[patternIdx]->topRightPoints[3].x, possiblePatternResults_[patternIdx]->topRightPoints[3].y));
            
            pts_src.push_back(cv::Point2f(dimension - offset1, offset1));
            pts_dst.push_back(cv::Point2f(possiblePatternResults_[patternIdx]->topRightPoints[2].x, possiblePatternResults_[patternIdx]->topRightPoints[2].y));
            flag = true;
        }
        if (possiblePatternResults_[patternIdx]->bottomLeftPoints.size() == 4)
        {
            pts_src.push_back(cv::Point2f(3.5, dimension - 3.5));
            pts_dst.push_back(cv::Point2f(bottomLeft->getX(), bottomLeft->getY()));
            
            pts_src.push_back(cv::Point2f(offset, dimension-offset1));
            pts_dst.push_back(cv::Point2f(possiblePatternResults_[patternIdx]->bottomLeftPoints[0].x, possiblePatternResults_[patternIdx]->bottomLeftPoints[0].y));
            
            pts_src.push_back(cv::Point2f(offset1, dimension-offset));
            pts_dst.push_back(cv::Point2f(possiblePatternResults_[patternIdx]->bottomLeftPoints[1].x, possiblePatternResults_[patternIdx]->bottomLeftPoints[1].y));
            
            pts_src.push_back(cv::Point2f(offset, dimension-offset));
            pts_dst.push_back(cv::Point2f(possiblePatternResults_[patternIdx]->bottomLeftPoints[2].x, possiblePatternResults_[patternIdx]->bottomLeftPoints[2].y));
            
            pts_src.push_back(cv::Point2f(offset1, dimension-offset1));
            pts_dst.push_back(cv::Point2f(possiblePatternResults_[patternIdx]->bottomLeftPoints[3].x, possiblePatternResults_[patternIdx]->bottomLeftPoints[3].y));
            flag = true;
        }
        if (!flag) return Ref<DetectorResult>();
    }
    
    // find aligments
    if (mode == 2 || mode == 3 || mode == 4 || mode == 5)
    {
        if (mode == 2 || mode == 3)
        {
            if (possiblePatternResults_[patternIdx]->topLeftPoints.size() == 4)
            {
                pts_src.push_back(cv::Point2f(3.5, 3.5));
                pts_dst.push_back(cv::Point2f(topLeft->getX(), topLeft->getY()));
            }
            if (possiblePatternResults_[patternIdx]->topRightPoints.size() == 4)
            {
                pts_src.push_back(cv::Point2f(dimension - 3.5, 3.5));
                pts_dst.push_back(cv::Point2f(topRight->getX(), topRight->getY()));
            }
            if (possiblePatternResults_[patternIdx]->bottomLeftPoints.size() == 4)
            {
                pts_src.push_back(cv::Point2f(3.5, dimension - 3.5));
                pts_dst.push_back(cv::Point2f(bottomLeft->getX(), bottomLeft->getY()));
            }
        }
        
        std::vector<int> alignmentPatternCenters = provisionalVersion->getAlignmentPatternCenters();
        if (getPossibleAlignmentPointsMore(patternIdx, alignmentIndex, dimension, module_size, alignmentPatternCenters.size()) == 0) {
            if (possiblePatternResults_[patternIdx]->moreAlignmentPoints[alignmentIndex].size() > 0)
            {
                if (possiblePatternResults_[patternIdx]->moreAlignmentPoints[alignmentIndex].size() == 5)
                {
                    float v1 = alignmentPatternCenters[0] + 0.5;
                    float v2 = alignmentPatternCenters[1] + 0.5;
                    float v3 = alignmentPatternCenters[2] + 0.5;
                    
                    for (size_t i = 0; i < 5; i++) {
                        float x = possiblePatternResults_[patternIdx]->moreAlignmentPoints[alignmentIndex][i].x;
                        float y = possiblePatternResults_[patternIdx]->moreAlignmentPoints[alignmentIndex][i].y;
                        
                        if (x != -1 && y != -1)
                        {
                            if (i == 0)
                            {
                                pts_src.push_back(cv::Point2f(v1, v2));
                            }
                            else if (i == 1)
                            {
                                pts_src.push_back(cv::Point2f(v2, v1));
                            }
                            else if (i == 2)
                            {
                                pts_src.push_back(cv::Point2f(v2, v2));
                            }
                            else if (i == 3)
                            {
                                pts_src.push_back(cv::Point2f(v2, v3));
                            }
                            else
                            {
                                pts_src.push_back(cv::Point2f(v3, v2));
                            }
                            pts_dst.push_back(cv::Point2f(x, y));
                        }
                    }
                }
                else if (possiblePatternResults_[patternIdx]->moreAlignmentPoints[alignmentIndex].size() == 12)
                {
                    float v1 = alignmentPatternCenters[0] + 0.5;
                    float v2 = alignmentPatternCenters[1] + 0.5;
                    float v3 = alignmentPatternCenters[2] + 0.5;
                    float v4 = alignmentPatternCenters[3] + 0.5;
                    
                    for (size_t i = 0; i < 12; i++) {
                        float x = possiblePatternResults_[patternIdx]->moreAlignmentPoints[alignmentIndex][i].x;
                        float y = possiblePatternResults_[patternIdx]->moreAlignmentPoints[alignmentIndex][i].y;
                        
                        if (x != -1 && y != -1)
                        {
                            if (i == 0)
                            {
                                pts_src.push_back(cv::Point2f(v1, v2));
                            }
                            else if (i == 1)
                            {
                                pts_src.push_back(cv::Point2f(v1, v3));
                            }
                            else if (i == 2)
                            {
                                pts_src.push_back(cv::Point2f(v2, v1));
                            }
                            else if (i == 3)
                            {
                                pts_src.push_back(cv::Point2f(v3, v1));
                            }
                            else if (i == 4)
                            {
                                pts_src.push_back(cv::Point2f(v2, v2));
                            }
                            else if (i == 5)
                            {
                                pts_src.push_back(cv::Point2f(v3, v3));
                            }
                            else if (i == 6)
                            {
                                pts_src.push_back(cv::Point2f(v3, v2));
                            }
                            else if (i == 7)
                            {
                                pts_src.push_back(cv::Point2f(v2, v3));
                            }
                            else if (i == 8)
                            {
                                pts_src.push_back(cv::Point2f(v2, v4));
                            }
                            else if (i == 9)
                            {
                                pts_src.push_back(cv::Point2f(v3, v4));
                            }
                            else if (i == 10)
                            {
                                pts_src.push_back(cv::Point2f(v4, v2));
                            }
                            else
                            {
                                pts_src.push_back(cv::Point2f(v4, v3));
                            }
                            pts_dst.push_back(cv::Point2f(x, y));
                        }
                    }
                }
            }
            else return Ref<DetectorResult>();
        }
        else
            return Ref<DetectorResult>();
    }
    
    int count = static_cast<int>(pts_src.size());
    if (mode == 0 || mode == 2 || mode == 4)  // use homograph to fit points
    {
        if (count < 4) return Ref<DetectorResult>();
        cv::Mat H = cv::findHomography(pts_src, pts_dst, cv::RANSAC);
        
        Ref<BitMatrix> bits(sampleGrid(image_, dimension, H, err_handler));
        if (err_handler.ErrCode())
            Ref<DetectorResult>();
        
        std::vector<cv::Point2f> corners(4);
        corners[0].x = 0.0f; corners[0].y = dimension;
        corners[1].x = 0.0f; corners[1].y = 0.0f;
        corners[2].x = dimension; corners[2].y = 0.0f;
        corners[3].x = dimension; corners[3].y = dimension;
        cv::perspectiveTransform(corners, corners, H);
        
        ArrayRef< Ref<ResultPoint> > corrners(new Array< Ref<ResultPoint> >(4));
        corrners[0].reset(Ref<FinderPattern>(new FinderPattern(corners[0].x, corners[0].y, 0)));
        corrners[1].reset(Ref<FinderPattern>(new FinderPattern(corners[1].x, corners[1].y, 0)));
        corrners[2].reset(Ref<FinderPattern>(new FinderPattern(corners[2].x, corners[2].y, 0)));
        corrners[3].reset(Ref<FinderPattern>(new FinderPattern(corners[3].x, corners[3].y, 0)));
        
        Ref<DetectorResult> result(new DetectorResult(bits, corrners, dimension));
        return result;
    }
    else if (mode == 1 || mode == 3 || mode == 5)  // use map function to fit points
    {
        if (count < 6) return Ref<DetectorResult>();
        
        // correct
        cv::Mat U(count, 6, CV_32F, cv::Scalar(0));
        cv::Mat X(count, 1, CV_32F, cv::Scalar(0));
        cv::Mat Y(count, 1, CV_32F, cv::Scalar(0));
        
        for (int i = 0; i < count; i++) {
            U.at<float>(i, 0) = 1;
            U.at<float>(i, 1) = pts_src[i].x;
            U.at<float>(i, 2) = pts_src[i].y;
            U.at<float>(i, 3) = pts_src[i].x * pts_src[i].x;
            U.at<float>(i, 4) = pts_src[i].y * pts_src[i].y;
            U.at<float>(i, 5) = pts_src[i].x * pts_src[i].y;
            
            X.at<float>(i, 0) = pts_dst[i].x;
            Y.at<float>(i, 0) = pts_dst[i].y;
        }
        
        cv::Mat temp = U.t() * U;
        cv::Mat inv;
        cv::invert(temp, inv);
        cv::Mat K1 = inv * U.t() * X;
        cv::Mat K2 = inv * U.t() * Y;
        float a_x = K1.at<float>(0, 0), b_x = K1.at<float>(1, 0), c_x = K1.at<float>(2, 0), d_x = K1.at<float>(3, 0), e_x = K1.at<float>(4, 0), f_x = K1.at<float>(5, 0);
        float a_y = K2.at<float>(0, 0), b_y = K2.at<float>(1, 0), c_y = K2.at<float>(2, 0), d_y = K2.at<float>(3, 0), e_y = K2.at<float>(4, 0), f_y = K2.at<float>(5, 0);
        
        Ref<BitMatrix> bits(sampleGrid(image_, dimension,
                                       a_x, b_x, c_x, d_x, e_x, f_x,
                                       a_y, b_y, c_y, d_y, e_y, f_y,
                                       err_handler));
        if (err_handler.ErrCode())
            return Ref<DetectorResult>();
        
        std::vector<cv::Point2f> corners(4);
        corners[0].x = 0.0f; corners[0].y = dimension;
        corners[1].x = 0.0f; corners[1].y = 0.0f;
        corners[2].x = dimension; corners[2].y = 0.0f;
        corners[3].x = dimension; corners[3].y = dimension;
        
        for (int i = 0; i < 4; i++) {
            float map_x = a_x + b_x * corners[i].x + c_x * corners[i].y + d_x * corners[i].x * corners[i].x + e_x * corners[i].y * corners[i].y + f_x * corners[i].x * corners[i].y;
            float map_y = a_y + b_y * corners[i].x + c_y * corners[i].y + d_y * corners[i].x * corners[i].x + e_y * corners[i].y * corners[i].y + f_y * corners[i].x * corners[i].y;
            corners[i].x = map_x;
            corners[i].y = map_y;
        }
        
        ArrayRef< Ref<ResultPoint> > corrners(new Array< Ref<ResultPoint> >(4));
        corrners[0].reset(Ref<FinderPattern>(new FinderPattern(corners[0].x, corners[0].y, 0)));
        corrners[1].reset(Ref<FinderPattern>(new FinderPattern(corners[1].x, corners[1].y, 0)));
        corrners[2].reset(Ref<FinderPattern>(new FinderPattern(corners[2].x, corners[2].y, 0)));
        corrners[3].reset(Ref<FinderPattern>(new FinderPattern(corners[3].x, corners[3].y, 0)));
        
        Ref<DetectorResult> result(new DetectorResult(bits, corrners, dimension));
        return result;
    }
    
    return Ref<DetectorResult>();
}

Ref<DetectorResult> Detector::getResultViaAlignment(size_t patternIdx, size_t alignmentIdx, int possibleDimension,  ErrorHandler & err_handler)
{
    if (patternIdx >= possiblePatternResults_.size() || patternIdx < 0)
    {
        return Ref<DetectorResult>(NULL);
    }
    
    if (alignmentIdx >= possiblePatternResults_[patternIdx]->possibleAlignmentPatterns.size() || alignmentIdx < 0)
    {
        return Ref<DetectorResult>(NULL);
    }
    
    // Default is the dimension
    if (possibleDimension <= 0)
    {
        possibleDimension = possiblePatternResults_[patternIdx]->getDimension();
    }
    
    Ref<FinderPattern> topLeft(possiblePatternResults_[patternIdx]->finderPatternInfo->getTopLeft());
    Ref<FinderPattern> topRight(possiblePatternResults_[patternIdx]->finderPatternInfo->getTopRight());
    Ref<FinderPattern> bottomLeft(possiblePatternResults_[patternIdx]->finderPatternInfo->getBottomLeft());
    
    if (alignmentIdx < possiblePatternResults_[patternIdx]->possibleAlignmentPatterns.size())
    {
        Ref<AlignmentPattern> alignment(possiblePatternResults_[patternIdx]->possibleAlignmentPatterns[alignmentIdx]);
        
        Ref<PerspectiveTransform> transform = createTransform(topLeft, topRight, bottomLeft, alignment, possibleDimension);
        Ref<BitMatrix> bits(sampleGrid(image_, possibleDimension, transform, err_handler));
        if (err_handler.ErrCode())
            Ref<DetectorResult>();
        
        ArrayRef< Ref<ResultPoint> > corrners(new Array< Ref<ResultPoint> >(4));
        std::vector<float> points(8, 0.0f);
        points[0] = 0.0f; points[1] =  possibleDimension;  // bottomLeft
        points[2] = 0.0f; points[3] = 0.0f;  // topLeft
        points[4] = possibleDimension; points[5] = 0.0f;  // topRight
        points[6] = possibleDimension; points[7] =  possibleDimension;  // bottomRight
        transform->transformPoints(points);
        corrners[0].reset(Ref<FinderPattern>(new FinderPattern(points[0], points[1], 0)));
        corrners[1].reset(Ref<FinderPattern>(new FinderPattern(points[2], points[3], 0)));
        corrners[2].reset(Ref<FinderPattern>(new FinderPattern(points[4], points[5], 0)));
        corrners[3].reset(Ref<FinderPattern>(new FinderPattern(points[6], points[7], 0)));
        
        Ref<DetectorResult> result(new DetectorResult(bits, corrners, possibleDimension));
        return result;
    }
    else
    {
        Ref<AlignmentPattern> alignment(possiblePatternResults_[patternIdx]->possibleAlignmentPatterns[0]);
        Ref<PerspectiveTransform> transform = createTransform(topLeft, topRight, bottomLeft, alignment, possibleDimension);
        Ref<BitMatrix> bits(sampleGrid(image_, possibleDimension, transform, err_handler));
        if (err_handler.ErrCode())
            Ref<DetectorResult>();
        
        ArrayRef< Ref<ResultPoint> > corrners(new Array< Ref<ResultPoint> >(4));
        std::vector<float> points(8, 0.0f);
        points[0] = 0.0f; points[1] = possibleDimension;  // bottomLeft
        points[2] = 0.0f; points[3] = 0.0f;  // topLeft
        points[4] = possibleDimension; points[5] = 0.0f;  // topRight
        points[6] = possibleDimension; points[7] = possibleDimension;  // bottomRight
        transform->transformPoints(points);
        corrners[0].reset(Ref<FinderPattern>(new FinderPattern(points[0], points[1], 0)));
        corrners[1].reset(Ref<FinderPattern>(new FinderPattern(points[2], points[3], 0)));
        corrners[2].reset(Ref<FinderPattern>(new FinderPattern(points[4], points[5], 0)));
        corrners[3].reset(Ref<FinderPattern>(new FinderPattern(points[6], points[7], 0)));
        
        Ref<DetectorResult> result(new DetectorResult(bits, corrners, possibleDimension));
        
        return result;
    }
}

Ref<DetectorResult> Detector::getResultViaPoints(DecodeHints const& hints, int dimension, std::vector<cv::Point2f>& pts_src, std::vector<cv::Point2f>& pts_dst, bool is_homo,  ErrorHandler & err_handler)
{
    (void)hints;

    if (is_homo)
    {
        if (pts_src.size() < 4) return Ref<DetectorResult>();
        
        cv::Mat H = cv::findHomography(pts_src, pts_dst, cv::RANSAC);
        
        Ref<BitMatrix> bits(sampleGrid(image_, dimension, H, err_handler));
        if (err_handler.ErrCode())
            Ref<DetectorResult>();
        
        std::vector<cv::Point2f> corners(4);
        corners[0].x = 0.0f; corners[0].y = dimension;
        corners[1].x = 0.0f; corners[1].y = 0.0f;
        corners[2].x = dimension; corners[2].y = 0.0f;
        corners[3].x = dimension; corners[3].y = dimension;
        cv::perspectiveTransform(corners, corners, H);
        
        ArrayRef< Ref<ResultPoint> > corrners(new Array< Ref<ResultPoint> >(4));
        corrners[0].reset(Ref<FinderPattern>(new FinderPattern(corners[0].x, corners[0].y, 0)));
        corrners[1].reset(Ref<FinderPattern>(new FinderPattern(corners[1].x, corners[1].y, 0)));
        corrners[2].reset(Ref<FinderPattern>(new FinderPattern(corners[2].x, corners[2].y, 0)));
        corrners[3].reset(Ref<FinderPattern>(new FinderPattern(corners[3].x, corners[3].y, 0)));
        
        Ref<DetectorResult> result(new DetectorResult(bits, corrners, dimension));
        return result;
    }
    else
    {
        int count = static_cast<int>(pts_src.size());
        if (count < 6) return Ref<DetectorResult>();
        
        // correct
        cv::Mat U(count, 6, CV_32F, cv::Scalar(0));
        cv::Mat X(count, 1, CV_32F, cv::Scalar(0));
        cv::Mat Y(count, 1, CV_32F, cv::Scalar(0));
        
        for (int i = 0; i < count; i++) {
            U.at<float>(i, 0) = 1;
            U.at<float>(i, 1) = pts_src[i].x;
            U.at<float>(i, 2) = pts_src[i].y;
            U.at<float>(i, 3) = pts_src[i].x * pts_src[i].x;
            U.at<float>(i, 4) = pts_src[i].y * pts_src[i].y;
            U.at<float>(i, 5) = pts_src[i].x * pts_src[i].y;
            
            X.at<float>(i, 0) = pts_dst[i].x;
            Y.at<float>(i, 0) = pts_dst[i].y;
        }
        
        cv::Mat temp = U.t() * U;
        cv::Mat inv;
        cv::invert(temp, inv);
        cv::Mat K1 = inv * U.t() * X;
        cv::Mat K2 = inv * U.t() * Y;
        float a_x = K1.at<float>(0, 0), b_x = K1.at<float>(1, 0), c_x = K1.at<float>(2, 0), d_x = K1.at<float>(3, 0), e_x = K1.at<float>(4, 0), f_x = K1.at<float>(5, 0);
        float a_y = K2.at<float>(0, 0), b_y = K2.at<float>(1, 0), c_y = K2.at<float>(2, 0), d_y = K2.at<float>(3, 0), e_y = K2.at<float>(4, 0), f_y = K2.at<float>(5, 0);
        
        Ref<BitMatrix> bits(sampleGrid(image_, dimension,
                                       a_x, b_x, c_x, d_x, e_x, f_x,
                                       a_y, b_y, c_y, d_y, e_y, f_y,
                                       err_handler));
        if (err_handler.ErrCode())
            return Ref<DetectorResult>();
        
        std::vector<cv::Point2f> corners(4);
        corners[0].x = 0.0f; corners[0].y = dimension;
        corners[1].x = 0.0f; corners[1].y = 0.0f;
        corners[2].x = dimension; corners[2].y = 0.0f;
        corners[3].x = dimension; corners[3].y = dimension;
        
        for (int i = 0; i < 4; i++) {
            float map_x = a_x + b_x * corners[i].x + c_x * corners[i].y + d_x * corners[i].x * corners[i].x + e_x * corners[i].y * corners[i].y + f_x * corners[i].x * corners[i].y;
            float map_y = a_y + b_y * corners[i].x + c_y * corners[i].y + d_y * corners[i].x * corners[i].x + e_y * corners[i].y * corners[i].y + f_y * corners[i].x * corners[i].y;
            corners[i].x = map_x;
            corners[i].y = map_y;
        }
        
        ArrayRef< Ref<ResultPoint> > corrners(new Array< Ref<ResultPoint> >(4));
        corrners[0].reset(Ref<FinderPattern>(new FinderPattern(corners[0].x, corners[0].y, 0)));
        corrners[1].reset(Ref<FinderPattern>(new FinderPattern(corners[1].x, corners[1].y, 0)));
        corrners[2].reset(Ref<FinderPattern>(new FinderPattern(corners[2].x, corners[2].y, 0)));
        corrners[3].reset(Ref<FinderPattern>(new FinderPattern(corners[3].x, corners[3].y, 0)));
        
        Ref<DetectorResult> result(new DetectorResult(bits, corrners, dimension));
        return result;
    }
    
    return Ref<DetectorResult>();
}

bool Detector::hasSameResult(std::vector<Ref<AlignmentPattern> > possibleAlignmentPatterns, Ref<AlignmentPattern> alignmentPattern)
{
    float moduleSize = alignmentPattern->getModuleSize() / 5.0;
    
    if (moduleSize < 1.0)
    {
        moduleSize = 1.0;
    }
    
    for (size_t i = 0; i < possibleAlignmentPatterns.size(); i++)
    {
        if (possibleAlignmentPatterns[i]->aboutEquals(moduleSize, alignmentPattern->getY(), alignmentPattern->getX()))
        {
            return true;
        }
    }
    return false;
}

Ref<AlignmentPattern> Detector::getNearestAlignmentPattern(int tryFindRange, float moduleSize, int estAlignmentX, int estAlignmentY)
{
    Ref<AlignmentPattern> alignmentPattern;
    
    ErrorHandler err_handler;
    for (int i = 2; i <= tryFindRange; i <<= 1) {
        err_handler.Reset();
        alignmentPattern = findAlignmentInRegion(moduleSize, estAlignmentX, estAlignmentY, static_cast<float>(i), err_handler);
        if (err_handler.ErrCode() == 0)
            break;
    }
    
    return alignmentPattern;
}

Ref<PatternResult> Detector::processFinderPatternInfo(::DecodeHints& hints, Ref<FinderPatternInfo> info, ErrorHandler & err_handler)
{
    (void)hints;
    
    Ref<FinderPattern> topLeft(info->getTopLeft());
    Ref<FinderPattern> topRight(info->getTopRight());
    Ref<FinderPattern> bottomLeft(info->getBottomLeft());
    
    Ref<PatternResult> result(new PatternResult(info));
    result->finderPatternInfo = info;
    result->possibleAlignmentPatterns.clear();
    
    float moduleSizeX_ = calculateModuleSizeOneWay(topLeft, topRight, topLeft->getHorizontalCheckState(),  topRight->getHorizontalCheckState());
    float moduleSizeY_ = calculateModuleSizeOneWay(topLeft, bottomLeft, topLeft->getVerticalCheckState(), bottomLeft->getVerticalCheckState());
    
    if (moduleSizeX_ < 1.0f || moduleSizeY_ < 1.0f)
    {
        err_handler = ReaderErrorHandler("bad midule size");
        return Ref<PatternResult>();
    }
    
    float moduleSize = (moduleSizeX_ + moduleSizeY_) / 2.0f;
    
    if (moduleSize > topLeft->getEstimatedModuleSize() * 1.05 &&
        moduleSize > topRight->getEstimatedModuleSize() * 1.05 &&
        moduleSize > bottomLeft->getEstimatedModuleSize() * 1.05)
    {
        moduleSize = (topLeft->getEstimatedModuleSize() + topRight->getEstimatedModuleSize() +
                      bottomLeft->getEstimatedModuleSize() + moduleSize) / 4;
        moduleSizeX_ = moduleSize;
        moduleSizeY_ = moduleSize;
    }
    
    if (moduleSize < 1.0f)
    {
        err_handler = ReaderErrorHandler("bad midule size");
        return Ref<PatternResult>();
    }
    result->possibleModuleSize = moduleSize;
    
    Version *provisionalVersion = NULL;
    int dimension = -1;
    int modulesBetweenFPCenters;
    
    dimension = computeDimension(topLeft, topRight, bottomLeft, moduleSizeX_, moduleSizeY_);
    
    // lincoln add version 0 should be 17, but we make as 19
    if (dimension == 17) dimension = Version::MICRO_DEMITION;
    
    // Try demension around if it cannot get a version
    int dimensionDiff[5] = {0, 1, -1, 2, -2};
    
    int oriDimension = dimension;
    
    for (int i = 0; i < 5; i++)
    {
        err_handler.Reset();
        dimension = oriDimension + dimensionDiff[i];
        
        provisionalVersion = Version::getProvisionalVersionForDimension(dimension, err_handler);
        if (err_handler.ErrCode() == 0)
        {
            // refine modulesize
            {
                float new_module_size = refineModuleSize(topLeft, topRight, bottomLeft, dimension);
                if ((fabs)(new_module_size - moduleSize) < 2)
                {
                    moduleSize = (new_module_size + moduleSize) / 2.0;
                    result->possibleModuleSize = moduleSize;
                }
                else
                {
                    result->possibleModuleSize = new_module_size;
                }
            }
            break;
        }
    }
    
    if (provisionalVersion == NULL)
    {
        err_handler = zxing::ReaderErrorHandler("Cannot get version number");
        return Ref<PatternResult>();
    }
    
    result->possibleDimension = dimension;
    result->possibleVersion = provisionalVersion->getVersionNumber();
    
    /*=========================================================================================*/
    
    // find aligment point
    modulesBetweenFPCenters = provisionalVersion->getDimensionForVersion(err_handler) - 7;
    if (err_handler.ErrCode())  {
        err_handler = zxing::ReaderErrorHandler("Cannot get version number");
        return Ref<PatternResult>();
    }
    
    Ref<AlignmentPattern> alignmentPattern;
    
    // Guess where a "bottom right" finder pattern would have been
    float bottomRightX = topRight->getX() - topLeft->getX() + bottomLeft->getX();
    float bottomRightY = topRight->getY() - topLeft->getY() + bottomLeft->getY();
    
    // Estimate that alignment pattern is closer by 3 modules from "bottom right" to known top left location
    float correctionToTopLeft = 1.0f - 3.0f / static_cast<float>(modulesBetweenFPCenters);
    int estAlignmentX = static_cast<int>(topLeft->getX() + correctionToTopLeft * (bottomRightX - topLeft->getX()));
    int estAlignmentY = static_cast<int>(topLeft->getY() + correctionToTopLeft * (bottomRightY - topLeft->getY()));
    
    Ref<AlignmentPattern> estimateCenter(new AlignmentPattern(estAlignmentX, estAlignmentY, moduleSize));
    
    bool foundFitLine = false;
    Ref<AlignmentPattern> fitLineCenter;
    
    fitLineCenter = findAlignmentWithFitLine(topLeft, topRight, bottomLeft, moduleSize, err_handler);
    if (err_handler.ErrCode() == 0)
    {
        if (fitLineCenter != NULL && MathUtils::isInRange(fitLineCenter->getX(), fitLineCenter->getY(), image_->getWidth(), image_->getHeight())){
            foundFitLine = true;
        }
    }
    err_handler.Reset();
    
    Ref<AlignmentPattern> fitAP, estAP;
    
    // Anything above version 1 has an alignment pattern
    if (provisionalVersion->getAlignmentPatternCenters().size() > 0) {
        int tryFindRange =  provisionalVersion->getDimensionForVersion(err_handler) / 4;
        if (err_handler.ErrCode())  return Ref<PatternResult>();
        
        if (foundFitLine == true)
        {
            fitAP = getNearestAlignmentPattern(tryFindRange, moduleSize, fitLineCenter->getX(), fitLineCenter->getY());
            
            if (fitAP != NULL && !hasSameResult(result->possibleAlignmentPatterns, fitAP))
            {
                result->possibleAlignmentPatterns.push_back(fitAP);
            }
        }
        
        estAP = getNearestAlignmentPattern(tryFindRange, moduleSize, estimateCenter->getX(), estimateCenter->getY());
        
        if (estAP != NULL && !hasSameResult(result->possibleAlignmentPatterns, estAP))
        {
            result->possibleAlignmentPatterns.push_back(estAP);
        }
    }
    
    // Any way use the fit line result
    if (foundFitLine == true && result->possibleAlignmentPatterns.size() == 0)  //!hasSameResult(result->possibleAlignmentPatterns, fitLineCenter))
    {
#ifdef FIX_ALIGNMENT_CENTER
        float alignmentX = fitLineCenter->getX();
        float alignmentY = fitLineCenter->getY();
        fixAlignmentPattern(alignmentX, alignmentY, topLeft,  topRight, bottomLeft, moduleSize);
        Ref<AlignmentPattern> fitLineCenterFixed = Ref<AlignmentPattern> (new AlignmentPattern(alignmentX, alignmentY, moduleSize));
#endif
        if (!hasSameResult(result->possibleAlignmentPatterns, fitLineCenterFixed))
        {
            result->possibleAlignmentPatterns.push_back(fitLineCenterFixed);
        }
        
        if (!hasSameResult(result->possibleAlignmentPatterns, fitLineCenter))
        {
            result->possibleAlignmentPatterns.push_back(fitLineCenter);
        }
    }
    
    if (result->possibleAlignmentPatterns.size() == 0)  //(!hasSameResult(result->possibleAlignmentPatterns, estimateCenter))
    {
#ifdef FIX_ALIGNMENT_CENTER
        float alignmentX = estimateCenter->getX();
        float alignmentY = estimateCenter->getY();
        fixAlignmentPattern(alignmentX, alignmentY, topLeft,  topRight, bottomLeft, moduleSize);
        Ref<AlignmentPattern> estimateCenterFixed = Ref<AlignmentPattern> (new AlignmentPattern(alignmentX, alignmentY, moduleSize));
#endif
        if (!hasSameResult(result->possibleAlignmentPatterns, estimateCenterFixed))
        {
            result->possibleAlignmentPatterns.push_back(estimateCenterFixed);
        }
        
        if (!hasSameResult(result->possibleAlignmentPatterns, estimateCenter))
        {
            result->possibleAlignmentPatterns.push_back(estimateCenter);
        }
    }
    
    if (dimension == Version::MICRO_DEMITION)
    {
        // may be version 1, try estimate
        float verOneCorrectionToTopLeft = 1.0f - 3.0f / 14;
        int estVerOneAlignmentX = static_cast<int>(topLeft->getX() + verOneCorrectionToTopLeft * (bottomRightX - topLeft->getX()));
        int estVerOneAlignmentY = static_cast<int>(topLeft->getY() + verOneCorrectionToTopLeft * (bottomRightY - topLeft->getY()));
        Ref<AlignmentPattern> estimateVerOneCenter(new AlignmentPattern(estVerOneAlignmentX, estVerOneAlignmentY, moduleSize));
        if (!hasSameResult(result->possibleAlignmentPatterns, estimateVerOneCenter))
        {
#ifdef FIX_ALIGNMENT_CENTER
            float alignmentX = estimateVerOneCenter->getX();
            float alignmentY = estimateVerOneCenter->getY();
            fixAlignmentPattern(alignmentX, alignmentY, topLeft, topRight, bottomLeft, moduleSize);
            Ref<AlignmentPattern> estimateVerOneCenterFixed = Ref<AlignmentPattern>(new AlignmentPattern(alignmentX, alignmentY, moduleSize));
#endif
            if (!hasSameResult(result->possibleAlignmentPatterns, estimateVerOneCenterFixed))
            {
                result->possibleAlignmentPatterns.push_back(estimateVerOneCenterFixed);
            }
            
            if (!hasSameResult(result->possibleAlignmentPatterns, estimateVerOneCenter))
            {
                result->possibleAlignmentPatterns.push_back(estimateVerOneCenter);
            }
        }
    }
    
    Ref<AlignmentPattern> NoneEstimateCenter =
    Ref<AlignmentPattern>(new AlignmentPattern(0, 0, moduleSize));
    result->possibleAlignmentPatterns.push_back(NoneEstimateCenter);
    
    if (result->possibleAlignmentPatterns.size() > 0)
    {
        result->confirmedAlignmentPattern = result->possibleAlignmentPatterns[0];
    }
    
    detectorState_= FINDALIGNPATTERN;
    
    return result;
}

// Computes an average estimated module size based on estimated derived from the positions
// of the three finder patterns.
float Detector::calculateModuleSize(Ref<ResultPoint> topLeft, Ref<ResultPoint> topRight, Ref<ResultPoint> bottomLeft)
{
    // Take the average
    return (calculateModuleSizeOneWay(topLeft, topRight, NORMAL, NORMAL) + calculateModuleSizeOneWay(topLeft, bottomLeft, NORMAL, NORMAL)) / 2.0f;
}

// Estimates module size based on two finder patterns 
// it uses sizeOfBlackWhiteBlackRunBothWays() to figure the width of each, 
// measuring along the axis between their centers.
float Detector::calculateModuleSizeOneWay(Ref<ResultPoint> pattern, Ref<ResultPoint> otherPattern, int patternState, int otherPatternState)
{
    float moduleSizeEst1 = sizeOfBlackWhiteBlackRunBothWays(static_cast<int>(pattern->getX()), static_cast<int>(pattern->getY()),
                                                            static_cast<int>(otherPattern->getX()), static_cast<int>(otherPattern->getY()), patternState, false);
    float moduleSizeEst2 = sizeOfBlackWhiteBlackRunBothWays(static_cast<int>(otherPattern->getX()), static_cast<int>(otherPattern->getY()),
                                                            static_cast<int>(pattern->getX()), static_cast<int>(pattern->getY()), otherPatternState, true);
    
    if (zxing::isnan(moduleSizeEst1) && zxing::isnan(moduleSizeEst2))
    {
        return 0;
    }
    if (zxing::isnan(moduleSizeEst1))
    {
        return moduleSizeEst2 / 7.0f;
    }
    if (zxing::isnan(moduleSizeEst2))
    {
        return moduleSizeEst1 / 7.0f;
    }
    
    // Average them, and divide by 7 since we've counted the width of 3 black modules,
    // and 1 white and 1 black module on either side. Ergo, divide sum by 14.
    return (moduleSizeEst1 + moduleSizeEst2) / 14.0f;
}

// Computes the total width of a finder pattern by looking for a black-white-black run from the center
// in the direction of another point (another finder pattern center), and in the opposite direction too.
float Detector::sizeOfBlackWhiteBlackRunBothWays(int fromX, int fromY, int toX, int toY, int patternState, bool isReverse)
{
    
    float result = 0.0;
    // Now count other way -- don't run off image though of course
    float scale = 1.0f;
    int otherToX = fromX - (toX - fromX);
    if (otherToX < 0)
    {
        scale = static_cast<float>(fromX) / static_cast<float>(fromX - otherToX);
        otherToX = 0;
    }
    else if (otherToX >= static_cast<int>(image_->getWidth()))
    {
        scale = static_cast<float>(image_->getWidth() - 1 - fromX) / static_cast<float>(otherToX - fromX);
        otherToX = image_->getWidth() - 1;
    }
    int otherToY = static_cast<int>(fromY - (toY - fromY) * scale);
    
    scale = 1.0f;
    if (otherToY < 0)
    {
        scale = static_cast<float>(fromY) / static_cast<float>(fromY - otherToY);
        otherToY = 0;
    }
    else if (otherToY >= static_cast<int>(image_->getHeight()))
    {
        scale = static_cast<float>(image_->getHeight() - 1 - fromY) / static_cast<float>(otherToY - fromY);
        otherToY = image_->getHeight() - 1;
    }
    otherToX = static_cast<int>(fromX + (otherToX - fromX) * scale);
    
    float result1 = sizeOfBlackWhiteBlackRun(fromX, fromY, toX, toY);
    float result2 = sizeOfBlackWhiteBlackRun(fromX, fromY, otherToX, otherToY);
    
    if (patternState == FinderPattern::HORIZONTAL_STATE_LEFT_SPILL || patternState == FinderPattern::VERTICAL_STATE_UP_SPILL)
    {
        if (!isReverse) 	result = result1 * 2;
        else result = result2 * 2;
    }
    else if (patternState == FinderPattern::HORIZONTAL_STATE_RIGHT_SPILL || patternState == FinderPattern::VERTICAL_STATE_DOWN_SPILL)
    {
        if (!isReverse) result = result2 * 2;
        else result = result1 * 2;
    }
    else
    {
        result = result1 + result2;
    }
    
    // Middle pixel is double-counted this way; subtract 1
    return result - 1.0f;
}

Ref<BitMatrix> Detector::sampleGrid(Ref<BitMatrix> image, int dimension, Ref<PerspectiveTransform> transform, ErrorHandler &err_handler)
{
    GridSampler &sampler = GridSampler::getInstance();
    Ref<BitMatrix> bits = sampler.sampleGrid(image, dimension, transform, err_handler);
    if (err_handler.ErrCode())
        return Ref<BitMatrix>();
    
    return bits;
}

Ref<BitMatrix> Detector::sampleGrid(Ref<BitMatrix> image, int dimension, cv::Mat& transform, ErrorHandler &err_handler)
{
    GridSampler &sampler = GridSampler::getInstance();
    Ref<BitMatrix> bits = sampler.sampleGrid(image, dimension, transform, err_handler);
    if (err_handler.ErrCode())
        return Ref<BitMatrix>();
    
    return bits;
}

Ref<BitMatrix> Detector::sampleGrid(Ref<BitMatrix> image, int dimension,
                                    float ax, float bx, float cx, float dx, float ex, float fx,
                                    float ay, float by, float cy, float dy, float ey, float fy,
                                    ErrorHandler &err_handler)
{
    GridSampler &sampler = GridSampler::getInstance();
    Ref<BitMatrix> bits = sampler.sampleGrid(image, dimension,
                                             ax, bx, cx, dx, ex, fx,
                                             ay, by, cy, dy, ey, fy,
                                             err_handler);
    if (err_handler.ErrCode())
        return Ref<BitMatrix>();
    
    return bits;
}

// This method traces a line from a point in the image, in the direction towards another point.
// It begins in a black region, and keeps going until it finds white, then black, then white again.
// It reports the distance from the start to this point.
float Detector::sizeOfBlackWhiteBlackRun(int fromX, int fromY, int toX, int toY)
{
    // Mild variant of Bresenham's algorithm;
    // see http:// en.wikipedia.org/wiki/Bresenham's_line_algorithm
    bool steep = abs(toY - fromY) > abs(toX - fromX);
    if (steep)
    {
        int temp = fromX;
        fromX = fromY;
        fromY = temp;
        temp = toX;
        toX = toY;
        toY = temp;
    }
    
    int dx = abs(toX - fromX);
    int dy = abs(toY - fromY);
    int error = -dx >> 1;
    int xstep = fromX < toX ? 1 : -1;
    int ystep = fromY < toY ? 1 : -1;
    // In black pixels, looking for white, first or second time.
    int state = 0;
    // Loop up until x == toX, but not beyond
    int xLimit = toX + xstep;
    for (int x = fromX, y = fromY; x != xLimit; x += xstep) {
        int realX = steep ? y : x;
        int realY = steep ? x : y;
        
        // Does current pixel mean we have moved white to black or vice versa?
        // Scanning black in state 0, 2 and white in state 1, so if we find the wrong color,
        // advance to next state or end if we are in state 2 already
        if (!((state == 1) ^ image_->get(realX, realY)))
        {
            if (state == 2)
            {
                return MathUtils::distance(x, y, fromX, fromY);
            }
            state++;
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
    // Found black-white-black; give the benefit of the doubt that the next pixel outside the image
    // is "white" so this last point at (toX+xStep, toY) is the right ending. This is really a
    // small approximation; (toX+xStep, toY+yStep) might be really correct. Ignore this.
    if (state == 2)
    {
        return MathUtils::distance(toX + xstep, toY, fromX, fromY);
    }
    // else we didn't find even black-white-black; no estimate is really possible
    return nan();
}

// Attempts to locate an alignment pattern in a limited region of the image, 
// which is guessed to contain it. 
Ref<AlignmentPattern> Detector::findAlignmentInRegion(float overallEstModuleSize, int estAlignmentX, int estAlignmentY, float allowanceFactor, ErrorHandler & err_handler)
{
    // Look for an alignment pattern (3 modules in size) around where it should be
    int allowance = static_cast<int>(allowanceFactor * overallEstModuleSize);
    int alignmentAreaLeftX = max(0, estAlignmentX - allowance);
    int alignmentAreaRightX = min(static_cast<int>(image_->getWidth() - 1), estAlignmentX + allowance);
    if (alignmentAreaRightX - alignmentAreaLeftX < overallEstModuleSize * 3)
    {
        err_handler = ReaderErrorHandler("region too small to hold alignment pattern");
        return Ref<AlignmentPattern>();
    }
    int alignmentAreaTopY = max(0, estAlignmentY - allowance);
    int alignmentAreaBottomY = min(static_cast<int>(image_->getHeight() - 1), estAlignmentY + allowance);
    if (alignmentAreaBottomY - alignmentAreaTopY < overallEstModuleSize * 3)
    {
        err_handler = ReaderErrorHandler("region too small to hold alignment pattern");
        return Ref<AlignmentPattern>();
    }
    
    AlignmentPatternFinder alignmentFinder(image_, alignmentAreaLeftX, alignmentAreaTopY, alignmentAreaRightX
                                           - alignmentAreaLeftX, alignmentAreaBottomY - alignmentAreaTopY, overallEstModuleSize, callback_);
    
    Ref<AlignmentPattern> ap = alignmentFinder.find(err_handler);
    if (err_handler.ErrCode())
        return Ref<AlignmentPattern>();
    return ap;
}

Ref<AlignmentPattern> Detector::findAlignmentWithFitLine(Ref<ResultPoint> topLeft, Ref<ResultPoint> topRight,
                                                         Ref<ResultPoint> bottomLeft,
                                                         float moduleSize, ErrorHandler & err_handler)
{
    float alignmentX = 0.0f, alignmentY = 0.0f;
    int imgWidth = image_->getWidth();
    int imgHeight = image_->getHeight();
    Rect bottomLeftRect, topRightRect;
    double rectSize = moduleSize * 7;
    
    bottomLeftRect.x = (bottomLeft->getX() - rectSize / 2.0f) > 0 ? (bottomLeft->getX() - rectSize / 2.0f) : 0;
    bottomLeftRect.y = (bottomLeft->getY() - rectSize / 2.0f) > 0 ? (bottomLeft->getY() - rectSize / 2.0f) : 0;
    bottomLeftRect.width = (bottomLeft->getX() - bottomLeftRect.x) * 2;
    if (bottomLeftRect.x + bottomLeftRect.width > imgWidth)
        bottomLeftRect.width = imgWidth - bottomLeftRect.x;
    bottomLeftRect.height = (bottomLeft->getY() - bottomLeftRect.y) * 2;
    if (bottomLeftRect.y + bottomLeftRect.height > imgHeight)
        bottomLeftRect.height = imgHeight - bottomLeftRect.y;
    
    topRightRect.x = (topRight->getX() - rectSize / 2.0f) > 0 ? (topRight->getX() - rectSize / 2.0f) : 0;
    topRightRect.y = (topRight->getY() - rectSize / 2.0f) > 0 ? (topRight->getY() - rectSize / 2.0f) : 0;
    topRightRect.width = (topRight->getX() - topRightRect.x) * 2;
    if (topRightRect.x + topRightRect.width > imgWidth)
        topRightRect.width = imgWidth - topRightRect.x;
    topRightRect.height = (topRight->getY() - topRightRect.y) * 2;
    if (topRightRect.y + topRightRect.height > imgHeight)
        topRightRect.height = imgHeight - topRightRect.y;
    
    std::vector<Ref<ResultPoint> > topRightPoints;
    std::vector<Ref<ResultPoint> > bottomLeftPoints;
    
    findPointsForLine(topLeft, topRight, bottomLeft, topRightRect, bottomLeftRect, topRightPoints, bottomLeftPoints, moduleSize);
    
    int a1; float k1, b1;
    int fitResult = fitLine(topRightPoints, k1, b1, a1);
    if (fitResult < 0)
    {
        err_handler = ReaderErrorHandler("Cannot find a valid divide for line fit");
        return Ref<AlignmentPattern>();
    }
    
    int a2; float k2, b2;
    int fitResult2 = fitLine(bottomLeftPoints, k2, b2, a2);
    if (fitResult2 < 0)
    {
        err_handler = ReaderErrorHandler("Cannot find a valid divide for line fit");
        return Ref<AlignmentPattern>();
    }
    
    int hasResult = 1;
    if (a1 == 0)
    {
        if (a2 == 0)
        {
            hasResult = 0;
        }
        else
        {
            alignmentX = -b1;
            alignmentY = b2 - b1 * k2;
        }
    }
    else
    {
        if (a2 == 0)
        {
            alignmentX = -b2;
            alignmentY = b1 - b2 * k1;
        }
        else
        {
            if (k1 == k2)
            {
                hasResult = 0;
            }
            else
            {
                alignmentX = (b2 - b1) / (k1 - k2);
                alignmentY = k1 * alignmentX + b1;
            }
        }
    }
    
    // Donot have a valid divide
    if (hasResult == 0)
    {
        err_handler = ReaderErrorHandler("Cannot find a valid divide for line fit");
        return Ref<AlignmentPattern>();
    }
    
    Ref<AlignmentPattern> result(new AlignmentPattern(alignmentX, alignmentY, moduleSize));
    return result;
}


void Detector::fixAlignmentPattern(float &alignmentX, float &alignmentY, 
                                   Ref<ResultPoint> topLeft, Ref<ResultPoint> topRight, Ref<ResultPoint> bottomLeft,
                                   float moduleSize)
{
    (void)topLeft;
    (void)topRight;
    (void)bottomLeft;

    int imgWidth = image_->getWidth();
    int imgHeight = image_->getHeight();
    int maxFixStep = moduleSize * 2;
    int fixStep = 0;
    
    while (alignmentX < imgWidth && alignmentY < imgHeight &&
          alignmentX > 0 && alignmentY > 0 &&
          !image_->get(alignmentX, alignmentY) && fixStep < maxFixStep)
    {
        ++fixStep;
        // Newest Version:  The fix process is like this:
        // 1  2  3
        // 4  0  5
        // 6  7  8
        for (int y = alignmentY - fixStep; y <= alignmentY + fixStep; y++)
        {
            if (y == alignmentY - fixStep || y == alignmentY + fixStep)
            {
                for (int x = alignmentX - fixStep; x <= alignmentX + fixStep; x++)
                {
                    if (x < imgWidth && y < imgHeight && x > 0 && y > 0 && image_->get(x, y))
                    {
                        alignmentX = x;
                        alignmentY = y;
                        return;
                    }
                }
            }
            else
            {
                int x = alignmentX - fixStep;
                if (x < imgWidth && y < imgHeight && x > 0 && y > 0 && image_->get(x, y))
                {
                    alignmentX = x;
                    alignmentY = y;
                    return;
                }
                x = alignmentX + fixStep;
                if (x < imgWidth && y < imgHeight && x > 0 && y > 0 && image_->get(x, y))
                {
                    alignmentX = x;
                    alignmentY = y;
                    return;
                }
            }
        }
    }
}

int Detector::fitLine(std::vector<Ref<ResultPoint> > &oldPoints, float& k, float& b, int& a)
{
    a = 1;
    k = 0.0f;
    b = 0.0f;
    int old_num = oldPoints.size();
    if (old_num < 2)
    {
        return -1;
    }
    
    float tolerance = 2.0f;
    std::vector<Ref<ResultPoint> > fitPoints;
    float pre_diff = -1;
    for (std::vector<Ref<ResultPoint> >::iterator it = oldPoints.begin() + 1; it != oldPoints.end() - 1; it++)
    {
        float diff_x = 0.0f, diff_y = 0.0f, diff = 0.0f;
        if (pre_diff < 0)
        {
            diff_x = (*(it-1))->getX() - (*it)->getX();
            diff_y = (*(it-1))->getY() - (*it)->getY();
            diff = (diff_x * diff_x + diff_y * diff_y);
            pre_diff = diff;
        }
        diff_x = (*(it+1))->getX() - (*it)->getX();
        diff_y = (*(it+1))->getY() - (*it)->getY();
        diff = (diff_x * diff_x+ diff_y * diff_y);
        if (pre_diff <= tolerance && diff <= tolerance){
            fitPoints.push_back(*(it));
        }
        pre_diff = diff;
    }
    
    int num = fitPoints.size();
    if (num < 2)
        return -1;
    
    double x = 0, y = 0, xx = 0, xy = 0, /* yy = 0, */ tem = 0;
    for (int i = 0; i < num; i++)
    {
        int point_x = fitPoints[i]->getX();
        int point_y = fitPoints[i]->getY();
        x += point_x;
        y += point_y;
        xx += point_x * point_x;
        xy += point_x * point_y;
        // yy += point_y * point_y;
    }
    
    tem = xx*num - x*x;
    if (abs(tem) < 0.0000001)
    {
        // Set b as average x
        b = -x / num;
        a = 0;
        k = 1;
        
        return 1;
    }
    
    k = (num * xy - x * y) / tem;
    b = (y - k * x) / num;
    a = 1;
    if (abs(k) < 0.01) k = 0;
    return 1;
}

bool Detector::checkTolerance(Ref<ResultPoint> &topLeft, Ref<ResultPoint> &topRight, Rect& topRightRect, 
                              double modelSize, Ref<ResultPoint>& p, int flag)
{
    int topLeftX = topLeft->getX(), topLeftY = topLeft->getY(), topRightX = topRight->getX(), topRightY = topRight->getY();
    double left_right_k = 0.0f, left_right_b = 0.0f, left_right_b_tolerance, tolerance_b1 = 0.0f, tolerance_b2 = 0.0f;
    
    if (flag < 2)
    {
        double tolerance_y1 = 0.0f, tolerance_y2 = 0.0f;
        double tolerance_x = topRightRect.x;
        if (flag == 1)
            tolerance_x = topRightRect.x + topRightRect.width;
        
        if (topRightX != topLeftX){
            left_right_k = (topRightY - topLeftY) / static_cast<double>(topRightX - topLeftX);
            left_right_b = (topRightY - left_right_k * topRightX);
            double tmp_1 = modelSize * 2.5f;
            double tmp_2 = tmp_1 * left_right_k;
            
            left_right_b_tolerance = sqrt(tmp_1 * tmp_1 + tmp_2 * tmp_2);
            tolerance_b1 = left_right_b - left_right_b_tolerance;
            tolerance_b2 = left_right_b + left_right_b_tolerance;
            tolerance_y1 = left_right_k * tolerance_x + tolerance_b1;
            tolerance_y2 = left_right_k * tolerance_x + tolerance_b2;
        }
        else
        {
            return false;
        }
        
        if (p->getY() < tolerance_y1 || p->getY() > tolerance_y2)
            return false;
        
        return true;
    }
    else
    {
        double tolerance_x1 = 0.0f, tolerance_x2 = 0.0f;
        if (topRightY != topLeftY)
        {
            double tolerance_y = topRightRect.y;
            if (flag == 3)
                tolerance_y = topRightRect.y + topRightRect.height;
            
            left_right_k = (topRightX - topLeftX) / static_cast<double>(topRightY - topLeftY);
            left_right_b = (topRightX - left_right_k * topRightY);
            double tmp_1 = modelSize * 2.5f;
            double tmp_2 = tmp_1 / left_right_k;
            left_right_b_tolerance = sqrt(tmp_1 * tmp_1 + tmp_2 * tmp_2);
            tolerance_b1 = left_right_b - left_right_b_tolerance;
            tolerance_b2 = left_right_b + left_right_b_tolerance;
            tolerance_x1 = left_right_k * tolerance_y + tolerance_b1;
            tolerance_x2 = left_right_k * tolerance_y + tolerance_b2;
            
            if (p->getX() < tolerance_x1 || p->getX() > tolerance_x2)
                return false;
            return true;
        }
        else
        {
            return false;
        }
    }
}

void Detector::findPointsForLine(Ref<ResultPoint> &topLeft, Ref<ResultPoint> &topRight, Ref<ResultPoint> &bottomLeft, 
                                 Rect topRightRect, Rect bottomLeftRect, std::vector<Ref<ResultPoint> > &topRightPoints,
                                 std::vector<Ref<ResultPoint> > &bottomLeftPoints, float modelSize)
{
    int topLeftX = topLeft->getX(), topLeftY = topLeft->getY(), topRightX = topRight->getX(), topRightY = topRight->getY();
    if (!topRightPoints.empty()) topRightPoints.clear();
    if (!bottomLeftPoints.empty()) bottomLeftPoints.clear();
    
    // Added by Valiantliu
    int xMin = 0;
    int xMax = 0;
    int yMin = 0;
    int yMax = 0;
    
    int imgWidth = image_->getWidth();
    int imgHeight = image_->getHeight();
    
    // [-45, 45] or [135, 180) or [-180, -45)
    if (topLeftY == topRightY || abs((topRightX - topLeftX) / (topRightY - topLeftY)) >= 1)
    {
        // [-45, 45] TopRight: left, black->white points; BottomLeft: top, black->white points
        if (topLeftX < topRightX)
        {
            xMin = topRightRect.x;
            xMax = topRightRect.x + modelSize * 2;
            yMin = topRightRect.y + modelSize;
            yMax = topRightRect.y - modelSize + topRightRect.height;
            
            MathUtils::getRangeValues(xMin, xMax, 0, imgWidth - 1);
            MathUtils::getRangeValues(yMin, yMax, 0, imgHeight - 1);
            
            for (int i = yMin; i < yMax; i++)
            {
                for (int j = xMin; j < xMax; j++)
                {
                    bool p = image_->get(j, i);
                    bool p_1 = image_->get(j + 1, i);
                    if (p && !p_1)
                    {
                        Ref<ResultPoint> topRightPoint(new ResultPoint(j, i));
                        if (checkTolerance(topLeft, topRight, topRightRect, modelSize, topRightPoint, 0))
                        {
                            topRightPoints.push_back(topRightPoint);
                            break;
                        }
                    }
                }
            }
            
            xMin = bottomLeftRect.x + modelSize;
            xMax = bottomLeftRect.x - modelSize + bottomLeftRect.width;
            yMin = bottomLeftRect.y;
            yMax = bottomLeftRect.y + 2 * modelSize;
            
            MathUtils::getRangeValues(xMin, xMax, 0, imgWidth - 1);
            MathUtils::getRangeValues(yMin, yMax, 0, imgHeight - 1);
            
            for (int j = xMin; j < xMax; j++)
            {
                for (int i = yMin; i < yMax; i++)
                {
                    bool p = image_->get(j, i);
                    bool p_1 = image_->get(j, i + 1);
                    if (p && !p_1)
                    {
                        Ref<ResultPoint> bottomLeftPoint(new ResultPoint(j, i));
                        if (checkTolerance(topLeft, bottomLeft, bottomLeftRect, modelSize, bottomLeftPoint, 2))
                        {
                            bottomLeftPoints.push_back(bottomLeftPoint);
                            break;
                        }
                    }
                }
            }
        }
        // [135, 180) or [-180, -45)  TopRight: right, white->black points; BottomLeft: bottom, white->black points
        else
        {
            xMin = topRightRect.x + topRightRect.width - 2 * modelSize;
            xMax = topRightRect.x + topRightRect.width;
            yMin = topRightRect.y + modelSize;
            yMax = topRightRect.y - modelSize + topRightRect.height;
            
            MathUtils::getRangeValues(xMin, xMax, 0, imgWidth - 1);
            MathUtils::getRangeValues(yMin, yMax, 0, imgHeight - 1);
            
            for (int i = yMin; i < yMax; i++)
            {
                for (int j = xMin; j < xMax; j++)
                {
                    bool p = image_->get(j, i);
                    bool p_1 = image_->get(j + 1, i);
                    if (!p && p_1)
                    {
                        Ref<ResultPoint> topRightPoint(new ResultPoint(j, i));
                        if (checkTolerance(topLeft, topRight, topRightRect, modelSize, topRightPoint, 1))
                        {
                            topRightPoints.push_back(topRightPoint);
                            break;
                        }
                    }
                }
            }
            
            xMin = bottomLeftRect.x + modelSize;
            xMax = bottomLeftRect.x - modelSize+bottomLeftRect.width;
            yMin = bottomLeftRect.y + bottomLeftRect.height - 2 * modelSize;
            yMax = bottomLeftRect.y + bottomLeftRect.height;
            
            MathUtils::getRangeValues(xMin, xMax, 0, imgWidth - 1);
            MathUtils::getRangeValues(yMin, yMax, 0, imgHeight - 1);
            
            for (int j = xMin; j < xMax; j++)
            {
                for (int i = yMin; i < yMax; i++)
                {
                    bool p = image_->get(j, i);
                    bool p_1 = image_->get(j, i + 1);
                    if (!p && p_1)
                    {
                        Ref<ResultPoint> bottomLeftPoint(new ResultPoint(j, i));
                        if (checkTolerance(topLeft, bottomLeft, bottomLeftRect, modelSize, bottomLeftPoint, 3))
                        {
                            bottomLeftPoints.push_back(bottomLeftPoint);
                            break;
                        }
                    }
                }
            }
        }
    }
    // (45, 135) or (-45, -135)
    else
    {
        // (45, 135) TopRight: top, black->white; BottomRight: right, black->white
        if (topLeftY < topRightY)
        {
            xMin = topRightRect.x + modelSize;
            xMax = topRightRect.x - modelSize + topRightRect.width;
            yMin = topRightRect.y;
            yMax = topRightRect.y + 2 * modelSize;
            
            MathUtils::getRangeValues(xMin, xMax, 0, imgWidth - 1);
            MathUtils::getRangeValues(yMin, yMax, 0, imgHeight - 1);
            
            for (int j = xMin; j < xMax; j++)
            {
                for (int i = yMin; i < yMax; i++)
                {
                    bool p = image_->get(j, i);
                    bool p_1 = image_->get(j, i + 1);
                    if (p && !p_1)
                    {
                        Ref<ResultPoint> topRightPoint(new ResultPoint(j, i));
                        if (checkTolerance(topLeft, topRight, topRightRect, modelSize, topRightPoint, 2))
                        {
                            topRightPoints.push_back(topRightPoint);
                            break;
                        }
                    }
                }
            }
            
            xMin = topRightRect.x + topRightRect.width - 2 * modelSize;
            xMax = topRightRect.x + topRightRect.width;
            yMin = topRightRect.y + modelSize;
            yMax = topRightRect.y - modelSize + topRightRect.height;
            
            MathUtils::getRangeValues(xMin, xMax, 0, imgWidth - 1);
            MathUtils::getRangeValues(yMin, yMax, 0, imgHeight - 1);
            
            for (int i = yMin; i < yMax; i++)
            {
                for (int j = xMin; j < xMax; j++)
                {
                    bool p = image_->get(j, i);
                    bool p_1 = image_->get(j + 1, i);
                    if (!p && p_1)
                    {
                        Ref<ResultPoint> bottomLeftPoint(new ResultPoint(j, i));
                        if (checkTolerance(topLeft, bottomLeft, bottomLeftRect, modelSize, bottomLeftPoint, 1))
                        {
                            bottomLeftPoints.push_back(bottomLeftPoint);
                            break;
                        }
                    }
                }
            }
        }
        // (-45, -135) TopRight: bottom, white->black; BottomRight: left, black->white
        else
        {
            xMin = topRightRect.x + modelSize;
            xMax = topRightRect.x - modelSize + topRightRect.width;
            yMin = topRightRect.y + topRightRect.height - 2 * modelSize;
            yMax = topRightRect.y + topRightRect.height;
            
            MathUtils::getRangeValues(xMin, xMax, 0, imgWidth - 1);
            MathUtils::getRangeValues(yMin, yMax, 0, imgHeight - 1);
            
            for (int j = xMin; j < xMax; j++)
            {
                for (int i = yMin; i < yMax; i++)
                {
                    bool p = image_->get(j, i);
                    bool p_1 = image_->get(j, i + 1);
                    if (!p && p_1)
                    {
                        Ref<ResultPoint> topRightPoint(new ResultPoint(j, i));
                        if (checkTolerance(topLeft, topRight, topRightRect, modelSize, topRightPoint, 3))
                        {
                            topRightPoints.push_back(topRightPoint);
                            break;
                        }
                    }
                }
            }
            
            xMin = bottomLeftRect.x;
            xMax = bottomLeftRect.x + 2 * modelSize;
            yMin = bottomLeftRect.y + modelSize;
            yMax = bottomLeftRect.y + bottomLeftRect.height - modelSize;
            
            MathUtils::getRangeValues(xMin, xMax, 0, imgWidth - 1);
            MathUtils::getRangeValues(yMin, yMax, 0, imgHeight - 1);
            
            for (int i = yMin; i < yMax; i++)
            {
                for (int j = xMin; j < xMax; j++){
                    bool p = image_->get(j, i);
                    bool p_1 = image_->get(j + 1, i);
                    if (p && !p_1)
                    {
                        Ref<ResultPoint> bottomLeftPoint(new ResultPoint(j, i));
                        if (checkTolerance(topLeft, bottomLeft, bottomLeftRect, modelSize, bottomLeftPoint, 0))
                        {
                            bottomLeftPoints.push_back(bottomLeftPoint);
                            break;
                        }
                    }
                }
            }
        }
    }
}


Ref<PerspectiveTransform>Detector:: createTransform(Ref<FinderPatternInfo>info,
                                                    Ref<ResultPoint> alignmentPattern, int dimension)
{
    Ref<FinderPattern> topLeft(info->getTopLeft());
    Ref<FinderPattern> topRight(info->getTopRight());
    Ref<FinderPattern> bottomLeft(info->getBottomLeft());
    Ref<PerspectiveTransform> transform = createTransform(topLeft, topRight, bottomLeft, alignmentPattern, dimension);
    return transform;
}


Ref<PerspectiveTransform> Detector::createTransform(Ref<ResultPoint> topLeft, Ref<ResultPoint> topRight,
                                                    Ref <ResultPoint > bottomLeft, Ref<ResultPoint> alignmentPattern, int dimension)
{
    float dimMinusThree = static_cast<float>(dimension) - 3.5f;
    float bottomRightX;
    float bottomRightY;
    float sourceBottomRightX;
    float sourceBottomRightY;
    
    if (alignmentPattern  && alignmentPattern->getX())
    {
        bottomRightX = alignmentPattern->getX();
        bottomRightY = alignmentPattern->getY();
        sourceBottomRightX = dimMinusThree - 3.0f;
        sourceBottomRightY = sourceBottomRightX;
    }
    else
    {
        // Don't have an alignment pattern, just make up the bottom-right point
        bottomRightX = (topRight->getX() - topLeft->getX()) + bottomLeft->getX();
        bottomRightY = (topRight->getY() - topLeft->getY()) + bottomLeft->getY();
        float deltaX = topLeft->getX() - bottomLeft->getX();
        float deltaY = topLeft->getY() - bottomLeft->getY();
        if (fabs(deltaX) < fabs(deltaY))
            deltaY = topLeft->getY() - topRight->getY();
        else
            deltaX = topLeft->getX() - topRight->getX();
        bottomRightX += 2 * deltaX;
        bottomRightY += 2 * deltaY;
        sourceBottomRightX = dimMinusThree;
        sourceBottomRightY = dimMinusThree;
    }
    Ref<PerspectiveTransform> transform(PerspectiveTransform::quadrilateralToQuadrilateral(3.5f, 3.5f, dimMinusThree, 3.5f, sourceBottomRightX,
                                                                                           sourceBottomRightY, 3.5f, dimMinusThree, topLeft->getX(), topLeft->getY(), topRight->getX(),
                                                                                           topRight->getY(), bottomRightX, bottomRightY, bottomLeft->getX(), bottomLeft->getY()));
    
    return transform;
}


Ref<PerspectiveTransform> Detector::createInvertedTransform(Ref<ResultPoint> topLeft, Ref<ResultPoint> topRight,
                                                            Ref <ResultPoint > bottomLeft, Ref<ResultPoint> alignmentPattern, int dimension)
{
    float dimMinusThree = static_cast<float>(dimension) - 3.5f;
    float bottomRightX;
    float bottomRightY;
    float sourceBottomRightX;
    float sourceBottomRightY;
    
    if (alignmentPattern != 0)
    {
        bottomRightX = alignmentPattern->getX();
        bottomRightY = alignmentPattern->getY();
        sourceBottomRightX = dimMinusThree - 3.0f;
        sourceBottomRightY = sourceBottomRightX;
    }
    else
    {
        // Don't have an alignment pattern, just make up the bottom-right point
        bottomRightX = (topRight->getX() - topLeft->getX()) + bottomLeft->getX();
        bottomRightY = (topRight->getY() - topLeft->getY()) + bottomLeft->getY();
        float deltaX = topLeft->getX() - bottomLeft->getX();
        float deltaY = topLeft->getY() - bottomLeft->getY();
        if (fabs(deltaX) < fabs(deltaY))
            deltaY = topLeft->getY() - topRight->getY();
        else
            deltaX = topLeft->getX() - topRight->getX();
        bottomRightX += 2 * deltaX;
        bottomRightY += 2 * deltaY;
        sourceBottomRightX = dimMinusThree;
        sourceBottomRightY = dimMinusThree;
    }
    
    Ref<PerspectiveTransform> transform(PerspectiveTransform::quadrilateralToQuadrilateral(topLeft->getX(), topLeft->getY(),
                                                                                           topRight->getX(), topRight->getY(),
                                                                                           bottomRightX, bottomRightY,
                                                                                           bottomLeft->getX(),
                                                                                           bottomLeft->getY(),
                                                                                           3.5f, 3.5f,
                                                                                           dimMinusThree, 3.5f,
                                                                                           sourceBottomRightX,
                                                                                           sourceBottomRightY,
                                                                                           3.5f, dimMinusThree
                                                                                          ));
    
    return transform;
}


// Computes the dimension (number of modules on a size) of the QR code based on the position
// of the finder patterns and estimated module size.
int Detector::computeDimension(Ref<ResultPoint> topLeft, Ref<ResultPoint> topRight, Ref<ResultPoint> bottomLeft,
                               float moduleSizeX, float moduleSizeY)
{
    int tltrCentersDimension =
    ResultPoint::distance(topLeft, topRight) / moduleSizeX;
    int tlblCentersDimension =
    ResultPoint::distance(topLeft, bottomLeft) / moduleSizeY;
    
    float tmp_dimension = ((tltrCentersDimension + tlblCentersDimension) / 2.0) + 7.0;
    int dimension = MathUtils::round(tmp_dimension);
    
    int mod = dimension & 0x03;  // mod 4
    
    switch (mod) {  // mod 4
        case 0:
            dimension++;
            break;
        case 2:
            dimension--;
            break;
    }
    return dimension;
}

float Detector::refineModuleSize(Ref<ResultPoint> topLeft, Ref<ResultPoint> topRight, Ref<ResultPoint> bottomLeft, int dimension) {
    float tmp_dimension = dimension - 7.0;
    float module_size_x = ResultPoint::distance(topLeft, topRight) / tmp_dimension;
    float module_size_y = ResultPoint::distance(topLeft, bottomLeft) / tmp_dimension;
    float module_size = (module_size_x + module_size_y) / 2.0;
    return module_size;
}

bool Detector::checkConvexQuadrilateral(Ref<ResultPoint> topLeft, Ref<ResultPoint> topRight, Ref<ResultPoint> bottomLeft,
                                        Ref<ResultPoint>bottomRight){
    float v1[2];
    float v2[2];
    float v3[2];
    float v4[2];
    
    v1[0] = topLeft->getX()-topRight->getX(); v1[1] = topLeft->getY()-topRight->getY();
    v2[0] = topRight->getX()-bottomRight->getX(); v2[1] = topRight->getY()-bottomRight->getY();
    v3[0] = bottomRight->getX()-bottomLeft->getX(); v3[1] = bottomRight->getY()-bottomLeft->getY();
    v4[0] = bottomLeft->getX()-topLeft->getX(); v4[1] = bottomLeft->getY()-topLeft->getY();
    
    float c1 = MathUtils::VecCross(v1, v2);
    float c2 = MathUtils::VecCross(v2, v3);
    float c3 = MathUtils::VecCross(v3, v4);
    float c4 = MathUtils::VecCross(v4, v1);
    
    // If it looks like a convex quadrilateral
    if  ((c1 < 0.0 && c2 < 0.0 && c3 < 0.0 && c4 < 0.0)||(c1 > 0.0 && c2 > 0.0 && c3 > 0.0 && c4 > 0.0))
        return true;
    else
        return false;
}
