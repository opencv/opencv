/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef __OPENCV_VIDEOSTAB_MOTION_STABILIZING_HPP__
#define __OPENCV_VIDEOSTAB_MOTION_STABILIZING_HPP__

#include <vector>
#include <utility>
#include "opencv2/core/core.hpp"

namespace cv
{
namespace videostab
{

class CV_EXPORTS IMotionStabilizer
{
public:
    virtual ~IMotionStabilizer() {}

    // assumes that [0, size-1) is in or equals to [range.first, range.second)
    virtual void stabilize(
            int size, const std::vector<Mat> &motions, std::pair<int,int> range,
            Mat *stabilizationMotions) const = 0;
};

class CV_EXPORTS MotionStabilizationPipeline : public IMotionStabilizer
{
public:
    void pushBack(Ptr<IMotionStabilizer> stabilizer) { stabilizers_.push_back(stabilizer); }
    bool empty() const { return stabilizers_.empty(); }

    virtual void stabilize(
            int size, const std::vector<Mat> &motions, std::pair<int,int> range,
            Mat *stabilizationMotions) const;

private:
    std::vector<Ptr<IMotionStabilizer> > stabilizers_;
};

class CV_EXPORTS MotionFilterBase : public IMotionStabilizer
{
public:
    virtual ~MotionFilterBase() {}

    virtual Mat stabilize(
            int idx, const std::vector<Mat> &motions, std::pair<int,int> range) const = 0;

    virtual void stabilize(
            int size, const std::vector<Mat> &motions, std::pair<int,int> range,
            Mat *stabilizationMotions) const;
};

class CV_EXPORTS GaussianMotionFilter : public MotionFilterBase
{
public:
    GaussianMotionFilter(int radius = 15, float stdev = -1.f) { setParams(radius, stdev); }

    void setParams(int radius, float stdev = -1.f);
    int radius() const { return radius_; }
    float stdev() const { return stdev_; }

    virtual Mat stabilize(
            int idx, const std::vector<Mat> &motions, std::pair<int,int> range) const;

private:
    int radius_;
    float stdev_;
    std::vector<float> weight_;
};

CV_EXPORTS Mat ensureInclusionConstraint(const Mat &M, Size size, float trimRatio);

CV_EXPORTS float estimateOptimalTrimRatio(const Mat &M, Size size);

} // namespace videostab
} // namespace

#endif
