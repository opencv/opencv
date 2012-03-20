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

#ifndef __OPENCV_VIDEOSTAB_GLOBAL_MOTION_HPP__
#define __OPENCV_VIDEOSTAB_GLOBAL_MOTION_HPP__

#include <vector>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/videostab/optical_flow.hpp"

namespace cv
{
namespace videostab
{

enum MotionModel
{
    TRANSLATION = 0,
    TRANSLATION_AND_SCALE = 1,
    AFFINE = 2
};

CV_EXPORTS Mat estimateGlobalMotionLeastSquares(
        const std::vector<Point2f> &points0, const std::vector<Point2f> &points1,
        int model = AFFINE, float *rmse = 0);

CV_EXPORTS struct RansacParams
{
    int size; // subset size
    float thresh; // max error to classify as inlier
    float eps; // max outliers ratio
    float prob; // probability of success

    RansacParams(int size, float thresh, float eps, float prob)
        : size(size), thresh(thresh), eps(eps), prob(prob) {}

    static RansacParams affine2dMotionStd() { return RansacParams(6, 0.5f, 0.5f, 0.99f); }
    static RansacParams translationAndScale2dMotionStd() { return RansacParams(3, 0.5f, 0.5f, 0.99f); }
    static RansacParams translationMotionStd() { return RansacParams(2, 0.5f, 0.5f, 0.99f); }
};

CV_EXPORTS Mat estimateGlobalMotionRobust(
        const std::vector<Point2f> &points0, const std::vector<Point2f> &points1,
        int model = AFFINE, const RansacParams &params = RansacParams::affine2dMotionStd(),
        float *rmse = 0, int *ninliers = 0);

class CV_EXPORTS IGlobalMotionEstimator
{
public:
    virtual ~IGlobalMotionEstimator() {}
    virtual Mat estimate(const Mat &frame0, const Mat &frame1) = 0;
};

class CV_EXPORTS PyrLkRobustMotionEstimator : public IGlobalMotionEstimator
{
public:
    PyrLkRobustMotionEstimator();

    void setDetector(Ptr<FeatureDetector> val) { detector_ = val; }
    Ptr<FeatureDetector> detector() const { return detector_; }

    void setOptFlowEstimator(Ptr<ISparseOptFlowEstimator> val) { optFlowEstimator_ = val; }
    Ptr<ISparseOptFlowEstimator> optFlowEstimator() const { return optFlowEstimator_; }

    void setMotionModel(MotionModel val) { motionModel_ = val; }
    MotionModel motionModel() const { return motionModel_; }

    void setRansacParams(const RansacParams &val) { ransacParams_ = val; }
    RansacParams ransacParams() const { return ransacParams_; }

    void setMaxRmse(float val) { maxRmse_ = val; }
    float maxRmse() const { return maxRmse_; }

    void setMinInlierRatio(float val) { minInlierRatio_ = val; }
    float minInlierRatio() const { return minInlierRatio_; }

    virtual Mat estimate(const Mat &frame0, const Mat &frame1);

private:
    Ptr<FeatureDetector> detector_;
    Ptr<ISparseOptFlowEstimator> optFlowEstimator_;
    MotionModel motionModel_;
    RansacParams ransacParams_;
    std::vector<uchar> status_;
    std::vector<KeyPoint> keypointsPrev_;
    std::vector<Point2f> pointsPrev_, points_;
    std::vector<Point2f> pointsPrevGood_, pointsGood_;
    float maxRmse_;
    float minInlierRatio_;
};

CV_EXPORTS Mat getMotion(int from, int to, const std::vector<Mat> &motions);

CV_EXPORTS Mat ensureInclusionConstraint(const Mat &M, Size size, float trimRatio);

CV_EXPORTS float estimateOptimalTrimRatio(const Mat &M, Size size);

// frame1 is non-transformed frame
CV_EXPORTS float alignementError(const Mat &M, const Mat &frame0, const Mat &mask0, const Mat &frame1);

} // namespace videostab
} // namespace cv

#endif
