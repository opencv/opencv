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
#include <string>
#include <fstream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/opencv_modules.hpp"
#include "opencv2/videostab/optical_flow.hpp"
#include "opencv2/videostab/motion_core.hpp"
#include "opencv2/videostab/outlier_rejection.hpp"

#if HAVE_OPENCV_GPU
  #include "opencv2/gpu/gpu.hpp"
#endif

namespace cv
{
namespace videostab
{

CV_EXPORTS Mat estimateGlobalMotionLeastSquares(
        int npoints, Point2f *points0, Point2f *points1, int model = MM_AFFINE, float *rmse = 0);


CV_EXPORTS Mat estimateGlobalMotionRobust(
        const std::vector<Point2f> &points0, const std::vector<Point2f> &points1,
        int model = MM_AFFINE, const RansacParams &params = RansacParams::default2dMotion(MM_AFFINE),
        float *rmse = 0, int *ninliers = 0);

class CV_EXPORTS GlobalMotionEstimatorBase
{
public:    
    virtual ~GlobalMotionEstimatorBase() {}

    virtual void setMotionModel(MotionModel val) { motionModel_ = val; }
    virtual MotionModel motionModel() const { return motionModel_; }

    virtual Mat estimate(const Mat &frame0, const Mat &frame1, bool *ok = 0) = 0;

protected:
    GlobalMotionEstimatorBase(MotionModel model) { setMotionModel(model); }

    MotionModel motionModel_;
};

class CV_EXPORTS FromFileMotionReader : public GlobalMotionEstimatorBase
{
public:
    FromFileMotionReader(const std::string &path);
    virtual Mat estimate(const Mat &frame0, const Mat &frame1, bool *ok = 0);

private:
    std::ifstream file_;
};

class CV_EXPORTS ToFileMotionWriter : public GlobalMotionEstimatorBase
{
public:
    ToFileMotionWriter(const std::string &path, Ptr<GlobalMotionEstimatorBase> estimator);
    virtual Mat estimate(const Mat &frame0, const Mat &frame1, bool *ok = 0);

private:
    std::ofstream file_;
    Ptr<GlobalMotionEstimatorBase> estimator_;
};

class CV_EXPORTS PyrLkRobustMotionEstimatorBase : public GlobalMotionEstimatorBase
{
public:
    virtual void setRansacParams(const RansacParams &val) { ransacParams_ = val; }
    virtual RansacParams ransacParams() const { return ransacParams_; }

    virtual void setOutlierRejector(Ptr<IOutlierRejector> val) { outlierRejector_ = val; }
    virtual Ptr<IOutlierRejector> outlierRejector() const { return outlierRejector_; }

    virtual void setMinInlierRatio(float val) { minInlierRatio_ = val; }
    virtual float minInlierRatio() const { return minInlierRatio_; }

protected:
    PyrLkRobustMotionEstimatorBase(MotionModel model);

    RansacParams ransacParams_;
    Ptr<IOutlierRejector> outlierRejector_;
    float minInlierRatio_;
};

class CV_EXPORTS PyrLkRobustMotionEstimator : public PyrLkRobustMotionEstimatorBase
{
public:
    PyrLkRobustMotionEstimator(MotionModel model = MM_AFFINE);

    void setDetector(Ptr<FeatureDetector> val) { detector_ = val; }
    Ptr<FeatureDetector> detector() const { return detector_; }

    void setOptFlowEstimator(Ptr<ISparseOptFlowEstimator> val) { optFlowEstimator_ = val; }
    Ptr<ISparseOptFlowEstimator> optFlowEstimator() const { return optFlowEstimator_; }

    void setGridSize(Size val) { gridSize_ = val; }
    Size gridSize() const { return gridSize_; }

    virtual Mat estimate(const Mat &frame0, const Mat &frame1, bool *ok = 0);

private:
    Ptr<FeatureDetector> detector_;
    Ptr<ISparseOptFlowEstimator> optFlowEstimator_;
    Size gridSize_;

    std::vector<uchar> status_;
    std::vector<KeyPoint> keypointsPrev_;
    std::vector<Point2f> pointsPrev_, points_;
    std::vector<Point2f> pointsPrevGood_, pointsGood_;
};

#if HAVE_OPENCV_GPU
class CV_EXPORTS PyrLkRobustMotionEstimatorGpu : public PyrLkRobustMotionEstimatorBase
{
public:
    PyrLkRobustMotionEstimatorGpu(MotionModel model = MM_AFFINE);

    virtual Mat estimate(const Mat &frame0, const Mat &frame1, bool *ok = 0);
    Mat estimate(const gpu::GpuMat &frame0, const gpu::GpuMat &frame1, bool *ok = 0);

private:
    gpu::GoodFeaturesToTrackDetector_GPU detector_;
    SparsePyrLkOptFlowEstimatorGpu optFlowEstimator_;

    gpu::GpuMat frame0_, grayFrame0_, frame1_;
    gpu::GpuMat pointsPrev_, points_;
    gpu::GpuMat status_;

    Mat hostPointsPrev_, hostPoints_;
    std::vector<Point2f> hostPointsPrevGood_, hostPointsGood_;
    std::vector<uchar> rejectionStatus_;
};
#endif

CV_EXPORTS Mat getMotion(int from, int to, const std::vector<Mat> &motions);

} // namespace videostab
} // namespace cv

#endif
