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

#ifndef OPENCV_VIDEOSTAB_GLOBAL_MOTION_HPP
#define OPENCV_VIDEOSTAB_GLOBAL_MOTION_HPP

#include <vector>
#include <fstream>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/opencv_modules.hpp"
#include "opencv2/videostab/optical_flow.hpp"
#include "opencv2/videostab/motion_core.hpp"
#include "opencv2/videostab/outlier_rejection.hpp"

#ifdef HAVE_OPENCV_CUDAIMGPROC
#  include "opencv2/cudaimgproc.hpp"
#endif

namespace cv
{
namespace videostab
{

//! @addtogroup videostab_motion
//! @{

/** @brief Estimates best global motion between two 2D point clouds in the least-squares sense.

@note Works in-place and changes input point arrays.

@param points0 Source set of 2D points (32F).
@param points1 Destination set of 2D points (32F).
@param model Motion model (up to MM_AFFINE).
@param rmse Final root-mean-square error.
@return 3x3 2D transformation matrix (32F).
 */
CV_EXPORTS Mat estimateGlobalMotionLeastSquares(
        InputOutputArray points0, InputOutputArray points1, int model = MM_AFFINE,
        float *rmse = 0);

/** @brief Estimates best global motion between two 2D point clouds robustly (using RANSAC method).

@param points0 Source set of 2D points (32F).
@param points1 Destination set of 2D points (32F).
@param model Motion model. See cv::videostab::MotionModel.
@param params RANSAC method parameters. See videostab::RansacParams.
@param rmse Final root-mean-square error.
@param ninliers Final number of inliers.
 */
CV_EXPORTS Mat estimateGlobalMotionRansac(
        InputArray points0, InputArray points1, int model = MM_AFFINE,
        const RansacParams &params = RansacParams::default2dMotion(MM_AFFINE),
        float *rmse = 0, int *ninliers = 0);

/** @brief Base class for all global motion estimation methods.
 */
class CV_EXPORTS MotionEstimatorBase
{
public:
    virtual ~MotionEstimatorBase() {}

    /** @brief Sets motion model.

    @param val Motion model. See cv::videostab::MotionModel.
     */
    virtual void setMotionModel(MotionModel val) { motionModel_ = val; }

    /**
    @return Motion model. See cv::videostab::MotionModel.
    */
    virtual MotionModel motionModel() const { return motionModel_; }

    /** @brief Estimates global motion between two 2D point clouds.

    @param points0 Source set of 2D points (32F).
    @param points1 Destination set of 2D points (32F).
    @param ok Indicates whether motion was estimated successfully.
    @return 3x3 2D transformation matrix (32F).
     */
    virtual Mat estimate(InputArray points0, InputArray points1, bool *ok = 0) = 0;

protected:
    MotionEstimatorBase(MotionModel model) { setMotionModel(model); }

private:
    MotionModel motionModel_;
};

/** @brief Describes a robust RANSAC-based global 2D motion estimation method which minimizes L2 error.
 */
class CV_EXPORTS MotionEstimatorRansacL2 : public MotionEstimatorBase
{
public:
    MotionEstimatorRansacL2(MotionModel model = MM_AFFINE);

    void setRansacParams(const RansacParams &val) { ransacParams_ = val; }
    RansacParams ransacParams() const { return ransacParams_; }

    void setMinInlierRatio(float val) { minInlierRatio_ = val; }
    float minInlierRatio() const { return minInlierRatio_; }

    virtual Mat estimate(InputArray points0, InputArray points1, bool *ok = 0);

private:
    RansacParams ransacParams_;
    float minInlierRatio_;
};

/** @brief Describes a global 2D motion estimation method which minimizes L1 error.

@note To be able to use this method you must build OpenCV with CLP library support. :
 */
class CV_EXPORTS MotionEstimatorL1 : public MotionEstimatorBase
{
public:
    MotionEstimatorL1(MotionModel model = MM_AFFINE);

    virtual Mat estimate(InputArray points0, InputArray points1, bool *ok = 0);

private:
    std::vector<double> obj_, collb_, colub_;
    std::vector<double> elems_, rowlb_, rowub_;
    std::vector<int> rows_, cols_;

    void set(int row, int col, double coef)
    {
        rows_.push_back(row);
        cols_.push_back(col);
        elems_.push_back(coef);
    }
};

/** @brief Base class for global 2D motion estimation methods which take frames as input.
 */
class CV_EXPORTS ImageMotionEstimatorBase
{
public:
    virtual ~ImageMotionEstimatorBase() {}

    virtual void setMotionModel(MotionModel val) { motionModel_ = val; }
    virtual MotionModel motionModel() const { return motionModel_; }

    virtual Mat estimate(const Mat &frame0, const Mat &frame1, bool *ok = 0) = 0;

protected:
    ImageMotionEstimatorBase(MotionModel model) { setMotionModel(model); }

private:
    MotionModel motionModel_;
};

class CV_EXPORTS FromFileMotionReader : public ImageMotionEstimatorBase
{
public:
    FromFileMotionReader(const String &path);

    virtual Mat estimate(const Mat &frame0, const Mat &frame1, bool *ok = 0);

private:
    std::ifstream file_;
};

class CV_EXPORTS ToFileMotionWriter : public ImageMotionEstimatorBase
{
public:
    ToFileMotionWriter(const String &path, Ptr<ImageMotionEstimatorBase> estimator);

    virtual void setMotionModel(MotionModel val) { motionEstimator_->setMotionModel(val); }
    virtual MotionModel motionModel() const { return motionEstimator_->motionModel(); }

    virtual Mat estimate(const Mat &frame0, const Mat &frame1, bool *ok = 0);

private:
    std::ofstream file_;
    Ptr<ImageMotionEstimatorBase> motionEstimator_;
};

/** @brief Describes a global 2D motion estimation method which uses keypoints detection and optical flow for
matching.
 */
class CV_EXPORTS KeypointBasedMotionEstimator : public ImageMotionEstimatorBase
{
public:
    KeypointBasedMotionEstimator(Ptr<MotionEstimatorBase> estimator);

    virtual void setMotionModel(MotionModel val) { motionEstimator_->setMotionModel(val); }
    virtual MotionModel motionModel() const { return motionEstimator_->motionModel(); }

    void setDetector(Ptr<FeatureDetector> val) { detector_ = val; }
    Ptr<FeatureDetector> detector() const { return detector_; }

    void setOpticalFlowEstimator(Ptr<ISparseOptFlowEstimator> val) { optFlowEstimator_ = val; }
    Ptr<ISparseOptFlowEstimator> opticalFlowEstimator() const { return optFlowEstimator_; }

    void setOutlierRejector(Ptr<IOutlierRejector> val) { outlierRejector_ = val; }
    Ptr<IOutlierRejector> outlierRejector() const { return outlierRejector_; }

    virtual Mat estimate(const Mat &frame0, const Mat &frame1, bool *ok = 0);
    Mat estimate(InputArray frame0, InputArray frame1, bool *ok = 0);

private:
    Ptr<MotionEstimatorBase> motionEstimator_;
    Ptr<FeatureDetector> detector_;
    Ptr<ISparseOptFlowEstimator> optFlowEstimator_;
    Ptr<IOutlierRejector> outlierRejector_;

    std::vector<uchar> status_;
    std::vector<KeyPoint> keypointsPrev_;
    std::vector<Point2f> pointsPrev_, points_;
    std::vector<Point2f> pointsPrevGood_, pointsGood_;
};

#if defined(HAVE_OPENCV_CUDAIMGPROC) && defined(HAVE_OPENCV_CUDAOPTFLOW)

class CV_EXPORTS KeypointBasedMotionEstimatorGpu : public ImageMotionEstimatorBase
{
public:
    KeypointBasedMotionEstimatorGpu(Ptr<MotionEstimatorBase> estimator);

    virtual void setMotionModel(MotionModel val) { motionEstimator_->setMotionModel(val); }
    virtual MotionModel motionModel() const { return motionEstimator_->motionModel(); }

    void setOutlierRejector(Ptr<IOutlierRejector> val) { outlierRejector_ = val; }
    Ptr<IOutlierRejector> outlierRejector() const { return outlierRejector_; }

    virtual Mat estimate(const Mat &frame0, const Mat &frame1, bool *ok = 0);
    Mat estimate(const cuda::GpuMat &frame0, const cuda::GpuMat &frame1, bool *ok = 0);

private:
    Ptr<MotionEstimatorBase> motionEstimator_;
    Ptr<cuda::CornersDetector> detector_;
    SparsePyrLkOptFlowEstimatorGpu optFlowEstimator_;
    Ptr<IOutlierRejector> outlierRejector_;

    cuda::GpuMat frame0_, grayFrame0_, frame1_;
    cuda::GpuMat pointsPrev_, points_;
    cuda::GpuMat status_;

    Mat hostPointsPrev_, hostPoints_;
    std::vector<Point2f> hostPointsPrevTmp_, hostPointsTmp_;
    std::vector<uchar> rejectionStatus_;
};

#endif // defined(HAVE_OPENCV_CUDAIMGPROC) && defined(HAVE_OPENCV_CUDAOPTFLOW)

/** @brief Computes motion between two frames assuming that all the intermediate motions are known.

@param from Source frame index.
@param to Destination frame index.
@param motions Pair-wise motions. motions[i] denotes motion from the frame i to the frame i+1
@return Motion from the Source frame to the Destination frame.
 */
CV_EXPORTS Mat getMotion(int from, int to, const std::vector<Mat> &motions);

//! @}

} // namespace videostab
} // namespace cv

#endif
