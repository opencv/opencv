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

#ifndef __OPENCV_VIDEOSTAB_WOBBLE_SUPPRESSION_HPP__
#define __OPENCV_VIDEOSTAB_WOBBLE_SUPPRESSION_HPP__

#include <vector>
#include "opencv2/core.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/videostab/global_motion.hpp"
#include "opencv2/videostab/log.hpp"

namespace cv
{
namespace videostab
{

//! @addtogroup videostab
//! @{

class CV_EXPORTS WobbleSuppressorBase
{
public:
    WobbleSuppressorBase();

    virtual ~WobbleSuppressorBase() {}

    void setMotionEstimator(Ptr<ImageMotionEstimatorBase> val) { motionEstimator_ = val; }
    Ptr<ImageMotionEstimatorBase> motionEstimator() const { return motionEstimator_; }

    virtual void suppress(int idx, const Mat &frame, Mat &result) = 0;


    // data from stabilizer

    virtual void setFrameCount(int val) { frameCount_ = val; }
    virtual int frameCount() const { return frameCount_; }

    virtual void setMotions(const std::vector<Mat> &val) { motions_ = &val; }
    virtual const std::vector<Mat>& motions() const { return *motions_; }

    virtual void setMotions2(const std::vector<Mat> &val) { motions2_ = &val; }
    virtual const std::vector<Mat>& motions2() const { return *motions2_; }

    virtual void setStabilizationMotions(const std::vector<Mat> &val) { stabilizationMotions_ = &val; }
    virtual const std::vector<Mat>& stabilizationMotions() const { return *stabilizationMotions_; }

protected:
    Ptr<ImageMotionEstimatorBase> motionEstimator_;
    int frameCount_;
    const std::vector<Mat> *motions_;
    const std::vector<Mat> *motions2_;
    const std::vector<Mat> *stabilizationMotions_;
};

class CV_EXPORTS NullWobbleSuppressor : public WobbleSuppressorBase
{
public:
    virtual void suppress(int idx, const Mat &frame, Mat &result);
};

class CV_EXPORTS MoreAccurateMotionWobbleSuppressorBase : public WobbleSuppressorBase
{
public:
    virtual void setPeriod(int val) { period_ = val; }
    virtual int period() const { return period_; }

protected:
    MoreAccurateMotionWobbleSuppressorBase() { setPeriod(30); }

    int period_;
};

class CV_EXPORTS MoreAccurateMotionWobbleSuppressor : public MoreAccurateMotionWobbleSuppressorBase
{
public:
    virtual void suppress(int idx, const Mat &frame, Mat &result);

private:
    Mat_<float> mapx_, mapy_;
};

#if defined(HAVE_OPENCV_CUDAWARPING)
class CV_EXPORTS MoreAccurateMotionWobbleSuppressorGpu : public MoreAccurateMotionWobbleSuppressorBase
{
public:
    void suppress(int idx, const cuda::GpuMat &frame, cuda::GpuMat &result);
    virtual void suppress(int idx, const Mat &frame, Mat &result);

private:
    cuda::GpuMat frameDevice_, resultDevice_;
    cuda::GpuMat mapx_, mapy_;
};
#endif

//! @}

} // namespace videostab
} // namespace cv

#endif
