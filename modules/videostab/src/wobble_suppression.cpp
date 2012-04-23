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

#include "precomp.hpp"
#include "opencv2/videostab/wobble_suppression.hpp"
#include "opencv2/videostab/ring_buffer.hpp"

using namespace std;

namespace cv
{
namespace videostab
{

WobbleSuppressorBase::WobbleSuppressorBase() : motions_(0), stabilizationMotions_(0)
{
    PyrLkRobustMotionEstimator *est = new PyrLkRobustMotionEstimator();
    est->setMotionModel(MM_HOMOGRAPHY);
    est->setRansacParams(RansacParams::default2dMotion(MM_HOMOGRAPHY));
}


void NullWobbleSuppressor::suppress(int /*idx*/, const Mat &frame, Mat &result)
{
    result = frame;
}


void MoreAccurateMotionWobbleSuppressor::suppress(int idx, const Mat &frame, Mat &result)
{
    CV_Assert(motions_ && stabilizationMotions_);

    if (idx % period_ == 0)
    {
        result = frame;
        return;
    }

    int k1 = idx / period_ * period_;
    int k2 = std::min(k1 + period_, frameCount_ - 1);

    Mat S1 = (*stabilizationMotions_)[idx];

    Mat_<float> ML = S1 * getMotion(k1, idx, *motions2_) * getMotion(k1, idx, *motions_).inv() * S1.inv();
    Mat_<float> MR = S1 * getMotion(idx, k2, *motions2_).inv() * getMotion(idx, k2, *motions_) * S1.inv();

    mapx_.create(frame.size());
    mapy_.create(frame.size());

    float xl, yl, zl, wl;
    float xr, yr, zr, wr;

    for (int y = 0; y < frame.rows; ++y)
    {
        for (int x = 0; x < frame.cols; ++x)
        {
            xl = ML(0,0)*x + ML(0,1)*y + ML(0,2);
            yl = ML(1,0)*x + ML(1,1)*y + ML(1,2);
            zl = ML(2,0)*x + ML(2,1)*y + ML(2,2);
            xl /= zl; yl /= zl;
            wl = idx - k1;

            xr = MR(0,0)*x + MR(0,1)*y + MR(0,2);
            yr = MR(1,0)*x + MR(1,1)*y + MR(1,2);
            zr = MR(2,0)*x + MR(2,1)*y + MR(2,2);
            xr /= zr; yr /= zr;
            wr = k2 - idx;

            mapx_(y,x) = (wr * xl + wl * xr) / (wl + wr);
            mapy_(y,x) = (wr * yl + wl * yr) / (wl + wr);
        }
    }

    if (result.data == frame.data)
        result = Mat(frame.size(), frame.type());

    remap(frame, result, mapx_, mapy_, INTER_LINEAR, BORDER_REPLICATE);
}


#if HAVE_OPENCV_GPU
void MoreAccurateMotionWobbleSuppressorGpu::suppress(int idx, const gpu::GpuMat &frame, gpu::GpuMat &result)
{
    CV_Assert(motions_ && stabilizationMotions_);

    if (idx % period_ == 0)
    {
        result = frame;
        return;
    }

    int k1 = idx / period_ * period_;
    int k2 = std::min(k1 + period_, frameCount_ - 1);

    Mat S1 = (*stabilizationMotions_)[idx];

    Mat ML = S1 * getMotion(k1, idx, *motions2_) * getMotion(k1, idx, *motions_).inv() * S1.inv();
    Mat MR = S1 * getMotion(idx, k2, *motions2_).inv() * getMotion(idx, k2, *motions_) * S1.inv();

    gpu::calcWobbleSuppressionMaps(k1, idx, k2, frame.size(), ML, MR, mapx_, mapy_);

    if (result.data == frame.data)
        result = gpu::GpuMat(frame.size(), frame.type());

    gpu::remap(frame, result, mapx_, mapy_, INTER_LINEAR, BORDER_REPLICATE);
}


void MoreAccurateMotionWobbleSuppressorGpu::suppress(int idx, const Mat &frame, Mat &result)
{
    frameDevice_.upload(frame);
    suppress(idx, frameDevice_, resultDevice_);
    resultDevice_.download(result);
}
#endif

} // namespace videostab
} // namespace cv

