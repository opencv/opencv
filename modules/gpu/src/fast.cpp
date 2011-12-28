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
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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
//     and/or other GpuMaterials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or bpied warranties, including, but not limited to, the bpied
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

using namespace cv;
using namespace cv::gpu;
using namespace std;

#if !defined (HAVE_CUDA)

cv::gpu::FAST_GPU::FAST_GPU(int, bool, double) { throw_nogpu(); }
void cv::gpu::FAST_GPU::operator ()(const GpuMat&, const GpuMat&, GpuMat&) { throw_nogpu(); }
void cv::gpu::FAST_GPU::operator ()(const GpuMat&, const GpuMat&, std::vector<KeyPoint>&) { throw_nogpu(); }
void cv::gpu::FAST_GPU::downloadKeypoints(const GpuMat&, std::vector<KeyPoint>&) { throw_nogpu(); }
void cv::gpu::FAST_GPU::convertKeypoints(const Mat&, std::vector<KeyPoint>&) { throw_nogpu(); }
void cv::gpu::FAST_GPU::release() { throw_nogpu(); }
int cv::gpu::FAST_GPU::calcKeyPointsLocation(const GpuMat&, const GpuMat&) { throw_nogpu(); return 0; }
int cv::gpu::FAST_GPU::getKeyPoints(GpuMat&) { throw_nogpu(); return 0; }

#else /* !defined (HAVE_CUDA) */

cv::gpu::FAST_GPU::FAST_GPU(int _threshold, bool _nonmaxSupression, double _keypointsRatio) : 
    nonmaxSupression(_nonmaxSupression), threshold(_threshold), keypointsRatio(_keypointsRatio), count_(0)
{
}

void cv::gpu::FAST_GPU::operator ()(const GpuMat& image, const GpuMat& mask, std::vector<KeyPoint>& keypoints)
{
    if (image.empty())
        return;

    (*this)(image, mask, d_keypoints_);
    downloadKeypoints(d_keypoints_, keypoints);
}

void cv::gpu::FAST_GPU::downloadKeypoints(const GpuMat& d_keypoints, std::vector<KeyPoint>& keypoints)
{
    if (d_keypoints.empty())
        return;

    Mat h_keypoints(d_keypoints);
    convertKeypoints(h_keypoints, keypoints);
}

void cv::gpu::FAST_GPU::convertKeypoints(const Mat& h_keypoints, std::vector<KeyPoint>& keypoints)
{
    if (h_keypoints.empty())
        return;

    CV_Assert(h_keypoints.rows == ROWS_COUNT && h_keypoints.elemSize() == 4);

    int npoints = h_keypoints.cols;

    keypoints.resize(npoints);

    const short2* loc_row = h_keypoints.ptr<short2>(LOCATION_ROW);
    const float* response_row = h_keypoints.ptr<float>(RESPONSE_ROW);

    for (int i = 0; i < npoints; ++i)
    {
        KeyPoint kp(loc_row[i].x, loc_row[i].y, static_cast<float>(FEATURE_SIZE), -1, response_row[i]);
        keypoints[i] = kp;
    }
}

void cv::gpu::FAST_GPU::operator ()(const GpuMat& img, const GpuMat& mask, GpuMat& keypoints)
{
    calcKeyPointsLocation(img, mask);
    keypoints.cols = getKeyPoints(keypoints);
}

namespace cv { namespace gpu { namespace device 
{
    namespace fast 
    {
        int calcKeypoints_gpu(DevMem2Db img, DevMem2Db mask, short2* kpLoc, int maxKeypoints, DevMem2Di score, int threshold);
        int nonmaxSupression_gpu(const short2* kpLoc, int count, DevMem2Di score, short2* loc, float* response);
    }
}}}

int cv::gpu::FAST_GPU::calcKeyPointsLocation(const GpuMat& img, const GpuMat& mask)
{
    using namespace cv::gpu::device::fast;

    CV_Assert(img.type() == CV_8UC1);
    CV_Assert(mask.empty() || (mask.type() == CV_8UC1 && mask.size() == img.size()));
    CV_Assert(TargetArchs::builtWith(GLOBAL_ATOMICS) && DeviceInfo().supports(GLOBAL_ATOMICS));

    int maxKeypoints = static_cast<int>(keypointsRatio * img.size().area());

    ensureSizeIsEnough(1, maxKeypoints, CV_16SC2, kpLoc_);

    if (nonmaxSupression)
    {
        ensureSizeIsEnough(img.size(), CV_32SC1, score_);
        score_.setTo(Scalar::all(0));
    }

    count_ = calcKeypoints_gpu(img, mask, kpLoc_.ptr<short2>(), maxKeypoints, nonmaxSupression ? score_ : DevMem2Di(), threshold);
    count_ = std::min(count_, maxKeypoints);

    return count_;
}

int cv::gpu::FAST_GPU::getKeyPoints(GpuMat& keypoints)
{
    using namespace cv::gpu::device::fast;

    CV_Assert(TargetArchs::builtWith(GLOBAL_ATOMICS) && DeviceInfo().supports(GLOBAL_ATOMICS));

    if (count_ == 0)
        return 0;

    ensureSizeIsEnough(ROWS_COUNT, count_, CV_32FC1, keypoints);

    if (nonmaxSupression)
        return nonmaxSupression_gpu(kpLoc_.ptr<short2>(), count_, score_, keypoints.ptr<short2>(LOCATION_ROW), keypoints.ptr<float>(RESPONSE_ROW));

    GpuMat locRow(1, count_, kpLoc_.type(), keypoints.ptr(0));
    kpLoc_.colRange(0, count_).copyTo(locRow);
    keypoints.row(1).setTo(Scalar::all(0));

    return count_;    
}

void cv::gpu::FAST_GPU::release()
{
    kpLoc_.release();
    score_.release();

    d_keypoints_.release();
}

#endif /* !defined (HAVE_CUDA) */
