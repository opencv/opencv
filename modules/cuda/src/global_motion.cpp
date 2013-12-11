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

using namespace cv;
using namespace cv::cuda;

#if !defined HAVE_CUDA || defined(CUDA_DISABLER)

void cv::cuda::compactPoints(GpuMat&, GpuMat&, const GpuMat&) { throw_no_cuda(); }
void cv::cuda::calcWobbleSuppressionMaps(
        int, int, int, Size, const Mat&, const Mat&, GpuMat&, GpuMat&) { throw_no_cuda(); }

#else

namespace cv { namespace cuda { namespace device { namespace globmotion {

    int compactPoints(int N, float *points0, float *points1, const uchar *mask);

    void calcWobbleSuppressionMaps(
            int left, int idx, int right, int width, int height,
            const float *ml, const float *mr, PtrStepSzf mapx, PtrStepSzf mapy);

}}}}

void cv::cuda::compactPoints(GpuMat &points0, GpuMat &points1, const GpuMat &mask)
{
    CV_Assert(points0.rows == 1 && points1.rows == 1 && mask.rows == 1);
    CV_Assert(points0.type() == CV_32FC2 && points1.type() == CV_32FC2 && mask.type() == CV_8U);
    CV_Assert(points0.cols == mask.cols && points1.cols == mask.cols);

    int npoints = points0.cols;
    int remaining = cv::cuda::device::globmotion::compactPoints(
            npoints, (float*)points0.data, (float*)points1.data, mask.data);

    points0 = points0.colRange(0, remaining);
    points1 = points1.colRange(0, remaining);
}


void cv::cuda::calcWobbleSuppressionMaps(
        int left, int idx, int right, Size size, const Mat &ml, const Mat &mr,
        GpuMat &mapx, GpuMat &mapy)
{
    CV_Assert(ml.size() == Size(3, 3) && ml.type() == CV_32F && ml.isContinuous());
    CV_Assert(mr.size() == Size(3, 3) && mr.type() == CV_32F && mr.isContinuous());

    mapx.create(size, CV_32F);
    mapy.create(size, CV_32F);

    cv::cuda::device::globmotion::calcWobbleSuppressionMaps(
                left, idx, right, size.width, size.height,
                ml.ptr<float>(), mr.ptr<float>(), mapx, mapy);
}

#endif
