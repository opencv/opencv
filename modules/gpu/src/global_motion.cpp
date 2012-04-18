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

using namespace std;
using namespace cv;
using namespace cv::gpu;

#ifndef HAVE_CUDA

void cv::gpu::compactPoints(GpuMat&, GpuMat&, const GpuMat&) { throw_nogpu(); }

#else

namespace cv { namespace gpu { namespace device {

    int compactPoints(int N, float *points0, float *points1, const uchar *mask);

}}}

void cv::gpu::compactPoints(GpuMat &points0, GpuMat &points1, const GpuMat &mask)
{
    CV_Assert(points0.rows == 1 && points1.rows == 1 && mask.rows == 1);
    CV_Assert(points0.type() == CV_32FC2 && points1.type() == CV_32FC2 && mask.type() == CV_8U);
    CV_Assert(points0.cols == mask.cols && points1.cols == mask.cols);

    int npoints = points0.cols;
    int remaining = cv::gpu::device::compactPoints(
            npoints, (float*)points0.data, (float*)points1.data, mask.data);

    points0 = points0.colRange(0, remaining);
    points1 = points1.colRange(0, remaining);
}

#endif
