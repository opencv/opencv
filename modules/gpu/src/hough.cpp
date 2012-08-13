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

namespace cv { namespace gpu { namespace device
{
    namespace hough
    {
        void linesAccum_gpu(DevMem2Db src, PtrStep_<uint> accum, float theta, int numangle, int numrho, float irho);
        int linesGetResult_gpu(DevMem2D_<uint> accum, float2* out, int* voices, int maxSize, float threshold, float theta, float rho, bool doSort);
    }
}}}

void cv::gpu::HoughLinesTransform(const GpuMat& src, GpuMat& accum, float rho, float theta)
{
    using namespace cv::gpu::device;

    CV_Assert(src.type() == CV_8UC1);

    const int numangle = cvRound(CV_PI / theta);
    const int numrho = cvRound(((src.cols + src.rows) * 2 + 1) / rho);
    const float irho = 1.0f / rho;

    accum.create(numangle + 2, numrho + 2, CV_32SC1);
    accum.setTo(cv::Scalar::all(0));

    hough::linesAccum_gpu(src, accum, theta, numangle, numrho, irho);
}

void cv::gpu::HoughLinesGet(const GpuMat& accum, GpuMat& lines, float rho, float theta, int threshold, bool doSort, int maxLines)
{
    using namespace cv::gpu::device;

    CV_Assert(accum.type() == CV_32SC1);

    lines.create(2, maxLines, CV_32FC2);
    lines.cols = hough::linesGetResult_gpu(accum, lines.ptr<float2>(0), lines.ptr<int>(1), maxLines, threshold, theta, rho, doSort);
}

void cv::gpu::HoughLines(const GpuMat& src, GpuMat& lines, float rho, float theta, int threshold, bool doSort, int maxLines)
{
    cv::gpu::GpuMat accum;
    HoughLines(src, lines, accum, rho, theta, threshold, doSort, maxLines);
}

void cv::gpu::HoughLines(const GpuMat& src, GpuMat& lines, GpuMat& accum, float rho, float theta, int threshold, bool doSort, int maxLines)
{
    HoughLinesTransform(src, accum, rho, theta);
    HoughLinesGet(accum, lines, rho, theta, threshold, doSort, maxLines);
}

void cv::gpu::HoughLinesDownload(const GpuMat& d_lines, OutputArray h_lines_, OutputArray h_voices_)
{
    h_lines_.create(1, d_lines.cols, CV_32FC2);
    cv::Mat h_lines = h_lines_.getMat();
    d_lines.row(0).download(h_lines);

    if (h_voices_.needed())
    {
        h_voices_.create(1, d_lines.cols, CV_32SC1);
        cv::Mat h_voices = h_voices_.getMat();
        cv::gpu::GpuMat d_voices(1, d_lines.cols, CV_32SC1, const_cast<int*>(d_lines.ptr<int>(1)));
        d_voices.download(h_voices);
    }
}
