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

#if !defined (HAVE_CUDA)

void cv::gpu::HoughLinesTransform(const GpuMat&, GpuMat&, GpuMat&, float, float) { throw_nogpu(); }
void cv::gpu::HoughLinesGet(const GpuMat&, GpuMat&, float, float, int, bool, int) { throw_nogpu(); }
void cv::gpu::HoughLines(const GpuMat&, GpuMat&, float, float, int, bool, int) { throw_nogpu(); }
void cv::gpu::HoughLines(const GpuMat&, GpuMat&, GpuMat&, GpuMat&, float, float, int, bool, int) { throw_nogpu(); }
void cv::gpu::HoughLinesDownload(const GpuMat&, OutputArray, OutputArray) { throw_nogpu(); }

#else /* !defined (HAVE_CUDA) */

namespace cv { namespace gpu { namespace device
{
    namespace hough
    {
        int buildPointList_gpu(DevMem2Db src, unsigned int* list);
        void linesAccum_gpu(const unsigned int* list, int count, DevMem2Di accum, float rho, float theta, size_t sharedMemPerBlock, bool has20);
        int linesGetResult_gpu(DevMem2Di accum, float2* out, int* votes, int maxSize, float rho, float theta, float threshold, bool doSort);
    }
}}}

void cv::gpu::HoughLinesTransform(const GpuMat& src, GpuMat& accum, GpuMat& buf, float rho, float theta)
{
    using namespace cv::gpu::device::hough;

    CV_Assert(src.type() == CV_8UC1);
    CV_Assert(src.cols < std::numeric_limits<unsigned short>::max());
    CV_Assert(src.rows < std::numeric_limits<unsigned short>::max());

    ensureSizeIsEnough(1, src.size().area(), CV_32SC1, buf);

    const int count = buildPointList_gpu(src, buf.ptr<unsigned int>());

    const int numangle = cvRound(CV_PI / theta);
    const int numrho = cvRound(((src.cols + src.rows) * 2 + 1) / rho);

    CV_Assert(numangle > 0 && numrho > 0);

    ensureSizeIsEnough(numangle + 2, numrho + 2, CV_32SC1, accum);
    accum.setTo(cv::Scalar::all(0));

    cv::gpu::DeviceInfo devInfo;

    if (count > 0)
        linesAccum_gpu(buf.ptr<unsigned int>(), count, accum, rho, theta, devInfo.sharedMemPerBlock(), devInfo.supports(cv::gpu::FEATURE_SET_COMPUTE_20));
}

void cv::gpu::HoughLinesGet(const GpuMat& accum, GpuMat& lines, float rho, float theta, int threshold, bool doSort, int maxLines)
{
    using namespace cv::gpu::device;

    CV_Assert(accum.type() == CV_32SC1);

    ensureSizeIsEnough(2, maxLines, CV_32FC2, lines);

    int count = hough::linesGetResult_gpu(accum, lines.ptr<float2>(0), lines.ptr<int>(1), maxLines, rho, theta, threshold, doSort);

    if (count > 0)
        lines.cols = count;
    else
        lines.release();
}

void cv::gpu::HoughLines(const GpuMat& src, GpuMat& lines, float rho, float theta, int threshold, bool doSort, int maxLines)
{
    cv::gpu::GpuMat accum, buf;
    HoughLines(src, lines, accum, buf, rho, theta, threshold, doSort, maxLines);
}

void cv::gpu::HoughLines(const GpuMat& src, GpuMat& lines, GpuMat& accum, GpuMat& buf, float rho, float theta, int threshold, bool doSort, int maxLines)
{
    HoughLinesTransform(src, accum, buf, rho, theta);
    HoughLinesGet(accum, lines, rho, theta, threshold, doSort, maxLines);
}

void cv::gpu::HoughLinesDownload(const GpuMat& d_lines, OutputArray h_lines_, OutputArray h_votes_)
{
    if (d_lines.empty())
    {
        h_lines_.release();
        if (h_votes_.needed())
            h_votes_.release();
        return;
    }

    CV_Assert(d_lines.rows == 2 && d_lines.type() == CV_32FC2);

    h_lines_.create(1, d_lines.cols, CV_32FC2);
    cv::Mat h_lines = h_lines_.getMat();
    d_lines.row(0).download(h_lines);

    if (h_votes_.needed())
    {
        h_votes_.create(1, d_lines.cols, CV_32SC1);
        cv::Mat h_votes = h_votes_.getMat();
        cv::gpu::GpuMat d_votes(1, d_lines.cols, CV_32SC1, const_cast<int*>(d_lines.ptr<int>(1)));
        d_votes.download(h_votes);
    }
}

#endif /* !defined (HAVE_CUDA) */
