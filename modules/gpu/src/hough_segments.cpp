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

using namespace std;
using namespace cv;
using namespace cv::gpu;

#if !defined (HAVE_CUDA) || defined (CUDA_DISABLER)

void cv::gpu::HoughLinesP(const GpuMat&, GpuMat&, HoughLinesBuf&, float, float, int, int, int) { throw_nogpu(); }

#else /* !defined (HAVE_CUDA) */

namespace cv { namespace gpu { namespace device
{
    namespace hough
    {
        int buildPointList_gpu(PtrStepSzb src, unsigned int* list);
    }
}}}

namespace cv { namespace gpu { namespace device
{
    namespace hough
    {
        void linesAccum_gpu(const unsigned int* list, int count, PtrStepSzi accum, float rho, float theta, size_t sharedMemPerBlock, bool has20);
        int houghLinesProbabilistic_gpu(PtrStepSzb mask, PtrStepSzi accum, int4* out, int maxSize, float rho, float theta, int lineGap, int lineLength);
    }
}}}

void cv::gpu::HoughLinesP(const GpuMat& src, GpuMat& lines, HoughLinesBuf& buf, float rho, float theta, int minLineLength, int maxLineGap, int maxLines)
{
    using namespace cv::gpu::device::hough;

    CV_Assert( src.type() == CV_8UC1 );
    CV_Assert( src.cols < std::numeric_limits<unsigned short>::max() );
    CV_Assert( src.rows < std::numeric_limits<unsigned short>::max() );

    ensureSizeIsEnough(1, src.size().area(), CV_32SC1, buf.list);
    unsigned int* srcPoints = buf.list.ptr<unsigned int>();

    const int pointsCount = buildPointList_gpu(src, srcPoints);
    if (pointsCount == 0)
    {
        lines.release();
        return;
    }

    const int numangle = cvRound(CV_PI / theta);
    const int numrho = cvRound(((src.cols + src.rows) * 2 + 1) / rho);
    CV_Assert( numangle > 0 && numrho > 0 );

    ensureSizeIsEnough(numangle + 2, numrho + 2, CV_32SC1, buf.accum);
    buf.accum.setTo(Scalar::all(0));

    DeviceInfo devInfo;
    linesAccum_gpu(srcPoints, pointsCount, buf.accum, rho, theta, devInfo.sharedMemPerBlock(), devInfo.supports(FEATURE_SET_COMPUTE_20));

    ensureSizeIsEnough(1, maxLines, CV_32SC4, lines);

    int linesCount = houghLinesProbabilistic_gpu(src, buf.accum, lines.ptr<int4>(), maxLines, rho, theta, maxLineGap, minLineLength);

    if (linesCount > 0)
        lines.cols = linesCount;
    else
        lines.release();
}

#endif /* !defined (HAVE_CUDA) */
