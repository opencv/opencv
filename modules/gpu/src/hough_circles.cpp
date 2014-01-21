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

void cv::gpu::HoughCircles(const GpuMat&, GpuMat&, int, float, float, int, int, int, int, int) { throw_nogpu(); }
void cv::gpu::HoughCircles(const GpuMat&, GpuMat&, HoughCirclesBuf&, int, float, float, int, int, int, int, int) { throw_nogpu(); }
void cv::gpu::HoughCirclesDownload(const GpuMat&, OutputArray) { throw_nogpu(); }

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
        void circlesAccumCenters_gpu(const unsigned int* list, int count, PtrStepi dx, PtrStepi dy, PtrStepSzi accum, int minRadius, int maxRadius, float idp);
        int buildCentersList_gpu(PtrStepSzi accum, unsigned int* centers, int threshold);
        int circlesAccumRadius_gpu(const unsigned int* centers, int centersCount, const unsigned int* list, int count,
                                   float3* circles, int maxCircles, float dp, int minRadius, int maxRadius, int threshold, bool has20);
    }
}}}

void cv::gpu::HoughCircles(const GpuMat& src, GpuMat& circles, int method, float dp, float minDist, int cannyThreshold, int votesThreshold, int minRadius, int maxRadius, int maxCircles)
{
    HoughCirclesBuf buf;
    HoughCircles(src, circles, buf, method, dp, minDist, cannyThreshold, votesThreshold, minRadius, maxRadius, maxCircles);
}

void cv::gpu::HoughCircles(const GpuMat& src, GpuMat& circles, HoughCirclesBuf& buf, int method,
                           float dp, float minDist, int cannyThreshold, int votesThreshold, int minRadius, int maxRadius, int maxCircles)
{
    using namespace cv::gpu::device::hough;

    CV_Assert(src.type() == CV_8UC1);
    CV_Assert(src.cols < std::numeric_limits<unsigned short>::max());
    CV_Assert(src.rows < std::numeric_limits<unsigned short>::max());
    CV_Assert(method == CV_HOUGH_GRADIENT);
    CV_Assert(dp > 0);
    CV_Assert(minRadius > 0 && maxRadius > minRadius);
    CV_Assert(cannyThreshold > 0);
    CV_Assert(votesThreshold > 0);
    CV_Assert(maxCircles > 0);

    const float idp = 1.0f / dp;

    cv::gpu::Canny(src, buf.cannyBuf, buf.edges, std::max(cannyThreshold / 2, 1), cannyThreshold);

    ensureSizeIsEnough(2, src.size().area(), CV_32SC1, buf.list);
    unsigned int* srcPoints = buf.list.ptr<unsigned int>(0);
    unsigned int* centers = buf.list.ptr<unsigned int>(1);

    const int pointsCount = buildPointList_gpu(buf.edges, srcPoints);
    if (pointsCount == 0)
    {
        circles.release();
        return;
    }

    ensureSizeIsEnough(cvCeil(src.rows * idp) + 2, cvCeil(src.cols * idp) + 2, CV_32SC1, buf.accum);
    buf.accum.setTo(Scalar::all(0));

    circlesAccumCenters_gpu(srcPoints, pointsCount, buf.cannyBuf.dx, buf.cannyBuf.dy, buf.accum, minRadius, maxRadius, idp);

    int centersCount = buildCentersList_gpu(buf.accum, centers, votesThreshold);
    if (centersCount == 0)
    {
        circles.release();
        return;
    }

    if (minDist > 1)
    {
        cv::AutoBuffer<ushort2> oldBuf_(centersCount);
        cv::AutoBuffer<ushort2> newBuf_(centersCount);
        int newCount = 0;

        ushort2* oldBuf = oldBuf_;
        ushort2* newBuf = newBuf_;

        cudaSafeCall( cudaMemcpy(oldBuf, centers, centersCount * sizeof(ushort2), cudaMemcpyDeviceToHost) );

        const int cellSize = cvRound(minDist);
        const int gridWidth = (src.cols + cellSize - 1) / cellSize;
        const int gridHeight = (src.rows + cellSize - 1) / cellSize;

        std::vector< std::vector<ushort2> > grid(gridWidth * gridHeight);

        const float minDist2 = minDist * minDist;

        for (int i = 0; i < centersCount; ++i)
        {
            ushort2 p = oldBuf[i];

            bool good = true;

            int xCell = static_cast<int>(p.x / cellSize);
            int yCell = static_cast<int>(p.y / cellSize);

            int x1 = xCell - 1;
            int y1 = yCell - 1;
            int x2 = xCell + 1;
            int y2 = yCell + 1;

            // boundary check
            x1 = std::max(0, x1);
            y1 = std::max(0, y1);
            x2 = std::min(gridWidth - 1, x2);
            y2 = std::min(gridHeight - 1, y2);

            for (int yy = y1; yy <= y2; ++yy)
            {
                for (int xx = x1; xx <= x2; ++xx)
                {
                    vector<ushort2>& m = grid[yy * gridWidth + xx];

                    for(size_t j = 0; j < m.size(); ++j)
                    {
                        float dx = (float)(p.x - m[j].x);
                        float dy = (float)(p.y - m[j].y);

                        if (dx * dx + dy * dy < minDist2)
                        {
                            good = false;
                            goto break_out;
                        }
                    }
                }
            }

            break_out:

            if(good)
            {
                grid[yCell * gridWidth + xCell].push_back(p);

                newBuf[newCount++] = p;
            }
        }

        cudaSafeCall( cudaMemcpy(centers, newBuf, newCount * sizeof(unsigned int), cudaMemcpyHostToDevice) );
        centersCount = newCount;
    }

    ensureSizeIsEnough(1, maxCircles, CV_32FC3, circles);

    const int circlesCount = circlesAccumRadius_gpu(centers, centersCount, srcPoints, pointsCount, circles.ptr<float3>(), maxCircles,
                                                    dp, minRadius, maxRadius, votesThreshold, deviceSupports(FEATURE_SET_COMPUTE_20));

    if (circlesCount > 0)
        circles.cols = circlesCount;
    else
        circles.release();
}

void cv::gpu::HoughCirclesDownload(const GpuMat& d_circles, cv::OutputArray h_circles_)
{
    if (d_circles.empty())
    {
        h_circles_.release();
        return;
    }

    CV_Assert(d_circles.rows == 1 && d_circles.type() == CV_32FC3);

    h_circles_.create(1, d_circles.cols, CV_32FC3);
    Mat h_circles = h_circles_.getMat();
    d_circles.download(h_circles);
}

#endif /* !defined (HAVE_CUDA) */
