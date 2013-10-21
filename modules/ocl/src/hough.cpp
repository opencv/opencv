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
#include "opencl_kernels.hpp"

using namespace cv;
using namespace cv::ocl;

#if !defined (HAVE_OPENCL)

void cv::ocl::HoughCircles(const oclMat&, oclMat&, int, float, float, int, int, int, int, int) { throw_nogpu(); }
void cv::ocl::HoughCircles(const oclMat&, oclMat&, HoughCirclesBuf&, int, float, float, int, int, int, int, int) { throw_nogpu(); }
void cv::ocl::HoughCirclesDownload(const oclMat&, OutputArray) { throw_nogpu(); }

#else /* !defined (HAVE_OPENCL) */

#define MUL_UP(a, b) ((a)/(b)+1)*(b)

//////////////////////////////////////////////////////////
// common functions

namespace
{
    int buildPointList_gpu(const oclMat& src, oclMat& list)
    {
        const int PIXELS_PER_THREAD = 16;

        int totalCount = 0;
        int err = CL_SUCCESS;
        cl_mem counter = clCreateBuffer(*(cl_context*)src.clCxt->getOpenCLContextPtr(),
                                        CL_MEM_COPY_HOST_PTR,
                                        sizeof(int),
                                        &totalCount,
                                        &err);
        openCLSafeCall(err);

        const size_t blkSizeX = 32;
        const size_t blkSizeY = 4;
        size_t localThreads[3] = { blkSizeX, blkSizeY, 1 };

        const int PIXELS_PER_BLOCK = blkSizeX * PIXELS_PER_THREAD;
        const size_t glbSizeX = src.cols % (PIXELS_PER_BLOCK) == 0 ? src.cols : MUL_UP(src.cols, PIXELS_PER_BLOCK);
        const size_t glbSizeY = src.rows % blkSizeY == 0 ? src.rows : MUL_UP(src.rows, blkSizeY);
        size_t globalThreads[3] = { glbSizeX, glbSizeY, 1 };

        std::vector<std::pair<size_t , const void *> > args;
        args.push_back( std::make_pair( sizeof(cl_mem)  , (void *)&src.data ));
        args.push_back( std::make_pair( sizeof(cl_int)  , (void *)&src.cols ));
        args.push_back( std::make_pair( sizeof(cl_int)  , (void *)&src.rows ));
        args.push_back( std::make_pair( sizeof(cl_int)  , (void *)&src.step ));
        args.push_back( std::make_pair( sizeof(cl_mem)  , (void *)&list.data ));
        args.push_back( std::make_pair( sizeof(cl_mem)  , (void *)&counter ));

        // WARNING: disabled until
        openCLExecuteKernel(src.clCxt, &imgproc_hough, "buildPointList", globalThreads, localThreads, args, -1, -1);
        openCLSafeCall(clEnqueueReadBuffer(*(cl_command_queue*)src.clCxt->getOpenCLCommandQueuePtr(), counter, CL_TRUE, 0, sizeof(int), &totalCount, 0, NULL, NULL));
        openCLSafeCall(clReleaseMemObject(counter));

        return totalCount;
    }
}

//////////////////////////////////////////////////////////
// HoughCircles

namespace
{
    void circlesAccumCenters_gpu(const oclMat& list, int count, const oclMat& dx, const oclMat& dy, oclMat& accum, int minRadius, int maxRadius, float idp)
    {
        const size_t blkSizeX = 256;
        size_t localThreads[3] = { 256, 1, 1 };

        const size_t glbSizeX = count % blkSizeX == 0 ? count : MUL_UP(count, blkSizeX);
        size_t globalThreads[3] = { glbSizeX, 1, 1 };

        const int width  = accum.cols - 2;
        const int height = accum.rows - 2;

        std::vector<std::pair<size_t , const void *> > args;
        args.push_back( std::make_pair( sizeof(cl_mem)  , (void *)&list.data ));
        args.push_back( std::make_pair( sizeof(cl_int)  , (void *)&count ));
        args.push_back( std::make_pair( sizeof(cl_mem)  , (void *)&dx.data ));
        args.push_back( std::make_pair( sizeof(cl_int)  , (void *)&dx.step ));
        args.push_back( std::make_pair( sizeof(cl_mem)  , (void *)&dy.data ));
        args.push_back( std::make_pair( sizeof(cl_int)  , (void *)&dy.step ));
        args.push_back( std::make_pair( sizeof(cl_mem)  , (void *)&accum.data ));
        args.push_back( std::make_pair( sizeof(cl_int)  , (void *)&accum.step ));
        args.push_back( std::make_pair( sizeof(cl_int)  , (void *)&width ));
        args.push_back( std::make_pair( sizeof(cl_int)  , (void *)&height ));
        args.push_back( std::make_pair( sizeof(cl_int)  , (void *)&minRadius));
        args.push_back( std::make_pair( sizeof(cl_int)  , (void *)&maxRadius));
        args.push_back( std::make_pair( sizeof(cl_float), (void *)&idp));

        openCLExecuteKernel(accum.clCxt, &imgproc_hough, "circlesAccumCenters", globalThreads, localThreads, args, -1, -1);
    }

    int buildCentersList_gpu(const oclMat& accum, oclMat& centers, int threshold)
    {
        int totalCount = 0;
        int err = CL_SUCCESS;
        cl_mem counter = clCreateBuffer(*(cl_context*)accum.clCxt->getOpenCLContextPtr(),
                                        CL_MEM_COPY_HOST_PTR,
                                        sizeof(int),
                                        &totalCount,
                                        &err);
        openCLSafeCall(err);

        const size_t blkSizeX = 32;
        const size_t blkSizeY = 8;
        size_t localThreads[3] = { blkSizeX, blkSizeY, 1 };

        const size_t glbSizeX = (accum.cols - 2) % blkSizeX == 0 ? accum.cols - 2 : MUL_UP(accum.cols - 2, blkSizeX);
        const size_t glbSizeY = (accum.rows - 2) % blkSizeY == 0 ? accum.rows - 2 : MUL_UP(accum.rows - 2, blkSizeY);
        size_t globalThreads[3] = { glbSizeX, glbSizeY, 1 };

        std::vector<std::pair<size_t , const void *> > args;
        args.push_back( std::make_pair( sizeof(cl_mem)  , (void *)&accum.data ));
        args.push_back( std::make_pair( sizeof(cl_int)  , (void *)&accum.cols ));
        args.push_back( std::make_pair( sizeof(cl_int)  , (void *)&accum.rows ));
        args.push_back( std::make_pair( sizeof(cl_int)  , (void *)&accum.step ));
        args.push_back( std::make_pair( sizeof(cl_mem)  , (void *)&centers.data ));
        args.push_back( std::make_pair( sizeof(cl_int)  , (void *)&threshold ));
        args.push_back( std::make_pair( sizeof(cl_mem)  , (void *)&counter ));

        openCLExecuteKernel(accum.clCxt, &imgproc_hough, "buildCentersList", globalThreads, localThreads, args, -1, -1);

        openCLSafeCall(clEnqueueReadBuffer(*(cl_command_queue*)accum.clCxt->getOpenCLCommandQueuePtr(), counter, CL_TRUE, 0, sizeof(int), &totalCount, 0, NULL, NULL));
        openCLSafeCall(clReleaseMemObject(counter));

        return totalCount;
    }

    int circlesAccumRadius_gpu(const oclMat& centers, int centersCount,
                               const oclMat& list, int count,
                               oclMat& circles, int maxCircles,
                               float dp, int minRadius, int maxRadius, int threshold)
    {
        int totalCount = 0;
        int err = CL_SUCCESS;
        cl_mem counter = clCreateBuffer(*(cl_context*)circles.clCxt->getOpenCLContextPtr(),
                                        CL_MEM_COPY_HOST_PTR,
                                        sizeof(int),
                                        &totalCount,
                                        &err);
        openCLSafeCall(err);

        const size_t blkSizeX = circles.clCxt->getDeviceInfo().maxWorkGroupSize;
        size_t localThreads[3] = { blkSizeX, 1, 1 };

        const size_t glbSizeX = centersCount * blkSizeX;
        size_t globalThreads[3] = { glbSizeX, 1, 1 };

        const int histSize = maxRadius - minRadius + 1;
        size_t smemSize = (histSize + 2) * sizeof(int);

        std::vector<std::pair<size_t , const void *> > args;
        args.push_back( std::make_pair( sizeof(cl_mem)  , (void *)&centers.data ));
        args.push_back( std::make_pair( sizeof(cl_mem)  , (void *)&list.data ));
        args.push_back( std::make_pair( sizeof(cl_int)  , (void *)&count ));
        args.push_back( std::make_pair( sizeof(cl_mem)  , (void *)&circles.data ));
        args.push_back( std::make_pair( sizeof(cl_int)  , (void *)&maxCircles ));
        args.push_back( std::make_pair( sizeof(cl_float), (void *)&dp ));
        args.push_back( std::make_pair( sizeof(cl_int)  , (void *)&minRadius ));
        args.push_back( std::make_pair( sizeof(cl_int)  , (void *)&maxRadius ));
        args.push_back( std::make_pair( sizeof(cl_int)  , (void *)&histSize ));
        args.push_back( std::make_pair( sizeof(cl_int)  , (void *)&threshold ));
        args.push_back( std::make_pair( smemSize        , (void *)NULL ));
        args.push_back( std::make_pair( sizeof(cl_mem)  , (void *)&counter ));

        CV_Assert(circles.offset == 0);

        openCLExecuteKernel(circles.clCxt, &imgproc_hough, "circlesAccumRadius", globalThreads, localThreads, args, -1, -1);

        openCLSafeCall(clEnqueueReadBuffer(*(cl_command_queue*)circles.clCxt->getOpenCLCommandQueuePtr(), counter, CL_TRUE, 0, sizeof(int), &totalCount, 0, NULL, NULL));

        openCLSafeCall(clReleaseMemObject(counter));

        totalCount = std::min(totalCount, maxCircles);

        return totalCount;
    }


} // namespace



void cv::ocl::HoughCircles(const oclMat& src, oclMat& circles, int method, float dp, float minDist, int cannyThreshold, int votesThreshold, int minRadius, int maxRadius, int maxCircles)
{
    HoughCirclesBuf buf;
    HoughCircles(src, circles, buf, method, dp, minDist, cannyThreshold, votesThreshold, minRadius, maxRadius, maxCircles);
}

void cv::ocl::HoughCircles(const oclMat& src, oclMat& circles, HoughCirclesBuf& buf, int method,
                           float dp, float minDist, int cannyThreshold, int votesThreshold, int minRadius, int maxRadius, int maxCircles)
{
    CV_Assert(src.type() == CV_8UC1);
    CV_Assert(src.cols < std::numeric_limits<unsigned short>::max());
    CV_Assert(src.rows < std::numeric_limits<unsigned short>::max());
    CV_Assert(method == HOUGH_GRADIENT);
    CV_Assert(dp > 0);
    CV_Assert(minRadius > 0 && maxRadius > minRadius);
    CV_Assert(cannyThreshold > 0);
    CV_Assert(votesThreshold > 0);
    CV_Assert(maxCircles > 0);

    const float idp = 1.0f / dp;

    cv::ocl::Canny(src, buf.cannyBuf, buf.edges, std::max(cannyThreshold / 2, 1), cannyThreshold);

    ensureSizeIsEnough(1, src.size().area(), CV_32SC1, buf.srcPoints);
    const int pointsCount = buildPointList_gpu(buf.edges, buf.srcPoints);
    if (pointsCount == 0)
    {
        circles.release();
        return;
    }

    ensureSizeIsEnough(cvCeil(src.rows * idp) + 2, cvCeil(src.cols * idp) + 2, CV_32SC1, buf.accum);
    buf.accum.setTo(Scalar::all(0));

    circlesAccumCenters_gpu(buf.srcPoints, pointsCount, buf.cannyBuf.dx, buf.cannyBuf.dy, buf.accum, minRadius, maxRadius, idp);

    ensureSizeIsEnough(1, src.size().area(), CV_32SC1, buf.centers);
    int centersCount = buildCentersList_gpu(buf.accum, buf.centers, votesThreshold);
    if (centersCount == 0)
    {
        circles.release();
        return;
    }

    if (minDist > 1)
    {
        cv::AutoBuffer<unsigned int> oldBuf_(centersCount);
        cv::AutoBuffer<unsigned int> newBuf_(centersCount);
        int newCount = 0;

        unsigned int* oldBuf = oldBuf_;
        unsigned int* newBuf = newBuf_;

        openCLSafeCall(clEnqueueReadBuffer(*(cl_command_queue*)buf.centers.clCxt->getOpenCLCommandQueuePtr(),
                                           (cl_mem)buf.centers.data,
                                           CL_TRUE,
                                           0,
                                           centersCount * sizeof(unsigned int),
                                           oldBuf,
                                           0,
                                           NULL,
                                           NULL));


        const int cellSize = cvRound(minDist);
        const int gridWidth = (src.cols + cellSize - 1) / cellSize;
        const int gridHeight = (src.rows + cellSize - 1) / cellSize;

        std::vector< std::vector<unsigned int> > grid(gridWidth * gridHeight);

        const float minDist2 = minDist * minDist;

        for (int i = 0; i < centersCount; ++i)
        {
            unsigned int p = oldBuf[i];
            const int px = p & 0xFFFF;
            const int py = (p >> 16) & 0xFFFF;

            bool good = true;

            int xCell = static_cast<int>(px / cellSize);
            int yCell = static_cast<int>(py / cellSize);

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
                    std::vector<unsigned int>& m = grid[yy * gridWidth + xx];

                    for(size_t j = 0; j < m.size(); ++j)
                    {
                        const int val = m[j];
                        const int jx = val & 0xFFFF;
                        const int jy = (val >> 16) & 0xFFFF;

                        float dx = (float)(px - jx);
                        float dy = (float)(py - jy);

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

        openCLSafeCall(clEnqueueWriteBuffer(*(cl_command_queue*)buf.centers.clCxt->getOpenCLCommandQueuePtr(),
                                            (cl_mem)buf.centers.data,
                                            CL_TRUE,
                                            0,
                                            newCount * sizeof(unsigned int),
                                            newBuf,
                                            0,
                                            0,
                                            0));
        centersCount = newCount;
    }

    ensureSizeIsEnough(1, maxCircles, CV_32FC3, circles);

    const int circlesCount = circlesAccumRadius_gpu(buf.centers, centersCount,
                                                           buf.srcPoints, pointsCount,
                                                           circles, maxCircles,
                                                           dp, minRadius, maxRadius, votesThreshold);

    if (circlesCount > 0)
        circles.cols = circlesCount;
    else
        circles.release();
}

void cv::ocl::HoughCirclesDownload(const oclMat& d_circles, cv::OutputArray h_circles_)
{
    // FIX ME: garbage values are copied!
    CV_Error(Error::StsNotImplemented, "HoughCirclesDownload is not implemented");

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

#endif /* !defined (HAVE_OPENCL) */
