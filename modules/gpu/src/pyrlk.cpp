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

#if !defined (HAVE_CUDA)

void cv::gpu::PyrLKOpticalFlow::sparse(const GpuMat&, const GpuMat&, const GpuMat&, GpuMat&, GpuMat&, GpuMat*) { throw_nogpu(); }
void cv::gpu::PyrLKOpticalFlow::dense(const GpuMat&, const GpuMat&, GpuMat&, GpuMat&, GpuMat*) { throw_nogpu(); }

#else /* !defined (HAVE_CUDA) */

namespace cv { namespace gpu { namespace device
{
    namespace pyrlk
    {
        void loadConstants(int2 winSize, int iters);

        void lkSparse1_gpu(DevMem2Df I, DevMem2Df J, const float2* prevPts, float2* nextPts, uchar* status, float* err, int ptcount,
            int level, dim3 block, dim3 patch, cudaStream_t stream = 0);
        void lkSparse4_gpu(DevMem2D_<float4> I, DevMem2D_<float4> J, const float2* prevPts, float2* nextPts, uchar* status, float* err, int ptcount,
            int level, dim3 block, dim3 patch, cudaStream_t stream = 0);

        void lkDense_gpu(DevMem2Db I, DevMem2Df J, DevMem2Df u, DevMem2Df v, DevMem2Df prevU, DevMem2Df prevV,
                         DevMem2Df err, int2 winSize, cudaStream_t stream = 0);
    }
}}}

namespace
{
    void calcPatchSize(cv::Size winSize, dim3& block, dim3& patch, bool isDeviceArch11)
    {
        if (winSize.width > 32 && winSize.width > 2 * winSize.height)
        {
            block.x = isDeviceArch11 ? 16 : 32;
            block.y = 8;
        }
        else
        {
            block.x = 16;
            block.y = isDeviceArch11 ? 8 : 16;
        }

        patch.x = (winSize.width  + block.x - 1) / block.x;
        patch.y = (winSize.height + block.y - 1) / block.y;

        block.z = patch.z = 1;
    }
}

void cv::gpu::PyrLKOpticalFlow::sparse(const GpuMat& prevImg, const GpuMat& nextImg, const GpuMat& prevPts, GpuMat& nextPts, GpuMat& status, GpuMat* err)
{
    using namespace cv::gpu::device::pyrlk;

    if (prevPts.empty())
    {
        nextPts.release();
        status.release();
        if (err) err->release();
        return;
    }

    dim3 block, patch;
    calcPatchSize(winSize, block, patch, isDeviceArch11_);

    CV_Assert(prevImg.type() == CV_8UC1 || prevImg.type() == CV_8UC3 || prevImg.type() == CV_8UC4);
    CV_Assert(prevImg.size() == nextImg.size() && prevImg.type() == nextImg.type());
    CV_Assert(maxLevel >= 0);
    CV_Assert(winSize.width > 2 && winSize.height > 2);
    CV_Assert(patch.x > 0 && patch.x < 6 && patch.y > 0 && patch.y < 6);
    CV_Assert(prevPts.rows == 1 && prevPts.type() == CV_32FC2);

    if (useInitialFlow)
        CV_Assert(nextPts.size() == prevPts.size() && nextPts.type() == CV_32FC2);
    else
        ensureSizeIsEnough(1, prevPts.cols, prevPts.type(), nextPts);

    GpuMat temp1 = (useInitialFlow ? nextPts : prevPts).reshape(1);
    GpuMat temp2 = nextPts.reshape(1);
    multiply(temp1, Scalar::all(1.0 / (1 << maxLevel) / 2.0), temp2);

    ensureSizeIsEnough(1, prevPts.cols, CV_8UC1, status);
    status.setTo(Scalar::all(1));

    if (err)
        ensureSizeIsEnough(1, prevPts.cols, CV_32FC1, *err);

    // build the image pyramids.

    prevPyr_.resize(maxLevel + 1);
    nextPyr_.resize(maxLevel + 1);

    int cn = prevImg.channels();

    if (cn == 1 || cn == 4)
    {
        prevImg.convertTo(prevPyr_[0], CV_32F);
        nextImg.convertTo(nextPyr_[0], CV_32F);
    }
    else
    {
        cvtColor(prevImg, dx_calcBuf_, COLOR_BGR2BGRA);
        dx_calcBuf_.convertTo(prevPyr_[0], CV_32F);

        cvtColor(nextImg, dx_calcBuf_, COLOR_BGR2BGRA);
        dx_calcBuf_.convertTo(nextPyr_[0], CV_32F);
    }

    for (int level = 1; level <= maxLevel; ++level)
    {
        pyrDown(prevPyr_[level - 1], prevPyr_[level]);
        pyrDown(nextPyr_[level - 1], nextPyr_[level]);
    }

    loadConstants(make_int2(winSize.width, winSize.height), iters);

    for (int level = maxLevel; level >= 0; level--)
    {
        if (cn == 1)
        {
            lkSparse1_gpu(prevPyr_[level], nextPyr_[level],
                prevPts.ptr<float2>(), nextPts.ptr<float2>(), status.ptr(), level == 0 && err ? err->ptr<float>() : 0, prevPts.cols,
                level, block, patch);
        }
        else
        {
            lkSparse4_gpu(prevPyr_[level], nextPyr_[level],
                prevPts.ptr<float2>(), nextPts.ptr<float2>(), status.ptr(), level == 0 && err ? err->ptr<float>() : 0, prevPts.cols,
                level, block, patch);
        }
    }
}

void cv::gpu::PyrLKOpticalFlow::dense(const GpuMat& prevImg, const GpuMat& nextImg, GpuMat& u, GpuMat& v, GpuMat* err)
{
    using namespace cv::gpu::device::pyrlk;

    CV_Assert(prevImg.type() == CV_8UC1);
    CV_Assert(prevImg.size() == nextImg.size() && prevImg.type() == nextImg.type());
    CV_Assert(maxLevel >= 0);
    CV_Assert(winSize.width > 2 && winSize.height > 2);

    if (err)
        err->create(prevImg.size(), CV_32FC1);

    // build the image pyramids.

    prevPyr_.resize(maxLevel + 1);
    nextPyr_.resize(maxLevel + 1);

    prevPyr_[0] = prevImg;
    nextImg.convertTo(nextPyr_[0], CV_32F);

    for (int level = 1; level <= maxLevel; ++level)
    {
        pyrDown(prevPyr_[level - 1], prevPyr_[level]);
        pyrDown(nextPyr_[level - 1], nextPyr_[level]);
    }

    uPyr_.resize(2);
    vPyr_.resize(2);

    ensureSizeIsEnough(prevImg.size(), CV_32FC1, uPyr_[0]);
    ensureSizeIsEnough(prevImg.size(), CV_32FC1, vPyr_[0]);
    ensureSizeIsEnough(prevImg.size(), CV_32FC1, uPyr_[1]);
    ensureSizeIsEnough(prevImg.size(), CV_32FC1, vPyr_[1]);
    uPyr_[1].setTo(Scalar::all(0));
    vPyr_[1].setTo(Scalar::all(0));

    int2 winSize2i = make_int2(winSize.width, winSize.height);
    loadConstants(winSize2i, iters);

    DevMem2Df derr = err ? *err : DevMem2Df();

    int idx = 0;

    for (int level = maxLevel; level >= 0; level--)
    {
        int idx2 = (idx + 1) & 1;

        lkDense_gpu(prevPyr_[level], nextPyr_[level], uPyr_[idx], vPyr_[idx], uPyr_[idx2], vPyr_[idx2],
            level == 0 ? derr : DevMem2Df(), winSize2i);

        if (level > 0)
            idx = idx2;
    }

    uPyr_[idx].copyTo(u);
    vPyr_[idx].copyTo(v);
}

#endif /* !defined (HAVE_CUDA) */
