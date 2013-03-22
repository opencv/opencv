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

cv::gpu::PyrLKOpticalFlow::PyrLKOpticalFlow() { throw_nogpu(); }
void cv::gpu::PyrLKOpticalFlow::sparse(const GpuMat&, const GpuMat&, const GpuMat&, GpuMat&, GpuMat&, GpuMat*) { throw_nogpu(); }
void cv::gpu::PyrLKOpticalFlow::dense(const GpuMat&, const GpuMat&, GpuMat&, GpuMat&, GpuMat*) { throw_nogpu(); }
void cv::gpu::PyrLKOpticalFlow::releaseMemory() {}

#else /* !defined (HAVE_CUDA) */

namespace pyrlk
{
    void loadConstants(int2 winSize, int iters);

    void sparse1(PtrStepSzf I, PtrStepSzf J, const float2* prevPts, float2* nextPts, uchar* status, float* err, int ptcount,
                 int level, dim3 block, dim3 patch, cudaStream_t stream = 0);
    void sparse4(PtrStepSz<float4> I, PtrStepSz<float4> J, const float2* prevPts, float2* nextPts, uchar* status, float* err, int ptcount,
                 int level, dim3 block, dim3 patch, cudaStream_t stream = 0);

    void dense(PtrStepSzb I, PtrStepSzf J, PtrStepSzf u, PtrStepSzf v, PtrStepSzf prevU, PtrStepSzf prevV,
               PtrStepSzf err, int2 winSize, cudaStream_t stream = 0);
}

cv::gpu::PyrLKOpticalFlow::PyrLKOpticalFlow()
{
    winSize = Size(21, 21);
    maxLevel = 3;
    iters = 30;
    useInitialFlow = false;
}

namespace
{
    void calcPatchSize(cv::Size winSize, dim3& block, dim3& patch)
    {
        if (winSize.width > 32 && winSize.width > 2 * winSize.height)
        {
            block.x = deviceSupports(FEATURE_SET_COMPUTE_12) ? 32 : 16;
            block.y = 8;
        }
        else
        {
            block.x = 16;
            block.y = deviceSupports(FEATURE_SET_COMPUTE_12) ? 16 : 8;
        }

        patch.x = (winSize.width  + block.x - 1) / block.x;
        patch.y = (winSize.height + block.y - 1) / block.y;

        block.z = patch.z = 1;
    }
}

void cv::gpu::PyrLKOpticalFlow::sparse(const GpuMat& prevImg, const GpuMat& nextImg, const GpuMat& prevPts, GpuMat& nextPts, GpuMat& status, GpuMat* err)
{
    if (prevPts.empty())
    {
        nextPts.release();
        status.release();
        if (err) err->release();
        return;
    }

    dim3 block, patch;
    calcPatchSize(winSize, block, patch);

    CV_Assert(prevImg.channels() == 1 || prevImg.channels() == 3 || prevImg.channels() == 4);
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
        buf_.resize(1);

        cvtColor(prevImg, buf_[0], COLOR_BGR2BGRA);
        buf_[0].convertTo(prevPyr_[0], CV_32F);

        cvtColor(nextImg, buf_[0], COLOR_BGR2BGRA);
        buf_[0].convertTo(nextPyr_[0], CV_32F);
    }

    for (int level = 1; level <= maxLevel; ++level)
    {
        pyrDown(prevPyr_[level - 1], prevPyr_[level]);
        pyrDown(nextPyr_[level - 1], nextPyr_[level]);
    }

    pyrlk::loadConstants(make_int2(winSize.width, winSize.height), iters);

    for (int level = maxLevel; level >= 0; level--)
    {
        if (cn == 1)
        {
            pyrlk::sparse1(prevPyr_[level], nextPyr_[level],
                prevPts.ptr<float2>(), nextPts.ptr<float2>(), status.ptr(), level == 0 && err ? err->ptr<float>() : 0, prevPts.cols,
                level, block, patch);
        }
        else
        {
            pyrlk::sparse4(prevPyr_[level], nextPyr_[level],
                prevPts.ptr<float2>(), nextPts.ptr<float2>(), status.ptr(), level == 0 && err ? err->ptr<float>() : 0, prevPts.cols,
                level, block, patch);
        }
    }
}

void cv::gpu::PyrLKOpticalFlow::dense(const GpuMat& prevImg, const GpuMat& nextImg, GpuMat& u, GpuMat& v, GpuMat* err)
{
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

    ensureSizeIsEnough(prevImg.size(), CV_32FC1, uPyr_[0]);
    ensureSizeIsEnough(prevImg.size(), CV_32FC1, vPyr_[0]);
    ensureSizeIsEnough(prevImg.size(), CV_32FC1, uPyr_[1]);
    ensureSizeIsEnough(prevImg.size(), CV_32FC1, vPyr_[1]);
    uPyr_[0].setTo(Scalar::all(0));
    vPyr_[0].setTo(Scalar::all(0));
    uPyr_[1].setTo(Scalar::all(0));
    vPyr_[1].setTo(Scalar::all(0));

    int2 winSize2i = make_int2(winSize.width, winSize.height);
    pyrlk::loadConstants(winSize2i, iters);

    PtrStepSzf derr = err ? *err : PtrStepSzf();

    int idx = 0;

    for (int level = maxLevel; level >= 0; level--)
    {
        int idx2 = (idx + 1) & 1;

        pyrlk::dense(prevPyr_[level], nextPyr_[level], uPyr_[idx], vPyr_[idx], uPyr_[idx2], vPyr_[idx2],
            level == 0 ? derr : PtrStepSzf(), winSize2i);

        if (level > 0)
            idx = idx2;
    }

    uPyr_[idx].copyTo(u);
    vPyr_[idx].copyTo(v);
}

void cv::gpu::PyrLKOpticalFlow::releaseMemory()
{
    prevPyr_.clear();
    nextPyr_.clear();

    buf_.clear();

    uPyr_[0].release();
    vPyr_[0].release();

    uPyr_[1].release();
    vPyr_[1].release();
}

#endif /* !defined (HAVE_CUDA) */
