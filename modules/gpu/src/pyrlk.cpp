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
        void loadConstants(int cn, float minEigThreshold, int2 winSize, int iters);

        void calcSharrDeriv_gpu(DevMem2Db src, DevMem2D_<short> dx_buf, DevMem2D_<short> dy_buf, DevMem2D_<short> dIdx, DevMem2D_<short> dIdy, int cn,
            cudaStream_t stream = 0);

        void lkSparse_gpu(DevMem2Db I, DevMem2Db J, DevMem2D_<short> dIdx, DevMem2D_<short> dIdy,
            const float2* prevPts, float2* nextPts, uchar* status, float* err, bool GET_MIN_EIGENVALS, int ptcount,
            int level, dim3 block, dim3 patch, cudaStream_t stream = 0);

        void lkDense_gpu(DevMem2Db I, DevMem2Db J, DevMem2D_<short> dIdx, DevMem2D_<short> dIdy,
            DevMem2Df u, DevMem2Df v, DevMem2Df* err, bool GET_MIN_EIGENVALS, cudaStream_t stream = 0);
    }
}}}

void cv::gpu::PyrLKOpticalFlow::calcSharrDeriv(const GpuMat& src, GpuMat& dIdx, GpuMat& dIdy)
{
    using namespace cv::gpu::device::pyrlk;

    CV_Assert(src.rows > 1 && src.cols > 1);
    CV_Assert(src.depth() == CV_8U);

    const int cn = src.channels();

    ensureSizeIsEnough(src.size(), CV_MAKETYPE(CV_16S, cn), dx_calcBuf_);
    ensureSizeIsEnough(src.size(), CV_MAKETYPE(CV_16S, cn), dy_calcBuf_);

    calcSharrDeriv_gpu(src, dx_calcBuf_, dy_calcBuf_, dIdx, dIdy, cn);
}

void cv::gpu::PyrLKOpticalFlow::buildImagePyramid(const GpuMat& img0, vector<GpuMat>& pyr, bool withBorder)
{
    pyr.resize(maxLevel + 1);

    Size sz = img0.size();

    for (int level = 0; level <= maxLevel; ++level)
    {
        GpuMat temp;

        if (withBorder)
        {
            temp.create(sz.height + winSize.height * 2, sz.width + winSize.width * 2, img0.type());
            pyr[level] = temp(Rect(winSize.width, winSize.height, sz.width, sz.height));
        }
        else
        {
            ensureSizeIsEnough(sz, img0.type(), pyr[level]);
        }

        if (level == 0)
            img0.copyTo(pyr[level]);
        else
            pyrDown(pyr[level - 1], pyr[level]);

        if (withBorder)
            copyMakeBorder(pyr[level], temp, winSize.height, winSize.height, winSize.width, winSize.width, BORDER_REFLECT_101);

        sz = Size((sz.width + 1) / 2, (sz.height + 1) / 2);

        if (sz.width <= winSize.width || sz.height <= winSize.height)
        {
            maxLevel = level;
            break;
        }
    }
}

namespace
{
    void calcPatchSize(cv::Size winSize, int cn, dim3& block, dim3& patch, bool isDeviceArch11)
    {
        winSize.width *= cn;

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

    derivLambda = std::min(std::max(derivLambda, 0.0), 1.0);

    iters = std::min(std::max(iters, 0), 100);

    const int cn = prevImg.channels();

    dim3 block, patch;
    calcPatchSize(winSize, cn, block, patch, isDeviceArch11_);   

    CV_Assert(derivLambda >= 0);
    CV_Assert(maxLevel >= 0 && winSize.width > 2 && winSize.height > 2);
    CV_Assert(prevImg.size() == nextImg.size() && prevImg.type() == nextImg.type());
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
    // we pad each level with +/-winSize.{width|height}
    // pixels to simplify the further patch extraction.

    buildImagePyramid(prevImg, prevPyr_, true);
    buildImagePyramid(nextImg, nextPyr_, true);

    // dI/dx ~ Ix, dI/dy ~ Iy

    ensureSizeIsEnough(prevImg.rows + winSize.height * 2, prevImg.cols + winSize.width * 2, CV_MAKETYPE(CV_16S, cn), dx_buf_);
    ensureSizeIsEnough(prevImg.rows + winSize.height * 2, prevImg.cols + winSize.width * 2, CV_MAKETYPE(CV_16S, cn), dy_buf_);

    loadConstants(cn, minEigThreshold, make_int2(winSize.width, winSize.height), iters);

    for (int level = maxLevel; level >= 0; level--)
    {
        Size imgSize = prevPyr_[level].size();

        GpuMat dxWhole(imgSize.height + winSize.height * 2, imgSize.width + winSize.width * 2, dx_buf_.type(), dx_buf_.data, dx_buf_.step);
        GpuMat dyWhole(imgSize.height + winSize.height * 2, imgSize.width + winSize.width * 2, dy_buf_.type(), dy_buf_.data, dy_buf_.step);
        dxWhole.setTo(Scalar::all(0));
        dyWhole.setTo(Scalar::all(0));
        GpuMat dIdx = dxWhole(Rect(winSize.width, winSize.height, imgSize.width, imgSize.height));
        GpuMat dIdy = dyWhole(Rect(winSize.width, winSize.height, imgSize.width, imgSize.height));

        calcSharrDeriv(prevPyr_[level], dIdx, dIdy);

        lkSparse_gpu(prevPyr_[level], nextPyr_[level], dIdx, dIdy,
            prevPts.ptr<float2>(), nextPts.ptr<float2>(), status.ptr(), level == 0 && err ? err->ptr<float>() : 0, getMinEigenVals, prevPts.cols,
            level, block, patch);
    }
}

void cv::gpu::PyrLKOpticalFlow::dense(const GpuMat& prevImg, const GpuMat& nextImg, GpuMat& u, GpuMat& v, GpuMat* err)
{
    using namespace cv::gpu::device::pyrlk;

    derivLambda = std::min(std::max(derivLambda, 0.0), 1.0);

    iters = std::min(std::max(iters, 0), 100);

    CV_Assert(prevImg.type() == CV_8UC1);
    CV_Assert(prevImg.size() == nextImg.size() && prevImg.type() == nextImg.type());
    CV_Assert(derivLambda >= 0);
    CV_Assert(maxLevel >= 0 && winSize.width > 2 && winSize.height > 2);

    if (useInitialFlow)
    {
        CV_Assert(u.size() == prevImg.size() && u.type() == CV_32FC1);
        CV_Assert(v.size() == prevImg.size() && v.type() == CV_32FC1);
    }
    else
    {
        u.create(prevImg.size(), CV_32FC1);
        v.create(prevImg.size(), CV_32FC1);

        u.setTo(Scalar::all(0));
        v.setTo(Scalar::all(0));
    }

    if (err)
        err->create(prevImg.size(), CV_32FC1);

    // build the image pyramids.
    // we pad each level with +/-winSize.{width|height}
    // pixels to simplify the further patch extraction.

    buildImagePyramid(prevImg, prevPyr_, true);
    buildImagePyramid(nextImg, nextPyr_, true);
    buildImagePyramid(u, uPyr_, false);
    buildImagePyramid(v, vPyr_, false);

    // dI/dx ~ Ix, dI/dy ~ Iy

    ensureSizeIsEnough(prevImg.rows + winSize.height * 2, prevImg.cols + winSize.width * 2, CV_16SC1, dx_buf_);
    ensureSizeIsEnough(prevImg.rows + winSize.height * 2, prevImg.cols + winSize.width * 2, CV_16SC1, dy_buf_);

    loadConstants(1, minEigThreshold, make_int2(winSize.width, winSize.height), iters);

    DevMem2Df derr = err ? *err : DevMem2Df();

    for (int level = maxLevel; level >= 0; level--)
    {
        Size imgSize = prevPyr_[level].size();

        GpuMat dxWhole(imgSize.height + winSize.height * 2, imgSize.width + winSize.width * 2, dx_buf_.type(), dx_buf_.data, dx_buf_.step);
        GpuMat dyWhole(imgSize.height + winSize.height * 2, imgSize.width + winSize.width * 2, dy_buf_.type(), dy_buf_.data, dy_buf_.step);
        dxWhole.setTo(Scalar::all(0));
        dyWhole.setTo(Scalar::all(0));
        GpuMat dIdx = dxWhole(Rect(winSize.width, winSize.height, imgSize.width, imgSize.height));
        GpuMat dIdy = dyWhole(Rect(winSize.width, winSize.height, imgSize.width, imgSize.height));

        calcSharrDeriv(prevPyr_[level], dIdx, dIdy);

        lkDense_gpu(prevPyr_[level], nextPyr_[level], dIdx, dIdy, uPyr_[level], vPyr_[level],
            level == 0 && err ? &derr : 0, getMinEigenVals);

        if (level == 0)
        {
            uPyr_[0].copyTo(u);
            vPyr_[0].copyTo(v);
        }
        else
        {
            pyrUp(uPyr_[level], uPyr_[level - 1]);
            pyrUp(vPyr_[level], vPyr_[level - 1]);

            multiply(uPyr_[level - 1], Scalar::all(2), uPyr_[level - 1]);
            multiply(vPyr_[level - 1], Scalar::all(2), vPyr_[level - 1]);
        }
    }
}

#endif /* !defined (HAVE_CUDA) */
