// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_HAL_ACCEL_HIP_IMGPROC_KERNELS_HPP
#define OPENCV_HAL_ACCEL_HIP_IMGPROC_KERNELS_HPP

#include <hip/hip_runtime_api.h>
#include <stddef.h>

// 8-bit imgproc kernels, queued on @p s; @p rows x @p cols is the size of @p dst.
// Borders are cv::BorderTypes without BORDER_ISOLATED; BORDER_CONSTANT reads zeros.
namespace cv { namespace accel { namespace hip_impl { namespace kernels {

//! rows a launch can cover: gridDim.y is limited and every block here covers at least 4 rows
const int IMGPROC_MAX_ROWS = 65535 * 4;

//! threads per block of both box filter kernels (64 x 4)
const int BOX_BLOCK_THREADS = 256;

//! cv::boxFilter's division: (sum + divDelta) * divScale >> 23 when divScale is set, else round(sum * scale)
struct BoxScale
{
    float scale;
    unsigned divScale, divDelta;
};

//! the LDS tile box filter for @p cn channels (1, 3, 4) and @p border; NULL unless ksize is 3 or 5
const void* boxLdsKernelFor(int cn, int ksize, int border, bool fixed);

//! the sliding column box filter, any ksize; NULL for borders it does not implement
const void* boxSlidingKernelFor(int cn, int border, bool fixed);

//! runs @p kernel from boxLdsKernelFor; @p ax, @p ay is the anchor
hipError_t boxFilterLds(const void* kernel, const unsigned char* src, size_t sstep,
                        unsigned char* dst, size_t dstep, int rows, int cols,
                        int ax, int ay, BoxScale scale, hipStream_t s);

//! runs @p kernel from boxSlidingKernelFor
hipError_t boxFilterSliding(const void* kernel, const unsigned char* src, size_t sstep,
                            unsigned char* dst, size_t dstep, int rows, int cols,
                            int kw, int kh, int ax, int ay, BoxScale scale, hipStream_t s);

//! NV12 (@p uIdx 0) or NV21 (1) to @p dcn channels (3, 4), BGR order unless @p swapBlue
const void* nv12ToBgrKernelFor(int dcn, bool swapBlue, int uIdx);

// runs @p kernel from nv12ToBgrKernelFor on @p rows Y rows followed by rows / 2 UV rows;
// each thread of a block (at most 256 threads) converts 2x2 pixels
hipError_t nv12ToBgr(const void* kernel, const unsigned char* src, size_t sstep,
                     unsigned char* dst, size_t dstep, int rows, int cols,
                     int blockX, int blockY, hipStream_t s);

}}}} // namespace cv::accel::hip_impl::kernels

#endif // OPENCV_HAL_ACCEL_HIP_IMGPROC_KERNELS_HPP
