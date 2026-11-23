// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_HAL_GPU_DEVICE_HPP
#define OPENCV_HAL_GPU_DEVICE_HPP

// Non-CPU HAL, CUDA and HIP: what the kernel files share, the launch geometry and the device
// functions the kernels are built from.

#include "kernels.hpp"

namespace cv { namespace accel { namespace CV_GPU_NS { namespace kernels {

namespace {

const int BLOCK_X = 32, BLOCK_Y = 8;

// gridDim.y is at most 65535: taller images are covered by a loop over y
const unsigned MAX_GRID_Y = 65535;

inline dim3 grid(int rows, int cols, int bx, int by)
{
    const unsigned gy = (unsigned)((rows + by - 1) / by);
    return dim3((unsigned)((cols + bx - 1) / bx), gy < MAX_GRID_Y ? gy : MAX_GRID_Y);
}

// cv::BorderTypes; this file does not include core's C++ headers
enum { BORDER_CONSTANT = 0, BORDER_REPLICATE = 1, BORDER_REFLECT = 2, BORDER_WRAP = 3, BORDER_REFLECT_101 = 4 };

//! cv::borderInterpolate in O(1): the index position @p p of a line of @p len reads, -1 for BORDER_CONSTANT
__device__ inline int borderIndex(int p, int len, int border)
{
    if ((unsigned)p < (unsigned)len)
        return p;
    if (border == BORDER_REPLICATE)
        return p < 0 ? 0 : len - 1;
    if (border == BORDER_WRAP)
    {
        const long long r = (long long)p % len;
        return (int)(r < 0 ? r + len : r);
    }
    if (border == BORDER_REFLECT || border == BORDER_REFLECT_101)
    {
        if (len == 1)
            return 0;
        const int delta = border == BORDER_REFLECT_101;
        const long long period = 2 * ((long long)len - delta);
        long long r = (long long)p % period;
        if (r < 0)
            r += period;
        return (int)(r < len ? r : period - 1 + delta - r);
    }
    return -1;
}

//! saturate_cast<T>(v): rounded half to even as cvRound, clamped to the range of T
template<typename T> __device__ inline T saturate(float v);
template<> __device__ inline float saturate<float>(float v) { return v; }
template<> __device__ inline unsigned char saturate<unsigned char>(float v)
{ return (unsigned char)fminf(fmaxf(rintf(v), 0.f), 255.f); }
template<> __device__ inline unsigned short saturate<unsigned short>(float v)
{ return (unsigned short)fminf(fmaxf(rintf(v), 0.f), 65535.f); }
template<> __device__ inline short saturate<short>(float v)
{ return (short)fminf(fmaxf(rintf(v), -32768.f), 32767.f); }

} // namespace

}}}} // namespace cv::accel::CV_GPU_NS::kernels

#endif // OPENCV_HAL_GPU_DEVICE_HPP
