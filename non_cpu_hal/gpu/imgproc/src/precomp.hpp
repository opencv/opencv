// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_NON_CPU_HAL_GPU_IMGPROC_PRECOMP_HPP
#define OPENCV_NON_CPU_HAL_GPU_IMGPROC_PRECOMP_HPP

// Non-CPU HAL, GPU, imgproc: what the families share, then the families, one file each (included by
// ../../precomp.hpp). Images are of @p depth (CV_8U, ...) with @p cn channels; borders are
// cv::BorderTypes.

#include "opencv2/core/hal/interface.h"
#include <string.h>

#ifdef CV_ACCEL_GPU_KERNELS

namespace cv { namespace accel { namespace CV_GPU_NS { namespace kernels {

namespace {

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

//! returns fn<T, cn>(...) for 1 to 4 channels
#define CV_ACCEL_GPU_BY_CN(fn, T, ...) \
    switch (cn) \
    { \
    case 1: return fn<T, 1>(__VA_ARGS__); \
    case 2: return fn<T, 2>(__VA_ARGS__); \
    case 3: return fn<T, 3>(__VA_ARGS__); \
    case 4: return fn<T, 4>(__VA_ARGS__); \
    default: return CV_GPU(ErrorInvalidValue); \
    }

}}}} // namespace cv::accel::CV_GPU_NS::kernels

#else // the operations

#include "gpu/imgproc/include/gpu_hal_imgproc.hpp"

namespace cv { namespace accel { namespace CV_GPU_NS {

namespace {

//! a 2-D image of up to 4 channels and no Mat view alive (UMat::handle() needs that)
bool usableImage(const UMat& m)
{
    return m.dims == 2 && m.channels() <= 4 && m.u->refcount == 0;
}

//! true when the kernels ran; false when nothing was queued because the build has no code for this GPU
bool launched(CV_GPU(Error_t) r, const char* what)
{
    if (noKernelImage(r))
    {
        (void)CV_GPU(GetLastError)();
        return false;
    }
    check(r, what);
    return true;
}

} // namespace

}}} // namespace cv::accel::CV_GPU_NS

#endif // CV_ACCEL_GPU_KERNELS

#include "smooth.impl.hpp"
#include "warp.impl.hpp"

#ifdef CV_ACCEL_GPU_KERNELS

#undef CV_ACCEL_GPU_BY_CN

#else

namespace cv { namespace accel { namespace CV_GPU_NS {

namespace {

//! the imgproc operations: one mixin per family
class GpuImgprocOps CV_FINAL : public GpuWarpOps<GpuSmoothOps<ImgprocOps> > {};

Ptr<ImgprocOps> createImgprocOps()
{
    return makePtr<GpuImgprocOps>();
}

} // namespace

}}} // namespace cv::accel::CV_GPU_NS

#endif // CV_ACCEL_GPU_KERNELS

#endif // OPENCV_NON_CPU_HAL_GPU_IMGPROC_PRECOMP_HPP
