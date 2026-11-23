// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_HAL_GPU_OPS_IMGPROC_IMPL_HPP
#define OPENCV_HAL_GPU_OPS_IMGPROC_IMPL_HPP

// Non-CPU HAL, CUDA and HIP: the imgproc operations (ImgprocOps), on the kernels of
// kernels_imgproc.impl.hpp.

#include "stream.impl.hpp"
#include "kernels.hpp"

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
    if (r == CV_GPU_ERROR_NO_KERNEL_IMAGE)
    {
        (void)CV_GPU(GetLastError)();
        return false;
    }
    check(r, what);
    return true;
}

class GpuImgprocOps CV_FINAL : public ImgprocOps
{
public:
    // two passes through a buffer, so the output may alias the input in any way; BORDER_WRAP is
    // left to the CPU, whose floating-point filters reject it
    bool gaussianBlur(const UMat& src, UMat& dst, const Mat& kx, const Mat& ky, int borderType,
                      Stream& stream) CV_OVERRIDE
    {
        const int depth = src.depth();
        const int tapType = depth == CV_8U ? CV_16U : depth == CV_16U ? CV_32U : CV_32F;
        if (!usableImage(src) || !usableImage(dst) || borderType == BORDER_WRAP || borderType > BORDER_REFLECT_101 ||
            (depth != CV_8U && depth != CV_16U && depth != CV_16S && depth != CV_32F) ||
            !validTaps(kx, tapType) || !validTaps(ky, tapType))
            return false;
        GpuStream& s = gpuStreamOf(stream);
        UMat buf;
        buf.createOn(getDevice(src), src.size(), CV_MAKETYPE(tapType, src.channels()));
        DeviceGuard g(s.device_);
        const unsigned char* sp = (const unsigned char*)src.handle(ACCESS_READ) + src.offset;
        unsigned char* bp = (unsigned char*)buf.handle(ACCESS_WRITE);
        unsigned char* dp = (unsigned char*)dst.handle(ACCESS_WRITE) + dst.offset;
        if (!launched(kernels::gaussianBlur(sp, src.step[0], bp, buf.step[0], dp, dst.step[0], src.rows, src.cols,
                                            depth, src.channels(), kx.ptr(), (int)kx.total(), ky.ptr(), (int)ky.total(),
                                            borderType, s.s_), "GaussianBlur"))
            return false;
        stream.retain({ src, buf, dst });
        return true;
    }

    // the types the CPU warps in floating point (warp_kernels.simd.hpp), which the GPU follows
    bool warpAffine(const UMat& src, UMat& dst, const Matx23d& M, int interpolation, int borderType,
                    const Scalar& borderValue, Stream& stream) CV_OVERRIDE
    {
        const int depth = src.depth(), cn = src.channels();
        if (!usableImage(src) || !usableImage(dst) || src.u == dst.u || cn == 2 ||
            (depth != CV_8U && depth != CV_16U && depth != CV_32F) ||
            (interpolation != kernels::INTER_NEAREST && interpolation != kernels::INTER_LINEAR) || borderType > BORDER_REFLECT_101)
            return false;
        float m[6], bval[4];
        for (int i = 0; i < 6; i++)
            m[i] = (float)M.val[i];
        for (int c = 0; c < 4; c++)
            bval[c] = depth == CV_8U ? saturate_cast<uchar>(borderValue[c])
                    : depth == CV_16U ? saturate_cast<ushort>(borderValue[c]) : (float)borderValue[c];
        GpuStream& s = gpuStreamOf(stream);
        DeviceGuard g(s.device_);
        const unsigned char* sp = (const unsigned char*)src.handle(ACCESS_READ) + src.offset;
        unsigned char* dp = (unsigned char*)dst.handle(ACCESS_WRITE) + dst.offset;
        if (!launched(kernels::warpAffine(sp, src.step[0], src.rows, src.cols, dp, dst.step[0], dst.rows, dst.cols,
                                          depth, cn, m, interpolation, borderType, bval, s.s_), "warpAffine"))
            return false;
        stream.retain({ src, dst });
        return true;
    }

private:
    static bool validTaps(const Mat& k, int type)
    {
        const int n = (int)k.total();
        return k.type() == type && k.isContinuous() && n % 2 == 1 && n <= kernels::GAUSSIAN_MAX_TAPS;
    }
};

} // namespace

}}} // namespace cv::accel::CV_GPU_NS

#endif // OPENCV_HAL_GPU_OPS_IMGPROC_IMPL_HPP
