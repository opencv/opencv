// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_HAL_GPU_OPS_CORE_IMPL_HPP
#define OPENCV_HAL_GPU_OPS_CORE_IMPL_HPP

// Non-CPU HAL, CUDA and HIP: the core operations (CoreOps), on the kernels of kernels_core.impl.hpp.

#include "stream.impl.hpp"
#include "kernels.hpp"

namespace cv { namespace accel { namespace CV_GPU_NS {

namespace {

/* Whether the kernels can run on @p src and @p dst: 2-D, an element size they move, no Mat view
   alive (UMat::handle() needs that), and either separate buffers or the same view (in place).
   Other views of one buffer may overlap, which the kernels do not handle. */
bool runnable(const UMat& src, const UMat& dst)
{
    return src.dims == 2 && kernels::supportedElementSize((int)src.elemSize()) &&
           src.u->refcount == 0 && dst.u->refcount == 0 &&
           (src.u != dst.u || (src.offset == dst.offset && src.step[0] == dst.step[0]));
}

class GpuCoreOps CV_FINAL : public CoreOps
{
public:
    bool flip(const UMat& src, UMat& dst, int flipCode, Stream& stream) CV_OVERRIDE
    {
        if (!runnable(src, dst))
            return false;
        GpuStream& s = gpuStreamOf(stream);
        DeviceGuard g(s.device_);
        const unsigned char* sp = (const unsigned char*)src.handle(ACCESS_READ) + src.offset;
        unsigned char* dp = (unsigned char*)dst.handle(ACCESS_WRITE) + dst.offset;
        const int esz = (int)src.elemSize();
        const CV_GPU(Error_t) r = src.u == dst.u
              ? kernels::flipInplace(dp, dst.step[0], src.rows, src.cols, esz, flipCode, s.s_)
              : kernels::flip(sp, src.step[0], dp, dst.step[0], src.rows, src.cols, esz, flipCode, s.s_);
        if (r == CV_GPU_ERROR_NO_KERNEL_IMAGE)
        {
            (void)CV_GPU(GetLastError)();
            return false;   // no code for this GPU in the build: nothing was queued, the CPU runs it
        }
        check(r, "flip");
        stream.retain({ src, dst });
        return true;
    }

    bool transpose(const UMat& src, UMat& dst, Stream& stream) CV_OVERRIDE
    {
        if (!runnable(src, dst) || (src.u == dst.u && src.rows != src.cols))
            return false;
        GpuStream& s = gpuStreamOf(stream);
        DeviceGuard g(s.device_);
        const unsigned char* sp = (const unsigned char*)src.handle(ACCESS_READ) + src.offset;
        unsigned char* dp = (unsigned char*)dst.handle(ACCESS_WRITE) + dst.offset;
        const int esz = (int)src.elemSize();
        const CV_GPU(Error_t) r = src.u == dst.u
              ? kernels::transposeInplace(dp, dst.step[0], src.rows, esz, s.s_)
              : kernels::transpose(sp, src.step[0], dp, dst.step[0], src.rows, src.cols, esz, s.s_);
        if (r == CV_GPU_ERROR_NO_KERNEL_IMAGE)
        {
            (void)CV_GPU(GetLastError)();
            return false;
        }
        check(r, "transpose");
        stream.retain({ src, dst });
        return true;
    }
};

} // namespace

}}} // namespace cv::accel::CV_GPU_NS

#endif // OPENCV_HAL_GPU_OPS_CORE_IMPL_HPP
