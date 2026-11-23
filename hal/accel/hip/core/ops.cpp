// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

// setTo and convertTo of the HIP backend. Each returns false, before queueing anything, when the
// device cannot run that call; OpenCV then runs it on the CPU.

#include "accel/hip/hip.hpp"

#ifdef HAVE_HIP

#include "kernels.hpp"

namespace cv { namespace accel { namespace hip_impl {

namespace {

// no Mat view alive (UMat::handle() needs that), and separate buffers or the same view (in place)
bool runnable(const UMat& src, const UMat& dst)
{
    return src.u->refcount == 0 && dst.u->refcount == 0 &&
           (src.u != dst.u || (src.offset == dst.offset && src.step[0] == dst.step[0]));
}

// false when the build has no code for this GPU: nothing was queued, the CPU runs the call
bool launched(hipError_t r, const char* what)
{
    if (r == hipErrorNoBinaryForGpu)
    {
        (void)hipGetLastError();
        return false;
    }
    hipCheck(r, what);
    return true;
}

class HipCoreOps CV_FINAL : public CoreOps
{
public:
    bool setTo(UMat& dst, const Scalar& value, Stream& stream) CV_OVERRIDE
    {
        if (dst.dims != 2 || dst.depth() > CV_64F || dst.channels() > 4 || dst.u->refcount != 0)
            return false;
        const HipStreamRef s = hipStreamOf(stream);
        HipDeviceGuard g(s.device);
        if (!launched(kernels::setTo((unsigned char*)dst.handle(ACCESS_WRITE) + dst.offset, dst.step[0],
                                     dst.rows, dst.cols, dst.depth(), dst.channels(), value.val, s.stream),
                      "setTo"))
            return false;
        stream.retain({ dst });
        return true;
    }

    bool convertTo(const UMat& src, UMat& dst, double alpha, double beta, Stream& stream) CV_OVERRIDE
    {
        if (src.dims != 2 || src.depth() > CV_64F || dst.depth() > CV_64F ||
            src.channels() != dst.channels() || !runnable(src, dst))
            return false;
        const HipStreamRef s = hipStreamOf(stream);
        HipDeviceGuard g(s.device);
        if (!launched(kernels::convertToScale((const unsigned char*)src.handle(ACCESS_READ) + src.offset,
                                              src.step[0], src.depth(),
                                              (unsigned char*)dst.handle(ACCESS_WRITE) + dst.offset,
                                              dst.step[0], dst.depth(),
                                              src.rows, src.cols, src.channels(), alpha, beta, s.stream),
                      "convertTo"))
            return false;
        stream.retain({ src, dst });
        return true;
    }
};

} // namespace

CoreOps* coreOps()
{
    static HipCoreOps ops;
    return &ops;
}

}}} // namespace cv::accel::hip_impl

#endif // HAVE_HIP
