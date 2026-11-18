// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

// Core-module operations of the HIP backend. Each returns false, before queueing anything,
// when the device cannot run that call; OpenCV then runs it on the CPU.

#ifdef HAVE_ACCEL_HIP

#include "accel/hip/hip.hpp"

#include "kernels.hpp"

namespace cv { namespace accel { namespace hip_impl {

namespace {

class HipCoreOps CV_FINAL : public CoreOps
{
public:
    bool flip(const UMat& src, UMat& dst, int flipCode, Stream& stream) CV_OVERRIDE
    {
        const int esz = (int)src.elemSize();
        if (src.dims != 2 || !kernels::supportedElementSize(esz))
            return false;
        HipStream& s = hipStreamOf(stream);
        DeviceGuard g(s.device_);
        check(kernels::flip((const unsigned char*)devicePtr(src), src.step[0],
                            (unsigned char*)devicePtr(dst), dst.step[0],
                            src.rows, src.cols, esz, flipCode, s.s_), "flip");
        return true;
    }

    bool transpose(const UMat& src, UMat& dst, Stream& stream) CV_OVERRIDE
    {
        const int esz = (int)src.elemSize();
        if (src.dims != 2 || !kernels::supportedElementSize(esz))
            return false;
        HipStream& s = hipStreamOf(stream);
        DeviceGuard g(s.device_);
        check(kernels::transpose((const unsigned char*)devicePtr(src), src.step[0],
                                 (unsigned char*)devicePtr(dst), dst.step[0],
                                 src.rows, src.cols, esz, s.s_), "transpose");
        return true;
    }

    bool setTo(UMat& dst, const Scalar& value, Stream& stream) CV_OVERRIDE
    {
        if (dst.dims != 2 || dst.depth() > CV_64F || dst.channels() > 4)
            return false;
        HipStream& s = hipStreamOf(stream);
        DeviceGuard g(s.device_);
        check(kernels::setTo((unsigned char*)devicePtr(dst), dst.step[0],
                             dst.rows, dst.cols, dst.depth(), dst.channels(),
                             value.val, s.s_), "setTo");
        return true;
    }

    bool convertTo(const UMat& src, UMat& dst, double alpha, double beta, Stream& stream) CV_OVERRIDE
    {
        if (src.dims != 2 || src.depth() > CV_64F || dst.depth() > CV_64F)
            return false;
        if (src.channels() != dst.channels())
            return false;
        HipStream& s = hipStreamOf(stream);
        DeviceGuard g(s.device_);
        check(kernels::convertToScale((const unsigned char*)devicePtr(src), src.step[0], src.depth(),
                                      (unsigned char*)devicePtr(dst), dst.step[0], dst.depth(),
                                      src.rows, src.cols, src.channels(), alpha, beta, s.s_),
              "convertToScale");
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

#endif // HAVE_ACCEL_HIP
