// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_CORE_UTILS_ACCEL_DISPATCH_PRIVATE_HPP
#define OPENCV_CORE_UTILS_ACCEL_DISPATCH_PRIVATE_HPP

#include "opencv2/core/accel_backend.hpp"
#include <functional>

/*
The hook of a cv:: function, one line before its OpenCL branch:

    static bool accel_flip(InputArray src, OutputArray dst, int code)
    {
        accel::detail::OpCall call("flip");
        call.input(src).output(dst, MatShape({src.rows(), src.cols()}), src.type());
        return call.run<accel::CoreOps>([&](accel::CoreOps& ops, std::vector<UMat>& a, accel::Stream& s) {
            return ops.flip(a[0], a[1], code, s);
        });
    }
    ...
    CV_ACCEL_RUN(_src.isUMat() || _dst.isUMat(), accel_flip(_src, _dst, flip_mode))
    CV_OCL_RUN(_dst.isUMat(), ocl_flip(_src, _dst, flip_mode))

The call runs on the device that holds the first UMat input (or on the device of the
thread's ExecutionScope). Outputs are created there; outputs that cannot be moved (a Mat, a
region of a larger UMat, memory with an allocator of its own) get the result copied in. When
the device does not run the call, the outputs are still created on it and the function goes
on to its CPU code, which reaches the device data through UMat::getMat().
*/

namespace cv { namespace accel { namespace detail {

//! true while device memory exists or this thread is inside an ExecutionScope
CV_EXPORTS bool isActive();

class CV_EXPORTS OpCall
{
public:
    //! @p name names the call in errors and logs
    explicit OpCall(const char* name);
    ~OpCall();

    //! arguments in the order the operation takes them; the first UMat input picks the device
    OpCall& input(InputArray arr);
    OpCall& output(OutputArray arr, const MatShape& shape, int type);

    /** @brief Runs the call on a device.
    @p fn receives the module's operations, one UMat per argument in device memory and the stream,
    and returns what the operation returned.
    @return true when the call ran on the device, false when the CPU has to run it */
    template<class OpsT, class Fn>
    bool run(Fn fn)
    {
        return runImpl(OpsT::module(), [&fn](Ops* ops, std::vector<UMat>& args, Stream& stream) -> bool {
            OpsT* typed = dynamic_cast<OpsT*>(ops);
            return typed != 0 && fn(*typed, args, stream);
        });
    }

private:
    struct Arg
    {
        bool isOutput;
        const _InputArray* in;
        const _OutputArray* out;
        MatShape shape;
        int type;
    };

    bool runImpl(const char* module, const std::function<bool(Ops*, std::vector<UMat>&, Stream&)>& fn);

    const char* name_;
    std::vector<Arg> args_;

    OpCall(const OpCall&);
    OpCall& operator=(const OpCall&);
};

}}} // namespace cv::accel::detail

//! runs @p func when @p condition holds and accelerators are in use; returns from the caller when it did the work
#define CV_ACCEL_RUN_(condition, func, ...) \
    if ((condition) && ::cv::accel::detail::isActive() && (func)) \
        return __VA_ARGS__;

#define CV_ACCEL_RUN(condition, func) CV_ACCEL_RUN_(condition, func)

#endif // OPENCV_CORE_UTILS_ACCEL_DISPATCH_PRIVATE_HPP
