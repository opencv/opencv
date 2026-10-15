// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_CORE_UTILS_ACCEL_DISPATCH_PRIVATE_HPP
#define OPENCV_CORE_UTILS_ACCEL_DISPATCH_PRIVATE_HPP

// Non-CPU HAL: the hook that cv:: functions place before CV_OCL_RUN.
//
//     static int accel_flip(InputArray src, OutputArray dst, int mode)
//     {
//         cv::accel::detail::OpCall call("core.flip", CV_ACCEL_OPS_CORE, CV_ACCEL_OPS_CORE_VERSION);
//         call.input(src);
//         call.output(dst, src.shape(), src.type());
//         return call.run([mode](const void* t, const cvAccelCall* c, const cvAccelView* v) {
//             const cvAccelOpsCore_v0* ops = (const cvAccelOpsCore_v0*)t;
//             return CV_ACCEL_OP_PRESENT(cvAccelOpsCore_v0, ops, flip) ? ops->flip(c, &v[0], &v[1], mode)
//                                                                       : CV_ACCEL_NOT_IMPLEMENTED;
//         });
//     }
//     ...
//     CV_ACCEL_RUN(_src.isUMat() || _dst.isUMat(), accel_flip(_src, _dst, flip_mode))
//     CV_OCL_RUN(_dst.isUMat(), ocl_flip(_src, _dst, flip_mode))
//
// The callback is first called with CV_ACCEL_QUERY (no handles, must not have side effects),
// then, if the backend accepted, with CV_ACCEL_EXECUTE. Everything between the two calls
// (output allocation, uploads, fence waits) and after them (recording fences, copying a
// temporary output back) is done by OpCall, so the coherence protocol lives in one place.

#include "opencv2/core.hpp"
#include "opencv2/core/llapi/accel_hal.h"

namespace cv { namespace accel { namespace detail {

enum RunResult
{
    NOT_HANDLED = 0,    //!< no argument is on an accelerator: continue with OpenCL / CPU as before
    HANDLED = 1,        //!< ran on an accelerator, outputs are complete (possibly still in flight)
    CPU_ONLY = 2        //!< arguments are on an accelerator but no backend accepted: run on the CPU
};

//! true once at least one accelerator backend is registered (cheap)
CV_EXPORTS bool isActive();

//! true while the calling thread runs the CPU fallback of an accelerator call
CV_EXPORTS bool isOpenCLSuppressed();

/** @brief Registers the built-in emulator backend (tests; or set OPENCV_ACCEL_EMULATOR=1).

The emulator runs on the CPU but behaves like six devices with different memory models:
"emulator:0" discrete (host access only through copies), "emulator:1" coherent shared
memory, "emulator:2" shared memory without cache coherence (it keeps a separate CPU view
that only begin/end_cpu_access synchronize, so a missing cache operation shows up as
wrong data), "emulator:3" protected memory (no CPU access), "emulator:4" managed memory
that the CPU may only touch while the device is idle, and "emulator:5", a discrete device
that fails on request (emulatorInjectFailure). Work runs asynchronously on worker
threads; OPENCV_ACCEL_EMULATOR_DELAY_US delays every task to expose missing waits.
@return the backend id */
CV_EXPORTS int registerEmulatorBackend();

//! number of emulator allocations alive right now, on one device or (-1) on all (tests of release)
CV_EXPORTS int64 emulatorLiveAllocations(int device = -1);

//! times device work and CPU access overlapped on "emulator:4" (a fault on real hardware)
CV_EXPORTS int64 emulatorViolations();

/** @brief Makes work on "emulator:5" fail from now on (@p status is a cvAccelStatus, 0 restores
normal operation).

By default the work is queued and fails later: fences recorded from now on report @p status.
With @p synchronous, operations and copies queue their work and then return @p status, like a
backend that fails halfway through a call. CV_ACCEL_DEVICE_LOST also makes every call on the
device fail at once; OpenCV keeps a lost device lost for the rest of the process, so only one
test may use it, at its end. */
CV_EXPORTS void emulatorInjectFailure(int status, bool synchronous = false);

/** @brief Disables the OpenCL path for the rest of the current cv:: call.

Data on an accelerator cannot be used by OpenCL kernels; letting the OpenCL path try would
only allocate outputs in the wrong place before it gives up. */
class CV_EXPORTS OpenCLSuppressor
{
public:
    OpenCLSuppressor() : active_(false) {}
    ~OpenCLSuppressor() { if (active_) release(); }
    void suppress();
private:
    void release();
    bool active_;
};

/** @brief One cv:: call on a non-CPU backend. */
class CV_EXPORTS OpCall
{
public:
    typedef cvAccelStatus (*Invoker)(const void* opTable, const cvAccelCall* call,
                                     const cvAccelView* views, void* userdata);

    OpCall(const char* opName, const char* module, uint32_t moduleVersion);
    ~OpCall();

    //! adds an input argument; views passed to the callback follow the order of input()/output() calls
    void input(InputArray arr);

    /** @brief Adds an output argument.
    @param shape, type what the operation produces
    @param fullOverwrite every element of the output is written (allows skipping the
           download of old content when the output already exists) */
    void output(OutputArray arr, const MatShape& shape, int type, bool fullOverwrite = true);

    RunResult run(Invoker fn, void* userdata);

    template<typename F>
    RunResult run(F f)
    {
        return run(&invoke<F>, &f);
    }

    struct Impl;
private:
    template<typename F>
    static cvAccelStatus invoke(const void* table, const cvAccelCall* call,
                                const cvAccelView* views, void* userdata)
    {
        return (*static_cast<F*>(userdata))(table, call, views);
    }

    OpCall(const OpCall&);
    OpCall& operator=(const OpCall&);
    Impl* p;
};

}}} // namespace cv::accel::detail

//! @cond IGNORED
#define CV_ACCEL_RUN_(condition, func, ...) \
    cv::accel::detail::OpenCLSuppressor __cv_accel_ocl_suppressor; \
    if ((condition) && cv::accel::detail::isActive()) \
    { \
        int __cv_accel_result = (func); \
        if (__cv_accel_result == cv::accel::detail::HANDLED) \
            return __VA_ARGS__; \
        if (__cv_accel_result == cv::accel::detail::CPU_ONLY) \
            __cv_accel_ocl_suppressor.suppress(); \
    }
//! @endcond

//! runs @p func when @p condition holds and an accelerator backend is present
#define CV_ACCEL_RUN(condition, func) CV_ACCEL_RUN_(condition, func, )

#endif // OPENCV_CORE_UTILS_ACCEL_DISPATCH_PRIVATE_HPP
