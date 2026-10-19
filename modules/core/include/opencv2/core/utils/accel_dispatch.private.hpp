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

//! true while an ExecutionScope that names a device is active on the calling thread (cheap)
CV_EXPORTS bool inExecutionScope();

/** A cv:: function starts and ends on the calling thread (the trace region that every cv::
function opens calls these while an accelerator is in use): placement by call looks at the
arguments of the innermost call only (see enterDeviceArgument). */
CV_EXPORTS void enterCall() CV_NOEXCEPT;
CV_EXPORTS void leaveCall() CV_NOEXCEPT;

//! true while the calling thread runs the CPU fallback of an accelerator call
CV_EXPORTS bool isOpenCLSuppressed();

/** @brief Registers a backend table that lives in the program rather than in a plugin library
(tests, with the emulator backend of modules/core/test; applications that link a backend in).
The table must outlive the process. A backend of the same name that is registered already is
kept. @return the backend id, 0 when rejected */
CV_EXPORTS int registerBackendTable(const OpenCV_Core_Accel_API* api, int priority = 0);

//! tests: an id of the physical memory behind @p m, the same for every UMat that reaches it (0: host memory)
CV_EXPORTS uint64 backingId(const UMat& m);
//! tests: how many backends and devices reach the physical memory behind @p m (imports of it)
CV_EXPORTS int backingImports(const UMat& m);
//! tests: the backend's allocation (or import) that @p m is a view of, NULL for other memory
CV_EXPORTS const cvAccelAllocation* allocationOf(const UMat& m);

//! sets the OPENCV_ACCEL_WAIT_TIMEOUT_MS limit at run time (0: wait forever); tests and diagnostics
CV_EXPORTS void setWaitTimeout(uint64 ms);
//! tests only: forgets that devices were lost (a lost device otherwise stays lost for the process)
CV_EXPORTS void reviveDevices();
//! turns OPENCV_ACCEL_VALIDATE on or off at run time (every Mat view opened for reading is mapped read-only)
CV_EXPORTS void setValidation(bool on);
//! tests: the page protection call after @p after successful ones fails, once (-1: none)
CV_EXPORTS void injectPageProtectionFailure(int after);

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
    if (cv::accel::detail::isActive() && ((condition) || cv::accel::detail::inExecutionScope())) \
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
