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

//! true while the calling thread runs the CPU fallback of an accelerator call
CV_EXPORTS bool isOpenCLSuppressed();

/** @brief Registers the built-in emulator backend (tests; or set OPENCV_ACCEL_EMULATOR=1).

The emulator runs on the CPU but behaves like six devices with different memory models:
"emulator:0" discrete (host access only through copies), "emulator:1" coherent shared
memory, "emulator:2" shared memory without cache coherence (it keeps a separate CPU view
that only begin/end_cpu_access synchronize, so a missing cache operation shows up as
wrong data; dma-bufs it imports are coherent), "emulator:3" protected memory (no CPU
access), "emulator:4" managed memory that the CPU may only touch while the device is idle,
and "emulator:5", a discrete device that fails on request (emulatorInjectFailure) and, like
a backend that predates protected memory, never attests protection of memory it imports.
Work runs asynchronously on worker threads; OPENCV_ACCEL_EMULATOR_DELAY_US delays every task
to expose missing waits.
@return the backend id */
CV_EXPORTS int registerEmulatorBackend();

//! number of emulator allocations alive right now, on one device or (-1) on all (tests of release)
CV_EXPORTS int64 emulatorLiveAllocations(int device = -1);

//! times device work and CPU access overlapped on "emulator:4" (a fault on real hardware)
CV_EXPORTS int64 emulatorViolations();

//! where emulatorInjectFailure makes "emulator:5" fail (flags)
enum EmulatorFailurePoint
{
    EMU_FAIL_FENCES = 1,    //!< work queued from now on fails later: its fences report the status
    EMU_FAIL_CALLS = 2,     //!< operations and copies queue their work, then return the status
    EMU_FAIL_RECORD = 4,    //!< no fence can be recorded (stream_record_fence, and inside device_synchronize)
    EMU_FAIL_SYNC = 8,      //!< device_synchronize returns the status
    EMU_FAIL_RETAIN = 16    //!< fence_retain returns the status and takes no reference
};

/** @brief Makes "emulator:5" fail from now on (@p status is a cvAccelStatus, 0 restores normal
operation) at the @p points given (EmulatorFailurePoint flags).

CV_ACCEL_DEVICE_LOST makes every call on the device fail at once, whatever @p points says, and
stops the device: its queued kernels and copies never run (backend rule 8); the call returns
once none runs any more. OpenCV keeps a lost device lost; tests that lose it call
reviveDevices() and restore the emulator with status 0 afterwards. */
CV_EXPORTS void emulatorInjectFailure(int status, int points = EMU_FAIL_FENCES);

//! number of emulator fences alive right now (no reference released twice, none leaked)
CV_EXPORTS int64 emulatorLiveFences();

//! tests: an id of the physical memory behind @p m, the same for every UMat that reaches it (0: host memory)
CV_EXPORTS uint64 backingId(const UMat& m);

//! the kernels of @p device report NOT_IMPLEMENTED while @p disable holds (a device without the operation)
CV_EXPORTS void emulatorDisableOps(int device, bool disable);

//! makes the next end_cpu_access after a CPU write fail (on any emulator device), once
CV_EXPORTS void emulatorFailNextCpuFlush();

//! sets the OPENCV_ACCEL_WAIT_TIMEOUT_MS limit at run time (0: wait forever); tests and diagnostics
CV_EXPORTS void setWaitTimeout(uint64 ms);
//! tests only: forgets that devices were lost (a lost device otherwise stays lost for the process)
CV_EXPORTS void reviveDevices();
//! turns OPENCV_ACCEL_VALIDATE on or off at run time (read-only Mat views are checksummed)
CV_EXPORTS void setValidation(bool on);

/** @brief Device memory of "another library" (not held by OpenCV), filled with @p value; returns
its device pointer for ExternalMemory::DEVICE_POINTER. With @p protectedContent the emulator
allocates it as protected memory and attests that when it is wrapped. Never freed. */
CV_EXPORTS int64 emulatorAllocateExternal(int device, size_t size, int value, bool protectedContent);

/** @brief A simulated dma-buf of @p size bytes (a memfd), shared with other "drivers"; returns its
fd for ExternalMemory::DMABUF_FD on "emulator:1" and "emulator:2". Linux only. */
CV_EXPORTS int64 emulatorCreateDmaBuf(size_t size);
//! another driver (a camera, say) writes @p value into the whole dma-buf @p delayUs from now; users wait for it
CV_EXPORTS void emulatorDmaBufProduce(int64 fd, int value, int delayUs);
//! how many fences OpenCV has attached to the dma-buf so far (end_device_access)
CV_EXPORTS int emulatorDmaBufAttachedFences(int64 fd);

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
