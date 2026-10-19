// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_CORE_TEST_ACCEL_EMULATOR_HPP
#define OPENCV_CORE_TEST_ACCEL_EMULATOR_HPP

#include "opencv2/core.hpp"
#include "opencv2/core/llapi/accel_hal.h"

// The emulator backend of the Non-CPU HAL tests (test_accel_emulator.cpp). It is part of the
// test binary, not of OpenCV: core sees it as any backend table (registerBackendTable).
namespace opencv_test { namespace accel_emu {

/** @brief Registers the emulator backend.

The emulator runs on the CPU but behaves like seven devices with different memory models:
"emulator:0" discrete (host access only through copies), "emulator:1" coherent shared
memory, "emulator:2" shared memory without cache coherence (it keeps a separate CPU view
that only begin/end_cpu_access synchronize, so a missing cache operation shows up as
wrong data; dma-bufs it imports are coherent; its streams are not thread-safe), "emulator:3"
protected memory (no CPU access), "emulator:4" managed memory that the CPU may only touch
while the device is idle, "emulator:5", a discrete device that fails on request
(emulatorInjectFailure) and, like a backend that predates protected memory, never attests
protection of memory it imports, and "emulator:6", a second discrete GPU with peer access to
emulator:0; neither of those two waits for fences of other devices on the device. Host
memory has another device address than its CPU address, as without unified addressing. Work
runs asynchronously on worker threads; emulatorSetTaskDelay (or OPENCV_ACCEL_EMULATOR_DELAY_US)
delays every task to expose missing waits.
@return the backend id */
int registerEmulatorBackend();

//! the same devices through a table that ends before begin_device_access ("emulator-v1")
int registerLegacyEmulatorBackend();

//! the emulator's table of entry points, to call them the way a careless core would (tests of the ABI rules)
const OpenCV_Core_Accel_API* emulatorBackendTable();

//! number of emulator allocations alive right now, on one device or (-1) on all (tests of release)
int64 emulatorLiveAllocations(int device = -1);

//! protocol violations the emulator detects (flags)
enum EmulatorViolation
{
    EMU_VIOLATION_MANAGED = 1,      //!< "emulator:4": device work outstanding while the CPU is in its memory (a fault)
    EMU_VIOLATION_CPU_ACCESS = 2,   //!< begin/end_cpu_access not balanced, ended with another access, or never ended
    EMU_VIOLATION_STREAM = 4,       //!< one stream used by two threads at once where streams are not thread-safe
    EMU_VIOLATION_FENCE = 8,        //!< a fence of another device waited for where that is not supported
    EMU_VIOLATION_ADDRESS = 16,     //!< a kernel given another address for memory than the one its device reaches it at
    EMU_VIOLATION_BRACKET = 32      //!< end_device_access without a begin_device_access it ends
};

//! violations of the kinds in @p kinds (EmulatorViolation flags) seen so far
int64 emulatorViolations(int kinds = EMU_VIOLATION_MANAGED);

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
cv::accel::detail::reviveDevices() and restore the emulator with status 0 afterwards. */
void emulatorInjectFailure(int status, int points = EMU_FAIL_FENCES);

//! number of emulator fences alive right now (no reference released twice, none leaked)
int64 emulatorLiveFences();

//! the kernels of @p device report NOT_IMPLEMENTED while @p disable holds (a device without the operation)
void emulatorDisableOps(int device, bool disable);

//! makes the next end_cpu_access after a CPU write fail (on any emulator device), once
void emulatorFailNextCpuFlush();
//! makes the next @p count end_cpu_access calls fail, whatever their access (the CPU access stays open)
void emulatorFailNextCpuEnds(int count);
/** @brief Makes the begin_cpu_access after the next @p after successful ones fail, once (-1:
none): with an error, or with @p nullAddress, with success but no address (a broken backend; the
access is open and must be ended). */
void emulatorFailNextCpuBegin(int after, bool nullAddress = false);
//! makes the next deallocate fail, once: the memory (an import of host pages, say) stays held
void emulatorFailNextDeallocate();
//! makes the begin_device_access after the next @p after successful ones fail, once (-1: none)
void emulatorFailNextDeviceAccessBegin(int after);
//! brackets begun on the dma-buf (begin_device_access) that have not ended yet
int emulatorDmaBufOpenBrackets(int64 fd);
//! makes the next end_device_access fail: the fence is not attached to the dma-buf
void emulatorFailNextDeviceAccessEnd();

/** @brief Device memory of "another library" (not held by OpenCV), filled with @p value; returns
its device pointer for ExternalMemory::DEVICE_POINTER. With @p protectedContent the emulator
allocates it as protected memory and attests that when it is wrapped. Never freed. */
int64 emulatorAllocateExternal(int device, size_t size, int value, bool protectedContent);

/** @brief A simulated dma-buf of @p size bytes (a memfd), shared with other "drivers"; returns its
fd for ExternalMemory::DMABUF_FD on "emulator:1" and "emulator:2". With @p protectedContent it is
a buffer of a secure heap: imported only as protected memory (CV_ACCEL_MEMF_PROTECTED asked
for), which the CPU never sees. Linux only: elsewhere, and without memfd, the test is skipped. */
int64 emulatorCreateDmaBuf(size_t size, bool protectedContent = false);
//! another driver (a camera, say) writes @p value into the whole dma-buf @p delayUs from now; users wait for it
void emulatorDmaBufProduce(int64 fd, int value, int delayUs);
/** @brief Exports the emulator allocation behind @p m as a dma-buf, the way another library holding
the allocation would, without telling OpenCV; returns the descriptor (the caller's). */
int64 emulatorExportDmaBuf(const cv::UMat& m);
//! how many fences OpenCV has attached to the dma-buf so far (end_device_access)
int emulatorDmaBufAttachedFences(int64 fd);

//! delays every task of the streams created from now on by @p delayUs (-1: OPENCV_ACCEL_EMULATOR_DELAY_US)
void emulatorSetTaskDelay(int delayUs);

//! disarms every injected failure and one-shot hook and enables every device's operations again
void emulatorReset();

}} // namespace opencv_test::accel_emu

#endif // OPENCV_CORE_TEST_ACCEL_EMULATOR_HPP
