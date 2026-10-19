// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

// Non-CPU HAL: the coherence protocol, checked on the built-in emulator.
// emulator:0 discrete, 1 coherent shared, 2 non-coherent shared, 3 protected, 4 managed
// (CPU access only while the device is idle), 5 discrete that fails on request.
// See accel_dispatch.private.hpp.

#include "test_precomp.hpp"

#include <opencv2/core/accel.hpp>
#ifdef HAVE_CUDA
#include <opencv2/core/cuda.hpp>
#endif
#include <opencv2/core/utils/accel_dispatch.private.hpp>

#include <atomic>
#include <chrono>
#include <fstream>
#include <limits>
#include <thread>
#ifdef __linux__
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace opencv_test { namespace {

enum { DISCRETE = 0, COHERENT = 1, NONCOHERENT = 2, PROTECTED_MEM = 3, MANAGED = 4, FAILING = 5, PEER = 6 };

static accel::Device emulator(int index)
{
    accel::detail::registerEmulatorBackend();
    accel::Device d = accel::Device::find(cv::format("emulator:%d", index));
    CV_Assert(!d.empty());
    return d;
}

//! emulator:5 in its normal state (tests that use it leave it failed or lost)
static accel::Device failingDevice()
{
    accel::Device d = emulator(FAILING);
    accel::detail::emulatorInjectFailure(CV_ACCEL_OK);
    accel::detail::reviveDevices();
    return d;
}

static UMat makeOn(const accel::Device& d, const Mat& content)
{
    UMat u;
    u.allocator = d.allocator();
    content.copyTo(u);
    return u;
}

static double maxDiff(InputArray a, InputArray b)
{
    return cvtest::norm(a, b, NORM_INF);
}

//! runs @p body on a new thread whose emulator streams delay every task by @p delayUs
template<typename F>
static void runWithDeviceDelay(int delayUs, F body)
{
#ifndef _WIN32
    setenv("OPENCV_ACCEL_EMULATOR_DELAY_US", cv::format("%d", delayUs).c_str(), 1);
#endif
    std::thread t(body);
    t.join();
#ifndef _WIN32
    unsetenv("OPENCV_ACCEL_EMULATOR_DELAY_US");
#endif
}

TEST(Core_Accel, emulator_devices)
{
    accel::Device d = emulator(DISCRETE);
    EXPECT_EQ("emulator", d.backend());
    EXPECT_EQ(accel::DEVICE_GPU_DISCRETE, d.type());
    EXPECT_EQ(accel::DEVICE_NPU, emulator(NONCOHERENT).type());
    EXPECT_TRUE(accel::Device::find("emulator:99").empty());
    EXPECT_TRUE(accel::Device::find("no-such-backend").empty());
    EXPECT_EQ(accel::DEVICE_HOST, accel::Device().type());
}

typedef testing::TestWithParam<int> Core_Accel_AnyDevice;

TEST_P(Core_Accel_AnyDevice, flip_runs_on_the_device_and_stays_there)
{
    accel::Device d = emulator(GetParam());
    Mat src(31, 47, CV_8UC3);
    randu(src, 0, 255);
    UMat usrc = makeOn(d, src);

    accel::resetTransferStats();
    UMat udst;
    cv::flip(usrc, udst, 1);
    accel::TransferStats s = accel::getTransferStats();
    EXPECT_EQ(1u, s.deviceOps);
    EXPECT_EQ(0u, s.cpuFallbacks);
    EXPECT_TRUE(d == accel::describe(udst).device);

    Mat ref;
    cv::flip(src, ref, 1);
    EXPECT_EQ(0, maxDiff(ref, udst.getMat(ACCESS_READ)));
}

INSTANTIATE_TEST_CASE_P(/**/, Core_Accel_AnyDevice, testing::Values(DISCRETE, COHERENT, NONCOHERENT, MANAGED));

TEST(Core_Accel, shared_memory_needs_no_copies)
{
    for (int index : {COHERENT, NONCOHERENT})
    {
        accel::Device d = emulator(index);
        UMat u;
        u.allocator = d.allocator();
        u.create(8, 8, CV_8UC1);
        accel::resetTransferStats();
        {
            // on non-coherent memory this view ends with a cache clean, or the device misses it
            Mat m = u.getMat(ACCESS_WRITE);
            m.setTo(Scalar(3));
            m.at<uchar>(0, 0) = 200;
        }
        UMat out;
        cv::flip(u, out, -1);
        Mat r = out.getMat(ACCESS_READ);   // and this one starts with an invalidate
        EXPECT_EQ(200, r.at<uchar>(7, 7)) << "device " << index;
        EXPECT_EQ(3, r.at<uchar>(0, 0)) << "device " << index;
        EXPECT_EQ(0u, accel::getTransferStats().transfers) << "device " << index;
    }
}

TEST(Core_Accel, read_only_view_never_writes_back)
{
    accel::Device d = emulator(DISCRETE);
    Mat src(64, 64, CV_8UC1, Scalar(7));
    UMat u = makeOn(d, src);

    accel::resetTransferStats();
    {
        Mat m = u.getMat(ACCESS_READ);
        EXPECT_EQ(7, m.at<uchar>(3, 3));
    }
    {
        Mat m = u.getMat(ACCESS_READ);  // the host copy is still current
    }
    UMat out;
    cv::flip(u, out, 0);                // the device copy is still current too
    accel::TransferStats s = accel::getTransferStats();
    EXPECT_EQ((uint64)u.total(), s.bytesDeviceToHost);  // one download
    EXPECT_EQ(0u, s.bytesHostToDevice);                 // no write-back, no re-upload
}

TEST(Core_Accel, cpu_writes_reach_the_device_with_one_upload)
{
    accel::Device d = emulator(DISCRETE);
    UMat u;
    u.allocator = d.allocator();
    u.create(20, 30, CV_8UC1);

    accel::resetTransferStats();
    {
        Mat m = u.getMat(ACCESS_WRITE);     // fresh buffer: nothing to download
        m.setTo(Scalar(5));
    }
    {
        Mat m = u.getMat(ACCESS_RW);        // host copy is current: still nothing
        m.at<uchar>(0, 0) = 9;
    }
    EXPECT_EQ(0u, accel::getTransferStats().transfers);

    UMat out;
    cv::flip(u, out, 1);                    // the device needs the data: one upload
    EXPECT_EQ((uint64)u.total(), accel::getTransferStats().bytesHostToDevice);
    Mat r = out.getMat(ACCESS_READ);
    EXPECT_EQ(9, r.at<uchar>(0, 29));
    EXPECT_EQ(5, r.at<uchar>(5, 5));
}

TEST(Core_Accel, cpu_fallback_keeps_placement_and_skips_transfers)
{
    accel::Device d = emulator(DISCRETE);
    Mat src(16, 24, CV_32FC3);
    randu(src, -1, 1);
    UMat usrc = makeOn(d, src);

    accel::resetTransferStats();
    UMat udst;
    cv::transpose(usrc, udst);              // the emulator does not cover 12-byte elements
    accel::TransferStats s = accel::getTransferStats();
    EXPECT_EQ(1u, s.cpuFallbacks);
    EXPECT_TRUE(d == accel::describe(udst).device);                         // where the device would have put it
    EXPECT_EQ((uint64)(src.total() * src.elemSize()), s.bytesDeviceToHost); // src only, not the new dst
    EXPECT_EQ(0u, s.bytesHostToDevice);                                     // dst goes up only when needed

    Mat ref;
    cv::transpose(src, ref);
    EXPECT_EQ(0, maxDiff(ref, udst.getMat(ACCESS_READ)));
}

TEST(Core_Accel, partial_device_write_keeps_the_rest_of_the_buffer)
{
    accel::Device d = emulator(DISCRETE);
    Mat src(16, 16, CV_8UC1);
    randu(src, 0, 255);
    UMat u = makeOn(d, src);
    {
        Mat m = u.getMat(ACCESS_RW);        // now the host copy is the newer one
        m.row(0).setTo(Scalar(1));
    }
    Mat patch(4, 4, CV_8UC1);
    randu(patch, 0, 255);
    UMat upatch = makeOn(d, patch);
    UMat roi = u(Rect(2, 2, 4, 4));
    cv::flip(upatch, roi, 1);               // writes 16 of 256 bytes on the device

    Mat ref = src.clone();
    ref.row(0).setTo(Scalar(1));
    cv::flip(patch, ref(Rect(2, 2, 4, 4)), 1);
    EXPECT_EQ(0, maxDiff(ref, u.getMat(ACCESS_READ)));
}

TEST(Core_Accel, partly_written_output_keeps_the_rest)
{
    // an operation that writes the top half of its output only (fullOverwrite = false); the
    // output is a Mat, so the device writes a temporary that is then copied into it
    accel::Device d = emulator(DISCRETE);
    Mat src(8, 8, CV_8UC1), dst(8, 8, CV_8UC1, Scalar(5));
    randu(src, 0, 255);
    UMat usrc = makeOn(d, src);
    accel::detail::OpCall call("test.flip_top_half", CV_ACCEL_OPS_CORE, CV_ACCEL_OPS_CORE_VERSION);
    call.input(usrc);
    call.output(dst, src.shape(), src.type(), false);
    EXPECT_EQ(accel::detail::HANDLED, call.run([](const void* t, const cvAccelCall* c, const cvAccelView* v) {
        cvAccelView half[2] = { v[0], v[1] };
        half[0].size[0] /= 2;
        half[1].size[0] /= 2;
        return ((const cvAccelOpsCore_v0*)t)->flip(c, &half[0], &half[1], 1);
    }));
    Mat ref(8, 8, CV_8UC1, Scalar(5));
    cv::flip(src.rowRange(0, 4), ref.rowRange(0, 4), 1);
    EXPECT_EQ(0, maxDiff(ref, dst));
}

TEST(Core_Accel, in_place_operation)
{
    accel::Device d = emulator(DISCRETE);
    Mat src(9, 13, CV_16UC2);
    randu(src, 0, 60000);
    UMat u = makeOn(d, src);
    cv::flip(u, u, 0);
    Mat ref;
    cv::flip(src, ref, 0);
    EXPECT_EQ(0, maxDiff(ref, u.getMat(ACCESS_READ)));
    EXPECT_TRUE(d == accel::describe(u).device);
}

TEST(Core_Accel, writable_mat_view_keeps_the_device_away)
{
    accel::Device d = emulator(DISCRETE);
    Mat src(10, 10, CV_8UC1);
    randu(src, 0, 255);
    UMat u = makeOn(d, src);
    Mat view = u.getMat(ACCESS_RW);
    view.at<uchar>(0, 0) = 42;

    accel::resetTransferStats();
    UMat out;
    cv::flip(u, out, 1);                    // must not race with the view: runs on the CPU
    EXPECT_EQ(1u, accel::getTransferStats().cpuFallbacks);
    EXPECT_EQ(42, out.getMat(ACCESS_READ).at<uchar>(0, 9));
    view.release();

    cv::flip(u, out, 1);                    // view gone: back on the device
    EXPECT_EQ(1u, accel::getTransferStats().deviceOps);
}

TEST(Core_Accel, mat_views_of_device_memory_are_device_memory)
{
    // A Mat from UMat::getMat maps accelerator memory the CPU has open; it is not host memory.
    // A call that takes one runs on the CPU (counted), and its results still go to the device.
    Mat src(16, 24, CV_8UC1), ref;
    randu(src, 0, 255);
    cv::flip(src, ref, 0);
    for (int index : {DISCRETE, COHERENT, NONCOHERENT})
    {
        accel::Device d = emulator(index);
        UMat x = makeOn(d, src);
        Mat view = x.getMat(ACCESS_READ);
        accel::resetTransferStats();
        UMat out;
        cv::flip(view, out, 0);
        accel::TransferStats s = accel::getTransferStats();
        EXPECT_EQ(0u, s.deviceOps) << "device " << index;
        EXPECT_EQ(1u, s.cpuFallbacks) << "device " << index;
        EXPECT_TRUE(d == accel::describe(out).device) << "device " << index;
        EXPECT_EQ(0, maxDiff(ref, out.getMat(ACCESS_READ))) << "device " << index;
    }

    // The same bytes under two names: a Mat view of rows 0..7 written from a UMat of rows 4..11
    // of the same memory. The result is the CPU's on host memory aliased the same way; the
    // device never writes the open view, and never misses the overlap.
    Mat host = src.clone();
    cv::flip(host.rowRange(4, 12), host.rowRange(0, 8), 0);
    for (int index : {DISCRETE, COHERENT, NONCOHERENT})
    {
        for (AccessFlag access : {ACCESS_READ, ACCESS_RW})
        {
            accel::Device d = emulator(index);
            UMat x = makeOn(d, src);
            Mat view = x.rowRange(0, 8).getMat(access);
            accel::resetTransferStats();
            cv::flip(x.rowRange(4, 12), view, 0);
            EXPECT_EQ(0u, accel::getTransferStats().deviceOps) << "device " << index;
            EXPECT_EQ(0, maxDiff(host, x.getMat(ACCESS_READ))) << "device " << index << " access " << (int)access;
        }
    }
}

TEST(Core_Accel, copy_out_while_a_mat_view_is_open)
{
    // non-coherent memory: the view's writes sit in CPU caches until the view ends
    accel::Device d = emulator(NONCOHERENT);
    UMat u = makeOn(d, Mat(8, 8, CV_8UC1, Scalar(1)));
    Mat view = u.getMat(ACCESS_RW);
    view.setTo(Scalar(42));
    Mat res;
    u.copyTo(res);
    EXPECT_EQ(0, maxDiff(Mat(8, 8, CV_8UC1, Scalar(42)), res));
    view.release();
    EXPECT_EQ(0, maxDiff(Mat(8, 8, CV_8UC1, Scalar(42)), u.getMat(ACCESS_READ)));   // the writes survived
}

TEST(Core_Accel, cpu_access_whose_end_failed_keeps_the_device_away)
{
    // A failed end_cpu_access, even after a read, may leave the memory in the CPU's hands: on
    // memory the CPU may touch only while the device is idle, device work would fault. The
    // device stays away until a retry of the end succeeds.
    const int kinds = accel::detail::EMU_VIOLATION_MANAGED | accel::detail::EMU_VIOLATION_CPU_ACCESS;
    const int64 violations = accel::detail::emulatorViolations(kinds);
    {
        // the device gate stays closed: no device work at all until the end went through
        accel::Device d = emulator(MANAGED);
        UMat u = makeOn(d, Mat(8, 8, CV_8UC1, Scalar(2)));
        accel::detail::emulatorFailNextCpuEnds(1);
        EXPECT_EQ(2, u.getMat(ACCESS_READ).at<uchar>(0, 0));
        accel::resetTransferStats();
        UMat out;
        cv::flip(u, out, 1);                        // the CPU still holds the memory: on the CPU, which ends it
        EXPECT_EQ(1u, accel::getTransferStats().cpuFallbacks);
        EXPECT_EQ(0u, accel::getTransferStats().deviceOps);
        cv::flip(u, out, 1);
        EXPECT_EQ(1u, accel::getTransferStats().deviceOps);
        EXPECT_EQ(2, out.getMat(ACCESS_READ).at<uchar>(7, 7));
    }
    {
        // other memory: the device stays away from this buffer while the retries fail
        accel::Device d = emulator(NONCOHERENT);
        UMat u = makeOn(d, Mat(8, 8, CV_8UC1, Scalar(3)));
        accel::detail::emulatorFailNextCpuEnds(2);  // the end, and its retry before the next device use
        EXPECT_EQ(3, u.getMat(ACCESS_READ).at<uchar>(0, 0));
        accel::resetTransferStats();
        UMat out;
        cv::flip(u, out, 1);
        EXPECT_EQ(1u, accel::getTransferStats().cpuFallbacks);
        cv::flip(u, out, 1);
        EXPECT_EQ(1u, accel::getTransferStats().deviceOps);
        EXPECT_EQ(3, out.getMat(ACCESS_READ).at<uchar>(7, 7));
    }
    EXPECT_EQ(violations, accel::detail::emulatorViolations(kinds));
}

TEST(Core_Accel, cpu_access_that_cannot_open_ends_every_begin)
{
    // A camera frame imported by two devices: CPU access begins on both imports. When the
    // second begin fails, the first is ended again; when that end fails too, it stays owed
    // and the devices stay away until it succeeds. A begin that succeeds without an address
    // is ended as well. Every begin ends once, innermost first.
    const int64 fd = accel::detail::emulatorCreateDmaBuf(256);
    accel::ExternalMemory mem;
    mem.type = accel::ExternalMemory::DMABUF_FD;
    mem.handle = fd;
    mem.size = 256;
    const int64 violations = accel::detail::emulatorViolations(accel::detail::EMU_VIOLATION_CPU_ACCESS);
    {
        UMat gpu = accel::wrapExternalMemory(emulator(COHERENT), mem, MatShape({16, 16}), CV_8UC1);
        UMat npu = accel::wrapExternalMemory(emulator(NONCOHERENT), mem, MatShape({16, 16}), CV_8UC1);
        gpu.setTo(Scalar(7));

        accel::detail::emulatorFailNextCpuBegin(1);
        accel::detail::emulatorFailNextCpuEnds(3);      // the end that undoes the first begin, and its next two retries
        EXPECT_THROW(gpu.getMat(ACCESS_READ), cv::Exception);
        accel::resetTransferStats();
        UMat out;
        cv::flip(gpu, out, 0);                          // the end is retried and fails again: on the CPU
        EXPECT_EQ(0u, accel::getTransferStats().deviceOps);
        EXPECT_EQ(1u, accel::getTransferStats().cpuFallbacks);
        cv::flip(gpu, out, 0);                          // ended by now: on the device
        EXPECT_EQ(1u, accel::getTransferStats().deviceOps);
        EXPECT_EQ(7, out.getMat(ACCESS_READ).at<uchar>(15, 15));

        accel::detail::emulatorFailNextCpuBegin(0, true);
        EXPECT_THROW(npu.getMat(ACCESS_READ), cv::Exception);
        EXPECT_EQ(7, npu.getMat(ACCESS_READ).at<uchar>(0, 0));
    }
    EXPECT_EQ(violations, accel::detail::emulatorViolations(accel::detail::EMU_VIOLATION_CPU_ACCESS));
}

TEST(Core_Accel, memory_whose_cpu_access_cannot_end_is_never_freed)
{
    // end_cpu_access fails for good: when the last UMat goes, the memory is still open for the
    // CPU in the backend, so it is leaked rather than freed under that access
    accel::Device d = emulator(COHERENT);
    const int64 live = accel::detail::emulatorLiveAllocations(COHERENT);
    const int64 violations = accel::detail::emulatorViolations(accel::detail::EMU_VIOLATION_CPU_ACCESS);
    {
        UMat u = makeOn(d, Mat(8, 8, CV_8UC1, Scalar(1)));
        Mat m = u.getMat(ACCESS_READ);
        accel::detail::emulatorFailNextCpuEnds(1000);
        m.release();
        u.release();
        accel::detail::emulatorFailNextCpuEnds(0);
    }
    EXPECT_EQ(live + 1, accel::detail::emulatorLiveAllocations(COHERENT));
    EXPECT_EQ(violations, accel::detail::emulatorViolations(accel::detail::EMU_VIOLATION_CPU_ACCESS));
}

TEST(Core_Accel, host_ranges_past_the_end_of_the_address_space_are_refused)
{
    // the pages of such a range cannot be computed: reserving them would wrap around to address 0
    accel::ExternalMemory mem;
    mem.type = accel::ExternalMemory::HOST_POINTER;
    mem.handle = (int64)(intptr_t)(std::numeric_limits<uintptr_t>::max() & ~(uintptr_t)4095);
    mem.size = 8192;
    EXPECT_THROW(accel::wrapExternalMemory(emulator(COHERENT), mem, MatShape({1, 8192}), CV_8UC1), cv::Exception);
    mem.size = 64;
    EXPECT_THROW(accel::wrapExternalMemory(emulator(COHERENT), mem, MatShape({1, 64}), CV_8UC1), cv::Exception);
}

TEST(Core_Accel, upgraded_cpu_access_ends_every_level)
{
    // a read mapping opened for writing too nests a second begin_cpu_access: each one ends,
    // innermost first, with the access it began with
    for (int dev : {NONCOHERENT, MANAGED})
    {
        accel::Device d = emulator(dev);
        const int64 violations = accel::detail::emulatorViolations(accel::detail::EMU_VIOLATION_CPU_ACCESS);
        UMat u = makeOn(d, Mat(8, 8, CV_8UC1, Scalar(1)));
        {
            Mat r = u.getMat(ACCESS_READ);
            Mat w = u.getMat(ACCESS_WRITE);
            w.setTo(Scalar(6));
        }
        UMat out;
        cv::flip(u, out, 0);    // the device sees the write: the outer level ended with a clean
        EXPECT_EQ(6, out.getMat(ACCESS_READ).at<uchar>(0, 0)) << "device " << dev;
        u.release();
        EXPECT_EQ(violations, accel::detail::emulatorViolations(accel::detail::EMU_VIOLATION_CPU_ACCESS)) << "device " << dev;
    }
}

TEST(Core_Accel, failed_cache_flush_never_publishes_cpu_writes)
{
    // non-coherent shared memory: CPU writes reach the device only through end_cpu_access
    accel::Device d = emulator(NONCOHERENT);
    UMat u = makeOn(d, Mat(8, 8, CV_8UC1, Scalar(1)));
    {
        Mat view = u.getMat(ACCESS_RW);
        view.setTo(Scalar(7));
        accel::detail::emulatorFailNextCpuFlush();
    }   // the flush fails: the device may hold old bytes, new bytes, or both
    UMat out;
    EXPECT_THROW(cv::flip(u, out, 1), cv::Exception);     // readers get an error, not stale data
    EXPECT_THROW(u.getMat(ACCESS_READ), cv::Exception);
    Mat(8, 8, CV_8UC1, Scalar(3)).copyTo(u);                // written whole: usable again
    EXPECT_EQ(3, u.getMat(ACCESS_READ).at<uchar>(0, 0));
}

#ifdef __linux__
//! permissions ("rw-p", ...) of the mapping that holds @p p, from /proc/self/maps
static std::string mappingPermissions(const void* p)
{
    std::ifstream maps("/proc/self/maps");
    std::string line;
    const uintptr_t a = (uintptr_t)p;
    while (std::getline(maps, line))
    {
        unsigned long long lo = 0, hi = 0;
        char perms[8] = {0};
        if (sscanf(line.c_str(), "%llx-%llx %7s", &lo, &hi, perms) == 3 && lo <= a && a < hi)
            return perms;
    }
    return "";
}

TEST(Core_Accel, read_only_memory_has_a_read_only_cpu_view)
{
    // read-only device memory (a wrapped pointer, say): a write through its Mat view faults
    accel::Device d = emulator(DISCRETE);
    accel::ExternalMemory mem;
    mem.type = accel::ExternalMemory::DEVICE_POINTER;
    mem.handle = accel::detail::emulatorAllocateExternal(DISCRETE, 64 * 64, 3, false);
    mem.size = 64 * 64;
    mem.readOnly = true;
    UMat ro = accel::wrapExternalMemory(d, mem, MatShape({64, 64}), CV_8UC1);
    Mat view = ro.getMat(ACCESS_READ);
    EXPECT_EQ(3, view.at<uchar>(63, 63));
    const std::string perms = mappingPermissions(view.data);
    ASSERT_EQ(4u, perms.size());
    EXPECT_EQ('r', perms[0]);
    EXPECT_EQ('-', perms[1]);
}


TEST(Core_Accel, raw_handles_keep_the_rules_of_the_memory)
{
    // UMat::handle() hands the memory to code OpenCV does not see: no write handle of read-only
    // memory, and none of a dma-buf (only DeviceAccess orders work with the other drivers)
    accel::ExternalMemory mem;
    mem.type = accel::ExternalMemory::DEVICE_POINTER;
    mem.handle = accel::detail::emulatorAllocateExternal(DISCRETE, 256, 3, false);
    mem.size = 256;
    mem.readOnly = true;
    UMat ro = accel::wrapExternalMemory(emulator(DISCRETE), mem, MatShape({16, 16}), CV_8UC1);
    EXPECT_THROW(ro.handle(ACCESS_RW), cv::Exception);
    EXPECT_TRUE(ro.handle(ACCESS_READ) != NULL);

    accel::ExternalMemory buf;
    buf.type = accel::ExternalMemory::DMABUF_FD;
    buf.handle = accel::detail::emulatorCreateDmaBuf(256);
    buf.size = 256;
    UMat frame = accel::wrapExternalMemory(emulator(COHERENT), buf, MatShape({16, 16}), CV_8UC1);
    EXPECT_THROW(frame.handle(ACCESS_READ), cv::Exception);
    accel::DeviceAccess access(frame, accel::Stream::getDefault(emulator(COHERENT)), ACCESS_READ);
    EXPECT_NE(0u, access.handle());
}

TEST(Core_Accel, read_only_views_of_shared_memory_fault_on_writes)
{
    // host memory the coherent device uses in place, wrapped read-only: the CPU addresses the
    // bytes directly, but the Mat view must not let it write them
    accel::Device d = emulator(COHERENT);
    std::vector<uchar> storage(4096 * 2, 5);
    uchar* p = alignPtr(storage.data(), 4096);
    accel::ExternalMemory mem;
    mem.type = accel::ExternalMemory::HOST_POINTER;
    mem.handle = (int64)(intptr_t)p;
    mem.size = 4096;
    mem.readOnly = true;
    UMat ro = accel::wrapExternalMemory(d, mem, MatShape({64, 64}), CV_8UC1);
    EXPECT_THROW(ro.getMat(ACCESS_WRITE), cv::Exception);
    Mat view = ro.getMat(ACCESS_READ);
    EXPECT_EQ(5, view.at<uchar>(63, 63));
    const std::string perms = mappingPermissions(view.data);
    ASSERT_EQ(4u, perms.size());
    EXPECT_EQ('-', perms[1]);
}

TEST(Core_Accel, read_only_dma_buf_is_mapped_read_only_without_a_copy)
{
    accel::Device d = emulator(COHERENT);
    const int64 fd = accel::detail::emulatorCreateDmaBuf(256);
    accel::detail::emulatorDmaBufProduce(fd, 6, 0);
    accel::ExternalMemory mem;
    mem.type = accel::ExternalMemory::DMABUF_FD;
    mem.handle = fd;
    mem.size = 256;
    mem.readOnly = true;
    UMat ro = accel::wrapExternalMemory(d, mem, MatShape({16, 16}), CV_8UC1);
    Mat view = ro.getMat(ACCESS_READ);
    EXPECT_EQ(6, view.at<uchar>(0, 0));
    const std::string perms = mappingPermissions(view.data);
    ASSERT_EQ(4u, perms.size());
    EXPECT_EQ("r--s", perms);   // the buffer's own read-only shared mapping: no copy
}

TEST(Core_Accel, read_only_views_fail_closed)
{
    // read-only device memory: its host copy lives in pages that must be protected, and
    // protected again after every download; when that fails, the CPU gets no view at all
    accel::Device d = emulator(DISCRETE);
    accel::ExternalMemory mem;
    mem.type = accel::ExternalMemory::DEVICE_POINTER;
    mem.handle = accel::detail::emulatorAllocateExternal(DISCRETE, 64 * 64, 3, false);
    mem.size = 64 * 64;
    mem.readOnly = true;
    {
        UMat ro = accel::wrapExternalMemory(d, mem, MatShape({64, 64}), CV_8UC1);
        accel::detail::injectPageProtectionFailure(0);     // the host copy cannot be protected
        EXPECT_THROW(ro.getMat(ACCESS_READ), cv::Exception);
    }
    {
        UMat ro = accel::wrapExternalMemory(d, mem, MatShape({64, 64}), CV_8UC1);
        accel::detail::injectPageProtectionFailure(2);     // protected, unprotected for the download, then not again
        EXPECT_THROW(ro.getMat(ACCESS_READ), cv::Exception);
        Mat view = ro.getMat(ACCESS_READ);                 // a new host copy, protected
        EXPECT_EQ(3, view.at<uchar>(10, 10));
        EXPECT_EQ('-', mappingPermissions(view.data)[1]);
    }
}

TEST(Core_Accel, validation_mode_maps_read_views_read_only)
{
    accel::detail::setValidation(true);
    for (int dev : {COHERENT, DISCRETE})
    {
        UMat u = makeOn(emulator(dev), Mat(8, 8, CV_8UC1, Scalar(1)));
        {
            Mat r = u.getMat(ACCESS_READ);              // a write through it faults right away
            EXPECT_EQ('-', mappingPermissions(r.data)[1]) << "device " << dev;
            Mat w = u.getMat(ACCESS_WRITE);             // the same mapping, opened for writing too
            w.at<uchar>(0, 0) = 9;
            EXPECT_EQ(9, r.at<uchar>(0, 0)) << "device " << dev;
        }
        EXPECT_EQ(9, u.getMat(ACCESS_READ).at<uchar>(0, 0)) << "device " << dev;   // the write reached the UMat
    }
    accel::detail::setValidation(false);
}
#endif


TEST(Core_Accel, protected_memory_never_reaches_the_cpu)
{
    accel::Device d = emulator(PROTECTED_MEM);
    Mat src(10, 12, CV_8UC1);
    randu(src, 0, 255);
    UMat u = makeOn(d, src);                // data may go in
    UMat out;
    cv::flip(u, out, 0);                    // and be processed on the device
    EXPECT_TRUE(d == accel::describe(out).device);
    EXPECT_EQ(accel::HOST_ACCESS_NONE, accel::describe(out).hostAccess);

    EXPECT_THROW(out.getMat(ACCESS_READ), cv::Exception);
    Mat back;
    EXPECT_THROW(out.copyTo(back), cv::Exception);

    // no silent CPU fallback either: the device does not cover 12-byte elements
    Mat f(4, 4, CV_32FC3, Scalar::all(1));
    UMat uf = makeOn(d, f), t;
    EXPECT_THROW(cv::transpose(uf, t), cv::Exception);
}

TEST(Core_Accel, protected_content_stays_in_protected_memory)
{
    // a protected frame (from a secure decoder, say) next to ordinary memory on one device
    accel::Device d = emulator(DISCRETE);
    accel::ExternalMemory mem;
    mem.type = accel::ExternalMemory::DEVICE_POINTER;
    mem.handle = accel::detail::emulatorAllocateExternal(DISCRETE, 64, 7, true);
    mem.size = 64;
    mem.protectedContent = true;
    UMat secure = accel::wrapExternalMemory(d, mem, MatShape({8, 8}), CV_8UC1);
    EXPECT_TRUE(accel::describe(secure).protectedContent);
    EXPECT_THROW(secure.getMat(ACCESS_READ), cv::Exception);

    UMat out;
    cv::flip(secure, out, 1);                   // runs on the device, into protected memory
    EXPECT_TRUE(accel::describe(out).protectedContent);
    EXPECT_THROW(out.getMat(ACCESS_READ), cv::Exception);

    UMat plain = makeOn(d, Mat(8, 8, CV_8UC1, Scalar(0)));
    Mat host;
    EXPECT_THROW(cv::flip(secure, plain, 1), cv::Exception);   // an ordinary output would expose it
    EXPECT_THROW(cv::flip(secure, host, 1), cv::Exception);    // and so would a Mat
    EXPECT_THROW(secure.copyTo(plain), cv::Exception);
    EXPECT_EQ(0, maxDiff(Mat(8, 8, CV_8UC1, Scalar(0)), plain));

    // protection comes from the backend: ordinary memory cannot be declared protected...
    accel::ExternalMemory claimed = mem;
    claimed.handle = accel::detail::emulatorAllocateExternal(DISCRETE, 64, 7, false);
    EXPECT_THROW(accel::wrapExternalMemory(d, claimed, MatShape({8, 8}), CV_8UC1), cv::Exception);
    // ...and protected memory stays protected without the caller saying so
    accel::ExternalMemory unclaimed = mem;
    unclaimed.handle = accel::detail::emulatorAllocateExternal(DISCRETE, 64, 7, true);
    unclaimed.protectedContent = false;
    EXPECT_TRUE(accel::describe(accel::wrapExternalMemory(d, unclaimed, MatShape({8, 8}), CV_8UC1)).protectedContent);
}

TEST(Core_Accel, transfer_policy_reports_hidden_copies)
{
    Mat src(8, 8, CV_32FC3, Scalar::all(1));
    UMat discrete = makeOn(emulator(DISCRETE), src);
    UMat shared = makeOn(emulator(COHERENT), src);
    UMat t;
    accel::setTransferPolicy(accel::TRANSFER_FORBID);
    EXPECT_THROW(cv::transpose(discrete, t), cv::Exception);   // a CPU fallback would copy device data
    EXPECT_NO_THROW(cv::transpose(shared, t));                 // shared memory: the CPU uses it in place
    accel::setTransferPolicy(accel::TRANSFER_ALLOW);
}

TEST(Core_Accel, host_mat_used_in_place_by_a_coherent_device)
{
    Mat src(32, 40, CV_8UC4);
    randu(src, 0, 255);
    Mat ref;
    cv::flip(src, ref, 0);
    {
        accel::ExecutionScope scope(emulator(COHERENT));
        accel::resetTransferStats();
        UMat out;
        cv::flip(src, out, 0);              // the device reads the Mat where it is
        EXPECT_EQ(1u, accel::getTransferStats().deviceOps);
        EXPECT_EQ(0u, accel::getTransferStats().transfers);
        EXPECT_EQ(0, maxDiff(ref, out.getMat(ACCESS_READ)));
    }
    {
        accel::ExecutionScope scope(emulator(DISCRETE));
        accel::resetTransferStats();
        UMat out;
        cv::flip(src, out, 0);              // a discrete device needs one upload
        EXPECT_EQ((uint64)(src.total() * src.elemSize()), accel::getTransferStats().bytesHostToDevice);
        EXPECT_EQ(0, maxDiff(ref, out.getMat(ACCESS_READ)));
    }
}

TEST(Core_Accel, mat_gets_back_what_was_written_through_its_umat)
{
    // coherent shared memory: the UMat holds its own copy of the Mat's bytes
    accel::Device d = emulator(COHERENT);
    Mat m(8, 8, CV_8UC1, Scalar(3));
    {
        accel::ExecutionScope scope(d);
        UMat u = m.getUMat(ACCESS_RW);
        u.getMat(ACCESS_WRITE).setTo(Scalar(9));    // a CPU write through the UMat
    }
    EXPECT_EQ(0, maxDiff(Mat(8, 8, CV_8UC1, Scalar(9)), m));
    {
        accel::ExecutionScope scope(d);
        UMat u = m.getUMat(ACCESS_RW);
        Mat(8, 8, CV_8UC1, Scalar(5)).copyTo(u);    // an upload
    }
    EXPECT_EQ(0, maxDiff(Mat(8, 8, CV_8UC1, Scalar(5)), m));
}

TEST(Core_Accel, mat_backed_umat_stays_inside_the_mat)
{
    // external memory whose last row ends before its step does, followed by other data
    const int rows = 4, cols = 6, step = 16;
    const size_t extent = (size_t)step * (rows - 1) + cols;
    std::vector<uchar> buf(extent + step, 0xEE);
    Mat m(rows, cols, CV_8UC1, buf.data(), step);
    Mat src(rows, cols, CV_8UC1), ref;
    randu(src, 0, 255);
    cv::flip(src, ref, 0);
    {
        accel::ExecutionScope scope(emulator(DISCRETE));
        UMat u = m.getUMat(ACCESS_RW);
        cv::flip(src, u, 0);                                // written on the device
        std::fill(buf.begin() + extent, buf.end(), 0x11);   // the data after the Mat changes meanwhile
    }
    EXPECT_EQ(0, maxDiff(ref, m));
    for (size_t i = extent; i < buf.size(); i++)
        ASSERT_EQ(0x11, buf[i]) << "byte " << i << " after the Mat was overwritten";
}

TEST(Core_Accel, mat_calls_inside_a_scope_run_on_its_device)
{
    Mat src(16, 24, CV_8UC1), ref;
    randu(src, 0, 255);
    cv::flip(src, ref, 0);
    for (int dev : {DISCRETE, COHERENT})
    {
        accel::ExecutionScope scope(emulator(dev));
        accel::resetTransferStats();
        Mat dst;
        cv::flip(src, dst, 0);          // Mat in, Mat out: still the scope's device
        EXPECT_EQ(1u, accel::getTransferStats().deviceOps) << "device " << dev;
        EXPECT_EQ(0, maxDiff(ref, dst));
    }
}

TEST(Core_Accel, host_outputs_are_written_in_place_when_the_device_can)
{
    // a coherent iGPU addresses host memory: no temporary, no copy back
    Mat src(32, 40, CV_8UC4), ref;
    randu(src, 0, 255);
    cv::flip(src, ref, 0);
    accel::ExecutionScope scope(emulator(COHERENT));
    accel::resetTransferStats();
    Mat dst;
    cv::flip(src, dst, 0);
    EXPECT_EQ(1u, accel::getTransferStats().deviceOps);
    EXPECT_EQ(0u, accel::getTransferStats().transfers);
    EXPECT_EQ(0, maxDiff(ref, dst));

    Mat same = src.clone();
    cv::flip(same, same, 0);            // output over its own input: through a temporary
    EXPECT_EQ(0, maxDiff(ref, same));
}

TEST(Core_Accel, device_choice_follows_coverage)
{
    // the input's device lacks the operation, the output's device has it
    Mat src(16, 16, CV_8UC1), ref;
    randu(src, 0, 255);
    cv::flip(src, ref, 1);
    UMat in = makeOn(emulator(DISCRETE), src);
    UMat out = makeOn(emulator(COHERENT), Mat::zeros(16, 16, CV_8UC1));
    accel::detail::emulatorDisableOps(DISCRETE, true);
    accel::resetTransferStats();
    cv::flip(in, out, 1);
    accel::detail::emulatorDisableOps(DISCRETE, false);
    EXPECT_EQ(1u, accel::getTransferStats().deviceOps);    // not a CPU fallback
    EXPECT_EQ(0u, accel::getTransferStats().cpuFallbacks);
    EXPECT_EQ(0, maxDiff(ref, out.getMat(ACCESS_READ)));
}

TEST(Core_Accel, device_choice_follows_transfer_cost)
{
    // a call with a small input on the discrete device, a large one on the coherent device: the
    // coherent device moves 64 bytes, the discrete one would move 4096
    UMat small = makeOn(emulator(DISCRETE), Mat(8, 8, CV_8UC1, Scalar(5)));
    UMat large = makeOn(emulator(COHERENT), Mat(64, 64, CV_8UC1, Scalar(1)));
    UMat out;
    accel::detail::OpCall call("test.two_inputs", CV_ACCEL_OPS_CORE, CV_ACCEL_OPS_CORE_VERSION);
    call.input(small);
    call.input(large);
    call.output(out, small.shape(), small.type());
    EXPECT_EQ(accel::detail::HANDLED, call.run([](const void* t, const cvAccelCall* c, const cvAccelView* v) {
        return ((const cvAccelOpsCore_v0*)t)->flip(c, &v[0], &v[2], 1);    // the large input is only read
    }));
    EXPECT_TRUE(emulator(COHERENT) == accel::describe(out).device);
    EXPECT_EQ(5, out.getMat(ACCESS_READ).at<uchar>(0, 0));
}

TEST(Core_Accel, clone_stays_on_the_device)
{
    accel::Device d = emulator(DISCRETE);
    Mat src(12, 12, CV_8UC1);
    randu(src, 0, 255);
    UMat u = makeOn(d, src);
    accel::resetTransferStats();
    UMat c = u.clone();
    EXPECT_TRUE(d == accel::describe(c).device);
    EXPECT_EQ((uint64)src.total(), accel::getTransferStats().bytesDeviceToDevice);
    EXPECT_EQ(0u, accel::getTransferStats().bytesDeviceToHost);
    EXPECT_EQ(0, maxDiff(src, c.getMat(ACCESS_READ)));
}

TEST(Core_Accel, wrapped_host_memory_and_overlapping_imports)
{
    accel::Device d = emulator(COHERENT);
    std::vector<uchar> storage(4096 * 3);
    uchar* p = alignPtr(storage.data(), 4096);
    Mat host(64, 64, CV_8UC1, p);
    randu(host, 0, 255);

    accel::ExternalMemory mem;
    mem.type = accel::ExternalMemory::HOST_POINTER;
    mem.handle = (int64)(intptr_t)p;
    mem.size = host.total();
    UMat w = accel::wrapExternalMemory(d, mem, MatShape({64, 64}), CV_8UC1);
    EXPECT_TRUE(accel::describe(w).imported);

    // the same memory wrapped again is one memory to OpenCV, whatever UMat reaches it
    UMat again = accel::wrapExternalMemory(d, mem, MatShape({64, 64}), CV_8UC1);
    EXPECT_EQ(accel::detail::backingId(w), accel::detail::backingId(again));
    again.release();
    // a range that overlaps it without lying inside it cannot be one memory: refused
    accel::ExternalMemory wider = mem;
    wider.size = host.total() + 64;
    EXPECT_THROW(accel::wrapExternalMemory(d, wider, MatShape({1, (int)wider.size}), CV_8UC1), cv::Exception);
    // other memory on the same pages cannot be imported (pinning works on whole pages)
    accel::ExternalMemory next = mem, neighbour = mem;
    next.handle = (int64)(intptr_t)(p + 4096);
    next.size = 64;
    UMat n = accel::wrapExternalMemory(d, next, MatShape({1, 64}), CV_8UC1);
    neighbour.handle = (int64)(intptr_t)(p + 4096 + 128);
    neighbour.size = 64;
    EXPECT_THROW(accel::wrapExternalMemory(d, neighbour, MatShape({1, 64}), CV_8UC1), cv::Exception);
    n.release();

    UMat out;
    cv::flip(w, out, 1);
    Mat ref;
    cv::flip(host, ref, 1);
    EXPECT_EQ(0, maxDiff(ref, out.getMat(ACCESS_READ)));

    w.release();    // waits for the device: the memory goes back to its owner
    EXPECT_NO_THROW(accel::wrapExternalMemory(d, mem, MatShape({64, 64}), CV_8UC1));

    // this device imports host memory on 64-byte boundaries only
    accel::ExternalMemory odd = mem;
    odd.handle = (int64)(intptr_t)(p + 1);
    odd.size = 63 * 64;
    EXPECT_THROW(accel::wrapExternalMemory(d, odd, MatShape({63, 64}), CV_8UC1), cv::Exception);
}

TEST(Core_Accel, a_mat_over_wrapped_host_memory_overlaps_its_umat)
{
    // a UMat over host memory and a Mat one row further: the output overlaps the input, so the
    // device writes a temporary, and the result is the flip of what the input was
    accel::Device d = emulator(COHERENT);
    std::vector<uchar> storage(4096 * 2);
    uchar* p = alignPtr(storage.data(), 4096);
    Mat host(16, 16, CV_8UC1, p), ref;
    randu(host, 0, 255);
    cv::flip(host.clone(), ref, 0);
    accel::ExternalMemory mem;
    mem.type = accel::ExternalMemory::HOST_POINTER;
    mem.handle = (int64)(intptr_t)p;
    mem.size = host.total();
    UMat w = accel::wrapExternalMemory(d, mem, MatShape({16, 16}), CV_8UC1);
    Mat out(16, 16, CV_8UC1, p + 16);
    cv::flip(w, out, 0);
    EXPECT_EQ(0, maxDiff(ref, out));
}

TEST(Core_Accel, host_pages_stay_reserved_while_their_import_is_not_released)
{
    // the backend fails to release an import of host memory: it may still pin the pages, so
    // no other import of them is let in
    accel::Device d = emulator(COHERENT);
    // never freed: the pages a run leaves reserved are never another test's, nor the next run's
    static uchar storage[4096 * (2 * 32 + 1)];
    static int run = 0;
    uchar* p = alignPtr(storage, 4096) + 2 * 4096 * (run++ % 32);
    accel::ExternalMemory mem;
    mem.type = accel::ExternalMemory::HOST_POINTER;
    mem.handle = (int64)(intptr_t)p;
    mem.size = 4096;
    UMat w = accel::wrapExternalMemory(d, mem, MatShape({1, 4096}), CV_8UC1);
    accel::detail::emulatorFailNextDeallocate();
    w.release();
    EXPECT_THROW(accel::wrapExternalMemory(d, mem, MatShape({1, 4096}), CV_8UC1), cv::Exception);
    accel::ExternalMemory next = mem;
    next.handle = (int64)(intptr_t)(p + 4096);     // other pages: fine
    EXPECT_NO_THROW(accel::wrapExternalMemory(d, next, MatShape({1, 4096}), CV_8UC1));
}

TEST(Core_Accel, wrapped_device_memory_reads_its_content)
{
    // like a device pointer from another library: the content is in device memory, and the
    // host copy that the CPU needs does not exist yet
    accel::Device d = emulator(DISCRETE);
    Mat src(24, 32, CV_8UC1);
    randu(src, 0, 255);
    UMat owner = makeOn(d, src);
    accel::ExternalMemory mem;
    mem.type = accel::ExternalMemory::DEVICE_POINTER;
    mem.handle = (int64)(intptr_t)owner.handle(ACCESS_READ);
    mem.size = src.total();
    {
        UMat w = accel::wrapExternalMemory(d, mem, MatShape({24, 32}), CV_8UC1);
        EXPECT_EQ(0, maxDiff(src, w.getMat(ACCESS_READ)));
    }
    {
        UMat w = accel::wrapExternalMemory(d, mem, MatShape({24, 32}), CV_8UC1);
        Mat res;
        w.copyTo(res);
        EXPECT_EQ(0, maxDiff(src, res));
    }
}

TEST(Core_Accel, ranges_outside_the_memory_are_refused)
{
    accel::Device d = emulator(DISCRETE);
    accel::ExternalMemory mem;
    mem.type = accel::ExternalMemory::DEVICE_POINTER;
    mem.handle = accel::detail::emulatorAllocateExternal(DISCRETE, 256, 0, false);
    mem.size = 256;
    mem.offset = std::numeric_limits<size_t>::max() - 16;  // offset + size wraps around
    EXPECT_THROW(accel::wrapExternalMemory(d, mem, MatShape({16, 16}), CV_8UC1), cv::Exception);
    mem.offset = 200;                                       // inside the allocation, but runs past its end
    mem.size = 100;
    EXPECT_THROW(accel::wrapExternalMemory(d, mem, MatShape({1, 100}), CV_8UC1), cv::Exception);
    // rows 2^63 bytes apart: the extent wraps around to 1 byte in 64 bits
    mem.offset = 0;
    mem.size = 256;
    const size_t huge[2] = { (size_t)1 << (sizeof(size_t) * 8 - 1), 1 };
    EXPECT_THROW(accel::wrapExternalMemory(d, mem, MatShape({3, 1}), CV_8UC1, huge), cv::Exception);
    // packed steps that overflow
    EXPECT_THROW(accel::wrapExternalMemory(d, mem, MatShape({INT_MAX, INT_MAX, INT_MAX}), CV_64FC4), cv::Exception);
    // a protected allocation is checked over the whole range too, not only where it starts
    accel::ExternalMemory secure;
    secure.type = accel::ExternalMemory::DEVICE_POINTER;
    secure.handle = accel::detail::emulatorAllocateExternal(DISCRETE, 64, 7, true);
    secure.offset = 32;
    secure.size = 64;
    EXPECT_THROW(accel::wrapExternalMemory(d, secure, MatShape({8, 8}), CV_8UC1), cv::Exception);
}

TEST(Core_Accel, released_memory_in_flight_is_revived_with_its_fences)
{
    // A UMat is released while a device write into it is still queued; its memory, still
    // allocated, is wrapped again through its device pointer. The wrap must reach the same
    // state, fences included, instead of a new one that knows nothing of the queued write.
    accel::Device d = emulator(DISCRETE);
    runWithDeviceDelay(30000, [&]() {
        UMat owner = makeOn(d, Mat(16, 16, CV_8UC1, Scalar(1)));
        const uint64 id = accel::detail::backingId(owner);
        accel::ExternalMemory mem;
        mem.type = accel::ExternalMemory::DEVICE_POINTER;
        mem.handle = (int64)(intptr_t)owner.handle(ACCESS_READ);
        mem.size = 256;
        UMat src = makeOn(d, Mat(16, 16, CV_8UC1, Scalar(9)));
        cv::flip(src, owner, 1);    // queued for 30 ms
        owner.release();            // freed only once the flip completed
        UMat again = accel::wrapExternalMemory(d, mem, MatShape({16, 16}), CV_8UC1);
        EXPECT_EQ(id, accel::detail::backingId(again));
        EXPECT_EQ(9, again.getMat(ACCESS_READ).at<uchar>(0, 0));   // waits for the flip
    });
}

TEST(Core_Accel, borrowed_memory_rewrapped_during_its_release_is_one_memory)
{
    // The last UMat of wrapped host memory goes away while device work through it is in flight,
    // so its release waits; meanwhile another thread wraps the same memory. The wrap gets the
    // state that orders that work, not a second one.
    accel::Device d = emulator(COHERENT);
    std::vector<uchar> storage(4096 * 2);
    uchar* p = alignPtr(storage.data(), 4096);
    accel::ExternalMemory mem;
    mem.type = accel::ExternalMemory::HOST_POINTER;
    mem.handle = (int64)(intptr_t)p;
    mem.size = 1024;
    uint64 id = 0;
    std::atomic<bool> released(false);
    std::thread owner;
    runWithDeviceDelay(50000, [&]() {
        UMat w = accel::wrapExternalMemory(d, mem, MatShape({32, 32}), CV_8UC1);
        id = accel::detail::backingId(w);
        accel::Stream s = accel::Stream::getDefault(d);
        {
            accel::DeviceAccess acc(w, s, ACCESS_WRITE);    // the caller's work on the memory...
            UMat a = makeOn(d, Mat(32, 32, CV_8UC1, Scalar(2))), b;
            cv::flip(a, b, 0);                              // ...ends after this, 50 ms from now
        }
        owner = std::thread([&released](UMat last) {
            last.release();         // the last reference: waits for the device
            released = true;
        }, std::move(w));
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    EXPECT_FALSE(released.load());
    UMat again = accel::wrapExternalMemory(d, mem, MatShape({32, 32}), CV_8UC1);
    EXPECT_EQ(id, accel::detail::backingId(again));   // the same memory, its fences included
    owner.join();
    EXPECT_TRUE(released.load());
}

#ifdef __linux__
TEST(Core_Accel, planes_of_one_dma_buf_are_one_memory)
{
    accel::Device d = emulator(COHERENT);
    // NV12 16x16 in one dma-buf: luma rows of 16 bytes, chroma at offset 256
    const int64 fd = accel::detail::emulatorCreateDmaBuf(256 + 128);
    accel::ExternalImage nv12;
    nv12.type = accel::ExternalMemory::DMABUF_FD;
    accel::ExternalPlane y, uv;
    y.handle = uv.handle = fd;
    y.rows = y.cols = 16;
    y.type = CV_8UC1;
    uv.offset = 256;
    uv.rows = uv.cols = 8;
    uv.type = CV_8UC2;
    nv12.planes.push_back(y);
    nv12.planes.push_back(uv);
    const int64 imports = accel::detail::emulatorLiveAllocations(COHERENT);
    std::vector<UMat> planes = accel::wrapExternalImage(d, nv12);
    EXPECT_EQ(imports + 1, accel::detail::emulatorLiveAllocations(COHERENT));  // one import of the buffer
    EXPECT_EQ(accel::detail::backingId(planes[0]), accel::detail::backingId(planes[1]));
    {
        // The CPU writes the luma plane. Cache maintenance of a dma-buf covers the whole buffer,
        // so no device may touch the chroma plane until that CPU access ends.
        Mat luma = planes[0].getMat(ACCESS_RW);
        luma.setTo(Scalar(1));
        Mat chroma = planes[1].getMat(ACCESS_READ);     // joins the open CPU access
        EXPECT_EQ(Vec2b(0, 0), chroma.at<Vec2b>(0, 0));
        accel::resetTransferStats();
        UMat src = makeOn(d, Mat(8, 8, CV_8UC2, Scalar(5, 6)));
        cv::flip(src, planes[1], 1);
        EXPECT_EQ(1u, accel::getTransferStats().cpuFallbacks);
    }
    UMat src = makeOn(d, Mat(8, 8, CV_8UC2, Scalar(7, 8)));
    accel::resetTransferStats();
    cv::flip(src, planes[1], 1);                        // the CPU is done: on the device
    EXPECT_EQ(1u, accel::getTransferStats().deviceOps);
    EXPECT_EQ(Vec2b(7, 8), planes[1].getMat(ACCESS_READ).at<Vec2b>(7, 7));
    EXPECT_EQ(1, planes[0].getMat(ACCESS_READ).at<uchar>(15, 15));
}

TEST(Core_Accel, one_dma_buf_imported_by_two_devices_is_one_memory)
{
    // a camera frame used by a GPU (emulator:1) and an NPU (emulator:2)
    const int64 fd = accel::detail::emulatorCreateDmaBuf(256);
    accel::ExternalMemory mem;
    mem.type = accel::ExternalMemory::DMABUF_FD;
    mem.handle = fd;
    mem.size = 256;
    UMat gpu = accel::wrapExternalMemory(emulator(COHERENT), mem, MatShape({16, 16}), CV_8UC1);
    UMat npu = accel::wrapExternalMemory(emulator(NONCOHERENT), mem, MatShape({16, 16}), CV_8UC1);
    EXPECT_EQ(accel::detail::backingId(gpu), accel::detail::backingId(npu));
    {
        Mat cpu = gpu.getMat(ACCESS_RW);    // the CPU writes through the GPU's import...
        cpu.setTo(Scalar(3));
        accel::resetTransferStats();
        UMat out;
        cv::flip(npu, out, 0);              // ...so the NPU may not read the frame yet
        EXPECT_EQ(1u, accel::getTransferStats().cpuFallbacks);
        EXPECT_EQ(3, out.getMat(ACCESS_READ).at<uchar>(0, 0));
    }
    accel::resetTransferStats();
    UMat out;
    cv::flip(npu, out, 0);
    EXPECT_EQ(1u, accel::getTransferStats().deviceOps);
    EXPECT_EQ(3, out.getMat(ACCESS_READ).at<uchar>(15, 15));
}
#endif

TEST(Core_Accel, backends_must_agree_that_memory_is_protected)
{
    // a protected frame of emulator:0, also imported by emulator:5, which never attests
    // protection: one memory cannot be protected for one importer and readable for the other
    const int64 live = accel::detail::emulatorLiveAllocations();
    accel::ExternalMemory mem;
    mem.type = accel::ExternalMemory::DEVICE_POINTER;
    mem.handle = accel::detail::emulatorAllocateExternal(DISCRETE, 64, 7, true);
    mem.size = 64;
    UMat secure = accel::wrapExternalMemory(emulator(DISCRETE), mem, MatShape({8, 8}), CV_8UC1);
    EXPECT_TRUE(accel::describe(secure).protectedContent);
    const int64 imported = accel::detail::emulatorLiveAllocations();
    EXPECT_THROW(accel::wrapExternalMemory(failingDevice(), mem, MatShape({8, 8}), CV_8UC1), cv::Exception);
    EXPECT_EQ(imported, accel::detail::emulatorLiveAllocations());     // the refused import was released
    secure.release();
    EXPECT_EQ(live + 1, accel::detail::emulatorLiveAllocations());     // the protected allocation itself stays
}

#ifdef __linux__
TEST(Core_Accel, dma_buf_fences_of_other_drivers_are_honoured)
{
    // a dma-buf that another driver (a camera) fills: the frame is valid once its fence completes
    accel::Device d = emulator(COHERENT);
    const int64 fd = accel::detail::emulatorCreateDmaBuf(16 * 16);
    accel::ExternalMemory mem;
    mem.type = accel::ExternalMemory::DMABUF_FD;
    mem.handle = fd;
    mem.size = 16 * 16;
    UMat frame = accel::wrapExternalMemory(d, mem, MatShape({16, 16}), CV_8UC1);

    accel::detail::emulatorDmaBufProduce(fd, 9, 20000);    // the camera writes 9s, 20 ms from now
    UMat out;
    cv::flip(frame, out, 1);                                // the device waits for the camera's fence
    EXPECT_EQ(9, out.getMat(ACCESS_READ).at<uchar>(0, 0));

    accel::detail::emulatorDmaBufProduce(fd, 4, 20000);
    EXPECT_EQ(4, frame.getMat(ACCESS_READ).at<uchar>(0, 0));   // and so does the CPU

    const int attached = accel::detail::emulatorDmaBufAttachedFences(fd);
    UMat src = makeOn(d, Mat(16, 16, CV_8UC1, Scalar(5)));
    cv::flip(src, frame, 0);                                // a device write into the dma-buf...
    EXPECT_GT(accel::detail::emulatorDmaBufAttachedFences(fd), attached);  // ...hands its fence on
}

TEST(Core_Accel, other_drivers_wait_for_opencv_reads)
{
    // dma_resv semantics: a camera that writes the next frame into the buffer waits for every
    // device read of the previous one, not only for the last write
    accel::Device d = emulator(COHERENT);
    const int64 fd = accel::detail::emulatorCreateDmaBuf(256);
    accel::detail::emulatorDmaBufProduce(fd, 1, 0);
    accel::ExternalMemory mem;
    mem.type = accel::ExternalMemory::DMABUF_FD;
    mem.handle = fd;
    mem.size = 256;
    UMat frame = accel::wrapExternalMemory(d, mem, MatShape({16, 16}), CV_8UC1);
    runWithDeviceDelay(30000, [&]() {
        UMat out;
        cv::flip(frame, out, 1);                        // reads the frame 30 ms from now
        accel::detail::emulatorDmaBufProduce(fd, 9, 0); // the next frame: after that read
        EXPECT_EQ(0, maxDiff(Mat(16, 16, CV_8UC1, Scalar(1)), out.getMat(ACCESS_READ)));
    });
    EXPECT_EQ(9, frame.getMat(ACCESS_READ).at<uchar>(0, 0));
}

TEST(Core_Accel, work_whose_fence_cannot_be_attached_ends_before_the_call)
{
    // when end_device_access fails, other drivers would not wait for the work: the call that
    // queued it waits instead, so the buffer holds the result as soon as the call returns
    accel::Device d = emulator(COHERENT);
    const int64 fd = accel::detail::emulatorCreateDmaBuf(256);
    accel::ExternalMemory mem;
    mem.type = accel::ExternalMemory::DMABUF_FD;
    mem.handle = fd;
    mem.size = 256;
    UMat frame = accel::wrapExternalMemory(d, mem, MatShape({16, 16}), CV_8UC1);
    const uchar* raw = (const uchar*)mmap(NULL, 256, PROT_READ, MAP_SHARED, (int)fd, 0);     // another driver's view
    ASSERT_TRUE(raw != MAP_FAILED);
    runWithDeviceDelay(30000, [&]() {
        UMat src = makeOn(d, Mat(16, 16, CV_8UC1, Scalar(4)));
        accel::detail::emulatorFailNextDeviceAccessEnd();
        accel::resetTransferStats();
        cv::flip(src, frame, 0);
        EXPECT_GE(accel::getTransferStats().hostWaits, 1u);
        EXPECT_EQ(4, raw[255]);
    });
    munmap((void*)raw, 256);
}

TEST(Core_Accel, every_device_access_bracket_ends)
{
#ifdef __linux__
    // two camera buffers in one call: when the second bracket cannot begin, the first one ends
    // anyway, and so does every bracket of a call that runs
    accel::Device d = emulator(COHERENT);
    const int64 fa = accel::detail::emulatorCreateDmaBuf(256), fb = accel::detail::emulatorCreateDmaBuf(256);
    accel::ExternalMemory ma, mb;
    ma.type = mb.type = accel::ExternalMemory::DMABUF_FD;
    ma.handle = fa;
    mb.handle = fb;
    ma.size = mb.size = 256;
    UMat a = accel::wrapExternalMemory(d, ma, MatShape({16, 16}), CV_8UC1);
    UMat b = accel::wrapExternalMemory(d, mb, MatShape({16, 16}), CV_8UC1);
    const int64 unmatched = accel::detail::emulatorViolations(accel::detail::EMU_VIOLATION_BRACKET);
    cv::flip(a, b, 0);
    EXPECT_EQ(0, accel::detail::emulatorDmaBufOpenBrackets(fa));
    EXPECT_EQ(0, accel::detail::emulatorDmaBufOpenBrackets(fb));
    accel::detail::emulatorFailNextDeviceAccessBegin(1);
    EXPECT_THROW(cv::flip(a, b, 0), cv::Exception);
    EXPECT_EQ(0, accel::detail::emulatorDmaBufOpenBrackets(fa));
    EXPECT_EQ(0, accel::detail::emulatorDmaBufOpenBrackets(fb));
    EXPECT_EQ(unmatched, accel::detail::emulatorViolations(accel::detail::EMU_VIOLATION_BRACKET));
#endif
}

TEST(Core_Accel, dma_bufs_need_a_backend_that_honours_their_fences)
{
    accel::detail::registerLegacyEmulatorBackend();
    accel::Device legacy = accel::Device::find("emulator-v1:1");   // a plugin of API version 1
    ASSERT_FALSE(legacy.empty());
    const int64 fd = accel::detail::emulatorCreateDmaBuf(256);
    accel::ExternalMemory mem;
    mem.type = accel::ExternalMemory::DMABUF_FD;
    mem.handle = fd;
    mem.size = 256;
    const int64 live = accel::detail::emulatorLiveAllocations(COHERENT);
    EXPECT_THROW(accel::wrapExternalMemory(legacy, mem, MatShape({16, 16}), CV_8UC1), cv::Exception);
    EXPECT_EQ(live, accel::detail::emulatorLiveAllocations(COHERENT));    // the import was given back
    EXPECT_NO_THROW(accel::wrapExternalMemory(emulator(COHERENT), mem, MatShape({16, 16}), CV_8UC1));
}

TEST(Core_Accel, multi_plane_images_in_one_or_several_buffers)
{
    accel::Device d = emulator(COHERENT);
    // NV12, 16x16, in one dma-buf: luma rows of 32 bytes, then chroma (8x8 pairs) at offset 512
    const int64 fd = accel::detail::emulatorCreateDmaBuf(512 + 32 * 8);
    accel::detail::emulatorDmaBufProduce(fd, 7, 0);
    accel::ExternalImage nv12;
    nv12.type = accel::ExternalMemory::DMABUF_FD;
    accel::ExternalPlane y, uv;
    y.handle = uv.handle = fd;
    y.pitch = uv.pitch = 32;
    y.rows = y.cols = 16;
    y.type = CV_8UC1;
    uv.offset = 512;
    uv.rows = uv.cols = 8;
    uv.type = CV_8UC2;
    nv12.planes.push_back(y);
    nv12.planes.push_back(uv);
    std::vector<UMat> planes = accel::wrapExternalImage(d, nv12);
    ASSERT_EQ(2u, planes.size());
    EXPECT_EQ(Size(16, 16), planes[0].size());
    EXPECT_EQ(32u, planes[0].step[0]);
    EXPECT_EQ(CV_8UC2, planes[1].type());
    EXPECT_EQ(Vec2b(7, 7), planes[1].getMat(ACCESS_READ).at<Vec2b>(7, 7));
    UMat flipped;
    accel::resetTransferStats();
    cv::flip(planes[0], flipped, 1);
    EXPECT_EQ(1u, accel::getTransferStats().deviceOps);
    EXPECT_EQ(7, flipped.getMat(ACCESS_READ).at<uchar>(0, 0));

    // one fd per plane
    const int64 fdY = accel::detail::emulatorCreateDmaBuf(256), fdUV = accel::detail::emulatorCreateDmaBuf(128);
    accel::detail::emulatorDmaBufProduce(fdUV, 3, 0);
    nv12.planes[0].handle = fdY;
    nv12.planes[0].pitch = 16;
    nv12.planes[1].handle = fdUV;
    nv12.planes[1].offset = 0;
    nv12.planes[1].pitch = 16;
    planes = accel::wrapExternalImage(d, nv12);
    EXPECT_EQ(Vec2b(3, 3), planes[1].getMat(ACCESS_READ).at<Vec2b>(0, 0));
}

static const unsigned FOURCC_NV12 = 'N' | ('V' << 8) | ('1' << 16) | ('2' << 24);

//! NV12 16x16 in one dma-buf: luma rows of 16 bytes at 0, chroma (8x8 pairs) at @p uvOffset
static accel::ExternalImage nv12Image(int64 fd, size_t uvOffset)
{
    accel::ExternalImage img;
    img.type = accel::ExternalMemory::DMABUF_FD;
    img.fourcc = FOURCC_NV12;
    accel::ExternalPlane y, uv;
    y.handle = uv.handle = fd;
    y.rows = y.cols = 16;
    y.type = CV_8UC1;
    uv.offset = uvOffset;
    uv.rows = uv.cols = 8;
    uv.type = CV_8UC2;
    img.planes.push_back(y);
    img.planes.push_back(uv);
    return img;
}

TEST(Core_Accel, image_planes_carry_their_image_to_the_backend)
{
    accel::Device d = emulator(COHERENT);
    const int64 fd = accel::detail::emulatorCreateDmaBuf(256 + 128);
    accel::detail::emulatorDmaBufProduce(fd, 5, 0);
    accel::ExternalImage img = nv12Image(fd, 256);
    img.color.model = accel::COLOR_MODEL_BT709;
    img.color.range = accel::COLOR_RANGE_LIMITED;
    img.color.chromaSitingX = accel::CHROMA_SITING_COSITED;
    img.color.chromaSitingY = accel::CHROMA_SITING_MIDPOINT;
    std::vector<UMat> planes = accel::wrapExternalImage(d, img);
    const accel::ImagePlaneInfo info = accel::describePlane(planes[1]);
    EXPECT_NE(0u, info.imageId);
    EXPECT_EQ(accel::describePlane(planes[0]).imageId, info.imageId);
    EXPECT_EQ(1, info.plane);
    EXPECT_EQ(2, info.planeCount);
    EXPECT_EQ(FOURCC_NV12, info.fourcc);
    EXPECT_EQ(accel::COLOR_RANGE_LIMITED, info.color.range);

    // an operation on both planes: QUERY and EXECUTE see one image, its format and its colours
    UMat out;
    int calls = 0;
    bool seen = true;
    accel::detail::OpCall call("test.nv12_luma", CV_ACCEL_OPS_CORE, CV_ACCEL_OPS_CORE_VERSION);
    call.input(planes[0]);
    call.input(planes[1]);
    call.output(out, planes[0].shape(), CV_8UC1);
    EXPECT_EQ(accel::detail::HANDLED, call.run([&](const void* t, const cvAccelCall* c, const cvAccelView* v) {
        calls++;
        seen = seen && v[0].image && v[0].image == v[1].image && v[0].plane == 0 && v[1].plane == 1 &&
               v[0].image->fourcc == FOURCC_NV12 && v[0].image->plane_count == 2 &&
               v[0].image->planes[1].type == CV_8UC2 && v[0].image->planes[1].offset == 256 &&
               v[0].image->color_model == CV_ACCEL_COLOR_MODEL_BT709 &&
               v[0].image->chroma_siting_y == CV_ACCEL_CHROMA_SITING_MIDPOINT && v[2].image == NULL;
        return ((const cvAccelOpsCore_v0*)t)->flip(c, &v[0], &v[2], 1);
    }));
    EXPECT_EQ(2, calls);
    EXPECT_TRUE(seen);
    EXPECT_EQ(5, out.getMat(ACCESS_READ).at<uchar>(0, 0));
}

TEST(Core_Accel, planes_copied_to_another_device_are_still_planes_of_their_image)
{
    // a frame of the coherent device used on the discrete one: its planes are uploaded into
    // temporaries, and each view still says which image and which plane it is, and where its
    // copy lies (also for a part of a plane, and for one plane copied twice)
    const int64 fd = accel::detail::emulatorCreateDmaBuf(256 + 128);
    accel::detail::emulatorDmaBufProduce(fd, 5, 0);
    accel::ExternalImage img = nv12Image(fd, 256);
    img.color.model = accel::COLOR_MODEL_BT709;
    std::vector<UMat> planes = accel::wrapExternalImage(emulator(COHERENT), img);
    const uint64 id = accel::describePlane(planes[0]).imageId;
    accel::ExecutionScope scope(emulator(DISCRETE));
    UMat out;
    int calls = 0;
    bool seen = true;
    accel::detail::OpCall call("test.nv12_luma", CV_ACCEL_OPS_CORE, CV_ACCEL_OPS_CORE_VERSION);
    call.input(planes[0]);
    call.input(planes[1]);
    call.input(planes[0].rowRange(0, 8));
    call.output(out, planes[0].shape(), CV_8UC1);
    EXPECT_EQ(accel::detail::HANDLED, call.run([&](const void* t, const cvAccelCall* c, const cvAccelView* v) {
        calls++;
        for (int i = 0; i < 3; i++)
        {
            const cvAccelImageDesc* im = v[i].image;
            const int k = i == 1 ? 1 : 0;
            seen = seen && im && im->id == id && (int)v[i].plane == k && im->fourcc == FOURCC_NV12 &&
                   im->color_model == CV_ACCEL_COLOR_MODEL_BT709 && im->planes[k].offset == v[i].offset &&
                   im->planes[k].pitch == v[i].step[0] && im->planes[k].rows == v[i].size[0];
        }
        seen = seen && v[0].image != v[2].image && v[0].image->planes[0].buffer != v[2].image->planes[0].buffer &&
               v[2].image->planes[0].rows == 8 && v[3].image == NULL;
        return ((const cvAccelOpsCore_v0*)t)->flip(c, &v[0], &v[3], 1);
    }));
    EXPECT_EQ(2, calls);
    EXPECT_TRUE(seen);
    EXPECT_EQ(5, out.getMat(ACCESS_READ).at<uchar>(0, 0));
}

TEST(Core_Accel, image_planes_are_checked)
{
    accel::Device d = emulator(COHERENT);
    const int64 fd = accel::detail::emulatorCreateDmaBuf(1024);
    const accel::ExternalImage img = nv12Image(fd, 256);
    EXPECT_NO_THROW(accel::wrapExternalImage(d, img));
    accel::ExternalImage bad = img;
    bad.planes[1].cols = 16;                    // NV12 chroma is subsampled horizontally
    EXPECT_THROW(accel::wrapExternalImage(d, bad), cv::Exception);
    bad = img;
    bad.planes[1].type = CV_8UC1;               // and interleaved
    EXPECT_THROW(accel::wrapExternalImage(d, bad), cv::Exception);
    bad = img;
    bad.planes.pop_back();                      // and has two planes
    EXPECT_THROW(accel::wrapExternalImage(d, bad), cv::Exception);
    bad = img;
    bad.planes[1].offset = 128;                 // over the luma plane
    EXPECT_THROW(accel::wrapExternalImage(d, bad), cv::Exception);
    bad = img;
    bad.planes[0].pitch = std::numeric_limits<size_t>::max() / 4;   // pitch * rows does not fit
    EXPECT_THROW(accel::wrapExternalImage(d, bad), cv::Exception);
    accel::ExternalImage other = img;
    other.fourcc = 0;                           // a format OpenCV does not know: only the generic checks
    other.planes[1].cols = 16;
    EXPECT_NO_THROW(accel::wrapExternalImage(d, other));
}

TEST(Core_Accel, planes_are_grouped_by_the_memory_they_lie_in)
{
#ifdef __linux__
    // the chroma plane through another descriptor of the same dma-buf: still one buffer, and
    // planes that overlap in it are refused whatever handles they came through
    accel::Device d = emulator(COHERENT);
    const int64 fd = accel::detail::emulatorCreateDmaBuf(1024);
    const int fd2 = dup((int)fd);
    ASSERT_GE(fd2, 0);
    accel::ExternalImage img = nv12Image(fd, 256);
    img.planes[1].handle = fd2;
    std::vector<UMat> planes = accel::wrapExternalImage(d, img);
    EXPECT_EQ(accel::describePlane(planes[0]).buffer, accel::describePlane(planes[1]).buffer);
    EXPECT_EQ(256u, accel::describePlane(planes[1]).offset);
    planes.clear();
    img.planes[1].offset = 128;
    EXPECT_THROW(accel::wrapExternalImage(d, img), cv::Exception);
    close(fd2);
#endif
}

TEST(Core_Accel, image_planes_have_their_own_layouts)
{
    accel::Device d = emulator(COHERENT);
    const int64 fd = accel::detail::emulatorCreateDmaBuf(4096);
    const uint64 tiled = 0x0100000000000002ull;
    accel::ExternalImage img = nv12Image(fd, 1024);
    img.planes[1].layoutModifier = tiled;       // the chroma plane is tiled, the luma plane linear
    img.planes[1].size = 512;                   // tiles padded past 8 rows of 16 bytes
    std::vector<UMat> planes = accel::wrapExternalImage(d, img);
    EXPECT_EQ(0u, accel::describePlane(planes[0]).layoutModifier);
    EXPECT_EQ(tiled, accel::describePlane(planes[1]).layoutModifier);
    EXPECT_NO_THROW(planes[0].getMat(ACCESS_READ));             // linear: the CPU reads it
    EXPECT_THROW(planes[1].getMat(ACCESS_READ), cv::Exception); // tiled: no CPU view
    accel::ExternalImage noSize = img;
    noSize.planes[1].size = 0;                  // the size of a tiled plane cannot be computed
    EXPECT_THROW(accel::wrapExternalImage(d, noSize), cv::Exception);
}

TEST(Core_Accel, tiled_layouts_never_reach_linear_code)
{
    accel::Device d = emulator(COHERENT);
    accel::ExternalMemory mem;
    mem.type = accel::ExternalMemory::DMABUF_FD;
    mem.handle = accel::detail::emulatorCreateDmaBuf(256);
    mem.size = 256;
    mem.layoutModifier = 0x0100000000000002ull;    // a vendor tiling
    UMat tiled = accel::wrapExternalMemory(d, mem, MatShape({16, 16}), CV_8UC1);
    EXPECT_EQ(accel::HOST_ACCESS_NONE, accel::describe(tiled).hostAccess);
    EXPECT_THROW(tiled.getMat(ACCESS_READ), cv::Exception);   // no linear CPU view
    UMat out;
    EXPECT_THROW(cv::flip(tiled, out, 1), cv::Exception);     // the device's kernels are linear, and no CPU fallback

    // a backend that does not know the layout reports linear memory: refused
    accel::ExternalMemory ptr;
    ptr.type = accel::ExternalMemory::DEVICE_POINTER;
    ptr.handle = accel::detail::emulatorAllocateExternal(DISCRETE, 256, 0, false);
    ptr.size = 256;
    ptr.layoutModifier = 0x0100000000000002ull;
    EXPECT_THROW(accel::wrapExternalMemory(emulator(DISCRETE), ptr, MatShape({16, 16}), CV_8UC1), cv::Exception);
}

#endif // __linux__

TEST(Core_Accel, memory_reaches_another_device_as_a_dma_buf_without_copies)
{
#ifdef __linux__
    // A GPU result (emulator:1) used by an NPU (emulator:2) that cannot address the GPU's
    // memory: the NPU imports the same memory as a dma-buf instead of receiving a copy, and the
    // devices' work on it stays ordered as work on one memory.
    accel::Device gpu = emulator(COHERENT), npu = emulator(NONCOHERENT);
    Mat src(16, 24, CV_8UC1), ref;
    randu(src, 0, 255);
    cv::flip(src, ref, 0);
    UMat x = makeOn(gpu, src);
    UMat out = makeOn(npu, Mat::zeros(16, 24, CV_8UC1));
    accel::resetTransferStats();
    cv::flip(x, out, 0);            // cheapest on the NPU: its output in place, its input shared
    accel::TransferStats s = accel::getTransferStats();
    EXPECT_EQ(1u, s.deviceOps);
    EXPECT_EQ(0u, s.transfers);
    EXPECT_EQ(0, maxDiff(ref, out.getMat(ACCESS_READ)));
    x.setTo(Scalar(9));             // the GPU's memory changes: the NPU's next read sees it
    cv::flip(x, out, 0);
    EXPECT_EQ(9, out.getMat(ACCESS_READ).at<uchar>(0, 0));

    // exported, the memory is one buffer whichever device wraps the descriptor
    accel::ExternalMemory e = accel::exportExternalMemory(x);
    UMat again = accel::wrapExternalMemory(npu, e, MatShape({16, 24}), CV_8UC1);
    EXPECT_EQ(accel::detail::backingId(x), accel::detail::backingId(again));
    close((int)e.handle);

    // exported by another library holding the allocation: its backend tells whose memory it is
    UMat y = makeOn(gpu, src);
    accel::ExternalMemory f;
    f.type = accel::ExternalMemory::DMABUF_FD;
    f.handle = accel::detail::emulatorExportDmaBuf(y);
    f.size = y.total();
    UMat viaFd = accel::wrapExternalMemory(npu, f, MatShape({16, 24}), CV_8UC1);
    EXPECT_EQ(accel::detail::backingId(y), accel::detail::backingId(viaFd));
    close((int)f.handle);
#endif
}

TEST(Core_Accel, exports_keep_the_layout_and_the_image)
{
#ifdef __linux__
    // a tiled frame exported keeps its layout, whole; the same bytes are never wrapped as linear
    accel::Device gpu = emulator(COHERENT), npu = emulator(NONCOHERENT);
    accel::ExternalMemory mem;
    mem.type = accel::ExternalMemory::DMABUF_FD;
    mem.handle = accel::detail::emulatorCreateDmaBuf(256);
    mem.size = 256;
    mem.layoutModifier = 0x0100000000000002ull;
    UMat tiled = accel::wrapExternalMemory(gpu, mem, MatShape({16, 16}), CV_8UC1);
    accel::ExternalMemory e = accel::exportExternalMemory(tiled);
    EXPECT_EQ(mem.layoutModifier, e.layoutModifier);
    EXPECT_EQ(256u, e.size);
    UMat again = accel::wrapExternalMemory(gpu, e, MatShape({16, 16}), CV_8UC1);
    EXPECT_EQ(accel::HOST_ACCESS_NONE, accel::describe(again).hostAccess);
    accel::ExternalMemory linear = e;
    linear.layoutModifier = 0;
    EXPECT_THROW(accel::wrapExternalMemory(gpu, linear, MatShape({16, 16}), CV_8UC1), cv::Exception);
    EXPECT_THROW(accel::exportExternalMemory(tiled.rowRange(8, 16)), cv::Exception);   // a part of a tiled plane
    close((int)e.handle);

    // an image exported with its format, colours and planes, wrapped again on another device
    const int64 fd = accel::detail::emulatorCreateDmaBuf(256 + 128);
    accel::ExternalImage img = nv12Image(fd, 256);
    img.color.model = accel::COLOR_MODEL_BT709;
    img.color.range = accel::COLOR_RANGE_LIMITED;
    std::vector<UMat> planes = accel::wrapExternalImage(gpu, img);
    accel::ExternalImage x = accel::exportExternalImage(planes);
    EXPECT_EQ(FOURCC_NV12, x.fourcc);
    EXPECT_EQ(accel::COLOR_MODEL_BT709, x.color.model);
    ASSERT_EQ(2u, x.planes.size());
    EXPECT_EQ(256u, x.planes[1].offset);
    std::vector<UMat> other = accel::wrapExternalImage(npu, x);
    EXPECT_EQ(accel::detail::backingId(planes[1]), accel::detail::backingId(other[1]));
    EXPECT_EQ(accel::COLOR_RANGE_LIMITED, accel::describePlane(other[1]).color.range);
    for (size_t i = 0; i < x.planes.size(); i++)
        close((int)x.planes[i].handle);
#endif
}

TEST(Core_Accel, exports_are_settled_snapshots)
{
#ifdef __linux__
    // nothing is handed to other drivers while the CPU may still write it, or after failed writes
    UMat x = makeOn(emulator(COHERENT), Mat(8, 8, CV_8UC1, Scalar(1)));
    {
        Mat view = x.getMat(ACCESS_RW);
        EXPECT_THROW(accel::exportExternalMemory(x), cv::Exception);
    }
    UMat y = makeOn(emulator(COHERENT), Mat(8, 8, CV_8UC1, Scalar(1)));
    {
        Mat view = y.getMat(ACCESS_WRITE);
        view.setTo(Scalar(2));
        accel::detail::emulatorFailNextCpuFlush();  // the CPU's writes may reach the memory only in part
    }
    EXPECT_THROW(accel::exportExternalMemory(y), cv::Exception);
    accel::ExternalMemory e = accel::exportExternalMemory(x);
    close((int)e.handle);
#endif
}

TEST(Core_Accel, protected_memory_is_shared_between_devices_as_protected_memory)
{
#ifdef __linux__
    // a frame of a secure heap, wrapped on the GPU, processed on the NPU: imported there as
    // protected memory (nothing copied, which protection would forbid), the result protected too
    accel::Device gpu = emulator(COHERENT), npu = emulator(NONCOHERENT);
    accel::ExternalMemory mem;
    mem.type = accel::ExternalMemory::DMABUF_FD;
    mem.handle = accel::detail::emulatorCreateDmaBuf(256, true);
    mem.size = 256;
    mem.protectedContent = true;
    UMat secure = accel::wrapExternalMemory(gpu, mem, MatShape({16, 16}), CV_8UC1);
    accel::ExecutionScope scope(npu);
    UMat out;
    accel::resetTransferStats();
    cv::flip(secure, out, 1);
    EXPECT_EQ(1u, accel::getTransferStats().deviceOps);
    EXPECT_EQ(2, accel::detail::backingImports(secure));
    EXPECT_TRUE(npu == accel::describe(out).device);
    EXPECT_TRUE(accel::describe(out).protectedContent);
#endif
}

TEST(Core_Accel, asking_a_device_imports_nothing)
{
#ifdef __linux__
    // the NPU is asked whether it runs the call on memory it would import (QUERY), and declines:
    // nothing was imported for the question, and the call runs on the CPU
    UMat x = makeOn(emulator(COHERENT), Mat(8, 8, CV_8UC1, Scalar(3)));
    accel::detail::emulatorDisableOps(NONCOHERENT, true);
    {
        accel::ExecutionScope scope(emulator(NONCOHERENT));
        UMat out;
        accel::resetTransferStats();
        cv::flip(x, out, 1);
        EXPECT_EQ(1u, accel::getTransferStats().cpuFallbacks);
        EXPECT_EQ(1, accel::detail::backingImports(x));
    }
    accel::detail::emulatorDisableOps(NONCOHERENT, false);
#endif
}

TEST(Core_Accel, memory_given_out_by_handle_is_never_shared_as_a_dma_buf)
{
#ifdef __linux__
    // code that got UMat::handle() orders its work with the device's default stream only: the
    // memory must never take implicit fences afterwards, so it is neither exported nor shared
    Mat src(16, 24, CV_8UC1), ref;
    randu(src, 0, 255);
    cv::flip(src, ref, 0);
    UMat x = makeOn(emulator(COHERENT), src);
    EXPECT_TRUE(x.handle(ACCESS_READ) != NULL);
    EXPECT_THROW(accel::exportExternalMemory(x), cv::Exception);
    accel::ExecutionScope scope(emulator(NONCOHERENT));
    UMat out;
    cv::flip(x, out, 0);            // copied to the NPU instead of imported there
    EXPECT_EQ(1, accel::detail::backingImports(x));
    EXPECT_EQ(0, maxDiff(ref, out.getMat(ACCESS_READ)));
#endif
}

TEST(Core_Accel, no_device_runs_a_plan_that_moves_data_the_cpu_cannot_carry)
{
#ifdef __linux__
    // A tiled frame of the coherent device as the output of a call whose input lives on the
    // non-coherent one. That device could run it only by copying a linear result into the
    // tiled memory, and the coherent device's kernels are linear: nobody runs it, and the
    // frame is left alone.
    accel::ExternalMemory mem;
    mem.type = accel::ExternalMemory::DMABUF_FD;
    mem.handle = accel::detail::emulatorCreateDmaBuf(256);
    mem.size = 256;
    mem.layoutModifier = 0x0100000000000002ull;    // a vendor tiling
    uchar* frame = (uchar*)mmap(NULL, 256, PROT_READ | PROT_WRITE, MAP_SHARED, (int)mem.handle, 0);
    ASSERT_TRUE(frame != MAP_FAILED);
    memset(frame, 5, 256);
    {
        UMat tiled = accel::wrapExternalMemory(emulator(COHERENT), mem, MatShape({16, 16}), CV_8UC1);
        Mat src(16, 16, CV_8UC1);
        randu(src, 0, 255);
        UMat in = makeOn(emulator(NONCOHERENT), src);
        EXPECT_THROW(cv::flip(in, tiled, 1), cv::Exception);
    }
    EXPECT_EQ(0, cvtest::norm(Mat(1, 256, CV_8UC1, frame), Mat(1, 256, CV_8UC1, Scalar(5)), NORM_INF));
    munmap(frame, 256);
#endif
}

TEST(Core_Accel, an_output_placed_by_its_allocator_decides_where_the_call_runs)
{
    // the output's own allocator puts it on a device: a call with host inputs runs there, and
    // one whose inputs are elsewhere still leaves its result where the allocator says
    accel::Device d = emulator(DISCRETE);
    Mat src(8, 8, CV_8UC1), ref;
    randu(src, 0, 255);
    cv::flip(src, ref, 1);
    accel::resetTransferStats();
    UMat out;
    out.allocator = d.allocator();
    cv::flip(src, out, 1);
    EXPECT_EQ(1u, accel::getTransferStats().deviceOps);
    EXPECT_TRUE(d == accel::describe(out).device);
    EXPECT_EQ(0, maxDiff(ref, out.getMat(ACCESS_READ)));

    UMat in = makeOn(emulator(COHERENT), src), placed;
    placed.allocator = d.allocator();
    cv::flip(in, placed, 1);
    EXPECT_TRUE(d == accel::describe(placed).device);
    EXPECT_EQ(0, maxDiff(ref, placed.getMat(ACCESS_READ)));
}

TEST(Core_Accel, device_access_checks_the_stream)
{
    UMat u = makeOn(emulator(DISCRETE), Mat(4, 4, CV_8UC1, Scalar(1)));
    accel::Stream other = accel::Stream::getDefault(emulator(COHERENT));
    EXPECT_THROW({ accel::DeviceAccess acc(u, other, ACCESS_READ); }, cv::Exception);  // a device that cannot reach it
    EXPECT_NO_THROW({ accel::DeviceAccess acc(u, accel::Stream::getDefault(emulator(DISCRETE)), ACCESS_READ); });
}

TEST(Core_Accel, aliases_of_one_memory_share_one_buffer)
{
    accel::Device d = emulator(DISCRETE);
    UMat owner = makeOn(d, Mat(16, 16, CV_8UC1, Scalar(1)));
    accel::ExternalMemory mem;
    mem.type = accel::ExternalMemory::DEVICE_POINTER;
    mem.handle = (int64)(intptr_t)owner.handle(ACCESS_READ);
    mem.size = 256;
    UMat w1 = accel::wrapExternalMemory(d, mem, MatShape({16, 16}), CV_8UC1);
    UMat w2 = accel::wrapExternalMemory(d, mem, MatShape({16, 16}), CV_8UC1);
    EXPECT_EQ(owner.u, w1.u);       // memory OpenCV already holds: views of that buffer
    EXPECT_EQ(owner.u, w2.u);
    mem.offset = 128;
    mem.size = 128;
    UMat half = accel::wrapExternalMemory(d, mem, MatShape({8, 16}), CV_8UC1);
    EXPECT_EQ(owner.u, half.u);
    EXPECT_EQ(128u, half.offset);
    mem.offset = 200;               // runs past the end of that buffer: refused
    mem.size = 100;
    EXPECT_THROW(accel::wrapExternalMemory(d, mem, MatShape({1, 100}), CV_8UC1), cv::Exception);
    mem.offset = 0;
    mem.size = 256;
    mem.readOnly = true;            // a writable buffer cannot also be read-only
    EXPECT_THROW(accel::wrapExternalMemory(d, mem, MatShape({16, 16}), CV_8UC1), cv::Exception);
}

TEST(Core_Accel, results_of_functions_without_a_hook_stay_on_the_device)
{
    // shape-changing functions create their outputs with UMat::create, not after an input's shape
    accel::Device d = emulator(DISCRETE);
    UMat src = makeOn(d, Mat(4, 6, CV_8UC1, Scalar(3)));
    UMat rep, red;
    cv::repeat(src, 2, 3, rep);
    EXPECT_TRUE(d == accel::describe(rep).device);
    EXPECT_EQ(Size(18, 8), rep.size());
    EXPECT_EQ(3, rep.getMat(ACCESS_READ).at<uchar>(7, 17));
    cv::reduce(src, red, 0, REDUCE_SUM, CV_32S);
    EXPECT_TRUE(d == accel::describe(red).device);
    EXPECT_EQ(12, red.getMat(ACCESS_READ).at<int>(0, 5));

    // an output on the device stays there when the function gives it another shape
    UMat out = makeOn(d, Mat(2, 2, CV_8UC1, Scalar(0)));
    cv::repeat(Mat(4, 6, CV_8UC1, Scalar(5)), 1, 2, out);
    EXPECT_TRUE(d == accel::describe(out).device);
    EXPECT_EQ(5, out.getMat(ACCESS_READ).at<uchar>(3, 11));

    // without device data, host memory as before
    UMat plain;
    cv::repeat(Mat(2, 2, CV_8UC1, Scalar(1)), 2, 2, plain);
    EXPECT_TRUE(accel::describe(plain).device.empty());
}

//! an application function that opens a trace region like a cv:: function, and calls cv::repeat with an argument of its own
static void outerCall(InputArray src, OutputArray dst, const UMat& innerSrc, UMat& innerDst)
{
    CV_TRACE_FUNCTION();
    cv::repeat(innerSrc, 1, 2, innerDst);
    src.copyTo(dst);
}

//! one that only forwards its arguments
static void forwardingCall(InputArray src, OutputArray dst)
{
    CV_TRACE_FUNCTION();
    cv::repeat(src, 1, 2, dst);
}

TEST(Core_Accel, placement_by_call_looks_at_the_innermost_call)
{
    // the arguments of an outer call say nothing about where an inner call with arguments of
    // its own puts its results
    accel::Device a = emulator(DISCRETE), b = emulator(COHERENT);
    UMat src = makeOn(a, Mat(4, 6, CV_8UC1, Scalar(3))), dst;
    UMat host, hostResult, deviceResult;
    Mat(2, 2, CV_8UC1, Scalar(1)).copyTo(host);
    ASSERT_TRUE(accel::describe(host).device.empty());
    outerCall(src, dst, host, hostResult);          // inner call: host data
    EXPECT_TRUE(accel::describe(hostResult).device.empty());
    UMat onB = makeOn(b, Mat(2, 2, CV_8UC1, Scalar(2)));
    outerCall(src, dst, onB, deviceResult);         // inner call: data on another device
    EXPECT_TRUE(b == accel::describe(deviceResult).device);
    EXPECT_EQ(2, deviceResult.getMat(ACCESS_READ).at<uchar>(1, 3));
    UMat forwarded;
    forwardingCall(src, forwarded);                 // forwarded arguments are the inner call's too
    EXPECT_TRUE(a == accel::describe(forwarded).device);

    // the elements of a vector of UMats are arguments too
    std::vector<UMat> parts;
    parts.push_back(makeOn(a, Mat(4, 6, CV_8UC1, Scalar(1))));
    parts.push_back(makeOn(a, Mat(4, 6, CV_8UC1, Scalar(2))));
    UMat merged;
    cv::merge(parts, merged);
    EXPECT_TRUE(a == accel::describe(merged).device);
    EXPECT_EQ(Vec2b(1, 2), merged.getMat(ACCESS_READ).at<Vec2b>(3, 5));
}

TEST(Core_Accel, a_host_scope_keeps_data_and_work_on_the_host)
{
    // the host scope is the CPU: new UMat data is host memory, not OpenCL's, and no OpenCL kernel runs
    accel::detail::registerEmulatorBackend();
    accel::ExecutionScope scope{accel::Device()};
    EXPECT_FALSE(cv::ocl::useOpenCL());
    UMat a(8, 8, CV_8UC1, Scalar(1)), out;
    EXPECT_EQ(Mat::getDefaultAllocator(), a.u->currAllocator);
    cv::repeat(a, 2, 2, out);
    EXPECT_EQ(Mat::getDefaultAllocator(), out.u->currAllocator);
    EXPECT_EQ(1, out.getMat(ACCESS_READ).at<uchar>(15, 15));
}

TEST(Core_Accel, a_device_scope_decides_even_when_its_device_declines)
{
    accel::Device d = emulator(COHERENT);
    Mat src(8, 8, CV_8UC1), ref;
    randu(src, 0, 255);
    cv::flip(src, ref, 0);
    accel::detail::emulatorDisableOps(COHERENT, true);
    {
        accel::ExecutionScope scope(d);
        accel::resetTransferStats();
        UMat out;
        cv::flip(src, out, 0);      // only host arguments, and the device declines
        EXPECT_EQ(1u, accel::getTransferStats().cpuFallbacks);     // a fallback, counted
        EXPECT_TRUE(d == accel::describe(out).device);              // the result on the scope's device
        EXPECT_EQ(0, maxDiff(ref, out.getMat(ACCESS_READ)));
    }
    accel::detail::emulatorDisableOps(COHERENT, false);
}

TEST(Core_Accel, device_choice_counts_copies_back_to_another_device_twice)
{
    // 64 bytes on the discrete device, 40 host bytes, a 64-byte output on the coherent device.
    // On the discrete device the result goes to the output through the CPU: downloaded from the
    // temporary, then uploaded (128 bytes, plus 40 uploaded). The coherent device gets the
    // discrete device's bytes (downloaded once, uploaded once: 128) and reads the host bytes in
    // place. The coherent device moves less.
    UMat d = makeOn(emulator(DISCRETE), Mat(8, 8, CV_8UC1, Scalar(1)));
    Mat h(5, 8, CV_8UC1, Scalar(3));
    UMat out = makeOn(emulator(COHERENT), Mat::zeros(8, 8, CV_8UC1));
    int ran = -1;
    accel::detail::OpCall call("test.cost", CV_ACCEL_OPS_CORE, CV_ACCEL_OPS_CORE_VERSION);
    call.input(d);
    call.input(h);
    call.output(out, d.shape(), CV_8UC1);
    EXPECT_EQ(accel::detail::HANDLED, call.run([&](const void*, const cvAccelCall* c, const cvAccelView*) {
        if (c->mode == CV_ACCEL_EXECUTE)
            ran = c->device;
        return CV_ACCEL_OK;
    }));
    EXPECT_EQ(COHERENT, ran);
}

TEST(Core_Accel, device_choice_counts_host_copies_that_must_move)
{
    // the large input sits on the discrete device, but the CPU wrote it last: using it there in
    // place costs an upload of all of it, more than copying the small input to the other device
    UMat large = makeOn(emulator(DISCRETE), Mat(64, 64, CV_8UC1, Scalar(1)));
    large.getMat(ACCESS_WRITE).setTo(Scalar(2));                   // the host copy is newer now
    UMat small = makeOn(emulator(COHERENT), Mat(32, 32, CV_8UC1, Scalar(5)));
    UMat out;
    accel::detail::OpCall call("test.two_inputs", CV_ACCEL_OPS_CORE, CV_ACCEL_OPS_CORE_VERSION);
    call.input(large);
    call.input(small);
    call.output(out, large.shape(), large.type());
    EXPECT_EQ(accel::detail::HANDLED, call.run([](const void* t, const cvAccelCall* c, const cvAccelView* v) {
        return ((const cvAccelOpsCore_v0*)t)->flip(c, &v[0], &v[2], 1);    // the small input is only read
    }));
    EXPECT_TRUE(emulator(COHERENT) == accel::describe(out).device);   // 4096 bytes moved instead of 5120
    EXPECT_EQ(2, out.getMat(ACCESS_READ).at<uchar>(0, 0));
}

TEST(Core_Accel, device_access_gives_the_address_kernels_use)
{
    // host memory a device addresses at another address than the CPU (no unified addressing)
    accel::Device d = emulator(COHERENT);
    std::vector<uchar> storage(4096 * 2);
    uchar* p = alignPtr(storage.data(), 4096);
    accel::ExternalMemory mem;
    mem.type = accel::ExternalMemory::HOST_POINTER;
    mem.handle = (int64)(intptr_t)p;
    mem.size = 1024;
    UMat w = accel::wrapExternalMemory(d, mem, MatShape({32, 32}), CV_8UC1);
    UMat part = w(Rect(0, 16, 32, 16));
    accel::DeviceAccess acc(part, accel::Stream::getDefault(d), ACCESS_READ);
    EXPECT_NE((uint64)(uintptr_t)p, acc.handle());                  // not the CPU address
    EXPECT_EQ((uint64)(uintptr_t)p | (1ull << 62), acc.handle());   // the emulator's device address of it
    EXPECT_EQ(16u * 32u, acc.offset());
}

TEST(Core_Accel, a_stream_shared_by_threads_is_used_by_one_at_a_time)
{
    // emulator:2's streams are not thread-safe: two threads in ExecutionScopes of one explicit
    // stream must never be inside the backend with it at the same time
    accel::Device d = emulator(NONCOHERENT);
    accel::Stream shared(d);
    const int64 races = accel::detail::emulatorViolations(accel::detail::EMU_VIOLATION_STREAM);
    auto work = [&](int value) {
        accel::ExecutionScope scope(shared);
        UMat u = makeOn(d, Mat(8, 8, CV_8UC1, Scalar(value))), out;
        for (int i = 0; i < 100; i++)
            cv::flip(u, out, i & 1);
        EXPECT_EQ(value, out.getMat(ACCESS_READ).at<uchar>(7, 7));
    };
    std::thread t1(work, 1), t2(work, 2);
    t1.join();
    t2.join();
    EXPECT_EQ(races, accel::detail::emulatorViolations(accel::detail::EMU_VIOLATION_STREAM));
}

TEST(Core_Accel, fences_of_another_device_are_waited_for_where_the_device_cannot)
{
    // emulator:6 reaches emulator:0's memory (peer access) but cannot wait for its fences on
    // the device: the wait for the queued upload into the output happens on the CPU instead
    const int64 foreign = accel::detail::emulatorViolations(accel::detail::EMU_VIOLATION_FENCE);
    Mat src(8, 8, CV_8UC1), ref;
    randu(src, 0, 255);
    cv::flip(src, ref, 1);
    runWithDeviceDelay(20000, [&]() {
        UMat in = makeOn(emulator(PEER), src);
        UMat out = makeOn(emulator(DISCRETE), Mat::zeros(8, 8, CV_8UC1));     // still being uploaded
        accel::resetTransferStats();
        cv::flip(in, out, 1);           // on emulator:6, writing emulator:0's memory in place
        EXPECT_EQ(1u, accel::getTransferStats().deviceOps);
        EXPECT_EQ(0, maxDiff(ref, out.getMat(ACCESS_READ)));
    });
    EXPECT_EQ(foreign, accel::detail::emulatorViolations(accel::detail::EMU_VIOLATION_FENCE));
}

TEST(Core_Accel, a_peer_device_gets_its_own_address_of_the_memory)
{
    // emulator:6 reaches emulator:0's memory through a window of its own, as without unified
    // addressing: its kernels, and DeviceAccess on its streams, get that address
    const int64 wrong = accel::detail::emulatorViolations(accel::detail::EMU_VIOLATION_ADDRESS);
    Mat src(8, 8, CV_8UC1), ref;
    randu(src, 0, 255);
    cv::flip(src, ref, 1);
    UMat in = makeOn(emulator(PEER), src);
    UMat out = makeOn(emulator(DISCRETE), Mat::zeros(8, 8, CV_8UC1));
    accel::resetTransferStats();
    cv::flip(in, out, 1);           // on emulator:6, writing emulator:0's memory in place
    EXPECT_EQ(1u, accel::getTransferStats().deviceOps);
    EXPECT_EQ(0, maxDiff(ref, out.getMat(ACCESS_READ)));
    EXPECT_EQ(wrong, accel::detail::emulatorViolations(accel::detail::EMU_VIOLATION_ADDRESS));

    const uint64 own = accel::DeviceAccess(out, accel::Stream::getDefault(emulator(DISCRETE)), ACCESS_READ).handle();
    const uint64 peer = accel::DeviceAccess(out, accel::Stream::getDefault(emulator(PEER)), ACCESS_READ).handle();
    EXPECT_NE(own, peer);
    EXPECT_EQ(own + (7ull << 52), peer);
}

TEST(Core_Accel, structs_too_short_for_their_fields_are_refused)
{
    const OpenCV_Core_Accel_API* api = accel::detail::emulatorBackendTable();
    struct
    {
        cvAccelDeviceInfo info;
        uchar canary[16];
    } buf;
    memset(&buf, 0xAB, sizeof(buf));
    buf.info.struct_size = 2;   // not even room for the struct_size field
    EXPECT_EQ(CV_ACCEL_INVALID_ARGUMENT, api->v0.get_device_info(0, &buf.info));
    for (size_t i = sizeof(uint32_t); i < sizeof(buf); i++)
        ASSERT_EQ(0xAB, ((const uchar*)&buf)[i]) << "byte " << i << " was written";
    cvAccelExternalMemory ext;
    memset(&ext, 0, sizeof(ext));
    ext.struct_size = 8;        // ends before the fields of version 0
    cvAccelAllocation out;
    memset(&out, 0, sizeof(out));
    out.struct_size = (uint32_t)sizeof(out);
    EXPECT_EQ(CV_ACCEL_INVALID_ARGUMENT, api->v0.import_memory(1, &ext, &out));

    // an out-struct is filled only as far as its provider sized it: one shorter than version 0 is refused
    struct
    {
        cvAccelAllocation alloc;
        uchar canary[16];
    } small;
    memset(&small, 0xAB, sizeof(small));
    small.alloc.struct_size = 8;
    EXPECT_EQ(CV_ACCEL_INVALID_ARGUMENT, api->v0.allocate(1, 64, CV_ACCEL_MEM_SHARED, 0, &small.alloc));
    for (size_t i = sizeof(uint32_t); i < sizeof(small); i++)
        ASSERT_EQ(0xAB, ((const uchar*)&small)[i]) << "byte " << i << " was written";

    // a table that claims a newer API version than core knows follows rules core does not: refused
    static OpenCV_Core_Accel_API newer = *accel::detail::emulatorBackendTable();
    newer.api_header.api_version = CV_ACCEL_API_VERSION + 1;
    newer.v0.backend_name = "emulator-from-the-future";
    EXPECT_EQ(0, accel::detail::registerBackendTable(&newer));

    // a table of ABI version 0 had other layouts of allocations and views: refused, never read
    static OpenCV_Core_Accel_API older = *accel::detail::emulatorBackendTable();
    older.api_header.min_api_version = 0;
    older.v0.backend_name = "emulator-of-abi-0";
    EXPECT_EQ(0, accel::detail::registerBackendTable(&older));
}

#ifdef __linux__
TEST(Core_Accel, wrapped_dma_buf_outlives_the_descriptor)
{
    // the descriptor is borrowed: the caller may close it once the wrap returned
    accel::Device d = emulator(COHERENT);
    const int64 fd = accel::detail::emulatorCreateDmaBuf(256);
    accel::detail::emulatorDmaBufProduce(fd, 8, 0);
    accel::ExternalMemory mem;
    mem.type = accel::ExternalMemory::DMABUF_FD;
    mem.handle = dup((int)fd);
    mem.size = 256;
    UMat frame = accel::wrapExternalMemory(d, mem, MatShape({16, 16}), CV_8UC1);
    close((int)mem.handle);
    UMat out;
    cv::flip(frame, out, 0);
    EXPECT_EQ(8, out.getMat(ACCESS_READ).at<uchar>(15, 15));
    mem.handle = fd;            // another descriptor of the same buffer: the same memory
    UMat again = accel::wrapExternalMemory(d, mem, MatShape({16, 16}), CV_8UC1);
    EXPECT_EQ(accel::detail::backingId(frame), accel::detail::backingId(again));
}
#endif

TEST(Core_Accel, functions_without_a_hook_still_work)
{
    // cv::add has no accelerator hook: OpenCL (if active) must decline the device memory
    // and the CPU path must reach it through the protocol
    accel::Device d = emulator(DISCRETE);
    Mat a(16, 16, CV_8UC1, Scalar(20)), b(16, 16, CV_8UC1, Scalar(22));
    UMat ua = makeOn(d, a), ub = makeOn(d, b), uc;
    cv::add(ua, ub, uc);
    EXPECT_TRUE(d == accel::describe(uc).device);  // the result stays with its inputs
    EXPECT_EQ(0, maxDiff(Mat(16, 16, CV_8UC1, Scalar(42)), uc.getMat(ACCESS_READ)));
}

TEST(Core_Accel, device_idle_memory_holds_back_device_work)
{
    accel::Device d = emulator(MANAGED);
    Mat src(32, 32, CV_8UC1);
    randu(src, 0, 255);
    UMat a = makeOn(d, src), b = makeOn(d, src);
    const int64 violations = accel::detail::emulatorViolations();

    accel::resetTransferStats();
    {
        Mat view = a.getMat(ACCESS_READ);   // the CPU is inside managed memory
        UMat out;
        cv::flip(b, out, 1);                // starting device work now would fault: CPU instead
        EXPECT_EQ(1u, accel::getTransferStats().cpuFallbacks);
    }
    UMat out2;
    cv::flip(b, out2, 1);                   // the view is gone: runs on the device
    EXPECT_EQ(1u, accel::getTransferStats().deviceOps);
    Mat ref;
    cv::flip(src, ref, 1);
    EXPECT_EQ(0, maxDiff(ref, out2.getMat(ACCESS_READ)));
    EXPECT_EQ(violations, accel::detail::emulatorViolations());
}

#ifndef _WIN32
TEST(Core_Accel, mat_view_chain_ends_its_mapping)
{
    // the CPU may touch this memory only while the device is idle: an open mapping keeps
    // device work away, so it has to end when the last view goes
    accel::Device d = emulator(MANAGED);
    Mat src(8, 8, CV_8UC1);
    randu(src, 0, 255);
    UMat u = makeOn(d, src);
    {
        UMat t = u.getMat(ACCESS_READ).getUMat(ACCESS_READ);    // the Mat in between is gone already
    }   // the last view of the mapping goes with t
    accel::resetTransferStats();
    UMat out;
    cv::flip(u, out, 1);
    EXPECT_EQ(1u, accel::getTransferStats().deviceOps);
}

TEST(Core_Accel, device_access_keeps_the_cpu_out_of_device_idle_memory)
{
    accel::Device d = emulator(MANAGED);
    UMat a = makeOn(d, Mat(8, 8, CV_8UC1, Scalar(1)));
    UMat b = makeOn(d, Mat(8, 8, CV_8UC1, Scalar(2)));
    accel::Stream s = accel::Stream::getDefault(d);
    {
        Mat view = a.getMat(ACCESS_READ);  // the CPU is inside device-idle memory...
        EXPECT_THROW({ accel::DeviceAccess acc(b, s, ACCESS_RW); }, cv::Exception);   // ...so no device work
    }
    {
        accel::DeviceAccess acc(b, s, ACCESS_RW);  // the caller's kernels may run at any time now
        EXPECT_THROW(a.getMat(ACCESS_READ), cv::Exception);   // an error, not a wait for itself
    }
    EXPECT_EQ(1, a.getMat(ACCESS_READ).at<uchar>(0, 0));
}

TEST(Core_Accel, cpu_waits_for_queued_device_work)
{
    accel::Device d = emulator(DISCRETE);
    runWithDeviceDelay(20000, [&]() {
        Mat src(32, 32, CV_8UC1);
        randu(src, 0, 255);
        UMat u = makeOn(d, src);
        UMat a, b;
        cv::flip(u, a, 0);
        cv::flip(a, b, 1);                  // both still queued (each task waits 20 ms)
        Mat ref;
        cv::flip(src, ref, -1);
        EXPECT_EQ(0, maxDiff(ref, b.getMat(ACCESS_READ)));
    });
}

TEST(Core_Accel, release_while_in_flight_is_deferred)
{
    accel::Device d = emulator(DISCRETE);
    runWithDeviceDelay(20000, [&]() {
        const int64 before = accel::detail::emulatorLiveAllocations();
        Mat src(32, 32, CV_8UC1, Scalar(1));
        {
            UMat u = makeOn(d, src);
            UMat out;
            cv::flip(u, out, 1);
            // both released while the flip is still queued
        }
        EXPECT_GT(accel::detail::emulatorLiveAllocations(), before);   // not freed under the device

        accel::Stream::getDefault(d).waitForCompletion();
        UMat trigger;                       // any later release collects finished buffers
        trigger.allocator = d.allocator();
        trigger.create(1, 1, CV_8UC1);
        trigger.release();
        EXPECT_EQ(before, accel::detail::emulatorLiveAllocations());
    });
}

TEST(Core_Accel, device_sync_waits_for_streams_of_finished_threads)
{
    // managed memory: the CPU may touch it only while the device is idle, which includes work
    // queued by a thread that has exited since (its stream is gone, its work is not)
    accel::Device d = emulator(MANAGED);
    UMat x = makeOn(d, Mat(64, 64, CV_8UC1, Scalar(1)));
    UMat y = makeOn(d, Mat(8, 8, CV_8UC1, Scalar(2)));
    UMat out;
    const int64 violations = accel::detail::emulatorViolations();
    runWithDeviceDelay(20000, [&]() { cv::flip(x, out, 0); });     // still queued when the thread ends
    EXPECT_EQ(2, y.getMat(ACCESS_READ).at<uchar>(0, 0));           // another buffer: the device drains first
    EXPECT_EQ(violations, accel::detail::emulatorViolations());
    EXPECT_EQ(1, out.getMat(ACCESS_READ).at<uchar>(0, 0));
}

TEST(Core_Accel, wait_limit_never_hands_back_memory_the_device_still_uses)
{
    accel::Device d = emulator(COHERENT);
    accel::detail::setWaitTimeout(5);  // ms; device work below takes 50 ms
    runWithDeviceDelay(50000, [&]() {
        std::vector<uchar> storage(4096 * 2);
        uchar* p = alignPtr(storage.data(), 4096);
        Mat host(32, 32, CV_8UC1, p);
        host.setTo(Scalar(0));
        accel::ExternalMemory mem;
        mem.type = accel::ExternalMemory::HOST_POINTER;
        mem.handle = (int64)(intptr_t)p;
        mem.size = host.total();
        UMat w = accel::wrapExternalMemory(d, mem, MatShape({32, 32}), CV_8UC1);
        UMat src = makeOn(d, Mat(32, 32, CV_8UC1, Scalar(7)));
        src.copyTo(w);                  // a device copy into the owner's memory: no limit on that wait
        w.release();
        EXPECT_EQ(0, maxDiff(Mat(32, 32, CV_8UC1, Scalar(7)), host));
    });
    accel::detail::setWaitTimeout(0);
    accel::detail::reviveDevices();
}

TEST(Core_Accel, aliases_are_ordered_through_their_shared_state)
{
    accel::Device d = emulator(DISCRETE);
    runWithDeviceDelay(20000, [&]() {
        UMat owner = makeOn(d, Mat(16, 16, CV_8UC1, Scalar(1)));
        accel::ExternalMemory mem;
        mem.type = accel::ExternalMemory::DEVICE_POINTER;
        mem.handle = (int64)(intptr_t)owner.handle(ACCESS_READ);
        mem.size = 256;
        UMat w1 = accel::wrapExternalMemory(d, mem, MatShape({16, 16}), CV_8UC1);
        UMat w2 = accel::wrapExternalMemory(d, mem, MatShape({16, 16}), CV_8UC1);
        UMat src = makeOn(d, Mat(16, 16, CV_8UC1, Scalar(9)));
        src.copyTo(w1);             // a device write through one wrapper, still queued...
        EXPECT_EQ(9, w2.getMat(ACCESS_READ).at<uchar>(15, 15));    // ...seen by a CPU read through the other
    });
}

TEST(Core_Accel, copy_into_wrapped_memory_is_done_on_return)
{
    accel::Device d = emulator(COHERENT);
    runWithDeviceDelay(20000, [&]() {
        std::vector<uchar> storage(4096 * 2);
        uchar* p = alignPtr(storage.data(), 4096);
        Mat host(32, 32, CV_8UC1, p);
        host.setTo(Scalar(0));
        accel::ExternalMemory mem;
        mem.type = accel::ExternalMemory::HOST_POINTER;
        mem.handle = (int64)(intptr_t)p;
        mem.size = host.total();
        UMat w = accel::wrapExternalMemory(d, mem, MatShape({32, 32}), CV_8UC1);
        UMat src = makeOn(d, Mat(32, 32, CV_8UC1, Scalar(7)));
        src.copyTo(w);                      // a device copy into the owner's memory
        EXPECT_EQ(0, maxDiff(Mat(32, 32, CV_8UC1, Scalar(7)), host));   // read directly, not through w
    });
}
#endif

TEST(Core_Accel, device_failures_never_escape_a_destructor)
{
    accel::Device d = failingDevice();
    const int64 base = accel::detail::emulatorLiveAllocations(FAILING);    // memory other tests leaked on purpose
    Mat src(16, 16, CV_8UC1), other(16, 16, CV_8UC1);
    randu(src, 0, 128);
    randu(other, 128, 255);

    // 1. work that fails asynchronously: readers get the error, releases free everything
    {
        UMat u = makeOn(d, src);
        accel::detail::emulatorInjectFailure(CV_ACCEL_INTERNAL_ERROR);
        UMat out;
        cv::flip(u, out, 1);
        EXPECT_THROW(out.getMat(ACCESS_READ), cv::Exception);
        accel::detail::emulatorInjectFailure(CV_ACCEL_OK);
    }
    EXPECT_EQ(base, accel::detail::emulatorLiveAllocations(FAILING));

    // 2. a UMat made from a Mat: a failed result is not copied back, the Mat keeps its content
    {
        accel::ExecutionScope scope(d);
        Mat m = src.clone();
        {
            UMat u = m.getUMat(ACCESS_RW);
            UMat dev = makeOn(d, other);
            accel::detail::emulatorInjectFailure(CV_ACCEL_INTERNAL_ERROR);
            dev.copyTo(u);
            accel::detail::emulatorInjectFailure(CV_ACCEL_OK);
        }
        EXPECT_EQ(0, maxDiff(src, m));
    }
    EXPECT_EQ(base, accel::detail::emulatorLiveAllocations(FAILING));

    // 3. calls that fail after queuing their work: what they write is invalid until written whole
    {
        UMat u = makeOn(d, src), out, c = makeOn(d, other);
        accel::detail::emulatorInjectFailure(CV_ACCEL_INTERNAL_ERROR, accel::detail::EMU_FAIL_CALLS);
        EXPECT_THROW(cv::flip(u, out, 1), cv::Exception);
        EXPECT_THROW(u.copyTo(c), cv::Exception);
        accel::detail::emulatorInjectFailure(CV_ACCEL_OK);
        EXPECT_THROW(out.getMat(ACCESS_READ), cv::Exception);
        EXPECT_THROW(c.getMat(ACCESS_READ), cv::Exception);

        cv::flip(u, out, 1);                // written whole: usable again
        Mat ref, res;
        cv::flip(src, ref, 1);
        out.copyTo(res);                    // a download into res, no host copy kept
        EXPECT_EQ(0, maxDiff(ref, res));
    }
    EXPECT_EQ(base, accel::detail::emulatorLiveAllocations(FAILING));

    // 4. a lost device: its memory is leaked rather than freed, the other devices keep working
    {
        UMat u = makeOn(d, src);
        const int64 live = accel::detail::emulatorLiveAllocations(FAILING);
        accel::detail::emulatorInjectFailure(CV_ACCEL_DEVICE_LOST);
        EXPECT_THROW(u.getMat(ACCESS_READ), cv::Exception);
        u.release();
        EXPECT_EQ(live, accel::detail::emulatorLiveAllocations(FAILING));
    }
    UMat a = makeOn(emulator(DISCRETE), src), b;
    accel::resetTransferStats();
    cv::flip(a, b, 1);
    EXPECT_EQ(1u, accel::getTransferStats().deviceOps);
    accel::detail::emulatorInjectFailure(CV_ACCEL_OK);
    accel::detail::reviveDevices();
}

TEST(Core_Accel, borrowed_memory_waits_until_failed_work_provably_stopped)
{
    // A copy into memory of "another library" queues its work and fails, and afterwards the
    // device can neither record a fence nor be synchronized: nothing proves that the copy
    // stopped writing. Marking the device lost does not stop its DMA, so the memory must not
    // go back to its owner until the device can be synchronized again.
    accel::Device d = failingDevice();
    accel::ExternalMemory mem;
    mem.type = accel::ExternalMemory::DEVICE_POINTER;
    mem.handle = accel::detail::emulatorAllocateExternal(FAILING, 256, 1, false);
    mem.size = 256;
    UMat borrowed = accel::wrapExternalMemory(d, mem, MatShape({16, 16}), CV_8UC1);
    UMat src = makeOn(d, Mat(16, 16, CV_8UC1, Scalar(9)));
    accel::detail::emulatorInjectFailure(CV_ACCEL_INTERNAL_ERROR,
                                         accel::detail::EMU_FAIL_CALLS | accel::detail::EMU_FAIL_RECORD);
    EXPECT_THROW(src.copyTo(borrowed), cv::Exception);

    std::atomic<bool> released(false);
    std::thread owner([&]() { borrowed.release(); released = true; });
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    EXPECT_FALSE(released.load());      // device_synchronize cannot record its fences: no proof yet
    accel::detail::emulatorInjectFailure(CV_ACCEL_INTERNAL_ERROR, accel::detail::EMU_FAIL_SYNC);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    EXPECT_FALSE(released.load());      // nor when it fails outright
    accel::detail::emulatorInjectFailure(CV_ACCEL_OK);
    owner.join();                       // the device can be synchronized again: handed back
    EXPECT_TRUE(released.load());
    accel::detail::reviveDevices();
}

// "emulator-wide": a backend with 70 devices, device i behaving as emulator:(i % 7). Core
// stamps the device index on what the backend returns, so only entries that take a device change.
enum { WIDE_DEVICES = 70 };
static const OpenCV_Core_Accel_API_v0_0_api_entries& emu() { return accel::detail::emulatorBackendTable()->v0; }
static int wideBase(int device) { return device < 0 ? device : device % 7; }
static cvAccelStatus CV_API_CALL wide_get_device_count(int* n) CV_ACCEL_NOEXCEPT
{
    if (!n)
        return CV_ACCEL_INVALID_ARGUMENT;
    *n = WIDE_DEVICES;
    return CV_ACCEL_OK;
}
static cvAccelStatus CV_API_CALL wide_get_device_info(int device, cvAccelDeviceInfo* info) CV_ACCEL_NOEXCEPT
{
    return device < 0 || device >= WIDE_DEVICES ? CV_ACCEL_INVALID_ARGUMENT : emu().get_device_info(wideBase(device), info);
}
static cvAccelStatus CV_API_CALL wide_allocate(int device, uint64_t size, uint32_t kind, uint32_t flags,
                                               cvAccelAllocation* out) CV_ACCEL_NOEXCEPT
{
    return emu().allocate(wideBase(device), size, kind, flags, out);
}
static cvAccelStatus CV_API_CALL wide_import_memory(int device, const cvAccelExternalMemory* ext,
                                                    cvAccelAllocation* out) CV_ACCEL_NOEXCEPT
{
    return emu().import_memory(wideBase(device), ext, out);
}
static cvAccelStatus CV_API_CALL wide_can_access(int device, const cvAccelMemoryDesc* desc, int own) CV_ACCEL_NOEXCEPT
{
    return emu().can_access(wideBase(device), desc, wideBase(own));
}
static cvAccelStatus CV_API_CALL wide_copy(int device, cvAccelStream stream, const cvAccelCopyRegion* region) CV_ACCEL_NOEXCEPT
{
    return emu().copy(wideBase(device), stream, region);
}
static cvAccelStatus CV_API_CALL wide_stream_create(int device, uint32_t flags, cvAccelStream* out) CV_ACCEL_NOEXCEPT
{
    return emu().stream_create(wideBase(device), flags, out);
}
static cvAccelStatus CV_API_CALL wide_device_synchronize(int device) CV_ACCEL_NOEXCEPT
{
    return emu().device_synchronize(wideBase(device));
}
static cvAccelStatus CV_API_CALL wide_describe_allocation(int device, uint64_t size, uint32_t kind, uint32_t flags,
                                                          cvAccelMemoryDesc* desc) CV_ACCEL_NOEXCEPT
{
    return emu().describe_allocation(wideBase(device), size, kind, flags, desc);
}

static accel::Device wideDevice(int index)
{
    static OpenCV_Core_Accel_API table = []() {
        OpenCV_Core_Accel_API t = *accel::detail::emulatorBackendTable();
        t.v0.backend_name = "emulator-wide";
        t.v0.get_device_count = wide_get_device_count;
        t.v0.get_device_info = wide_get_device_info;
        t.v0.allocate = wide_allocate;
        t.v0.import_memory = wide_import_memory;
        t.v0.can_access = wide_can_access;
        t.v0.copy = wide_copy;
        t.v0.stream_create = wide_stream_create;
        t.v0.device_synchronize = wide_device_synchronize;
        t.v0.describe_allocation = wide_describe_allocation;
        return t;
    }();
    accel::detail::registerEmulatorBackend();
    CV_Assert(accel::detail::registerBackendTable(&table, -102) > 0);
    accel::Device d = accel::Device::find(cv::format("emulator-wide:%d", index));
    CV_Assert(!d.empty());
    return d;
}

TEST(Core_Accel, devices_share_no_state_however_many_a_backend_has)
{
    // Device 68 (failing) loses track of a copy into borrowed memory: nothing proves it
    // stopped. Device 67 (managed) is synchronized meanwhile, and device 64 keeps working:
    // neither says anything about device 68, past the 64th device as before it.
    accel::Device failing = wideDevice(68), managed = wideDevice(67), coherent = wideDevice(64);
    failingDevice();
    accel::ExternalMemory mem;
    mem.type = accel::ExternalMemory::DEVICE_POINTER;
    mem.handle = accel::detail::emulatorAllocateExternal(FAILING, 256, 1, false);
    mem.size = 256;
    UMat borrowed = accel::wrapExternalMemory(failing, mem, MatShape({16, 16}), CV_8UC1);
    UMat src = makeOn(failing, Mat(16, 16, CV_8UC1, Scalar(9)));
    UMat idle = makeOn(managed, Mat(16, 16, CV_8UC1, Scalar(5)));
    accel::detail::emulatorInjectFailure(CV_ACCEL_INTERNAL_ERROR,
                                         accel::detail::EMU_FAIL_CALLS | accel::detail::EMU_FAIL_RECORD);
    EXPECT_THROW(src.copyTo(borrowed), cv::Exception);

    std::atomic<bool> released(false);
    std::thread owner([&]() { borrowed.release(); released = true; });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_EQ(5, idle.getMat(ACCESS_READ).at<uchar>(0, 0));  // device 67 synchronized: proves nothing about 68
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_FALSE(released.load());

    Mat ref;
    Mat in(8, 8, CV_8UC1);
    randu(in, 0, 255);
    cv::flip(in, ref, 1);
    UMat uin = makeOn(coherent, in), out;
    accel::resetTransferStats();
    cv::flip(uin, out, 1);                      // device 64 is not lost with device 68
    EXPECT_EQ(1u, accel::getTransferStats().deviceOps);
    EXPECT_EQ(0, maxDiff(ref, out.getMat(ACCESS_READ)));

    accel::detail::emulatorInjectFailure(CV_ACCEL_OK);
    owner.join();                               // device 68 can be synchronized again: handed back
    EXPECT_TRUE(released.load());
    accel::detail::reviveDevices();
}

TEST(Core_Accel, memory_reaches_a_device_of_another_backend_as_a_dma_buf)
{
#ifdef __linux__
    // a result of emulator:1 used by a device of another backend that imports dma-bufs (it
    // behaves as emulator:2): imported there as the same memory, nothing copied
    Mat src(16, 24, CV_8UC1), ref;
    randu(src, 0, 255);
    cv::flip(src, ref, 0);
    UMat x = makeOn(emulator(COHERENT), src), out;
    accel::Device other = wideDevice(65);
    accel::ExecutionScope scope(other);
    accel::resetTransferStats();
    cv::flip(x, out, 0);
    accel::TransferStats s = accel::getTransferStats();
    EXPECT_EQ(1u, s.deviceOps);
    EXPECT_EQ(0u, s.transfers);
    EXPECT_TRUE(other == accel::describe(out).device);
    EXPECT_EQ(0, maxDiff(ref, out.getMat(ACCESS_READ)));
#endif
}

TEST(Core_Accel, memory_exported_by_one_backend_is_one_memory_in_another)
{
#ifdef __linux__
    // another library exports an emulator allocation as a dma-buf without telling OpenCV, and the
    // descriptor is wrapped through another backend: its owner recognises it, so it is one memory
    UMat y = makeOn(emulator(COHERENT), Mat(16, 24, CV_8UC1, Scalar(4)));
    accel::ExternalMemory f;
    f.type = accel::ExternalMemory::DMABUF_FD;
    f.handle = accel::detail::emulatorExportDmaBuf(y);
    f.size = y.total();
    UMat other = accel::wrapExternalMemory(wideDevice(65), f, MatShape({16, 24}), CV_8UC1);
    EXPECT_EQ(accel::detail::backingId(y), accel::detail::backingId(other));
    close((int)f.handle);
#endif
}

TEST(Core_Accel, a_later_write_does_not_hide_an_earlier_failure)
{
    // two writes on one stream: the first fails later, the second (to half of the memory) does
    // not. The stream orders them, so nobody waits for the first; its failure still counts.
    accel::Device d = failingDevice();
    runWithDeviceDelay(20000, [&]() {
        Mat src(16, 16, CV_8UC1);
        randu(src, 0, 255);
        UMat u = makeOn(d, src), half = makeOn(d, src.rowRange(0, 8));
        UMat out = makeOn(d, Mat::zeros(16, 16, CV_8UC1));
        accel::detail::emulatorInjectFailure(CV_ACCEL_INTERNAL_ERROR, accel::detail::EMU_FAIL_FENCES);
        cv::flip(u, out, 1);                        // fails 20 ms from now
        accel::detail::emulatorInjectFailure(CV_ACCEL_OK);
        UMat top = out.rowRange(0, 8);
        cv::flip(half, top, 1);                     // succeeds, after it on the same stream
        EXPECT_THROW(out.getMat(ACCESS_READ), cv::Exception);
    });
    accel::detail::reviveDevices();
}

TEST(Core_Accel, failed_work_stays_failed_however_its_fence_is_shared)
{
    // Everything that waits for some work shares the one fence reference recorded for it, and
    // never asks the backend for another (this device refuses them): the result of work that
    // fails later is invalid however many buffers keep its fence, and each reference is
    // released once.
    accel::Device d = failingDevice();
    const int64 fences = accel::detail::emulatorLiveFences();
    runWithDeviceDelay(20000, [&]() {
        Mat src(16, 16, CV_8UC1);
        randu(src, 0, 255);
        UMat u = makeOn(d, src), out;
        accel::detail::emulatorInjectFailure(CV_ACCEL_INTERNAL_ERROR,
                                             accel::detail::EMU_FAIL_RETAIN | accel::detail::EMU_FAIL_FENCES);
        cv::flip(u, out, 1);                                // fails 20 ms from now
        accel::detail::emulatorInjectFailure(CV_ACCEL_OK);
        EXPECT_THROW(out.getMat(ACCESS_READ), cv::Exception);
        EXPECT_THROW(out.getMat(ACCESS_READ), cv::Exception);  // and stays failed
        EXPECT_EQ(0, maxDiff(src, u.getMat(ACCESS_READ)));     // what it only read is intact
    });
    UMat trigger = makeOn(d, Mat(1, 1, CV_8UC1, Scalar(0)));   // releases collect finished buffers
    trigger.release();
    EXPECT_EQ(fences, accel::detail::emulatorLiveFences());    // every reference released once
    accel::detail::reviveDevices();
}

#ifdef HAVE_CUDA
// The CUDA backend on a real GPU: memory, streams and fences. It has no kernels yet, so calls
// run on the CPU, with their results on the GPU.

static accel::Device cudaDevice()
{
    accel::Device d = accel::Device::find("cuda:0");
    if (d.empty())
        throw SkipTestException("no CUDA device");
    return d;
}

TEST(Core_Accel_CUDA, host_round_trip_costs_one_copy_each_way)
{
    accel::Device d = cudaDevice();
    Mat src(480, 640, CV_8UC3);
    randu(src, 0, 255);
    const uint64 bytes = (uint64)(src.total() * src.elemSize());
    accel::resetTransferStats();
    UMat u = makeOn(d, src);
    EXPECT_EQ(accel::MEMORY_DEVICE, accel::describe(u).kind);
    EXPECT_EQ(0, maxDiff(src, u.getMat(ACCESS_READ)));
    EXPECT_EQ(0, maxDiff(src, u.getMat(ACCESS_READ)));     // the host copy is current: no second download
    accel::TransferStats s = accel::getTransferStats();
    EXPECT_EQ(bytes, s.bytesHostToDevice);                   // read-only CPU access never writes back
    EXPECT_EQ(bytes, s.bytesDeviceToHost);
}

TEST(Core_Accel_CUDA, device_copies_stay_on_the_device)
{
    accel::Device d = cudaDevice();
    Mat src(256, 256, CV_32FC1);
    randu(src, -1, 1);
    UMat u = makeOn(d, src);
    accel::resetTransferStats();
    UMat c = u.clone();
    accel::TransferStats s = accel::getTransferStats();
    EXPECT_TRUE(d == accel::describe(c).device);
    EXPECT_EQ((uint64)(src.total() * src.elemSize()), s.bytesDeviceToDevice);
    EXPECT_EQ(0u, s.bytesDeviceToHost + s.bytesHostToDevice);
    EXPECT_EQ(0, maxDiff(src, c.getMat(ACCESS_READ)));
}

TEST(Core_Accel_CUDA, calls_without_kernels_keep_their_results_on_the_gpu)
{
    accel::Device d = cudaDevice();
    Mat src(64, 48, CV_8UC1), ref;
    randu(src, 0, 255);
    cv::flip(src, ref, 0);
    UMat u = makeOn(d, src), out, sum;
    accel::resetTransferStats();
    cv::flip(u, out, 0);                    // a hook, but no CUDA kernel yet
    EXPECT_EQ(1u, accel::getTransferStats().cpuFallbacks);
    EXPECT_TRUE(d == accel::describe(out).device);
    EXPECT_EQ(0, maxDiff(ref, out.getMat(ACCESS_READ)));
    cv::add(u, u, sum);                     // no hook at all
    EXPECT_TRUE(d == accel::describe(sum).device);
}

TEST(Core_Accel_CUDA, wrapping_a_umat_pointer_gives_a_view_of_it)
{
    accel::Device d = cudaDevice();
    UMat owner = makeOn(d, Mat(32, 32, CV_8UC1, Scalar(4)));
    accel::ExternalMemory mem;
    mem.type = accel::ExternalMemory::DEVICE_POINTER;
    mem.handle = (int64)(intptr_t)owner.handle(ACCESS_READ);
    mem.offset = 32 * 8;
    mem.size = 32 * 8;
    UMat rows = accel::wrapExternalMemory(d, mem, MatShape({8, 32}), CV_8UC1);
    EXPECT_EQ(owner.u, rows.u);             // cuMemGetAddressRange: the same allocation, one state
    EXPECT_EQ(256u, rows.offset);
    EXPECT_EQ(4, rows.getMat(ACCESS_READ).at<uchar>(0, 0));
}

TEST(Core_Accel_CUDA, device_pointer_ranges_are_checked_against_their_allocation)
{
    accel::Device d = cudaDevice();
    UMat owner = makeOn(d, Mat(16, 16, CV_8UC1, Scalar(1)));
    accel::ExternalMemory mem;
    mem.type = accel::ExternalMemory::DEVICE_POINTER;
    mem.handle = (int64)(intptr_t)owner.handle(ACCESS_READ);
    mem.offset = 128;
    mem.size = 1 << 20;         // runs far past the end of the cudaMalloc it starts in
    EXPECT_THROW(accel::wrapExternalMemory(d, mem, MatShape({1024, 1024}), CV_8UC1), cv::Exception);
}

TEST(Core_Accel_CUDA, device_access_to_pinned_memory_gives_the_device_address)
{
    // mapped pinned memory: the kernel gets its device address, whatever its CPU address is
    accel::Device d = cudaDevice();
    UMat u;
    u.allocator = d.allocator();
    u.create(16, 16, CV_8UC1, USAGE_ALLOCATE_HOST_MEMORY);
    EXPECT_EQ(accel::MEMORY_HOST_PINNED, accel::describe(u).kind);
    {
        accel::DeviceAccess acc(u, accel::Stream::getDefault(d), ACCESS_WRITE);
        cuda::GpuMat g(u.rows, u.cols, u.type(), (void*)(uintptr_t)(acc.handle() + acc.offset()), u.step[0]);
        g.setTo(Scalar(7));     // a kernel on the legacy default stream, at that address
    }
    EXPECT_EQ(7, u.getMat(ACCESS_READ).at<uchar>(15, 15));
}

TEST(Core_Accel_CUDA, user_kernels_on_the_default_stream_are_ordered)
{
    accel::Device d = cudaDevice();
    UMat u = makeOn(d, Mat(64, 64, CV_8UC1, Scalar(0)));
    {
        accel::DeviceAccess acc(u, accel::Stream::getDefault(d), ACCESS_WRITE);
        cuda::GpuMat g(u.rows, u.cols, u.type(), (void*)(uintptr_t)(acc.handle() + acc.offset()), u.step[0]);
        g.setTo(Scalar(9));                 // the caller's kernel, on the legacy default stream
    }
    EXPECT_EQ(9, u.getMat(ACCESS_READ).at<uchar>(63, 63));
}

TEST(Core_Accel_CUDA, mat_backed_umat_gets_device_results_back)
{
    accel::Device d = cudaDevice();
    Mat m(32, 32, CV_8UC1, Scalar(1));
    {
        accel::ExecutionScope scope(d);
        UMat u = m.getUMat(ACCESS_RW);
        makeOn(d, Mat(32, 32, CV_8UC1, Scalar(6))).copyTo(u);  // a copy on the GPU into it
    }
    EXPECT_EQ(0, maxDiff(Mat(32, 32, CV_8UC1, Scalar(6)), m));
}
#endif // HAVE_CUDA

}} // namespace
