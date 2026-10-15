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

#include <fstream>
#include <thread>

namespace opencv_test { namespace {

enum { DISCRETE = 0, COHERENT = 1, NONCOHERENT = 2, PROTECTED_MEM = 3, MANAGED = 4, FAILING = 5 };

static accel::Device emulator(int index)
{
    accel::detail::registerEmulatorBackend();
    accel::Device d = accel::Device::find(cv::format("emulator:%d", index));
    CV_Assert(!d.empty());
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
#endif

TEST(Core_Accel, validation_mode_catches_writes_through_read_views)
{
    accel::detail::setValidation(true);
    UMat u = makeOn(emulator(COHERENT), Mat(8, 8, CV_8UC1, Scalar(1)));
    {
        Mat view = u.getMat(ACCESS_READ);
        view.at<uchar>(0, 0) = 9;           // a write the protocol never hears about
    }
    EXPECT_THROW(u.getMat(ACCESS_READ), cv::Exception);    // the buffer is no longer trusted
    accel::detail::setValidation(false);
}

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

    // the same pages cannot be imported twice
    EXPECT_THROW(accel::wrapExternalMemory(d, mem, MatShape({64, 64}), CV_8UC1), cv::Exception);

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

// The only test that uses emulator:5: OpenCV keeps a lost device lost for the whole process.
TEST(Core_Accel, device_failures_never_escape_a_destructor)
{
    accel::Device d = emulator(FAILING);
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
    EXPECT_EQ(0, accel::detail::emulatorLiveAllocations(FAILING));

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
    EXPECT_EQ(0, accel::detail::emulatorLiveAllocations(FAILING));

    // 3. calls that fail after queuing their work: what they write is invalid until written whole
    {
        UMat u = makeOn(d, src), out, c = makeOn(d, other);
        accel::detail::emulatorInjectFailure(CV_ACCEL_INTERNAL_ERROR, true);
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
    EXPECT_EQ(0, accel::detail::emulatorLiveAllocations(FAILING));

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
