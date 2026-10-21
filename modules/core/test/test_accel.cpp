// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "test_precomp.hpp"
#include "opencv2/core/accel_backend.hpp"

#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <map>
#include <mutex>
#include <thread>

namespace opencv_test { namespace {

/* "emu", a test backend with three devices:
   emu:0  a discrete GPU: its memory sits at addresses the CPU cannot use, so OpenCV must copy;
   emu:1  shared, coherent memory: device addresses are host addresses;
   emu:2  shared, non-coherent memory: the CPU sees a separate "cache" that beginCpuAccess()
          refreshes from device memory and endCpuAccess() writes back, so a missing call shows up
          as stale data.
   Each stream runs its work on its own thread, after an optional delay, so a missing wait shows
   up as wrong data. Work on freed or unknown memory is counted as a violation. */

enum { EMU_DISCRETE = 0, EMU_SHARED = 1, EMU_NONCOHERENT = 2, EMU_DEVICES = 3 };

std::atomic<int> g_delayUs(0);

class EmuEvent CV_FINAL : public accel::Event
{
public:
    bool query() CV_OVERRIDE
    {
        std::lock_guard<std::mutex> lock(m_);
        if (done_ && !error_.empty())
            CV_Error(Error::StsError, error_);
        return done_;
    }
    void wait() CV_OVERRIDE
    {
        waitQuiet();
        std::lock_guard<std::mutex> lock(m_);
        if (!error_.empty())
            CV_Error(Error::StsError, error_);
    }
    void waitQuiet()
    {
        std::unique_lock<std::mutex> lock(m_);
        cond_.wait(lock, [this] { return done_; });
    }
    void complete(const std::string& error)
    {
        {
            std::lock_guard<std::mutex> lock(m_);
            done_ = true;
            error_ = error;
        }
        cond_.notify_all();
    }
private:
    std::mutex m_;
    std::condition_variable cond_;
    bool done_ = false;
    std::string error_;
};

class EmuStream CV_FINAL : public accel::Stream
{
public:
    explicit EmuStream(int device) : device_(device), stop_(false), worker_([this] { loop(); }) {}
    ~EmuStream()
    {
        {
            std::lock_guard<std::mutex> lock(m_);
            stop_ = true;
        }
        wake_.notify_all();
        worker_.join();
    }

    void enqueue(std::function<void()> task)
    {
        {
            std::lock_guard<std::mutex> lock(m_);
            queue_.push_back(std::move(task));
        }
        wake_.notify_all();
    }

    Ptr<accel::Event> record() CV_OVERRIDE
    {
        Ptr<EmuEvent> e = makePtr<EmuEvent>();
        enqueue([this, e] { e->complete(error()); });
        return e;
    }

    bool waitEvent(const Ptr<accel::Event>& event) CV_OVERRIDE
    {
        Ptr<EmuEvent> e = event.dynamicCast<EmuEvent>();
        if (!e)
            return false;
        enqueue([e] { e->waitQuiet(); });
        return true;
    }

    void synchronize() CV_OVERRIDE { record()->wait(); }

    //! from a task: the device failed; like CUDA's sticky errors, every later event reports it
    void fail(const std::string& message)
    {
        std::lock_guard<std::mutex> lock(m_);
        error_ = message;
    }
    void clearError()
    {
        std::lock_guard<std::mutex> lock(m_);
        error_.clear();
    }

    const int device_;

private:
    std::string error()
    {
        std::lock_guard<std::mutex> lock(m_);
        return error_;
    }

    void loop()
    {
        for (;;)
        {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(m_);
                wake_.wait(lock, [this] { return stop_ || !queue_.empty(); });
                if (queue_.empty())
                    return;
                task = std::move(queue_.front());
                queue_.pop_front();
            }
            const int delay = g_delayUs.load();
            if (delay > 0)
                std::this_thread::sleep_for(std::chrono::microseconds(delay));
            task();
        }
    }

    std::mutex m_;
    std::condition_variable wake_;
    std::deque<std::function<void()> > queue_;
    bool stop_;
    std::string error_;
    std::thread worker_;    // last: starts once everything above exists
};

EmuStream& emuStreamOf(accel::Stream& s)
{
    EmuStream* e = dynamic_cast<EmuStream*>(&s);
    CV_Assert(e);
    return *e;
}

class EmuBackend;

class EmuCoreOps CV_FINAL : public accel::CoreOps
{
public:
    explicit EmuCoreOps(EmuBackend* b) : b_(b) {}
    bool flip(const UMat& src, UMat& dst, int flipCode, accel::Stream& stream) CV_OVERRIDE;
    bool transpose(const UMat& src, UMat& dst, accel::Stream& stream) CV_OVERRIDE;
private:
    EmuBackend* b_;
};

class EmuBackend CV_FINAL : public accel::Backend
{
public:
    EmuBackend() : core_(this), nextFake_((uintptr_t)0xDE0000000000ull)
    {
        for (int i = 0; i < EMU_DEVICES; i++)
        {
            opsRun_[i] = 0;
            live_[i] = 0;
        }
    }

    String name() const CV_OVERRIDE { return "emu"; }
    int deviceCount() const CV_OVERRIDE { return EMU_DEVICES; }
    String deviceDescription(int d) const CV_OVERRIDE
    {
        return d == EMU_DISCRETE ? "emulated discrete GPU" : d == EMU_SHARED ? "emulated coherent memory"
                                                                              : "emulated non-coherent memory";
    }
    bool sharesHostMemory(int d) const CV_OVERRIDE { return d != EMU_DISCRETE; }

    void* allocate(int d, size_t size) CV_OVERRIDE
    {
        std::lock_guard<std::mutex> lock(mem_);
        Block* b = new Block();
        b->device = d;
        b->mem.resize(size);
        if (d == EMU_NONCOHERENT)
            b->cache.resize(size);
        if (d == EMU_SHARED)
            b->addr = (uintptr_t)b->mem.data();
        else
        {
            b->addr = nextFake_;    // an address the CPU cannot dereference
            nextFake_ += ((size >> 12) + 2) << 12;
        }
        blocks_[b->addr] = b;
        live_[d]++;
        return (void*)b->addr;
    }

    void deallocate(int d, void* ptr) CV_OVERRIDE
    {
        std::lock_guard<std::mutex> lock(mem_);
        std::map<uintptr_t, Block*>::iterator it = blocks_.find((uintptr_t)ptr);
        if (it == blocks_.end() || it->second->device != d)
        {
            violations_++;
            return;
        }
        live_[d]--;
        delete it->second;
        blocks_.erase(it);
    }

    void* hostAddress(int d, void* ptr) CV_OVERRIDE
    {
        if (d == EMU_SHARED)
            return ptr;
        std::lock_guard<std::mutex> lock(mem_);
        Block* b = find((uintptr_t)ptr, 1);
        if (!b || d != EMU_NONCOHERENT)
        {
            violations_++;
            return 0;
        }
        return b->cache.data() + ((uintptr_t)ptr - b->addr);
    }

    void beginCpuAccess(int d, void* ptr, size_t size, bool) CV_OVERRIDE
    {
        begins_++;
        if (d != EMU_NONCOHERENT)
            return;
        std::lock_guard<std::mutex> lock(mem_);
        Block* b = find((uintptr_t)ptr, size);
        if (!b)
        {
            violations_++;
            return;
        }
        const size_t off = (uintptr_t)ptr - b->addr;
        memcpy(b->cache.data() + off, b->mem.data() + off, size);   // the CPU now sees the device's bytes
    }

    void endCpuAccess(int d, void* ptr, size_t size, bool written) CV_OVERRIDE
    {
        ends_++;
        if (d != EMU_NONCOHERENT || !written)
            return;
        std::lock_guard<std::mutex> lock(mem_);
        Block* b = find((uintptr_t)ptr, size);
        if (!b)
        {
            violations_++;
            return;
        }
        const size_t off = (uintptr_t)ptr - b->addr;
        memcpy(b->mem.data() + off, b->cache.data() + off, size);   // the device now sees the CPU's bytes
    }

    void copy(int d, accel::CopyKind kind, void* dst, size_t dstStep, const void* src, size_t srcStep,
              size_t w, size_t rows, accel::Stream& stream, bool hostStable) CV_OVERRIDE
    {
        EmuStream& s = emuStreamOf(stream);
        CV_Assert(s.device_ == d);
        const uintptr_t da = (uintptr_t)dst, sa = (uintptr_t)src;
        if (kind == accel::COPY_HOST_TO_DEVICE && !hostStable)
        {
            // the caller's memory may change once we return: read it now, as a staging copy would
            std::shared_ptr<std::vector<uchar> > staged = std::make_shared<std::vector<uchar> >(w * rows);
            for (size_t r = 0; r < rows; r++)
                memcpy(staged->data() + r * w, (const uchar*)src + r * srcStep, w);
            s.enqueue([this, staged, da, dstStep, w, rows] {
                std::lock_guard<std::mutex> lock(mem_);
                uchar* p = deviceBytes(da, dstStep, w, rows);
                if (p)
                    for (size_t r = 0; r < rows; r++)
                        memcpy(p + r * dstStep, staged->data() + r * w, w);
            });
            return;
        }
        s.enqueue([this, kind, da, dstStep, sa, srcStep, w, rows] {
            std::lock_guard<std::mutex> lock(mem_);
            const uchar* sp = kind == accel::COPY_HOST_TO_DEVICE ? (const uchar*)sa : deviceBytes(sa, srcStep, w, rows);
            uchar* dp = kind == accel::COPY_DEVICE_TO_HOST ? (uchar*)da : deviceBytes(da, dstStep, w, rows);
            if (sp && dp)
                for (size_t r = 0; r < rows; r++)
                    memcpy(dp + r * dstStep, sp + r * srcStep, w);
        });
    }

    Ptr<accel::Stream> createStream(int d) CV_OVERRIDE
    {
        Ptr<EmuStream> s = makePtr<EmuStream>(d);
        std::lock_guard<std::mutex> lock(mem_);
        streams_.push_back(s.get());   // default streams live for the whole process
        return s;
    }

    accel::Ops* ops(const String& module) CV_OVERRIDE
    {
        return module == accel::CoreOps::module() ? &core_ : 0;
    }

    //! the bytes of [addr, addr + (rows-1)*step + width) in device memory; NULL (a violation) if unknown
    uchar* deviceBytes(uintptr_t addr, size_t step, size_t width, size_t rows)
    {
        Block* b = find(addr, rows > 0 ? (rows - 1) * step + width : 0);
        if (!b)
        {
            violations_++;
            return 0;
        }
        return b->mem.data() + (addr - b->addr);
    }

    void reset()
    {
        g_delayUs = 0;
        throwNext = false;
        failNextAsync = false;
        violations_ = 0;
        begins_ = 0;
        ends_ = 0;
        for (int i = 0; i < EMU_DEVICES; i++)
            opsRun_[i] = 0;
        clearErrors();
    }

    void clearErrors()
    {
        std::lock_guard<std::mutex> lock(mem_);
        for (size_t i = 0; i < streams_.size(); i++)
            streams_[i]->clearError();
    }

    int live(int d)
    {
        std::lock_guard<std::mutex> lock(mem_);
        return live_[d];
    }

    std::atomic<bool> throwNext{false};        // the next operation throws
    std::atomic<bool> failNextAsync{false};    // the next operation fails on the device, later
    std::atomic<int> violations_{0};
    std::atomic<int> begins_{0}, ends_{0};
    std::atomic<int> opsRun_[EMU_DEVICES];
    std::mutex mem_;    // held by every task while it touches device memory

private:
    struct Block
    {
        int device;
        uintptr_t addr;
        std::vector<uchar> mem;
        std::vector<uchar> cache;
    };

    Block* find(uintptr_t addr, size_t len)
    {
        std::map<uintptr_t, Block*>::iterator it = blocks_.upper_bound(addr);
        if (it == blocks_.begin())
            return 0;
        --it;
        Block* b = it->second;
        return (addr >= b->addr && addr + len <= b->addr + b->mem.size()) ? b : 0;
    }

    EmuCoreOps core_;
    std::map<uintptr_t, Block*> blocks_;
    uintptr_t nextFake_;
    int live_[EMU_DEVICES];
    std::vector<EmuStream*> streams_;
};

bool EmuCoreOps::flip(const UMat& src, UMat& dst, int flipCode, accel::Stream& stream)
{
    if (src.depth() == CV_64F)
        return false;   // the emulated device has no doubles: the CPU runs these calls
    if (b_->throwNext.exchange(false))
        CV_Error(Error::StsError, "emu: flip failed");
    EmuStream* s = &emuStreamOf(stream);
    b_->opsRun_[s->device_]++;
    const uintptr_t sa = (uintptr_t)accel::devicePtr(src), da = (uintptr_t)accel::devicePtr(dst);
    const size_t ss = src.step[0], ds = dst.step[0];
    const int rows = src.rows, cols = src.cols, esz = (int)src.elemSize();
    const bool fail = b_->failNextAsync.exchange(false);
    EmuBackend* b = b_;
    s->enqueue([=] {
        if (fail)
        {
            s->fail("emu: flip failed on the device");
            return;
        }
        std::lock_guard<std::mutex> lock(b->mem_);
        const uchar* sp = b->deviceBytes(sa, ss, (size_t)cols * esz, rows);
        uchar* dp = b->deviceBytes(da, ds, (size_t)cols * esz, rows);
        if (!sp || !dp)
            return;
        const bool flipRows = flipCode <= 0, flipCols = flipCode != 0;
        for (int y = 0; y < rows; y++)
            for (int x = 0; x < cols; x++)
                memcpy(dp + y * ds + (size_t)x * esz,
                       sp + (flipRows ? rows - 1 - y : y) * ss + (size_t)(flipCols ? cols - 1 - x : x) * esz, esz);
    });
    return true;
}

bool EmuCoreOps::transpose(const UMat& src, UMat& dst, accel::Stream& stream)
{
    if (src.depth() == CV_64F)
        return false;
    EmuStream* s = &emuStreamOf(stream);
    b_->opsRun_[s->device_]++;
    const uintptr_t sa = (uintptr_t)accel::devicePtr(src), da = (uintptr_t)accel::devicePtr(dst);
    const size_t ss = src.step[0], ds = dst.step[0];
    const int rows = src.rows, cols = src.cols, esz = (int)src.elemSize();
    EmuBackend* b = b_;
    s->enqueue([=] {
        std::lock_guard<std::mutex> lock(b->mem_);
        const uchar* sp = b->deviceBytes(sa, ss, (size_t)cols * esz, rows);
        uchar* dp = b->deviceBytes(da, ds, (size_t)rows * esz, cols);
        if (!sp || !dp)
            return;
        for (int y = 0; y < rows; y++)
            for (int x = 0; x < cols; x++)
                memcpy(dp + x * ds + (size_t)y * esz, sp + y * ss + (size_t)x * esz, esz);
    });
    return true;
}

EmuBackend& emu()
{
    static EmuBackend* backend = 0;
    static std::once_flag once;
    std::call_once(once, [] {
        Ptr<EmuBackend> b = makePtr<EmuBackend>();
        backend = b.get();
        accel::registerBackend(b);
    });
    return *backend;
}

accel::Device emuDevice(int i)
{
    emu();
    return accel::Device::find(cv::format("emu:%d", i));
}

//! a UMat in @p d's memory holding @p m; its allocator field stays empty
UMat onDevice(const accel::Device& d, const Mat& m)
{
    UMat u;
    u.allocator = d.allocator();
    u.create(m.size(), m.type());
    u.allocator = 0;
    m.copyTo(u);
    return u;
}

Mat toHost(const UMat& u)
{
    Mat m;
    u.copyTo(m);
    return m;
}

Mat randomMat(Size size, int type)
{
    Mat m(size, type);
    randu(m, 0, 255);
    return m;
}

class Core_Accel : public testing::Test
{
protected:
    void SetUp() CV_OVERRIDE
    {
        if (sizeof(void*) < 8)
            throw SkipTestException("the emulator needs 64-bit addresses");
        emu().reset();
        accel::setTransferPolicy(accel::TRANSFER_ALLOW);
        accel::resetTransferStats();
    }
    void TearDown() CV_OVERRIDE
    {
        if (sizeof(void*) < 8)
            return;
        for (int i = 0; i < EMU_DEVICES; i++)
            accel::synchronize(emuDevice(i));
        accel::setTransferPolicy(accel::TRANSFER_ALLOW);
        g_delayUs = 0;
        EXPECT_EQ(0, emu().violations_.load());
        EXPECT_EQ(emu().begins_.load(), emu().ends_.load());
    }
};

TEST_F(Core_Accel, devices)
{
    accel::Device d0 = emuDevice(EMU_DISCRETE), d1 = emuDevice(EMU_SHARED), d2 = emuDevice(EMU_NONCOHERENT);
    ASSERT_FALSE(d0.empty());
    ASSERT_FALSE(d1.empty());
    ASSERT_FALSE(d2.empty());
    EXPECT_EQ("emu:1", d1.name());
    EXPECT_EQ("emu", d1.backend());
    EXPECT_EQ(1, d1.index());
    EXPECT_EQ(d0, accel::Device::find("emu"));
    EXPECT_TRUE(accel::Device::find("emu:7").empty());
    EXPECT_TRUE(accel::Device::find("nonexistent").empty());
    EXPECT_FALSE(d0.sharesHostMemory());
    EXPECT_TRUE(d1.sharesHostMemory());
    EXPECT_TRUE(d2.sharesHostMemory());
    EXPECT_EQ("cpu", accel::Device().name());
    EXPECT_EQ(Mat::getDefaultAllocator(), accel::Device().allocator());

    int found = 0;
    std::vector<accel::Device> all = accel::Device::enumerate();
    for (size_t i = 0; i < all.size(); i++)
        found += all[i].backend() == "emu";
    EXPECT_EQ(EMU_DEVICES, found);
}

TEST_F(Core_Accel, round_trip_on_every_device)
{
    const Mat src = randomMat(Size(53, 37), CV_8UC3);
    for (int d = 0; d < EMU_DEVICES; d++)
    {
        accel::resetTransferStats();
        UMat u = onDevice(emuDevice(d), src);
        EXPECT_EQ(emuDevice(d), accel::getDevice(u));
        EXPECT_EQ(0, cvtest::norm(src, toHost(u), NORM_INF)) << "device " << d;
        const accel::TransferStats st = accel::getTransferStats();
        if (d == EMU_DISCRETE)
        {
            EXPECT_EQ(src.total() * src.elemSize(), st.bytesHostToDevice);
            EXPECT_EQ(src.total() * src.elemSize(), st.bytesDeviceToHost);
        }
        else
            EXPECT_EQ(0u, st.transfers) << "shared memory is never copied";
    }
}

TEST_F(Core_Accel, flip_runs_on_the_device_of_its_input)
{
    const Mat src = randomMat(Size(31, 17), CV_8UC3);
    for (int d = 0; d < EMU_DEVICES; d++)
    {
        for (int code = -1; code <= 1; code++)
        {
            UMat u = onDevice(emuDevice(d), src), dst;
            emu().opsRun_[d] = 0;
            accel::resetTransferStats();
            cv::flip(u, dst, code);
            const accel::TransferStats st = accel::getTransferStats();
            EXPECT_EQ(1u, st.deviceOps);
            EXPECT_EQ(0u, st.cpuFallbacks);
            EXPECT_EQ(0u, st.transfers);
            EXPECT_EQ(1, emu().opsRun_[d].load());
            EXPECT_EQ(emuDevice(d), accel::getDevice(dst)) << "the output is created where the call ran";
            Mat expected;
            cv::flip(src, expected, code);
            EXPECT_EQ(0, cvtest::norm(expected, toHost(dst), NORM_INF)) << "device " << d << ", code " << code;
        }
    }
}

TEST_F(Core_Accel, transpose_runs_on_the_device)
{
    const Mat src = randomMat(Size(29, 13), CV_32FC2);
    for (int d = 0; d < EMU_DEVICES; d++)
    {
        UMat u = onDevice(emuDevice(d), src), dst;
        accel::resetTransferStats();
        cv::transpose(u, dst);
        EXPECT_EQ(1u, accel::getTransferStats().deviceOps);
        EXPECT_EQ(emuDevice(d), accel::getDevice(dst));
        EXPECT_EQ(Size(13, 29), dst.size());
        EXPECT_EQ(0, cvtest::norm(src.t(), toHost(dst), NORM_INF)) << "device " << d;
    }
}

TEST_F(Core_Accel, declined_call_runs_on_the_cpu_and_its_output_stays_on_the_device)
{
    const Mat src = randomMat(Size(20, 10), CV_64FC1);  // the emulator declines doubles
    UMat u = onDevice(emuDevice(EMU_DISCRETE), src), dst;
    accel::resetTransferStats();
    cv::flip(u, dst, 1);
    const accel::TransferStats st = accel::getTransferStats();
    EXPECT_EQ(0u, st.deviceOps);
    EXPECT_EQ(1u, st.cpuFallbacks);
    EXPECT_EQ(emuDevice(EMU_DISCRETE), accel::getDevice(dst));
    Mat expected;
    cv::flip(src, expected, 1);
    EXPECT_EQ(0, cvtest::norm(expected, toHost(dst), NORM_INF));
}

TEST_F(Core_Accel, cpu_fallback_reads_its_input_without_uploading_it_again)
{
    const Mat src = randomMat(Size(20, 10), CV_64FC1);
    const uint64 bytes = src.total() * src.elemSize();
    UMat u = onDevice(emuDevice(EMU_DISCRETE), src), dst;
    accel::resetTransferStats();
    cv::flip(u, dst, 0);
    const accel::TransferStats st = accel::getTransferStats();
    EXPECT_EQ(bytes, st.bytesDeviceToHost) << "the input is downloaded once";
    EXPECT_EQ(bytes, st.bytesHostToDevice) << "only the output is uploaded; a fresh output is not downloaded";
}

TEST_F(Core_Accel, shared_memory_is_used_in_place)
{
    const Mat src = randomMat(Size(16, 8), CV_64FC1);
    UMat u = onDevice(emuDevice(EMU_SHARED), src);
    accel::resetTransferStats();
    {
        Mat view = u.getMat(ACCESS_READ);
        EXPECT_EQ((uchar*)u.u->handle + u.offset, view.data) << "the CPU reads the device's own bytes";
    }
    UMat dst;
    cv::flip(u, dst, 1);    // declined: the CPU runs it, on the same memory
    EXPECT_EQ(1u, accel::getTransferStats().cpuFallbacks);
    EXPECT_EQ(0u, accel::getTransferStats().transfers);
    Mat expected;
    cv::flip(src, expected, 1);
    EXPECT_EQ(0, cvtest::norm(expected, toHost(dst), NORM_INF));
}

TEST_F(Core_Accel, noncoherent_memory_keeps_the_caches_right)
{
    accel::Device d = emuDevice(EMU_NONCOHERENT);
    UMat u;
    u.allocator = d.allocator();
    u.create(12, 9, CV_8UC1);
    u.allocator = 0;
    {
        Mat view = u.getMat(ACCESS_WRITE);
        for (int y = 0; y < view.rows; y++)
            for (int x = 0; x < view.cols; x++)
                view.at<uchar>(y, x) = (uchar)(y * 16 + x);
    }   // releasing the view writes the CPU cache back
    UMat dst;
    cv::flip(u, dst, -1);   // runs on the device, on device memory
    Mat expected(12, 9, CV_8UC1);
    for (int y = 0; y < 12; y++)
        for (int x = 0; x < 9; x++)
            expected.at<uchar>(y, x) = (uchar)((11 - y) * 16 + (8 - x));
    Mat got = dst.getMat(ACCESS_READ).clone();     // the view refreshes the CPU cache first
    EXPECT_EQ(0, cvtest::norm(expected, got, NORM_INF));
    EXPECT_EQ(1, emu().opsRun_[EMU_NONCOHERENT].load());
}

TEST_F(Core_Accel, cpu_waits_for_device_work)
{
    const Mat src = randomMat(Size(40, 30), CV_8UC4);
    UMat u = onDevice(emuDevice(EMU_DISCRETE), src), dst;
    accel::synchronize(emuDevice(EMU_DISCRETE));
    g_delayUs = 20000;
    accel::resetTransferStats();
    cv::flip(u, dst, 1);
    Mat got = toHost(dst);
    EXPECT_GE(accel::getTransferStats().hostWaits, 1u);
    Mat expected;
    cv::flip(src, expected, 1);
    EXPECT_EQ(0, cvtest::norm(expected, got, NORM_INF));
}

TEST_F(Core_Accel, cpu_write_waits_for_device_reads)
{
    const Mat src = randomMat(Size(40, 30), CV_8UC1);
    UMat u = onDevice(emuDevice(EMU_DISCRETE), src), dst;
    accel::synchronize(emuDevice(EMU_DISCRETE));
    g_delayUs = 20000;
    cv::flip(u, dst, 0);    // reads u later, on the device
    u.setTo(Scalar(7));     // the CPU write has to wait for that read
    Mat expected;
    cv::flip(src, expected, 0);
    EXPECT_EQ(0, cvtest::norm(expected, toHost(dst), NORM_INF));
    EXPECT_EQ(0, cvtest::norm(Mat(src.size(), CV_8UC1, Scalar(7)), toHost(u), NORM_INF));
}

TEST_F(Core_Accel, released_memory_outlives_the_device_work_using_it)
{
    const Mat src = randomMat(Size(64, 48), CV_8UC1);
    const accel::Device d = emuDevice(EMU_DISCRETE);
    accel::synchronize(d);
    const int before = emu().live(EMU_DISCRETE);
    UMat dst;
    g_delayUs = 20000;
    {
        UMat u = onDevice(d, src);
        cv::flip(u, dst, 1);
    }   // u is released while the flip still reads it
    accel::synchronize(d);
    g_delayUs = 0;
    Mat expected;
    cv::flip(src, expected, 1);
    EXPECT_EQ(0, cvtest::norm(expected, toHost(dst), NORM_INF));
    accel::synchronize(d);
    EXPECT_EQ(before + 1, emu().live(EMU_DISCRETE)) << "only dst is left";
}

TEST_F(Core_Accel, output_on_another_device_moves_to_the_input)
{
    const Mat src = randomMat(Size(10, 6), CV_16UC1);
    UMat u = onDevice(emuDevice(EMU_DISCRETE), src);
    UMat dst = onDevice(emuDevice(EMU_SHARED), Mat::zeros(6, 10, CV_16UC1));
    cv::flip(u, dst, 1);
    EXPECT_EQ(emuDevice(EMU_DISCRETE), accel::getDevice(dst));
    EXPECT_EQ(1u, accel::getTransferStats().deviceOps);
    Mat expected;
    cv::flip(src, expected, 1);
    EXPECT_EQ(0, cvtest::norm(expected, toHost(dst), NORM_INF));
}

TEST_F(Core_Accel, region_output_is_written_in_place)
{
    const Mat src = randomMat(Size(8, 6), CV_8UC2);
    UMat u = onDevice(emuDevice(EMU_DISCRETE), src);
    UMat big = onDevice(emuDevice(EMU_DISCRETE), Mat::zeros(20, 20, CV_8UC2));
    cv::flip(u, big(Rect(4, 5, 8, 6)), 1);
    EXPECT_EQ(1u, accel::getTransferStats().deviceOps);
    EXPECT_EQ(emuDevice(EMU_DISCRETE), accel::getDevice(big));
    Mat expected = Mat::zeros(20, 20, CV_8UC2);
    cv::flip(src, expected(Rect(4, 5, 8, 6)), 1);
    EXPECT_EQ(0, cvtest::norm(expected, toHost(big), NORM_INF));
}

TEST_F(Core_Accel, region_output_on_another_device_gets_the_result_copied_in)
{
    const Mat src = randomMat(Size(8, 6), CV_8UC1);
    UMat u = onDevice(emuDevice(EMU_DISCRETE), src);
    UMat big = onDevice(emuDevice(EMU_SHARED), Mat::zeros(10, 12, CV_8UC1));
    cv::flip(u, big(Rect(2, 3, 8, 6)), 0);
    EXPECT_EQ(1u, accel::getTransferStats().deviceOps);
    EXPECT_EQ(emuDevice(EMU_SHARED), accel::getDevice(big)) << "a region cannot move";
    Mat expected = Mat::zeros(10, 12, CV_8UC1);
    cv::flip(src, expected(Rect(2, 3, 8, 6)), 0);
    EXPECT_EQ(0, cvtest::norm(expected, toHost(big), NORM_INF));
}

TEST_F(Core_Accel, mat_output_is_downloaded)
{
    const Mat src = randomMat(Size(11, 7), CV_32FC1);
    UMat u = onDevice(emuDevice(EMU_DISCRETE), src);
    Mat dst;
    accel::resetTransferStats();
    cv::flip(u, dst, -1);
    EXPECT_EQ(1u, accel::getTransferStats().deviceOps);
    EXPECT_EQ(src.total() * src.elemSize(), accel::getTransferStats().bytesDeviceToHost);
    Mat expected;
    cv::flip(src, expected, -1);
    EXPECT_EQ(0, cvtest::norm(expected, dst, NORM_INF));
}

TEST_F(Core_Accel, in_place_call)
{
    const Mat src = randomMat(Size(9, 9), CV_8UC3);
    for (int d = 0; d < EMU_DEVICES; d++)
    {
        UMat u = onDevice(emuDevice(d), src);
        cv::flip(u, u, 1);
        cv::transpose(u, u);
        Mat expected;
        cv::flip(src, expected, 1);
        expected = expected.t();
        EXPECT_EQ(0, cvtest::norm(expected, toHost(u), NORM_INF)) << "device " << d;
        EXPECT_EQ(emuDevice(d), accel::getDevice(u));
    }
}

TEST_F(Core_Accel, execution_scope_places_new_data_and_runs_calls)
{
    const accel::Device d = emuDevice(EMU_DISCRETE);
    const Mat src = randomMat(Size(12, 5), CV_8UC1);
    UMat host;
    src.copyTo(host);   // OpenCL or host memory
    ASSERT_TRUE(accel::getDevice(host).empty());
    {
        accel::ExecutionScope scope(d);
        UMat fresh(5, 5, CV_8UC1);
        EXPECT_EQ(d, accel::getDevice(fresh));
        UMat dst;
        accel::resetTransferStats();
        cv::flip(host, dst, 1);     // the input is copied to the scope's device
        EXPECT_EQ(1u, accel::getTransferStats().deviceOps);
        EXPECT_EQ(d, accel::getDevice(dst));
        Mat expected;
        cv::flip(src, expected, 1);
        EXPECT_EQ(0, cvtest::norm(expected, toHost(dst), NORM_INF));
        {
            accel::ExecutionScope cpu((accel::Device()));
            UMat onCpu(4, 4, CV_8UC1);
            EXPECT_TRUE(accel::getDevice(onCpu).empty());
            UMat dst2;
            cv::flip(dst, dst2, 1);     // runs on the CPU
            EXPECT_EQ(1u, accel::getTransferStats().deviceOps);
            EXPECT_EQ(0, cvtest::norm(src, toHost(dst2), NORM_INF));
        }
    }
    UMat after(3, 3, CV_8UC1);
    EXPECT_TRUE(accel::getDevice(after).empty());
}

TEST_F(Core_Accel, transfer_forbid_reports_hidden_copies)
{
    const Mat src8 = randomMat(Size(8, 8), CV_8UC1), src64 = randomMat(Size(8, 8), CV_64FC1);
    UMat u8 = onDevice(emuDevice(EMU_DISCRETE), src8);
    UMat u64 = onDevice(emuDevice(EMU_DISCRETE), src64);
    UMat shared64 = onDevice(emuDevice(EMU_SHARED), src64);
    accel::setTransferPolicy(accel::TRANSFER_FORBID);
    UMat dst;
    EXPECT_NO_THROW(cv::flip(u8, dst, 1)) << "everything stays on the device";
    Mat matDst;
    EXPECT_THROW(cv::flip(u8, matDst, 1), cv::Exception) << "the result would be downloaded";
    UMat dst64;
    EXPECT_THROW(cv::flip(u64, dst64, 1), cv::Exception) << "the CPU would copy device data";
    UMat sharedDst;
    EXPECT_NO_THROW(cv::flip(shared64, sharedDst, 1)) << "the CPU runs on shared memory without copies";
    accel::setTransferPolicy(accel::TRANSFER_ALLOW);
}

TEST_F(Core_Accel, device_error_surfaces_when_the_cpu_waits)
{
    const Mat src = randomMat(Size(8, 8), CV_8UC1);
    UMat u = onDevice(emuDevice(EMU_DISCRETE), src), dst;
    accel::synchronize(emuDevice(EMU_DISCRETE));
    emu().failNextAsync = true;
    cv::flip(u, dst, 1);    // queued: the failure happens later, on the device
    EXPECT_THROW(toHost(dst), cv::Exception);
    EXPECT_THROW(accel::synchronize(emuDevice(EMU_DISCRETE)), cv::Exception) << "the error is sticky";
    emu().clearErrors();
}

TEST_F(Core_Accel, operation_errors_propagate)
{
    UMat u = onDevice(emuDevice(EMU_DISCRETE), randomMat(Size(8, 8), CV_8UC1)), dst;
    emu().throwNext = true;
    EXPECT_THROW(cv::flip(u, dst, 1), cv::Exception);
}

TEST_F(Core_Accel, handle_waits_for_device_work)
{
    UMat u = onDevice(emuDevice(EMU_DISCRETE), randomMat(Size(32, 32), CV_8UC1)), dst;
    accel::synchronize(emuDevice(EMU_DISCRETE));
    g_delayUs = 20000;
    cv::flip(u, dst, 1);
    accel::resetTransferStats();
    void* h = dst.handle(ACCESS_READ);
    EXPECT_EQ(dst.u->handle, h);
    EXPECT_GE(accel::getTransferStats().hostWaits, 1u);
}

TEST_F(Core_Accel, copies_between_umats)
{
    const Mat src = randomMat(Size(15, 11), CV_8UC3);
    UMat a = onDevice(emuDevice(EMU_DISCRETE), src);

    UMat same;
    same.allocator = emuDevice(EMU_DISCRETE).allocator();
    accel::resetTransferStats();
    a.copyTo(same);
    EXPECT_EQ(0u, accel::getTransferStats().transfers) << "within one device";
    EXPECT_EQ(src.total() * src.elemSize(), accel::getTransferStats().bytesDeviceToDevice);

    UMat other;
    other.allocator = emuDevice(EMU_NONCOHERENT).allocator();
    a.copyTo(other);    // through the host
    EXPECT_EQ(emuDevice(EMU_NONCOHERENT), accel::getDevice(other));
    EXPECT_EQ(0, cvtest::norm(src, toHost(same), NORM_INF));
    EXPECT_EQ(0, cvtest::norm(src, toHost(other), NORM_INF));
}

TEST_F(Core_Accel, opencl_declines_device_memory)
{
    if (!cv::ocl::haveOpenCL())
        throw SkipTestException("OpenCL is not available");
    const bool prev = cv::ocl::useOpenCL();
    cv::ocl::setUseOpenCL(true);
    const Mat src = randomMat(Size(20, 10), CV_64FC1);  // declined by the emulator: the call reaches the OpenCL branch
    UMat u = onDevice(emuDevice(EMU_DISCRETE), src), dst;
    cv::flip(u, dst, 1);
    cv::ocl::setUseOpenCL(prev);
    Mat expected;
    cv::flip(src, expected, 1);
    EXPECT_EQ(0, cvtest::norm(expected, toHost(dst), NORM_INF));
}

TEST_F(Core_Accel, threads_share_a_device)
{
    const accel::Device d = emuDevice(EMU_DISCRETE);
    std::vector<std::thread> threads;
    std::atomic<int> failures(0);
    for (int t = 0; t < 4; t++)
    {
        threads.push_back(std::thread([&, t] {
            try
            {
                const Mat src = randomMat(Size(17 + t, 9), CV_8UC1);
                UMat u = onDevice(d, src), a, b;
                for (int i = 0; i < 10; i++)
                {
                    cv::flip(u, a, i % 3 - 1);
                    cv::flip(a, b, i % 3 - 1);
                    if (cvtest::norm(src, toHost(b), NORM_INF) != 0)
                        failures++;
                }
            }
            catch (...)
            {
                failures++;
            }
        }));
    }
    for (size_t i = 0; i < threads.size(); i++)
        threads[i].join();
    EXPECT_EQ(0, failures.load());
}

TEST_F(Core_Accel, mat_calls_are_not_affected)
{
    UMat keepActive = onDevice(emuDevice(EMU_DISCRETE), Mat::ones(2, 2, CV_8UC1));
    const Mat src = randomMat(Size(10, 10), CV_8UC1);
    Mat dst;
    accel::resetTransferStats();
    cv::flip(src, dst, 1);
    const accel::TransferStats st = accel::getTransferStats();
    EXPECT_EQ(0u, st.deviceOps);
    EXPECT_EQ(0u, st.cpuFallbacks);
    EXPECT_EQ(0u, st.transfers);
}

#ifdef HAVE_CUDA

accel::Device cudaDevice()
{
    accel::Device d = accel::Device::find("cuda:0");
    if (d.empty())
        throw SkipTestException("no CUDA device");
    return d;
}

TEST(Core_Accel_CUDA, flip_and_transpose)
{
    const accel::Device d = cudaDevice();
    const int types[] = { CV_8UC1, CV_8UC3, CV_16UC2, CV_32FC1, CV_32FC4, CV_64FC3 };
    for (size_t t = 0; t < sizeof(types) / sizeof(types[0]); t++)
    {
        const Mat src = randomMat(Size(1923, 1081), types[t]);
        UMat u = onDevice(d, src), f, tr;
        accel::resetTransferStats();
        cv::flip(u, f, -1);
        cv::transpose(u, tr);
        EXPECT_EQ(2u, accel::getTransferStats().deviceOps) << typeToString(types[t]);
        Mat ef, et;
        cv::flip(src, ef, -1);
        cv::transpose(src, et);
        EXPECT_EQ(0, cvtest::norm(ef, toHost(f), NORM_INF)) << typeToString(types[t]);
        EXPECT_EQ(0, cvtest::norm(et, toHost(tr), NORM_INF)) << typeToString(types[t]);
        EXPECT_EQ(d, accel::getDevice(f));
    }
}

TEST(Core_Accel_CUDA, unsupported_element_size_falls_back)
{
    const accel::Device d = cudaDevice();
    const Mat src = randomMat(Size(33, 21), CV_8UC(5));    // 5-byte elements: no kernel
    UMat u = onDevice(d, src), dst;
    accel::resetTransferStats();
    cv::flip(u, dst, 1);
    EXPECT_EQ(1u, accel::getTransferStats().cpuFallbacks);
    EXPECT_EQ(d, accel::getDevice(dst));
    Mat expected;
    cv::flip(src, expected, 1);
    EXPECT_EQ(0, cvtest::norm(expected, toHost(dst), NORM_INF));
}

TEST(Core_Accel_CUDA, region_and_chain)
{
    const accel::Device d = cudaDevice();
    const Mat src = randomMat(Size(640, 480), CV_8UC3);
    UMat u = onDevice(d, src), a, b;
    UMat big = onDevice(d, Mat::zeros(500, 700, CV_8UC3));
    for (int i = 0; i < 20; i++)
    {
        cv::flip(u, a, 1);
        cv::flip(a, b, 1);
    }
    cv::flip(b, big(Rect(30, 10, 640, 480)), 0);
    Mat expected = Mat::zeros(500, 700, CV_8UC3);
    cv::flip(src, expected(Rect(30, 10, 640, 480)), 0);
    EXPECT_EQ(0, cvtest::norm(expected, toHost(big), NORM_INF));
}

#endif // HAVE_CUDA

}} // namespace
