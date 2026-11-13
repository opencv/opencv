// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "test_precomp.hpp"
#include "opencv2/core/accel_backend.hpp"

#include <atomic>
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
   emu:1  memory the CPU addresses: device addresses are host addresses;
   emu:2  a second discrete GPU, for transfers between devices.
   Each stream runs its work on its own thread, after an optional delay, so a missing wait shows
   up as wrong data. Work on freed memory or on another device's memory is counted as a violation. */
enum { EMU_DISCRETE = 0, EMU_SHARED = 1, EMU_DISCRETE2 = 2, EMU_DEVICES = 3 };

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
        worker_.join();     // after the queued work
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
    //! false for what the emulated devices decline: doubles, Mat views alive, overlapping views
    bool runnable(const UMat& src, const UMat& dst);
    //! queues fn(srcBytes, dstBytes) on @p stream, then keeps the arguments alive as backends do
    void launch(const UMat& src, UMat& dst, accel::Stream& stream,
                const std::function<void(const uchar*, uchar*)>& fn);

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
        static const char* names[] = { "emulated discrete GPU", "emulated memory the CPU addresses",
                                       "emulated second discrete GPU" };
        return names[d];
    }

    void* allocate(int d, size_t size) CV_OVERRIDE
    {
        std::lock_guard<std::mutex> lock(mem_);
        Block* b = new Block();
        b->device = d;
        b->size = size;
        b->storage.resize(size);
        b->base = b->storage.data();
        if (d == EMU_SHARED)
            b->addr = (uintptr_t)b->base;   // the CPU and the device use the same address
        else
        {
            b->addr = nextFake_;            // an address the CPU cannot use
            nextFake_ += ((size >> 12) + 2) << 12;
        }
        blocks_[b->addr] = b;
        live_[d]++;
        return (void*)b->addr;
    }

    void deallocate(int d, void* handle) CV_OVERRIDE
    {
        std::lock_guard<std::mutex> lock(mem_);
        std::map<uintptr_t, Block*>::iterator it = blocks_.find((uintptr_t)handle);
        if (it == blocks_.end() || it->second->device != d)
        {
            violations_++;
            return;
        }
        live_[d]--;
        delete it->second;
        blocks_.erase(it);
    }

    void* hostAddress(int d, void* handle) CV_OVERRIDE
    {
        return d == EMU_SHARED ? handle : 0;
    }

    void copy(int d, accel::CopyKind kind, accel::DeviceRef dst, size_t dstStep,
              accel::DeviceRef src, size_t srcStep, size_t w, size_t rows, accel::Stream& stream) CV_OVERRIDE
    {
        EmuStream& s = emuStreamOf(stream);
        CV_Assert(s.device_ == d);
        (kind == accel::COPY_HOST_TO_DEVICE ? h2d_ : kind == accel::COPY_DEVICE_TO_HOST ? d2h_ : d2d_) += w * rows;
        // this backend's handles are addresses of its own making, so the offset folds into them
        const uintptr_t da = (uintptr_t)dst.handle + dst.offset;
        const uintptr_t sa = (uintptr_t)src.handle + src.offset;
        s.enqueue([this, d, kind, da, dstStep, sa, srcStep, w, rows] {
            std::lock_guard<std::mutex> lock(mem_);
            const uchar* sp = kind == accel::COPY_HOST_TO_DEVICE ? (const uchar*)sa : deviceBytes(d, sa, srcStep, w, rows);
            uchar* dp = kind == accel::COPY_DEVICE_TO_HOST ? (uchar*)da : deviceBytes(d, da, dstStep, w, rows);
            if (sp && dp)
                for (size_t r = 0; r < rows; r++)
                    memcpy(dp + r * dstStep, sp + r * srcStep, w);
        });
        if (kind != accel::COPY_DEVICE_TO_DEVICE)
            s.synchronize();    // copies to and from the host finish before returning
    }

    Ptr<accel::Stream> createStream(int d) CV_OVERRIDE
    {
        Ptr<EmuStream> s = makePtr<EmuStream>(d);
        std::lock_guard<std::mutex> lock(mem_);
        streams_.push_back(std::weak_ptr<EmuStream>(s));
        streamsCreated_++;
        return s;
    }

    accel::Ops* ops(const char* module) CV_OVERRIDE
    {
        return strcmp(module, "core") == 0 ? &core_ : 0;
    }

    //! the bytes of [addr, addr + (rows-1)*step + width) in @p d's memory; NULL (a violation) if not there
    uchar* deviceBytes(int d, uintptr_t addr, size_t step, size_t width, size_t rows)
    {
        Block* b = find(addr, rows > 0 ? (rows - 1) * step + width : 0);
        if (!b || b->device != d)
        {
            violations_++;
            return 0;
        }
        return b->base + (addr - b->addr);
    }

    void reset()
    {
        g_delayUs = 0;
        throwNext = false;
        failNextAsync = false;
        violations_ = 0;
        resetCounters();
        clearErrors();
    }

    void resetCounters()
    {
        h2d_ = 0;
        d2h_ = 0;
        d2d_ = 0;
        for (int i = 0; i < EMU_DEVICES; i++)
            opsRun_[i] = 0;
    }

    //! bytes moved between host and device memory
    uint64 transferred() const { return h2d_.load() + d2h_.load(); }

    void clearErrors()
    {
        std::lock_guard<std::mutex> lock(mem_);
        for (size_t i = 0; i < streams_.size(); i++)
        {
            std::shared_ptr<EmuStream> s = streams_[i].lock();
            if (s)
                s->clearError();
        }
    }

    int live(int d)
    {
        std::lock_guard<std::mutex> lock(mem_);
        return live_[d];
    }

    std::atomic<bool> throwNext{false};        // the next operation throws
    std::atomic<bool> failNextAsync{false};    // the next operation fails on the device, later
    std::atomic<int> violations_{0};
    std::atomic<int> streamsCreated_{0};
    std::atomic<uint64> h2d_{0}, d2h_{0}, d2d_{0};
    std::atomic<int> opsRun_[EMU_DEVICES];
    std::mutex mem_;    // held by every task while it touches device memory

private:
    struct Block
    {
        Block() : device(0), addr(0), size(0), base(0) {}
        int device;
        uintptr_t addr;
        size_t size;
        uchar* base;
        std::vector<uchar> storage;
    };

    Block* find(uintptr_t addr, size_t len)
    {
        std::map<uintptr_t, Block*>::iterator it = blocks_.upper_bound(addr);
        if (it == blocks_.begin())
            return 0;
        --it;
        Block* b = it->second;
        return (addr >= b->addr && addr + len <= b->addr + b->size) ? b : 0;
    }

    EmuCoreOps core_;
    std::map<uintptr_t, Block*> blocks_;
    uintptr_t nextFake_;
    int live_[EMU_DEVICES];
    std::vector<std::weak_ptr<EmuStream> > streams_;
};

bool EmuCoreOps::runnable(const UMat& src, const UMat& dst)
{
    if (src.depth() == CV_64F)
        return false;   // the emulated devices have no doubles: the CPU runs these calls
    if (b_->throwNext.exchange(false))
        CV_Error(Error::StsError, "emu: the operation failed");
    return src.dims == 2 && src.u->refcount == 0 && dst.u->refcount == 0 &&
           (src.u != dst.u || src.offset == dst.offset);
}

void EmuCoreOps::launch(const UMat& src, UMat& dst, accel::Stream& stream,
                        const std::function<void(const uchar*, uchar*)>& fn)
{
    EmuStream* s = &emuStreamOf(stream);
    b_->opsRun_[s->device_]++;
    const uintptr_t sa = (uintptr_t)src.handle(ACCESS_READ) + src.offset;
    const uintptr_t da = (uintptr_t)dst.handle(ACCESS_WRITE) + dst.offset;
    const size_t ss = src.step[0], ds = dst.step[0];
    const size_t sw = src.cols * src.elemSize(), dw = dst.cols * dst.elemSize();
    const int srows = src.rows, drows = dst.rows;
    const bool fail = b_->failNextAsync.exchange(false);
    EmuBackend* b = b_;
    s->enqueue([=] {
        if (fail)
        {
            s->fail("emu: the operation failed on the device");
            return;
        }
        std::lock_guard<std::mutex> lock(b->mem_);
        const uchar* sp = b->deviceBytes(s->device_, sa, ss, sw, srows);
        uchar* dp = b->deviceBytes(s->device_, da, ds, dw, drows);
        if (sp && dp)
            fn(sp, dp);
    });
    stream.retain({ src, dst });
}

bool EmuCoreOps::flip(const UMat& src, UMat& dst, int flipCode, accel::Stream& stream)
{
    if (!runnable(src, dst))
        return false;
    const size_t ss = src.step[0], ds = dst.step[0], esz = src.elemSize();
    const int rows = src.rows, cols = src.cols;
    launch(src, dst, stream, [=](const uchar* sp, uchar* dp) {
        // through a copy of the source, which makes an in-place call work too
        std::vector<uchar> tmp((size_t)rows * cols * esz);
        for (int y = 0; y < rows; y++)
            memcpy(&tmp[(size_t)y * cols * esz], sp + y * ss, cols * esz);
        const bool flipRows = flipCode <= 0, flipCols = flipCode != 0;
        for (int y = 0; y < rows; y++)
            for (int x = 0; x < cols; x++)
                memcpy(dp + y * ds + x * esz,
                       &tmp[((size_t)(flipRows ? rows - 1 - y : y) * cols + (flipCols ? cols - 1 - x : x)) * esz], esz);
    });
    return true;
}

bool EmuCoreOps::transpose(const UMat& src, UMat& dst, accel::Stream& stream)
{
    if (!runnable(src, dst))
        return false;
    const size_t ss = src.step[0], ds = dst.step[0], esz = src.elemSize();
    const int rows = src.rows, cols = src.cols;
    launch(src, dst, stream, [=](const uchar* sp, uchar* dp) {
        std::vector<uchar> tmp((size_t)rows * cols * esz);
        for (int y = 0; y < rows; y++)
            memcpy(&tmp[(size_t)y * cols * esz], sp + y * ss, cols * esz);
        for (int y = 0; y < rows; y++)
            for (int x = 0; x < cols; x++)
                memcpy(dp + x * ds + y * esz, &tmp[((size_t)y * cols + x) * esz], esz);
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
    }
    void TearDown() CV_OVERRIDE
    {
        if (sizeof(void*) < 8)
            return;
        g_delayUs = 0;
        for (int i = 0; i < EMU_DEVICES; i++)
            accel::synchronize(emuDevice(i));
        EXPECT_EQ(0, emu().violations_.load());
    }
};

TEST_F(Core_Accel, devices)
{
    const accel::Device d0 = emuDevice(EMU_DISCRETE), d1 = emuDevice(EMU_SHARED);
    ASSERT_FALSE(d0.empty());
    ASSERT_FALSE(d1.empty());
    EXPECT_EQ("emu:1", d1.name());
    EXPECT_EQ("emu", d1.backend());
    EXPECT_EQ(1, d1.index());
    EXPECT_EQ("emulated memory the CPU addresses", d1.description());
    EXPECT_EQ(d0, accel::Device::find("emu"));
    EXPECT_TRUE(accel::Device::find("emu:7").empty());
    EXPECT_TRUE(accel::Device::find("emu:x").empty());
    EXPECT_TRUE(accel::Device::find("nonexistent").empty());
    EXPECT_EQ("cpu", accel::Device().name());

    EXPECT_TRUE(d0.ops<accel::CoreOps>("core") != NULL);
    EXPECT_TRUE(d0.ops<accel::CoreOps>("imgproc") == NULL);
    EXPECT_TRUE(accel::Device().ops<accel::CoreOps>("core") == NULL);

    int found = 0;
    std::vector<accel::Device> all = accel::Device::enumerate();
    for (size_t i = 0; i < all.size(); i++)
        found += all[i].backend() == "emu";
    EXPECT_EQ(EMU_DEVICES, found);
}

TEST_F(Core_Accel, create_on_places_the_data)
{
    const accel::Device d0 = emuDevice(EMU_DISCRETE), d2 = emuDevice(EMU_DISCRETE2);
    UMat u;
    u.createOn(d0, Size(16, 8), CV_8UC3);
    EXPECT_EQ(d0, accel::getDevice(u));
    EXPECT_EQ(Size(16, 8), u.size());
    EXPECT_EQ(CV_8UC3, u.type());
    const UMatData* first = u.u;
    u.createOn(d0, MatShape({8, 16}), CV_8UC3);
    EXPECT_EQ(first, u.u) << "the same shape and type on the same device keep the buffer";
    u.createOn(d2, Size(16, 8), CV_8UC3);
    EXPECT_EQ(d2, accel::getDevice(u)) << "data on another device is released";
    u.createOn(accel::Device(), Size(16, 8), CV_8UC3);
    EXPECT_TRUE(accel::getDevice(u).empty());
    EXPECT_EQ(Size(16, 8), u.size());

    UMat nd;
    nd.createOn(d0, MatShape({3, 4, 5}), CV_32FC1);
    EXPECT_EQ(3, nd.dims);
    EXPECT_EQ(MatShape({3, 4, 5}), nd.shape());
    EXPECT_EQ(d0, accel::getDevice(nd));
}

TEST_F(Core_Accel, upload_and_read_back_on_every_device)
{
    const Mat src = randomMat(Size(53, 37), CV_8UC3);
    const uint64 bytes = src.total() * src.elemSize();
    for (int d = 0; d < EMU_DEVICES; d++)
    {
        emu().resetCounters();
        UMat u = src.upload(emuDevice(d));
        EXPECT_EQ(emuDevice(d), accel::getDevice(u));
        EXPECT_EQ(0, cvtest::norm(src, toHost(u), NORM_INF)) << "device " << d;
        if (d == EMU_SHARED)
            EXPECT_EQ(0u, emu().transferred()) << "the CPU uses the device's memory in place";
        else
        {
            EXPECT_EQ(bytes, emu().h2d_.load());
            EXPECT_EQ(bytes, emu().d2h_.load());
        }
    }
}

TEST_F(Core_Accel, mat_upload_has_read_the_matrix_when_it_returns)
{
    Mat src = randomMat(Size(32, 16), CV_8UC1);
    const Mat expected = src.clone();
    g_delayUs = 10000;
    UMat u = src.upload(emuDevice(EMU_DISCRETE));
    src.setTo(Scalar::all(0));
    src.release();
    EXPECT_EQ(0, cvtest::norm(expected, toHost(u), NORM_INF));
}

TEST_F(Core_Accel, umat_upload_is_shallow_on_the_same_device)
{
    const Mat src = randomMat(Size(20, 10), CV_16UC2);
    const accel::Device d0 = emuDevice(EMU_DISCRETE), d2 = emuDevice(EMU_DISCRETE2);
    UMat u = src.upload(d0);
    EXPECT_EQ(u.u, u.upload(d0).u);

    UMat other = u.upload(d2);
    EXPECT_EQ(d2, accel::getDevice(other));
    EXPECT_NE(u.u, other.u);
    EXPECT_EQ(0, cvtest::norm(src, toHost(other), NORM_INF));

    UMat back = other.upload(accel::Device());
    EXPECT_TRUE(accel::getDevice(back).empty());
    EXPECT_EQ(0, cvtest::norm(src, toHost(back), NORM_INF));
    EXPECT_EQ(back.u, back.upload(accel::Device()).u);
}

TEST_F(Core_Accel, flip_runs_on_the_device_of_its_input)
{
    const Mat src = randomMat(Size(31, 17), CV_8UC3);
    for (int d = 0; d < EMU_DEVICES; d++)
    {
        for (int code = -1; code <= 1; code++)
        {
            UMat u = src.upload(emuDevice(d)), dst;
            emu().resetCounters();
            cv::flip(u, dst, code);
            EXPECT_EQ(1, emu().opsRun_[d].load());
            EXPECT_EQ(0u, emu().transferred());
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
        UMat u = src.upload(emuDevice(d)), dst;
        emu().resetCounters();
        cv::transpose(u, dst);
        EXPECT_EQ(1, emu().opsRun_[d].load());
        EXPECT_EQ(emuDevice(d), accel::getDevice(dst));
        EXPECT_EQ(Size(13, 29), dst.size());
        EXPECT_EQ(0, cvtest::norm(src.t(), toHost(dst), NORM_INF)) << "device " << d;
    }
}

TEST_F(Core_Accel, declined_call_runs_on_the_cpu_and_its_output_stays_on_the_device)
{
    const Mat src = randomMat(Size(20, 10), CV_64FC1);  // the emulator declines doubles
    UMat u = src.upload(emuDevice(EMU_DISCRETE)), dst;
    emu().resetCounters();
    cv::flip(u, dst, 1);
    EXPECT_EQ(0, emu().opsRun_[EMU_DISCRETE].load());
    EXPECT_EQ(emuDevice(EMU_DISCRETE), accel::getDevice(dst));
    Mat expected;
    cv::flip(src, expected, 1);
    EXPECT_EQ(0, cvtest::norm(expected, toHost(dst), NORM_INF));
}

TEST_F(Core_Accel, cpu_fallback_does_not_upload_its_input_again)
{
    const Mat src = randomMat(Size(20, 10), CV_64FC1);
    const uint64 bytes = src.total() * src.elemSize();
    UMat u = src.upload(emuDevice(EMU_DISCRETE)), dst;
    emu().resetCounters();
    cv::flip(u, dst, 0);
    EXPECT_EQ(bytes, emu().d2h_.load()) << "the input is downloaded once";
    EXPECT_EQ(bytes, emu().h2d_.load()) << "only the output is uploaded; a new output is not downloaded";
}

TEST_F(Core_Accel, cpu_uses_memory_it_addresses_in_place)
{
    const Mat src = randomMat(Size(16, 8), CV_64FC1);
    UMat u = src.upload(emuDevice(EMU_SHARED));
    emu().resetCounters();
    {
        Mat view = u.getMat(ACCESS_READ);
        EXPECT_EQ((uchar*)u.u->handle + u.offset, view.data) << "the CPU reads the device's own bytes";
    }
    UMat dst;
    cv::flip(u, dst, 1);    // declined: the CPU runs it, on the same memory
    EXPECT_EQ(0u, emu().transferred());
    Mat expected;
    cv::flip(src, expected, 1);
    EXPECT_EQ(0, cvtest::norm(expected, toHost(dst), NORM_INF));
}

TEST_F(Core_Accel, in_place_calls_run_on_the_device)
{
    const int devices[] = { EMU_DISCRETE, EMU_SHARED };
    for (int i = 0; i < 2; i++)
    {
        const int d = devices[i];
        const Mat rect = randomMat(Size(9, 6), CV_8UC3), square = randomMat(Size(7, 7), CV_16UC1);
        UMat u = rect.upload(emuDevice(d));
        emu().resetCounters();
        cv::flip(u, u, 1);
        cv::flip(u, u, -1);
        Mat expected;
        cv::flip(rect, expected, 0);
        EXPECT_EQ(0, cvtest::norm(expected, toHost(u), NORM_INF)) << "device " << d;

        UMat v = square.upload(emuDevice(d));
        cv::transpose(v, v);
        EXPECT_EQ(0, cvtest::norm(square.t(), toHost(v), NORM_INF)) << "device " << d;

        cv::transpose(u, u);    // a new buffer: the shape changes
        EXPECT_EQ(Size(6, 9), u.size());
        EXPECT_EQ(emuDevice(d), accel::getDevice(u));
        EXPECT_EQ(0, cvtest::norm(expected.t(), toHost(u), NORM_INF)) << "device " << d;
        EXPECT_EQ(4, emu().opsRun_[d].load());
    }
}

TEST_F(Core_Accel, declined_in_place_calls_run_on_the_cpu)
{
    const Mat src = randomMat(Size(9, 4), CV_64FC1);
    UMat u = src.upload(emuDevice(EMU_DISCRETE));
    cv::flip(u, u, -1);
    cv::transpose(u, u);
    EXPECT_EQ(0, emu().opsRun_[EMU_DISCRETE].load());
    EXPECT_EQ(emuDevice(EMU_DISCRETE), accel::getDevice(u));
    Mat expected;
    cv::flip(src, expected, -1);
    EXPECT_EQ(0, cvtest::norm(expected.t(), toHost(u), NORM_INF));
}

TEST_F(Core_Accel, output_on_another_device_moves_to_the_input)
{
    const Mat src = randomMat(Size(10, 6), CV_16UC1);
    UMat u = src.upload(emuDevice(EMU_DISCRETE));
    UMat dst = Mat(Mat::zeros(6, 10, CV_16UC1)).upload(emuDevice(EMU_SHARED));
    cv::flip(u, dst, 1);
    EXPECT_EQ(emuDevice(EMU_DISCRETE), accel::getDevice(dst));
    EXPECT_EQ(1, emu().opsRun_[EMU_DISCRETE].load());
    Mat expected;
    cv::flip(src, expected, 1);
    EXPECT_EQ(0, cvtest::norm(expected, toHost(dst), NORM_INF));
}

TEST_F(Core_Accel, region_output_on_the_device_is_written_in_place)
{
    const Mat src = randomMat(Size(8, 6), CV_8UC2);
    UMat u = src.upload(emuDevice(EMU_DISCRETE));
    UMat big = Mat(Mat::zeros(20, 20, CV_8UC2)).upload(emuDevice(EMU_DISCRETE));
    cv::flip(u, big(Rect(4, 5, 8, 6)), 1);
    EXPECT_EQ(1, emu().opsRun_[EMU_DISCRETE].load());
    EXPECT_EQ(emuDevice(EMU_DISCRETE), accel::getDevice(big));
    Mat expected = Mat::zeros(20, 20, CV_8UC2);
    cv::flip(src, expected(Rect(4, 5, 8, 6)), 1);
    EXPECT_EQ(0, cvtest::norm(expected, toHost(big), NORM_INF));
}

TEST_F(Core_Accel, region_output_elsewhere_stays_where_it_is)
{
    const Mat src = randomMat(Size(8, 6), CV_8UC1);
    UMat u = src.upload(emuDevice(EMU_DISCRETE));
    UMat onShared = Mat(Mat::zeros(10, 12, CV_8UC1)).upload(emuDevice(EMU_SHARED));
    UMat onHost;
    Mat(Mat::zeros(10, 12, CV_8UC1)).copyTo(onHost);
    UMat bigs[] = { onShared, onHost };
    for (int i = 0; i < 2; i++)
    {
        const accel::Device where = accel::getDevice(bigs[i]);
        cv::flip(u, bigs[i](Rect(2, 3, 8, 6)), 0);
        UMat roi = bigs[i](Rect(1, 1, 8, 6));
        cv::flip(u, roi, 1);
        EXPECT_EQ(where, accel::getDevice(bigs[i])) << "a region cannot move";
        EXPECT_EQ(where, accel::getDevice(roi));
        Mat expected = Mat::zeros(10, 12, CV_8UC1);
        cv::flip(src, expected(Rect(2, 3, 8, 6)), 0);
        cv::flip(src, expected(Rect(1, 1, 8, 6)), 1);
        EXPECT_EQ(0, cvtest::norm(expected, toHost(bigs[i]), NORM_INF)) << i;
    }
    EXPECT_EQ(0, emu().opsRun_[EMU_DISCRETE].load());
}

TEST_F(Core_Accel, mat_output_is_written_by_the_cpu)
{
    const Mat src = randomMat(Size(11, 7), CV_32FC1);
    UMat u = src.upload(emuDevice(EMU_DISCRETE));
    Mat dst;
    cv::flip(u, dst, -1);
    Mat expected;
    cv::flip(src, expected, -1);
    EXPECT_EQ(0, cvtest::norm(expected, dst, NORM_INF));
}

TEST_F(Core_Accel, cpu_waits_for_the_work_of_its_thread)
{
    const Mat src = randomMat(Size(40, 30), CV_8UC4);
    Mat expected;
    cv::flip(src, expected, 1);
    for (int d = 0; d < 2; d++)
    {
        UMat u = src.upload(emuDevice(d)), dst;
        g_delayUs = 20000;
        cv::flip(u, dst, 1);
        EXPECT_EQ(0, cvtest::norm(expected, dst.getMat(ACCESS_READ), NORM_INF)) << "device " << d;
        g_delayUs = 0;
    }
}

TEST_F(Core_Accel, cpu_write_waits_for_device_reads)
{
    const Mat src = randomMat(Size(40, 30), CV_8UC1);
    Mat expected;
    cv::flip(src, expected, 0);
    for (int d = 0; d < 2; d++)
    {
        UMat u = src.upload(emuDevice(d)), dst;
        g_delayUs = 20000;
        cv::flip(u, dst, 0);    // reads u later, on the device
        u.setTo(Scalar(7));     // on the CPU, after that read
        g_delayUs = 0;
        EXPECT_EQ(0, cvtest::norm(expected, toHost(dst), NORM_INF)) << "device " << d;
        EXPECT_EQ(0, cvtest::norm(Mat(src.size(), CV_8UC1, Scalar(7)), toHost(u), NORM_INF)) << "device " << d;
    }
}

TEST_F(Core_Accel, released_buffers_outlive_the_device_work)
{
    const Mat src = randomMat(Size(64, 48), CV_8UC1);
    const accel::Device d = emuDevice(EMU_DISCRETE);
    accel::synchronize(d);
    const int before = emu().live(EMU_DISCRETE);
    UMat dst;
    g_delayUs = 20000;
    {
        UMat u = src.upload(d);
        cv::flip(u, dst, 1);
    }   // u is released while the flip still reads it
    EXPECT_EQ(before + 2, emu().live(EMU_DISCRETE)) << "the stream keeps u until the flip is done";
    accel::synchronize(d);
    g_delayUs = 0;
    EXPECT_EQ(before + 1, emu().live(EMU_DISCRETE)) << "only dst is left";
    Mat expected;
    cv::flip(src, expected, 1);
    EXPECT_EQ(0, cvtest::norm(expected, toHost(dst), NORM_INF));
}

TEST_F(Core_Accel, device_error_surfaces_when_the_cpu_waits)
{
    const accel::Device d = emuDevice(EMU_DISCRETE);
    UMat u = randomMat(Size(8, 8), CV_8UC1).upload(d), dst;
    emu().failNextAsync = true;
    cv::flip(u, dst, 1);    // queued: the failure happens later, on the device
    EXPECT_THROW(toHost(dst), cv::Exception);
    EXPECT_THROW(accel::synchronize(d), cv::Exception) << "the error is sticky";
    emu().clearErrors();
}

TEST_F(Core_Accel, operation_errors_propagate)
{
    UMat u = randomMat(Size(8, 8), CV_8UC1).upload(emuDevice(EMU_DISCRETE)), dst;
    emu().throwNext = true;
    EXPECT_THROW(cv::flip(u, dst, 1), cv::Exception);
}

TEST_F(Core_Accel, copies_between_umats)
{
    const Mat src = randomMat(Size(15, 11), CV_8UC3);
    const uint64 bytes = src.total() * src.elemSize();
    UMat a = src.upload(emuDevice(EMU_DISCRETE));

    UMat same;
    same.createOn(emuDevice(EMU_DISCRETE), src.size(), src.type());
    emu().resetCounters();
    a.copyTo(same);
    EXPECT_EQ(0u, emu().transferred()) << "within one device";
    EXPECT_EQ(bytes, emu().d2d_.load());

    UMat other;
    other.createOn(emuDevice(EMU_DISCRETE2), src.size(), src.type());
    emu().resetCounters();
    a.copyTo(other);    // through the host
    EXPECT_EQ(bytes, emu().d2h_.load());
    EXPECT_EQ(bytes, emu().h2d_.load());
    EXPECT_EQ(emuDevice(EMU_DISCRETE2), accel::getDevice(other));
    EXPECT_EQ(0, cvtest::norm(src, toHost(same), NORM_INF));
    EXPECT_EQ(0, cvtest::norm(src, toHost(other), NORM_INF));
}

TEST_F(Core_Accel, cpu_read_is_not_uploaded_back)
{
    const Mat src = randomMat(Size(16, 16), CV_8UC1);
    UMat u = src.upload(emuDevice(EMU_DISCRETE));
    emu().resetCounters();
    {
        Mat a = u.getMat(ACCESS_READ), b = u.getMat(ACCESS_READ);
        EXPECT_EQ(0, cvtest::norm(src, a, NORM_INF));
    }
    EXPECT_EQ(src.total(), emu().d2h_.load());
    EXPECT_EQ(0u, emu().h2d_.load());
    {
        Mat r = u.getMat(ACCESS_READ), w = u.getMat(ACCESS_WRITE);
        w.setTo(Scalar(3));
    }
    EXPECT_EQ(src.total(), emu().h2d_.load()) << "a write view is uploaded when the views are released";
    EXPECT_EQ(0, cvtest::norm(Mat(src.size(), CV_8UC1, Scalar(3)), toHost(u), NORM_INF));
}

TEST_F(Core_Accel, opencl_declines_device_memory)
{
    if (!cv::ocl::haveOpenCL())
        throw SkipTestException("OpenCL is not available");
    const bool prev = cv::ocl::useOpenCL();
    cv::ocl::setUseOpenCL(true);
    const Mat src = randomMat(Size(20, 10), CV_64FC1);  // declined by the emulator: the call reaches the OpenCL branch
    UMat u = src.upload(emuDevice(EMU_DISCRETE)), dst;
#ifdef HAVE_OPENCL
    EXPECT_FALSE(cv::ocl::internal::isCLBuffer(u));    // defined only in OpenCL builds
#endif
    cv::flip(u, dst, 1);
    cv::ocl::setUseOpenCL(prev);
    Mat expected;
    cv::flip(src, expected, 1);
    EXPECT_EQ(0, cvtest::norm(expected, toHost(dst), NORM_INF));
}

TEST_F(Core_Accel, each_thread_has_its_own_stream)
{
    const accel::Device d = emuDevice(EMU_DISCRETE);
    accel::Stream* mine = &d.threadStream();
    EXPECT_EQ(mine, &d.threadStream());
    EXPECT_NE(mine, &emuDevice(EMU_DISCRETE2).threadStream());
    accel::Stream* other = 0;
    std::thread([&] { other = &d.threadStream(); }).join();
    EXPECT_NE(mine, other);
    EXPECT_THROW(accel::Device().threadStream(), cv::Exception);
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
                UMat u = src.upload(d), a, b;
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

TEST_F(Core_Accel, synchronize_hands_the_data_to_another_thread)
{
    const accel::Device d = emuDevice(EMU_DISCRETE);
    const Mat src = randomMat(Size(32, 24), CV_8UC1);
    UMat u = src.upload(d), dst;
    std::mutex m;
    std::condition_variable cv_;
    bool queued = false, read = false;
    std::thread producer([&] {
        g_delayUs = 20000;
        cv::flip(u, dst, 1);
        accel::synchronize(d);      // without it, the reader's stream would not wait for this one
        g_delayUs = 0;
        {
            std::lock_guard<std::mutex> lock(m);
            queued = true;
        }
        cv_.notify_all();
        std::unique_lock<std::mutex> lock(m);
        cv_.wait(lock, [&] { return read; });   // alive until the reader is done
    });
    Mat got;
    {
        std::unique_lock<std::mutex> lock(m);
        cv_.wait(lock, [&] { return queued; });
    }
    got = toHost(dst);
    {
        std::lock_guard<std::mutex> lock(m);
        read = true;
    }
    cv_.notify_all();
    producer.join();
    Mat expected;
    cv::flip(src, expected, 1);
    EXPECT_EQ(0, cvtest::norm(expected, got, NORM_INF));
}

TEST_F(Core_Accel, thread_exit_finishes_its_work_and_releases_its_buffers)
{
    const accel::Device d = emuDevice(EMU_DISCRETE);
    const Mat src = randomMat(Size(32, 24), CV_8UC1);
    accel::synchronize(d);
    const int before = emu().live(EMU_DISCRETE);
    UMat result;
    std::thread([&] {
        g_delayUs = 20000;
        UMat u = src.upload(d), dst;
        cv::flip(u, dst, 0);
        result = dst;
    }).join();
    g_delayUs = 0;
    EXPECT_EQ(before + 1, emu().live(EMU_DISCRETE)) << "u is released once the flip is done";
    Mat expected;
    cv::flip(src, expected, 0);
    EXPECT_EQ(0, cvtest::norm(expected, toHost(result), NORM_INF));
}

TEST_F(Core_Accel, mat_calls_are_not_affected)
{
    UMat onDevice = Mat(Mat::ones(2, 2, CV_8UC1)).upload(emuDevice(EMU_DISCRETE));
    const Mat src = randomMat(Size(10, 10), CV_8UC1);
    Mat dst;
    emu().resetCounters();
    cv::flip(src, dst, 1);
    cv::transpose(src, dst);
    for (int d = 0; d < EMU_DEVICES; d++)
        EXPECT_EQ(0, emu().opsRun_[d].load());
    EXPECT_EQ(0u, emu().transferred());
}

#ifdef HAVE_CUDA

accel::Device cudaDevice()
{
    accel::Device d = accel::Device::find("cuda:0");
    if (d.empty())
        throw SkipTestException("no CUDA device");
    return d;
}

//! a kernel wrote @p m (UMat::handle(ACCESS_WRITE)); the CPU path leaves the host copy current
bool writtenByDevice(const UMat& m)
{
    return m.u->hostCopyObsolete();
}

TEST(Core_Accel_CUDA, flip_and_transpose)
{
    const accel::Device d = cudaDevice();
    const int types[] = { CV_8UC1, CV_8UC3, CV_16UC2, CV_32FC1, CV_32FC4, CV_64FC3 };
    for (size_t t = 0; t < sizeof(types) / sizeof(types[0]); t++)
    {
        const Mat src = randomMat(Size(1923, 1081), types[t]);
        UMat u = src.upload(d), f, tr;
        cv::flip(u, f, -1);
        cv::transpose(u, tr);
        EXPECT_EQ(d, accel::getDevice(f));
        EXPECT_EQ(d, accel::getDevice(tr));
        EXPECT_TRUE(writtenByDevice(f));
        EXPECT_TRUE(writtenByDevice(tr));
        Mat ef, et;
        cv::flip(src, ef, -1);
        cv::transpose(src, et);
        EXPECT_EQ(0, cvtest::norm(ef, toHost(f), NORM_INF)) << typeToString(types[t]);
        EXPECT_EQ(0, cvtest::norm(et, toHost(tr), NORM_INF)) << typeToString(types[t]);
    }
}

TEST(Core_Accel_CUDA, in_place)
{
    const accel::Device d = cudaDevice();
    const int types[] = { CV_8UC1, CV_8UC3, CV_32FC1, CV_64FC3 };
    for (size_t t = 0; t < sizeof(types) / sizeof(types[0]); t++)
    {
        const Mat rect = randomMat(Size(333, 117), types[t]), square = randomMat(Size(101, 101), types[t]);
        for (int code = -1; code <= 1; code++)
        {
            UMat u = rect.upload(d);
            cv::flip(u, u, code);
            EXPECT_TRUE(writtenByDevice(u));
            Mat expected;
            cv::flip(rect, expected, code);
            EXPECT_EQ(0, cvtest::norm(expected, toHost(u), NORM_INF)) << typeToString(types[t]) << ", code " << code;
        }
        UMat v = square.upload(d), w = rect.upload(d);
        cv::transpose(v, v);
        cv::transpose(w, w);
        EXPECT_EQ(d, accel::getDevice(w));
        EXPECT_TRUE(writtenByDevice(v));
        EXPECT_TRUE(writtenByDevice(w));
        EXPECT_EQ(0, cvtest::norm(square.t(), toHost(v), NORM_INF)) << typeToString(types[t]);
        EXPECT_EQ(0, cvtest::norm(rect.t(), toHost(w), NORM_INF)) << typeToString(types[t]);
    }
}

TEST(Core_Accel_CUDA, unsupported_element_size_falls_back)
{
    const accel::Device d = cudaDevice();
    const Mat src = randomMat(Size(33, 21), CV_8UC(5));    // 5-byte elements: no kernel
    UMat u = src.upload(d), dst;
    cv::flip(u, dst, 1);
    EXPECT_EQ(d, accel::getDevice(dst));
    EXPECT_FALSE(writtenByDevice(dst));
    Mat expected;
    cv::flip(src, expected, 1);
    EXPECT_EQ(0, cvtest::norm(expected, toHost(dst), NORM_INF));
}

TEST(Core_Accel_CUDA, region_and_chain)
{
    const accel::Device d = cudaDevice();
    const Mat src = randomMat(Size(640, 480), CV_8UC3);
    UMat u = src.upload(d), a, b;
    UMat big = Mat(Mat::zeros(500, 700, CV_8UC3)).upload(d);
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
