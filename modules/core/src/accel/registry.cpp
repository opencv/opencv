// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

// Non-CPU HAL: backends and their devices, ExecutionScope, counters and the transfer policy.

#include "../precomp.hpp"
#include "accel_impl.hpp"
#include "opencv2/core/utils/configuration.private.hpp"
#include "opencv2/core/utils/logger.hpp"

namespace cv { namespace accel {

// defaults of the backend interface

Event::~Event() {}
Stream::~Stream() {}
Ops::~Ops() {}
Backend::~Backend() {}

bool CoreOps::flip(const UMat&, UMat&, int, Stream&) { return false; }
bool CoreOps::transpose(const UMat&, UMat&, Stream&) { return false; }

void* Backend::hostAddress(int, void* ptr) { return ptr; }
void Backend::beginCpuAccess(int, void*, size_t, bool) {}
void Backend::endCpuAccess(int, void*, size_t, bool) {}
Ops* Backend::ops(const String&) { return 0; }

void* Backend::allocateHost(int, size_t size)
{
    try
    {
        return fastMalloc(size);
    }
    catch (const cv::Exception&)
    {
        return 0;
    }
}

void Backend::deallocateHost(int, void* ptr)
{
    fastFree(ptr);
}

Device::Impl::Impl(const Ptr<Backend>& b, int i)
    : backend(b), index(i), name(cv::format("%s:%d", b->name().c_str(), i)),
      shared(b->sharesHostMemory(i)), allocator(0)
{
    allocator = new detail::DeviceAllocator(this);
}

Stream& Device::Impl::defaultStream()
{
    std::lock_guard<std::mutex> lock(streamMutex_);
    if (!stream_)
    {
        stream_ = backend->createStream(index);
        CV_Assert(stream_ && "accel: the backend returned no stream");
    }
    return *stream_;
}

namespace {

/* Backends and devices registered in this process. Nothing is ever removed, so Device::Impl
   pointers stay valid; the registry, its devices and their default streams live until the
   process exits (destroying them at exit would race with the backends' own teardown). */
class Registry
{
public:
    static Registry& get()
    {
        static Registry* r = new Registry();
        return *r;
    }

    void loadBuiltins()
    {
        std::call_once(builtinsOnce_, [this]() {
#ifdef HAVE_CUDA
            if (utils::getConfigurationParameterBool("OPENCV_ACCEL_CUDA", true))
            {
                try
                {
                    Ptr<Backend> cuda = impl::createCudaBackend();
                    if (cuda && cuda->deviceCount() > 0)
                        add(cuda);
                }
                catch (const cv::Exception& e)
                {
                    CV_LOG_WARNING(NULL, "accel: the CUDA backend is not available: " << e.what());
                }
            }
#endif
        });
    }

    void add(const Ptr<Backend>& b)
    {
        CV_Assert(b);
        const String name = b->name();
        CV_Assert(!name.empty());
        const int n = b->deviceCount();
        CV_Assert(n >= 0);
        std::vector<Device::Impl*> created;    // outside the lock: this calls into the backend
        for (int i = 0; i < n; i++)
            created.push_back(new Device::Impl(b, i));
        std::lock_guard<std::mutex> lock(mutex_);
        for (size_t i = 0; i < backends_.size(); i++)
        {
            if (backends_[i]->name() == name)
            {
                for (size_t j = 0; j < created.size(); j++)
                {
                    delete created[j]->allocator;
                    delete created[j];
                }
                CV_Error_(Error::StsBadArg, ("accel: a backend named '%s' is already registered", name.c_str()));
            }
        }
        backends_.push_back(b);
        devices_.insert(devices_.end(), created.begin(), created.end());
    }

    std::vector<Device::Impl*> devices()
    {
        loadBuiltins();
        std::lock_guard<std::mutex> lock(mutex_);
        return devices_;
    }

private:
    Registry() {}
    std::once_flag builtinsOnce_;
    std::mutex mutex_;
    std::vector<Ptr<Backend> > backends_;
    std::vector<Device::Impl*> devices_;
};

struct ScopeTLS
{
    bool active;
    Device::Impl* dev;
};
thread_local ScopeTLS tlsScope = { false, 0 };

struct Counters
{
    std::atomic<uint64> h2d, d2h, d2d, transfers, ops, fallbacks, waits;
    Counters() : h2d(0), d2h(0), d2d(0), transfers(0), ops(0), fallbacks(0), waits(0) {}
};

Counters& counters()
{
    static Counters* c = new Counters();
    return *c;
}

std::atomic<int>& policyValue()
{
    static std::atomic<int>* p = new std::atomic<int>(
        utils::getConfigurationParameterString("OPENCV_ACCEL_TRANSFER_POLICY", "allow") == "forbid"
            ? TRANSFER_FORBID : TRANSFER_ALLOW);
    return *p;
}

} // namespace

void registerBackend(const Ptr<Backend>& backend)
{
    Registry& r = Registry::get();
    r.loadBuiltins();   // built-in backends come first in Device::enumerate()
    r.add(backend);
}

std::vector<Device> Device::enumerate()
{
    std::vector<Device::Impl*> all = Registry::get().devices();
    std::vector<Device> result;
    for (size_t i = 0; i < all.size(); i++)
        result.push_back(Device(all[i]));
    return result;
}

Device Device::find(const String& spec)
{
    String base = spec;
    int index = -1;
    const size_t colon = spec.find(':');
    if (colon != String::npos)
    {
        base = spec.substr(0, colon);
        const String num = spec.substr(colon + 1);
        if (num.empty() || num.find_first_not_of("0123456789") != String::npos)
            return Device();
        index = atoi(num.c_str());
    }
    std::vector<Device::Impl*> all = Registry::get().devices();
    for (size_t i = 0; i < all.size(); i++)
        if (all[i]->backend->name() == base && (index < 0 || all[i]->index == index))
            return Device(all[i]);
    return Device();
}

String Device::backend() const { return impl_ ? impl_->backend->name() : String(); }
int Device::index() const { return impl_ ? impl_->index : -1; }
String Device::name() const { return impl_ ? impl_->name : String("cpu"); }
String Device::description() const { return impl_ ? impl_->backend->deviceDescription(impl_->index) : String("CPU"); }
bool Device::sharesHostMemory() const { return impl_ ? impl_->shared : true; }

MatAllocator* Device::allocator() const
{
    return impl_ ? (MatAllocator*)impl_->allocator : Mat::getDefaultAllocator();
}

Device getDevice(const UMat& m)
{
    detail::BufferState* st = detail::stateOf(m.u);
    return st ? Device(st->dev) : Device();
}

ExecutionScope::ExecutionScope(const Device& device)
    : prevActive_(tlsScope.active), prev_(tlsScope.dev)
{
    tlsScope.active = true;
    tlsScope.dev = device.impl();
}

ExecutionScope::~ExecutionScope()
{
    tlsScope.active = prevActive_;
    tlsScope.dev = prev_.impl();
}

void synchronize(const Device& device)
{
    if (device.empty())
        return;
    device.impl()->defaultStream().synchronize();
    detail::collectRetired(false);
}

void setTransferPolicy(TransferPolicy policy)
{
    CV_Assert(policy == TRANSFER_ALLOW || policy == TRANSFER_FORBID);
    policyValue().store(policy);
}

TransferPolicy getTransferPolicy()
{
    return (TransferPolicy)policyValue().load();
}

TransferStats getTransferStats()
{
    Counters& c = counters();
    TransferStats s;
    s.bytesHostToDevice = c.h2d.load();
    s.bytesDeviceToHost = c.d2h.load();
    s.bytesDeviceToDevice = c.d2d.load();
    s.transfers = c.transfers.load();
    s.deviceOps = c.ops.load();
    s.cpuFallbacks = c.fallbacks.load();
    s.hostWaits = c.waits.load();
    return s;
}

void resetTransferStats()
{
    Counters& c = counters();
    c.h2d = 0; c.d2h = 0; c.d2d = 0; c.transfers = 0;
    c.ops = 0; c.fallbacks = 0; c.waits = 0;
}

namespace detail {

bool isActive()
{
    return g_liveBuffers.load(std::memory_order_relaxed) > 0 || tlsScope.active;
}

MatAllocator* scopeAllocator()
{
    return (tlsScope.active && tlsScope.dev) ? (MatAllocator*)tlsScope.dev->allocator : 0;
}

Device::Impl* scopeDevice(bool& active)
{
    active = tlsScope.active;
    return tlsScope.dev;
}

void countCopy(CopyKind kind, size_t bytes)
{
    Counters& c = counters();
    switch (kind)
    {
    case COPY_HOST_TO_DEVICE: c.h2d += bytes; c.transfers++; break;
    case COPY_DEVICE_TO_HOST: c.d2h += bytes; c.transfers++; break;
    case COPY_DEVICE_TO_DEVICE: c.d2d += bytes; break;
    }
}

void countDeviceOp() { counters().ops++; }
void countFallback() { counters().fallbacks++; }
void countHostWait() { counters().waits++; }

} // namespace detail

}} // namespace cv::accel
