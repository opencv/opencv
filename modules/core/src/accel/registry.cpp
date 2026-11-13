// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

// Non-CPU HAL: backends and devices, the per-thread streams, and the buffers a stream keeps alive.

#include "../precomp.hpp"
#include "accel_impl.hpp"
#include "opencv2/core/utils/logger.hpp"
#include "opencv2/core/utils/tls.hpp"

#include <deque>

namespace cv { namespace accel {

Event::~Event() {}
Ops::~Ops() {}
Backend::~Backend() {}

bool CoreOps::flip(const UMat&, UMat&, int, Stream&) { return false; }
bool CoreOps::transpose(const UMat&, UMat&, Stream&) { return false; }

void* Backend::hostAddress(int, void*) { return 0; }
Ops* Backend::ops(const char*) { return 0; }

struct Stream::Impl
{
    cv::Mutex mutex;
    std::deque<std::pair<Ptr<Event>, std::vector<UMat> > > pending;    // in the order of the work
};

namespace {

//! failed work counts as finished: the buffers it used can be released
bool finished(const Ptr<Event>& e)
{
    try
    {
        return e->query();
    }
    catch (const cv::Exception& ex)
    {
        CV_LOG_WARNING(NULL, "accel: device work failed: " << ex.what());
        return true;
    }
}

} // namespace

Stream::Stream() : p_(new Impl()) {}

Stream::~Stream()
{
    for (size_t i = 0; i < p_->pending.size(); i++)
    {
        try
        {
            p_->pending[i].first->wait();
        }
        catch (const cv::Exception& ex)
        {
            CV_LOG_WARNING(NULL, "accel: device work failed: " << ex.what());
        }
    }
    delete p_;
}

void Stream::retain(const std::vector<UMat>& arrays)
{
    Ptr<Event> e = record();
    releaseFinished();
    cv::AutoLock lock(p_->mutex);
    p_->pending.push_back(std::make_pair(e, arrays));
}

void Stream::releaseFinished()
{
    std::vector<std::vector<UMat> > done;   // released after the lock: that may free device memory
    cv::AutoLock lock(p_->mutex);
    while (!p_->pending.empty() && finished(p_->pending.front().first))
    {
        done.push_back(std::vector<UMat>());
        done.back().swap(p_->pending.front().second);
        p_->pending.pop_front();
    }
}

Device::Impl::Impl(const Ptr<Backend>& b, int i, int id_, detail::BackendAllocator* a)
    : backend(b), index(i), id(id_), name(cv::format("%s:%d", b->name().c_str(), i)), allocator(a)
{}

namespace {

// never shrinks, so Device::Impl pointers stay valid; leaked at exit, not to race backend teardown
class Registry
{
public:
    static Registry& get()
    {
        static Registry* r = new Registry();
        return *r;
    }

    //! the built-in backends, once, before anything else is registered
    void loadBuiltins()
    {
        static const bool loaded = addBuiltins();
        CV_UNUSED(loaded);
    }

    void add(const Ptr<Backend>& b)
    {
        CV_Assert(b);
        const String name = b->name();
        CV_Assert(!name.empty());
        const int n = b->deviceCount();
        CV_Assert(n >= 0);
        cv::AutoLock lock(mutex_);
        for (size_t i = 0; i < backends_.size(); i++)
            if (backends_[i]->name() == name)
                CV_Error_(Error::StsBadArg, ("accel: a backend named '%s' is already registered", name.c_str()));
        backends_.push_back(b);
        detail::BackendAllocator* allocator = new detail::BackendAllocator();
        for (int i = 0; i < n; i++)
            devices_.push_back(new Device::Impl(b, i, (int)devices_.size(), allocator));
    }

    std::vector<Device::Impl*> devices()
    {
        loadBuiltins();
        cv::AutoLock lock(mutex_);
        return devices_;
    }

private:
    Registry() {}

    bool addBuiltins()
    {
#ifdef HAVE_CUDA
        try
        {
            Ptr<Backend> cuda = cuda_impl::createCudaBackend();
            if (cuda && cuda->deviceCount() > 0)
                add(cuda);
        }
        catch (const cv::Exception& e)
        {
            CV_LOG_WARNING(NULL, "accel: the CUDA backend is not available: " << e.what());
        }
#endif
        return true;
    }

    cv::Mutex mutex_;
    std::vector<Ptr<Backend> > backends_;
    std::vector<Device::Impl*> devices_;
};

// streams of one thread, by Device::Impl::id; destroyed with the thread, after their work
struct ThreadStreams
{
    std::vector<Ptr<Stream> > streams;
};

TLSData<ThreadStreams>& threadStreams()
{
    CV_SINGLETON_LAZY_INIT_REF(TLSData<ThreadStreams>, new TLSData<ThreadStreams>())
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

Stream& Device::threadStream() const
{
    CV_Assert(impl_ && "accel: the CPU has no stream");
    return detail::threadStream(impl_);
}

Ops* Device::opsImpl(const char* module) const
{
    return impl_ ? impl_->backend->ops(module) : 0;
}

Device getDevice(const UMat& m)
{
    if (!m.u || !detail::isBackendAllocator(m.u->currAllocator))
        return Device();
    return Device(static_cast<Device::Impl*>(m.u->allocatorContext.get()));
}

void synchronize(const Device& device)
{
    if (device.empty())
        return;
    Stream& s = device.threadStream();
    s.synchronize();
    s.releaseFinished();
}

namespace detail {

Stream& threadStream(Device::Impl* dev)
{
    std::vector<Ptr<Stream> >& streams = threadStreams().getRef().streams;
    if ((int)streams.size() <= dev->id)
        streams.resize(dev->id + 1);
    Ptr<Stream>& s = streams[dev->id];
    if (!s)
    {
        s = dev->backend->createStream(dev->index);
        CV_Assert(s && "accel: the backend returned no stream");
    }
    return *s;
}

} // namespace detail

}} // namespace cv::accel
