// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

// Non-CPU HAL: backends, plugins and devices, streams, ExecutionScope, preferred devices,
// counters and the transfer policy.

#include "../precomp.hpp"
#include "accel_impl.hpp"
#include "opencv2/core/utils/configuration.private.hpp"
#include "opencv2/core/utils/filesystem.hpp"
#include "opencv2/core/utils/filesystem.private.hpp"
#include "opencv2/core/utils/logger.hpp"

#if OPENCV_HAVE_FILESYSTEM_SUPPORT
#include "opencv2/core/utils/plugin_loader.private.hpp"
#endif

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
void* Backend::importDmaBuf(int, int, size_t) { return 0; }
int Backend::exportDmaBuf(int, void*, size_t) { return -1; }
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

const Ptr<Stream>& Device::Impl::defaultStream()
{
    std::lock_guard<std::mutex> lock(streamMutex_);
    if (!stream_)
    {
        stream_ = backend->createStream(index);
        CV_Assert(stream_ && "accel: the backend returned no stream");
    }
    return stream_;
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

    //! built-in backends and plugins found on disk, once, before anything else is registered
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
            if (utils::getConfigurationParameterBool("OPENCV_ACCEL_PLUGINS_ENABLE", true))
            {
                std::vector<std::string> plugins = pluginCandidates();
                for (size_t i = 0; i < plugins.size(); i++)
                    loadPluginFile(plugins[i]);
            }
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

    bool loadPluginFile(const std::string& path)
    {
#if OPENCV_HAVE_FILESYSTEM_SUPPORT
        using namespace cv::plugin::impl;
        std::unique_ptr<DynamicLib> lib(new DynamicLib(toFileSystemPath(path)));
        if (!lib->isLoaded())
        {
            CV_LOG_WARNING(NULL, "accel: cannot load the plugin " << path);
            return false;
        }
        FN_opencv_core_accel_plugin_init_v1 init =
            reinterpret_cast<FN_opencv_core_accel_plugin_init_v1>(lib->getSymbol(CV_ACCEL_PLUGIN_INIT_NAME));
        if (!init)
        {
            CV_LOG_WARNING(NULL, "accel: " << path << " is not a backend plugin (no " CV_ACCEL_PLUGIN_INIT_NAME ")");
            return false;
        }
        Backend* raw = init(CV_VERSION_MAJOR, CV_VERSION_MINOR, CV_VERSION_REVISION);
        if (!raw)
        {
            CV_LOG_WARNING(NULL, "accel: the plugin " << path << " was built for another OpenCV version or failed to start");
            return false;
        }
        Ptr<Backend> backend(raw);
        // the backend's code lives in the library for as long as the backend: for the whole process
        lib->disableAutomaticLibraryUnloading();
        lib.release();
        try
        {
            add(backend);
        }
        catch (const cv::Exception& e)
        {
            CV_LOG_WARNING(NULL, "accel: the plugin " << path << " was not registered: " << e.what());
            return false;
        }
        CV_LOG_INFO(NULL, "accel: loaded the plugin " << path << " (" << backend->name() << ", "
                          << backend->deviceCount() << " device(s))");
        return true;
#else
        CV_UNUSED(path);
        return false;
#endif
    }

private:
    Registry() {}

    //! OPENCV_ACCEL_PLUGINS, otherwise libopencv_core_accel_*.so in the plugin directories
    static std::vector<std::string> pluginCandidates()
    {
        std::vector<std::string> result;
#if OPENCV_HAVE_FILESYSTEM_SUPPORT
        const std::vector<std::string> explicitFiles =
            utils::getConfigurationParameterPaths("OPENCV_ACCEL_PLUGINS", std::vector<std::string>());
        if (!explicitFiles.empty())
            return explicitFiles;
        std::vector<std::string> dirs =
            utils::getConfigurationParameterPaths("OPENCV_CORE_PLUGIN_PATH", std::vector<std::string>());
        if (dirs.empty())
        {
            std::string binary;
            if (utils::getBinLocation(binary))
                dirs.push_back(utils::fs::getParent(binary));
        }
        const std::string pattern = plugin::impl::libraryPrefix() + "opencv_core_accel_*" + plugin::impl::librarySuffix();
        for (size_t i = 0; i < dirs.size(); i++)
        {
            if (dirs[i].empty())
                continue;
            std::vector<std::string> found;
            try
            {
                cv::glob(utils::fs::join(dirs[i], pattern), found, false);
            }
            catch (const cv::Exception&)
            {
                continue;
            }
            std::sort(found.begin(), found.end());
            result.insert(result.end(), found.begin(), found.end());
        }
#endif
        return result;
    }

    std::once_flag builtinsOnce_;
    std::mutex mutex_;
    std::vector<Ptr<Backend> > backends_;
    std::vector<Device::Impl*> devices_;
};

struct ScopeTLS
{
    bool active;
    Device::Impl* dev;
    const Ptr<Stream>* stream;  // the ExecutionScope's own member, alive as long as the scope
};
thread_local ScopeTLS tlsScope = { false, 0, 0 };

std::mutex& preferredMutex()
{
    static std::mutex* m = new std::mutex();
    return *m;
}

std::vector<Device::Impl*>& preferredList()
{
    static std::vector<Device::Impl*>* l = new std::vector<Device::Impl*>();
    return *l;
}

std::atomic<bool> g_havePreferred(false);

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

bool loadPlugin(const String& path)
{
    Registry& r = Registry::get();
    r.loadBuiltins();
    return r.loadPluginFile(path);
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

Ptr<Stream> createStream(const Device& device)
{
    CV_Assert(!device.empty() && "accel: streams belong to a device");
    Ptr<Stream> s = device.impl()->backend->createStream(device.index());
    CV_Assert(s && "accel: the backend returned no stream");
    return s;
}

ExecutionScope::ExecutionScope(const Device& device)
    : prevActive_(tlsScope.active), prev_(tlsScope.dev), prevStream_(tlsScope.stream)
{
    tlsScope.active = true;
    tlsScope.dev = device.impl();
    tlsScope.stream = 0;
}

ExecutionScope::ExecutionScope(const Device& device, const Ptr<Stream>& stream)
    : prevActive_(tlsScope.active), prev_(tlsScope.dev), prevStream_(tlsScope.stream), stream_(stream)
{
    CV_Assert(!device.empty() && stream && "accel: a stream needs its device");
    tlsScope.active = true;
    tlsScope.dev = device.impl();
    tlsScope.stream = &stream_;
}

ExecutionScope::~ExecutionScope()
{
    tlsScope.active = prevActive_;
    tlsScope.dev = prev_.impl();
    tlsScope.stream = prevStream_;
}

void synchronize(const Device& device)
{
    if (device.empty())
        return;
    device.impl()->defaultStream()->synchronize();
    detail::collectRetired(false);
}

void setPreferredDevices(const std::vector<Device>& devices)
{
    std::vector<Device::Impl*> list;
    for (size_t i = 0; i < devices.size(); i++)
    {
        CV_Assert(!devices[i].empty() && "accel: the CPU cannot be a preferred device");
        list.push_back(devices[i].impl());
    }
    std::lock_guard<std::mutex> lock(preferredMutex());
    preferredList().swap(list);
    g_havePreferred = !preferredList().empty();
}

std::vector<Device> getPreferredDevices()
{
    std::lock_guard<std::mutex> lock(preferredMutex());
    std::vector<Device> result;
    for (size_t i = 0; i < preferredList().size(); i++)
        result.push_back(Device(preferredList()[i]));
    return result;
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

const Ptr<Stream>& currentStream(Device::Impl* dev)
{
    if (tlsScope.active && tlsScope.dev == dev && tlsScope.stream)
        return *tlsScope.stream;
    return dev->defaultStream();
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

std::vector<Device::Impl*> preferredDevices()
{
    if (!g_havePreferred.load(std::memory_order_relaxed))
        return std::vector<Device::Impl*>();
    std::lock_guard<std::mutex> lock(preferredMutex());
    return preferredList();
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
