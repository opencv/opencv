// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "precomp.hpp"
#include "umatrix.hpp"
#include <opencv2/core/utils/logger.hpp>

namespace cv {

struct ComputeDevice::Impl
{
    String backend;
    int index;
    MatAllocator* allocator;
};

namespace {

/** @brief The non-CPU devices of this build present on this machine, found once per process. */
struct DeviceRegistry
{
    DeviceRegistry()
    {
        if (ocl::haveOpenCL() && ocl::getOpenCLAllocator())
            add("opencl", ocl::getOpenCLAllocator());
        // one entry: the CUDA allocator uses the thread's current CUDA device
        if (cuda::getCudaEnabledDeviceCount() > 0 && cuda::getCudaAllocator())
            add("cuda", cuda::getCudaAllocator());
    }

    /** @brief Registers device 0 of @p backend, whose UMat buffers @p allocator creates. */
    void add(const char* backend, MatAllocator* allocator)
    {
        ComputeDevice::Impl* d = new ComputeDevice::Impl();
        d->backend = backend;
        d->index = 0;
        d->allocator = allocator;
        devices.push_back(d);
    }

    std::vector<ComputeDevice::Impl*> devices;
};

/** @brief The process-wide registry, built on first use. */
DeviceRegistry& registry()
{
    CV_SINGLETON_LAZY_INIT_REF(DeviceRegistry, new DeviceRegistry())
}

} // namespace

ComputeDevice ComputeDevice::cpu()
{
    return ComputeDevice();
}

std::vector<ComputeDevice> ComputeDevice::enumerate()
{
    const std::vector<Impl*>& devices = registry().devices;
    std::vector<ComputeDevice> result(1, cpu());
    for (size_t i = 0; i < devices.size(); i++)
        result.push_back(ComputeDevice(devices[i]));
    return result;
}

ComputeDevice ComputeDevice::find(const String& spec)
{
    String backend = spec;
    int index = 0;
    const size_t colon = spec.find(':');
    if (colon != String::npos)
    {
        backend = spec.substr(0, colon);
        const String number = spec.substr(colon + 1);
        char* end = 0;
        const long value = strtol(number.c_str(), &end, 10);
        if (number.empty() || *end != '\0' || value < 0 || value > INT_MAX)
            backend.clear();
        index = (int)value;
    }
    if (backend == "cpu")
        return cpu();
    const std::vector<Impl*>& devices = registry().devices;
    for (size_t i = 0; i < devices.size(); i++)
        if (devices[i]->backend == backend && devices[i]->index == index)
            return ComputeDevice(devices[i]);
    CV_LOG_INFO(NULL, "ComputeDevice: no device matches '" << spec << "', using the CPU");
    return cpu();
}

ComputeDevice ComputeDevice::of(const UMat& m)
{
    const MatAllocator* a = m.u ? m.u->currAllocator : m.allocator;
    if (!a || a == Mat::getDefaultAllocator() || a == Mat::getStdAllocator())
        return cpu();
    const std::vector<Impl*>& devices = registry().devices;
    for (size_t i = 0; i < devices.size(); i++)
        if (devices[i]->allocator == a)
            return ComputeDevice(devices[i]);
    return cpu();
}

ComputeDevice ComputeDevice::current()
{
    return ComputeDevice(getCoreTlsData().scopeDevice);
}

String ComputeDevice::backend() const
{
    return impl_ ? impl_->backend : String("cpu");
}

int ComputeDevice::index() const
{
    return impl_ ? impl_->index : 0;
}

String ComputeDevice::name() const
{
    return impl_ ? cv::format("%s:%d", impl_->backend.c_str(), impl_->index) : String("cpu");
}

String ComputeDevice::description() const
{
    if (!impl_)
        return "CPU";
    if (impl_->backend == "opencl")
        return ocl::Device::getDefault().name();
    if (impl_->backend == "cuda")
        return cuda::DeviceInfo(cuda::getDevice()).name();
    return String();
}

MatAllocator* ComputeDevice::allocator() const
{
    return impl_ ? impl_->allocator : Mat::getDefaultAllocator();
}

DeviceScope::DeviceScope(const ComputeDevice& device)
    : prev_(ComputeDevice::current())
{
    getCoreTlsData().scopeDevice = device.impl();
}

DeviceScope::DeviceScope(const String& spec)
    : DeviceScope(ComputeDevice::find(spec))
{
}

DeviceScope::~DeviceScope()
{
    getCoreTlsData().scopeDevice = prev_.impl();
}

MatAllocator* getDeviceScopeAllocator()
{
    const ComputeDevice::Impl* d = getCoreTlsData().scopeDevice;
    return d ? d->allocator : 0;
}

} // namespace cv
