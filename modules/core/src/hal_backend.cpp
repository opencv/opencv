// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.
#include "precomp.hpp"
#include "opencv2/core/hal/backend_registry.hpp"
#include <vector>
#include <dlfcn.h>      // dlopen, dlsym, dlclose
#include <mutex>        // std::once_flag, std::call_once
#include <string>
#include <cstring>      // memcpy
#include <cstdlib>      // getenv

namespace cv { namespace hal {

namespace {

// Global backend list. Static local avoids static init order issues.
std::vector<Backend*>& getRegistry() {
    static std::vector<Backend*> registry;
    return registry;
}

} // anonymous namespace

void registerBackend(Backend* backend) {
    CV_Assert(backend != nullptr);
    getRegistry().push_back(backend);
}

Backend* findBackend() {
    std::vector<Backend*>& reg = getRegistry();
    return reg.empty() ? nullptr : reg.front();
}

void clearBackends() {
    getRegistry().clear();
}

namespace {

typedef Backend* (*BackendFactory)();

bool tryLoadPlugin(const std::string& path) {
    // RTLD_GLOBAL so the plugin can see opencv_core symbols.
    void* handle = dlopen(path.c_str(), RTLD_LAZY | RTLD_GLOBAL);
    if (!handle) return false;

    dlerror();
    BackendFactory factory = reinterpret_cast<BackendFactory>(
        dlsym(handle, "cv_hal_createCudaBackend"));
    if (dlerror() != nullptr || factory == nullptr) {
        dlclose(handle);
        return false;
    }

    Backend* backend = factory();
    if (backend == nullptr) { dlclose(handle); return false; }

    registerBackend(backend);                  // handle stays open for life
    return true;
}

void doLoadBackendPlugins() {
    std::vector<std::string> candidates;
    const char* envPath = std::getenv("OPENCV_GPU_BACKEND_PATH");
    if (envPath != nullptr) candidates.push_back(std::string(envPath));
    candidates.push_back("libopencv_cuda_backend.so");
    for (const std::string& path : candidates)
        if (tryLoadPlugin(path)) break;
}

} // anonymous namespace

void loadBackendPlugins() {
    static std::once_flag flag;
    std::call_once(flag, doLoadBackendPlugins);
}

// Static initializer: loads plugins when libopencv_core loads, before main().
namespace {
struct BackendPluginInitializer {
    BackendPluginInitializer() { loadBackendPlugins(); }
};
static BackendPluginInitializer g_backendPluginInit;
} // anonymous namespace

}} // cv::hal

// Python-friendly bridge so cv2 can create/read GPU-HAL-resident UMats.
namespace cv {

UMat gpuUpload(InputArray _src)
{
    hal::Backend* b = hal::findBackend();
    CV_Assert(b && b->allocator() && "no GPU backend / allocator registered");
    Mat src = _src.getMat();
    CV_Assert(src.isContinuous());
    MatAllocator* a = b->allocator();
    UMat u;
    u.allocator = a;
    u.create(src.rows, src.cols, src.type());
    a->map(u.u, ACCESS_WRITE);
    std::memcpy(u.u->data, src.data, u.u->size);
    a->unmap(u.u);                                  // host -> device
    return u;
}

void gpuDownload(InputArray _src, OutputArray _dst)
{
    UMat u = _src.getUMat();
    CV_Assert(u.u && u.u->currAllocator);
    const_cast<MatAllocator*>(u.u->currAllocator)->map(u.u, ACCESS_READ);  // device -> host
    Mat hdr(u.rows, u.cols, u.type(), u.u->data, u.step[0]);
    hdr.copyTo(_dst);
}

} // namespace cv
