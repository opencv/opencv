// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <cstring>
#include <memory>
#include "dawn_utils.hpp"
#include "opencv2/core/base.hpp"
namespace cv { namespace dnn { namespace webgpu {
#if defined(__EMSCRIPTEN__) && defined(DAWN_EMSDK)
wgpu::Device createCppDawnDevice() {
    return wgpu::Device::Acquire(emscripten_webgpu_get_device());
}
#else
#ifdef HAVE_WEBGPU

static std::shared_ptr<dawn_native::Instance> instance;

void PrintDeviceError(WGPUErrorType errorType, const char* message, void*) {
    String errorTypeName = "";
    switch (errorType) {
        case WGPUErrorType_Validation:
            errorTypeName = "WGPUErrorTyp Validation";
            break;
        case WGPUErrorType_OutOfMemory:
            errorTypeName = "WGPUErrorTyp Out of memory";
            break;
        case WGPUErrorType_Unknown:
            errorTypeName = "WGPUErrorTyp Unknown";
            break;
        case WGPUErrorType_DeviceLost:
            errorTypeName = "WGPUErrorTyp Device lost";
            break;
        default:
            errorTypeName = "WGPUErrorTyp Unknown";
            return;
    }
    errorTypeName += "Error message: ";
    errorTypeName += message;
    CV_Error(Error::StsError, errorTypeName);
}

wgpu::Device createCppDawnDevice() {
    instance = std::make_shared<dawn_native::Instance>();
    instance->DiscoverDefaultAdapters();
    // Get an adapter for the backend to use, and create the device.
    dawn_native::Adapter backendAdapter;
    {
        std::vector<dawn_native::Adapter> adapters = instance->GetAdapters();
        auto adapterIt = std::find_if(adapters.begin(), adapters.end(),
                                    [](const dawn_native::Adapter adapter) -> bool {
                                        wgpu::AdapterProperties properties;
                                        adapter.GetProperties(&properties);
#ifdef DAWN_METAL
                                        return properties.backendType == wgpu::BackendType::Metal;
#else
                                        return properties.backendType == wgpu::BackendType::Vulkan;
#endif
                                    });
        backendAdapter = *adapterIt;
    }
    WGPUDevice backendDevice = backendAdapter.CreateDevice();
    DawnProcTable backendProcs = dawn_native::GetProcs();
    dawnProcSetProcs(&backendProcs);
    backendProcs.deviceSetUncapturedErrorCallback(backendDevice, PrintDeviceError, nullptr);
    return wgpu::Device::Acquire(backendDevice);
}

#endif  // HAVE_WEBGPU
#endif  //__EMSCRIPTEN__
}}}  //namespace cv::dnn::webgpu