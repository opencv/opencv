// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

/*
The code has been borrowed from ncnn inference engine (https://github.com/Tencent/ncnn/blob/20230223/src/gpu.cpp)
and adapted for OpenCV by Zihao Mu.
Below is the original copyright:
*/

// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#ifndef OPENCV_CONTEXT_VULKAN_HPP
#define OPENCV_CONTEXT_VULKAN_HPP

#include "../../precomp.hpp"

#ifdef HAVE_VULKAN
#include <vulkan/vulkan.h>
#endif // HAVE_VULKAN

#include "command.hpp"
#include "pipeline.hpp"

namespace cv { namespace dnn { namespace vkcom {

#ifdef HAVE_VULKAN

// NOTE: Manually set true to enable ValidationLayers, default is false.
const bool enableValidationLayers = false;

enum GPU_TYPE {
    GPU_TYPE_NOFOUND = -1,
    GPU_TYPE_DISCRETE = 0,
    GPU_TYPE_INTEGRATED = 1,
    GPU_TYPE_VIRTUAL = 2,
    GPU_TYPE_CPU_ONLY = 3,
};

// GPUInfo will parse GPU hardware information and save it in param.
struct GPUInfo
{
    // memory properties
    VkPhysicalDeviceMemoryProperties physicalDeviceMemoryProperties;

    // basic info
    GPU_TYPE type; // cpu, integrated GPU, discrete GPU.
    uint32_t apiVersion;
    uint32_t driverVersion;
    uint32_t vendorId;
    uint32_t deviceId;
    char deviceName[VK_MAX_PHYSICAL_DEVICE_NAME_SIZE];
    uint8_t pipelineCacheUUID[VK_UUID_SIZE];

    // hardware limit
    uint32_t maxSharedMemorySize;
    uint32_t maxWorkgroupCount_x;
    uint32_t maxWorkgroupCount_y;
    uint32_t maxWorkgroupCount_z;
    uint32_t maxWorkgroup_invocations;
    uint32_t maxWorkgroupSize_x;
    uint32_t maxWorkgroupSize_y;
    uint32_t maxWorkgroupSize_z;
    size_t memoryMapAlignment;
    size_t bufferOffsetAlignment;
    size_t non_coherent_atom_size;
    size_t bufferImageGranularity;

    uint32_t maxImageDimension_1d;
    uint32_t maxImageDimension_2d;
    uint32_t maxImageDimension_3d;
    float timestampPeriod;

    // runtime
    uint32_t computeQueueFamilyIndex;
    uint32_t graphicsQueueFamilyIndex;
    uint32_t transferQueueFamilyIndex;
    bool unifiedComputeTransferQueue;

    uint32_t computeQueueCount;
    uint32_t graphicsQueueCount;
    uint32_t transferQueueCount;

    // subgroup
    uint32_t subgroupSize;
    bool supportSubgroupBasic;
    bool supportSubgroupVote;
    bool supportSubgroupBallot;
    bool supportSubgroupShuffle;

    // TODO! Maybe in OpenCV we just care about if the device supports the FP16 or INT8.
    // fp16 and int8 feature
    bool support_fp16_packed;
    bool support_fp16_storage;
    bool support_fp16_arithmetic;
    bool support_int8_packed;
    bool support_int8_storage;
    bool support_int8_arithmetic;

    // cooperative matrix
    bool support_cooperative_matrix;

    // extension capability
    int support_VK_KHR_8bit_storage;
    int support_VK_KHR_16bit_storage;
    int support_VK_KHR_bind_memory2;
    int support_VK_KHR_create_renderpass2;
    int support_VK_KHR_dedicated_allocation;
    int support_VK_KHR_descriptor_update_template;
    int support_VK_KHR_external_memory;
    int support_VK_KHR_get_memory_requirements2;
    int support_VK_KHR_maintenance1;
    int support_VK_KHR_maintenance2;
    int support_VK_KHR_maintenance3;
    int support_VK_KHR_multiview;
    int support_VK_KHR_portability_subset;
    int support_VK_KHR_push_descriptor;
    int support_VK_KHR_sampler_ycbcr_conversion;
    int support_VK_KHR_shader_float16_int8;
    int support_VK_KHR_shader_float_controls;
    int support_VK_KHR_storage_buffer_storage_class;
    int support_VK_KHR_swapchain;
    int support_VK_EXT_descriptor_indexing;
    int support_VK_EXT_memory_budget;
    int support_VK_EXT_queue_family_foreign;
#if defined(__ANDROID_API__) && __ANDROID_API__ >= 26
    int support_VK_ANDROID_external_memory_android_hardware_buffer;
#endif // __ANDROID_API__ >= 26
    int support_VK_NV_cooperative_matrix;
};

// It contains all source we need in Vulkan Backend.
// every class may need use the resource from context.
class Context
{
public:
    static Ptr<Context> create();

    void operator=(const Context &) = delete;
    Context(Context &other) = delete;
    ~Context(); // TODO deconstruct this class when net was deconstructed.
    void reset();
private:
    GPUInfo parseGPUInfo(VkPhysicalDevice& device);

    // The following function will create kInstance.
    void createInstance();
    int findBestPhysicalGPUIndex();
    Context();

    // Vulkan related resource.
    VkInstance kInstance = VK_NULL_HANDLE;
    VkPhysicalDevice kPhysicalDevice = VK_NULL_HANDLE;

    uint32_t kQueueFamilyIndex;

    std::vector<GPUInfo> gpuInfoList; // store all available GPU information.
    int bestGPUIndex;

    std::vector<const char *> kEnabledLayers;
    VkDebugReportCallbackEXT kDebugReportCallback = VK_NULL_HANDLE;

    // Extension things
    std::vector<const char *> enabledExtensions;
    uint32_t instanceExtensionPropertyCount;
    std::vector<VkExtensionProperties> instanceExtensionProperties;
    uint32_t instanceApiVersion;
};

#endif // HAVE_VULKAN
}}} // namespace cv::dnn::vkcom
#endif //OPENCV_CONTEXT_VULKAN_HPP
