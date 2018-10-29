// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../../precomp.hpp"
#include "common.hpp"
#include "internal.hpp"
#include "../include/op_conv.hpp"
#include "../include/op_pool.hpp"
#include "../include/op_lrn.hpp"
#include "../include/op_concat.hpp"
#include "../include/op_softmax.hpp"
#include "../vulkan/vk_loader.hpp"

namespace cv { namespace dnn { namespace vkcom {

#ifdef HAVE_VULKAN

static bool enableValidationLayers = false;
static VkInstance kInstance;
static VkPhysicalDevice kPhysicalDevice;
static VkDebugReportCallbackEXT kDebugReportCallback;
static uint32_t kQueueFamilyIndex;
std::vector<const char *> kEnabledLayers;
typedef std::map<std::thread::id, Context*> IdToContextMap;
IdToContextMap kThreadResources;
static std::map<std::string, std::vector<uint32_t>> kShaders;
static int init_count = 0;
static bool init();
static void release();
static uint32_t getComputeQueueFamilyIndex();
static bool checkExtensionAvailability(const char *extension_name,
                                       const std::vector<VkExtensionProperties>
                                       &available_extensions);
static VKAPI_ATTR VkBool32 VKAPI_CALL debugReportCallbackFn(
       VkDebugReportFlagsEXT                       flags,
       VkDebugReportObjectTypeEXT                  objectType,
       uint64_t                                    object,
       size_t                                      location,
       int32_t                                     messageCode,
       const char*                                 pLayerPrefix,
       const char*                                 pMessage,
       void*                                       pUserData);

static void setContext(Context* ctx)
{
    cv::AutoLock lock(getInitializationMutex());
    std::thread::id tid = std::this_thread::get_id();
    if (kThreadResources.find(tid) != kThreadResources.end())
    {
        return;
    }
    kThreadResources.insert(std::pair<std::thread::id, Context*>(tid, ctx));
}

Context* getContext()
{
    Context* ctx = NULL;

    cv::AutoLock lock(getInitializationMutex());
    std::thread::id tid = std::this_thread::get_id();
    IdToContextMap::iterator it = kThreadResources.find(tid);
    if (it != kThreadResources.end())
    {
        ctx = it->second;
    }
    return ctx;
}

static void removeContext()
{
    cv::AutoLock lock(getInitializationMutex());
    std::thread::id tid = std::this_thread::get_id();
    IdToContextMap::iterator it = kThreadResources.find(tid);
    if (it == kThreadResources.end())
    {
        return;
    }
    kThreadResources.erase(it);
}

bool initPerThread()
{
    VkDevice device;
    VkQueue queue;
    VkCommandPool cmd_pool;

    VKCOM_CHECK_BOOL_RET_VAL(init(), false);
    Context* ctx = getContext();
    if (ctx)
    {
        ctx->ref++;
        return true;
    }

    // create device, queue, command pool
    VkDeviceQueueCreateInfo queueCreateInfo = {};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = kQueueFamilyIndex;
    queueCreateInfo.queueCount = 1; // create one queue in this family. We don't need more.
    float queuePriorities = 1.0;  // we only have one queue, so this is not that imporant.
    queueCreateInfo.pQueuePriorities = &queuePriorities;

    VkDeviceCreateInfo deviceCreateInfo = {};

    // Specify any desired device features here. We do not need any for this application, though.
    VkPhysicalDeviceFeatures deviceFeatures = {};

    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCreateInfo.enabledLayerCount = kEnabledLayers.size();
    deviceCreateInfo.ppEnabledLayerNames = kEnabledLayers.data();
    deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
    deviceCreateInfo.queueCreateInfoCount = 1;
    deviceCreateInfo.pEnabledFeatures = &deviceFeatures;

    VK_CHECK_RESULT(vkCreateDevice(kPhysicalDevice, &deviceCreateInfo, NULL, &device));

    // Get a handle to the only member of the queue family.
    vkGetDeviceQueue(device, kQueueFamilyIndex, 0, &queue);

    // create command pool
    VkCommandPoolCreateInfo commandPoolCreateInfo = {};
    commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    commandPoolCreateInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    // the queue family of this command pool. All command buffers allocated from this command pool,
    // must be submitted to queues of this family ONLY.
    commandPoolCreateInfo.queueFamilyIndex = kQueueFamilyIndex;
    VK_CHECK_RESULT(vkCreateCommandPool(device, &commandPoolCreateInfo, NULL, &cmd_pool));

    ctx = new Context();
    ctx->device = device;
    ctx->queue = queue;
    ctx->cmd_pool = cmd_pool;
    ctx->ref = 1;
    setContext(ctx);
    return true;
}

void deinitPerThread()
{
    Context* ctx = getContext();
    if (ctx == NULL)
    {
        release();
        return;
    }

    if (ctx->ref > 1)
    {
        ctx->ref--;
    }
    else if (ctx->ref == 1)
    {
        for(auto &kv: ctx->shader_modules)
        {
            vkDestroyShaderModule(ctx->device, kv.second, NULL);
        }
        ctx->shader_modules.clear();
        vkDestroyCommandPool(ctx->device, ctx->cmd_pool, NULL);
        vkDestroyDevice(ctx->device, NULL);
        removeContext();
        delete ctx;
    }
    else
        CV_Assert(0);
    release();
}

static bool init()
{
    cv::AutoLock lock(getInitializationMutex());

    if (init_count == 0)
    {
        if(!loadVulkanLibrary())
        {
            return false;
        }
        else if (!loadVulkanEntry())
        {
            return false;
        }
        else if (!loadVulkanGlobalFunctions())
        {
            return false;
        }

        // create VkInstance, VkPhysicalDevice
        std::vector<const char *> enabledExtensions;
        if (enableValidationLayers)
        {
            uint32_t layerCount;
            vkEnumerateInstanceLayerProperties(&layerCount, NULL);

            std::vector<VkLayerProperties> layerProperties(layerCount);
            vkEnumerateInstanceLayerProperties(&layerCount, layerProperties.data());

            bool foundLayer = false;
            for (VkLayerProperties prop : layerProperties)
            {
                if (strcmp("VK_LAYER_LUNARG_standard_validation", prop.layerName) == 0)
                {
                    foundLayer = true;
                    break;
                }
            }

            if (!foundLayer)
            {
                throw std::runtime_error("Layer VK_LAYER_LUNARG_standard_validation not supported\n");
            }
            kEnabledLayers.push_back("VK_LAYER_LUNARG_standard_validation");

            uint32_t extensionCount;

            vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, NULL);
            std::vector<VkExtensionProperties> extensionProperties(extensionCount);
            vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensionProperties.data());

            bool foundExtension = false;
            for (VkExtensionProperties prop : extensionProperties)
            {
                if (strcmp(VK_EXT_DEBUG_REPORT_EXTENSION_NAME, prop.extensionName) == 0)
                {
                    foundExtension = true;
                    break;
                }
            }

            if (!foundExtension) {
                throw std::runtime_error("Extension VK_EXT_DEBUG_REPORT_EXTENSION_NAME not supported\n");
            }
            enabledExtensions.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
        }

        VkApplicationInfo applicationInfo = {};
        applicationInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        applicationInfo.pApplicationName = "VkCom Library";
        applicationInfo.applicationVersion = 0;
        applicationInfo.pEngineName = "vkcom";
        applicationInfo.engineVersion = 0;
        applicationInfo.apiVersion = VK_API_VERSION_1_0;;

        VkInstanceCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.flags = 0;
        createInfo.pApplicationInfo = &applicationInfo;

        // Give our desired layers and extensions to vulkan.
        createInfo.enabledLayerCount = kEnabledLayers.size();
        createInfo.ppEnabledLayerNames = kEnabledLayers.data();
        createInfo.enabledExtensionCount = enabledExtensions.size();
        createInfo.ppEnabledExtensionNames = enabledExtensions.data();

        VK_CHECK_RESULT(vkCreateInstance(&createInfo, NULL, &kInstance));

        if (!loadVulkanFunctions(kInstance))
        {
            return false;
        }

        if (enableValidationLayers && vkCreateDebugReportCallbackEXT)
        {
            VkDebugReportCallbackCreateInfoEXT createInfo = {};
            createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT;
            createInfo.flags = VK_DEBUG_REPORT_ERROR_BIT_EXT |
                               VK_DEBUG_REPORT_WARNING_BIT_EXT |
                               VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT;
            createInfo.pfnCallback = &debugReportCallbackFn;

            // Create and register callback.
            VK_CHECK_RESULT(vkCreateDebugReportCallbackEXT(kInstance, &createInfo,
                                                           NULL, &kDebugReportCallback));
        }

        // find physical device
        uint32_t deviceCount;
        vkEnumeratePhysicalDevices(kInstance, &deviceCount, NULL);
        if (deviceCount == 0)
        {
            throw std::runtime_error("could not find a device with vulkan support");
        }

        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(kInstance, &deviceCount, devices.data());

        for (VkPhysicalDevice device : devices)
        {
            if (true)
            {
                kPhysicalDevice = device;
                break;
            }
        }

        kQueueFamilyIndex = getComputeQueueFamilyIndex();
    }

    init_count++;
    return true;
}

static void release()
{
    cv::AutoLock lock(getInitializationMutex());
    if (init_count == 0)
    {
        return;
    }

    init_count--;
    if (init_count == 0)
    {
        if (enableValidationLayers) {
            auto func = (PFN_vkDestroyDebugReportCallbackEXT)
                vkGetInstanceProcAddr(kInstance, "vkDestroyDebugReportCallbackEXT");
            if (func == nullptr) {
                throw std::runtime_error("Could not load vkDestroyDebugReportCallbackEXT");
            }
            func(kInstance, kDebugReportCallback, NULL);
        }
        kShaders.clear();
        vkDestroyInstance(kInstance, NULL);
    }

    return;
}

// Returns the index of a queue family that supports compute operations.
static uint32_t getComputeQueueFamilyIndex()
{
    uint32_t queueFamilyCount;

    vkGetPhysicalDeviceQueueFamilyProperties(kPhysicalDevice, &queueFamilyCount, NULL);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(kPhysicalDevice,
                                             &queueFamilyCount,
                                             queueFamilies.data());

    uint32_t i = 0;
    for (; i < queueFamilies.size(); ++i)
    {
        VkQueueFamilyProperties props = queueFamilies[i];

        if (props.queueCount > 0 && (props.queueFlags & VK_QUEUE_COMPUTE_BIT))
        {
            break;
        }
    }

    if (i == queueFamilies.size())
    {
        throw std::runtime_error("could not find a queue family that supports operations");
    }

    return i;
}

bool checkExtensionAvailability(const char *extension_name,
                                const std::vector<VkExtensionProperties> &available_extensions)
{
    for( size_t i = 0; i < available_extensions.size(); ++i )
    {
      if( strcmp( available_extensions[i].extensionName, extension_name ) == 0 )
      {
        return true;
      }
    }
    return false;
}

VKAPI_ATTR VkBool32 VKAPI_CALL debugReportCallbackFn(
        VkDebugReportFlagsEXT                       flags,
        VkDebugReportObjectTypeEXT                  objectType,
        uint64_t                                    object,
        size_t                                      location,
        int32_t                                     messageCode,
        const char*                                 pLayerPrefix,
        const char*                                 pMessage,
        void*                                       pUserData)
{
        std::cout << "Debug Report: " << pLayerPrefix << ":" << pMessage << std::endl;
        return VK_FALSE;
}

// internally used functions
VkPhysicalDevice getPhysicalDevice()
{
    return kPhysicalDevice;
}

bool isAvailable()
{
    return getContext() != NULL;
}

#endif // HAVE_VULKAN

}}} // namespace cv::dnn::vkcom
