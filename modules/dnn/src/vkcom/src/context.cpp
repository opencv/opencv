// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../../precomp.hpp"
#include "../vulkan/vk_loader.hpp"
#include "common.hpp"
#include "context.hpp"

namespace cv { namespace dnn { namespace vkcom {

#ifdef HAVE_VULKAN

std::shared_ptr<Context> kCtx;
bool enableValidationLayers = false;
VkInstance kInstance;
VkPhysicalDevice kPhysicalDevice;
VkDevice kDevice;
VkQueue kQueue;
VkCommandPool kCmdPool;
VkDebugReportCallbackEXT kDebugReportCallback;
uint32_t kQueueFamilyIndex;
std::vector<const char *> kEnabledLayers;
std::map<std::string, std::vector<uint32_t>> kShaders;
cv::Mutex kContextMtx;

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

// internally used
void createContext()
{
    cv::AutoLock lock(kContextMtx);
    if (!kCtx)
    {
        kCtx.reset(new Context());
    }
}

bool isAvailable()
{
    try
    {
        createContext();
    }
    catch (const cv::Exception& e)
    {
        CV_LOG_ERROR(NULL, "Failed to init Vulkan environment. " << e.what());
        return false;
    }

    return true;
}

Context::Context()
{
    if(!loadVulkanLibrary())
    {
        CV_Error(Error::StsError, "loadVulkanLibrary failed");
        return;
    }
    else if (!loadVulkanEntry())
    {
        CV_Error(Error::StsError, "loadVulkanEntry failed");
        return;
    }
    else if (!loadVulkanGlobalFunctions())
    {
        CV_Error(Error::StsError, "loadVulkanGlobalFunctions failed");
        return;
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
        CV_Error(Error::StsError, "loadVulkanFunctions failed");
        return;
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

    VK_CHECK_RESULT(vkCreateDevice(kPhysicalDevice, &deviceCreateInfo, NULL, &kDevice));

    // Get a handle to the only member of the queue family.
    vkGetDeviceQueue(kDevice, kQueueFamilyIndex, 0, &kQueue);

    // create command pool
    VkCommandPoolCreateInfo commandPoolCreateInfo = {};
    commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    commandPoolCreateInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    // the queue family of this command pool. All command buffers allocated from this command pool,
    // must be submitted to queues of this family ONLY.
    commandPoolCreateInfo.queueFamilyIndex = kQueueFamilyIndex;
    VK_CHECK_RESULT(vkCreateCommandPool(kDevice, &commandPoolCreateInfo, NULL, &kCmdPool));
}

Context::~Context()
{
    vkDestroyCommandPool(kDevice, kCmdPool, NULL);
    vkDestroyDevice(kDevice, NULL);

    if (enableValidationLayers) {
        auto func = (PFN_vkDestroyDebugReportCallbackEXT)
            vkGetInstanceProcAddr(kInstance, "vkDestroyDebugReportCallbackEXT");
        if (func == nullptr)
        {
            CV_LOG_FATAL(NULL, "Could not load vkDestroyDebugReportCallbackEXT");
        }
        else
        {
            func(kInstance, kDebugReportCallback, NULL);
        }
    }
    kShaders.clear();
    vkDestroyInstance(kInstance, NULL);

    return;
}

#endif // HAVE_VULKAN

}}} // namespace cv::dnn::vkcom
