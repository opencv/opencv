// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../../precomp.hpp"
#ifdef HAVE_VULKAN
#include <vulkan/vulkan.h>
#endif
#include "vk_functions.hpp"

#if defined(_WIN32)
#include <windows.h>
typedef HMODULE VulkanHandle;
#define DEFAULT_VK_LIBRARY_PATH "vulkan-1.dll"
#define LOAD_VK_LIBRARY(path) LoadLibrary(path)
#define FREE_VK_LIBRARY(handle) FreeLibrary(handle)
#define GET_VK_ENTRY_POINT(handle) \
        (PFN_vkGetInstanceProcAddr)GetProcAddress(handle, "vkGetInstanceProcAddr");
#endif // _WIN32

#if defined(__linux__)
#include <dlfcn.h>
#include <stdio.h>
typedef void* VulkanHandle;
#define DEFAULT_VK_LIBRARY_PATH "libvulkan.so"
#define LOAD_VK_LIBRARY(path) dlopen(path, RTLD_LAZY | RTLD_GLOBAL)
#define FREE_VK_LIBRARY(handle) dlclose(handle)
#define GET_VK_ENTRY_POINT(handle) \
        (PFN_vkGetInstanceProcAddr)dlsym(handle, "vkGetInstanceProcAddr");
#endif // __linux__

#if defined(__APPLE__)
#include <dlfcn.h>
#include <stdio.h>
typedef void* VulkanHandle;

#if defined(__x86_64__)
#define DEFAULT_VK_LIBRARY_PATH "libvulkan.dylib"
#else // For Apple ARM chip, we use MoltenVK lib.
#define DEFAULT_VK_LIBRARY_PATH "libMoltenVK.dylib"
#endif

#define LOAD_VK_LIBRARY(path) dlopen(path, RTLD_LAZY | RTLD_GLOBAL)
#define FREE_VK_LIBRARY(handle) dlclose(handle)
#define GET_VK_ENTRY_POINT(handle) \
        (PFN_vkGetInstanceProcAddr)dlsym(handle, "vkGetInstanceProcAddr");
#endif // Macos

#ifndef DEFAULT_VK_LIBRARY_PATH
#define DEFAULT_VK_LIBRARY_PATH ""
#define LOAD_VK_LIBRARY(path) nullptr
#define FREE_VK_LIBRARY(handle)
#define GET_VK_ENTRY_POINT(handle) nullptr
#endif

namespace cv { namespace dnn { namespace vkcom {

#ifdef HAVE_VULKAN
static VulkanHandle handle = nullptr;

bool loadVulkanFunctions(VkInstance& instance)
{
#define VK_FUNC(fun) \
    fun = (PFN_##fun)vkGetInstanceProcAddr(instance, #fun);

#define VK_FUNC_MANDATORY(fun) \
    VK_FUNC(fun) \
    if(!fun) \
    { \
      fprintf(stderr, "Could not load Vulkan function: %s !\n", #fun); \
      return false; \
    }

#include "function_list.inl.hpp"
    return true;
}

bool loadVulkanGlobalFunctions()
{
#define VK_GLOBAL_LEVEL_FUNC(fun) \
    fun = (PFN_##fun)vkGetInstanceProcAddr(nullptr, #fun);

#define VK_GLOBAL_LEVEL_FUNC_MANDATORY(fun) \
    VK_GLOBAL_LEVEL_FUNC(fun) \
    if(!fun) \
    { \
      fprintf(stderr, "Could not load global Vulkan function: %s !\n", #fun); \
      return false; \
    }

#include "function_list.inl.hpp"
    return true;
}

bool loadVulkanEntry()
{
    if (handle == nullptr)
        return false;

    vkGetInstanceProcAddr = GET_VK_ENTRY_POINT(handle);
    if (!vkGetInstanceProcAddr)
    {
        fprintf(stderr, "Could not load Vulkan entry function: vkGetInstanceProcAddr!\n");
        return false;
    }

    return true;
}

bool loadVulkanLibrary()
{
    if (handle != nullptr)
        return true;

    const std::string path = cv::utils::getConfigurationParameterString("OPENCV_VULKAN_RUNTIME", DEFAULT_VK_LIBRARY_PATH);

    handle = LOAD_VK_LIBRARY(path.c_str());
    if( handle == nullptr )
    {
        fprintf(stderr, "Could not load Vulkan library: %s!\n", path.c_str());
        fprintf(stderr, "Please download the Vulkan SDK and set the environment variable of OPENCV_VULKAN_RUNTIME according "
                        "to your system environment.\n");
        fprintf(stderr, "For M1 Mac and IOS, we use MoltenVK to map the Vulkan code to native apple Metal code.\n");
        fprintf(stderr, "You can download the SDK from https://vulkan.lunarg.com/sdk/home.\n");
        fprintf(stderr, "The following are some examples:\n");
        fprintf(stderr, "For Windows, OPENCV_VULKAN_RUNTIME=D:\\VulkanSDK\\1.3.236.0\\Bin\\vulkan-1.dll\n");
        fprintf(stderr, "For Linux, OPENCV_VULKAN_RUNTIME=/opt/vulkan/1.3.236.0/x86_64/libvulkan.so\n");
        fprintf(stderr, "For MacOS of x86, OPENCV_VULKAN_RUNTIME=/opt/vulkan/1.3.236.0/x86_64/libvulkan.dylib\n");
        fprintf(stderr, "For MacOS of M1 or IOS, OPENCV_VULKAN_RUNTIME=/opt/VulkanSDK/1.3.231.1/MoltenVK/dylib/macOS/libMoltenVK.dylib\n");
        return false;
    }

    return true;
}

#endif // HAVE_VULKAN

}}} // namespace cv::dnn::vkcom
