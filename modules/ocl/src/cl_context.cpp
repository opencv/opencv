/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2010-2012, Institute Of Software Chinese Academy Of Science, all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Guoping Long, longguoping@gmail.com
//    Niko Li, newlife20080214@gmail.com
//    Yao Wang, bitwangyaoyao@gmail.com
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other oclMaterials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"
#include <iomanip>
#include <fstream>
#include "binarycaching.hpp"

#undef __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

namespace cv { namespace ocl {

extern void fft_teardown();
extern void clBlasTeardown();

struct PlatformInfoImpl
{
    cl_platform_id platform_id;

    std::vector<int> deviceIDs;

    PlatformInfo info;

    PlatformInfoImpl()
        : platform_id(NULL)
    {
    }
};

struct DeviceInfoImpl
{
    cl_platform_id platform_id;
    cl_device_id device_id;

    DeviceInfo info;

    DeviceInfoImpl()
        : platform_id(NULL), device_id(NULL)
    {
    }
};

static std::vector<PlatformInfoImpl> global_platforms;
static std::vector<DeviceInfoImpl> global_devices;

static bool parseOpenCLVersion(const std::string& versionStr, int& major, int& minor)
{
    size_t p0 = versionStr.find(' ');
    while (true)
    {
        if (p0 == std::string::npos)
            break;
        if (p0 + 1 >= versionStr.length())
            break;
        char c = versionStr[p0 + 1];
        if (isdigit(c))
            break;
        p0 = versionStr.find(' ', p0 + 1);
    }
    size_t p1 = versionStr.find('.', p0);
    size_t p2 = versionStr.find(' ', p1);
    if (p0 == std::string::npos || p1 == std::string::npos || p2 == std::string::npos)
    {
        major = 0;
        minor = 0;
        return false;
    }
    std::string majorStr = versionStr.substr(p0 + 1, p1 - p0 - 1);
    std::string minorStr = versionStr.substr(p1 + 1, p2 - p1 - 1);
    major = atoi(majorStr.c_str());
    minor = atoi(minorStr.c_str());
    return true;
}

static int initializeOpenCLDevices()
{
    assert(global_devices.size() == 0);

    std::vector<cl::Platform> platforms;
    try
    {
        openCLSafeCall(cl::Platform::get(&platforms));
    }
    catch (cv::Exception& e)
    {
        return 0; // OpenCL not found
    }

    global_platforms.resize(platforms.size());

    for (size_t i = 0; i < platforms.size(); ++i)
    {
        PlatformInfoImpl& platformInfo = global_platforms[i];
        platformInfo.info._id = i;

        cl::Platform& platform = platforms[i];

        platformInfo.platform_id = platform();
        openCLSafeCall(platform.getInfo(CL_PLATFORM_PROFILE, &platformInfo.info.platformProfile));
        openCLSafeCall(platform.getInfo(CL_PLATFORM_VERSION, &platformInfo.info.platformVersion));
        openCLSafeCall(platform.getInfo(CL_PLATFORM_NAME, &platformInfo.info.platformName));
        openCLSafeCall(platform.getInfo(CL_PLATFORM_VENDOR, &platformInfo.info.platformVendor));
        openCLSafeCall(platform.getInfo(CL_PLATFORM_EXTENSIONS, &platformInfo.info.platformExtensons));

        parseOpenCLVersion(platformInfo.info.platformVersion,
                platformInfo.info.platformVersionMajor, platformInfo.info.platformVersionMinor);

        std::vector<cl::Device> devices;
        cl_int status = platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
        if(status != CL_DEVICE_NOT_FOUND)
            openCLVerifyCall(status);

        if(devices.size() > 0)
        {
            int baseIndx = global_devices.size();
            global_devices.resize(baseIndx + devices.size());
            platformInfo.deviceIDs.resize(devices.size());
            platformInfo.info.devices.resize(devices.size());

            for(size_t j = 0; j < devices.size(); ++j)
            {
                cl::Device& device = devices[j];

                DeviceInfoImpl& deviceInfo = global_devices[baseIndx + j];
                deviceInfo.info._id = baseIndx + j;
                deviceInfo.platform_id = platform();
                deviceInfo.device_id = device();

                deviceInfo.info.platform = &platformInfo.info;
                platformInfo.deviceIDs[j] = deviceInfo.info._id;

                cl_device_type type = -1;
                openCLSafeCall(device.getInfo(CL_DEVICE_TYPE, &type));
                deviceInfo.info.deviceType = DeviceType(type);

                openCLSafeCall(device.getInfo(CL_DEVICE_PROFILE, &deviceInfo.info.deviceProfile));
                openCLSafeCall(device.getInfo(CL_DEVICE_VERSION, &deviceInfo.info.deviceVersion));
                openCLSafeCall(device.getInfo(CL_DEVICE_NAME, &deviceInfo.info.deviceName));
                openCLSafeCall(device.getInfo(CL_DEVICE_VENDOR, &deviceInfo.info.deviceVendor));
                cl_uint vendorID = -1;
                openCLSafeCall(device.getInfo(CL_DEVICE_VENDOR_ID, &vendorID));
                deviceInfo.info.deviceVendorId = vendorID;
                openCLSafeCall(device.getInfo(CL_DRIVER_VERSION, &deviceInfo.info.deviceDriverVersion));
                openCLSafeCall(device.getInfo(CL_DEVICE_EXTENSIONS, &deviceInfo.info.deviceExtensions));

                parseOpenCLVersion(deviceInfo.info.deviceVersion,
                        deviceInfo.info.deviceVersionMajor, deviceInfo.info.deviceVersionMinor);

                size_t maxWorkGroupSize = 0;
                openCLSafeCall(device.getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &maxWorkGroupSize));
                deviceInfo.info.maxWorkGroupSize = maxWorkGroupSize;

                cl_uint maxDimensions = 0;
                openCLSafeCall(device.getInfo(CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, &maxDimensions));
                std::vector<size_t> maxWorkItemSizes(maxDimensions);
                openCLSafeCall(clGetDeviceInfo(device(), CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t) * maxDimensions,
                        (void *)&maxWorkItemSizes[0], 0));
                deviceInfo.info.maxWorkItemSizes = maxWorkItemSizes;

                cl_uint maxComputeUnits = 0;
                openCLSafeCall(device.getInfo(CL_DEVICE_MAX_COMPUTE_UNITS, &maxComputeUnits));
                deviceInfo.info.maxComputeUnits = maxComputeUnits;

                cl_ulong localMemorySize = 0;
                openCLSafeCall(device.getInfo(CL_DEVICE_LOCAL_MEM_SIZE, &localMemorySize));
                deviceInfo.info.localMemorySize = (size_t)localMemorySize;


                cl_bool unifiedMemory = false;
                openCLSafeCall(device.getInfo(CL_DEVICE_HOST_UNIFIED_MEMORY, &unifiedMemory));
                deviceInfo.info.isUnifiedMemory = unifiedMemory != 0;

                //initialize extra options for compilation. Currently only fp64 is included.
                //Assume 4KB is enough to store all possible extensions.
                openCLSafeCall(device.getInfo(CL_DEVICE_EXTENSIONS, &deviceInfo.info.deviceExtensions));

                size_t fp64_khr = deviceInfo.info.deviceExtensions.find("cl_khr_fp64");
                if(fp64_khr != std::string::npos)
                {
                    deviceInfo.info.compilationExtraOptions += "-D DOUBLE_SUPPORT";
                    deviceInfo.info.haveDoubleSupport = true;
                }
                else
                {
                    deviceInfo.info.haveDoubleSupport = false;
                }
            }
        }
    }

    for (size_t i = 0; i < platforms.size(); ++i)
    {
        PlatformInfoImpl& platformInfo = global_platforms[i];
        for(size_t j = 0; j < platformInfo.deviceIDs.size(); ++j)
        {
            DeviceInfoImpl& deviceInfo = global_devices[platformInfo.deviceIDs[j]];
            platformInfo.info.devices[j] = &deviceInfo.info;
        }
    }

    return global_devices.size();
}


DeviceInfo::DeviceInfo()
    : _id(-1), deviceType(DeviceType(0)),
      deviceVendorId(-1),
      maxWorkGroupSize(0), maxComputeUnits(0), localMemorySize(0),
      deviceVersionMajor(0), deviceVersionMinor(0),
      haveDoubleSupport(false), isUnifiedMemory(false),
      platform(NULL)
{
    // nothing
}

PlatformInfo::PlatformInfo()
    : _id(-1),
      platformVersionMajor(0), platformVersionMinor(0)
{
    // nothing
}

//////////////////////////////// OpenCL context ////////////////////////
//This is a global singleton class used to represent a OpenCL context.
class ContextImpl : public Context
{
public:
    const cl_device_id clDeviceID;
    cl_context clContext;
    cl_command_queue clCmdQueue;
    const DeviceInfo& deviceInfo;

protected:
    ContextImpl(const DeviceInfo& deviceInfo, cl_device_id clDeviceID)
        : clDeviceID(clDeviceID), clContext(NULL), clCmdQueue(NULL), deviceInfo(deviceInfo)
    {
        // nothing
    }
    ~ContextImpl();
public:

    static ContextImpl* getContext();
    static void setContext(const DeviceInfo* deviceInfo);

    bool supportsFeature(FEATURE_TYPE featureType) const;

    static void cleanupContext(void);
};

static cv::Mutex currentContextMutex;
static ContextImpl* currentContext = NULL;

Context* Context::getContext()
{
    return currentContext;
}

bool Context::supportsFeature(FEATURE_TYPE featureType) const
{
    return ((ContextImpl*)this)->supportsFeature(featureType);
}

const DeviceInfo& Context::getDeviceInfo() const
{
    return ((ContextImpl*)this)->deviceInfo;
}

const void* Context::getOpenCLContextPtr() const
{
    return &(((ContextImpl*)this)->clContext);
}

const void* Context::getOpenCLCommandQueuePtr() const
{
    return &(((ContextImpl*)this)->clCmdQueue);
}

const void* Context::getOpenCLDeviceIDPtr() const
{
    return &(((ContextImpl*)this)->clDeviceID);
}


bool ContextImpl::supportsFeature(FEATURE_TYPE featureType) const
{
    switch (featureType)
    {
    case FEATURE_CL_DOUBLE:
        return deviceInfo.haveDoubleSupport;
    case FEATURE_CL_UNIFIED_MEM:
        return deviceInfo.isUnifiedMemory;
    case FEATURE_CL_VER_1_2:
        return deviceInfo.deviceVersionMajor > 1 || (deviceInfo.deviceVersionMajor == 1 && deviceInfo.deviceVersionMinor >= 2);
    }
    CV_Error(CV_StsBadArg, "Invalid feature type");
    return false;
}

#if defined(WIN32)
static bool __termination = false;
#endif

ContextImpl::~ContextImpl()
{
    fft_teardown();
    clBlasTeardown();

#ifdef WIN32
    // if process is on termination stage (ExitProcess was called and other threads were terminated)
    // then disable command queue release because it may cause program hang
    if (!__termination)
#endif
    {
        if(clCmdQueue)
        {
            openCLSafeCall(clReleaseCommandQueue(clCmdQueue)); // some cleanup problems are here
        }

        if(clContext)
        {
            openCLSafeCall(clReleaseContext(clContext));
        }
    }
    clCmdQueue = NULL;
    clContext = NULL;
}

void ContextImpl::cleanupContext(void)
{
    cv::AutoLock lock(currentContextMutex);
    if (currentContext)
        delete currentContext;
    currentContext = NULL;
}

void ContextImpl::setContext(const DeviceInfo* deviceInfo)
{
    CV_Assert(deviceInfo->_id >= 0 && deviceInfo->_id < (int)global_devices.size());

    DeviceInfoImpl& infoImpl = global_devices[deviceInfo->_id];
    CV_Assert(deviceInfo == &infoImpl.info);

    cl_int status = 0;
    cl_context_properties cps[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)(infoImpl.platform_id), 0 };
    cl_context clContext = clCreateContext(cps, 1, &infoImpl.device_id, NULL, NULL, &status);
    openCLVerifyCall(status);
    // TODO add CL_QUEUE_PROFILING_ENABLE
    cl_command_queue clCmdQueue = clCreateCommandQueue(clContext, infoImpl.device_id, 0, &status);
    openCLVerifyCall(status);

    ContextImpl* ctx = new ContextImpl(infoImpl.info, infoImpl.device_id);
    ctx->clCmdQueue = clCmdQueue;
    ctx->clContext = clContext;

    ContextImpl* old = NULL;
    {
        cv::AutoLock lock(currentContextMutex);
        old = currentContext;
        currentContext = ctx;
    }
    if (old != NULL)
    {
        delete old;
    }
}

ContextImpl* ContextImpl::getContext()
{
    return currentContext;
}

int getOpenCLPlatforms(PlatformsInfo& platforms)
{
    platforms.clear();

    for (size_t id = 0; id < global_platforms.size(); ++id)
    {
        PlatformInfoImpl& impl = global_platforms[id];
        platforms.push_back(&impl.info);
    }

    return platforms.size();
}

int getOpenCLDevices(std::vector<const DeviceInfo*> &devices, int deviceType, const PlatformInfo* platform)
{
    devices.clear();

    switch(deviceType)
    {
    case CVCL_DEVICE_TYPE_DEFAULT:
    case CVCL_DEVICE_TYPE_CPU:
    case CVCL_DEVICE_TYPE_GPU:
    case CVCL_DEVICE_TYPE_ACCELERATOR:
    case CVCL_DEVICE_TYPE_ALL:
        break;
    default:
        return 0;
    }

    if (platform == NULL)
    {
        for (size_t id = 0; id < global_devices.size(); ++id)
        {
            DeviceInfoImpl& deviceInfo = global_devices[id];
            if (((int)deviceInfo.info.deviceType & deviceType) == deviceType)
            {
                devices.push_back(&deviceInfo.info);
            }
        }
    }
    else
    {
        for (size_t id = 0; id < platform->devices.size(); ++id)
        {
            const DeviceInfo* deviceInfo = platform->devices[id];
            if (((int)deviceInfo->deviceType & deviceType) == deviceType)
            {
                devices.push_back(deviceInfo);
            }
        }
    }

    return (int)devices.size();
}

void setDevice(const DeviceInfo* info)
{
    ContextImpl::setContext(info);
}

bool supportsFeature(FEATURE_TYPE featureType)
{
    return Context::getContext()->supportsFeature(featureType);
}

struct __Module
{
    __Module() { initializeOpenCLDevices(); }
    ~__Module() { ContextImpl::cleanupContext(); }
};
static __Module __module;


}//namespace ocl
}//namespace cv


#if defined(WIN32) && defined(CVAPI_EXPORTS)

extern "C"
BOOL WINAPI DllMain(HINSTANCE /*hInst*/, DWORD fdwReason, LPVOID lpReserved)
{
    if (fdwReason == DLL_PROCESS_DETACH)
    {
        if (lpReserved != NULL) // called after ExitProcess() call
            cv::ocl::__termination = true;
    }
    return TRUE;
}

#endif
