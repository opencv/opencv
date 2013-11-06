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
//     and/or other materials provided with the distribution.
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
#include "cl_programcache.hpp"

#include "opencv2/ocl/private/opencl_utils.hpp"

namespace cv {
namespace ocl {

struct __Module
{
    __Module();
    ~__Module();
    cv::Mutex initializationMutex;
    cv::Mutex currentContextMutex;
};
static __Module __module;

cv::Mutex& getInitializationMutex()
{
    return __module.initializationMutex;
}


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

static void split(const std::string &s, char delim, std::vector<std::string> &elems) {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
}

static std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, elems);
    return elems;
}

// Layout: <Platform>:<CPU|GPU|ACCELERATOR|nothing=GPU/CPU>:<deviceName>
// Sample: AMD:GPU:
// Sample: AMD:GPU:Tahiti
// Sample: :GPU|CPU: = '' = ':' = '::'
static bool parseOpenCLDeviceConfiguration(const std::string& configurationStr,
        std::string& platform, std::vector<std::string>& deviceTypes, std::string& deviceNameOrID)
{
    std::string deviceTypesStr;
    size_t p0 = configurationStr.find(':');
    if (p0 != std::string::npos)
    {
        size_t p1 = configurationStr.find(':', p0 + 1);
        if (p1 != std::string::npos)
        {
            size_t p2 = configurationStr.find(':', p1 + 1);
            if (p2 != std::string::npos)
            {
                std::cerr << "ERROR: Invalid configuration string for OpenCL device" << std::endl;
                return false;
            }
            else
            {
                // assume platform + device types + device name/id
                platform = configurationStr.substr(0, p0);
                deviceTypesStr = configurationStr.substr(p0 + 1, p1 - (p0 + 1));
                deviceNameOrID = configurationStr.substr(p1 + 1, configurationStr.length() - (p1 + 1));
            }
        }
        else
        {
            // assume platform + device types
            platform = configurationStr.substr(0, p0);
            deviceTypesStr = configurationStr.substr(p0 + 1, configurationStr.length() - (p0 + 1));
        }
    }
    else
    {
        // assume only platform
        platform = configurationStr;
    }
    deviceTypes = split(deviceTypesStr, '|');
    return true;
}

static bool __deviceSelected = false;
static bool selectOpenCLDevice()
{
    __deviceSelected = true;

    std::string platform;
    std::vector<std::string> deviceTypes;
    std::string deviceName;
    const char* configuration = getenv("OPENCV_OPENCL_DEVICE");
    if (configuration)
    {
        if (!parseOpenCLDeviceConfiguration(std::string(configuration), platform, deviceTypes, deviceName))
            return false;
    }

    bool isID = false;
    int deviceID = -1;
    if (deviceName.length() == 1)
    // We limit ID range to 0..9, because we want to write:
    // - '2500' to mean i5-2500
    // - '8350' to mean AMD FX-8350
    // - '650' to mean GeForce 650
    // To extend ID range change condition to '> 0'
    {
        isID = true;
        for (size_t i = 0; i < deviceName.length(); i++)
        {
            if (!isdigit(deviceName[i]))
            {
                isID = false;
                break;
            }
        }
        if (isID)
        {
            deviceID = atoi(deviceName.c_str());
            CV_Assert(deviceID >= 0);
        }
    }

    const PlatformInfo* platformInfo = NULL;
    if (platform.length() > 0)
    {
        PlatformsInfo platforms;
        getOpenCLPlatforms(platforms);
        for (size_t i = 0; i < platforms.size(); i++)
        {
            if (platforms[i]->platformName.find(platform) != std::string::npos)
            {
                platformInfo = platforms[i];
                break;
            }
        }
        if (platformInfo == NULL)
        {
            std::cerr << "ERROR: Can't find OpenCL platform by name: " << platform << std::endl;
            goto not_found;
        }
    }

    if (deviceTypes.size() == 0)
    {
        if (!isID)
        {
            deviceTypes.push_back("GPU");
            deviceTypes.push_back("CPU");
        }
        else
        {
            deviceTypes.push_back("ALL");
        }
    }
    for (size_t t = 0; t < deviceTypes.size(); t++)
    {
        int deviceType = 0;
        if (deviceTypes[t] == "GPU")
        {
            deviceType = CVCL_DEVICE_TYPE_GPU;
        }
        else if (deviceTypes[t] == "CPU")
        {
            deviceType = CVCL_DEVICE_TYPE_CPU;
        }
        else if (deviceTypes[t] == "ACCELERATOR")
        {
            deviceType = CVCL_DEVICE_TYPE_ACCELERATOR;
        }
        else if (deviceTypes[t] == "ALL")
        {
            deviceType = CVCL_DEVICE_TYPE_ALL;
        }
        else
        {
            std::cerr << "ERROR: Unsupported device type for OpenCL device (GPU, CPU, ACCELERATOR): " << deviceTypes[t] << std::endl;
            goto not_found;
        }

        DevicesInfo devices;
        getOpenCLDevices(devices, deviceType, platformInfo);

        for (size_t i = (isID ? deviceID : 0);
             (isID ? (i == (size_t)deviceID) : true) && (i < devices.size());
             i++)
        {
            if (isID || devices[i]->deviceName.find(deviceName) != std::string::npos)
            {
                // check for OpenCL 1.1
                if (devices[i]->deviceVersionMajor < 1 ||
                        (devices[i]->deviceVersionMajor == 1 && devices[i]->deviceVersionMinor < 1))
                {
                    std::cerr << "Skip unsupported version of OpenCL device: " << devices[i]->deviceName
                            << "(" << devices[i]->platform->platformName << ")" << std::endl;
                    continue; // unsupported version of device, skip it
                }
                try
                {
                    setDevice(devices[i]);
                }
                catch (...)
                {
                    std::cerr << "ERROR: Can't select OpenCL device: " << devices[i]->deviceName
                            << "(" << devices[i]->platform->platformName << ")" << std::endl;
                    goto not_found;
                }
                return true;
            }
        }
    }
not_found:
    std::cerr << "ERROR: Required OpenCL device not found, check configuration: " << (configuration == NULL ? "" : configuration) << std::endl
            << "    Platform: " << (platform.length() == 0 ? "any" : platform) << std::endl
            << "    Device types: ";
    for (size_t t = 0; t < deviceTypes.size(); t++)
    {
        std::cerr << deviceTypes[t] << " ";
    }
    std::cerr << std::endl << "    Device name: " << (deviceName.length() == 0 ? "any" : deviceName) << std::endl;
    return false;
}

static bool __initialized = false;
static int initializeOpenCLDevices()
{
    using namespace cl_utils;

    assert(!__initialized);
    __initialized = true;

    assert(global_devices.size() == 0);

    std::vector<cl_platform_id> platforms;
    try
    {
        openCLSafeCall(getPlatforms(platforms));
    }
    catch (cv::Exception&)
    {
        return 0; // OpenCL not found
    }

    global_platforms.resize(platforms.size());

    for (size_t i = 0; i < platforms.size(); ++i)
    {
        PlatformInfoImpl& platformInfo = global_platforms[i];
        platformInfo.info._id = i;

        cl_platform_id platform = platforms[i];

        platformInfo.platform_id = platform;
        openCLSafeCall(getStringInfo(clGetPlatformInfo, platform, CL_PLATFORM_PROFILE, platformInfo.info.platformProfile));
        openCLSafeCall(getStringInfo(clGetPlatformInfo, platform, CL_PLATFORM_VERSION, platformInfo.info.platformVersion));
        openCLSafeCall(getStringInfo(clGetPlatformInfo, platform, CL_PLATFORM_NAME, platformInfo.info.platformName));
        openCLSafeCall(getStringInfo(clGetPlatformInfo, platform, CL_PLATFORM_VENDOR, platformInfo.info.platformVendor));
        openCLSafeCall(getStringInfo(clGetPlatformInfo, platform, CL_PLATFORM_EXTENSIONS, platformInfo.info.platformExtensons));

        parseOpenCLVersion(platformInfo.info.platformVersion,
                platformInfo.info.platformVersionMajor, platformInfo.info.platformVersionMinor);

        std::vector<cl_device_id> devices;
        cl_int status = getDevices(platform, CL_DEVICE_TYPE_ALL, devices);
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
                cl_device_id device = devices[j];

                DeviceInfoImpl& deviceInfo = global_devices[baseIndx + j];
                deviceInfo.info._id = baseIndx + j;
                deviceInfo.platform_id = platform;
                deviceInfo.device_id = device;

                deviceInfo.info.platform = &platformInfo.info;
                platformInfo.deviceIDs[j] = deviceInfo.info._id;

                cl_device_type type = cl_device_type(-1);
                openCLSafeCall(getScalarInfo(clGetDeviceInfo, device, CL_DEVICE_TYPE, type));
                deviceInfo.info.deviceType = DeviceType(type);

                openCLSafeCall(getStringInfo(clGetDeviceInfo, device, CL_DEVICE_PROFILE, deviceInfo.info.deviceProfile));
                openCLSafeCall(getStringInfo(clGetDeviceInfo, device, CL_DEVICE_VERSION, deviceInfo.info.deviceVersion));
                openCLSafeCall(getStringInfo(clGetDeviceInfo, device, CL_DEVICE_NAME, deviceInfo.info.deviceName));
                openCLSafeCall(getStringInfo(clGetDeviceInfo, device, CL_DEVICE_VENDOR, deviceInfo.info.deviceVendor));
                cl_uint vendorID = 0;
                openCLSafeCall(getScalarInfo(clGetDeviceInfo, device, CL_DEVICE_VENDOR_ID, vendorID));
                deviceInfo.info.deviceVendorId = vendorID;
                openCLSafeCall(getStringInfo(clGetDeviceInfo, device, CL_DRIVER_VERSION, deviceInfo.info.deviceDriverVersion));
                openCLSafeCall(getStringInfo(clGetDeviceInfo, device, CL_DEVICE_EXTENSIONS, deviceInfo.info.deviceExtensions));

                parseOpenCLVersion(deviceInfo.info.deviceVersion,
                        deviceInfo.info.deviceVersionMajor, deviceInfo.info.deviceVersionMinor);

                size_t maxWorkGroupSize = 0;
                openCLSafeCall(getScalarInfo(clGetDeviceInfo, device, CL_DEVICE_MAX_WORK_GROUP_SIZE, maxWorkGroupSize));
                deviceInfo.info.maxWorkGroupSize = maxWorkGroupSize;

                cl_uint maxDimensions = 0;
                openCLSafeCall(getScalarInfo(clGetDeviceInfo, device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, maxDimensions));
                std::vector<size_t> maxWorkItemSizes(maxDimensions);
                openCLSafeCall(clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t) * maxDimensions,
                        (void *)&maxWorkItemSizes[0], 0));
                deviceInfo.info.maxWorkItemSizes = maxWorkItemSizes;

                cl_uint maxComputeUnits = 0;
                openCLSafeCall(getScalarInfo(clGetDeviceInfo, device, CL_DEVICE_MAX_COMPUTE_UNITS, maxComputeUnits));
                deviceInfo.info.maxComputeUnits = maxComputeUnits;

                cl_ulong localMemorySize = 0;
                openCLSafeCall(getScalarInfo(clGetDeviceInfo, device, CL_DEVICE_LOCAL_MEM_SIZE, localMemorySize));
                deviceInfo.info.localMemorySize = (size_t)localMemorySize;

                cl_ulong maxMemAllocSize = 0;
                openCLSafeCall(getScalarInfo(clGetDeviceInfo, device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, maxMemAllocSize));
                deviceInfo.info.maxMemAllocSize = (size_t)maxMemAllocSize;

                cl_bool unifiedMemory = false;
                openCLSafeCall(getScalarInfo(clGetDeviceInfo, device, CL_DEVICE_HOST_UNIFIED_MEMORY, unifiedMemory));
                deviceInfo.info.isUnifiedMemory = unifiedMemory != 0;

                //initialize extra options for compilation. Currently only fp64 is included.
                //Assume 4KB is enough to store all possible extensions.
                openCLSafeCall(getStringInfo(clGetDeviceInfo, device, CL_DEVICE_EXTENSIONS, deviceInfo.info.deviceExtensions));

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

                size_t intel_platform = platformInfo.info.platformVendor.find("Intel");
                if(intel_platform != std::string::npos)
                {
                    deviceInfo.info.compilationExtraOptions += " -D INTEL_DEVICE";
                    deviceInfo.info.isIntelDevice = true;
                }
                else
                {
                    deviceInfo.info.isIntelDevice = false;
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
      maxWorkGroupSize(0), maxComputeUnits(0), localMemorySize(0), maxMemAllocSize(0),
      deviceVersionMajor(0), deviceVersionMinor(0),
      haveDoubleSupport(false), isUnifiedMemory(false),isIntelDevice(false),
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
    static void setContext(const DeviceInfo* deviceInfo);

    bool supportsFeature(FEATURE_TYPE featureType) const;

    static void cleanupContext(void);

private:
    ContextImpl(const ContextImpl&); // disabled
    ContextImpl& operator=(const ContextImpl&); // disabled
};

static ContextImpl* currentContext = NULL;

Context* Context::getContext()
{
    if (currentContext == NULL)
    {
        if (!__initialized || !__deviceSelected)
        {
            cv::AutoLock lock(getInitializationMutex());
            if (!__initialized)
            {
                if (initializeOpenCLDevices() == 0)
                {
                    CV_Error(Error::OpenCLInitError, "OpenCL not available");
                }
            }
            if (!__deviceSelected)
            {
                if (!selectOpenCLDevice())
                {
                    CV_Error(Error::OpenCLInitError, "Can't select OpenCL device");
                }
            }
        }
        CV_Assert(currentContext != NULL);
    }
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
    case FEATURE_CL_INTEL_DEVICE:
        return deviceInfo.isIntelDevice;
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

void fft_teardown();
void clBlasTeardown();

void ContextImpl::cleanupContext(void)
{
    fft_teardown();
    clBlasTeardown();

    cv::AutoLock lock(__module.currentContextMutex);
    if (currentContext)
        delete currentContext;
    currentContext = NULL;
}

void ContextImpl::setContext(const DeviceInfo* deviceInfo)
{
    CV_Assert(deviceInfo->_id >= 0 && deviceInfo->_id < (int)global_devices.size());

    {
        cv::AutoLock lock(__module.currentContextMutex);
        if (currentContext)
        {
            if (currentContext->deviceInfo._id == deviceInfo->_id)
                return;
        }
    }

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
        cv::AutoLock lock(__module.currentContextMutex);
        old = currentContext;
        currentContext = ctx;
    }
    if (old != NULL)
    {
        delete old;
    }
}

int getOpenCLPlatforms(PlatformsInfo& platforms)
{
    if (!__initialized)
        initializeOpenCLDevices();

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
    if (!__initialized)
        initializeOpenCLDevices();

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
            if (((int)deviceInfo.info.deviceType & deviceType) != 0)
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
    if (!__deviceSelected)
        __deviceSelected = true;

    ContextImpl::setContext(info);
}

bool supportsFeature(FEATURE_TYPE featureType)
{
    return Context::getContext()->supportsFeature(featureType);
}

__Module::__Module()
{
    /* moved to Context::getContext(): initializeOpenCLDevices(); */
}

__Module::~__Module()
{
#if defined(WIN32) && defined(CVAPI_EXPORTS)
    // nothing, see DllMain
#else
    ContextImpl::cleanupContext();
#endif
}

} // namespace ocl
} // namespace cv


#if defined(WIN32) && defined(CVAPI_EXPORTS)

extern "C"
BOOL WINAPI DllMain(HINSTANCE /*hInst*/, DWORD fdwReason, LPVOID lpReserved);

extern "C"
BOOL WINAPI DllMain(HINSTANCE /*hInst*/, DWORD fdwReason, LPVOID lpReserved)
{
    if (fdwReason == DLL_PROCESS_DETACH)
    {
        if (lpReserved != NULL) // called after ExitProcess() call
            cv::ocl::__termination = true;
        cv::ocl::ContextImpl::cleanupContext();
    }
    return TRUE;
}

#endif
