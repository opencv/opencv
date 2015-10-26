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
// Copyright (C) 2010-2014, Advanced Micro Devices, Inc., all rights reserved.
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
#include <stdlib.h>
#include <ctype.h>
#include <iomanip>
#include <fstream>
#include "cl_programcache.hpp"

#include "opencv2/ocl/private/opencl_utils.hpp"

namespace cv {
namespace ocl {

using namespace cl_utils;

#if defined(WIN32)
static bool __termination = false;
#endif

struct __Module
{
    __Module();
    ~__Module();
    cv::Mutex initializationMutex;
    cv::Mutex currentContextMutex;
};

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

struct PlatformInfoImpl : public PlatformInfo
{
    cl_platform_id platform_id;

    std::vector<int> deviceIDs;

    PlatformInfoImpl()
        : platform_id(NULL)
    {
    }

    void init(int id, cl_platform_id platform)
    {
        CV_Assert(platform_id == NULL);

        this->_id = id;
        platform_id = platform;

        openCLSafeCall(getStringInfo(clGetPlatformInfo, platform, CL_PLATFORM_PROFILE, this->platformProfile));
        openCLSafeCall(getStringInfo(clGetPlatformInfo, platform, CL_PLATFORM_VERSION, this->platformVersion));
        openCLSafeCall(getStringInfo(clGetPlatformInfo, platform, CL_PLATFORM_NAME, this->platformName));
        openCLSafeCall(getStringInfo(clGetPlatformInfo, platform, CL_PLATFORM_VENDOR, this->platformVendor));
        openCLSafeCall(getStringInfo(clGetPlatformInfo, platform, CL_PLATFORM_EXTENSIONS, this->platformExtensons));

        parseOpenCLVersion(this->platformVersion,
                this->platformVersionMajor, this->platformVersionMinor);
    }

};

struct DeviceInfoImpl: public DeviceInfo
{
    cl_platform_id platform_id;
    cl_device_id device_id;

    DeviceInfoImpl()
        : platform_id(NULL), device_id(NULL)
    {
    }

    void init(int id, PlatformInfoImpl& platformInfoImpl, cl_device_id device)
    {
        CV_Assert(device_id == NULL);

        this->_id = id;
        platform_id = platformInfoImpl.platform_id;
        device_id = device;

        this->platform = &platformInfoImpl;

        cl_device_type type = cl_device_type(-1);
        openCLSafeCall(getScalarInfo(clGetDeviceInfo, device, CL_DEVICE_TYPE, type));
        this->deviceType = DeviceType(type);

        openCLSafeCall(getStringInfo(clGetDeviceInfo, device, CL_DEVICE_PROFILE, this->deviceProfile));
        openCLSafeCall(getStringInfo(clGetDeviceInfo, device, CL_DEVICE_VERSION, this->deviceVersion));
        openCLSafeCall(getStringInfo(clGetDeviceInfo, device, CL_DEVICE_NAME, this->deviceName));
        openCLSafeCall(getStringInfo(clGetDeviceInfo, device, CL_DEVICE_VENDOR, this->deviceVendor));
        cl_uint vendorID = 0;
        openCLSafeCall(getScalarInfo(clGetDeviceInfo, device, CL_DEVICE_VENDOR_ID, vendorID));
        this->deviceVendorId = vendorID;
        openCLSafeCall(getStringInfo(clGetDeviceInfo, device, CL_DRIVER_VERSION, this->deviceDriverVersion));
        openCLSafeCall(getStringInfo(clGetDeviceInfo, device, CL_DEVICE_EXTENSIONS, this->deviceExtensions));

        parseOpenCLVersion(this->deviceVersion,
                this->deviceVersionMajor, this->deviceVersionMinor);

        size_t maxWGS = 0;
        openCLSafeCall(getScalarInfo(clGetDeviceInfo, device, CL_DEVICE_MAX_WORK_GROUP_SIZE, maxWGS));
        this->maxWorkGroupSize = maxWGS;

        cl_uint maxDimensions = 0;
        openCLSafeCall(getScalarInfo(clGetDeviceInfo, device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, maxDimensions));
        std::vector<size_t> maxWIS(maxDimensions);
        openCLSafeCall(clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t) * maxDimensions,
                (void *)&maxWIS[0], 0));
        this->maxWorkItemSizes = maxWIS;

        cl_uint maxCU = 0;
        openCLSafeCall(getScalarInfo(clGetDeviceInfo, device, CL_DEVICE_MAX_COMPUTE_UNITS, maxCU));
        this->maxComputeUnits = maxCU;

        cl_ulong localMS = 0;
        openCLSafeCall(getScalarInfo(clGetDeviceInfo, device, CL_DEVICE_LOCAL_MEM_SIZE, localMS));
        this->localMemorySize = (size_t)localMS;

        cl_ulong maxMAS = 0;
        openCLSafeCall(getScalarInfo(clGetDeviceInfo, device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, maxMAS));
        this->maxMemAllocSize = (size_t)maxMAS;

        cl_bool unifiedMemory = false;
        openCLSafeCall(getScalarInfo(clGetDeviceInfo, device, CL_DEVICE_HOST_UNIFIED_MEMORY, unifiedMemory));
        this->isUnifiedMemory = unifiedMemory != 0;

        //initialize extra options for compilation. Currently only fp64 is included.
        //Assume 4KB is enough to store all possible extensions.
        openCLSafeCall(getStringInfo(clGetDeviceInfo, device, CL_DEVICE_EXTENSIONS, this->deviceExtensions));

        size_t fp64_khr = this->deviceExtensions.find("cl_khr_fp64");
        if(fp64_khr != std::string::npos)
        {
            this->compilationExtraOptions += "-D DOUBLE_SUPPORT";
            this->haveDoubleSupport = true;
        }
        else
        {
            this->haveDoubleSupport = false;
        }

        size_t intel_platform = platformInfoImpl.platformVendor.find("Intel");
        if(intel_platform != std::string::npos)
        {
            this->compilationExtraOptions += " -D INTEL_DEVICE";
            this->isIntelDevice = true;
        }
        else
        {
            this->isIntelDevice = false;
        }

        if (id < 0)
        {
#ifdef CL_VERSION_1_2
            if (this->deviceVersionMajor > 1 || (this->deviceVersionMajor == 1 && this->deviceVersionMinor >= 2))
            {
                ::clRetainDevice(device);
            }
#endif
        }
    }
};

static std::vector<PlatformInfoImpl> global_platforms;
static std::vector<DeviceInfoImpl> global_devices;
static __Module __module;

cv::Mutex& getInitializationMutex()
{
    return __module.initializationMutex;
}

static cv::Mutex& getCurrentContextMutex()
{
    return __module.currentContextMutex;
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

static bool selectOpenCLDevice()
{
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

        cl_platform_id platform = platforms[i];
        platformInfo.init(i, platform);

        std::vector<cl_device_id> devices;
        cl_int status = getDevices(platform, CL_DEVICE_TYPE_ALL, devices);
        if(status != CL_DEVICE_NOT_FOUND)
            openCLVerifyCall(status);

        if(devices.size() > 0)
        {
            int baseIndx = global_devices.size();
            global_devices.resize(baseIndx + devices.size());
            platformInfo.deviceIDs.resize(devices.size());
            platformInfo.devices.resize(devices.size());

            for(size_t j = 0; j < devices.size(); ++j)
            {
                cl_device_id device = devices[j];

                DeviceInfoImpl& deviceInfo = global_devices[baseIndx + j];
                platformInfo.deviceIDs[j] = baseIndx + j;
                deviceInfo.init(baseIndx + j, platformInfo, device);
            }
        }
    }

    for (size_t i = 0; i < platforms.size(); ++i)
    {
        PlatformInfoImpl& platformInfo = global_platforms[i];
        for(size_t j = 0; j < platformInfo.deviceIDs.size(); ++j)
        {
            DeviceInfoImpl& deviceInfo = global_devices[platformInfo.deviceIDs[j]];
            platformInfo.devices[j] = &deviceInfo;
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

DeviceInfo::~DeviceInfo() { }

PlatformInfo::PlatformInfo()
    : _id(-1),
      platformVersionMajor(0), platformVersionMinor(0)
{
    // nothing
}

PlatformInfo::~PlatformInfo() { }

class ContextImpl;

struct CommandQueue
{
    ContextImpl* context_;
    cl_command_queue clQueue_;

    CommandQueue() : context_(NULL), clQueue_(NULL) { }
    ~CommandQueue() { release(); }

    void create(ContextImpl* context_);
    void release()
    {
#ifdef WIN32
        // if process is on termination stage (ExitProcess was called and other threads were terminated)
        // then disable command queue release because it may cause program hang
        if (!__termination)
#endif
        {
            if(clQueue_)
            {
                openCLSafeCall(clReleaseCommandQueue(clQueue_)); // some cleanup problems are here
            }

        }
        clQueue_ = NULL;
        context_ = NULL;
    }
};

cv::TLSData<CommandQueue> commandQueueTLSData;

//////////////////////////////// OpenCL context ////////////////////////
//This is a global singleton class used to represent a OpenCL context.
class ContextImpl : public Context
{
public:
    cl_device_id clDeviceID;
    cl_context clContext;
    const DeviceInfoImpl& deviceInfoImpl;

protected:
    ContextImpl(const DeviceInfoImpl& _deviceInfoImpl, cl_context context)
        : clDeviceID(_deviceInfoImpl.device_id), clContext(context), deviceInfoImpl(_deviceInfoImpl)
    {
#ifdef CL_VERSION_1_2
        if (supportsFeature(FEATURE_CL_VER_1_2))
        {
            openCLSafeCall(clRetainDevice(clDeviceID));
        }
#endif
        openCLSafeCall(clRetainContext(clContext));

        ContextImpl* old = NULL;
        {
            cv::AutoLock lock(getCurrentContextMutex());
            old = currentContext;
            currentContext = this;
        }
        if (old != NULL)
        {
            delete old;
        }
    }
    ~ContextImpl()
    {
        CV_Assert(this != currentContext);

#ifdef CL_VERSION_1_2
#ifdef WIN32
        // if process is on termination stage (ExitProcess was called and other threads were terminated)
        // then disable device release because it may cause program hang
        if (!__termination)
#endif
        {
            if (supportsFeature(FEATURE_CL_VER_1_2))
            {
                openCLSafeCall(clReleaseDevice(clDeviceID));
            }
        }
#endif
        if (deviceInfoImpl._id < 0) // not in the global registry, so we should cleanup it
        {
#ifdef CL_VERSION_1_2
            if (supportsFeature(FEATURE_CL_VER_1_2))
            {
                openCLSafeCall(clReleaseDevice(deviceInfoImpl.device_id));
            }
#endif
            PlatformInfoImpl* platformImpl = (PlatformInfoImpl*)(deviceInfoImpl.platform);
            delete platformImpl;
            delete const_cast<DeviceInfoImpl*>(&deviceInfoImpl);
        }
        clDeviceID = NULL;

#ifdef WIN32
        // if process is on termination stage (ExitProcess was called and other threads were terminated)
        // then disable context release because it may cause program hang
        if (!__termination)
#endif
        {
            if(clContext)
            {
                openCLSafeCall(clReleaseContext(clContext));
            }
        }
        clContext = NULL;
    }
public:
    static void setContext(const DeviceInfo* deviceInfo);
    static void initializeContext(void* pClPlatform, void* pClContext, void* pClDevice);

    bool supportsFeature(FEATURE_TYPE featureType) const;

    static void cleanupContext(void);

    static ContextImpl* getContext();
private:
    ContextImpl(const ContextImpl&); // disabled
    ContextImpl& operator=(const ContextImpl&); // disabled

    static ContextImpl* currentContext;
};

ContextImpl* ContextImpl::currentContext = NULL;

static bool __deviceSelected = false;

Context* Context::getContext()
{
    return ContextImpl::getContext();
}

ContextImpl* ContextImpl::getContext()
{
    if (currentContext == NULL)
    {
        static bool defaultInitiaization = false;
        if (!defaultInitiaization)
        {
            cv::AutoLock lock(getInitializationMutex());
            try
            {
                if (!__initialized)
                {
                    if (initializeOpenCLDevices() == 0)
                    {
                        CV_Error(CV_OpenCLInitError, "OpenCL not available");
                    }
                }
                if (!__deviceSelected)
                {
                    if (!selectOpenCLDevice())
                    {
                        CV_Error(CV_OpenCLInitError, "Can't select OpenCL device");
                    }
                }
                defaultInitiaization = true;
            }
            catch (...)
            {
                defaultInitiaization = true;
                throw;
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
    return ((ContextImpl*)this)->deviceInfoImpl;
}

const void* Context::getOpenCLContextPtr() const
{
    return &(((ContextImpl*)this)->clContext);
}

const void* Context::getOpenCLCommandQueuePtr() const
{
    ContextImpl* pThis = (ContextImpl*)this;
    CommandQueue* commandQueue = commandQueueTLSData.get();
    if (commandQueue->context_ != pThis)
    {
        commandQueue->create(pThis);
    }
    return &commandQueue->clQueue_;
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
        return deviceInfoImpl.isIntelDevice;
    case FEATURE_CL_DOUBLE:
        return deviceInfoImpl.haveDoubleSupport;
    case FEATURE_CL_UNIFIED_MEM:
        return deviceInfoImpl.isUnifiedMemory;
    case FEATURE_CL_VER_1_2:
        return deviceInfoImpl.deviceVersionMajor > 1 || (deviceInfoImpl.deviceVersionMajor == 1 && deviceInfoImpl.deviceVersionMinor >= 2);
    }
    CV_Error(CV_StsBadArg, "Invalid feature type");
    return false;
}

void fft_teardown();
void clBlasTeardown();

void ContextImpl::cleanupContext(void)
{
    fft_teardown();
    clBlasTeardown();

    cv::AutoLock lock(getCurrentContextMutex());
    if (currentContext)
    {
        ContextImpl* ctx = currentContext;
        currentContext = NULL;
        delete ctx;
    }
}

void ContextImpl::setContext(const DeviceInfo* deviceInfo)
{
    CV_Assert(deviceInfo->_id >= 0); // we can't specify custom devices
    CV_Assert(deviceInfo->_id < (int)global_devices.size());

    {
        cv::AutoLock lock(getCurrentContextMutex());
        if (currentContext)
        {
            if (currentContext->deviceInfoImpl._id == deviceInfo->_id)
                return;
        }
    }

    DeviceInfoImpl& infoImpl = global_devices[deviceInfo->_id];
    CV_Assert(deviceInfo == &infoImpl);

    cl_int status = 0;
    cl_context_properties cps[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)(infoImpl.platform_id), 0 };
    cl_context clContext = clCreateContext(cps, 1, &infoImpl.device_id, NULL, NULL, &status);
    openCLVerifyCall(status);

    ContextImpl* ctx = new ContextImpl(infoImpl, clContext);
    clReleaseContext(clContext);
    (void)ctx;
}

void ContextImpl::initializeContext(void* pClPlatform, void* pClContext, void* pClDevice)
{
    CV_Assert(pClPlatform != NULL);
    CV_Assert(pClContext != NULL);
    CV_Assert(pClDevice != NULL);
    cl_platform_id platform = *(cl_platform_id*)pClPlatform;
    cl_context context = *(cl_context*)pClContext;
    cl_device_id device = *(cl_device_id*)pClDevice;

    PlatformInfoImpl* platformInfoImpl = new PlatformInfoImpl();
    platformInfoImpl->init(-1, platform);
    DeviceInfoImpl* deviceInfoImpl = new DeviceInfoImpl();
    deviceInfoImpl->init(-1, *platformInfoImpl, device);

    ContextImpl* ctx = new ContextImpl(*deviceInfoImpl, context);
    (void)ctx;
}

void CommandQueue::create(ContextImpl* context)
{
    release();
    cl_int status = 0;
    // TODO add CL_QUEUE_PROFILING_ENABLE
    cl_command_queue clCmdQueue = clCreateCommandQueue(context->clContext, context->clDeviceID, 0, &status);
    openCLVerifyCall(status);
    context_ = context;
    clQueue_ = clCmdQueue;
}

int getOpenCLPlatforms(PlatformsInfo& platforms)
{
    if (!__initialized)
        initializeOpenCLDevices();

    platforms.clear();

    for (size_t id = 0; id < global_platforms.size(); ++id)
    {
        PlatformInfoImpl& impl = global_platforms[id];
        platforms.push_back(&impl);
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
            if (((int)deviceInfo.deviceType & deviceType) != 0)
            {
                devices.push_back(&deviceInfo);
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
    try
    {
        ContextImpl::setContext(info);
        __deviceSelected = true;
    }
    catch (...)
    {
        __deviceSelected = true;
        throw;
    }
}

void initializeContext(void* pClPlatform, void* pClContext, void* pClDevice)
{
    try
    {
        ContextImpl::initializeContext(pClPlatform, pClContext, pClDevice);
        __deviceSelected = true;
    }
    catch (...)
    {
        __deviceSelected = true;
        throw;
    }
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
