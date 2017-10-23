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
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
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
// In no event shall the OpenCV Foundation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"
#include <list>
#include <map>
#include <deque>
#include <set>
#include <string>
#include <sstream>
#include <iostream> // std::cerr
#if !(defined _MSC_VER) || (defined _MSC_VER && _MSC_VER > 1700)
#include <inttypes.h>
#endif

#include <opencv2/core/utils/configuration.private.hpp>

#include "opencv2/core/ocl_genbase.hpp"
#include "opencl_kernels_core.hpp"

#define CV_OPENCL_ALWAYS_SHOW_BUILD_LOG 0
#define CV_OPENCL_SHOW_RUN_ERRORS       0
#define CV_OPENCL_SHOW_SVM_ERROR_LOG    1
#define CV_OPENCL_SHOW_SVM_LOG          0

#include "opencv2/core/bufferpool.hpp"
#ifndef LOG_BUFFER_POOL
# if 0
#   define LOG_BUFFER_POOL printf
# else
#   define LOG_BUFFER_POOL(...)
# endif
#endif

#if CV_OPENCL_SHOW_SVM_LOG
// TODO add timestamp logging
#define CV_OPENCL_SVM_TRACE_P printf("line %d (ocl.cpp): ", __LINE__); printf
#else
#define CV_OPENCL_SVM_TRACE_P(...)
#endif

#if CV_OPENCL_SHOW_SVM_ERROR_LOG
// TODO add timestamp logging
#define CV_OPENCL_SVM_TRACE_ERROR_P printf("Error on line %d (ocl.cpp): ", __LINE__); printf
#else
#define CV_OPENCL_SVM_TRACE_ERROR_P(...)
#endif

#include "opencv2/core/opencl/runtime/opencl_clamdblas.hpp"
#include "opencv2/core/opencl/runtime/opencl_clamdfft.hpp"

#ifdef HAVE_OPENCL
#include "opencv2/core/opencl/runtime/opencl_core.hpp"
#else
// TODO FIXIT: This file can't be build without OPENCL
#include "ocl_deprecated.hpp"
#endif // HAVE_OPENCL

#ifdef _DEBUG
#define CV_OclDbgAssert CV_DbgAssert
#else
static bool isRaiseError()
{
    static bool initialized = false;
    static bool value = false;
    if (!initialized)
    {
        value = cv::utils::getConfigurationParameterBool("OPENCV_OPENCL_RAISE_ERROR", false);
        initialized = true;
    }
    return value;
}
#define CV_OclDbgAssert(expr) do { if (isRaiseError()) { CV_Assert(expr); } else { (void)(expr); } } while ((void)0, 0)
#endif

#ifdef HAVE_OPENCL_SVM
#include "opencv2/core/opencl/runtime/opencl_svm_20.hpp"
#include "opencv2/core/opencl/runtime/opencl_svm_hsa_extension.hpp"
#include "opencv2/core/opencl/opencl_svm.hpp"
#endif

namespace cv { namespace ocl {

struct UMat2D
{
    UMat2D(const UMat& m)
    {
        offset = (int)m.offset;
        step = (int)m.step;
        rows = m.rows;
        cols = m.cols;
    }
    int offset;
    int step;
    int rows;
    int cols;
};

struct UMat3D
{
    UMat3D(const UMat& m)
    {
        offset = (int)m.offset;
        step = (int)m.step.p[1];
        slicestep = (int)m.step.p[0];
        slices = (int)m.size.p[0];
        rows = m.size.p[1];
        cols = m.size.p[2];
    }
    int offset;
    int slicestep;
    int step;
    int slices;
    int rows;
    int cols;
};

// Computes 64-bit "cyclic redundancy check" sum, as specified in ECMA-182
static uint64 crc64( const uchar* data, size_t size, uint64 crc0=0 )
{
    static uint64 table[256];
    static bool initialized = false;

    if( !initialized )
    {
        for( int i = 0; i < 256; i++ )
        {
            uint64 c = i;
            for( int j = 0; j < 8; j++ )
                c = ((c & 1) ? CV_BIG_UINT(0xc96c5795d7870f42) : 0) ^ (c >> 1);
            table[i] = c;
        }
        initialized = true;
    }

    uint64 crc = ~crc0;
    for( size_t idx = 0; idx < size; idx++ )
        crc = table[(uchar)crc ^ data[idx]] ^ (crc >> 8);

    return ~crc;
}

bool haveOpenCL()
{
#ifdef HAVE_OPENCL
    static bool g_isOpenCLInitialized = false;
    static bool g_isOpenCLAvailable = false;

    if (!g_isOpenCLInitialized)
    {
        try
        {
            cl_uint n = 0;
            g_isOpenCLAvailable = ::clGetPlatformIDs(0, NULL, &n) == CL_SUCCESS;
        }
        catch (...)
        {
            g_isOpenCLAvailable = false;
        }
        g_isOpenCLInitialized = true;
    }
    return g_isOpenCLAvailable;
#else
    return false;
#endif
}

bool useOpenCL()
{
    CoreTLSData* data = getCoreTlsData().get();
    if( data->useOpenCL < 0 )
    {
        try
        {
            data->useOpenCL = (int)haveOpenCL() && Device::getDefault().ptr() && Device::getDefault().available();
        }
        catch (...)
        {
            data->useOpenCL = 0;
        }
    }
    return data->useOpenCL > 0;
}

void setUseOpenCL(bool flag)
{
    if( haveOpenCL() )
    {
        CoreTLSData* data = getCoreTlsData().get();
        data->useOpenCL = (flag && Device::getDefault().ptr() != NULL) ? 1 : 0;
    }
}

#ifdef HAVE_CLAMDBLAS

class AmdBlasHelper
{
public:
    static AmdBlasHelper & getInstance()
    {
        CV_SINGLETON_LAZY_INIT_REF(AmdBlasHelper, new AmdBlasHelper())
    }

    bool isAvailable() const
    {
        return g_isAmdBlasAvailable;
    }

    ~AmdBlasHelper()
    {
        try
        {
            clAmdBlasTeardown();
        }
        catch (...) { }
    }

protected:
    AmdBlasHelper()
    {
        if (!g_isAmdBlasInitialized)
        {
            AutoLock lock(getInitializationMutex());

            if (!g_isAmdBlasInitialized)
            {
                if (haveOpenCL())
                {
                    try
                    {
                        g_isAmdBlasAvailable = clAmdBlasSetup() == clAmdBlasSuccess;
                    }
                    catch (...)
                    {
                        g_isAmdBlasAvailable = false;
                    }
                }
                else
                    g_isAmdBlasAvailable = false;

                g_isAmdBlasInitialized = true;
            }
        }
    }

private:
    static bool g_isAmdBlasInitialized;
    static bool g_isAmdBlasAvailable;
};

bool AmdBlasHelper::g_isAmdBlasAvailable = false;
bool AmdBlasHelper::g_isAmdBlasInitialized = false;

bool haveAmdBlas()
{
    return AmdBlasHelper::getInstance().isAvailable();
}

#else

bool haveAmdBlas()
{
    return false;
}

#endif

#ifdef HAVE_CLAMDFFT

class AmdFftHelper
{
public:
    static AmdFftHelper & getInstance()
    {
        CV_SINGLETON_LAZY_INIT_REF(AmdFftHelper, new AmdFftHelper())
    }

    bool isAvailable() const
    {
        return g_isAmdFftAvailable;
    }

    ~AmdFftHelper()
    {
        try
        {
//            clAmdFftTeardown();
        }
        catch (...) { }
    }

protected:
    AmdFftHelper()
    {
        if (!g_isAmdFftInitialized)
        {
            AutoLock lock(getInitializationMutex());

            if (!g_isAmdFftInitialized)
            {
                if (haveOpenCL())
                {
                    try
                    {
                        cl_uint major, minor, patch;
                        CV_Assert(clAmdFftInitSetupData(&setupData) == CLFFT_SUCCESS);

                        // it throws exception in case AmdFft binaries are not found
                        CV_Assert(clAmdFftGetVersion(&major, &minor, &patch) == CLFFT_SUCCESS);
                        g_isAmdFftAvailable = true;
                    }
                    catch (const Exception &)
                    {
                        g_isAmdFftAvailable = false;
                    }
                }
                else
                    g_isAmdFftAvailable = false;

                g_isAmdFftInitialized = true;
            }
        }
    }

private:
    static clAmdFftSetupData setupData;
    static bool g_isAmdFftInitialized;
    static bool g_isAmdFftAvailable;
};

clAmdFftSetupData AmdFftHelper::setupData;
bool AmdFftHelper::g_isAmdFftAvailable = false;
bool AmdFftHelper::g_isAmdFftInitialized = false;

bool haveAmdFft()
{
    return AmdFftHelper::getInstance().isAvailable();
}

#else

bool haveAmdFft()
{
    return false;
}

#endif

bool haveSVM()
{
#ifdef HAVE_OPENCL_SVM
    return true;
#else
    return false;
#endif
}

void finish()
{
    Queue::getDefault().finish();
}

#define IMPLEMENT_REFCOUNTABLE() \
    void addref() { CV_XADD(&refcount, 1); } \
    void release() { if( CV_XADD(&refcount, -1) == 1 && !cv::__termination) delete this; } \
    int refcount

/////////////////////////////////////////// Platform /////////////////////////////////////////////

struct Platform::Impl
{
    Impl()
    {
        refcount = 1;
        handle = 0;
        initialized = false;
    }

    ~Impl() {}

    void init()
    {
        if( !initialized )
        {
            //cl_uint num_entries
            cl_uint n = 0;
            if( clGetPlatformIDs(1, &handle, &n) != CL_SUCCESS || n == 0 )
                handle = 0;
            if( handle != 0 )
            {
                char buf[1000];
                size_t len = 0;
                CV_OclDbgAssert(clGetPlatformInfo(handle, CL_PLATFORM_VENDOR, sizeof(buf), buf, &len) == CL_SUCCESS);
                buf[len] = '\0';
                vendor = String(buf);
            }

            initialized = true;
        }
    }

    IMPLEMENT_REFCOUNTABLE();

    cl_platform_id handle;
    String vendor;
    bool initialized;
};

Platform::Platform()
{
    p = 0;
}

Platform::~Platform()
{
    if(p)
        p->release();
}

Platform::Platform(const Platform& pl)
{
    p = (Impl*)pl.p;
    if(p)
        p->addref();
}

Platform& Platform::operator = (const Platform& pl)
{
    Impl* newp = (Impl*)pl.p;
    if(newp)
        newp->addref();
    if(p)
        p->release();
    p = newp;
    return *this;
}

void* Platform::ptr() const
{
    return p ? p->handle : 0;
}

Platform& Platform::getDefault()
{
    static Platform p;
    if( !p.p )
    {
        p.p = new Impl;
        p.p->init();
    }
    return p;
}

/////////////////////////////////////// Device ////////////////////////////////////////////

// deviceVersion has format
//   OpenCL<space><major_version.minor_version><space><vendor-specific information>
// by specification
//   http://www.khronos.org/registry/cl/sdk/1.1/docs/man/xhtml/clGetDeviceInfo.html
//   http://www.khronos.org/registry/cl/sdk/1.2/docs/man/xhtml/clGetDeviceInfo.html
static void parseDeviceVersion(const String &deviceVersion, int &major, int &minor)
{
    major = minor = 0;
    if (10 >= deviceVersion.length())
        return;
    const char *pstr = deviceVersion.c_str();
    if (0 != strncmp(pstr, "OpenCL ", 7))
        return;
    size_t ppos = deviceVersion.find('.', 7);
    if (String::npos == ppos)
        return;
    String temp = deviceVersion.substr(7, ppos - 7);
    major = atoi(temp.c_str());
    temp = deviceVersion.substr(ppos + 1);
    minor = atoi(temp.c_str());
}

struct Device::Impl
{
    Impl(void* d)
    {
        handle = (cl_device_id)d;
        refcount = 1;

        name_ = getStrProp(CL_DEVICE_NAME);
        version_ = getStrProp(CL_DEVICE_VERSION);
        extensions_ = getStrProp(CL_DEVICE_EXTENSIONS);
        doubleFPConfig_ = getProp<cl_device_fp_config, int>(CL_DEVICE_DOUBLE_FP_CONFIG);
        hostUnifiedMemory_ = getBoolProp(CL_DEVICE_HOST_UNIFIED_MEMORY);
        maxComputeUnits_ = getProp<cl_uint, int>(CL_DEVICE_MAX_COMPUTE_UNITS);
        maxWorkGroupSize_ = getProp<size_t, size_t>(CL_DEVICE_MAX_WORK_GROUP_SIZE);
        type_ = getProp<cl_device_type, int>(CL_DEVICE_TYPE);
        driverVersion_ = getStrProp(CL_DRIVER_VERSION);

        String deviceVersion_ = getStrProp(CL_DEVICE_VERSION);
        parseDeviceVersion(deviceVersion_, deviceVersionMajor_, deviceVersionMinor_);

        size_t pos = 0;
        while (pos < extensions_.size())
        {
            size_t pos2 = extensions_.find(' ', pos);
            if (pos2 == String::npos)
                pos2 = extensions_.size();
            if (pos2 > pos)
            {
                std::string extensionName = extensions_.substr(pos, pos2 - pos);
                extensions_set_.insert(extensionName);
            }
            pos = pos2 + 1;
        }

        intelSubgroupsSupport_ = isExtensionSupported("cl_intel_subgroups");

        vendorName_ = getStrProp(CL_DEVICE_VENDOR);
        if (vendorName_ == "Advanced Micro Devices, Inc." ||
            vendorName_ == "AMD")
            vendorID_ = VENDOR_AMD;
        else if (vendorName_ == "Intel(R) Corporation" || vendorName_ == "Intel" || strstr(name_.c_str(), "Iris") != 0)
            vendorID_ = VENDOR_INTEL;
        else if (vendorName_ == "NVIDIA Corporation")
            vendorID_ = VENDOR_NVIDIA;
        else
            vendorID_ = UNKNOWN_VENDOR;
    }

    template<typename _TpCL, typename _TpOut>
    _TpOut getProp(cl_device_info prop) const
    {
        _TpCL temp=_TpCL();
        size_t sz = 0;

        return clGetDeviceInfo(handle, prop, sizeof(temp), &temp, &sz) == CL_SUCCESS &&
            sz == sizeof(temp) ? _TpOut(temp) : _TpOut();
    }

    bool getBoolProp(cl_device_info prop) const
    {
        cl_bool temp = CL_FALSE;
        size_t sz = 0;

        return clGetDeviceInfo(handle, prop, sizeof(temp), &temp, &sz) == CL_SUCCESS &&
            sz == sizeof(temp) ? temp != 0 : false;
    }

    String getStrProp(cl_device_info prop) const
    {
        char buf[1024];
        size_t sz=0;
        return clGetDeviceInfo(handle, prop, sizeof(buf)-16, buf, &sz) == CL_SUCCESS &&
            sz < sizeof(buf) ? String(buf) : String();
    }

    bool isExtensionSupported(const std::string& extensionName) const
    {
        return extensions_set_.count(extensionName) > 0;
    }


    IMPLEMENT_REFCOUNTABLE();

    cl_device_id handle;

    String name_;
    String version_;
    std::string extensions_;
    int doubleFPConfig_;
    bool hostUnifiedMemory_;
    int maxComputeUnits_;
    size_t maxWorkGroupSize_;
    int type_;
    int deviceVersionMajor_;
    int deviceVersionMinor_;
    String driverVersion_;
    String vendorName_;
    int vendorID_;
    bool intelSubgroupsSupport_;

    std::set<std::string> extensions_set_;
};


Device::Device()
{
    p = 0;
}

Device::Device(void* d)
{
    p = 0;
    set(d);
}

Device::Device(const Device& d)
{
    p = d.p;
    if(p)
        p->addref();
}

Device& Device::operator = (const Device& d)
{
    Impl* newp = (Impl*)d.p;
    if(newp)
        newp->addref();
    if(p)
        p->release();
    p = newp;
    return *this;
}

Device::~Device()
{
    if(p)
        p->release();
}

void Device::set(void* d)
{
    if(p)
        p->release();
    p = new Impl(d);
}

void* Device::ptr() const
{
    return p ? p->handle : 0;
}

String Device::name() const
{ return p ? p->name_ : String(); }

String Device::extensions() const
{ return p ? String(p->extensions_) : String(); }

bool Device::isExtensionSupported(const String& extensionName) const
{ return p ? p->isExtensionSupported(extensionName) : false; }

String Device::version() const
{ return p ? p->version_ : String(); }

String Device::vendorName() const
{ return p ? p->vendorName_ : String(); }

int Device::vendorID() const
{ return p ? p->vendorID_ : 0; }

String Device::OpenCL_C_Version() const
{ return p ? p->getStrProp(CL_DEVICE_OPENCL_C_VERSION) : String(); }

String Device::OpenCLVersion() const
{ return p ? p->getStrProp(CL_DEVICE_VERSION) : String(); }

int Device::deviceVersionMajor() const
{ return p ? p->deviceVersionMajor_ : 0; }

int Device::deviceVersionMinor() const
{ return p ? p->deviceVersionMinor_ : 0; }

String Device::driverVersion() const
{ return p ? p->driverVersion_ : String(); }

int Device::type() const
{ return p ? p->type_ : 0; }

int Device::addressBits() const
{ return p ? p->getProp<cl_uint, int>(CL_DEVICE_ADDRESS_BITS) : 0; }

bool Device::available() const
{ return p ? p->getBoolProp(CL_DEVICE_AVAILABLE) : false; }

bool Device::compilerAvailable() const
{ return p ? p->getBoolProp(CL_DEVICE_COMPILER_AVAILABLE) : false; }

bool Device::linkerAvailable() const
#ifdef CL_VERSION_1_2
{ return p ? p->getBoolProp(CL_DEVICE_LINKER_AVAILABLE) : false; }
#else
{ CV_REQUIRE_OPENCL_1_2_ERROR; }
#endif

int Device::doubleFPConfig() const
{ return p ? p->doubleFPConfig_ : 0; }

int Device::singleFPConfig() const
{ return p ? p->getProp<cl_device_fp_config, int>(CL_DEVICE_SINGLE_FP_CONFIG) : 0; }

int Device::halfFPConfig() const
#ifdef CL_VERSION_1_2
{ return p ? p->getProp<cl_device_fp_config, int>(CL_DEVICE_HALF_FP_CONFIG) : 0; }
#else
{ CV_REQUIRE_OPENCL_1_2_ERROR; }
#endif

bool Device::endianLittle() const
{ return p ? p->getBoolProp(CL_DEVICE_ENDIAN_LITTLE) : false; }

bool Device::errorCorrectionSupport() const
{ return p ? p->getBoolProp(CL_DEVICE_ERROR_CORRECTION_SUPPORT) : false; }

int Device::executionCapabilities() const
{ return p ? p->getProp<cl_device_exec_capabilities, int>(CL_DEVICE_EXECUTION_CAPABILITIES) : 0; }

size_t Device::globalMemCacheSize() const
{ return p ? p->getProp<cl_ulong, size_t>(CL_DEVICE_GLOBAL_MEM_CACHE_SIZE) : 0; }

int Device::globalMemCacheType() const
{ return p ? p->getProp<cl_device_mem_cache_type, int>(CL_DEVICE_GLOBAL_MEM_CACHE_TYPE) : 0; }

int Device::globalMemCacheLineSize() const
{ return p ? p->getProp<cl_uint, int>(CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE) : 0; }

size_t Device::globalMemSize() const
{ return p ? p->getProp<cl_ulong, size_t>(CL_DEVICE_GLOBAL_MEM_SIZE) : 0; }

size_t Device::localMemSize() const
{ return p ? p->getProp<cl_ulong, size_t>(CL_DEVICE_LOCAL_MEM_SIZE) : 0; }

int Device::localMemType() const
{ return p ? p->getProp<cl_device_local_mem_type, int>(CL_DEVICE_LOCAL_MEM_TYPE) : 0; }

bool Device::hostUnifiedMemory() const
{ return p ? p->hostUnifiedMemory_ : false; }

bool Device::imageSupport() const
{ return p ? p->getBoolProp(CL_DEVICE_IMAGE_SUPPORT) : false; }

bool Device::imageFromBufferSupport() const
{
    return p ? p->isExtensionSupported("cl_khr_image2d_from_buffer") : false;
}

uint Device::imagePitchAlignment() const
{
#ifdef CL_DEVICE_IMAGE_PITCH_ALIGNMENT
    return p ? p->getProp<cl_uint, uint>(CL_DEVICE_IMAGE_PITCH_ALIGNMENT) : 0;
#else
    return 0;
#endif
}

uint Device::imageBaseAddressAlignment() const
{
#ifdef CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT
    return p ? p->getProp<cl_uint, uint>(CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT) : 0;
#else
    return 0;
#endif
}

size_t Device::image2DMaxWidth() const
{ return p ? p->getProp<size_t, size_t>(CL_DEVICE_IMAGE2D_MAX_WIDTH) : 0; }

size_t Device::image2DMaxHeight() const
{ return p ? p->getProp<size_t, size_t>(CL_DEVICE_IMAGE2D_MAX_HEIGHT) : 0; }

size_t Device::image3DMaxWidth() const
{ return p ? p->getProp<size_t, size_t>(CL_DEVICE_IMAGE3D_MAX_WIDTH) : 0; }

size_t Device::image3DMaxHeight() const
{ return p ? p->getProp<size_t, size_t>(CL_DEVICE_IMAGE3D_MAX_HEIGHT) : 0; }

size_t Device::image3DMaxDepth() const
{ return p ? p->getProp<size_t, size_t>(CL_DEVICE_IMAGE3D_MAX_DEPTH) : 0; }

size_t Device::imageMaxBufferSize() const
#ifdef CL_VERSION_1_2
{ return p ? p->getProp<size_t, size_t>(CL_DEVICE_IMAGE_MAX_BUFFER_SIZE) : 0; }
#else
{ CV_REQUIRE_OPENCL_1_2_ERROR; }
#endif

size_t Device::imageMaxArraySize() const
#ifdef CL_VERSION_1_2
{ return p ? p->getProp<size_t, size_t>(CL_DEVICE_IMAGE_MAX_ARRAY_SIZE) : 0; }
#else
{ CV_REQUIRE_OPENCL_1_2_ERROR; }
#endif

bool Device::intelSubgroupsSupport() const
{ return p ? p->intelSubgroupsSupport_ : false; }

int Device::maxClockFrequency() const
{ return p ? p->getProp<cl_uint, int>(CL_DEVICE_MAX_CLOCK_FREQUENCY) : 0; }

int Device::maxComputeUnits() const
{ return p ? p->maxComputeUnits_ : 0; }

int Device::maxConstantArgs() const
{ return p ? p->getProp<cl_uint, int>(CL_DEVICE_MAX_CONSTANT_ARGS) : 0; }

size_t Device::maxConstantBufferSize() const
{ return p ? p->getProp<cl_ulong, size_t>(CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE) : 0; }

size_t Device::maxMemAllocSize() const
{ return p ? p->getProp<cl_ulong, size_t>(CL_DEVICE_MAX_MEM_ALLOC_SIZE) : 0; }

size_t Device::maxParameterSize() const
{ return p ? p->getProp<cl_ulong, size_t>(CL_DEVICE_MAX_PARAMETER_SIZE) : 0; }

int Device::maxReadImageArgs() const
{ return p ? p->getProp<cl_uint, int>(CL_DEVICE_MAX_READ_IMAGE_ARGS) : 0; }

int Device::maxWriteImageArgs() const
{ return p ? p->getProp<cl_uint, int>(CL_DEVICE_MAX_WRITE_IMAGE_ARGS) : 0; }

int Device::maxSamplers() const
{ return p ? p->getProp<cl_uint, int>(CL_DEVICE_MAX_SAMPLERS) : 0; }

size_t Device::maxWorkGroupSize() const
{ return p ? p->maxWorkGroupSize_ : 0; }

int Device::maxWorkItemDims() const
{ return p ? p->getProp<cl_uint, int>(CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS) : 0; }

void Device::maxWorkItemSizes(size_t* sizes) const
{
    if(p)
    {
        const int MAX_DIMS = 32;
        size_t retsz = 0;
        CV_OclDbgAssert(clGetDeviceInfo(p->handle, CL_DEVICE_MAX_WORK_ITEM_SIZES,
                MAX_DIMS*sizeof(sizes[0]), &sizes[0], &retsz) == CL_SUCCESS);
    }
}

int Device::memBaseAddrAlign() const
{ return p ? p->getProp<cl_uint, int>(CL_DEVICE_MEM_BASE_ADDR_ALIGN) : 0; }

int Device::nativeVectorWidthChar() const
{ return p ? p->getProp<cl_uint, int>(CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR) : 0; }

int Device::nativeVectorWidthShort() const
{ return p ? p->getProp<cl_uint, int>(CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT) : 0; }

int Device::nativeVectorWidthInt() const
{ return p ? p->getProp<cl_uint, int>(CL_DEVICE_NATIVE_VECTOR_WIDTH_INT) : 0; }

int Device::nativeVectorWidthLong() const
{ return p ? p->getProp<cl_uint, int>(CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG) : 0; }

int Device::nativeVectorWidthFloat() const
{ return p ? p->getProp<cl_uint, int>(CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT) : 0; }

int Device::nativeVectorWidthDouble() const
{ return p ? p->getProp<cl_uint, int>(CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE) : 0; }

int Device::nativeVectorWidthHalf() const
{ return p ? p->getProp<cl_uint, int>(CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF) : 0; }

int Device::preferredVectorWidthChar() const
{ return p ? p->getProp<cl_uint, int>(CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR) : 0; }

int Device::preferredVectorWidthShort() const
{ return p ? p->getProp<cl_uint, int>(CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT) : 0; }

int Device::preferredVectorWidthInt() const
{ return p ? p->getProp<cl_uint, int>(CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT) : 0; }

int Device::preferredVectorWidthLong() const
{ return p ? p->getProp<cl_uint, int>(CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG) : 0; }

int Device::preferredVectorWidthFloat() const
{ return p ? p->getProp<cl_uint, int>(CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT) : 0; }

int Device::preferredVectorWidthDouble() const
{ return p ? p->getProp<cl_uint, int>(CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE) : 0; }

int Device::preferredVectorWidthHalf() const
{ return p ? p->getProp<cl_uint, int>(CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF) : 0; }

size_t Device::printfBufferSize() const
#ifdef CL_VERSION_1_2
{ return p ? p->getProp<size_t, size_t>(CL_DEVICE_PRINTF_BUFFER_SIZE) : 0; }
#else
{ CV_REQUIRE_OPENCL_1_2_ERROR; }
#endif


size_t Device::profilingTimerResolution() const
{ return p ? p->getProp<size_t, size_t>(CL_DEVICE_PROFILING_TIMER_RESOLUTION) : 0; }

const Device& Device::getDefault()
{
    const Context& ctx = Context::getDefault();
    int idx = getCoreTlsData().get()->device;
    const Device& device = ctx.device(idx);
    return device;
}

////////////////////////////////////// Context ///////////////////////////////////////////////////

template <typename Functor, typename ObjectType>
inline cl_int getStringInfo(Functor f, ObjectType obj, cl_uint name, std::string& param)
{
    ::size_t required;
    cl_int err = f(obj, name, 0, NULL, &required);
    if (err != CL_SUCCESS)
        return err;

    param.clear();
    if (required > 0)
    {
        AutoBuffer<char> buf(required + 1);
        char* ptr = (char*)buf; // cleanup is not needed
        err = f(obj, name, required, ptr, NULL);
        if (err != CL_SUCCESS)
            return err;
        param = ptr;
    }

    return CL_SUCCESS;
}

static void split(const std::string &s, char delim, std::vector<std::string> &elems)
{
    elems.clear();
    if (s.size() == 0)
        return;
    std::istringstream ss(s);
    std::string item;
    while (!ss.eof())
    {
        std::getline(ss, item, delim);
        elems.push_back(item);
    }
}

// Layout: <Platform>:<CPU|GPU|ACCELERATOR|nothing=GPU/CPU>:<deviceName>
// Sample: AMD:GPU:
// Sample: AMD:GPU:Tahiti
// Sample: :GPU|CPU: = '' = ':' = '::'
static bool parseOpenCLDeviceConfiguration(const std::string& configurationStr,
        std::string& platform, std::vector<std::string>& deviceTypes, std::string& deviceNameOrID)
{
    std::vector<std::string> parts;
    split(configurationStr, ':', parts);
    if (parts.size() > 3)
    {
        std::cerr << "ERROR: Invalid configuration string for OpenCL device" << std::endl;
        return false;
    }
    if (parts.size() > 2)
        deviceNameOrID = parts[2];
    if (parts.size() > 1)
    {
        split(parts[1], '|', deviceTypes);
    }
    if (parts.size() > 0)
    {
        platform = parts[0];
    }
    return true;
}

#ifdef WINRT
static cl_device_id selectOpenCLDevice()
{
    return NULL;
}
#else
// std::tolower is int->int
static char char_tolower(char ch)
{
    return (char)std::tolower((int)ch);
}
static cl_device_id selectOpenCLDevice()
{
    std::string platform, deviceName;
    std::vector<std::string> deviceTypes;

    const char* configuration = getenv("OPENCV_OPENCL_DEVICE");
    if (configuration &&
            (strcmp(configuration, "disabled") == 0 ||
             !parseOpenCLDeviceConfiguration(std::string(configuration), platform, deviceTypes, deviceName)
            ))
        return NULL;

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
            if (deviceID < 0)
                return NULL;
        }
    }

    std::vector<cl_platform_id> platforms;
    {
        cl_uint numPlatforms = 0;
        CV_OclDbgAssert(clGetPlatformIDs(0, NULL, &numPlatforms) == CL_SUCCESS);

        if (numPlatforms == 0)
            return NULL;
        platforms.resize((size_t)numPlatforms);
        CV_OclDbgAssert(clGetPlatformIDs(numPlatforms, &platforms[0], &numPlatforms) == CL_SUCCESS);
        platforms.resize(numPlatforms);
    }

    int selectedPlatform = -1;
    if (platform.length() > 0)
    {
        for (size_t i = 0; i < platforms.size(); i++)
        {
            std::string name;
            CV_OclDbgAssert(getStringInfo(clGetPlatformInfo, platforms[i], CL_PLATFORM_NAME, name) == CL_SUCCESS);
            if (name.find(platform) != std::string::npos)
            {
                selectedPlatform = (int)i;
                break;
            }
        }
        if (selectedPlatform == -1)
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
            if (configuration)
                deviceTypes.push_back("CPU");
        }
        else
            deviceTypes.push_back("ALL");
    }
    for (size_t t = 0; t < deviceTypes.size(); t++)
    {
        int deviceType = 0;
        std::string tempStrDeviceType = deviceTypes[t];
        std::transform(tempStrDeviceType.begin(), tempStrDeviceType.end(), tempStrDeviceType.begin(), char_tolower);

        if (tempStrDeviceType == "gpu" || tempStrDeviceType == "dgpu" || tempStrDeviceType == "igpu")
            deviceType = Device::TYPE_GPU;
        else if (tempStrDeviceType == "cpu")
            deviceType = Device::TYPE_CPU;
        else if (tempStrDeviceType == "accelerator")
            deviceType = Device::TYPE_ACCELERATOR;
        else if (tempStrDeviceType == "all")
            deviceType = Device::TYPE_ALL;
        else
        {
            std::cerr << "ERROR: Unsupported device type for OpenCL device (GPU, CPU, ACCELERATOR): " << deviceTypes[t] << std::endl;
            goto not_found;
        }

        std::vector<cl_device_id> devices; // TODO Use clReleaseDevice to cleanup
        for (int i = selectedPlatform >= 0 ? selectedPlatform : 0;
                (selectedPlatform >= 0 ? i == selectedPlatform : true) && (i < (int)platforms.size());
                i++)
        {
            cl_uint count = 0;
            cl_int status = clGetDeviceIDs(platforms[i], deviceType, 0, NULL, &count);
            CV_OclDbgAssert(status == CL_SUCCESS || status == CL_DEVICE_NOT_FOUND);
            if (count == 0)
                continue;
            size_t base = devices.size();
            devices.resize(base + count);
            status = clGetDeviceIDs(platforms[i], deviceType, count, &devices[base], &count);
            CV_OclDbgAssert(status == CL_SUCCESS || status == CL_DEVICE_NOT_FOUND);
        }

        for (size_t i = (isID ? deviceID : 0);
             (isID ? (i == (size_t)deviceID) : true) && (i < devices.size());
             i++)
        {
            std::string name;
            CV_OclDbgAssert(getStringInfo(clGetDeviceInfo, devices[i], CL_DEVICE_NAME, name) == CL_SUCCESS);
            cl_bool useGPU = true;
            if(tempStrDeviceType == "dgpu" || tempStrDeviceType == "igpu")
            {
                cl_bool isIGPU = CL_FALSE;
                clGetDeviceInfo(devices[i], CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(isIGPU), &isIGPU, NULL);
                useGPU = tempStrDeviceType == "dgpu" ? !isIGPU : isIGPU;
            }
            if ( (isID || name.find(deviceName) != std::string::npos) && useGPU)
            {
                // TODO check for OpenCL 1.1
                return devices[i];
            }
        }
    }

not_found:
    if (!configuration)
        return NULL; // suppress messages on stderr

    std::cerr << "ERROR: Requested OpenCL device not found, check configuration: " << configuration << std::endl
            << "    Platform: " << (platform.length() == 0 ? "any" : platform) << std::endl
            << "    Device types: ";
    for (size_t t = 0; t < deviceTypes.size(); t++)
        std::cerr << deviceTypes[t] << " ";

    std::cerr << std::endl << "    Device name: " << (deviceName.length() == 0 ? "any" : deviceName) << std::endl;
    return NULL;
}
#endif

#ifdef HAVE_OPENCL_SVM
namespace svm {

enum AllocatorFlags { // don't use first 16 bits
        OPENCL_SVM_COARSE_GRAIN_BUFFER = 1 << 16, // clSVMAlloc + SVM map/unmap
        OPENCL_SVM_FINE_GRAIN_BUFFER = 2 << 16, // clSVMAlloc
        OPENCL_SVM_FINE_GRAIN_SYSTEM = 3 << 16, // direct access
        OPENCL_SVM_BUFFER_MASK = 3 << 16,
        OPENCL_SVM_BUFFER_MAP = 4 << 16
};

static bool checkForceSVMUmatUsage()
{
    static bool initialized = false;
    static bool force = false;
    if (!initialized)
    {
        force = utils::getConfigurationParameterBool("OPENCV_OPENCL_SVM_FORCE_UMAT_USAGE", false);
        initialized = true;
    }
    return force;
}
static bool checkDisableSVMUMatUsage()
{
    static bool initialized = false;
    static bool force = false;
    if (!initialized)
    {
        force = utils::getConfigurationParameterBool("OPENCV_OPENCL_SVM_DISABLE_UMAT_USAGE", false);
        initialized = true;
    }
    return force;
}
static bool checkDisableSVM()
{
    static bool initialized = false;
    static bool force = false;
    if (!initialized)
    {
        force = utils::getConfigurationParameterBool("OPENCV_OPENCL_SVM_DISABLE", false);
        initialized = true;
    }
    return force;
}
// see SVMCapabilities
static unsigned int getSVMCapabilitiesMask()
{
    static bool initialized = false;
    static unsigned int mask = 0;
    if (!initialized)
    {
        const char* envValue = getenv("OPENCV_OPENCL_SVM_CAPABILITIES_MASK");
        if (envValue == NULL)
        {
            return ~0U; // all bits 1
        }
        mask = atoi(envValue);
        initialized = true;
    }
    return mask;
}
} // namespace
#endif

static size_t getProgramCountLimit()
{
    static bool initialized = false;
    static size_t count = 0;
    if (!initialized)
    {
        count = utils::getConfigurationParameterSizeT("OPENCV_OPENCL_PROGRAM_CACHE", 0);
        initialized = true;
    }
    return count;
}

struct Context::Impl
{
    static Context::Impl* get(Context& context) { return context.p; }

    void __init()
    {
        refcount = 1;
        handle = 0;
#ifdef HAVE_OPENCL_SVM
        svmInitialized = false;
#endif
    }

    Impl()
    {
        __init();
    }

    void setDefault()
    {
        CV_Assert(handle == NULL);

        cl_device_id d = selectOpenCLDevice();

        if (d == NULL)
            return;

        cl_platform_id pl = NULL;
        CV_OclDbgAssert(clGetDeviceInfo(d, CL_DEVICE_PLATFORM, sizeof(cl_platform_id), &pl, NULL) == CL_SUCCESS);

        cl_context_properties prop[] =
        {
            CL_CONTEXT_PLATFORM, (cl_context_properties)pl,
            0
        };

        // !!! in the current implementation force the number of devices to 1 !!!
        cl_uint nd = 1;
        cl_int status;

        handle = clCreateContext(prop, nd, &d, 0, 0, &status);

        bool ok = handle != 0 && status == CL_SUCCESS;
        if( ok )
        {
            devices.resize(nd);
            devices[0].set(d);
        }
        else
            handle = NULL;
    }

    Impl(int dtype0)
    {
        __init();

        cl_int retval = 0;
        cl_platform_id pl = (cl_platform_id)Platform::getDefault().ptr();
        cl_context_properties prop[] =
        {
            CL_CONTEXT_PLATFORM, (cl_context_properties)pl,
            0
        };

        cl_uint i, nd0 = 0, nd = 0;
        int dtype = dtype0 & 15;
        CV_OclDbgAssert(clGetDeviceIDs( pl, dtype, 0, 0, &nd0 ) == CL_SUCCESS);

        AutoBuffer<void*> dlistbuf(nd0*2+1);
        cl_device_id* dlist = (cl_device_id*)(void**)dlistbuf;
        cl_device_id* dlist_new = dlist + nd0;
        CV_OclDbgAssert(clGetDeviceIDs( pl, dtype, nd0, dlist, &nd0 ) == CL_SUCCESS);
        String name0;

        for(i = 0; i < nd0; i++)
        {
            Device d(dlist[i]);
            if( !d.available() || !d.compilerAvailable() )
                continue;
            if( dtype0 == Device::TYPE_DGPU && d.hostUnifiedMemory() )
                continue;
            if( dtype0 == Device::TYPE_IGPU && !d.hostUnifiedMemory() )
                continue;
            String name = d.name();
            if( nd != 0 && name != name0 )
                continue;
            name0 = name;
            dlist_new[nd++] = dlist[i];
        }

        if(nd == 0)
            return;

        // !!! in the current implementation force the number of devices to 1 !!!
        nd = 1;

        handle = clCreateContext(prop, nd, dlist_new, 0, 0, &retval);
        bool ok = handle != 0 && retval == CL_SUCCESS;
        if( ok )
        {
            devices.resize(nd);
            for( i = 0; i < nd; i++ )
                devices[i].set(dlist_new[i]);
        }
    }

    ~Impl()
    {
        if(handle)
        {
            clReleaseContext(handle);
            handle = NULL;
        }
        devices.clear();
    }

    Program getProg(const ProgramSource& src,
                    const String& buildflags, String& errmsg)
    {
        size_t limit = getProgramCountLimit();
        String key = cv::format("codehash=%08llx ", src.hash()) + Program::getPrefix(buildflags);
        {
            cv::AutoLock lock(program_cache_mutex);
            phash_t::iterator it = phash.find(key);
            if (it != phash.end())
            {
                // TODO LRU cache
                CacheList::iterator i = std::find(cacheList.begin(), cacheList.end(), key);
                if (i != cacheList.end() && i != cacheList.begin())
                {
                    cacheList.erase(i);
                    cacheList.push_front(key);
                }
                return it->second;
            }
            { // cleanup program cache
                size_t sz = phash.size();
                if (limit > 0 && sz >= limit)
                {
                    static bool warningFlag = false;
                    if (!warningFlag)
                    {
                        printf("\nWARNING: OpenCV-OpenCL:\n"
                            "    In-memory cache for OpenCL programs is full, older programs will be unloaded.\n"
                            "    You can change cache size via OPENCV_OPENCL_PROGRAM_CACHE environment variable\n\n");
                        warningFlag = true;
                    }
                    while (!cacheList.empty())
                    {
                        size_t c = phash.erase(cacheList.back());
                        cacheList.pop_back();
                        if (c != 0)
                            break;
                    }
                }
            }
        }
        Program prog(src, buildflags, errmsg);
        // Cache result of build failures too (to prevent unnecessary compiler invocations)
        {
            cv::AutoLock lock(program_cache_mutex);
            phash.insert(std::pair<std::string, Program>(key, prog));
            cacheList.push_front(key);
        }
        return prog;
    }

    void unloadProg(Program& prog)
    {
        cv::AutoLock lock(program_cache_mutex);
        for (CacheList::iterator i = cacheList.begin(); i != cacheList.end(); ++i)
        {
              phash_t::iterator it = phash.find(*i);
              if (it != phash.end())
              {
                  if (it->second.ptr() == prog.ptr())
                  {
                      phash.erase(*i);
                      cacheList.erase(i);
                      return;
                  }
              }
        }
    }

    IMPLEMENT_REFCOUNTABLE();

    cl_context handle;
    std::vector<Device> devices;

    cv::Mutex program_cache_mutex;
    typedef std::map<std::string, Program> phash_t;
    phash_t phash;
    typedef std::list<cv::String> CacheList;
    CacheList cacheList;

#ifdef HAVE_OPENCL_SVM
    bool svmInitialized;
    bool svmAvailable;
    bool svmEnabled;
    svm::SVMCapabilities svmCapabilities;
    svm::SVMFunctions svmFunctions;

    void svmInit()
    {
        CV_Assert(handle != NULL);
        const Device& device = devices[0];
        cl_device_svm_capabilities deviceCaps = 0;
        CV_Assert(((void)0, CL_DEVICE_SVM_CAPABILITIES == CL_DEVICE_SVM_CAPABILITIES_AMD)); // Check assumption
        cl_int status = clGetDeviceInfo((cl_device_id)device.ptr(), CL_DEVICE_SVM_CAPABILITIES, sizeof(deviceCaps), &deviceCaps, NULL);
        if (status != CL_SUCCESS)
        {
            CV_OPENCL_SVM_TRACE_ERROR_P("CL_DEVICE_SVM_CAPABILITIES via clGetDeviceInfo failed: %d\n", status);
            goto noSVM;
        }
        CV_OPENCL_SVM_TRACE_P("CL_DEVICE_SVM_CAPABILITIES returned: 0x%x\n", (int)deviceCaps);
        CV_Assert(((void)0, CL_DEVICE_SVM_COARSE_GRAIN_BUFFER == CL_DEVICE_SVM_COARSE_GRAIN_BUFFER_AMD)); // Check assumption
        svmCapabilities.value_ =
                ((deviceCaps & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER) ? svm::SVMCapabilities::SVM_COARSE_GRAIN_BUFFER : 0) |
                ((deviceCaps & CL_DEVICE_SVM_FINE_GRAIN_BUFFER) ? svm::SVMCapabilities::SVM_FINE_GRAIN_BUFFER : 0) |
                ((deviceCaps & CL_DEVICE_SVM_FINE_GRAIN_SYSTEM) ? svm::SVMCapabilities::SVM_FINE_GRAIN_SYSTEM : 0) |
                ((deviceCaps & CL_DEVICE_SVM_ATOMICS) ? svm::SVMCapabilities::SVM_ATOMICS : 0);
        svmCapabilities.value_ &= svm::getSVMCapabilitiesMask();
        if (svmCapabilities.value_ == 0)
        {
            CV_OPENCL_SVM_TRACE_ERROR_P("svmCapabilities is empty\n");
            goto noSVM;
        }
        try
        {
            // Try OpenCL 2.0
            CV_OPENCL_SVM_TRACE_P("Try SVM from OpenCL 2.0 ...\n");
            void* ptr = clSVMAlloc(handle, CL_MEM_READ_WRITE, 100, 0);
            if (!ptr)
            {
                CV_OPENCL_SVM_TRACE_ERROR_P("clSVMAlloc returned NULL...\n");
                CV_ErrorNoReturn(Error::StsBadArg, "clSVMAlloc returned NULL");
            }
            try
            {
                bool error = false;
                cl_command_queue q = (cl_command_queue)Queue::getDefault().ptr();
                if (CL_SUCCESS != clEnqueueSVMMap(q, CL_TRUE, CL_MAP_WRITE, ptr, 100, 0, NULL, NULL))
                {
                    CV_OPENCL_SVM_TRACE_ERROR_P("clEnqueueSVMMap failed...\n");
                    CV_ErrorNoReturn(Error::StsBadArg, "clEnqueueSVMMap FAILED");
                }
                clFinish(q);
                try
                {
                    ((int*)ptr)[0] = 100;
                }
                catch (...)
                {
                    CV_OPENCL_SVM_TRACE_ERROR_P("SVM buffer access test FAILED\n");
                    error = true;
                }
                if (CL_SUCCESS != clEnqueueSVMUnmap(q, ptr, 0, NULL, NULL))
                {
                    CV_OPENCL_SVM_TRACE_ERROR_P("clEnqueueSVMUnmap failed...\n");
                    CV_ErrorNoReturn(Error::StsBadArg, "clEnqueueSVMUnmap FAILED");
                }
                clFinish(q);
                if (error)
                {
                    CV_ErrorNoReturn(Error::StsBadArg, "OpenCL SVM buffer access test was FAILED");
                }
            }
            catch (...)
            {
                CV_OPENCL_SVM_TRACE_ERROR_P("OpenCL SVM buffer access test was FAILED\n");
                clSVMFree(handle, ptr);
                throw;
            }
            clSVMFree(handle, ptr);
            svmFunctions.fn_clSVMAlloc = clSVMAlloc;
            svmFunctions.fn_clSVMFree = clSVMFree;
            svmFunctions.fn_clSetKernelArgSVMPointer = clSetKernelArgSVMPointer;
            //svmFunctions.fn_clSetKernelExecInfo = clSetKernelExecInfo;
            //svmFunctions.fn_clEnqueueSVMFree = clEnqueueSVMFree;
            svmFunctions.fn_clEnqueueSVMMemcpy = clEnqueueSVMMemcpy;
            svmFunctions.fn_clEnqueueSVMMemFill = clEnqueueSVMMemFill;
            svmFunctions.fn_clEnqueueSVMMap = clEnqueueSVMMap;
            svmFunctions.fn_clEnqueueSVMUnmap = clEnqueueSVMUnmap;
        }
        catch (...)
        {
            CV_OPENCL_SVM_TRACE_P("clSVMAlloc failed, trying HSA extension...\n");
            try
            {
                // Try HSA extension
                String extensions = device.extensions();
                if (extensions.find("cl_amd_svm") == String::npos)
                {
                    CV_OPENCL_SVM_TRACE_P("Device extension doesn't have cl_amd_svm: %s\n", extensions.c_str());
                    goto noSVM;
                }
                cl_platform_id p = NULL;
                status = clGetDeviceInfo((cl_device_id)device.ptr(), CL_DEVICE_PLATFORM, sizeof(cl_platform_id), &p, NULL);
                CV_Assert(status == CL_SUCCESS);
                svmFunctions.fn_clSVMAlloc = (clSVMAllocAMD_fn)clGetExtensionFunctionAddressForPlatform(p, "clSVMAllocAMD");
                svmFunctions.fn_clSVMFree = (clSVMFreeAMD_fn)clGetExtensionFunctionAddressForPlatform(p, "clSVMFreeAMD");
                svmFunctions.fn_clSetKernelArgSVMPointer = (clSetKernelArgSVMPointerAMD_fn)clGetExtensionFunctionAddressForPlatform(p, "clSetKernelArgSVMPointerAMD");
                //svmFunctions.fn_clSetKernelExecInfo = (clSetKernelExecInfoAMD_fn)clGetExtensionFunctionAddressForPlatform(p, "clSetKernelExecInfoAMD");
                //svmFunctions.fn_clEnqueueSVMFree = (clEnqueueSVMFreeAMD_fn)clGetExtensionFunctionAddressForPlatform(p, "clEnqueueSVMFreeAMD");
                svmFunctions.fn_clEnqueueSVMMemcpy = (clEnqueueSVMMemcpyAMD_fn)clGetExtensionFunctionAddressForPlatform(p, "clEnqueueSVMMemcpyAMD");
                svmFunctions.fn_clEnqueueSVMMemFill = (clEnqueueSVMMemFillAMD_fn)clGetExtensionFunctionAddressForPlatform(p, "clEnqueueSVMMemFillAMD");
                svmFunctions.fn_clEnqueueSVMMap = (clEnqueueSVMMapAMD_fn)clGetExtensionFunctionAddressForPlatform(p, "clEnqueueSVMMapAMD");
                svmFunctions.fn_clEnqueueSVMUnmap = (clEnqueueSVMUnmapAMD_fn)clGetExtensionFunctionAddressForPlatform(p, "clEnqueueSVMUnmapAMD");
                CV_Assert(svmFunctions.isValid());
            }
            catch (...)
            {
                CV_OPENCL_SVM_TRACE_P("Something is totally wrong\n");
                goto noSVM;
            }
        }

        svmAvailable = true;
        svmEnabled = !svm::checkDisableSVM();
        svmInitialized = true;
        CV_OPENCL_SVM_TRACE_P("OpenCV OpenCL SVM support initialized\n");
        return;
    noSVM:
        CV_OPENCL_SVM_TRACE_P("OpenCL SVM is not detected\n");
        svmAvailable = false;
        svmEnabled = false;
        svmCapabilities.value_ = 0;
        svmInitialized = true;
        svmFunctions.fn_clSVMAlloc = NULL;
        return;
    }
#endif
};


Context::Context()
{
    p = 0;
}

Context::Context(int dtype)
{
    p = 0;
    create(dtype);
}

bool Context::create()
{
    if( !haveOpenCL() )
        return false;
    if(p)
        p->release();
    p = new Impl();
    if(!p->handle)
    {
        delete p;
        p = 0;
    }
    return p != 0;
}

bool Context::create(int dtype0)
{
    if( !haveOpenCL() )
        return false;
    if(p)
        p->release();
    p = new Impl(dtype0);
    if(!p->handle)
    {
        delete p;
        p = 0;
    }
    return p != 0;
}

Context::~Context()
{
    if (p)
    {
        p->release();
        p = NULL;
    }
}

Context::Context(const Context& c)
{
    p = (Impl*)c.p;
    if(p)
        p->addref();
}

Context& Context::operator = (const Context& c)
{
    Impl* newp = (Impl*)c.p;
    if(newp)
        newp->addref();
    if(p)
        p->release();
    p = newp;
    return *this;
}

void* Context::ptr() const
{
    return p == NULL ? NULL : p->handle;
}

size_t Context::ndevices() const
{
    return p ? p->devices.size() : 0;
}

const Device& Context::device(size_t idx) const
{
    static Device dummy;
    return !p || idx >= p->devices.size() ? dummy : p->devices[idx];
}

Context& Context::getDefault(bool initialize)
{
    static Context* ctx = new Context();
    if(!ctx->p && haveOpenCL())
    {
        if (!ctx->p)
            ctx->p = new Impl();
        if (initialize)
        {
            // do not create new Context right away.
            // First, try to retrieve existing context of the same type.
            // In its turn, Platform::getContext() may call Context::create()
            // if there is no such context.
            if (ctx->p->handle == NULL)
                ctx->p->setDefault();
        }
    }

    return *ctx;
}

Program Context::getProg(const ProgramSource& prog,
                         const String& buildopts, String& errmsg)
{
    return p ? p->getProg(prog, buildopts, errmsg) : Program();
}

void Context::unloadProg(Program& prog)
{
    if (p)
        p->unloadProg(prog);
}

#ifdef HAVE_OPENCL_SVM
bool Context::useSVM() const
{
    Context::Impl* i = p;
    CV_Assert(i);
    if (!i->svmInitialized)
        i->svmInit();
    return i->svmEnabled;
}
void Context::setUseSVM(bool enabled)
{
    Context::Impl* i = p;
    CV_Assert(i);
    if (!i->svmInitialized)
        i->svmInit();
    if (enabled && !i->svmAvailable)
    {
        CV_ErrorNoReturn(Error::StsError, "OpenCL Shared Virtual Memory (SVM) is not supported by OpenCL device");
    }
    i->svmEnabled = enabled;
}
#else
bool Context::useSVM() const { return false; }
void Context::setUseSVM(bool enabled) { CV_Assert(!enabled); }
#endif

#ifdef HAVE_OPENCL_SVM
namespace svm {

const SVMCapabilities getSVMCapabilitites(const ocl::Context& context)
{
    Context::Impl* i = context.p;
    CV_Assert(i);
    if (!i->svmInitialized)
        i->svmInit();
    return i->svmCapabilities;
}

CV_EXPORTS const SVMFunctions* getSVMFunctions(const ocl::Context& context)
{
    Context::Impl* i = context.p;
    CV_Assert(i);
    CV_Assert(i->svmInitialized); // getSVMCapabilitites() must be called first
    CV_Assert(i->svmFunctions.fn_clSVMAlloc != NULL);
    return &i->svmFunctions;
}

CV_EXPORTS bool useSVM(UMatUsageFlags usageFlags)
{
    if (checkForceSVMUmatUsage())
        return true;
    if (checkDisableSVMUMatUsage())
        return false;
    if ((usageFlags & USAGE_ALLOCATE_SHARED_MEMORY) != 0)
        return true;
    return false; // don't use SVM by default
}

} // namespace cv::ocl::svm
#endif // HAVE_OPENCL_SVM


static void get_platform_name(cl_platform_id id, String& name)
{
    // get platform name string length
    size_t sz = 0;
    if (CL_SUCCESS != clGetPlatformInfo(id, CL_PLATFORM_NAME, 0, 0, &sz))
        CV_ErrorNoReturn(cv::Error::OpenCLApiCallError, "clGetPlatformInfo failed!");

    // get platform name string
    AutoBuffer<char> buf(sz + 1);
    if (CL_SUCCESS != clGetPlatformInfo(id, CL_PLATFORM_NAME, sz, buf, 0))
        CV_ErrorNoReturn(cv::Error::OpenCLApiCallError, "clGetPlatformInfo failed!");

    // just in case, ensure trailing zero for ASCIIZ string
    buf[sz] = 0;

    name = (const char*)buf;
}

/*
// Attaches OpenCL context to OpenCV
*/
void attachContext(const String& platformName, void* platformID, void* context, void* deviceID)
{
    cl_uint cnt = 0;

    if(CL_SUCCESS != clGetPlatformIDs(0, 0, &cnt))
        CV_ErrorNoReturn(cv::Error::OpenCLApiCallError, "clGetPlatformIDs failed!");

    if (cnt == 0)
        CV_ErrorNoReturn(cv::Error::OpenCLApiCallError, "no OpenCL platform available!");

    std::vector<cl_platform_id> platforms(cnt);

    if(CL_SUCCESS != clGetPlatformIDs(cnt, &platforms[0], 0))
        CV_ErrorNoReturn(cv::Error::OpenCLApiCallError, "clGetPlatformIDs failed!");

    bool platformAvailable = false;

    // check if external platformName contained in list of available platforms in OpenCV
    for (unsigned int i = 0; i < cnt; i++)
    {
        String availablePlatformName;
        get_platform_name(platforms[i], availablePlatformName);
        // external platform is found in the list of available platforms
        if (platformName == availablePlatformName)
        {
            platformAvailable = true;
            break;
        }
    }

    if (!platformAvailable)
        CV_ErrorNoReturn(cv::Error::OpenCLApiCallError, "No matched platforms available!");

    // check if platformID corresponds to platformName
    String actualPlatformName;
    get_platform_name((cl_platform_id)platformID, actualPlatformName);
    if (platformName != actualPlatformName)
        CV_ErrorNoReturn(cv::Error::OpenCLApiCallError, "No matched platforms available!");

    // do not initialize OpenCL context
    Context ctx = Context::getDefault(false);

    // attach supplied context to OpenCV
    initializeContextFromHandle(ctx, platformID, context, deviceID);

    if(CL_SUCCESS != clRetainContext((cl_context)context))
        CV_ErrorNoReturn(cv::Error::OpenCLApiCallError, "clRetainContext failed!");

    // clear command queue, if any
    getCoreTlsData().get()->oclQueue.finish();
    Queue q;
    getCoreTlsData().get()->oclQueue = q;

    return;
} // attachContext()


void initializeContextFromHandle(Context& ctx, void* platform, void* _context, void* _device)
{
    cl_context context = (cl_context)_context;
    cl_device_id device = (cl_device_id)_device;

    // cleanup old context
    Context::Impl * impl = ctx.p;
    if (impl->handle)
    {
        CV_OclDbgAssert(clReleaseContext(impl->handle) == CL_SUCCESS);
    }
    impl->devices.clear();

    impl->handle = context;
    impl->devices.resize(1);
    impl->devices[0].set(device);

    Platform& p = Platform::getDefault();
    Platform::Impl* pImpl = p.p;
    pImpl->handle = (cl_platform_id)platform;
}

/////////////////////////////////////////// Queue /////////////////////////////////////////////

struct Queue::Impl
{
    inline void __init()
    {
        refcount = 1;
        handle = 0;
        isProfilingQueue_ = false;
    }

    Impl(cl_command_queue q)
    {
        __init();
        handle = q;

        cl_command_queue_properties props = 0;
        cl_int result = clGetCommandQueueInfo(handle, CL_QUEUE_PROPERTIES, sizeof(cl_command_queue_properties), &props, NULL);
        CV_Assert(result && "clGetCommandQueueInfo(CL_QUEUE_PROPERTIES)");
        isProfilingQueue_ = !!(props & CL_QUEUE_PROFILING_ENABLE);
    }

    Impl(cl_command_queue q, bool isProfilingQueue)
    {
        __init();
        handle = q;
        isProfilingQueue_ = isProfilingQueue;
    }

    Impl(const Context& c, const Device& d, bool withProfiling = false)
    {
        __init();

        const Context* pc = &c;
        cl_context ch = (cl_context)pc->ptr();
        if( !ch )
        {
            pc = &Context::getDefault();
            ch = (cl_context)pc->ptr();
        }
        cl_device_id dh = (cl_device_id)d.ptr();
        if( !dh )
            dh = (cl_device_id)pc->device(0).ptr();
        cl_int retval = 0;
        cl_command_queue_properties props = withProfiling ? CL_QUEUE_PROFILING_ENABLE : 0;
        handle = clCreateCommandQueue(ch, dh, props, &retval);
        CV_OclDbgAssert(retval == CL_SUCCESS);
        isProfilingQueue_ = withProfiling;
    }

    ~Impl()
    {
#ifdef _WIN32
        if (!cv::__termination)
#endif
        {
            if(handle)
            {
                clFinish(handle);
                clReleaseCommandQueue(handle);
                handle = NULL;
            }
        }
    }

    const cv::ocl::Queue& getProfilingQueue(const cv::ocl::Queue& self)
    {
        if (isProfilingQueue_)
            return self;

        if (profiling_queue_.ptr())
            return profiling_queue_;

        cl_context ctx = 0;
        CV_Assert(CL_SUCCESS == clGetCommandQueueInfo(handle, CL_QUEUE_CONTEXT, sizeof(cl_context), &ctx, NULL));

        cl_device_id device = 0;
        CV_Assert(CL_SUCCESS == clGetCommandQueueInfo(handle, CL_QUEUE_DEVICE, sizeof(cl_device_id), &device, NULL));

        cl_int result = CL_SUCCESS;
        cl_command_queue_properties props = CL_QUEUE_PROFILING_ENABLE;
        cl_command_queue q = clCreateCommandQueue(ctx, device, props, &result);
        CV_Assert(result == CL_SUCCESS && "clCreateCommandQueue(with CL_QUEUE_PROFILING_ENABLE)");

        Queue queue;
        queue.p = new Impl(q, true);
        profiling_queue_ = queue;

        return profiling_queue_;
    }

    IMPLEMENT_REFCOUNTABLE();

    cl_command_queue handle;
    bool isProfilingQueue_;
    cv::ocl::Queue profiling_queue_;
};

Queue::Queue()
{
    p = 0;
}

Queue::Queue(const Context& c, const Device& d)
{
    p = 0;
    create(c, d);
}

Queue::Queue(const Queue& q)
{
    p = q.p;
    if(p)
        p->addref();
}

Queue& Queue::operator = (const Queue& q)
{
    Impl* newp = (Impl*)q.p;
    if(newp)
        newp->addref();
    if(p)
        p->release();
    p = newp;
    return *this;
}

Queue::~Queue()
{
    if(p)
        p->release();
}

bool Queue::create(const Context& c, const Device& d)
{
    if(p)
        p->release();
    p = new Impl(c, d);
    return p->handle != 0;
}

void Queue::finish()
{
    if(p && p->handle)
    {
        CV_OclDbgAssert(clFinish(p->handle) == CL_SUCCESS);
    }
}

const Queue& Queue::getProfilingQueue() const
{
    CV_Assert(p);
    return p->getProfilingQueue(*this);
}

void* Queue::ptr() const
{
    return p ? p->handle : 0;
}

Queue& Queue::getDefault()
{
    Queue& q = getCoreTlsData().get()->oclQueue;
    if( !q.p && haveOpenCL() )
        q.create(Context::getDefault());
    return q;
}

static cl_command_queue getQueue(const Queue& q)
{
    cl_command_queue qq = (cl_command_queue)q.ptr();
    if(!qq)
        qq = (cl_command_queue)Queue::getDefault().ptr();
    return qq;
}

/////////////////////////////////////////// KernelArg /////////////////////////////////////////////

KernelArg::KernelArg()
    : flags(0), m(0), obj(0), sz(0), wscale(1), iwscale(1)
{
}

KernelArg::KernelArg(int _flags, UMat* _m, int _wscale, int _iwscale, const void* _obj, size_t _sz)
    : flags(_flags), m(_m), obj(_obj), sz(_sz), wscale(_wscale), iwscale(_iwscale)
{
    CV_Assert(_flags == LOCAL || _flags == CONSTANT || _m != NULL);
}

KernelArg KernelArg::Constant(const Mat& m)
{
    CV_Assert(m.isContinuous());
    return KernelArg(CONSTANT, 0, 0, 0, m.ptr(), m.total()*m.elemSize());
}

/////////////////////////////////////////// Kernel /////////////////////////////////////////////

struct Kernel::Impl
{
    Impl(const char* kname, const Program& prog) :
        refcount(1), isInProgress(false), nu(0)
    {
        cl_program ph = (cl_program)prog.ptr();
        cl_int retval = 0;
#ifdef ENABLE_INSTRUMENTATION
        name = kname;
#endif
        handle = ph != 0 ?
            clCreateKernel(ph, kname, &retval) : 0;
        CV_OclDbgAssert(retval == CL_SUCCESS);
        for( int i = 0; i < MAX_ARRS; i++ )
            u[i] = 0;
        haveTempDstUMats = false;
    }

    void cleanupUMats()
    {
        for( int i = 0; i < MAX_ARRS; i++ )
            if( u[i] )
            {
                if( CV_XADD(&u[i]->urefcount, -1) == 1 )
                {
                    u[i]->flags |= UMatData::ASYNC_CLEANUP;
                    u[i]->currAllocator->deallocate(u[i]);
                }
                u[i] = 0;
            }
        nu = 0;
        haveTempDstUMats = false;
    }

    void addUMat(const UMat& m, bool dst)
    {
        CV_Assert(nu < MAX_ARRS && m.u && m.u->urefcount > 0);
        u[nu] = m.u;
        CV_XADD(&m.u->urefcount, 1);
        nu++;
        if(dst && m.u->tempUMat())
            haveTempDstUMats = true;
    }

    void addImage(const Image2D& image)
    {
        images.push_back(image);
    }

    void finit(cl_event e)
    {
        CV_UNUSED(e);
#if 0
        printf("event::callback(%p)\n", e); fflush(stdout);
#endif
        cleanupUMats();
        images.clear();
        isInProgress = false;
        release();
    }

    bool run(int dims, size_t _globalsize[], size_t _localsize[],
            bool sync, int64* timeNS, const Queue& q);

    ~Impl()
    {
        if(handle)
            clReleaseKernel(handle);
    }

    IMPLEMENT_REFCOUNTABLE();

#ifdef ENABLE_INSTRUMENTATION
    cv::String name;
#endif
    cl_kernel handle;
    enum { MAX_ARRS = 16 };
    UMatData* u[MAX_ARRS];
    bool isInProgress;
    int nu;
    std::list<Image2D> images;
    bool haveTempDstUMats;
};

}} // namespace cv::ocl

extern "C" {

static void CL_CALLBACK oclCleanupCallback(cl_event e, cl_int, void *p)
{
    ((cv::ocl::Kernel::Impl*)p)->finit(e);
}

}

namespace cv { namespace ocl {

Kernel::Kernel()
{
    p = 0;
}

Kernel::Kernel(const char* kname, const Program& prog)
{
    p = 0;
    create(kname, prog);
}

Kernel::Kernel(const char* kname, const ProgramSource& src,
               const String& buildopts, String* errmsg)
{
    p = 0;
    create(kname, src, buildopts, errmsg);
}

Kernel::Kernel(const Kernel& k)
{
    p = k.p;
    if(p)
        p->addref();
}

Kernel& Kernel::operator = (const Kernel& k)
{
    Impl* newp = (Impl*)k.p;
    if(newp)
        newp->addref();
    if(p)
        p->release();
    p = newp;
    return *this;
}

Kernel::~Kernel()
{
    if(p)
        p->release();
}

bool Kernel::create(const char* kname, const Program& prog)
{
    if(p)
        p->release();
    p = new Impl(kname, prog);
    if(p->handle == 0)
    {
        p->release();
        p = 0;
    }
#ifdef CV_OPENCL_RUN_ASSERT // check kernel compilation fails
    CV_Assert(p);
#endif
    return p != 0;
}

bool Kernel::create(const char* kname, const ProgramSource& src,
                    const String& buildopts, String* errmsg)
{
    if(p)
    {
        p->release();
        p = 0;
    }
    String tempmsg;
    if( !errmsg ) errmsg = &tempmsg;
    const Program& prog = Context::getDefault().getProg(src, buildopts, *errmsg);
    return create(kname, prog);
}

void* Kernel::ptr() const
{
    return p ? p->handle : 0;
}

bool Kernel::empty() const
{
    return ptr() == 0;
}

int Kernel::set(int i, const void* value, size_t sz)
{
    if (!p || !p->handle)
        return -1;
    if (i < 0)
        return i;
    if( i == 0 )
        p->cleanupUMats();

    cl_int retval = clSetKernelArg(p->handle, (cl_uint)i, sz, value);
    CV_OclDbgAssert(retval == CL_SUCCESS);
    if (retval != CL_SUCCESS)
        return -1;
    return i+1;
}

int Kernel::set(int i, const Image2D& image2D)
{
    p->addImage(image2D);
    cl_mem h = (cl_mem)image2D.ptr();
    return set(i, &h, sizeof(h));
}

int Kernel::set(int i, const UMat& m)
{
    return set(i, KernelArg(KernelArg::READ_WRITE, (UMat*)&m));
}

int Kernel::set(int i, const KernelArg& arg)
{
    if( !p || !p->handle )
        return -1;
    if (i < 0)
        return i;
    if( i == 0 )
        p->cleanupUMats();
    if( arg.m )
    {
        int accessFlags = ((arg.flags & KernelArg::READ_ONLY) ? ACCESS_READ : 0) +
                          ((arg.flags & KernelArg::WRITE_ONLY) ? ACCESS_WRITE : 0);
        bool ptronly = (arg.flags & KernelArg::PTR_ONLY) != 0;
        cl_mem h = (cl_mem)arg.m->handle(accessFlags);

        if (!h)
        {
            p->release();
            p = 0;
            return -1;
        }

#ifdef HAVE_OPENCL_SVM
        if ((arg.m->u->allocatorFlags_ & svm::OPENCL_SVM_BUFFER_MASK) != 0)
        {
            const Context& ctx = Context::getDefault();
            const svm::SVMFunctions* svmFns = svm::getSVMFunctions(ctx);
            uchar*& svmDataPtr = (uchar*&)arg.m->u->handle;
            CV_OPENCL_SVM_TRACE_P("clSetKernelArgSVMPointer: %p\n", svmDataPtr);
#if 1 // TODO
            cl_int status = svmFns->fn_clSetKernelArgSVMPointer(p->handle, (cl_uint)i, svmDataPtr);
#else
            cl_int status = svmFns->fn_clSetKernelArgSVMPointer(p->handle, (cl_uint)i, &svmDataPtr);
#endif
            CV_Assert(status == CL_SUCCESS);
        }
        else
#endif
        {
            CV_OclDbgAssert(clSetKernelArg(p->handle, (cl_uint)i, sizeof(h), &h) == CL_SUCCESS);
        }

        if (ptronly)
        {
            i++;
        }
        else if( arg.m->dims <= 2 )
        {
            UMat2D u2d(*arg.m);
            CV_OclDbgAssert(clSetKernelArg(p->handle, (cl_uint)(i+1), sizeof(u2d.step), &u2d.step) == CL_SUCCESS);
            CV_OclDbgAssert(clSetKernelArg(p->handle, (cl_uint)(i+2), sizeof(u2d.offset), &u2d.offset) == CL_SUCCESS);
            i += 3;

            if( !(arg.flags & KernelArg::NO_SIZE) )
            {
                int cols = u2d.cols*arg.wscale/arg.iwscale;
                CV_OclDbgAssert(clSetKernelArg(p->handle, (cl_uint)i, sizeof(u2d.rows), &u2d.rows) == CL_SUCCESS);
                CV_OclDbgAssert(clSetKernelArg(p->handle, (cl_uint)(i+1), sizeof(cols), &cols) == CL_SUCCESS);
                i += 2;
            }
        }
        else
        {
            UMat3D u3d(*arg.m);
            CV_OclDbgAssert(clSetKernelArg(p->handle, (cl_uint)(i+1), sizeof(u3d.slicestep), &u3d.slicestep) == CL_SUCCESS);
            CV_OclDbgAssert(clSetKernelArg(p->handle, (cl_uint)(i+2), sizeof(u3d.step), &u3d.step) == CL_SUCCESS);
            CV_OclDbgAssert(clSetKernelArg(p->handle, (cl_uint)(i+3), sizeof(u3d.offset), &u3d.offset) == CL_SUCCESS);
            i += 4;
            if( !(arg.flags & KernelArg::NO_SIZE) )
            {
                int cols = u3d.cols*arg.wscale/arg.iwscale;
                CV_OclDbgAssert(clSetKernelArg(p->handle, (cl_uint)i, sizeof(u3d.slices), &u3d.slices) == CL_SUCCESS);
                CV_OclDbgAssert(clSetKernelArg(p->handle, (cl_uint)(i+1), sizeof(u3d.rows), &u3d.rows) == CL_SUCCESS);
                CV_OclDbgAssert(clSetKernelArg(p->handle, (cl_uint)(i+2), sizeof(u3d.cols), &cols) == CL_SUCCESS);
                i += 3;
            }
        }
        p->addUMat(*arg.m, (accessFlags & ACCESS_WRITE) != 0);
        return i;
    }
    CV_OclDbgAssert(clSetKernelArg(p->handle, (cl_uint)i, arg.sz, arg.obj) == CL_SUCCESS);
    return i+1;
}

bool Kernel::run(int dims, size_t _globalsize[], size_t _localsize[],
                 bool sync, const Queue& q)
{
    if (!p)
        return false;

    size_t globalsize[CV_MAX_DIM] = {1,1,1};
    size_t total = 1;
    CV_Assert(_globalsize != NULL);
    for (int i = 0; i < dims; i++)
    {
        size_t val = _localsize ? _localsize[i] :
            dims == 1 ? 64 : dims == 2 ? (i == 0 ? 256 : 8) : dims == 3 ? (8>>(int)(i>0)) : 1;
        CV_Assert( val > 0 );
        total *= _globalsize[i];
        if (_globalsize[i] == 1)
            val = 1;
        globalsize[i] = divUp(_globalsize[i], (unsigned int)val) * val;
    }
    CV_Assert(total > 0);

    return p->run(dims, globalsize, _localsize, sync, NULL, q);
}


bool Kernel::Impl::run(int dims, size_t globalsize[], size_t localsize[],
        bool sync, int64* timeNS, const Queue& q)
{
    CV_INSTRUMENT_REGION_OPENCL_RUN(p->name.c_str());

    if (!handle || isInProgress)
        return false;

    cl_command_queue qq = getQueue(q);
    if (haveTempDstUMats)
        sync = true;
    if (timeNS)
        sync = true;
    cl_event asyncEvent = 0;
    cl_int retval = clEnqueueNDRangeKernel(qq, handle, (cl_uint)dims,
                                           NULL, globalsize, localsize, 0, 0,
                                           (sync && !timeNS) ? 0 : &asyncEvent);
#if CV_OPENCL_SHOW_RUN_ERRORS
    if (retval != CL_SUCCESS)
    {
        printf("OpenCL program returns error: %d\n", retval);
        fflush(stdout);
    }
#endif
    if (sync || retval != CL_SUCCESS)
    {
        CV_OclDbgAssert(clFinish(qq) == CL_SUCCESS);
        if (timeNS)
        {
            if (retval == CL_SUCCESS)
            {
                clWaitForEvents(1, &asyncEvent);
                cl_ulong startTime, stopTime;
                CV_Assert(CL_SUCCESS == clGetEventProfilingInfo(asyncEvent, CL_PROFILING_COMMAND_START, sizeof(startTime), &startTime, NULL));
                CV_Assert(CL_SUCCESS == clGetEventProfilingInfo(asyncEvent, CL_PROFILING_COMMAND_END, sizeof(stopTime), &stopTime, NULL));
                *timeNS = (int64)(stopTime - startTime);
            }
            else
            {
                *timeNS = -1;
            }
        }
        cleanupUMats();
    }
    else
    {
        addref();
        isInProgress = true;
        CV_OclDbgAssert(clSetEventCallback(asyncEvent, CL_COMPLETE, oclCleanupCallback, this) == CL_SUCCESS);
    }
    if (asyncEvent)
        clReleaseEvent(asyncEvent);
    return retval == CL_SUCCESS;
}

bool Kernel::runTask(bool sync, const Queue& q)
{
    if(!p || !p->handle || p->isInProgress)
        return false;

    cl_command_queue qq = getQueue(q);
    cl_event asyncEvent = 0;
    cl_int retval = clEnqueueTask(qq, p->handle, 0, 0, sync ? 0 : &asyncEvent);
    if( sync || retval != CL_SUCCESS )
    {
        CV_OclDbgAssert(clFinish(qq) == CL_SUCCESS);
        p->cleanupUMats();
    }
    else
    {
        p->addref();
        p->isInProgress = true;
        CV_OclDbgAssert(clSetEventCallback(asyncEvent, CL_COMPLETE, oclCleanupCallback, p) == CL_SUCCESS);
    }
    if (asyncEvent)
        clReleaseEvent(asyncEvent);
    return retval == CL_SUCCESS;
}

int64 Kernel::runProfiling(int dims, size_t globalsize[], size_t localsize[], const Queue& q_)
{
    CV_Assert(p && p->handle && !p->isInProgress);
    Queue q = q_.ptr() ? q_ : Queue::getDefault();
    CV_Assert(q.ptr());
    q.finish(); // call clFinish() on base queue
    Queue profilingQueue = q.getProfilingQueue();
    int64 timeNs = -1;
    bool res = p->run(dims, globalsize, localsize, true, &timeNs, profilingQueue);
    return res ? timeNs : -1;
}

size_t Kernel::workGroupSize() const
{
    if(!p || !p->handle)
        return 0;
    size_t val = 0, retsz = 0;
    cl_device_id dev = (cl_device_id)Device::getDefault().ptr();
    return clGetKernelWorkGroupInfo(p->handle, dev, CL_KERNEL_WORK_GROUP_SIZE,
                                    sizeof(val), &val, &retsz) == CL_SUCCESS ? val : 0;
}

size_t Kernel::preferedWorkGroupSizeMultiple() const
{
    if(!p || !p->handle)
        return 0;
    size_t val = 0, retsz = 0;
    cl_device_id dev = (cl_device_id)Device::getDefault().ptr();
    return clGetKernelWorkGroupInfo(p->handle, dev, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
                                    sizeof(val), &val, &retsz) == CL_SUCCESS ? val : 0;
}

bool Kernel::compileWorkGroupSize(size_t wsz[]) const
{
    if(!p || !p->handle || !wsz)
        return 0;
    size_t retsz = 0;
    cl_device_id dev = (cl_device_id)Device::getDefault().ptr();
    return clGetKernelWorkGroupInfo(p->handle, dev, CL_KERNEL_COMPILE_WORK_GROUP_SIZE,
                                    sizeof(wsz[0])*3, wsz, &retsz) == CL_SUCCESS;
}

size_t Kernel::localMemSize() const
{
    if(!p || !p->handle)
        return 0;
    size_t retsz = 0;
    cl_ulong val = 0;
    cl_device_id dev = (cl_device_id)Device::getDefault().ptr();
    return clGetKernelWorkGroupInfo(p->handle, dev, CL_KERNEL_LOCAL_MEM_SIZE,
                                    sizeof(val), &val, &retsz) == CL_SUCCESS ? (size_t)val : 0;
}



///////////////////////////////////////// ProgramSource ///////////////////////////////////////////////

struct ProgramSource::Impl
{
    Impl(const String& src)
    {
        init(cv::String(), cv::String(), src, cv::String());
    }
    Impl(const String& module, const String& name, const String& codeStr, const String& codeHash)
    {
        init(module, name, codeStr, codeHash);
    }
    void init(const String& module, const String& name, const String& codeStr, const String& codeHash)
    {
        refcount = 1;
        module_ = module;
        name_ = name;
        codeStr_ = codeStr;
        codeHash_ = codeHash;

        isHashUpdated = false;
        if (codeHash_.empty())
        {
            updateHash();
            codeHash_ = cv::format("%08llx", hash_);
        }
    }

    void updateHash()
    {
        hash_ = crc64((uchar*)codeStr_.c_str(), codeStr_.size());
        isHashUpdated = true;
    }

    IMPLEMENT_REFCOUNTABLE();

    String module_;
    String name_;
    String codeStr_;
    String codeHash_;
    // TODO std::vector<ProgramSource> includes_;

    bool isHashUpdated;
    ProgramSource::hash_t hash_;
};


ProgramSource::ProgramSource()
{
    p = 0;
}

ProgramSource::ProgramSource(const String& module, const String& name, const String& codeStr, const String& codeHash)
{
    p = new Impl(module, name, codeStr, codeHash);
}

ProgramSource::ProgramSource(const char* prog)
{
    p = new Impl(prog);
}

ProgramSource::ProgramSource(const String& prog)
{
    p = new Impl(prog);
}

ProgramSource::~ProgramSource()
{
    if(p)
        p->release();
}

ProgramSource::ProgramSource(const ProgramSource& prog)
{
    p = prog.p;
    if(p)
        p->addref();
}

ProgramSource& ProgramSource::operator = (const ProgramSource& prog)
{
    Impl* newp = (Impl*)prog.p;
    if(newp)
        newp->addref();
    if(p)
        p->release();
    p = newp;
    return *this;
}

const String& ProgramSource::source() const
{
    CV_Assert(p);
    return p->codeStr_;
}

ProgramSource::hash_t ProgramSource::hash() const
{
    CV_Assert(p);
    if (!p->isHashUpdated)
        p->updateHash();
    return p->hash_;
}


internal::ProgramEntry::operator ProgramSource&() const
{
    if (this->pProgramSource == NULL)
    {
        cv::AutoLock lock(cv::getInitializationMutex());
        if (this->pProgramSource == NULL)
        {
            ProgramSource* ps = new ProgramSource(this->module, this->name, this->programCode, this->programHash);
            const_cast<ProgramEntry*>(this)->pProgramSource = ps;
        }
    }
    return *this->pProgramSource;
}



/////////////////////////////////////////// Program /////////////////////////////////////////////

struct Program::Impl
{
    Impl(const ProgramSource& _src,
         const String& _buildflags, String& errmsg) :
         src(_src),
         buildflags(_buildflags),
         handle(NULL)
    {
        refcount = 1;
        compile(Context::getDefault(), errmsg);
    }

    bool compile(const Context& ctx, String& errmsg)
    {
        CV_Assert(handle == NULL);
        CV_INSTRUMENT_REGION_OPENCL_COMPILE(cv::format("Compile: %" PRIx64 " options: %s", src.hash(), buildflags.c_str()).c_str());
        const String& srcstr = src.source();
        const char* srcptr = srcstr.c_str();
        size_t srclen = srcstr.size();
        cl_int retval = 0;

        handle = clCreateProgramWithSource((cl_context)ctx.ptr(), 1, &srcptr, &srclen, &retval);
        CV_OclDbgAssert(handle && retval == CL_SUCCESS);
        if (handle && retval == CL_SUCCESS)
        {
            int i, n = (int)ctx.ndevices();
            AutoBuffer<void*> deviceListBuf(n+1);
            void** deviceList = deviceListBuf;
            for( i = 0; i < n; i++ )
                deviceList[i] = ctx.device(i).ptr();

            Device device = Device::getDefault();
            if (device.isAMD())
                buildflags += " -D AMD_DEVICE";
            else if (device.isIntel())
                buildflags += " -D INTEL_DEVICE";

            retval = clBuildProgram(handle, n,
                                    (const cl_device_id*)deviceList,
                                    buildflags.c_str(), 0, 0);
#if !CV_OPENCL_ALWAYS_SHOW_BUILD_LOG
            if (retval != CL_SUCCESS)
#endif
            {
                AutoBuffer<char, 4096> buffer; buffer[0] = 0;

                size_t retsz = 0;
                cl_int log_retval = clGetProgramBuildInfo(handle, (cl_device_id)deviceList[0],
                                                          CL_PROGRAM_BUILD_LOG, 0, 0, &retsz);
                if (log_retval == CL_SUCCESS && retsz > 1)
                {
                    buffer.resize(retsz + 16);
                    log_retval = clGetProgramBuildInfo(handle, (cl_device_id)deviceList[0],
                                                       CL_PROGRAM_BUILD_LOG, retsz+1, (char*)buffer, &retsz);
                    if (log_retval == CL_SUCCESS)
                    {
                        if (retsz < buffer.size())
                            buffer[retsz] = 0;
                        else
                            buffer[buffer.size() - 1] = 0;
                    }
                    else
                    {
                        buffer[0] = 0;
                    }
                }

                errmsg = String(buffer);
                printf("OpenCL program build log: %s (%s)\nStatus %d: %s\n%s\n%s\n",
                        src.getImpl()->name_.c_str(), src.getImpl()->module_.c_str(),
                        retval, getOpenCLErrorString(retval),
                        buildflags.c_str(), errmsg.c_str());
                fflush(stdout);

                // don't remove "retval != CL_SUCCESS" condition here:
                // it would break CV_OPENCL_ALWAYS_SHOW_BUILD_LOG mode
                if (retval != CL_SUCCESS && handle)
                {
                    clReleaseProgram(handle);
                    handle = NULL;
                }
            }
        }
        return handle != NULL;
    }

    Impl(const String& _buf, const String& _buildflags)
    {
        refcount = 1;
        handle = 0;
        buildflags = _buildflags;
        if(_buf.empty())
            return;
        String prefix0 = Program::getPrefix(buildflags);
        const Context& ctx = Context::getDefault();
        const Device& dev = Device::getDefault();
        const char* pos0 = _buf.c_str();
        const char* pos1 = strchr(pos0, '\n');
        if(!pos1)
            return;
        const char* pos2 = strchr(pos1+1, '\n');
        if(!pos2)
            return;
        const char* pos3 = strchr(pos2+1, '\n');
        if(!pos3)
            return;
        size_t prefixlen = (pos3 - pos0)+1;
        String prefix(pos0, prefixlen);
        if( prefix != prefix0 )
            return;
        const uchar* bin = (uchar*)(pos3+1);
        void* devid = dev.ptr();
        size_t codelen = _buf.length() - prefixlen;
        cl_int binstatus = 0, retval = 0;
        handle = clCreateProgramWithBinary((cl_context)ctx.ptr(), 1, (cl_device_id*)&devid,
                                           &codelen, &bin, &binstatus, &retval);
        CV_OclDbgAssert(retval == CL_SUCCESS);
    }

    String store()
    {
        if(!handle)
            return String();
        size_t progsz = 0, retsz = 0;
        String prefix = Program::getPrefix(buildflags);
        size_t prefixlen = prefix.length();
        if(clGetProgramInfo(handle, CL_PROGRAM_BINARY_SIZES, sizeof(progsz), &progsz, &retsz) != CL_SUCCESS)
            return String();
        AutoBuffer<uchar> bufbuf(prefixlen + progsz + 16);
        uchar* buf = bufbuf;
        memcpy(buf, prefix.c_str(), prefixlen);
        buf += prefixlen;
        if(clGetProgramInfo(handle, CL_PROGRAM_BINARIES, sizeof(buf), &buf, &retsz) != CL_SUCCESS)
            return String();
        buf[progsz] = (uchar)'\0';
        return String((const char*)(uchar*)bufbuf, prefixlen + progsz);
    }

    ~Impl()
    {
        if( handle )
        {
#ifdef _WIN32
            if (!cv::__termination)
#endif
            {
                clReleaseProgram(handle);
            }
            handle = NULL;
        }
    }

    IMPLEMENT_REFCOUNTABLE();

    ProgramSource src;
    String buildflags;
    cl_program handle;
};


Program::Program() { p = 0; }

Program::Program(const ProgramSource& src,
        const String& buildflags, String& errmsg)
{
    p = 0;
    create(src, buildflags, errmsg);
}

Program::Program(const Program& prog)
{
    p = prog.p;
    if(p)
        p->addref();
}

Program& Program::operator = (const Program& prog)
{
    Impl* newp = (Impl*)prog.p;
    if(newp)
        newp->addref();
    if(p)
        p->release();
    p = newp;
    return *this;
}

Program::~Program()
{
    if(p)
        p->release();
}

bool Program::create(const ProgramSource& src,
            const String& buildflags, String& errmsg)
{
    if(p)
        p->release();
    p = new Impl(src, buildflags, errmsg);
    if(!p->handle)
    {
        p->release();
        p = 0;
    }
    return p != 0;
}

const ProgramSource& Program::source() const
{
    static ProgramSource dummy;
    return p ? p->src : dummy;
}

void* Program::ptr() const
{
    return p ? p->handle : 0;
}

bool Program::read(const String& bin, const String& buildflags)
{
    if(p)
        p->release();
    p = new Impl(bin, buildflags);
    return p->handle != 0;
}

bool Program::write(String& bin) const
{
    if(!p)
        return false;
    bin = p->store();
    return !bin.empty();
}

String Program::getPrefix() const
{
    if(!p)
        return String();
    return getPrefix(p->buildflags);
}

String Program::getPrefix(const String& buildflags)
{
    const Context& ctx = Context::getDefault();
    const Device& dev = ctx.device(0);
    return format("name=%s\ndriver=%s\nbuildflags=%s\n",
                  dev.name().c_str(), dev.driverVersion().c_str(), buildflags.c_str());
}



//////////////////////////////////////////// OpenCLAllocator //////////////////////////////////////////////////

template<typename T>
class OpenCLBufferPool
{
protected:
    ~OpenCLBufferPool() { }
public:
    virtual T allocate(size_t size) = 0;
    virtual void release(T buffer) = 0;
};

template <typename Derived, typename BufferEntry, typename T>
class OpenCLBufferPoolBaseImpl : public BufferPoolController, public OpenCLBufferPool<T>
{
private:
    inline Derived& derived() { return *static_cast<Derived*>(this); }
protected:
    Mutex mutex_;

    size_t currentReservedSize;
    size_t maxReservedSize;

    std::list<BufferEntry> allocatedEntries_; // Allocated and used entries
    std::list<BufferEntry> reservedEntries_; // LRU order. Allocated, but not used entries

    // synchronized
    bool _findAndRemoveEntryFromAllocatedList(CV_OUT BufferEntry& entry, T buffer)
    {
        typename std::list<BufferEntry>::iterator i = allocatedEntries_.begin();
        for (; i != allocatedEntries_.end(); ++i)
        {
            BufferEntry& e = *i;
            if (e.clBuffer_ == buffer)
            {
                entry = e;
                allocatedEntries_.erase(i);
                return true;
            }
        }
        return false;
    }

    // synchronized
    bool _findAndRemoveEntryFromReservedList(CV_OUT BufferEntry& entry, const size_t size)
    {
        if (reservedEntries_.empty())
            return false;
        typename std::list<BufferEntry>::iterator i = reservedEntries_.begin();
        typename std::list<BufferEntry>::iterator result_pos = reservedEntries_.end();
        BufferEntry result;
        size_t minDiff = (size_t)(-1);
        for (; i != reservedEntries_.end(); ++i)
        {
            BufferEntry& e = *i;
            if (e.capacity_ >= size)
            {
                size_t diff = e.capacity_ - size;
                if (diff < std::max((size_t)4096, size / 8) && (result_pos == reservedEntries_.end() || diff < minDiff))
                {
                    minDiff = diff;
                    result_pos = i;
                    result = e;
                    if (diff == 0)
                        break;
                }
            }
        }
        if (result_pos != reservedEntries_.end())
        {
            //CV_DbgAssert(result == *result_pos);
            reservedEntries_.erase(result_pos);
            entry = result;
            currentReservedSize -= entry.capacity_;
            allocatedEntries_.push_back(entry);
            return true;
        }
        return false;
    }

    // synchronized
    void _checkSizeOfReservedEntries()
    {
        while (currentReservedSize > maxReservedSize)
        {
            CV_DbgAssert(!reservedEntries_.empty());
            const BufferEntry& entry = reservedEntries_.back();
            CV_DbgAssert(currentReservedSize >= entry.capacity_);
            currentReservedSize -= entry.capacity_;
            derived()._releaseBufferEntry(entry);
            reservedEntries_.pop_back();
        }
    }

    inline size_t _allocationGranularity(size_t size)
    {
        // heuristic values
        if (size < 1024*1024)
            return 4096;  // don't work with buffers smaller than 4Kb (hidden allocation overhead issue)
        else if (size < 16*1024*1024)
            return 64*1024;
        else
            return 1024*1024;
    }

public:
    OpenCLBufferPoolBaseImpl()
        : currentReservedSize(0),
          maxReservedSize(0)
    {
        // nothing
    }
    virtual ~OpenCLBufferPoolBaseImpl()
    {
        freeAllReservedBuffers();
        CV_Assert(reservedEntries_.empty());
    }
public:
    virtual T allocate(size_t size)
    {
        AutoLock locker(mutex_);
        BufferEntry entry;
        if (maxReservedSize > 0 && _findAndRemoveEntryFromReservedList(entry, size))
        {
            CV_DbgAssert(size <= entry.capacity_);
            LOG_BUFFER_POOL("Reuse reserved buffer: %p\n", entry.clBuffer_);
        }
        else
        {
            derived()._allocateBufferEntry(entry, size);
        }
        return entry.clBuffer_;
    }
    virtual void release(T buffer)
    {
        AutoLock locker(mutex_);
        BufferEntry entry;
        CV_Assert(_findAndRemoveEntryFromAllocatedList(entry, buffer));
        if (maxReservedSize == 0 || entry.capacity_ > maxReservedSize / 8)
        {
            derived()._releaseBufferEntry(entry);
        }
        else
        {
            reservedEntries_.push_front(entry);
            currentReservedSize += entry.capacity_;
            _checkSizeOfReservedEntries();
        }
    }

    virtual size_t getReservedSize() const { return currentReservedSize; }
    virtual size_t getMaxReservedSize() const { return maxReservedSize; }
    virtual void setMaxReservedSize(size_t size)
    {
        AutoLock locker(mutex_);
        size_t oldMaxReservedSize = maxReservedSize;
        maxReservedSize = size;
        if (maxReservedSize < oldMaxReservedSize)
        {
            typename std::list<BufferEntry>::iterator i = reservedEntries_.begin();
            for (; i != reservedEntries_.end();)
            {
                const BufferEntry& entry = *i;
                if (entry.capacity_ > maxReservedSize / 8)
                {
                    CV_DbgAssert(currentReservedSize >= entry.capacity_);
                    currentReservedSize -= entry.capacity_;
                    derived()._releaseBufferEntry(entry);
                    i = reservedEntries_.erase(i);
                    continue;
                }
                ++i;
            }
            _checkSizeOfReservedEntries();
        }
    }
    virtual void freeAllReservedBuffers()
    {
        AutoLock locker(mutex_);
        typename std::list<BufferEntry>::const_iterator i = reservedEntries_.begin();
        for (; i != reservedEntries_.end(); ++i)
        {
            const BufferEntry& entry = *i;
            derived()._releaseBufferEntry(entry);
        }
        reservedEntries_.clear();
        currentReservedSize = 0;
    }
};

struct CLBufferEntry
{
    cl_mem clBuffer_;
    size_t capacity_;
    CLBufferEntry() : clBuffer_((cl_mem)NULL), capacity_(0) { }
};

class OpenCLBufferPoolImpl : public OpenCLBufferPoolBaseImpl<OpenCLBufferPoolImpl, CLBufferEntry, cl_mem>
{
public:
    typedef struct CLBufferEntry BufferEntry;
protected:
    int createFlags_;
public:
    OpenCLBufferPoolImpl(int createFlags = 0)
        : createFlags_(createFlags)
    {
    }

    void _allocateBufferEntry(BufferEntry& entry, size_t size)
    {
        CV_DbgAssert(entry.clBuffer_ == NULL);
        entry.capacity_ = alignSize(size, (int)_allocationGranularity(size));
        Context& ctx = Context::getDefault();
        cl_int retval = CL_SUCCESS;
        entry.clBuffer_ = clCreateBuffer((cl_context)ctx.ptr(), CL_MEM_READ_WRITE|createFlags_, entry.capacity_, 0, &retval);
        CV_Assert(retval == CL_SUCCESS);
        CV_Assert(entry.clBuffer_ != NULL);
        if(retval == CL_SUCCESS)
        {
            CV_IMPL_ADD(CV_IMPL_OCL);
        }
        LOG_BUFFER_POOL("OpenCL allocate %lld (0x%llx) bytes: %p\n",
                (long long)entry.capacity_, (long long)entry.capacity_, entry.clBuffer_);
        allocatedEntries_.push_back(entry);
    }

    void _releaseBufferEntry(const BufferEntry& entry)
    {
        CV_Assert(entry.capacity_ != 0);
        CV_Assert(entry.clBuffer_ != NULL);
        LOG_BUFFER_POOL("OpenCL release buffer: %p, %lld (0x%llx) bytes\n",
                entry.clBuffer_, (long long)entry.capacity_, (long long)entry.capacity_);
        clReleaseMemObject(entry.clBuffer_);
    }
};

#ifdef HAVE_OPENCL_SVM
struct CLSVMBufferEntry
{
    void* clBuffer_;
    size_t capacity_;
    CLSVMBufferEntry() : clBuffer_(NULL), capacity_(0) { }
};
class OpenCLSVMBufferPoolImpl : public OpenCLBufferPoolBaseImpl<OpenCLSVMBufferPoolImpl, CLSVMBufferEntry, void*>
{
public:
    typedef struct CLSVMBufferEntry BufferEntry;
public:
    OpenCLSVMBufferPoolImpl()
    {
    }

    void _allocateBufferEntry(BufferEntry& entry, size_t size)
    {
        CV_DbgAssert(entry.clBuffer_ == NULL);
        entry.capacity_ = alignSize(size, (int)_allocationGranularity(size));

        Context& ctx = Context::getDefault();
        const svm::SVMCapabilities svmCaps = svm::getSVMCapabilitites(ctx);
        bool isFineGrainBuffer = svmCaps.isSupportFineGrainBuffer();
        cl_svm_mem_flags memFlags = CL_MEM_READ_WRITE |
                (isFineGrainBuffer ? CL_MEM_SVM_FINE_GRAIN_BUFFER : 0);

        const svm::SVMFunctions* svmFns = svm::getSVMFunctions(ctx);
        CV_DbgAssert(svmFns->isValid());

        CV_OPENCL_SVM_TRACE_P("clSVMAlloc: %d\n", (int)entry.capacity_);
        void *buf = svmFns->fn_clSVMAlloc((cl_context)ctx.ptr(), memFlags, entry.capacity_, 0);
        CV_Assert(buf);

        entry.clBuffer_ = buf;
        {
            CV_IMPL_ADD(CV_IMPL_OCL);
        }
        LOG_BUFFER_POOL("OpenCL SVM allocate %lld (0x%llx) bytes: %p\n",
                (long long)entry.capacity_, (long long)entry.capacity_, entry.clBuffer_);
        allocatedEntries_.push_back(entry);
    }

    void _releaseBufferEntry(const BufferEntry& entry)
    {
        CV_Assert(entry.capacity_ != 0);
        CV_Assert(entry.clBuffer_ != NULL);
        LOG_BUFFER_POOL("OpenCL release SVM buffer: %p, %lld (0x%llx) bytes\n",
                entry.clBuffer_, (long long)entry.capacity_, (long long)entry.capacity_);
        Context& ctx = Context::getDefault();
        const svm::SVMFunctions* svmFns = svm::getSVMFunctions(ctx);
        CV_DbgAssert(svmFns->isValid());
        CV_OPENCL_SVM_TRACE_P("clSVMFree: %p\n",  entry.clBuffer_);
        svmFns->fn_clSVMFree((cl_context)ctx.ptr(), entry.clBuffer_);
    }
};
#endif



#if defined _MSC_VER
#pragma warning(disable:4127) // conditional expression is constant
#endif
template <bool readAccess, bool writeAccess>
class AlignedDataPtr
{
protected:
    const size_t size_;
    uchar* const originPtr_;
    const size_t alignment_;
    uchar* ptr_;
    uchar* allocatedPtr_;

public:
    AlignedDataPtr(uchar* ptr, size_t size, size_t alignment)
        : size_(size), originPtr_(ptr), alignment_(alignment), ptr_(ptr), allocatedPtr_(NULL)
    {
        CV_DbgAssert((alignment & (alignment - 1)) == 0); // check for 2^n
        if (((size_t)ptr_ & (alignment - 1)) != 0)
        {
            allocatedPtr_ = new uchar[size_ + alignment - 1];
            ptr_ = (uchar*)(((uintptr_t)allocatedPtr_ + (alignment - 1)) & ~(alignment - 1));
            if (readAccess)
            {
                memcpy(ptr_, originPtr_, size_);
            }
        }
    }

    uchar* getAlignedPtr() const
    {
        CV_DbgAssert(((size_t)ptr_ & (alignment_ - 1)) == 0);
        return ptr_;
    }

    ~AlignedDataPtr()
    {
        if (allocatedPtr_)
        {
            if (writeAccess)
            {
                memcpy(originPtr_, ptr_, size_);
            }
            delete[] allocatedPtr_;
            allocatedPtr_ = NULL;
        }
        ptr_ = NULL;
    }
private:
    AlignedDataPtr(const AlignedDataPtr&); // disabled
    AlignedDataPtr& operator=(const AlignedDataPtr&); // disabled
};

template <bool readAccess, bool writeAccess>
class AlignedDataPtr2D
{
protected:
    const size_t size_;
    uchar* const originPtr_;
    const size_t alignment_;
    uchar* ptr_;
    uchar* allocatedPtr_;
    size_t rows_;
    size_t cols_;
    size_t step_;

public:
    AlignedDataPtr2D(uchar* ptr, size_t rows, size_t cols, size_t step, size_t alignment)
        : size_(rows*step), originPtr_(ptr), alignment_(alignment), ptr_(ptr), allocatedPtr_(NULL), rows_(rows), cols_(cols), step_(step)
    {
        CV_DbgAssert((alignment & (alignment - 1)) == 0); // check for 2^n
        if (((size_t)ptr_ & (alignment - 1)) != 0)
        {
            allocatedPtr_ = new uchar[size_ + alignment - 1];
            ptr_ = (uchar*)(((uintptr_t)allocatedPtr_ + (alignment - 1)) & ~(alignment - 1));
            if (readAccess)
            {
                for (size_t i = 0; i < rows_; i++)
                    memcpy(ptr_ + i*step_, originPtr_ + i*step_, cols_);
            }
        }
    }

    uchar* getAlignedPtr() const
    {
        CV_DbgAssert(((size_t)ptr_ & (alignment_ - 1)) == 0);
        return ptr_;
    }

    ~AlignedDataPtr2D()
    {
        if (allocatedPtr_)
        {
            if (writeAccess)
            {
                for (size_t i = 0; i < rows_; i++)
                    memcpy(originPtr_ + i*step_, ptr_ + i*step_, cols_);
            }
            delete[] allocatedPtr_;
            allocatedPtr_ = NULL;
        }
        ptr_ = NULL;
    }
private:
    AlignedDataPtr2D(const AlignedDataPtr2D&); // disabled
    AlignedDataPtr2D& operator=(const AlignedDataPtr2D&); // disabled
};
#if defined _MSC_VER
#pragma warning(default:4127) // conditional expression is constant
#endif

#ifndef CV_OPENCL_DATA_PTR_ALIGNMENT
#define CV_OPENCL_DATA_PTR_ALIGNMENT 16
#endif

class OpenCLAllocator : public MatAllocator
{
    mutable OpenCLBufferPoolImpl bufferPool;
    mutable OpenCLBufferPoolImpl bufferPoolHostPtr;
#ifdef  HAVE_OPENCL_SVM
    mutable OpenCLSVMBufferPoolImpl bufferPoolSVM;
#endif

    enum AllocatorFlags
    {
        ALLOCATOR_FLAGS_BUFFER_POOL_USED = 1 << 0,
        ALLOCATOR_FLAGS_BUFFER_POOL_HOST_PTR_USED = 1 << 1
#ifdef HAVE_OPENCL_SVM
        ,ALLOCATOR_FLAGS_BUFFER_POOL_SVM_USED = 1 << 2
#endif
    };
public:
    OpenCLAllocator()
        : bufferPool(0),
          bufferPoolHostPtr(CL_MEM_ALLOC_HOST_PTR)
    {
        size_t defaultPoolSize, poolSize;
        defaultPoolSize = ocl::Device::getDefault().isIntel() ? 1 << 27 : 0;
        poolSize = utils::getConfigurationParameterSizeT("OPENCV_OPENCL_BUFFERPOOL_LIMIT", defaultPoolSize);
        bufferPool.setMaxReservedSize(poolSize);
        poolSize = utils::getConfigurationParameterSizeT("OPENCV_OPENCL_HOST_PTR_BUFFERPOOL_LIMIT", defaultPoolSize);
        bufferPoolHostPtr.setMaxReservedSize(poolSize);
#ifdef HAVE_OPENCL_SVM
        poolSize = utils::getConfigurationParameterSizeT("OPENCV_OPENCL_SVM_BUFFERPOOL_LIMIT", defaultPoolSize);
        bufferPoolSVM.setMaxReservedSize(poolSize);
#endif

        matStdAllocator = Mat::getDefaultAllocator();
    }
    ~OpenCLAllocator()
    {
        flushCleanupQueue();
    }

    UMatData* defaultAllocate(int dims, const int* sizes, int type, void* data, size_t* step,
            int flags, UMatUsageFlags usageFlags) const
    {
        UMatData* u = matStdAllocator->allocate(dims, sizes, type, data, step, flags, usageFlags);
        return u;
    }

    void getBestFlags(const Context& ctx, int /*flags*/, UMatUsageFlags usageFlags, int& createFlags, int& flags0) const
    {
        const Device& dev = ctx.device(0);
        createFlags = 0;
        if ((usageFlags & USAGE_ALLOCATE_HOST_MEMORY) != 0)
            createFlags |= CL_MEM_ALLOC_HOST_PTR;

        if( dev.hostUnifiedMemory() )
            flags0 = 0;
        else
            flags0 = UMatData::COPY_ON_MAP;
    }

    UMatData* allocate(int dims, const int* sizes, int type,
                       void* data, size_t* step, int flags, UMatUsageFlags usageFlags) const
    {
        if(!useOpenCL())
            return defaultAllocate(dims, sizes, type, data, step, flags, usageFlags);
        CV_Assert(data == 0);
        size_t total = CV_ELEM_SIZE(type);
        for( int i = dims-1; i >= 0; i-- )
        {
            if( step )
                step[i] = total;
            total *= sizes[i];
        }

        Context& ctx = Context::getDefault();
        flushCleanupQueue();

        int createFlags = 0, flags0 = 0;
        getBestFlags(ctx, flags, usageFlags, createFlags, flags0);

        void* handle = NULL;
        int allocatorFlags = 0;

#ifdef HAVE_OPENCL_SVM
        const svm::SVMCapabilities svmCaps = svm::getSVMCapabilitites(ctx);
        if (ctx.useSVM() && svm::useSVM(usageFlags) && !svmCaps.isNoSVMSupport())
        {
            allocatorFlags = ALLOCATOR_FLAGS_BUFFER_POOL_SVM_USED;
            handle = bufferPoolSVM.allocate(total);

            // this property is constant, so single buffer pool can be used here
            bool isFineGrainBuffer = svmCaps.isSupportFineGrainBuffer();
            allocatorFlags |= isFineGrainBuffer ? svm::OPENCL_SVM_FINE_GRAIN_BUFFER : svm::OPENCL_SVM_COARSE_GRAIN_BUFFER;
        }
        else
#endif
        if (createFlags == 0)
        {
            allocatorFlags = ALLOCATOR_FLAGS_BUFFER_POOL_USED;
            handle = bufferPool.allocate(total);
        }
        else if (createFlags == CL_MEM_ALLOC_HOST_PTR)
        {
            allocatorFlags = ALLOCATOR_FLAGS_BUFFER_POOL_HOST_PTR_USED;
            handle = bufferPoolHostPtr.allocate(total);
        }
        else
        {
            CV_Assert(handle != NULL); // Unsupported, throw
        }

        if (!handle)
            return defaultAllocate(dims, sizes, type, data, step, flags, usageFlags);

        UMatData* u = new UMatData(this);
        u->data = 0;
        u->size = total;
        u->handle = handle;
        u->flags = flags0;
        u->allocatorFlags_ = allocatorFlags;
        CV_DbgAssert(!u->tempUMat()); // for bufferPool.release() consistency in deallocate()
        u->markHostCopyObsolete(true);
        return u;
    }

    bool allocate(UMatData* u, int accessFlags, UMatUsageFlags usageFlags) const
    {
        if(!u)
            return false;

        flushCleanupQueue();

        UMatDataAutoLock lock(u);

        if(u->handle == 0)
        {
            CV_Assert(u->origdata != 0);
            Context& ctx = Context::getDefault();
            int createFlags = 0, flags0 = 0;
            getBestFlags(ctx, accessFlags, usageFlags, createFlags, flags0);

            cl_context ctx_handle = (cl_context)ctx.ptr();
            int allocatorFlags = 0;
            int tempUMatFlags = 0;
            void* handle = NULL;
            cl_int retval = CL_SUCCESS;

#ifdef HAVE_OPENCL_SVM
            svm::SVMCapabilities svmCaps = svm::getSVMCapabilitites(ctx);
            bool useSVM = ctx.useSVM() && svm::useSVM(usageFlags);
            if (useSVM && svmCaps.isSupportFineGrainSystem())
            {
                allocatorFlags = svm::OPENCL_SVM_FINE_GRAIN_SYSTEM;
                tempUMatFlags = UMatData::TEMP_UMAT;
                handle = u->origdata;
                CV_OPENCL_SVM_TRACE_P("Use fine grain system: %d (%p)\n", (int)u->size, handle);
            }
            else if (useSVM && (svmCaps.isSupportFineGrainBuffer() || svmCaps.isSupportCoarseGrainBuffer()))
            {
                if (!(accessFlags & ACCESS_FAST)) // memcpy used
                {
                    bool isFineGrainBuffer = svmCaps.isSupportFineGrainBuffer();

                    cl_svm_mem_flags memFlags = createFlags |
                            (isFineGrainBuffer ? CL_MEM_SVM_FINE_GRAIN_BUFFER : 0);

                    const svm::SVMFunctions* svmFns = svm::getSVMFunctions(ctx);
                    CV_DbgAssert(svmFns->isValid());

                    CV_OPENCL_SVM_TRACE_P("clSVMAlloc + copy: %d\n", (int)u->size);
                    handle = svmFns->fn_clSVMAlloc((cl_context)ctx.ptr(), memFlags, u->size, 0);
                    CV_Assert(handle);

                    cl_command_queue q = NULL;
                    if (!isFineGrainBuffer)
                    {
                        q = (cl_command_queue)Queue::getDefault().ptr();
                        CV_OPENCL_SVM_TRACE_P("clEnqueueSVMMap: %p (%d)\n", handle, (int)u->size);
                        cl_int status = svmFns->fn_clEnqueueSVMMap(q, CL_TRUE, CL_MAP_WRITE,
                                handle, u->size,
                                0, NULL, NULL);
                        CV_Assert(status == CL_SUCCESS);

                    }
                    memcpy(handle, u->origdata, u->size);
                    if (!isFineGrainBuffer)
                    {
                        CV_OPENCL_SVM_TRACE_P("clEnqueueSVMUnmap: %p\n", handle);
                        cl_int status = svmFns->fn_clEnqueueSVMUnmap(q, handle, 0, NULL, NULL);
                        CV_Assert(status == CL_SUCCESS);
                    }

                    tempUMatFlags = UMatData::TEMP_UMAT | UMatData::TEMP_COPIED_UMAT;
                    allocatorFlags |= isFineGrainBuffer ? svm::OPENCL_SVM_FINE_GRAIN_BUFFER
                                                : svm::OPENCL_SVM_COARSE_GRAIN_BUFFER;
                }
            }
            else
#endif
            {
                tempUMatFlags = UMatData::TEMP_UMAT;
                if (u->origdata == cv::alignPtr(u->origdata, 4)) // There are OpenCL runtime issues for less aligned data
                {
                    handle = clCreateBuffer(ctx_handle, CL_MEM_USE_HOST_PTR|createFlags,
                                            u->size, u->origdata, &retval);
                }
                if((!handle || retval < 0) && !(accessFlags & ACCESS_FAST))
                {
                    handle = clCreateBuffer(ctx_handle, CL_MEM_COPY_HOST_PTR|CL_MEM_READ_WRITE|createFlags,
                                               u->size, u->origdata, &retval);
                    tempUMatFlags |= UMatData::TEMP_COPIED_UMAT;
                }
            }
            if(!handle || retval != CL_SUCCESS)
                return false;
            u->handle = handle;
            u->prevAllocator = u->currAllocator;
            u->currAllocator = this;
            u->flags |= tempUMatFlags;
            u->allocatorFlags_ = allocatorFlags;
        }
        if(accessFlags & ACCESS_WRITE)
            u->markHostCopyObsolete(true);
        return true;
    }

    /*void sync(UMatData* u) const
    {
        cl_command_queue q = (cl_command_queue)Queue::getDefault().ptr();
        UMatDataAutoLock lock(u);

        if( u->hostCopyObsolete() && u->handle && u->refcount > 0 && u->origdata)
        {
            if( u->tempCopiedUMat() )
            {
                clEnqueueReadBuffer(q, (cl_mem)u->handle, CL_TRUE, 0,
                                    u->size, u->origdata, 0, 0, 0);
            }
            else
            {
                cl_int retval = 0;
                void* data = clEnqueueMapBuffer(q, (cl_mem)u->handle, CL_TRUE,
                                                (CL_MAP_READ | CL_MAP_WRITE),
                                                0, u->size, 0, 0, 0, &retval);
                clEnqueueUnmapMemObject(q, (cl_mem)u->handle, data, 0, 0, 0);
                clFinish(q);
            }
            u->markHostCopyObsolete(false);
        }
        else if( u->copyOnMap() && u->deviceCopyObsolete() && u->data )
        {
            clEnqueueWriteBuffer(q, (cl_mem)u->handle, CL_TRUE, 0,
                                 u->size, u->data, 0, 0, 0);
        }
    }*/

    void deallocate(UMatData* u) const
    {
        if(!u)
            return;

        CV_Assert(u->urefcount == 0);
        CV_Assert(u->refcount == 0 && "UMat deallocation error: some derived Mat is still alive");

        CV_Assert(u->handle != 0);
        CV_Assert(u->mapcount == 0);

        if (u->flags & UMatData::ASYNC_CLEANUP)
            addToCleanupQueue(u);
        else
            deallocate_(u);
    }

    void deallocate_(UMatData* u) const
    {
        if(u->tempUMat())
        {
            CV_Assert(u->origdata);
//            UMatDataAutoLock lock(u);

            if (u->hostCopyObsolete())
            {
#ifdef HAVE_OPENCL_SVM
                if ((u->allocatorFlags_ & svm::OPENCL_SVM_BUFFER_MASK) != 0)
                {
                    Context& ctx = Context::getDefault();
                    const svm::SVMFunctions* svmFns = svm::getSVMFunctions(ctx);
                    CV_DbgAssert(svmFns->isValid());

                    if( u->tempCopiedUMat() )
                    {
                        CV_DbgAssert((u->allocatorFlags_ & svm::OPENCL_SVM_BUFFER_MASK) == svm::OPENCL_SVM_FINE_GRAIN_BUFFER ||
                                (u->allocatorFlags_ & svm::OPENCL_SVM_BUFFER_MASK) == svm::OPENCL_SVM_COARSE_GRAIN_BUFFER);
                        bool isFineGrainBuffer = (u->allocatorFlags_ & svm::OPENCL_SVM_BUFFER_MASK) == svm::OPENCL_SVM_FINE_GRAIN_BUFFER;
                        cl_command_queue q = NULL;
                        if (!isFineGrainBuffer)
                        {
                            CV_DbgAssert(((u->allocatorFlags_ & svm::OPENCL_SVM_BUFFER_MAP) == 0));
                            q = (cl_command_queue)Queue::getDefault().ptr();
                            CV_OPENCL_SVM_TRACE_P("clEnqueueSVMMap: %p (%d)\n", u->handle, (int)u->size);
                            cl_int status = svmFns->fn_clEnqueueSVMMap(q, CL_FALSE, CL_MAP_READ,
                                    u->handle, u->size,
                                    0, NULL, NULL);
                            CV_Assert(status == CL_SUCCESS);
                        }
                        clFinish(q);
                        memcpy(u->origdata, u->handle, u->size);
                        if (!isFineGrainBuffer)
                        {
                            CV_OPENCL_SVM_TRACE_P("clEnqueueSVMUnmap: %p\n", u->handle);
                            cl_int status = svmFns->fn_clEnqueueSVMUnmap(q, u->handle, 0, NULL, NULL);
                            CV_Assert(status == CL_SUCCESS);
                        }
                    }
                    else
                    {
                        CV_DbgAssert((u->allocatorFlags_ & svm::OPENCL_SVM_BUFFER_MASK) == svm::OPENCL_SVM_FINE_GRAIN_SYSTEM);
                        // nothing
                    }
                }
                else
#endif
                {
                    cl_command_queue q = (cl_command_queue)Queue::getDefault().ptr();
                    if( u->tempCopiedUMat() )
                    {
                        AlignedDataPtr<false, true> alignedPtr(u->origdata, u->size, CV_OPENCL_DATA_PTR_ALIGNMENT);
                        CV_OclDbgAssert(clEnqueueReadBuffer(q, (cl_mem)u->handle, CL_TRUE, 0,
                                            u->size, alignedPtr.getAlignedPtr(), 0, 0, 0) == CL_SUCCESS);
                    }
                    else
                    {
                        cl_int retval = 0;
                        if (u->tempUMat())
                        {
                            CV_Assert(u->mapcount == 0);
                            void* data = clEnqueueMapBuffer(q, (cl_mem)u->handle, CL_TRUE,
                                (CL_MAP_READ | CL_MAP_WRITE),
                                0, u->size, 0, 0, 0, &retval);
                            CV_Assert(u->origdata == data);
                            CV_OclDbgAssert(retval == CL_SUCCESS);
                            if (u->originalUMatData)
                            {
                                CV_Assert(u->originalUMatData->data == data);
                            }
                            CV_OclDbgAssert(clEnqueueUnmapMemObject(q, (cl_mem)u->handle, data, 0, 0, 0) == CL_SUCCESS);
                            CV_OclDbgAssert(clFinish(q) == CL_SUCCESS);
                        }
                    }
                }
                u->markHostCopyObsolete(false);
            }
            else
            {
                // nothing
            }
#ifdef HAVE_OPENCL_SVM
            if ((u->allocatorFlags_ & svm::OPENCL_SVM_BUFFER_MASK) != 0)
            {
                if( u->tempCopiedUMat() )
                {
                    Context& ctx = Context::getDefault();
                    const svm::SVMFunctions* svmFns = svm::getSVMFunctions(ctx);
                    CV_DbgAssert(svmFns->isValid());

                    CV_OPENCL_SVM_TRACE_P("clSVMFree: %p\n", u->handle);
                    svmFns->fn_clSVMFree((cl_context)ctx.ptr(), u->handle);
                }
            }
            else
#endif
            {
                clReleaseMemObject((cl_mem)u->handle);
            }
            u->handle = 0;
            u->markDeviceCopyObsolete(true);
            u->currAllocator = u->prevAllocator;
            u->prevAllocator = NULL;
            if(u->data && u->copyOnMap() && u->data != u->origdata)
                fastFree(u->data);
            u->data = u->origdata;
            u->currAllocator->deallocate(u);
            u = NULL;
        }
        else
        {
            CV_Assert(u->origdata == NULL);
            if(u->data && u->copyOnMap() && u->data != u->origdata)
            {
                fastFree(u->data);
                u->data = 0;
                u->markHostCopyObsolete(true);
            }
            if (u->allocatorFlags_ & ALLOCATOR_FLAGS_BUFFER_POOL_USED)
            {
                bufferPool.release((cl_mem)u->handle);
            }
            else if (u->allocatorFlags_ & ALLOCATOR_FLAGS_BUFFER_POOL_HOST_PTR_USED)
            {
                bufferPoolHostPtr.release((cl_mem)u->handle);
            }
#ifdef HAVE_OPENCL_SVM
            else if (u->allocatorFlags_ & ALLOCATOR_FLAGS_BUFFER_POOL_SVM_USED)
            {
                if ((u->allocatorFlags_ & svm::OPENCL_SVM_BUFFER_MASK) == svm::OPENCL_SVM_FINE_GRAIN_SYSTEM)
                {
                    //nothing
                }
                else if ((u->allocatorFlags_ & svm::OPENCL_SVM_BUFFER_MASK) == svm::OPENCL_SVM_FINE_GRAIN_BUFFER ||
                        (u->allocatorFlags_ & svm::OPENCL_SVM_BUFFER_MASK) == svm::OPENCL_SVM_COARSE_GRAIN_BUFFER)
                {
                    Context& ctx = Context::getDefault();
                    const svm::SVMFunctions* svmFns = svm::getSVMFunctions(ctx);
                    CV_DbgAssert(svmFns->isValid());
                    cl_command_queue q = (cl_command_queue)Queue::getDefault().ptr();

                    if ((u->allocatorFlags_ & svm::OPENCL_SVM_BUFFER_MAP) != 0)
                    {
                        CV_OPENCL_SVM_TRACE_P("clEnqueueSVMUnmap: %p\n", u->handle);
                        cl_int status = svmFns->fn_clEnqueueSVMUnmap(q, u->handle, 0, NULL, NULL);
                        CV_Assert(status == CL_SUCCESS);
                    }
                }
                bufferPoolSVM.release((void*)u->handle);
            }
#endif
            else
            {
                clReleaseMemObject((cl_mem)u->handle);
            }
            u->handle = 0;
            u->markDeviceCopyObsolete(true);
            delete u;
            u = NULL;
        }
        CV_Assert(u == NULL);
    }

    // synchronized call (external UMatDataAutoLock, see UMat::getMat)
    void map(UMatData* u, int accessFlags) const
    {
        CV_Assert(u && u->handle);

        if(accessFlags & ACCESS_WRITE)
            u->markDeviceCopyObsolete(true);

        cl_command_queue q = (cl_command_queue)Queue::getDefault().ptr();

        {
            if( !u->copyOnMap() )
            {
                // TODO
                // because there can be other map requests for the same UMat with different access flags,
                // we use the universal (read-write) access mode.
#ifdef HAVE_OPENCL_SVM
                if ((u->allocatorFlags_ & svm::OPENCL_SVM_BUFFER_MASK) != 0)
                {
                    if ((u->allocatorFlags_ & svm::OPENCL_SVM_BUFFER_MASK) == svm::OPENCL_SVM_COARSE_GRAIN_BUFFER)
                    {
                        Context& ctx = Context::getDefault();
                        const svm::SVMFunctions* svmFns = svm::getSVMFunctions(ctx);
                        CV_DbgAssert(svmFns->isValid());

                        if ((u->allocatorFlags_ & svm::OPENCL_SVM_BUFFER_MAP) == 0)
                        {
                            CV_OPENCL_SVM_TRACE_P("clEnqueueSVMMap: %p (%d)\n", u->handle, (int)u->size);
                            cl_int status = svmFns->fn_clEnqueueSVMMap(q, CL_FALSE, CL_MAP_READ | CL_MAP_WRITE,
                                    u->handle, u->size,
                                    0, NULL, NULL);
                            CV_Assert(status == CL_SUCCESS);
                            u->allocatorFlags_ |= svm::OPENCL_SVM_BUFFER_MAP;
                        }
                    }
                    clFinish(q);
                    u->data = (uchar*)u->handle;
                    u->markHostCopyObsolete(false);
                    u->markDeviceMemMapped(true);
                    return;
                }
#endif

                cl_int retval = CL_SUCCESS;
                if (!u->deviceMemMapped())
                {
                    CV_Assert(u->refcount == 1);
                    CV_Assert(u->mapcount++ == 0);
                    u->data = (uchar*)clEnqueueMapBuffer(q, (cl_mem)u->handle, CL_TRUE,
                                                         (CL_MAP_READ | CL_MAP_WRITE),
                                                         0, u->size, 0, 0, 0, &retval);
                }
                if (u->data && retval == CL_SUCCESS)
                {
                    u->markHostCopyObsolete(false);
                    u->markDeviceMemMapped(true);
                    return;
                }

                // TODO Is it really a good idea and was it tested well?
                // if map failed, switch to copy-on-map mode for the particular buffer
                u->flags |= UMatData::COPY_ON_MAP;
            }

            if(!u->data)
            {
                u->data = (uchar*)fastMalloc(u->size);
                u->markHostCopyObsolete(true);
            }
        }

        if( (accessFlags & ACCESS_READ) != 0 && u->hostCopyObsolete() )
        {
            AlignedDataPtr<false, true> alignedPtr(u->data, u->size, CV_OPENCL_DATA_PTR_ALIGNMENT);
#ifdef HAVE_OPENCL_SVM
            CV_DbgAssert((u->allocatorFlags_ & svm::OPENCL_SVM_BUFFER_MASK) == 0);
#endif
            CV_Assert( clEnqueueReadBuffer(q, (cl_mem)u->handle, CL_TRUE, 0,
                                           u->size, alignedPtr.getAlignedPtr(), 0, 0, 0) == CL_SUCCESS );
            u->markHostCopyObsolete(false);
        }
    }

    void unmap(UMatData* u) const
    {
        if(!u)
            return;


        CV_Assert(u->handle != 0);

        UMatDataAutoLock autolock(u);

        cl_command_queue q = (cl_command_queue)Queue::getDefault().ptr();
        cl_int retval = 0;
        if( !u->copyOnMap() && u->deviceMemMapped() )
        {
            CV_Assert(u->data != NULL);
#ifdef HAVE_OPENCL_SVM
            if ((u->allocatorFlags_ & svm::OPENCL_SVM_BUFFER_MASK) != 0)
            {
                if ((u->allocatorFlags_ & svm::OPENCL_SVM_BUFFER_MASK) == svm::OPENCL_SVM_COARSE_GRAIN_BUFFER)
                {
                    Context& ctx = Context::getDefault();
                    const svm::SVMFunctions* svmFns = svm::getSVMFunctions(ctx);
                    CV_DbgAssert(svmFns->isValid());

                    CV_DbgAssert((u->allocatorFlags_ & svm::OPENCL_SVM_BUFFER_MAP) != 0);
                    {
                        CV_OPENCL_SVM_TRACE_P("clEnqueueSVMUnmap: %p\n", u->handle);
                        cl_int status = svmFns->fn_clEnqueueSVMUnmap(q, u->handle,
                                0, NULL, NULL);
                        CV_Assert(status == CL_SUCCESS);
                        clFinish(q);
                        u->allocatorFlags_ &= ~svm::OPENCL_SVM_BUFFER_MAP;
                    }
                }
                if (u->refcount == 0)
                    u->data = 0;
                u->markDeviceCopyObsolete(false);
                u->markHostCopyObsolete(true);
                return;
            }
#endif
            if (u->refcount == 0)
            {
                CV_Assert(u->mapcount-- == 1);
                CV_Assert((retval = clEnqueueUnmapMemObject(q,
                          (cl_mem)u->handle, u->data, 0, 0, 0)) == CL_SUCCESS);
                if (Device::getDefault().isAMD())
                {
                    // required for multithreaded applications (see stitching test)
                    CV_OclDbgAssert(clFinish(q) == CL_SUCCESS);
                }
                u->markDeviceMemMapped(false);
                u->data = 0;
                u->markDeviceCopyObsolete(false);
                u->markHostCopyObsolete(true);
            }
        }
        else if( u->copyOnMap() && u->deviceCopyObsolete() )
        {
            AlignedDataPtr<true, false> alignedPtr(u->data, u->size, CV_OPENCL_DATA_PTR_ALIGNMENT);
#ifdef HAVE_OPENCL_SVM
            CV_DbgAssert((u->allocatorFlags_ & svm::OPENCL_SVM_BUFFER_MASK) == 0);
#endif
            CV_Assert( (retval = clEnqueueWriteBuffer(q, (cl_mem)u->handle, CL_TRUE, 0,
                                u->size, alignedPtr.getAlignedPtr(), 0, 0, 0)) == CL_SUCCESS );
            u->markDeviceCopyObsolete(false);
            u->markHostCopyObsolete(true);
        }
    }

    bool checkContinuous(int dims, const size_t sz[],
                         const size_t srcofs[], const size_t srcstep[],
                         const size_t dstofs[], const size_t dststep[],
                         size_t& total, size_t new_sz[],
                         size_t& srcrawofs, size_t new_srcofs[], size_t new_srcstep[],
                         size_t& dstrawofs, size_t new_dstofs[], size_t new_dststep[]) const
    {
        bool iscontinuous = true;
        srcrawofs = srcofs ? srcofs[dims-1] : 0;
        dstrawofs = dstofs ? dstofs[dims-1] : 0;
        total = sz[dims-1];
        for( int i = dims-2; i >= 0; i-- )
        {
            if( i >= 0 && (total != srcstep[i] || total != dststep[i]) )
                iscontinuous = false;
            total *= sz[i];
            if( srcofs )
                srcrawofs += srcofs[i]*srcstep[i];
            if( dstofs )
                dstrawofs += dstofs[i]*dststep[i];
        }

        if( !iscontinuous )
        {
            // OpenCL uses {x, y, z} order while OpenCV uses {z, y, x} order.
            if( dims == 2 )
            {
                new_sz[0] = sz[1]; new_sz[1] = sz[0]; new_sz[2] = 1;
                // we assume that new_... arrays are initialized by caller
                // with 0's, so there is no else branch
                if( srcofs )
                {
                    new_srcofs[0] = srcofs[1];
                    new_srcofs[1] = srcofs[0];
                    new_srcofs[2] = 0;
                }

                if( dstofs )
                {
                    new_dstofs[0] = dstofs[1];
                    new_dstofs[1] = dstofs[0];
                    new_dstofs[2] = 0;
                }

                new_srcstep[0] = srcstep[0]; new_srcstep[1] = 0;
                new_dststep[0] = dststep[0]; new_dststep[1] = 0;
            }
            else
            {
                // we could check for dims == 3 here,
                // but from user perspective this one is more informative
                CV_Assert(dims <= 3);
                new_sz[0] = sz[2]; new_sz[1] = sz[1]; new_sz[2] = sz[0];
                if( srcofs )
                {
                    new_srcofs[0] = srcofs[2];
                    new_srcofs[1] = srcofs[1];
                    new_srcofs[2] = srcofs[0];
                }

                if( dstofs )
                {
                    new_dstofs[0] = dstofs[2];
                    new_dstofs[1] = dstofs[1];
                    new_dstofs[2] = dstofs[0];
                }

                new_srcstep[0] = srcstep[1]; new_srcstep[1] = srcstep[0];
                new_dststep[0] = dststep[1]; new_dststep[1] = dststep[0];
            }
        }
        return iscontinuous;
    }

    void download(UMatData* u, void* dstptr, int dims, const size_t sz[],
                  const size_t srcofs[], const size_t srcstep[],
                  const size_t dststep[]) const
    {
        if(!u)
            return;
        UMatDataAutoLock autolock(u);

        if( u->data && !u->hostCopyObsolete() )
        {
            Mat::getDefaultAllocator()->download(u, dstptr, dims, sz, srcofs, srcstep, dststep);
            return;
        }
        CV_Assert( u->handle != 0 );

        cl_command_queue q = (cl_command_queue)Queue::getDefault().ptr();

        size_t total = 0, new_sz[] = {0, 0, 0};
        size_t srcrawofs = 0, new_srcofs[] = {0, 0, 0}, new_srcstep[] = {0, 0, 0};
        size_t dstrawofs = 0, new_dstofs[] = {0, 0, 0}, new_dststep[] = {0, 0, 0};

        bool iscontinuous = checkContinuous(dims, sz, srcofs, srcstep, 0, dststep,
                                            total, new_sz,
                                            srcrawofs, new_srcofs, new_srcstep,
                                            dstrawofs, new_dstofs, new_dststep);

#ifdef HAVE_OPENCL_SVM
        if ((u->allocatorFlags_ & svm::OPENCL_SVM_BUFFER_MASK) != 0)
        {
            CV_DbgAssert(u->data == NULL || u->data == u->handle);
            Context& ctx = Context::getDefault();
            const svm::SVMFunctions* svmFns = svm::getSVMFunctions(ctx);
            CV_DbgAssert(svmFns->isValid());

            CV_DbgAssert((u->allocatorFlags_ & svm::OPENCL_SVM_BUFFER_MAP) == 0);
            if ((u->allocatorFlags_ & svm::OPENCL_SVM_BUFFER_MASK) == svm::OPENCL_SVM_COARSE_GRAIN_BUFFER)
            {
                CV_OPENCL_SVM_TRACE_P("clEnqueueSVMMap: %p (%d)\n", u->handle, (int)u->size);
                cl_int status = svmFns->fn_clEnqueueSVMMap(q, CL_FALSE, CL_MAP_READ,
                        u->handle, u->size,
                        0, NULL, NULL);
                CV_Assert(status == CL_SUCCESS);
            }
            clFinish(q);
            if( iscontinuous )
            {
                memcpy(dstptr, (uchar*)u->handle + srcrawofs, total);
            }
            else
            {
                // This code is from MatAllocator::download()
                int isz[CV_MAX_DIM];
                uchar* srcptr = (uchar*)u->handle;
                for( int i = 0; i < dims; i++ )
                {
                    CV_Assert( sz[i] <= (size_t)INT_MAX );
                    if( sz[i] == 0 )
                    return;
                    if( srcofs )
                    srcptr += srcofs[i]*(i <= dims-2 ? srcstep[i] : 1);
                    isz[i] = (int)sz[i];
                }

                Mat src(dims, isz, CV_8U, srcptr, srcstep);
                Mat dst(dims, isz, CV_8U, dstptr, dststep);

                const Mat* arrays[] = { &src, &dst };
                uchar* ptrs[2];
                NAryMatIterator it(arrays, ptrs, 2);
                size_t j, planesz = it.size;

                for( j = 0; j < it.nplanes; j++, ++it )
                    memcpy(ptrs[1], ptrs[0], planesz);
            }
            if ((u->allocatorFlags_ & svm::OPENCL_SVM_BUFFER_MASK) == svm::OPENCL_SVM_COARSE_GRAIN_BUFFER)
            {
                CV_OPENCL_SVM_TRACE_P("clEnqueueSVMUnmap: %p\n", u->handle);
                cl_int status = svmFns->fn_clEnqueueSVMUnmap(q, u->handle,
                        0, NULL, NULL);
                CV_Assert(status == CL_SUCCESS);
                clFinish(q);
            }
        }
        else
#endif
        {
            if( iscontinuous )
            {
                AlignedDataPtr<false, true> alignedPtr((uchar*)dstptr, total, CV_OPENCL_DATA_PTR_ALIGNMENT);
                CV_Assert(clEnqueueReadBuffer(q, (cl_mem)u->handle, CL_TRUE,
                    srcrawofs, total, alignedPtr.getAlignedPtr(), 0, 0, 0) >= 0 );
            }
            else
            {
                AlignedDataPtr2D<false, true> alignedPtr((uchar*)dstptr, new_sz[1], new_sz[0], new_dststep[0], CV_OPENCL_DATA_PTR_ALIGNMENT);
                uchar* ptr = alignedPtr.getAlignedPtr();

                CV_Assert( clEnqueueReadBufferRect(q, (cl_mem)u->handle, CL_TRUE,
                    new_srcofs, new_dstofs, new_sz,
                    new_srcstep[0], 0,
                    new_dststep[0], 0,
                    ptr, 0, 0, 0) >= 0 );
            }
        }
    }

    void upload(UMatData* u, const void* srcptr, int dims, const size_t sz[],
                const size_t dstofs[], const size_t dststep[],
                const size_t srcstep[]) const
    {
        if(!u)
            return;

        // there should be no user-visible CPU copies of the UMat which we are going to copy to
        CV_Assert(u->refcount == 0 || u->tempUMat());

        size_t total = 0, new_sz[] = {0, 0, 0};
        size_t srcrawofs = 0, new_srcofs[] = {0, 0, 0}, new_srcstep[] = {0, 0, 0};
        size_t dstrawofs = 0, new_dstofs[] = {0, 0, 0}, new_dststep[] = {0, 0, 0};

        bool iscontinuous = checkContinuous(dims, sz, 0, srcstep, dstofs, dststep,
                                            total, new_sz,
                                            srcrawofs, new_srcofs, new_srcstep,
                                            dstrawofs, new_dstofs, new_dststep);

        UMatDataAutoLock autolock(u);

        // if there is cached CPU copy of the GPU matrix,
        // we could use it as a destination.
        // we can do it in 2 cases:
        //    1. we overwrite the whole content
        //    2. we overwrite part of the matrix, but the GPU copy is out-of-date
        if( u->data && (u->hostCopyObsolete() < u->deviceCopyObsolete() || total == u->size))
        {
            Mat::getDefaultAllocator()->upload(u, srcptr, dims, sz, dstofs, dststep, srcstep);
            u->markHostCopyObsolete(false);
            u->markDeviceCopyObsolete(true);
            return;
        }

        CV_Assert( u->handle != 0 );
        cl_command_queue q = (cl_command_queue)Queue::getDefault().ptr();

#ifdef HAVE_OPENCL_SVM
        if ((u->allocatorFlags_ & svm::OPENCL_SVM_BUFFER_MASK) != 0)
        {
            CV_DbgAssert(u->data == NULL || u->data == u->handle);
            Context& ctx = Context::getDefault();
            const svm::SVMFunctions* svmFns = svm::getSVMFunctions(ctx);
            CV_DbgAssert(svmFns->isValid());

            CV_DbgAssert((u->allocatorFlags_ & svm::OPENCL_SVM_BUFFER_MAP) == 0);
            if ((u->allocatorFlags_ & svm::OPENCL_SVM_BUFFER_MASK) == svm::OPENCL_SVM_COARSE_GRAIN_BUFFER)
            {
                CV_OPENCL_SVM_TRACE_P("clEnqueueSVMMap: %p (%d)\n", u->handle, (int)u->size);
                cl_int status = svmFns->fn_clEnqueueSVMMap(q, CL_FALSE, CL_MAP_WRITE,
                        u->handle, u->size,
                        0, NULL, NULL);
                CV_Assert(status == CL_SUCCESS);
            }
            clFinish(q);
            if( iscontinuous )
            {
                memcpy((uchar*)u->handle + dstrawofs, srcptr, total);
            }
            else
            {
                // This code is from MatAllocator::upload()
                int isz[CV_MAX_DIM];
                uchar* dstptr = (uchar*)u->handle;
                for( int i = 0; i < dims; i++ )
                {
                    CV_Assert( sz[i] <= (size_t)INT_MAX );
                    if( sz[i] == 0 )
                    return;
                    if( dstofs )
                    dstptr += dstofs[i]*(i <= dims-2 ? dststep[i] : 1);
                    isz[i] = (int)sz[i];
                }

                Mat src(dims, isz, CV_8U, (void*)srcptr, srcstep);
                Mat dst(dims, isz, CV_8U, dstptr, dststep);

                const Mat* arrays[] = { &src, &dst };
                uchar* ptrs[2];
                NAryMatIterator it(arrays, ptrs, 2);
                size_t j, planesz = it.size;

                for( j = 0; j < it.nplanes; j++, ++it )
                    memcpy(ptrs[1], ptrs[0], planesz);
            }
            if ((u->allocatorFlags_ & svm::OPENCL_SVM_BUFFER_MASK) == svm::OPENCL_SVM_COARSE_GRAIN_BUFFER)
            {
                CV_OPENCL_SVM_TRACE_P("clEnqueueSVMUnmap: %p\n", u->handle);
                cl_int status = svmFns->fn_clEnqueueSVMUnmap(q, u->handle,
                        0, NULL, NULL);
                CV_Assert(status == CL_SUCCESS);
                clFinish(q);
            }
        }
        else
#endif
        {
            if( iscontinuous )
            {
                AlignedDataPtr<true, false> alignedPtr((uchar*)srcptr, total, CV_OPENCL_DATA_PTR_ALIGNMENT);
                CV_Assert(clEnqueueWriteBuffer(q, (cl_mem)u->handle, CL_TRUE,
                    dstrawofs, total, alignedPtr.getAlignedPtr(), 0, 0, 0) >= 0);
            }
            else
            {
                AlignedDataPtr2D<true, false> alignedPtr((uchar*)srcptr, new_sz[1], new_sz[0], new_srcstep[0], CV_OPENCL_DATA_PTR_ALIGNMENT);
                uchar* ptr = alignedPtr.getAlignedPtr();

                CV_Assert(clEnqueueWriteBufferRect(q, (cl_mem)u->handle, CL_TRUE,
                    new_dstofs, new_srcofs, new_sz,
                    new_dststep[0], 0,
                    new_srcstep[0], 0,
                    ptr, 0, 0, 0) >= 0 );
            }
        }
        u->markHostCopyObsolete(true);
#ifdef HAVE_OPENCL_SVM
        if ((u->allocatorFlags_ & svm::OPENCL_SVM_BUFFER_MASK) == svm::OPENCL_SVM_FINE_GRAIN_BUFFER ||
                (u->allocatorFlags_ & svm::OPENCL_SVM_BUFFER_MASK) == svm::OPENCL_SVM_FINE_GRAIN_SYSTEM)
        {
            // nothing
        }
        else
#endif
        {
            u->markHostCopyObsolete(true);
        }
        u->markDeviceCopyObsolete(false);
    }

    void copy(UMatData* src, UMatData* dst, int dims, const size_t sz[],
              const size_t srcofs[], const size_t srcstep[],
              const size_t dstofs[], const size_t dststep[], bool _sync) const
    {
        if(!src || !dst)
            return;

        size_t total = 0, new_sz[] = {0, 0, 0};
        size_t srcrawofs = 0, new_srcofs[] = {0, 0, 0}, new_srcstep[] = {0, 0, 0};
        size_t dstrawofs = 0, new_dstofs[] = {0, 0, 0}, new_dststep[] = {0, 0, 0};

        bool iscontinuous = checkContinuous(dims, sz, srcofs, srcstep, dstofs, dststep,
                                            total, new_sz,
                                            srcrawofs, new_srcofs, new_srcstep,
                                            dstrawofs, new_dstofs, new_dststep);

        UMatDataAutoLock src_autolock(src);
        UMatDataAutoLock dst_autolock(dst);

        if( !src->handle || (src->data && src->hostCopyObsolete() < src->deviceCopyObsolete()) )
        {
            upload(dst, src->data + srcrawofs, dims, sz, dstofs, dststep, srcstep);
            return;
        }
        if( !dst->handle || (dst->data && dst->hostCopyObsolete() < dst->deviceCopyObsolete()) )
        {
            download(src, dst->data + dstrawofs, dims, sz, srcofs, srcstep, dststep);
            dst->markHostCopyObsolete(false);
#ifdef HAVE_OPENCL_SVM
            if ((dst->allocatorFlags_ & svm::OPENCL_SVM_BUFFER_MASK) == svm::OPENCL_SVM_FINE_GRAIN_BUFFER ||
                    (dst->allocatorFlags_ & svm::OPENCL_SVM_BUFFER_MASK) == svm::OPENCL_SVM_FINE_GRAIN_SYSTEM)
            {
                // nothing
            }
            else
#endif
            {
                dst->markDeviceCopyObsolete(true);
            }
            return;
        }

        // there should be no user-visible CPU copies of the UMat which we are going to copy to
        CV_Assert(dst->refcount == 0);
        cl_command_queue q = (cl_command_queue)Queue::getDefault().ptr();

        cl_int retval = CL_SUCCESS;
#ifdef HAVE_OPENCL_SVM
        if ((src->allocatorFlags_ & svm::OPENCL_SVM_BUFFER_MASK) != 0 ||
                (dst->allocatorFlags_ & svm::OPENCL_SVM_BUFFER_MASK) != 0)
        {
            if ((src->allocatorFlags_ & svm::OPENCL_SVM_BUFFER_MASK) != 0 &&
                            (dst->allocatorFlags_ & svm::OPENCL_SVM_BUFFER_MASK) != 0)
            {
                Context& ctx = Context::getDefault();
                const svm::SVMFunctions* svmFns = svm::getSVMFunctions(ctx);
                CV_DbgAssert(svmFns->isValid());

                if( iscontinuous )
                {
                    CV_OPENCL_SVM_TRACE_P("clEnqueueSVMMemcpy: %p <-- %p (%d)\n",
                            (uchar*)dst->handle + dstrawofs, (uchar*)src->handle + srcrawofs, (int)total);
                    cl_int status = svmFns->fn_clEnqueueSVMMemcpy(q, CL_TRUE,
                            (uchar*)dst->handle + dstrawofs, (uchar*)src->handle + srcrawofs,
                            total, 0, NULL, NULL);
                    CV_Assert(status == CL_SUCCESS);
                }
                else
                {
                    clFinish(q);
                    // This code is from MatAllocator::download()/upload()
                    int isz[CV_MAX_DIM];
                    uchar* srcptr = (uchar*)src->handle;
                    for( int i = 0; i < dims; i++ )
                    {
                        CV_Assert( sz[i] <= (size_t)INT_MAX );
                        if( sz[i] == 0 )
                        return;
                        if( srcofs )
                        srcptr += srcofs[i]*(i <= dims-2 ? srcstep[i] : 1);
                        isz[i] = (int)sz[i];
                    }
                    Mat m_src(dims, isz, CV_8U, srcptr, srcstep);

                    uchar* dstptr = (uchar*)dst->handle;
                    for( int i = 0; i < dims; i++ )
                    {
                        if( dstofs )
                        dstptr += dstofs[i]*(i <= dims-2 ? dststep[i] : 1);
                    }
                    Mat m_dst(dims, isz, CV_8U, dstptr, dststep);

                    const Mat* arrays[] = { &m_src, &m_dst };
                    uchar* ptrs[2];
                    NAryMatIterator it(arrays, ptrs, 2);
                    size_t j, planesz = it.size;

                    for( j = 0; j < it.nplanes; j++, ++it )
                        memcpy(ptrs[1], ptrs[0], planesz);
                }
            }
            else
            {
                if ((src->allocatorFlags_ & svm::OPENCL_SVM_BUFFER_MASK) != 0)
                {
                    map(src, ACCESS_READ);
                    upload(dst, src->data + srcrawofs, dims, sz, dstofs, dststep, srcstep);
                    unmap(src);
                }
                else
                {
                    map(dst, ACCESS_WRITE);
                    download(src, dst->data + dstrawofs, dims, sz, srcofs, srcstep, dststep);
                    unmap(dst);
                }
            }
        }
        else
#endif
        {
            if( iscontinuous )
            {
                CV_Assert( (retval = clEnqueueCopyBuffer(q, (cl_mem)src->handle, (cl_mem)dst->handle,
                                               srcrawofs, dstrawofs, total, 0, 0, 0)) == CL_SUCCESS );
            }
            else
            {
                CV_Assert( (retval = clEnqueueCopyBufferRect(q, (cl_mem)src->handle, (cl_mem)dst->handle,
                                                   new_srcofs, new_dstofs, new_sz,
                                                   new_srcstep[0], 0,
                                                   new_dststep[0], 0,
                                                   0, 0, 0)) == CL_SUCCESS );
            }
        }
        if (retval == CL_SUCCESS)
        {
            CV_IMPL_ADD(CV_IMPL_OCL)
        }

#ifdef HAVE_OPENCL_SVM
        if ((dst->allocatorFlags_ & svm::OPENCL_SVM_BUFFER_MASK) == svm::OPENCL_SVM_FINE_GRAIN_BUFFER ||
            (dst->allocatorFlags_ & svm::OPENCL_SVM_BUFFER_MASK) == svm::OPENCL_SVM_FINE_GRAIN_SYSTEM)
        {
            // nothing
        }
        else
#endif
        {
            dst->markHostCopyObsolete(true);
        }
        dst->markDeviceCopyObsolete(false);

        if( _sync )
        {
            CV_OclDbgAssert(clFinish(q) == CL_SUCCESS);
        }
    }

    BufferPoolController* getBufferPoolController(const char* id) const {
#ifdef HAVE_OPENCL_SVM
        if ((svm::checkForceSVMUmatUsage() && (id == NULL || strcmp(id, "OCL") == 0)) || (id != NULL && strcmp(id, "SVM") == 0))
        {
            return &bufferPoolSVM;
        }
#endif
        if (id != NULL && strcmp(id, "HOST_ALLOC") == 0)
        {
            return &bufferPoolHostPtr;
        }
        if (id != NULL && strcmp(id, "OCL") != 0)
        {
            CV_ErrorNoReturn(cv::Error::StsBadArg, "getBufferPoolController(): unknown BufferPool ID\n");
        }
        return &bufferPool;
    }

    MatAllocator* matStdAllocator;

    mutable cv::Mutex cleanupQueueMutex;
    mutable std::deque<UMatData*> cleanupQueue;

    void flushCleanupQueue() const
    {
        if (!cleanupQueue.empty())
        {
            std::deque<UMatData*> q;
            {
                cv::AutoLock lock(cleanupQueueMutex);
                q.swap(cleanupQueue);
            }
            for (std::deque<UMatData*>::const_iterator i = q.begin(); i != q.end(); ++i)
            {
                deallocate_(*i);
            }
        }
    }
    void addToCleanupQueue(UMatData* u) const
    {
        //TODO: Validation check: CV_Assert(!u->tempUMat());
        {
            cv::AutoLock lock(cleanupQueueMutex);
            cleanupQueue.push_back(u);
        }
    }
};

MatAllocator* getOpenCLAllocator()
{
    CV_SINGLETON_LAZY_INIT(MatAllocator, new OpenCLAllocator())
}

}} // namespace cv::ocl


namespace cv {

// three funcs below are implemented in umatrix.cpp
void setSize( UMat& m, int _dims, const int* _sz, const size_t* _steps,
              bool autoSteps = false );

void updateContinuityFlag(UMat& m);
void finalizeHdr(UMat& m);

} // namespace cv


namespace cv { namespace ocl {

/*
// Convert OpenCL buffer memory to UMat
*/
void convertFromBuffer(void* cl_mem_buffer, size_t step, int rows, int cols, int type, UMat& dst)
{
    int d = 2;
    int sizes[] = { rows, cols };

    CV_Assert(0 <= d && d <= CV_MAX_DIM);

    dst.release();

    dst.flags      = (type & Mat::TYPE_MASK) | Mat::MAGIC_VAL;
    dst.usageFlags = USAGE_DEFAULT;

    setSize(dst, d, sizes, 0, true);
    dst.offset = 0;

    cl_mem             memobj = (cl_mem)cl_mem_buffer;
    cl_mem_object_type mem_type = 0;

    CV_Assert(clGetMemObjectInfo(memobj, CL_MEM_TYPE, sizeof(cl_mem_object_type), &mem_type, 0) == CL_SUCCESS);

    CV_Assert(CL_MEM_OBJECT_BUFFER == mem_type);

    size_t total = 0;
    CV_Assert(clGetMemObjectInfo(memobj, CL_MEM_SIZE, sizeof(size_t), &total, 0) == CL_SUCCESS);

    CV_Assert(clRetainMemObject(memobj) == CL_SUCCESS);

    CV_Assert((int)step >= cols * CV_ELEM_SIZE(type));
    CV_Assert(total >= rows * step);

    // attach clBuffer to UMatData
    dst.u = new UMatData(getOpenCLAllocator());
    dst.u->data            = 0;
    dst.u->allocatorFlags_ = 0; // not allocated from any OpenCV buffer pool
    dst.u->flags           = 0;
    dst.u->handle          = cl_mem_buffer;
    dst.u->origdata        = 0;
    dst.u->prevAllocator   = 0;
    dst.u->size            = total;

    finalizeHdr(dst);
    dst.addref();

    return;
} // convertFromBuffer()


/*
// Convert OpenCL image2d_t memory to UMat
*/
void convertFromImage(void* cl_mem_image, UMat& dst)
{
    cl_mem             clImage = (cl_mem)cl_mem_image;
    cl_mem_object_type mem_type = 0;

    CV_Assert(clGetMemObjectInfo(clImage, CL_MEM_TYPE, sizeof(cl_mem_object_type), &mem_type, 0) == CL_SUCCESS);

    CV_Assert(CL_MEM_OBJECT_IMAGE2D == mem_type);

    cl_image_format fmt = { 0, 0 };
    CV_Assert(clGetImageInfo(clImage, CL_IMAGE_FORMAT, sizeof(cl_image_format), &fmt, 0) == CL_SUCCESS);

    int depth = CV_8U;
    switch (fmt.image_channel_data_type)
    {
    case CL_UNORM_INT8:
    case CL_UNSIGNED_INT8:
        depth = CV_8U;
        break;

    case CL_SNORM_INT8:
    case CL_SIGNED_INT8:
        depth = CV_8S;
        break;

    case CL_UNORM_INT16:
    case CL_UNSIGNED_INT16:
        depth = CV_16U;
        break;

    case CL_SNORM_INT16:
    case CL_SIGNED_INT16:
        depth = CV_16S;
        break;

    case CL_SIGNED_INT32:
        depth = CV_32S;
        break;

    case CL_FLOAT:
        depth = CV_32F;
        break;

    default:
        CV_Error(cv::Error::OpenCLApiCallError, "Not supported image_channel_data_type");
    }

    int type = CV_8UC1;
    switch (fmt.image_channel_order)
    {
    case CL_R:
        type = CV_MAKE_TYPE(depth, 1);
        break;

    case CL_RGBA:
    case CL_BGRA:
    case CL_ARGB:
         type = CV_MAKE_TYPE(depth, 4);
        break;

    default:
        CV_Error(cv::Error::OpenCLApiCallError, "Not supported image_channel_order");
        break;
    }

    size_t step = 0;
    CV_Assert(clGetImageInfo(clImage, CL_IMAGE_ROW_PITCH, sizeof(size_t), &step, 0) == CL_SUCCESS);

    size_t w = 0;
    CV_Assert(clGetImageInfo(clImage, CL_IMAGE_WIDTH, sizeof(size_t), &w, 0) == CL_SUCCESS);

    size_t h = 0;
    CV_Assert(clGetImageInfo(clImage, CL_IMAGE_HEIGHT, sizeof(size_t), &h, 0) == CL_SUCCESS);

    dst.create((int)h, (int)w, type);

    cl_mem clBuffer = (cl_mem)dst.handle(ACCESS_READ);

    cl_command_queue q = (cl_command_queue)Queue::getDefault().ptr();

    size_t offset = 0;
    size_t src_origin[3] = { 0, 0, 0 };
    size_t region[3] = { w, h, 1 };
    CV_Assert(clEnqueueCopyImageToBuffer(q, clImage, clBuffer, src_origin, region, offset, 0, NULL, NULL) == CL_SUCCESS);

    CV_Assert(clFinish(q) == CL_SUCCESS);

    return;
} // convertFromImage()


///////////////////////////////////////////// Utility functions /////////////////////////////////////////////////

static void getDevices(std::vector<cl_device_id>& devices, cl_platform_id platform)
{
    cl_uint numDevices = 0;
    CV_OclDbgAssert(clGetDeviceIDs(platform, (cl_device_type)Device::TYPE_ALL,
                                0, NULL, &numDevices) == CL_SUCCESS);

    if (numDevices == 0)
    {
        devices.clear();
        return;
    }

    devices.resize((size_t)numDevices);
    CV_OclDbgAssert(clGetDeviceIDs(platform, (cl_device_type)Device::TYPE_ALL,
                                numDevices, &devices[0], &numDevices) == CL_SUCCESS);
}

struct PlatformInfo::Impl
{
    Impl(void* id)
    {
        refcount = 1;
        handle = *(cl_platform_id*)id;
        getDevices(devices, handle);
    }

    String getStrProp(cl_platform_info prop) const
    {
        char buf[1024];
        size_t sz=0;
        return clGetPlatformInfo(handle, prop, sizeof(buf)-16, buf, &sz) == CL_SUCCESS &&
            sz < sizeof(buf) ? String(buf) : String();
    }

    IMPLEMENT_REFCOUNTABLE();
    std::vector<cl_device_id> devices;
    cl_platform_id handle;
};

PlatformInfo::PlatformInfo()
{
    p = 0;
}

PlatformInfo::PlatformInfo(void* platform_id)
{
    p = new Impl(platform_id);
}

PlatformInfo::~PlatformInfo()
{
    if(p)
        p->release();
}

PlatformInfo::PlatformInfo(const PlatformInfo& i)
{
    if (i.p)
        i.p->addref();
    p = i.p;
}

PlatformInfo& PlatformInfo::operator =(const PlatformInfo& i)
{
    if (i.p != p)
    {
        if (i.p)
            i.p->addref();
        if (p)
            p->release();
        p = i.p;
    }
    return *this;
}

int PlatformInfo::deviceNumber() const
{
    return p ? (int)p->devices.size() : 0;
}

void PlatformInfo::getDevice(Device& device, int d) const
{
    CV_Assert(p && d < (int)p->devices.size() );
    if(p)
        device.set(p->devices[d]);
}

String PlatformInfo::name() const
{
    return p ? p->getStrProp(CL_PLATFORM_NAME) : String();
}

String PlatformInfo::vendor() const
{
    return p ? p->getStrProp(CL_PLATFORM_VENDOR) : String();
}

String PlatformInfo::version() const
{
    return p ? p->getStrProp(CL_PLATFORM_VERSION) : String();
}

static void getPlatforms(std::vector<cl_platform_id>& platforms)
{
    cl_uint numPlatforms = 0;
    CV_OclDbgAssert(clGetPlatformIDs(0, NULL, &numPlatforms) == CL_SUCCESS);

    if (numPlatforms == 0)
    {
        platforms.clear();
        return;
    }

    platforms.resize((size_t)numPlatforms);
    CV_OclDbgAssert(clGetPlatformIDs(numPlatforms, &platforms[0], &numPlatforms) == CL_SUCCESS);
}

void getPlatfomsInfo(std::vector<PlatformInfo>& platformsInfo)
{
    std::vector<cl_platform_id> platforms;
    getPlatforms(platforms);

    for (size_t i = 0; i < platforms.size(); i++)
        platformsInfo.push_back( PlatformInfo((void*)&platforms[i]) );
}

const char* typeToStr(int type)
{
    static const char* tab[]=
    {
        "uchar", "uchar2", "uchar3", "uchar4", 0, 0, 0, "uchar8", 0, 0, 0, 0, 0, 0, 0, "uchar16",
        "char", "char2", "char3", "char4", 0, 0, 0, "char8", 0, 0, 0, 0, 0, 0, 0, "char16",
        "ushort", "ushort2", "ushort3", "ushort4",0, 0, 0, "ushort8", 0, 0, 0, 0, 0, 0, 0, "ushort16",
        "short", "short2", "short3", "short4", 0, 0, 0, "short8", 0, 0, 0, 0, 0, 0, 0, "short16",
        "int", "int2", "int3", "int4", 0, 0, 0, "int8", 0, 0, 0, 0, 0, 0, 0, "int16",
        "float", "float2", "float3", "float4", 0, 0, 0, "float8", 0, 0, 0, 0, 0, 0, 0, "float16",
        "double", "double2", "double3", "double4", 0, 0, 0, "double8", 0, 0, 0, 0, 0, 0, 0, "double16",
        "?", "?", "?", "?", "?", "?", "?", "?", "?", "?", "?", "?", "?", "?", "?", "?"
    };
    int cn = CV_MAT_CN(type), depth = CV_MAT_DEPTH(type);
    return cn > 16 ? "?" : tab[depth*16 + cn-1];
}

const char* memopTypeToStr(int type)
{
    static const char* tab[] =
    {
        "uchar", "uchar2", "uchar3", "uchar4", 0, 0, 0, "uchar8", 0, 0, 0, 0, 0, 0, 0, "uchar16",
        "char", "char2", "char3", "char4", 0, 0, 0, "char8", 0, 0, 0, 0, 0, 0, 0, "char16",
        "ushort", "ushort2", "ushort3", "ushort4",0, 0, 0, "ushort8", 0, 0, 0, 0, 0, 0, 0, "ushort16",
        "short", "short2", "short3", "short4", 0, 0, 0, "short8", 0, 0, 0, 0, 0, 0, 0, "short16",
        "int", "int2", "int3", "int4", 0, 0, 0, "int8", 0, 0, 0, 0, 0, 0, 0, "int16",
        "int", "int2", "int3", "int4", 0, 0, 0, "int8", 0, 0, 0, 0, 0, 0, 0, "int16",
        "ulong", "ulong2", "ulong3", "ulong4", 0, 0, 0, "ulong8", 0, 0, 0, 0, 0, 0, 0, "ulong16",
        "?", "?", "?", "?", "?", "?", "?", "?", "?", "?", "?", "?", "?", "?", "?", "?"
    };
    int cn = CV_MAT_CN(type), depth = CV_MAT_DEPTH(type);
    return cn > 16 ? "?" : tab[depth*16 + cn-1];
}

const char* vecopTypeToStr(int type)
{
    static const char* tab[] =
    {
        "uchar", "short", "uchar3", "int", 0, 0, 0, "int2", 0, 0, 0, 0, 0, 0, 0, "int4",
        "char", "short", "char3", "int", 0, 0, 0, "int2", 0, 0, 0, 0, 0, 0, 0, "int4",
        "ushort", "int", "ushort3", "int2",0, 0, 0, "int4", 0, 0, 0, 0, 0, 0, 0, "int8",
        "short", "int", "short3", "int2", 0, 0, 0, "int4", 0, 0, 0, 0, 0, 0, 0, "int8",
        "int", "int2", "int3", "int4", 0, 0, 0, "int8", 0, 0, 0, 0, 0, 0, 0, "int16",
        "int", "int2", "int3", "int4", 0, 0, 0, "int8", 0, 0, 0, 0, 0, 0, 0, "int16",
        "ulong", "ulong2", "ulong3", "ulong4", 0, 0, 0, "ulong8", 0, 0, 0, 0, 0, 0, 0, "ulong16",
        "?", "?", "?", "?", "?", "?", "?", "?", "?", "?", "?", "?", "?", "?", "?", "?"
    };
    int cn = CV_MAT_CN(type), depth = CV_MAT_DEPTH(type);
    return cn > 16 ? "?" : tab[depth*16 + cn-1];
}

const char* convertTypeStr(int sdepth, int ddepth, int cn, char* buf)
{
    if( sdepth == ddepth )
        return "noconvert";
    const char *typestr = typeToStr(CV_MAKETYPE(ddepth, cn));
    if( ddepth >= CV_32F ||
        (ddepth == CV_32S && sdepth < CV_32S) ||
        (ddepth == CV_16S && sdepth <= CV_8S) ||
        (ddepth == CV_16U && sdepth == CV_8U))
    {
        sprintf(buf, "convert_%s", typestr);
    }
    else if( sdepth >= CV_32F )
        sprintf(buf, "convert_%s%s_rte", typestr, (ddepth < CV_32S ? "_sat" : ""));
    else
        sprintf(buf, "convert_%s_sat", typestr);

    return buf;
}

const char* getOpenCLErrorString(int errorCode)
{
    switch (errorCode)
    {
    case   0: return "CL_SUCCESS";
    case  -1: return "CL_DEVICE_NOT_FOUND";
    case  -2: return "CL_DEVICE_NOT_AVAILABLE";
    case  -3: return "CL_COMPILER_NOT_AVAILABLE";
    case  -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case  -5: return "CL_OUT_OF_RESOURCES";
    case  -6: return "CL_OUT_OF_HOST_MEMORY";
    case  -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case  -8: return "CL_MEM_COPY_OVERLAP";
    case  -9: return "CL_IMAGE_FORMAT_MISMATCH";
    case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case -11: return "CL_BUILD_PROGRAM_FAILURE";
    case -12: return "CL_MAP_FAILURE";
    case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
    case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
    case -15: return "CL_COMPILE_PROGRAM_FAILURE";
    case -16: return "CL_LINKER_NOT_AVAILABLE";
    case -17: return "CL_LINK_PROGRAM_FAILURE";
    case -18: return "CL_DEVICE_PARTITION_FAILED";
    case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
    case -30: return "CL_INVALID_VALUE";
    case -31: return "CL_INVALID_DEVICE_TYPE";
    case -32: return "CL_INVALID_PLATFORM";
    case -33: return "CL_INVALID_DEVICE";
    case -34: return "CL_INVALID_CONTEXT";
    case -35: return "CL_INVALID_QUEUE_PROPERTIES";
    case -36: return "CL_INVALID_COMMAND_QUEUE";
    case -37: return "CL_INVALID_HOST_PTR";
    case -38: return "CL_INVALID_MEM_OBJECT";
    case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case -40: return "CL_INVALID_IMAGE_SIZE";
    case -41: return "CL_INVALID_SAMPLER";
    case -42: return "CL_INVALID_BINARY";
    case -43: return "CL_INVALID_BUILD_OPTIONS";
    case -44: return "CL_INVALID_PROGRAM";
    case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
    case -46: return "CL_INVALID_KERNEL_NAME";
    case -47: return "CL_INVALID_KERNEL_DEFINITION";
    case -48: return "CL_INVALID_KERNEL";
    case -49: return "CL_INVALID_ARG_INDEX";
    case -50: return "CL_INVALID_ARG_VALUE";
    case -51: return "CL_INVALID_ARG_SIZE";
    case -52: return "CL_INVALID_KERNEL_ARGS";
    case -53: return "CL_INVALID_WORK_DIMENSION";
    case -54: return "CL_INVALID_WORK_GROUP_SIZE";
    case -55: return "CL_INVALID_WORK_ITEM_SIZE";
    case -56: return "CL_INVALID_GLOBAL_OFFSET";
    case -57: return "CL_INVALID_EVENT_WAIT_LIST";
    case -58: return "CL_INVALID_EVENT";
    case -59: return "CL_INVALID_OPERATION";
    case -60: return "CL_INVALID_GL_OBJECT";
    case -61: return "CL_INVALID_BUFFER_SIZE";
    case -62: return "CL_INVALID_MIP_LEVEL";
    case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
    case -64: return "CL_INVALID_PROPERTY";
    case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
    case -66: return "CL_INVALID_COMPILER_OPTIONS";
    case -67: return "CL_INVALID_LINKER_OPTIONS";
    case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";
    case -69: return "CL_INVALID_PIPE_SIZE";
    case -70: return "CL_INVALID_DEVICE_QUEUE";
    case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
    case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
    case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
    case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
    case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
    case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
    case -1024: return "clBLAS: Functionality is not implemented";
    case -1023: return "clBLAS: Library is not initialized yet";
    case -1022: return "clBLAS: Matrix A is not a valid memory object";
    case -1021: return "clBLAS: Matrix B is not a valid memory object";
    case -1020: return "clBLAS: Matrix C is not a valid memory object";
    case -1019: return "clBLAS: Vector X is not a valid memory object";
    case -1018: return "clBLAS: Vector Y is not a valid memory object";
    case -1017: return "clBLAS: An input dimension (M:N:K) is invalid";
    case -1016: return "clBLAS: Leading dimension A must not be less than the "
                       "size of the first dimension";
    case -1015: return "clBLAS: Leading dimension B must not be less than the "
                       "size of the second dimension";
    case -1014: return "clBLAS: Leading dimension C must not be less than the "
                       "size of the third dimension";
    case -1013: return "clBLAS: The increment for a vector X must not be 0";
    case -1012: return "clBLAS: The increment for a vector Y must not be 0";
    case -1011: return "clBLAS: The memory object for Matrix A is too small";
    case -1010: return "clBLAS: The memory object for Matrix B is too small";
    case -1009: return "clBLAS: The memory object for Matrix C is too small";
    case -1008: return "clBLAS: The memory object for Vector X is too small";
    case -1007: return "clBLAS: The memory object for Vector Y is too small";
    default: return "Unknown OpenCL error";
    }
}

template <typename T>
static std::string kerToStr(const Mat & k)
{
    int width = k.cols - 1, depth = k.depth();
    const T * const data = k.ptr<T>();

    std::ostringstream stream;
    stream.precision(10);

    if (depth <= CV_8S)
    {
        for (int i = 0; i < width; ++i)
            stream << "DIG(" << (int)data[i] << ")";
        stream << "DIG(" << (int)data[width] << ")";
    }
    else if (depth == CV_32F)
    {
        stream.setf(std::ios_base::showpoint);
        for (int i = 0; i < width; ++i)
            stream << "DIG(" << data[i] << "f)";
        stream << "DIG(" << data[width] << "f)";
    }
    else
    {
        for (int i = 0; i < width; ++i)
            stream << "DIG(" << data[i] << ")";
        stream << "DIG(" << data[width] << ")";
    }

    return stream.str();
}

String kernelToStr(InputArray _kernel, int ddepth, const char * name)
{
    Mat kernel = _kernel.getMat().reshape(1, 1);

    int depth = kernel.depth();
    if (ddepth < 0)
        ddepth = depth;

    if (ddepth != depth)
        kernel.convertTo(kernel, ddepth);

    typedef std::string (* func_t)(const Mat &);
    static const func_t funcs[] = { kerToStr<uchar>, kerToStr<char>, kerToStr<ushort>, kerToStr<short>,
                                    kerToStr<int>, kerToStr<float>, kerToStr<double>, 0 };
    const func_t func = funcs[ddepth];
    CV_Assert(func != 0);

    return cv::format(" -D %s=%s", name ? name : "COEFF", func(kernel).c_str());
}

#define PROCESS_SRC(src) \
    do \
    { \
        if (!src.empty()) \
        { \
            CV_Assert(src.isMat() || src.isUMat()); \
            Size csize = src.size(); \
            int ctype = src.type(), ccn = CV_MAT_CN(ctype), cdepth = CV_MAT_DEPTH(ctype), \
                ckercn = vectorWidths[cdepth], cwidth = ccn * csize.width; \
            if (cwidth < ckercn || ckercn <= 0) \
                return 1; \
            cols.push_back(cwidth); \
            if (strat == OCL_VECTOR_OWN && ctype != ref_type) \
                return 1; \
            offsets.push_back(src.offset()); \
            steps.push_back(src.step()); \
            dividers.push_back(ckercn * CV_ELEM_SIZE1(ctype)); \
            kercns.push_back(ckercn); \
        } \
    } \
    while ((void)0, 0)

int predictOptimalVectorWidth(InputArray src1, InputArray src2, InputArray src3,
                              InputArray src4, InputArray src5, InputArray src6,
                              InputArray src7, InputArray src8, InputArray src9,
                              OclVectorStrategy strat)
{
    const ocl::Device & d = ocl::Device::getDefault();

    int vectorWidths[] = { d.preferredVectorWidthChar(), d.preferredVectorWidthChar(),
        d.preferredVectorWidthShort(), d.preferredVectorWidthShort(),
        d.preferredVectorWidthInt(), d.preferredVectorWidthFloat(),
        d.preferredVectorWidthDouble(), -1 };

    // if the device says don't use vectors
    if (vectorWidths[0] == 1)
    {
        // it's heuristic
        vectorWidths[CV_8U] = vectorWidths[CV_8S] = 4;
        vectorWidths[CV_16U] = vectorWidths[CV_16S] = 2;
        vectorWidths[CV_32S] = vectorWidths[CV_32F] = vectorWidths[CV_64F] = 1;
    }

    return checkOptimalVectorWidth(vectorWidths, src1, src2, src3, src4, src5, src6, src7, src8, src9, strat);
}

int checkOptimalVectorWidth(const int *vectorWidths,
                            InputArray src1, InputArray src2, InputArray src3,
                            InputArray src4, InputArray src5, InputArray src6,
                            InputArray src7, InputArray src8, InputArray src9,
                            OclVectorStrategy strat)
{
    CV_Assert(vectorWidths);

    int ref_type = src1.type();

    std::vector<size_t> offsets, steps, cols;
    std::vector<int> dividers, kercns;
    PROCESS_SRC(src1);
    PROCESS_SRC(src2);
    PROCESS_SRC(src3);
    PROCESS_SRC(src4);
    PROCESS_SRC(src5);
    PROCESS_SRC(src6);
    PROCESS_SRC(src7);
    PROCESS_SRC(src8);
    PROCESS_SRC(src9);

    size_t size = offsets.size();

    for (size_t i = 0; i < size; ++i)
        while (offsets[i] % dividers[i] != 0 || steps[i] % dividers[i] != 0 || cols[i] % kercns[i] != 0)
            dividers[i] >>= 1, kercns[i] >>= 1;

    // default strategy
    int kercn = *std::min_element(kercns.begin(), kercns.end());

    return kercn;
}

int predictOptimalVectorWidthMax(InputArray src1, InputArray src2, InputArray src3,
                                 InputArray src4, InputArray src5, InputArray src6,
                                 InputArray src7, InputArray src8, InputArray src9)
{
    return predictOptimalVectorWidth(src1, src2, src3, src4, src5, src6, src7, src8, src9, OCL_VECTOR_MAX);
}

#undef PROCESS_SRC


// TODO Make this as a method of OpenCL "BuildOptions" class
void buildOptionsAddMatrixDescription(String& buildOptions, const String& name, InputArray _m)
{
    if (!buildOptions.empty())
        buildOptions += " ";
    int type = _m.type(), depth = CV_MAT_DEPTH(type);
    buildOptions += format(
            "-D %s_T=%s -D %s_T1=%s -D %s_CN=%d -D %s_TSIZE=%d -D %s_T1SIZE=%d -D %s_DEPTH=%d",
            name.c_str(), ocl::typeToStr(type),
            name.c_str(), ocl::typeToStr(CV_MAKE_TYPE(depth, 1)),
            name.c_str(), (int)CV_MAT_CN(type),
            name.c_str(), (int)CV_ELEM_SIZE(type),
            name.c_str(), (int)CV_ELEM_SIZE1(type),
            name.c_str(), (int)depth
            );
}


struct Image2D::Impl
{
    Impl(const UMat &src, bool norm, bool alias)
    {
        handle = 0;
        refcount = 1;
        init(src, norm, alias);
    }

    ~Impl()
    {
        if (handle)
            clReleaseMemObject(handle);
    }

    static cl_image_format getImageFormat(int depth, int cn, bool norm)
    {
        cl_image_format format;
        static const int channelTypes[] = { CL_UNSIGNED_INT8, CL_SIGNED_INT8, CL_UNSIGNED_INT16,
                                       CL_SIGNED_INT16, CL_SIGNED_INT32, CL_FLOAT, -1, -1 };
        static const int channelTypesNorm[] = { CL_UNORM_INT8, CL_SNORM_INT8, CL_UNORM_INT16,
                                                CL_SNORM_INT16, -1, -1, -1, -1 };
        static const int channelOrders[] = { -1, CL_R, CL_RG, -1, CL_RGBA };

        int channelType = norm ? channelTypesNorm[depth] : channelTypes[depth];
        int channelOrder = channelOrders[cn];
        format.image_channel_data_type = (cl_channel_type)channelType;
        format.image_channel_order = (cl_channel_order)channelOrder;
        return format;
    }

    static bool isFormatSupported(cl_image_format format)
    {
        if (!haveOpenCL())
            CV_Error(Error::OpenCLApiCallError, "OpenCL runtime not found!");

        cl_context context = (cl_context)Context::getDefault().ptr();
        // Figure out how many formats are supported by this context.
        cl_uint numFormats = 0;
        cl_int err = clGetSupportedImageFormats(context, CL_MEM_READ_WRITE,
                                                CL_MEM_OBJECT_IMAGE2D, numFormats,
                                                NULL, &numFormats);
        AutoBuffer<cl_image_format> formats(numFormats);
        err = clGetSupportedImageFormats(context, CL_MEM_READ_WRITE,
                                         CL_MEM_OBJECT_IMAGE2D, numFormats,
                                         formats, NULL);
        CV_OclDbgAssert(err == CL_SUCCESS);
        for (cl_uint i = 0; i < numFormats; ++i)
        {
            if (!memcmp(&formats[i], &format, sizeof(format)))
            {
                return true;
            }
        }
        return false;
    }

    void init(const UMat &src, bool norm, bool alias)
    {
        if (!haveOpenCL())
            CV_Error(Error::OpenCLApiCallError, "OpenCL runtime not found!");

        CV_Assert(!src.empty());
        CV_Assert(ocl::Device::getDefault().imageSupport());

        int err, depth = src.depth(), cn = src.channels();
        CV_Assert(cn <= 4);
        cl_image_format format = getImageFormat(depth, cn, norm);

        if (!isFormatSupported(format))
            CV_Error(Error::OpenCLApiCallError, "Image format is not supported");

        if (alias && !src.handle(ACCESS_RW))
            CV_Error(Error::OpenCLApiCallError, "Incorrect UMat, handle is null");

        cl_context context = (cl_context)Context::getDefault().ptr();
        cl_command_queue queue = (cl_command_queue)Queue::getDefault().ptr();

#ifdef CL_VERSION_1_2
        // this enables backwards portability to
        // run on OpenCL 1.1 platform if library binaries are compiled with OpenCL 1.2 support
        const Device & d = ocl::Device::getDefault();
        int minor = d.deviceVersionMinor(), major = d.deviceVersionMajor();
        CV_Assert(!alias || canCreateAlias(src));
        if (1 < major || (1 == major && 2 <= minor))
        {
            cl_image_desc desc;
            desc.image_type       = CL_MEM_OBJECT_IMAGE2D;
            desc.image_width      = src.cols;
            desc.image_height     = src.rows;
            desc.image_depth      = 0;
            desc.image_array_size = 1;
            desc.image_row_pitch  = alias ? src.step[0] : 0;
            desc.image_slice_pitch = 0;
            desc.buffer           = alias ? (cl_mem)src.handle(ACCESS_RW) : 0;
            desc.num_mip_levels   = 0;
            desc.num_samples      = 0;
            handle = clCreateImage(context, CL_MEM_READ_WRITE, &format, &desc, NULL, &err);
        }
        else
#endif
        {
            CV_SUPPRESS_DEPRECATED_START
            CV_Assert(!alias);  // This is an OpenCL 1.2 extension
            handle = clCreateImage2D(context, CL_MEM_READ_WRITE, &format, src.cols, src.rows, 0, NULL, &err);
            CV_SUPPRESS_DEPRECATED_END
        }
        CV_OclDbgAssert(err == CL_SUCCESS);

        size_t origin[] = { 0, 0, 0 };
        size_t region[] = { static_cast<size_t>(src.cols), static_cast<size_t>(src.rows), 1 };

        cl_mem devData;
        if (!alias && !src.isContinuous())
        {
            devData = clCreateBuffer(context, CL_MEM_READ_ONLY, src.cols * src.rows * src.elemSize(), NULL, &err);
            CV_OclDbgAssert(err == CL_SUCCESS);

            const size_t roi[3] = {static_cast<size_t>(src.cols) * src.elemSize(), static_cast<size_t>(src.rows), 1};
            CV_Assert(clEnqueueCopyBufferRect(queue, (cl_mem)src.handle(ACCESS_READ), devData, origin, origin,
                roi, src.step, 0, src.cols * src.elemSize(), 0, 0, NULL, NULL) == CL_SUCCESS);
            CV_OclDbgAssert(clFlush(queue) == CL_SUCCESS);
        }
        else
        {
            devData = (cl_mem)src.handle(ACCESS_READ);
        }
        CV_Assert(devData != NULL);

        if (!alias)
        {
            CV_OclDbgAssert(clEnqueueCopyBufferToImage(queue, devData, handle, 0, origin, region, 0, NULL, 0) == CL_SUCCESS);
            if (!src.isContinuous())
            {
                CV_OclDbgAssert(clFlush(queue) == CL_SUCCESS);
                CV_OclDbgAssert(clReleaseMemObject(devData) == CL_SUCCESS);
            }
        }
    }

    IMPLEMENT_REFCOUNTABLE();

    cl_mem handle;
};

Image2D::Image2D()
{
    p = NULL;
}

Image2D::Image2D(const UMat &src, bool norm, bool alias)
{
    p = new Impl(src, norm, alias);
}

bool Image2D::canCreateAlias(const UMat &m)
{
    bool ret = false;
    const Device & d = ocl::Device::getDefault();
    if (d.imageFromBufferSupport() && !m.empty())
    {
        // This is the required pitch alignment in pixels
        uint pitchAlign = d.imagePitchAlignment();
        if (pitchAlign && !(m.step % (pitchAlign * m.elemSize())))
        {
            // We don't currently handle the case where the buffer was created
            // with CL_MEM_USE_HOST_PTR
            if (!m.u->tempUMat())
            {
                ret = true;
            }
        }
    }
    return ret;
}

bool Image2D::isFormatSupported(int depth, int cn, bool norm)
{
    cl_image_format format = Impl::getImageFormat(depth, cn, norm);

    return Impl::isFormatSupported(format);
}

Image2D::Image2D(const Image2D & i)
{
    p = i.p;
    if (p)
        p->addref();
}

Image2D & Image2D::operator = (const Image2D & i)
{
    if (i.p != p)
    {
        if (i.p)
            i.p->addref();
        if (p)
            p->release();
        p = i.p;
    }
    return *this;
}

Image2D::~Image2D()
{
    if (p)
        p->release();
}

void* Image2D::ptr() const
{
    return p ? p->handle : 0;
}

bool internal::isOpenCLForced()
{
    static bool initialized = false;
    static bool value = false;
    if (!initialized)
    {
        value = utils::getConfigurationParameterBool("OPENCV_OPENCL_FORCE", false);
        initialized = true;
    }
    return value;
}

bool internal::isPerformanceCheckBypassed()
{
    static bool initialized = false;
    static bool value = false;
    if (!initialized)
    {
        value = utils::getConfigurationParameterBool("OPENCV_OPENCL_PERF_CHECK_BYPASS", false);
        initialized = true;
    }
    return value;
}

bool internal::isCLBuffer(UMat& u)
{
    void* h = u.handle(ACCESS_RW);
    if (!h)
        return true;
    CV_DbgAssert(u.u->currAllocator == getOpenCLAllocator());
#if 1
    if ((u.u->allocatorFlags_ & 0xffff0000) != 0) // OpenCL SVM flags are stored here
        return false;
#else
    cl_mem_object_type type = 0;
    cl_int ret = clGetMemObjectInfo((cl_mem)h, CL_MEM_TYPE, sizeof(type), &type, NULL);
    if (ret != CL_SUCCESS || type != CL_MEM_OBJECT_BUFFER)
        return false;
#endif
    return true;
}

struct Timer::Impl
{
    const Queue queue;

    Impl(const Queue& q)
        : queue(q)
    {
    }

    ~Impl(){}

    void start()
    {
#ifdef HAVE_OPENCL
        clFinish((cl_command_queue)queue.ptr());
        timer.start();
#endif
    }

    void stop()
    {
#ifdef HAVE_OPENCL
        clFinish((cl_command_queue)queue.ptr());
        timer.stop();
#endif
    }

    uint64 durationNS() const
    {
#ifdef HAVE_OPENCL
        return (uint64)(timer.getTimeSec() * 1e9);
#else
        return 0;
#endif
    }

    TickMeter timer;
};

Timer::Timer(const Queue& q) : p(new Impl(q)) { }
Timer::~Timer() { delete p; }

void Timer::start()
{
    CV_Assert(p);
    p->start();
}

void Timer::stop()
{
    CV_Assert(p);
    p->stop();
}

uint64 Timer::durationNS() const
{
    CV_Assert(p);
    return p->durationNS();
}

}} // namespace
