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
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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

using namespace cv;
using namespace cv::cuda;

int cv::cuda::getCudaEnabledDeviceCount()
{
#ifndef HAVE_CUDA
    return 0;
#else
    int count;
    cudaError_t error = cudaGetDeviceCount(&count);

    if (error == cudaErrorInsufficientDriver)
        return -1;

    if (error == cudaErrorNoDevice)
        return 0;

    cudaSafeCall( error );
    return count;
#endif
}

void cv::cuda::setDevice(int device)
{
#ifndef HAVE_CUDA
    (void) device;
    throw_no_cuda();
#else
    cudaSafeCall( cudaSetDevice(device) );
#endif
}

int cv::cuda::getDevice()
{
#ifndef HAVE_CUDA
    throw_no_cuda();
    return 0;
#else
    int device;
    cudaSafeCall( cudaGetDevice(&device) );
    return device;
#endif
}

void cv::cuda::resetDevice()
{
#ifndef HAVE_CUDA
    throw_no_cuda();
#else
    cudaSafeCall( cudaDeviceReset() );
#endif
}

bool cv::cuda::deviceSupports(FeatureSet feature_set)
{
#ifndef HAVE_CUDA
    (void) feature_set;
    throw_no_cuda();
    return false;
#else
    static int versions[] =
    {
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
    };
    static const int cache_size = static_cast<int>(sizeof(versions) / sizeof(versions[0]));

    const int devId = getDevice();

    int version;

    if (devId < cache_size && versions[devId] >= 0)
    {
        version = versions[devId];
    }
    else
    {
        DeviceInfo dev(devId);
        version = dev.majorVersion() * 10 + dev.minorVersion();
        if (devId < cache_size)
            versions[devId] = version;
    }

    return TargetArchs::builtWith(feature_set) && (version >= feature_set);
#endif
}

////////////////////////////////////////////////////////////////////////
// TargetArchs

#ifdef HAVE_CUDA

namespace
{
    class CudaArch
    {
    public:
        CudaArch();

        bool builtWith(FeatureSet feature_set) const;
        bool hasPtx(int major, int minor) const;
        bool hasBin(int major, int minor) const;
        bool hasEqualOrLessPtx(int major, int minor) const;
        bool hasEqualOrGreaterPtx(int major, int minor) const;
        bool hasEqualOrGreaterBin(int major, int minor) const;

    private:
        static void fromStr(const char* set_as_str, std::vector<int>& arr);

        std::vector<int> bin;
        std::vector<int> ptx;
        std::vector<int> features;
    };

    const CudaArch cudaArch;

    CudaArch::CudaArch()
    {
        fromStr(CUDA_ARCH_BIN, bin);
        fromStr(CUDA_ARCH_PTX, ptx);
        fromStr(CUDA_ARCH_FEATURES, features);
    }

    bool CudaArch::builtWith(FeatureSet feature_set) const
    {
        return !features.empty() && (features.back() >= feature_set);
    }

    bool CudaArch::hasPtx(int major, int minor) const
    {
        return std::find(ptx.begin(), ptx.end(), major * 10 + minor) != ptx.end();
    }

    bool CudaArch::hasBin(int major, int minor) const
    {
        return std::find(bin.begin(), bin.end(), major * 10 + minor) != bin.end();
    }

    bool CudaArch::hasEqualOrLessPtx(int major, int minor) const
    {
        return !ptx.empty() && (ptx.front() <= major * 10 + minor);
    }

    bool CudaArch::hasEqualOrGreaterPtx(int major, int minor) const
    {
        return !ptx.empty() && (ptx.back() >= major * 10 + minor);
    }

    bool CudaArch::hasEqualOrGreaterBin(int major, int minor) const
    {
        return !bin.empty() && (bin.back() >= major * 10 + minor);
    }

    void CudaArch::fromStr(const char* set_as_str, std::vector<int>& arr)
    {
        arr.clear();

        const size_t len = strlen(set_as_str);

        size_t pos = 0;
        while (pos < len)
        {
            if (isspace(set_as_str[pos]))
            {
                ++pos;
            }
            else
            {
                int cur_value;
                int chars_read;
                int args_read = sscanf(set_as_str + pos, "%d%n", &cur_value, &chars_read);
                CV_Assert( args_read == 1 );

                arr.push_back(cur_value);
                pos += chars_read;
            }
        }

        std::sort(arr.begin(), arr.end());
    }
}

#endif

bool cv::cuda::TargetArchs::builtWith(cv::cuda::FeatureSet feature_set)
{
#ifndef HAVE_CUDA
    (void) feature_set;
    throw_no_cuda();
    return false;
#else
    return cudaArch.builtWith(feature_set);
#endif
}

bool cv::cuda::TargetArchs::hasPtx(int major, int minor)
{
#ifndef HAVE_CUDA
    (void) major;
    (void) minor;
    throw_no_cuda();
    return false;
#else
    return cudaArch.hasPtx(major, minor);
#endif
}

bool cv::cuda::TargetArchs::hasBin(int major, int minor)
{
#ifndef HAVE_CUDA
    (void) major;
    (void) minor;
    throw_no_cuda();
    return false;
#else
    return cudaArch.hasBin(major, minor);
#endif
}

bool cv::cuda::TargetArchs::hasEqualOrLessPtx(int major, int minor)
{
#ifndef HAVE_CUDA
    (void) major;
    (void) minor;
    throw_no_cuda();
    return false;
#else
    return cudaArch.hasEqualOrLessPtx(major, minor);
#endif
}

bool cv::cuda::TargetArchs::hasEqualOrGreaterPtx(int major, int minor)
{
#ifndef HAVE_CUDA
    (void) major;
    (void) minor;
    throw_no_cuda();
    return false;
#else
    return cudaArch.hasEqualOrGreaterPtx(major, minor);
#endif
}

bool cv::cuda::TargetArchs::hasEqualOrGreaterBin(int major, int minor)
{
#ifndef HAVE_CUDA
    (void) major;
    (void) minor;
    throw_no_cuda();
    return false;
#else
    return cudaArch.hasEqualOrGreaterBin(major, minor);
#endif
}

////////////////////////////////////////////////////////////////////////
// DeviceInfo

#ifdef HAVE_CUDA

namespace
{
    class DeviceProps
    {
    public:
        DeviceProps();

        const cudaDeviceProp* get(int devID) const;

    private:
        std::vector<cudaDeviceProp> props_;
    };

    DeviceProps::DeviceProps()
    {
        int count = getCudaEnabledDeviceCount();

        if (count > 0)
        {
            props_.resize(count);

            for (int devID = 0; devID < count; ++devID)
            {
                cudaSafeCall( cudaGetDeviceProperties(&props_[devID], devID) );
            }
        }
    }

    const cudaDeviceProp* DeviceProps::get(int devID) const
    {
        CV_Assert( static_cast<size_t>(devID) < props_.size() );

        return &props_[devID];
    }

    DeviceProps& deviceProps()
    {
        static DeviceProps props;
        return props;
    }
}

#endif

const char* cv::cuda::DeviceInfo::name() const
{
#ifndef HAVE_CUDA
    throw_no_cuda();
    return "";
#else
    return deviceProps().get(device_id_)->name;
#endif
}

size_t cv::cuda::DeviceInfo::totalGlobalMem() const
{
#ifndef HAVE_CUDA
    throw_no_cuda();
    return 0;
#else
    return deviceProps().get(device_id_)->totalGlobalMem;
#endif
}

size_t cv::cuda::DeviceInfo::sharedMemPerBlock() const
{
#ifndef HAVE_CUDA
    throw_no_cuda();
    return 0;
#else
    return deviceProps().get(device_id_)->sharedMemPerBlock;
#endif
}

int cv::cuda::DeviceInfo::regsPerBlock() const
{
#ifndef HAVE_CUDA
    throw_no_cuda();
    return 0;
#else
    return deviceProps().get(device_id_)->regsPerBlock;
#endif
}

int cv::cuda::DeviceInfo::warpSize() const
{
#ifndef HAVE_CUDA
    throw_no_cuda();
    return 0;
#else
    return deviceProps().get(device_id_)->warpSize;
#endif
}

size_t cv::cuda::DeviceInfo::memPitch() const
{
#ifndef HAVE_CUDA
    throw_no_cuda();
    return 0;
#else
    return deviceProps().get(device_id_)->memPitch;
#endif
}

int cv::cuda::DeviceInfo::maxThreadsPerBlock() const
{
#ifndef HAVE_CUDA
    throw_no_cuda();
    return 0;
#else
    return deviceProps().get(device_id_)->maxThreadsPerBlock;
#endif
}

Vec3i cv::cuda::DeviceInfo::maxThreadsDim() const
{
#ifndef HAVE_CUDA
    throw_no_cuda();
    return Vec3i();
#else
    return Vec3i(deviceProps().get(device_id_)->maxThreadsDim);
#endif
}

Vec3i cv::cuda::DeviceInfo::maxGridSize() const
{
#ifndef HAVE_CUDA
    throw_no_cuda();
    return Vec3i();
#else
    return Vec3i(deviceProps().get(device_id_)->maxGridSize);
#endif
}

int cv::cuda::DeviceInfo::clockRate() const
{
#ifndef HAVE_CUDA
    throw_no_cuda();
    return 0;
#else
    return deviceProps().get(device_id_)->clockRate;
#endif
}

size_t cv::cuda::DeviceInfo::totalConstMem() const
{
#ifndef HAVE_CUDA
    throw_no_cuda();
    return 0;
#else
    return deviceProps().get(device_id_)->totalConstMem;
#endif
}

int cv::cuda::DeviceInfo::majorVersion() const
{
#ifndef HAVE_CUDA
    throw_no_cuda();
    return 0;
#else
    return deviceProps().get(device_id_)->major;
#endif
}

int cv::cuda::DeviceInfo::minorVersion() const
{
#ifndef HAVE_CUDA
    throw_no_cuda();
    return 0;
#else
    return deviceProps().get(device_id_)->minor;
#endif
}

size_t cv::cuda::DeviceInfo::textureAlignment() const
{
#ifndef HAVE_CUDA
    throw_no_cuda();
    return 0;
#else
    return deviceProps().get(device_id_)->textureAlignment;
#endif
}

size_t cv::cuda::DeviceInfo::texturePitchAlignment() const
{
#ifndef HAVE_CUDA
    throw_no_cuda();
    return 0;
#else
    return deviceProps().get(device_id_)->texturePitchAlignment;
#endif
}

int cv::cuda::DeviceInfo::multiProcessorCount() const
{
#ifndef HAVE_CUDA
    throw_no_cuda();
    return 0;
#else
    return deviceProps().get(device_id_)->multiProcessorCount;
#endif
}

bool cv::cuda::DeviceInfo::kernelExecTimeoutEnabled() const
{
#ifndef HAVE_CUDA
    throw_no_cuda();
    return false;
#else
    return deviceProps().get(device_id_)->kernelExecTimeoutEnabled != 0;
#endif
}

bool cv::cuda::DeviceInfo::integrated() const
{
#ifndef HAVE_CUDA
    throw_no_cuda();
    return false;
#else
    return deviceProps().get(device_id_)->integrated != 0;
#endif
}

bool cv::cuda::DeviceInfo::canMapHostMemory() const
{
#ifndef HAVE_CUDA
    throw_no_cuda();
    return false;
#else
    return deviceProps().get(device_id_)->canMapHostMemory != 0;
#endif
}

DeviceInfo::ComputeMode cv::cuda::DeviceInfo::computeMode() const
{
#ifndef HAVE_CUDA
    throw_no_cuda();
    return ComputeModeDefault;
#else
    static const ComputeMode tbl[] =
    {
        ComputeModeDefault,
        ComputeModeExclusive,
        ComputeModeProhibited,
        ComputeModeExclusiveProcess
    };

    return tbl[deviceProps().get(device_id_)->computeMode];
#endif
}

int cv::cuda::DeviceInfo::maxTexture1D() const
{
#ifndef HAVE_CUDA
    throw_no_cuda();
    return 0;
#else
    return deviceProps().get(device_id_)->maxTexture1D;
#endif
}

int cv::cuda::DeviceInfo::maxTexture1DMipmap() const
{
#ifndef HAVE_CUDA
    throw_no_cuda();
    return 0;
#else
    #if CUDA_VERSION >= 5000
        return deviceProps().get(device_id_)->maxTexture1DMipmap;
    #else
        CV_Error(Error::StsNotImplemented, "This function requires CUDA 5.0");
        return 0;
    #endif
#endif
}

int cv::cuda::DeviceInfo::maxTexture1DLinear() const
{
#ifndef HAVE_CUDA
    throw_no_cuda();
    return 0;
#else
    return deviceProps().get(device_id_)->maxTexture1DLinear;
#endif
}

Vec2i cv::cuda::DeviceInfo::maxTexture2D() const
{
#ifndef HAVE_CUDA
    throw_no_cuda();
    return Vec2i();
#else
    return Vec2i(deviceProps().get(device_id_)->maxTexture2D);
#endif
}

Vec2i cv::cuda::DeviceInfo::maxTexture2DMipmap() const
{
#ifndef HAVE_CUDA
    throw_no_cuda();
    return Vec2i();
#else
    #if CUDA_VERSION >= 5000
        return Vec2i(deviceProps().get(device_id_)->maxTexture2DMipmap);
    #else
        CV_Error(Error::StsNotImplemented, "This function requires CUDA 5.0");
        return Vec2i();
    #endif
#endif
}

Vec3i cv::cuda::DeviceInfo::maxTexture2DLinear() const
{
#ifndef HAVE_CUDA
    throw_no_cuda();
    return Vec3i();
#else
    return Vec3i(deviceProps().get(device_id_)->maxTexture2DLinear);
#endif
}

Vec2i cv::cuda::DeviceInfo::maxTexture2DGather() const
{
#ifndef HAVE_CUDA
    throw_no_cuda();
    return Vec2i();
#else
    return Vec2i(deviceProps().get(device_id_)->maxTexture2DGather);
#endif
}

Vec3i cv::cuda::DeviceInfo::maxTexture3D() const
{
#ifndef HAVE_CUDA
    throw_no_cuda();
    return Vec3i();
#else
    return Vec3i(deviceProps().get(device_id_)->maxTexture3D);
#endif
}

int cv::cuda::DeviceInfo::maxTextureCubemap() const
{
#ifndef HAVE_CUDA
    throw_no_cuda();
    return 0;
#else
    return deviceProps().get(device_id_)->maxTextureCubemap;
#endif
}

Vec2i cv::cuda::DeviceInfo::maxTexture1DLayered() const
{
#ifndef HAVE_CUDA
    throw_no_cuda();
    return Vec2i();
#else
    return Vec2i(deviceProps().get(device_id_)->maxTexture1DLayered);
#endif
}

Vec3i cv::cuda::DeviceInfo::maxTexture2DLayered() const
{
#ifndef HAVE_CUDA
    throw_no_cuda();
    return Vec3i();
#else
    return Vec3i(deviceProps().get(device_id_)->maxTexture2DLayered);
#endif
}

Vec2i cv::cuda::DeviceInfo::maxTextureCubemapLayered() const
{
#ifndef HAVE_CUDA
    throw_no_cuda();
    return Vec2i();
#else
    return Vec2i(deviceProps().get(device_id_)->maxTextureCubemapLayered);
#endif
}

int cv::cuda::DeviceInfo::maxSurface1D() const
{
#ifndef HAVE_CUDA
    throw_no_cuda();
    return 0;
#else
    return deviceProps().get(device_id_)->maxSurface1D;
#endif
}

Vec2i cv::cuda::DeviceInfo::maxSurface2D() const
{
#ifndef HAVE_CUDA
    throw_no_cuda();
    return Vec2i();
#else
    return Vec2i(deviceProps().get(device_id_)->maxSurface2D);
#endif
}

Vec3i cv::cuda::DeviceInfo::maxSurface3D() const
{
#ifndef HAVE_CUDA
    throw_no_cuda();
    return Vec3i();
#else
    return Vec3i(deviceProps().get(device_id_)->maxSurface3D);
#endif
}

Vec2i cv::cuda::DeviceInfo::maxSurface1DLayered() const
{
#ifndef HAVE_CUDA
    throw_no_cuda();
    return Vec2i();
#else
    return Vec2i(deviceProps().get(device_id_)->maxSurface1DLayered);
#endif
}

Vec3i cv::cuda::DeviceInfo::maxSurface2DLayered() const
{
#ifndef HAVE_CUDA
    throw_no_cuda();
    return Vec3i();
#else
    return Vec3i(deviceProps().get(device_id_)->maxSurface2DLayered);
#endif
}

int cv::cuda::DeviceInfo::maxSurfaceCubemap() const
{
#ifndef HAVE_CUDA
    throw_no_cuda();
    return 0;
#else
    return deviceProps().get(device_id_)->maxSurfaceCubemap;
#endif
}

Vec2i cv::cuda::DeviceInfo::maxSurfaceCubemapLayered() const
{
#ifndef HAVE_CUDA
    throw_no_cuda();
    return Vec2i();
#else
    return Vec2i(deviceProps().get(device_id_)->maxSurfaceCubemapLayered);
#endif
}

size_t cv::cuda::DeviceInfo::surfaceAlignment() const
{
#ifndef HAVE_CUDA
    throw_no_cuda();
    return 0;
#else
    return deviceProps().get(device_id_)->surfaceAlignment;
#endif
}

bool cv::cuda::DeviceInfo::concurrentKernels() const
{
#ifndef HAVE_CUDA
    throw_no_cuda();
    return false;
#else
    return deviceProps().get(device_id_)->concurrentKernels != 0;
#endif
}

bool cv::cuda::DeviceInfo::ECCEnabled() const
{
#ifndef HAVE_CUDA
    throw_no_cuda();
    return false;
#else
    return deviceProps().get(device_id_)->ECCEnabled != 0;
#endif
}

int cv::cuda::DeviceInfo::pciBusID() const
{
#ifndef HAVE_CUDA
    throw_no_cuda();
    return 0;
#else
    return deviceProps().get(device_id_)->pciBusID;
#endif
}

int cv::cuda::DeviceInfo::pciDeviceID() const
{
#ifndef HAVE_CUDA
    throw_no_cuda();
    return 0;
#else
    return deviceProps().get(device_id_)->pciDeviceID;
#endif
}

int cv::cuda::DeviceInfo::pciDomainID() const
{
#ifndef HAVE_CUDA
    throw_no_cuda();
    return 0;
#else
    return deviceProps().get(device_id_)->pciDomainID;
#endif
}

bool cv::cuda::DeviceInfo::tccDriver() const
{
#ifndef HAVE_CUDA
    throw_no_cuda();
    return false;
#else
    return deviceProps().get(device_id_)->tccDriver != 0;
#endif
}

int cv::cuda::DeviceInfo::asyncEngineCount() const
{
#ifndef HAVE_CUDA
    throw_no_cuda();
    return 0;
#else
    return deviceProps().get(device_id_)->asyncEngineCount;
#endif
}

bool cv::cuda::DeviceInfo::unifiedAddressing() const
{
#ifndef HAVE_CUDA
    throw_no_cuda();
    return false;
#else
    return deviceProps().get(device_id_)->unifiedAddressing != 0;
#endif
}

int cv::cuda::DeviceInfo::memoryClockRate() const
{
#ifndef HAVE_CUDA
    throw_no_cuda();
    return 0;
#else
    return deviceProps().get(device_id_)->memoryClockRate;
#endif
}

int cv::cuda::DeviceInfo::memoryBusWidth() const
{
#ifndef HAVE_CUDA
    throw_no_cuda();
    return 0;
#else
    return deviceProps().get(device_id_)->memoryBusWidth;
#endif
}

int cv::cuda::DeviceInfo::l2CacheSize() const
{
#ifndef HAVE_CUDA
    throw_no_cuda();
    return 0;
#else
    return deviceProps().get(device_id_)->l2CacheSize;
#endif
}

int cv::cuda::DeviceInfo::maxThreadsPerMultiProcessor() const
{
#ifndef HAVE_CUDA
    throw_no_cuda();
    return 0;
#else
    return deviceProps().get(device_id_)->maxThreadsPerMultiProcessor;
#endif
}

void cv::cuda::DeviceInfo::queryMemory(size_t& _totalMemory, size_t& _freeMemory) const
{
#ifndef HAVE_CUDA
    (void) _totalMemory;
    (void) _freeMemory;
    throw_no_cuda();
#else
    int prevDeviceID = getDevice();
    if (prevDeviceID != device_id_)
        setDevice(device_id_);

    cudaSafeCall( cudaMemGetInfo(&_freeMemory, &_totalMemory) );

    if (prevDeviceID != device_id_)
        setDevice(prevDeviceID);
#endif
}

bool cv::cuda::DeviceInfo::isCompatible() const
{
#ifndef HAVE_CUDA
    throw_no_cuda();
    return false;
#else
    // Check PTX compatibility
    if (TargetArchs::hasEqualOrLessPtx(majorVersion(), minorVersion()))
        return true;

    // Check BIN compatibility
    for (int i = minorVersion(); i >= 0; --i)
        if (TargetArchs::hasBin(majorVersion(), i))
            return true;

    return false;
#endif
}

////////////////////////////////////////////////////////////////////////
// print info

#ifdef HAVE_CUDA

namespace
{
    int convertSMVer2Cores(int major, int minor)
    {
        // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
        typedef struct {
            int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
            int Cores;
        } SMtoCores;

        SMtoCores gpuArchCoresPerSM[] =  { { 0x10,  8 }, { 0x11,  8 }, { 0x12,  8 }, { 0x13,  8 }, { 0x20, 32 }, { 0x21, 48 }, {0x30, 192}, {0x35, 192}, { -1, -1 }  };

        int index = 0;
        while (gpuArchCoresPerSM[index].SM != -1)
        {
            if (gpuArchCoresPerSM[index].SM == ((major << 4) + minor) )
                return gpuArchCoresPerSM[index].Cores;
            index++;
        }

        return -1;
    }
}

#endif

void cv::cuda::printCudaDeviceInfo(int device)
{
#ifndef HAVE_CUDA
    (void) device;
    throw_no_cuda();
#else
    int count = getCudaEnabledDeviceCount();
    bool valid = (device >= 0) && (device < count);

    int beg = valid ? device   : 0;
    int end = valid ? device+1 : count;

    printf("*** CUDA Device Query (Runtime API) version (CUDART static linking) *** \n\n");
    printf("Device count: %d\n", count);

    int driverVersion = 0, runtimeVersion = 0;
    cudaSafeCall( cudaDriverGetVersion(&driverVersion) );
    cudaSafeCall( cudaRuntimeGetVersion(&runtimeVersion) );

    const char *computeMode[] = {
        "Default (multiple host threads can use ::cudaSetDevice() with device simultaneously)",
        "Exclusive (only one host thread in one process is able to use ::cudaSetDevice() with this device)",
        "Prohibited (no host thread can use ::cudaSetDevice() with this device)",
        "Exclusive Process (many threads in one process is able to use ::cudaSetDevice() with this device)",
        "Unknown",
        NULL
    };

    for(int dev = beg; dev < end; ++dev)
    {
        cudaDeviceProp prop;
        cudaSafeCall( cudaGetDeviceProperties(&prop, dev) );

        printf("\nDevice %d: \"%s\"\n", dev, prop.name);
        printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n", driverVersion/1000, driverVersion%100, runtimeVersion/1000, runtimeVersion%100);
        printf("  CUDA Capability Major/Minor version number:    %d.%d\n", prop.major, prop.minor);
        printf("  Total amount of global memory:                 %.0f MBytes (%llu bytes)\n", (float)prop.totalGlobalMem/1048576.0f, (unsigned long long) prop.totalGlobalMem);

        int cores = convertSMVer2Cores(prop.major, prop.minor);
        if (cores > 0)
            printf("  (%2d) Multiprocessors x (%2d) CUDA Cores/MP:     %d CUDA Cores\n", prop.multiProcessorCount, cores, cores * prop.multiProcessorCount);

        printf("  GPU Clock Speed:                               %.2f GHz\n", prop.clockRate * 1e-6f);

        printf("  Max Texture Dimension Size (x,y,z)             1D=(%d), 2D=(%d,%d), 3D=(%d,%d,%d)\n",
            prop.maxTexture1D, prop.maxTexture2D[0], prop.maxTexture2D[1],
            prop.maxTexture3D[0], prop.maxTexture3D[1], prop.maxTexture3D[2]);
        printf("  Max Layered Texture Size (dim) x layers        1D=(%d) x %d, 2D=(%d,%d) x %d\n",
            prop.maxTexture1DLayered[0], prop.maxTexture1DLayered[1],
            prop.maxTexture2DLayered[0], prop.maxTexture2DLayered[1], prop.maxTexture2DLayered[2]);

        printf("  Total amount of constant memory:               %u bytes\n", (int)prop.totalConstMem);
        printf("  Total amount of shared memory per block:       %u bytes\n", (int)prop.sharedMemPerBlock);
        printf("  Total number of registers available per block: %d\n", prop.regsPerBlock);
        printf("  Warp size:                                     %d\n", prop.warpSize);
        printf("  Maximum number of threads per block:           %d\n", prop.maxThreadsPerBlock);
        printf("  Maximum sizes of each dimension of a block:    %d x %d x %d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  Maximum sizes of each dimension of a grid:     %d x %d x %d\n", prop.maxGridSize[0], prop.maxGridSize[1],  prop.maxGridSize[2]);
        printf("  Maximum memory pitch:                          %u bytes\n", (int)prop.memPitch);
        printf("  Texture alignment:                             %u bytes\n", (int)prop.textureAlignment);

        printf("  Concurrent copy and execution:                 %s with %d copy engine(s)\n", (prop.deviceOverlap ? "Yes" : "No"), prop.asyncEngineCount);
        printf("  Run time limit on kernels:                     %s\n", prop.kernelExecTimeoutEnabled ? "Yes" : "No");
        printf("  Integrated GPU sharing Host Memory:            %s\n", prop.integrated ? "Yes" : "No");
        printf("  Support host page-locked memory mapping:       %s\n", prop.canMapHostMemory ? "Yes" : "No");

        printf("  Concurrent kernel execution:                   %s\n", prop.concurrentKernels ? "Yes" : "No");
        printf("  Alignment requirement for Surfaces:            %s\n", prop.surfaceAlignment ? "Yes" : "No");
        printf("  Device has ECC support enabled:                %s\n", prop.ECCEnabled ? "Yes" : "No");
        printf("  Device is using TCC driver mode:               %s\n", prop.tccDriver ? "Yes" : "No");
        printf("  Device supports Unified Addressing (UVA):      %s\n", prop.unifiedAddressing ? "Yes" : "No");
        printf("  Device PCI Bus ID / PCI location ID:           %d / %d\n", prop.pciBusID, prop.pciDeviceID );
        printf("  Compute Mode:\n");
        printf("      %s \n", computeMode[prop.computeMode]);
    }

    printf("\n");
    printf("deviceQuery, CUDA Driver = CUDART");
    printf(", CUDA Driver Version  = %d.%d", driverVersion / 1000, driverVersion % 100);
    printf(", CUDA Runtime Version = %d.%d", runtimeVersion/1000, runtimeVersion%100);
    printf(", NumDevs = %d\n\n", count);

    fflush(stdout);
#endif
}

void cv::cuda::printShortCudaDeviceInfo(int device)
{
#ifndef HAVE_CUDA
    (void) device;
    throw_no_cuda();
#else
    int count = getCudaEnabledDeviceCount();
    bool valid = (device >= 0) && (device < count);

    int beg = valid ? device   : 0;
    int end = valid ? device+1 : count;

    int driverVersion = 0, runtimeVersion = 0;
    cudaSafeCall( cudaDriverGetVersion(&driverVersion) );
    cudaSafeCall( cudaRuntimeGetVersion(&runtimeVersion) );

    for(int dev = beg; dev < end; ++dev)
    {
        cudaDeviceProp prop;
        cudaSafeCall( cudaGetDeviceProperties(&prop, dev) );

        const char *arch_str = prop.major < 2 ? " (not Fermi)" : "";
        printf("Device %d:  \"%s\"  %.0fMb", dev, prop.name, (float)prop.totalGlobalMem/1048576.0f);
        printf(", sm_%d%d%s", prop.major, prop.minor, arch_str);

        int cores = convertSMVer2Cores(prop.major, prop.minor);
        if (cores > 0)
            printf(", %d cores", cores * prop.multiProcessorCount);

        printf(", Driver/Runtime ver.%d.%d/%d.%d\n", driverVersion/1000, driverVersion%100, runtimeVersion/1000, runtimeVersion%100);
    }

    fflush(stdout);
#endif
}

////////////////////////////////////////////////////////////////////////
// Error handling

#ifdef HAVE_CUDA

namespace
{
    #define error_entry(entry)  { entry, #entry }

    struct ErrorEntry
    {
        int code;
        const char* str;
    };

    struct ErrorEntryComparer
    {
        int code;
        ErrorEntryComparer(int code_) : code(code_) {}
        bool operator()(const ErrorEntry& e) const { return e.code == code; }
    };

    const ErrorEntry npp_errors [] =
    {
    #if defined (_MSC_VER)
        error_entry( NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY ),
    #endif

    #if NPP_VERSION < 5500
        error_entry( NPP_BAD_ARG_ERROR ),
        error_entry( NPP_COEFF_ERROR ),
        error_entry( NPP_RECT_ERROR ),
        error_entry( NPP_QUAD_ERROR ),
        error_entry( NPP_MEMFREE_ERR ),
        error_entry( NPP_MEMSET_ERR ),
        error_entry( NPP_MEM_ALLOC_ERR ),
        error_entry( NPP_HISTO_NUMBER_OF_LEVELS_ERROR ),
        error_entry( NPP_MIRROR_FLIP_ERR ),
        error_entry( NPP_INVALID_INPUT ),
        error_entry( NPP_POINTER_ERROR ),
        error_entry( NPP_WARNING ),
        error_entry( NPP_ODD_ROI_WARNING ),
    #else
        error_entry( NPP_INVALID_HOST_POINTER_ERROR ),
        error_entry( NPP_INVALID_DEVICE_POINTER_ERROR ),
        error_entry( NPP_LUT_PALETTE_BITSIZE_ERROR ),
        error_entry( NPP_ZC_MODE_NOT_SUPPORTED_ERROR ),
        error_entry( NPP_MEMFREE_ERROR ),
        error_entry( NPP_MEMSET_ERROR ),
        error_entry( NPP_QUALITY_INDEX_ERROR ),
        error_entry( NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR ),
        error_entry( NPP_CHANNEL_ORDER_ERROR ),
        error_entry( NPP_ZERO_MASK_VALUE_ERROR ),
        error_entry( NPP_QUADRANGLE_ERROR ),
        error_entry( NPP_RECTANGLE_ERROR ),
        error_entry( NPP_COEFFICIENT_ERROR ),
        error_entry( NPP_NUMBER_OF_CHANNELS_ERROR ),
        error_entry( NPP_COI_ERROR ),
        error_entry( NPP_DIVISOR_ERROR ),
        error_entry( NPP_CHANNEL_ERROR ),
        error_entry( NPP_STRIDE_ERROR ),
        error_entry( NPP_ANCHOR_ERROR ),
        error_entry( NPP_MASK_SIZE_ERROR ),
        error_entry( NPP_MIRROR_FLIP_ERROR ),
        error_entry( NPP_MOMENT_00_ZERO_ERROR ),
        error_entry( NPP_THRESHOLD_NEGATIVE_LEVEL_ERROR ),
        error_entry( NPP_THRESHOLD_ERROR ),
        error_entry( NPP_CONTEXT_MATCH_ERROR ),
        error_entry( NPP_FFT_FLAG_ERROR ),
        error_entry( NPP_FFT_ORDER_ERROR ),
        error_entry( NPP_SCALE_RANGE_ERROR ),
        error_entry( NPP_DATA_TYPE_ERROR ),
        error_entry( NPP_OUT_OFF_RANGE_ERROR ),
        error_entry( NPP_DIVIDE_BY_ZERO_ERROR ),
        error_entry( NPP_MEMORY_ALLOCATION_ERR ),
        error_entry( NPP_RANGE_ERROR ),
        error_entry( NPP_BAD_ARGUMENT_ERROR ),
        error_entry( NPP_NO_MEMORY_ERROR ),
        error_entry( NPP_ERROR_RESERVED ),
        error_entry( NPP_NO_OPERATION_WARNING ),
        error_entry( NPP_DIVIDE_BY_ZERO_WARNING ),
        error_entry( NPP_WRONG_INTERSECTION_ROI_WARNING ),
    #endif

        error_entry( NPP_NOT_SUPPORTED_MODE_ERROR ),
        error_entry( NPP_ROUND_MODE_NOT_SUPPORTED_ERROR ),
        error_entry( NPP_RESIZE_NO_OPERATION_ERROR ),
        error_entry( NPP_LUT_NUMBER_OF_LEVELS_ERROR ),
        error_entry( NPP_TEXTURE_BIND_ERROR ),
        error_entry( NPP_WRONG_INTERSECTION_ROI_ERROR ),
        error_entry( NPP_NOT_EVEN_STEP_ERROR ),
        error_entry( NPP_INTERPOLATION_ERROR ),
        error_entry( NPP_RESIZE_FACTOR_ERROR ),
        error_entry( NPP_HAAR_CLASSIFIER_PIXEL_MATCH_ERROR ),
        error_entry( NPP_MEMCPY_ERROR ),
        error_entry( NPP_ALIGNMENT_ERROR ),
        error_entry( NPP_STEP_ERROR ),
        error_entry( NPP_SIZE_ERROR ),
        error_entry( NPP_NULL_POINTER_ERROR ),
        error_entry( NPP_CUDA_KERNEL_EXECUTION_ERROR ),
        error_entry( NPP_NOT_IMPLEMENTED_ERROR ),
        error_entry( NPP_ERROR ),
        error_entry( NPP_NO_ERROR ),
        error_entry( NPP_SUCCESS ),
        error_entry( NPP_WRONG_INTERSECTION_QUAD_WARNING ),
        error_entry( NPP_MISALIGNED_DST_ROI_WARNING ),
        error_entry( NPP_AFFINE_QUAD_INCORRECT_WARNING ),
        error_entry( NPP_DOUBLE_SIZE_WARNING )
    };

    const size_t npp_error_num = sizeof(npp_errors) / sizeof(npp_errors[0]);

    const ErrorEntry cu_errors [] =
    {
        error_entry( CUDA_SUCCESS                              ),
        error_entry( CUDA_ERROR_INVALID_VALUE                  ),
        error_entry( CUDA_ERROR_OUT_OF_MEMORY                  ),
        error_entry( CUDA_ERROR_NOT_INITIALIZED                ),
        error_entry( CUDA_ERROR_DEINITIALIZED                  ),
        error_entry( CUDA_ERROR_PROFILER_DISABLED              ),
        error_entry( CUDA_ERROR_PROFILER_NOT_INITIALIZED       ),
        error_entry( CUDA_ERROR_PROFILER_ALREADY_STARTED       ),
        error_entry( CUDA_ERROR_PROFILER_ALREADY_STOPPED       ),
        error_entry( CUDA_ERROR_NO_DEVICE                      ),
        error_entry( CUDA_ERROR_INVALID_DEVICE                 ),
        error_entry( CUDA_ERROR_INVALID_IMAGE                  ),
        error_entry( CUDA_ERROR_INVALID_CONTEXT                ),
        error_entry( CUDA_ERROR_CONTEXT_ALREADY_CURRENT        ),
        error_entry( CUDA_ERROR_MAP_FAILED                     ),
        error_entry( CUDA_ERROR_UNMAP_FAILED                   ),
        error_entry( CUDA_ERROR_ARRAY_IS_MAPPED                ),
        error_entry( CUDA_ERROR_ALREADY_MAPPED                 ),
        error_entry( CUDA_ERROR_NO_BINARY_FOR_GPU              ),
        error_entry( CUDA_ERROR_ALREADY_ACQUIRED               ),
        error_entry( CUDA_ERROR_NOT_MAPPED                     ),
        error_entry( CUDA_ERROR_NOT_MAPPED_AS_ARRAY            ),
        error_entry( CUDA_ERROR_NOT_MAPPED_AS_POINTER          ),
        error_entry( CUDA_ERROR_ECC_UNCORRECTABLE              ),
        error_entry( CUDA_ERROR_UNSUPPORTED_LIMIT              ),
        error_entry( CUDA_ERROR_CONTEXT_ALREADY_IN_USE         ),
        error_entry( CUDA_ERROR_INVALID_SOURCE                 ),
        error_entry( CUDA_ERROR_FILE_NOT_FOUND                 ),
        error_entry( CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND ),
        error_entry( CUDA_ERROR_SHARED_OBJECT_INIT_FAILED      ),
        error_entry( CUDA_ERROR_OPERATING_SYSTEM               ),
        error_entry( CUDA_ERROR_INVALID_HANDLE                 ),
        error_entry( CUDA_ERROR_NOT_FOUND                      ),
        error_entry( CUDA_ERROR_NOT_READY                      ),
        error_entry( CUDA_ERROR_LAUNCH_FAILED                  ),
        error_entry( CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES        ),
        error_entry( CUDA_ERROR_LAUNCH_TIMEOUT                 ),
        error_entry( CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING  ),
        error_entry( CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED    ),
        error_entry( CUDA_ERROR_PEER_ACCESS_NOT_ENABLED        ),
        error_entry( CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE         ),
        error_entry( CUDA_ERROR_CONTEXT_IS_DESTROYED           ),
        error_entry( CUDA_ERROR_ASSERT                         ),
        error_entry( CUDA_ERROR_TOO_MANY_PEERS                 ),
        error_entry( CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED ),
        error_entry( CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED     ),
        error_entry( CUDA_ERROR_UNKNOWN                        )
    };

    const size_t cu_errors_num = sizeof(cu_errors) / sizeof(cu_errors[0]);

    cv::String getErrorString(int code, const ErrorEntry* errors, size_t n)
    {
        size_t idx = std::find_if(errors, errors + n, ErrorEntryComparer(code)) - errors;

        const char* msg = (idx != n) ? errors[idx].str : "Unknown error code";
        cv::String str = cv::format("%s [Code = %d]", msg, code);

        return str;
    }
}

#endif

String cv::cuda::getNppErrorMessage(int code)
{
#ifndef HAVE_CUDA
    (void) code;
    return String();
#else
    return getErrorString(code, npp_errors, npp_error_num);
#endif
}

String cv::cuda::getCudaDriverApiErrorMessage(int code)
{
#ifndef HAVE_CUDA
    (void) code;
    return String();
#else
    return getErrorString(code, cu_errors, cu_errors_num);
#endif
}
