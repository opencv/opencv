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
#include "opencv2/core/gpumat.hpp"
#include <iostream>

#ifdef HAVE_CUDA
    #include <cuda_runtime.h>
    #include <npp.h>

    #define CUDART_MINIMUM_REQUIRED_VERSION 4020
    #define NPP_MINIMUM_REQUIRED_VERSION 4200

    #if (CUDART_VERSION < CUDART_MINIMUM_REQUIRED_VERSION)
        #error "Insufficient Cuda Runtime library version, please update it."
    #endif

    #if (NPP_VERSION_MAJOR * 1000 + NPP_VERSION_MINOR * 100 + NPP_VERSION_BUILD < NPP_MINIMUM_REQUIRED_VERSION)
        #error "Insufficient NPP version, please update it."
    #endif
#endif

using namespace cv;
using namespace cv::gpu;

#ifndef HAVE_CUDA

#define throw_nogpu CV_Error(CV_GpuNotSupported, "The library is compiled without CUDA support")

#else // HAVE_CUDA

namespace
{
#if defined(__GNUC__)
    #define nppSafeCall(expr)  ___nppSafeCall(expr, __FILE__, __LINE__, __func__)
#else /* defined(__CUDACC__) || defined(__MSVC__) */
    #define nppSafeCall(expr)  ___nppSafeCall(expr, __FILE__, __LINE__)
#endif

    inline void ___nppSafeCall(int err, const char *file, const int line, const char *func = "")
    {
        if (err < 0)
        {
            std::ostringstream msg;
            msg << "NPP API Call Error: " << err;
            cv::gpu::error(msg.str().c_str(), file, line, func);
        }
    }
}

#endif // HAVE_CUDA

//////////////////////////////// Initialization & Info ////////////////////////

#ifndef HAVE_CUDA

int cv::gpu::getCudaEnabledDeviceCount() { return 0; }

void cv::gpu::setDevice(int) { throw_nogpu; }
int cv::gpu::getDevice() { throw_nogpu; return 0; }

void cv::gpu::resetDevice() { throw_nogpu; }

bool cv::gpu::deviceSupports(FeatureSet) { throw_nogpu; return false; }

bool cv::gpu::TargetArchs::builtWith(FeatureSet) { throw_nogpu; return false; }
bool cv::gpu::TargetArchs::has(int, int) { throw_nogpu; return false; }
bool cv::gpu::TargetArchs::hasPtx(int, int) { throw_nogpu; return false; }
bool cv::gpu::TargetArchs::hasBin(int, int) { throw_nogpu; return false; }
bool cv::gpu::TargetArchs::hasEqualOrLessPtx(int, int) { throw_nogpu; return false; }
bool cv::gpu::TargetArchs::hasEqualOrGreater(int, int) { throw_nogpu; return false; }
bool cv::gpu::TargetArchs::hasEqualOrGreaterPtx(int, int) { throw_nogpu; return false; }
bool cv::gpu::TargetArchs::hasEqualOrGreaterBin(int, int) { throw_nogpu; return false; }

size_t cv::gpu::DeviceInfo::sharedMemPerBlock() const { throw_nogpu; return 0; }
void cv::gpu::DeviceInfo::queryMemory(size_t&, size_t&) const { throw_nogpu; }
size_t cv::gpu::DeviceInfo::freeMemory() const { throw_nogpu; return 0; }
size_t cv::gpu::DeviceInfo::totalMemory() const { throw_nogpu; return 0; }
bool cv::gpu::DeviceInfo::supports(FeatureSet) const { throw_nogpu; return false; }
bool cv::gpu::DeviceInfo::isCompatible() const { throw_nogpu; return false; }
void cv::gpu::DeviceInfo::query() { throw_nogpu; }

void cv::gpu::printCudaDeviceInfo(int) { throw_nogpu; }
void cv::gpu::printShortCudaDeviceInfo(int) { throw_nogpu; }

#else // HAVE_CUDA

int cv::gpu::getCudaEnabledDeviceCount()
{
    int count;
    cudaError_t error = cudaGetDeviceCount( &count );

    if (error == cudaErrorInsufficientDriver)
        return -1;

    if (error == cudaErrorNoDevice)
        return 0;

    cudaSafeCall( error );
    return count;
}

void cv::gpu::setDevice(int device)
{
    cudaSafeCall( cudaSetDevice( device ) );
}

int cv::gpu::getDevice()
{
    int device;
    cudaSafeCall( cudaGetDevice( &device ) );
    return device;
}

void cv::gpu::resetDevice()
{
    cudaSafeCall( cudaDeviceReset() );
}

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
        static void fromStr(const String& set_as_str, std::vector<int>& arr);

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

    void CudaArch::fromStr(const String& set_as_str, std::vector<int>& arr)
    {
        if (set_as_str.find_first_not_of(" ") == String::npos)
            return;

        std::istringstream stream(set_as_str);
        int cur_value;

        while (!stream.eof())
        {
            stream >> cur_value;
            arr.push_back(cur_value);
        }

        std::sort(arr.begin(), arr.end());
    }
}

bool cv::gpu::TargetArchs::builtWith(cv::gpu::FeatureSet feature_set)
{
    return cudaArch.builtWith(feature_set);
}

bool cv::gpu::TargetArchs::has(int major, int minor)
{
    return hasPtx(major, minor) || hasBin(major, minor);
}

bool cv::gpu::TargetArchs::hasPtx(int major, int minor)
{
    return cudaArch.hasPtx(major, minor);
}

bool cv::gpu::TargetArchs::hasBin(int major, int minor)
{
    return cudaArch.hasBin(major, minor);
}

bool cv::gpu::TargetArchs::hasEqualOrLessPtx(int major, int minor)
{
    return cudaArch.hasEqualOrLessPtx(major, minor);
}

bool cv::gpu::TargetArchs::hasEqualOrGreater(int major, int minor)
{
    return hasEqualOrGreaterPtx(major, minor) || hasEqualOrGreaterBin(major, minor);
}

bool cv::gpu::TargetArchs::hasEqualOrGreaterPtx(int major, int minor)
{
    return cudaArch.hasEqualOrGreaterPtx(major, minor);
}

bool cv::gpu::TargetArchs::hasEqualOrGreaterBin(int major, int minor)
{
    return cudaArch.hasEqualOrGreaterBin(major, minor);
}

bool cv::gpu::deviceSupports(FeatureSet feature_set)
{
    static int versions[] =
    {
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
    };
    static const int cache_size = static_cast<int>(sizeof(versions) / sizeof(versions[0]));

    const int devId = getDevice();

    int version;

    if (devId < cache_size && versions[devId] >= 0)
        version = versions[devId];
    else
    {
        DeviceInfo dev(devId);
        version = dev.majorVersion() * 10 + dev.minorVersion();
        if (devId < cache_size)
            versions[devId] = version;
    }

    return TargetArchs::builtWith(feature_set) && (version >= feature_set);
}

namespace
{
    class DeviceProps
    {
    public:
        DeviceProps();
        ~DeviceProps();

        cudaDeviceProp* get(int devID);

    private:
        std::vector<cudaDeviceProp*> props_;
    };

    DeviceProps::DeviceProps()
    {
        props_.resize(10, 0);
    }

    DeviceProps::~DeviceProps()
    {
        for (size_t i = 0; i < props_.size(); ++i)
        {
            if (props_[i])
                delete props_[i];
        }
        props_.clear();
    }

    cudaDeviceProp* DeviceProps::get(int devID)
    {
        if (devID >= (int) props_.size())
            props_.resize(devID + 5, 0);

        if (!props_[devID])
        {
            props_[devID] = new cudaDeviceProp;
            cudaSafeCall( cudaGetDeviceProperties(props_[devID], devID) );
        }

        return props_[devID];
    }

    DeviceProps deviceProps;
}

size_t cv::gpu::DeviceInfo::sharedMemPerBlock() const
{
    return deviceProps.get(device_id_)->sharedMemPerBlock;
}

void cv::gpu::DeviceInfo::queryMemory(size_t& _totalMemory, size_t& _freeMemory) const
{
    int prevDeviceID = getDevice();
    if (prevDeviceID != device_id_)
        setDevice(device_id_);

    cudaSafeCall( cudaMemGetInfo(&_freeMemory, &_totalMemory) );

    if (prevDeviceID != device_id_)
        setDevice(prevDeviceID);
}

size_t cv::gpu::DeviceInfo::freeMemory() const
{
    size_t _totalMemory, _freeMemory;
    queryMemory(_totalMemory, _freeMemory);
    return _freeMemory;
}

size_t cv::gpu::DeviceInfo::totalMemory() const
{
    size_t _totalMemory, _freeMemory;
    queryMemory(_totalMemory, _freeMemory);
    return _totalMemory;
}

bool cv::gpu::DeviceInfo::supports(FeatureSet feature_set) const
{
    int version = majorVersion() * 10 + minorVersion();
    return version >= feature_set;
}

bool cv::gpu::DeviceInfo::isCompatible() const
{
    // Check PTX compatibility
    if (TargetArchs::hasEqualOrLessPtx(majorVersion(), minorVersion()))
        return true;

    // Check BIN compatibility
    for (int i = minorVersion(); i >= 0; --i)
        if (TargetArchs::hasBin(majorVersion(), i))
            return true;

    return false;
}

void cv::gpu::DeviceInfo::query()
{
    const cudaDeviceProp* prop = deviceProps.get(device_id_);

    name_ = prop->name;
    multi_processor_count_ = prop->multiProcessorCount;
    majorVersion_ = prop->major;
    minorVersion_ = prop->minor;
}

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

void cv::gpu::printCudaDeviceInfo(int device)
{
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
}

void cv::gpu::printShortCudaDeviceInfo(int device)
{
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
}

#endif // HAVE_CUDA

//////////////////////////////// GpuMat ///////////////////////////////

cv::gpu::GpuMat::GpuMat(const GpuMat& m)
    : flags(m.flags), rows(m.rows), cols(m.cols), step(m.step), data(m.data), refcount(m.refcount), datastart(m.datastart), dataend(m.dataend)
{
    if (refcount)
        CV_XADD(refcount, 1);
}

cv::gpu::GpuMat::GpuMat(int rows_, int cols_, int type_, void* data_, size_t step_) :
    flags(Mat::MAGIC_VAL + (type_ & TYPE_MASK)), rows(rows_), cols(cols_),
    step(step_), data((uchar*)data_), refcount(0),
    datastart((uchar*)data_), dataend((uchar*)data_)
{
    size_t minstep = cols * elemSize();

    if (step == Mat::AUTO_STEP)
    {
        step = minstep;
        flags |= Mat::CONTINUOUS_FLAG;
    }
    else
    {
        if (rows == 1)
            step = minstep;

        CV_DbgAssert(step >= minstep);

        flags |= step == minstep ? Mat::CONTINUOUS_FLAG : 0;
    }
    dataend += step * (rows - 1) + minstep;
}

cv::gpu::GpuMat::GpuMat(Size size_, int type_, void* data_, size_t step_) :
    flags(Mat::MAGIC_VAL + (type_ & TYPE_MASK)), rows(size_.height), cols(size_.width),
    step(step_), data((uchar*)data_), refcount(0),
    datastart((uchar*)data_), dataend((uchar*)data_)
{
    size_t minstep = cols * elemSize();

    if (step == Mat::AUTO_STEP)
    {
        step = minstep;
        flags |= Mat::CONTINUOUS_FLAG;
    }
    else
    {
        if (rows == 1)
            step = minstep;

        CV_DbgAssert(step >= minstep);

        flags |= step == minstep ? Mat::CONTINUOUS_FLAG : 0;
    }
    dataend += step * (rows - 1) + minstep;
}

cv::gpu::GpuMat::GpuMat(const GpuMat& m, Range _rowRange, Range _colRange)
{
    flags = m.flags;
    step = m.step; refcount = m.refcount;
    data = m.data; datastart = m.datastart; dataend = m.dataend;

    if (_rowRange == Range::all())
        rows = m.rows;
    else
    {
        CV_Assert(0 <= _rowRange.start && _rowRange.start <= _rowRange.end && _rowRange.end <= m.rows);

        rows = _rowRange.size();
        data += step*_rowRange.start;
    }

    if (_colRange == Range::all())
        cols = m.cols;
    else
    {
        CV_Assert(0 <= _colRange.start && _colRange.start <= _colRange.end && _colRange.end <= m.cols);

        cols = _colRange.size();
        data += _colRange.start*elemSize();
        flags &= cols < m.cols ? ~Mat::CONTINUOUS_FLAG : -1;
    }

    if (rows == 1)
        flags |= Mat::CONTINUOUS_FLAG;

    if (refcount)
        CV_XADD(refcount, 1);

    if (rows <= 0 || cols <= 0)
        rows = cols = 0;
}

cv::gpu::GpuMat::GpuMat(const GpuMat& m, Rect roi) :
    flags(m.flags), rows(roi.height), cols(roi.width),
    step(m.step), data(m.data + roi.y*step), refcount(m.refcount),
    datastart(m.datastart), dataend(m.dataend)
{
    flags &= roi.width < m.cols ? ~Mat::CONTINUOUS_FLAG : -1;
    data += roi.x * elemSize();

    CV_Assert(0 <= roi.x && 0 <= roi.width && roi.x + roi.width <= m.cols && 0 <= roi.y && 0 <= roi.height && roi.y + roi.height <= m.rows);

    if (refcount)
        CV_XADD(refcount, 1);

    if (rows <= 0 || cols <= 0)
        rows = cols = 0;
}

cv::gpu::GpuMat::GpuMat(const Mat& m) :
    flags(0), rows(0), cols(0), step(0), data(0), refcount(0), datastart(0), dataend(0)
{
    upload(m);
}

GpuMat& cv::gpu::GpuMat::operator = (const GpuMat& m)
{
    if (this != &m)
    {
        GpuMat temp(m);
        swap(temp);
    }

    return *this;
}

void cv::gpu::GpuMat::swap(GpuMat& b)
{
    std::swap(flags, b.flags);
    std::swap(rows, b.rows);
    std::swap(cols, b.cols);
    std::swap(step, b.step);
    std::swap(data, b.data);
    std::swap(datastart, b.datastart);
    std::swap(dataend, b.dataend);
    std::swap(refcount, b.refcount);
}

void cv::gpu::GpuMat::locateROI(Size& wholeSize, Point& ofs) const
{
    size_t esz = elemSize();
    ptrdiff_t delta1 = data - datastart;
    ptrdiff_t delta2 = dataend - datastart;

    CV_DbgAssert(step > 0);

    if (delta1 == 0)
        ofs.x = ofs.y = 0;
    else
    {
        ofs.y = static_cast<int>(delta1 / step);
        ofs.x = static_cast<int>((delta1 - step * ofs.y) / esz);

        CV_DbgAssert(data == datastart + ofs.y * step + ofs.x * esz);
    }

    size_t minstep = (ofs.x + cols) * esz;

    wholeSize.height = std::max(static_cast<int>((delta2 - minstep) / step + 1), ofs.y + rows);
    wholeSize.width = std::max(static_cast<int>((delta2 - step * (wholeSize.height - 1)) / esz), ofs.x + cols);
}

GpuMat& cv::gpu::GpuMat::adjustROI(int dtop, int dbottom, int dleft, int dright)
{
    Size wholeSize;
    Point ofs;
    locateROI(wholeSize, ofs);

    size_t esz = elemSize();

    int row1 = std::max(ofs.y - dtop, 0);
    int row2 = std::min(ofs.y + rows + dbottom, wholeSize.height);

    int col1 = std::max(ofs.x - dleft, 0);
    int col2 = std::min(ofs.x + cols + dright, wholeSize.width);

    data += (row1 - ofs.y) * step + (col1 - ofs.x) * esz;
    rows = row2 - row1;
    cols = col2 - col1;

    if (esz * cols == step || rows == 1)
        flags |= Mat::CONTINUOUS_FLAG;
    else
        flags &= ~Mat::CONTINUOUS_FLAG;

    return *this;
}

GpuMat cv::gpu::GpuMat::reshape(int new_cn, int new_rows) const
{
    GpuMat hdr = *this;

    int cn = channels();
    if (new_cn == 0)
        new_cn = cn;

    int total_width = cols * cn;

    if ((new_cn > total_width || total_width % new_cn != 0) && new_rows == 0)
        new_rows = rows * total_width / new_cn;

    if (new_rows != 0 && new_rows != rows)
    {
        int total_size = total_width * rows;

        if (!isContinuous())
            CV_Error(CV_BadStep, "The matrix is not continuous, thus its number of rows can not be changed");

        if ((unsigned)new_rows > (unsigned)total_size)
            CV_Error(CV_StsOutOfRange, "Bad new number of rows");

        total_width = total_size / new_rows;

        if (total_width * new_rows != total_size)
            CV_Error(CV_StsBadArg, "The total number of matrix elements is not divisible by the new number of rows");

        hdr.rows = new_rows;
        hdr.step = total_width * elemSize1();
    }

    int new_width = total_width / new_cn;

    if (new_width * new_cn != total_width)
        CV_Error(CV_BadNumChannels, "The total width is not divisible by the new number of channels");

    hdr.cols = new_width;
    hdr.flags = (hdr.flags & ~CV_MAT_CN_MASK) | ((new_cn - 1) << CV_CN_SHIFT);

    return hdr;
}

cv::Mat::Mat(const GpuMat& m) : flags(0), dims(0), rows(0), cols(0), data(0), refcount(0), datastart(0), dataend(0), datalimit(0), allocator(0), size(&rows)
{
    m.download(*this);
}

void cv::gpu::createContinuous(int rows, int cols, int type, GpuMat& m)
{
    int area = rows * cols;
    if (m.empty() || m.type() != type || !m.isContinuous() || m.size().area() < area)
        m.create(1, area, type);

    m.cols = cols;
    m.rows = rows;
    m.step = m.elemSize() * cols;
    m.flags |= Mat::CONTINUOUS_FLAG;
}

void cv::gpu::ensureSizeIsEnough(int rows, int cols, int type, GpuMat& m)
{
    if (m.empty() || m.type() != type || m.data != m.datastart)
        m.create(rows, cols, type);
    else
    {
        const size_t esz = m.elemSize();
        const ptrdiff_t delta2 = m.dataend - m.datastart;

        const size_t minstep = m.cols * esz;

        Size wholeSize;
        wholeSize.height = std::max(static_cast<int>((delta2 - minstep) / m.step + 1), m.rows);
        wholeSize.width = std::max(static_cast<int>((delta2 - m.step * (wholeSize.height - 1)) / esz), m.cols);

        if (wholeSize.height < rows || wholeSize.width < cols)
            m.create(rows, cols, type);
        else
        {
            m.cols = cols;
            m.rows = rows;
        }
    }
}

GpuMat cv::gpu::allocMatFromBuf(int rows, int cols, int type, GpuMat &mat)
{
    if (!mat.empty() && mat.type() == type && mat.rows >= rows && mat.cols >= cols)
        return mat(Rect(0, 0, cols, rows));
    return mat = GpuMat(rows, cols, type);
}

namespace
{
    class GpuFuncTable
    {
    public:
        virtual ~GpuFuncTable() {}

        virtual void copy(const Mat& src, GpuMat& dst) const = 0;
        virtual void copy(const GpuMat& src, Mat& dst) const = 0;
        virtual void copy(const GpuMat& src, GpuMat& dst) const = 0;

        virtual void copyWithMask(const GpuMat& src, GpuMat& dst, const GpuMat& mask) const = 0;

        virtual void convert(const GpuMat& src, GpuMat& dst) const = 0;
        virtual void convert(const GpuMat& src, GpuMat& dst, double alpha, double beta) const = 0;

        virtual void setTo(GpuMat& m, Scalar s, const GpuMat& mask) const = 0;

        virtual void mallocPitch(void** devPtr, size_t* step, size_t width, size_t height) const = 0;
        virtual void free(void* devPtr) const = 0;
    };
}

#ifndef HAVE_CUDA

namespace
{
    class EmptyFuncTable : public GpuFuncTable
    {
    public:
        void copy(const Mat&, GpuMat&) const { throw_nogpu; }
        void copy(const GpuMat&, Mat&) const { throw_nogpu; }
        void copy(const GpuMat&, GpuMat&) const { throw_nogpu; }

        void copyWithMask(const GpuMat&, GpuMat&, const GpuMat&) const { throw_nogpu; }

        void convert(const GpuMat&, GpuMat&) const { throw_nogpu; }
        void convert(const GpuMat&, GpuMat&, double, double) const { throw_nogpu; }

        void setTo(GpuMat&, Scalar, const GpuMat&) const { throw_nogpu; }

        void mallocPitch(void**, size_t*, size_t, size_t) const { throw_nogpu; }
        void free(void*) const {}
    };

    const GpuFuncTable* gpuFuncTable()
    {
        static EmptyFuncTable empty;
        return &empty;
    }
}

#else // HAVE_CUDA

namespace cv { namespace gpu { namespace device
{
    void copyToWithMask_gpu(PtrStepSzb src, PtrStepSzb dst, size_t elemSize1, int cn, PtrStepSzb mask, bool colorMask, cudaStream_t stream);

    template <typename T>
    void set_to_gpu(PtrStepSzb mat, const T* scalar, int channels, cudaStream_t stream);

    template <typename T>
    void set_to_gpu(PtrStepSzb mat, const T* scalar, PtrStepSzb mask, int channels, cudaStream_t stream);

    void convert_gpu(PtrStepSzb src, int sdepth, PtrStepSzb dst, int ddepth, double alpha, double beta, cudaStream_t stream);
}}}

namespace
{
    template <typename T> void kernelSetCaller(GpuMat& src, Scalar s, cudaStream_t stream)
    {
        Scalar_<T> sf = s;
        cv::gpu::device::set_to_gpu(src, sf.val, src.channels(), stream);
    }

    template <typename T> void kernelSetCaller(GpuMat& src, Scalar s, const GpuMat& mask, cudaStream_t stream)
    {
        Scalar_<T> sf = s;
        cv::gpu::device::set_to_gpu(src, sf.val, mask, src.channels(), stream);
    }
}


namespace cv { namespace gpu
{
    CV_EXPORTS void copyWithMask(const cv::gpu::GpuMat&, cv::gpu::GpuMat&, const cv::gpu::GpuMat&, CUstream_st*);
    CV_EXPORTS void convertTo(const cv::gpu::GpuMat&, cv::gpu::GpuMat&);
    CV_EXPORTS void convertTo(const cv::gpu::GpuMat&, cv::gpu::GpuMat&, double, double, CUstream_st*);
    CV_EXPORTS void setTo(cv::gpu::GpuMat&, cv::Scalar, CUstream_st*);
    CV_EXPORTS void setTo(cv::gpu::GpuMat&, cv::Scalar, const cv::gpu::GpuMat&, CUstream_st*);
    CV_EXPORTS void setTo(cv::gpu::GpuMat&, cv::Scalar);
    CV_EXPORTS void setTo(cv::gpu::GpuMat&, cv::Scalar, const cv::gpu::GpuMat&);
}}


namespace cv { namespace gpu
{
    void copyWithMask(const GpuMat& src, GpuMat& dst, const GpuMat& mask, cudaStream_t stream = 0)
    {
        CV_Assert(src.size() == dst.size() && src.type() == dst.type());
        CV_Assert(src.size() == mask.size() && mask.depth() == CV_8U && (mask.channels() == 1 || mask.channels() == src.channels()));

        cv::gpu::device::copyToWithMask_gpu(src.reshape(1), dst.reshape(1), src.elemSize1(), src.channels(), mask.reshape(1), mask.channels() != 1, stream);
    }

    void convertTo(const GpuMat& src, GpuMat& dst)
    {
        cv::gpu::device::convert_gpu(src.reshape(1), src.depth(), dst.reshape(1), dst.depth(), 1.0, 0.0, 0);
    }

    void convertTo(const GpuMat& src, GpuMat& dst, double alpha, double beta, cudaStream_t stream = 0)
    {
        cv::gpu::device::convert_gpu(src.reshape(1), src.depth(), dst.reshape(1), dst.depth(), alpha, beta, stream);
    }

    void setTo(GpuMat& src, Scalar s, cudaStream_t stream)
    {
        typedef void (*caller_t)(GpuMat& src, Scalar s, cudaStream_t stream);

        static const caller_t callers[] =
        {
            kernelSetCaller<uchar>, kernelSetCaller<schar>, kernelSetCaller<ushort>, kernelSetCaller<short>, kernelSetCaller<int>,
            kernelSetCaller<float>, kernelSetCaller<double>
        };

        callers[src.depth()](src, s, stream);
    }

    void setTo(GpuMat& src, Scalar s, const GpuMat& mask, cudaStream_t stream)
    {
        typedef void (*caller_t)(GpuMat& src, Scalar s, const GpuMat& mask, cudaStream_t stream);

        static const caller_t callers[] =
        {
            kernelSetCaller<uchar>, kernelSetCaller<schar>, kernelSetCaller<ushort>, kernelSetCaller<short>, kernelSetCaller<int>,
            kernelSetCaller<float>, kernelSetCaller<double>
        };

        callers[src.depth()](src, s, mask, stream);
    }

    void setTo(GpuMat& src, Scalar s)
    {
        setTo(src, s, 0);
    }

    void setTo(GpuMat& src, Scalar s, const GpuMat& mask)
    {
        setTo(src, s, mask, 0);
    }
}}

namespace
{
    template<int n> struct NPPTypeTraits;
    template<> struct NPPTypeTraits<CV_8U>  { typedef Npp8u npp_type; };
    template<> struct NPPTypeTraits<CV_8S>  { typedef Npp8s npp_type; };
    template<> struct NPPTypeTraits<CV_16U> { typedef Npp16u npp_type; };
    template<> struct NPPTypeTraits<CV_16S> { typedef Npp16s npp_type; };
    template<> struct NPPTypeTraits<CV_32S> { typedef Npp32s npp_type; };
    template<> struct NPPTypeTraits<CV_32F> { typedef Npp32f npp_type; };
    template<> struct NPPTypeTraits<CV_64F> { typedef Npp64f npp_type; };

    //////////////////////////////////////////////////////////////////////////
    // Convert

    template<int SDEPTH, int DDEPTH> struct NppConvertFunc
    {
        typedef typename NPPTypeTraits<SDEPTH>::npp_type src_t;
        typedef typename NPPTypeTraits<DDEPTH>::npp_type dst_t;

        typedef NppStatus (*func_ptr)(const src_t* pSrc, int nSrcStep, dst_t* pDst, int nDstStep, NppiSize oSizeROI);
    };
    template<int DDEPTH> struct NppConvertFunc<CV_32F, DDEPTH>
    {
        typedef typename NPPTypeTraits<DDEPTH>::npp_type dst_t;

        typedef NppStatus (*func_ptr)(const Npp32f* pSrc, int nSrcStep, dst_t* pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode);
    };

    template<int SDEPTH, int DDEPTH, typename NppConvertFunc<SDEPTH, DDEPTH>::func_ptr func> struct NppCvt
    {
        typedef typename NPPTypeTraits<SDEPTH>::npp_type src_t;
        typedef typename NPPTypeTraits<DDEPTH>::npp_type dst_t;

        static void call(const GpuMat& src, GpuMat& dst)
        {
            NppiSize sz;
            sz.width = src.cols;
            sz.height = src.rows;

            nppSafeCall( func(src.ptr<src_t>(), static_cast<int>(src.step), dst.ptr<dst_t>(), static_cast<int>(dst.step), sz) );

            cudaSafeCall( cudaDeviceSynchronize() );
        }
    };
    template<int DDEPTH, typename NppConvertFunc<CV_32F, DDEPTH>::func_ptr func> struct NppCvt<CV_32F, DDEPTH, func>
    {
        typedef typename NPPTypeTraits<DDEPTH>::npp_type dst_t;

        static void call(const GpuMat& src, GpuMat& dst)
        {
            NppiSize sz;
            sz.width = src.cols;
            sz.height = src.rows;

            nppSafeCall( func(src.ptr<Npp32f>(), static_cast<int>(src.step), dst.ptr<dst_t>(), static_cast<int>(dst.step), sz, NPP_RND_NEAR) );

            cudaSafeCall( cudaDeviceSynchronize() );
        }
    };

    //////////////////////////////////////////////////////////////////////////
    // Set

    template<int SDEPTH, int SCN> struct NppSetFunc
    {
        typedef typename NPPTypeTraits<SDEPTH>::npp_type src_t;

        typedef NppStatus (*func_ptr)(const src_t values[], src_t* pSrc, int nSrcStep, NppiSize oSizeROI);
    };
    template<int SDEPTH> struct NppSetFunc<SDEPTH, 1>
    {
        typedef typename NPPTypeTraits<SDEPTH>::npp_type src_t;

        typedef NppStatus (*func_ptr)(src_t val, src_t* pSrc, int nSrcStep, NppiSize oSizeROI);
    };
    template<int SCN> struct NppSetFunc<CV_8S, SCN>
    {
        typedef NppStatus (*func_ptr)(Npp8s values[], Npp8s* pSrc, int nSrcStep, NppiSize oSizeROI);
    };
    template<> struct NppSetFunc<CV_8S, 1>
    {
        typedef NppStatus (*func_ptr)(Npp8s val, Npp8s* pSrc, int nSrcStep, NppiSize oSizeROI);
    };

    template<int SDEPTH, int SCN, typename NppSetFunc<SDEPTH, SCN>::func_ptr func> struct NppSet
    {
        typedef typename NPPTypeTraits<SDEPTH>::npp_type src_t;

        static void call(GpuMat& src, Scalar s)
        {
            NppiSize sz;
            sz.width = src.cols;
            sz.height = src.rows;

            Scalar_<src_t> nppS = s;

            nppSafeCall( func(nppS.val, src.ptr<src_t>(), static_cast<int>(src.step), sz) );

            cudaSafeCall( cudaDeviceSynchronize() );
        }
    };
    template<int SDEPTH, typename NppSetFunc<SDEPTH, 1>::func_ptr func> struct NppSet<SDEPTH, 1, func>
    {
        typedef typename NPPTypeTraits<SDEPTH>::npp_type src_t;

        static void call(GpuMat& src, Scalar s)
        {
            NppiSize sz;
            sz.width = src.cols;
            sz.height = src.rows;

            Scalar_<src_t> nppS = s;

            nppSafeCall( func(nppS[0], src.ptr<src_t>(), static_cast<int>(src.step), sz) );

            cudaSafeCall( cudaDeviceSynchronize() );
        }
    };

    template<int SDEPTH, int SCN> struct NppSetMaskFunc
    {
        typedef typename NPPTypeTraits<SDEPTH>::npp_type src_t;

        typedef NppStatus (*func_ptr)(const src_t values[], src_t* pSrc, int nSrcStep, NppiSize oSizeROI, const Npp8u* pMask, int nMaskStep);
    };
    template<int SDEPTH> struct NppSetMaskFunc<SDEPTH, 1>
    {
        typedef typename NPPTypeTraits<SDEPTH>::npp_type src_t;

        typedef NppStatus (*func_ptr)(src_t val, src_t* pSrc, int nSrcStep, NppiSize oSizeROI, const Npp8u* pMask, int nMaskStep);
    };

    template<int SDEPTH, int SCN, typename NppSetMaskFunc<SDEPTH, SCN>::func_ptr func> struct NppSetMask
    {
        typedef typename NPPTypeTraits<SDEPTH>::npp_type src_t;

        static void call(GpuMat& src, Scalar s, const GpuMat& mask)
        {
            NppiSize sz;
            sz.width = src.cols;
            sz.height = src.rows;

            Scalar_<src_t> nppS = s;

            nppSafeCall( func(nppS.val, src.ptr<src_t>(), static_cast<int>(src.step), sz, mask.ptr<Npp8u>(), static_cast<int>(mask.step)) );

            cudaSafeCall( cudaDeviceSynchronize() );
        }
    };
    template<int SDEPTH, typename NppSetMaskFunc<SDEPTH, 1>::func_ptr func> struct NppSetMask<SDEPTH, 1, func>
    {
        typedef typename NPPTypeTraits<SDEPTH>::npp_type src_t;

        static void call(GpuMat& src, Scalar s, const GpuMat& mask)
        {
            NppiSize sz;
            sz.width = src.cols;
            sz.height = src.rows;

            Scalar_<src_t> nppS = s;

            nppSafeCall( func(nppS[0], src.ptr<src_t>(), static_cast<int>(src.step), sz, mask.ptr<Npp8u>(), static_cast<int>(mask.step)) );

            cudaSafeCall( cudaDeviceSynchronize() );
        }
    };

    //////////////////////////////////////////////////////////////////////////
    // CopyMasked

    template<int SDEPTH> struct NppCopyMaskedFunc
    {
        typedef typename NPPTypeTraits<SDEPTH>::npp_type src_t;

        typedef NppStatus (*func_ptr)(const src_t* pSrc, int nSrcStep, src_t* pDst, int nDstStep, NppiSize oSizeROI, const Npp8u* pMask, int nMaskStep);
    };

    template<int SDEPTH, typename NppCopyMaskedFunc<SDEPTH>::func_ptr func> struct NppCopyMasked
    {
        typedef typename NPPTypeTraits<SDEPTH>::npp_type src_t;

        static void call(const GpuMat& src, GpuMat& dst, const GpuMat& mask, cudaStream_t /*stream*/)
        {
            NppiSize sz;
            sz.width = src.cols;
            sz.height = src.rows;

            nppSafeCall( func(src.ptr<src_t>(), static_cast<int>(src.step), dst.ptr<src_t>(), static_cast<int>(dst.step), sz, mask.ptr<Npp8u>(), static_cast<int>(mask.step)) );

            cudaSafeCall( cudaDeviceSynchronize() );
        }
    };

    template <typename T> static inline bool isAligned(const T* ptr, size_t size)
    {
        return reinterpret_cast<size_t>(ptr) % size == 0;
    }

    //////////////////////////////////////////////////////////////////////////
    // CudaFuncTable

    class CudaFuncTable : public GpuFuncTable
    {
    public:
        void copy(const Mat& src, GpuMat& dst) const
        {
            cudaSafeCall( cudaMemcpy2D(dst.data, dst.step, src.data, src.step, src.cols * src.elemSize(), src.rows, cudaMemcpyHostToDevice) );
        }
        void copy(const GpuMat& src, Mat& dst) const
        {
            cudaSafeCall( cudaMemcpy2D(dst.data, dst.step, src.data, src.step, src.cols * src.elemSize(), src.rows, cudaMemcpyDeviceToHost) );
        }
        void copy(const GpuMat& src, GpuMat& dst) const
        {
            cudaSafeCall( cudaMemcpy2D(dst.data, dst.step, src.data, src.step, src.cols * src.elemSize(), src.rows, cudaMemcpyDeviceToDevice) );
        }

        void copyWithMask(const GpuMat& src, GpuMat& dst, const GpuMat& mask) const
        {
            CV_Assert(src.depth() <= CV_64F && src.channels() <= 4);
            CV_Assert(src.size() == dst.size() && src.type() == dst.type());
            CV_Assert(src.size() == mask.size() && mask.depth() == CV_8U && (mask.channels() == 1 || mask.channels() == src.channels()));

            if (src.depth() == CV_64F)
            {
                if (!TargetArchs::builtWith(NATIVE_DOUBLE) || !DeviceInfo().supports(NATIVE_DOUBLE))
                    CV_Error(CV_StsUnsupportedFormat, "The device doesn't support double");
            }

            typedef void (*func_t)(const GpuMat& src, GpuMat& dst, const GpuMat& mask, cudaStream_t stream);
            static const func_t funcs[7][4] =
            {
                /*  8U */ {NppCopyMasked<CV_8U , nppiCopy_8u_C1MR >::call, cv::gpu::copyWithMask, NppCopyMasked<CV_8U , nppiCopy_8u_C3MR >::call, NppCopyMasked<CV_8U , nppiCopy_8u_C4MR >::call},
                /*  8S */ {cv::gpu::copyWithMask                         , cv::gpu::copyWithMask, cv::gpu::copyWithMask                         , cv::gpu::copyWithMask                         },
                /* 16U */ {NppCopyMasked<CV_16U, nppiCopy_16u_C1MR>::call, cv::gpu::copyWithMask, NppCopyMasked<CV_16U, nppiCopy_16u_C3MR>::call, NppCopyMasked<CV_16U, nppiCopy_16u_C4MR>::call},
                /* 16S */ {NppCopyMasked<CV_16S, nppiCopy_16s_C1MR>::call, cv::gpu::copyWithMask, NppCopyMasked<CV_16S, nppiCopy_16s_C3MR>::call, NppCopyMasked<CV_16S, nppiCopy_16s_C4MR>::call},
                /* 32S */ {NppCopyMasked<CV_32S, nppiCopy_32s_C1MR>::call, cv::gpu::copyWithMask, NppCopyMasked<CV_32S, nppiCopy_32s_C3MR>::call, NppCopyMasked<CV_32S, nppiCopy_32s_C4MR>::call},
                /* 32F */ {NppCopyMasked<CV_32F, nppiCopy_32f_C1MR>::call, cv::gpu::copyWithMask, NppCopyMasked<CV_32F, nppiCopy_32f_C3MR>::call, NppCopyMasked<CV_32F, nppiCopy_32f_C4MR>::call},
                /* 64F */ {cv::gpu::copyWithMask                         , cv::gpu::copyWithMask, cv::gpu::copyWithMask                         , cv::gpu::copyWithMask                         }
            };

            const func_t func =  mask.channels() == src.channels() ? funcs[src.depth()][src.channels() - 1] : cv::gpu::copyWithMask;

            func(src, dst, mask, 0);
        }

        void convert(const GpuMat& src, GpuMat& dst) const
        {
            typedef void (*func_t)(const GpuMat& src, GpuMat& dst);
            static const func_t funcs[7][7][4] =
            {
                {
                    /*  8U ->  8U */ {0, 0, 0, 0},
                    /*  8U ->  8S */ {cv::gpu::convertTo                                , cv::gpu::convertTo, cv::gpu::convertTo, cv::gpu::convertTo                                },
                    /*  8U -> 16U */ {NppCvt<CV_8U, CV_16U, nppiConvert_8u16u_C1R>::call, cv::gpu::convertTo, cv::gpu::convertTo, NppCvt<CV_8U, CV_16U, nppiConvert_8u16u_C4R>::call},
                    /*  8U -> 16S */ {NppCvt<CV_8U, CV_16S, nppiConvert_8u16s_C1R>::call, cv::gpu::convertTo, cv::gpu::convertTo, NppCvt<CV_8U, CV_16S, nppiConvert_8u16s_C4R>::call},
                    /*  8U -> 32S */ {cv::gpu::convertTo                                , cv::gpu::convertTo, cv::gpu::convertTo, cv::gpu::convertTo                                },
                    /*  8U -> 32F */ {NppCvt<CV_8U, CV_32F, nppiConvert_8u32f_C1R>::call, cv::gpu::convertTo, cv::gpu::convertTo, cv::gpu::convertTo                                },
                    /*  8U -> 64F */ {cv::gpu::convertTo                                , cv::gpu::convertTo, cv::gpu::convertTo, cv::gpu::convertTo                                }
                },
                {
                    /*  8S ->  8U */ {cv::gpu::convertTo, cv::gpu::convertTo, cv::gpu::convertTo, cv::gpu::convertTo},
                    /*  8S ->  8S */ {0,0,0,0},
                    /*  8S -> 16U */ {cv::gpu::convertTo, cv::gpu::convertTo, cv::gpu::convertTo, cv::gpu::convertTo},
                    /*  8S -> 16S */ {cv::gpu::convertTo, cv::gpu::convertTo, cv::gpu::convertTo, cv::gpu::convertTo},
                    /*  8S -> 32S */ {cv::gpu::convertTo, cv::gpu::convertTo, cv::gpu::convertTo, cv::gpu::convertTo},
                    /*  8S -> 32F */ {cv::gpu::convertTo, cv::gpu::convertTo, cv::gpu::convertTo, cv::gpu::convertTo},
                    /*  8S -> 64F */ {cv::gpu::convertTo, cv::gpu::convertTo, cv::gpu::convertTo, cv::gpu::convertTo}
                },
                {
                    /* 16U ->  8U */ {NppCvt<CV_16U, CV_8U , nppiConvert_16u8u_C1R >::call, cv::gpu::convertTo, cv::gpu::convertTo, NppCvt<CV_16U, CV_8U, nppiConvert_16u8u_C4R>::call},
                    /* 16U ->  8S */ {cv::gpu::convertTo                                  , cv::gpu::convertTo, cv::gpu::convertTo, cv::gpu::convertTo                                },
                    /* 16U -> 16U */ {0,0,0,0},
                    /* 16U -> 16S */ {cv::gpu::convertTo                                  , cv::gpu::convertTo, cv::gpu::convertTo, cv::gpu::convertTo                                },
                    /* 16U -> 32S */ {NppCvt<CV_16U, CV_32S, nppiConvert_16u32s_C1R>::call, cv::gpu::convertTo, cv::gpu::convertTo, cv::gpu::convertTo                                },
                    /* 16U -> 32F */ {NppCvt<CV_16U, CV_32F, nppiConvert_16u32f_C1R>::call, cv::gpu::convertTo, cv::gpu::convertTo, cv::gpu::convertTo                                },
                    /* 16U -> 64F */ {cv::gpu::convertTo                                  , cv::gpu::convertTo, cv::gpu::convertTo, cv::gpu::convertTo                                }
                },
                {
                    /* 16S ->  8U */ {NppCvt<CV_16S, CV_8U , nppiConvert_16s8u_C1R >::call, cv::gpu::convertTo, cv::gpu::convertTo, NppCvt<CV_16S, CV_8U, nppiConvert_16s8u_C4R>::call},
                    /* 16S ->  8S */ {cv::gpu::convertTo                                  , cv::gpu::convertTo, cv::gpu::convertTo, cv::gpu::convertTo                                },
                    /* 16S -> 16U */ {cv::gpu::convertTo                                  , cv::gpu::convertTo, cv::gpu::convertTo, cv::gpu::convertTo                                },
                    /* 16S -> 16S */ {0,0,0,0},
                    /* 16S -> 32S */ {NppCvt<CV_16S, CV_32S, nppiConvert_16s32s_C1R>::call, cv::gpu::convertTo, cv::gpu::convertTo, cv::gpu::convertTo                                },
                    /* 16S -> 32F */ {NppCvt<CV_16S, CV_32F, nppiConvert_16s32f_C1R>::call, cv::gpu::convertTo, cv::gpu::convertTo, cv::gpu::convertTo                                },
                    /* 16S -> 64F */ {cv::gpu::convertTo                                  , cv::gpu::convertTo, cv::gpu::convertTo, cv::gpu::convertTo                                }
                },
                {
                    /* 32S ->  8U */ {cv::gpu::convertTo, cv::gpu::convertTo, cv::gpu::convertTo, cv::gpu::convertTo},
                    /* 32S ->  8S */ {cv::gpu::convertTo, cv::gpu::convertTo, cv::gpu::convertTo, cv::gpu::convertTo},
                    /* 32S -> 16U */ {cv::gpu::convertTo, cv::gpu::convertTo, cv::gpu::convertTo, cv::gpu::convertTo},
                    /* 32S -> 16S */ {cv::gpu::convertTo, cv::gpu::convertTo, cv::gpu::convertTo, cv::gpu::convertTo},
                    /* 32S -> 32S */ {0,0,0,0},
                    /* 32S -> 32F */ {cv::gpu::convertTo, cv::gpu::convertTo, cv::gpu::convertTo, cv::gpu::convertTo},
                    /* 32S -> 64F */ {cv::gpu::convertTo, cv::gpu::convertTo, cv::gpu::convertTo, cv::gpu::convertTo}
                },
                {
                    /* 32F ->  8U */ {NppCvt<CV_32F, CV_8U , nppiConvert_32f8u_C1R >::call, cv::gpu::convertTo, cv::gpu::convertTo, cv::gpu::convertTo},
                    /* 32F ->  8S */ {cv::gpu::convertTo                                  , cv::gpu::convertTo, cv::gpu::convertTo, cv::gpu::convertTo},
                    /* 32F -> 16U */ {NppCvt<CV_32F, CV_16U, nppiConvert_32f16u_C1R>::call, cv::gpu::convertTo, cv::gpu::convertTo, cv::gpu::convertTo},
                    /* 32F -> 16S */ {NppCvt<CV_32F, CV_16S, nppiConvert_32f16s_C1R>::call, cv::gpu::convertTo, cv::gpu::convertTo, cv::gpu::convertTo},
                    /* 32F -> 32S */ {cv::gpu::convertTo                                  , cv::gpu::convertTo, cv::gpu::convertTo, cv::gpu::convertTo},
                    /* 32F -> 32F */ {0,0,0,0},
                    /* 32F -> 64F */ {cv::gpu::convertTo                                  , cv::gpu::convertTo, cv::gpu::convertTo, cv::gpu::convertTo}
                },
                {
                    /* 64F ->  8U */ {cv::gpu::convertTo, cv::gpu::convertTo, cv::gpu::convertTo, cv::gpu::convertTo},
                    /* 64F ->  8S */ {cv::gpu::convertTo, cv::gpu::convertTo, cv::gpu::convertTo, cv::gpu::convertTo},
                    /* 64F -> 16U */ {cv::gpu::convertTo, cv::gpu::convertTo, cv::gpu::convertTo, cv::gpu::convertTo},
                    /* 64F -> 16S */ {cv::gpu::convertTo, cv::gpu::convertTo, cv::gpu::convertTo, cv::gpu::convertTo},
                    /* 64F -> 32S */ {cv::gpu::convertTo, cv::gpu::convertTo, cv::gpu::convertTo, cv::gpu::convertTo},
                    /* 64F -> 32F */ {cv::gpu::convertTo, cv::gpu::convertTo, cv::gpu::convertTo, cv::gpu::convertTo},
                    /* 64F -> 64F */ {0,0,0,0}
                }
            };

            CV_Assert(src.depth() <= CV_64F && src.channels() <= 4);
            CV_Assert(dst.depth() <= CV_64F);
            CV_Assert(src.size() == dst.size() && src.channels() == dst.channels());

            if (src.depth() == CV_64F || dst.depth() == CV_64F)
            {
                if (!TargetArchs::builtWith(NATIVE_DOUBLE) || !DeviceInfo().supports(NATIVE_DOUBLE))
                    CV_Error(CV_StsUnsupportedFormat, "The device doesn't support double");
            }

            bool aligned = isAligned(src.data, 16) && isAligned(dst.data, 16);
            if (!aligned)
            {
                cv::gpu::convertTo(src, dst);
                return;
            }

            const func_t func = funcs[src.depth()][dst.depth()][src.channels() - 1];
            CV_DbgAssert(func != 0);

            func(src, dst);
        }

        void convert(const GpuMat& src, GpuMat& dst, double alpha, double beta) const
        {
            CV_Assert(src.depth() <= CV_64F && src.channels() <= 4);
            CV_Assert(dst.depth() <= CV_64F);

            if (src.depth() == CV_64F || dst.depth() == CV_64F)
            {
                if (!TargetArchs::builtWith(NATIVE_DOUBLE) || !DeviceInfo().supports(NATIVE_DOUBLE))
                    CV_Error(CV_StsUnsupportedFormat, "The device doesn't support double");
            }

            cv::gpu::convertTo(src, dst, alpha, beta);
        }

        void setTo(GpuMat& m, Scalar s, const GpuMat& mask) const
        {
            if (mask.empty())
            {
                if (s[0] == 0.0 && s[1] == 0.0 && s[2] == 0.0 && s[3] == 0.0)
                {
                    cudaSafeCall( cudaMemset2D(m.data, m.step, 0, m.cols * m.elemSize(), m.rows) );
                    return;
                }

                if (m.depth() == CV_8U)
                {
                    int cn = m.channels();

                    if (cn == 1 || (cn == 2 && s[0] == s[1]) || (cn == 3 && s[0] == s[1] && s[0] == s[2]) || (cn == 4 && s[0] == s[1] && s[0] == s[2] && s[0] == s[3]))
                    {
                        int val = saturate_cast<uchar>(s[0]);
                        cudaSafeCall( cudaMemset2D(m.data, m.step, val, m.cols * m.elemSize(), m.rows) );
                        return;
                    }
                }

                typedef void (*func_t)(GpuMat& src, Scalar s);
                static const func_t funcs[7][4] =
                {
                    {NppSet<CV_8U , 1, nppiSet_8u_C1R >::call, cv::gpu::setTo                          , cv::gpu::setTo                        , NppSet<CV_8U , 4, nppiSet_8u_C4R >::call},
                    {NppSet<CV_8S , 1, nppiSet_8s_C1R >::call, NppSet<CV_8S , 2, nppiSet_8s_C2R >::call, NppSet<CV_8S, 3, nppiSet_8s_C3R>::call, NppSet<CV_8S , 4, nppiSet_8s_C4R >::call},
                    {NppSet<CV_16U, 1, nppiSet_16u_C1R>::call, NppSet<CV_16U, 2, nppiSet_16u_C2R>::call, cv::gpu::setTo                        , NppSet<CV_16U, 4, nppiSet_16u_C4R>::call},
                    {NppSet<CV_16S, 1, nppiSet_16s_C1R>::call, NppSet<CV_16S, 2, nppiSet_16s_C2R>::call, cv::gpu::setTo                        , NppSet<CV_16S, 4, nppiSet_16s_C4R>::call},
                    {NppSet<CV_32S, 1, nppiSet_32s_C1R>::call, cv::gpu::setTo                          , cv::gpu::setTo                        , NppSet<CV_32S, 4, nppiSet_32s_C4R>::call},
                    {NppSet<CV_32F, 1, nppiSet_32f_C1R>::call, cv::gpu::setTo                          , cv::gpu::setTo                        , NppSet<CV_32F, 4, nppiSet_32f_C4R>::call},
                    {cv::gpu::setTo                          , cv::gpu::setTo                          , cv::gpu::setTo                        , cv::gpu::setTo                          }
                };

                CV_Assert(m.depth() <= CV_64F && m.channels() <= 4);

                if (m.depth() == CV_64F)
                {
                    if (!TargetArchs::builtWith(NATIVE_DOUBLE) || !DeviceInfo().supports(NATIVE_DOUBLE))
                        CV_Error(CV_StsUnsupportedFormat, "The device doesn't support double");
                }

                funcs[m.depth()][m.channels() - 1](m, s);
            }
            else
            {
                typedef void (*func_t)(GpuMat& src, Scalar s, const GpuMat& mask);
                static const func_t funcs[7][4] =
                {
                    {NppSetMask<CV_8U , 1, nppiSet_8u_C1MR >::call, cv::gpu::setTo, cv::gpu::setTo, NppSetMask<CV_8U , 4, nppiSet_8u_C4MR >::call},
                    {cv::gpu::setTo                               , cv::gpu::setTo, cv::gpu::setTo, cv::gpu::setTo                               },
                    {NppSetMask<CV_16U, 1, nppiSet_16u_C1MR>::call, cv::gpu::setTo, cv::gpu::setTo, NppSetMask<CV_16U, 4, nppiSet_16u_C4MR>::call},
                    {NppSetMask<CV_16S, 1, nppiSet_16s_C1MR>::call, cv::gpu::setTo, cv::gpu::setTo, NppSetMask<CV_16S, 4, nppiSet_16s_C4MR>::call},
                    {NppSetMask<CV_32S, 1, nppiSet_32s_C1MR>::call, cv::gpu::setTo, cv::gpu::setTo, NppSetMask<CV_32S, 4, nppiSet_32s_C4MR>::call},
                    {NppSetMask<CV_32F, 1, nppiSet_32f_C1MR>::call, cv::gpu::setTo, cv::gpu::setTo, NppSetMask<CV_32F, 4, nppiSet_32f_C4MR>::call},
                    {cv::gpu::setTo                               , cv::gpu::setTo, cv::gpu::setTo, cv::gpu::setTo                               }
                };

                CV_Assert(m.depth() <= CV_64F && m.channels() <= 4);

                if (m.depth() == CV_64F)
                {
                    if (!TargetArchs::builtWith(NATIVE_DOUBLE) || !DeviceInfo().supports(NATIVE_DOUBLE))
                        CV_Error(CV_StsUnsupportedFormat, "The device doesn't support double");
                }

                funcs[m.depth()][m.channels() - 1](m, s, mask);
            }
        }

        void mallocPitch(void** devPtr, size_t* step, size_t width, size_t height) const
        {
            cudaSafeCall( cudaMallocPitch(devPtr, step, width, height) );
        }

        void free(void* devPtr) const
        {
            cudaFree(devPtr);
        }
    };

    const GpuFuncTable* gpuFuncTable()
    {
        static CudaFuncTable funcTable;
        return &funcTable;
    }
}

#endif // HAVE_CUDA

void cv::gpu::GpuMat::upload(const Mat& m)
{
    CV_DbgAssert(!m.empty());

    create(m.size(), m.type());

    gpuFuncTable()->copy(m, *this);
}

void cv::gpu::GpuMat::download(Mat& m) const
{
    CV_DbgAssert(!empty());

    m.create(size(), type());

    gpuFuncTable()->copy(*this, m);
}

void cv::gpu::GpuMat::copyTo(GpuMat& m) const
{
    CV_DbgAssert(!empty());

    m.create(size(), type());

    gpuFuncTable()->copy(*this, m);
}

void cv::gpu::GpuMat::copyTo(GpuMat& mat, const GpuMat& mask) const
{
    if (mask.empty())
        copyTo(mat);
    else
    {
        mat.create(size(), type());

        gpuFuncTable()->copyWithMask(*this, mat, mask);
    }
}

void cv::gpu::GpuMat::convertTo(GpuMat& dst, int rtype, double alpha, double beta) const
{
    bool noScale = fabs(alpha - 1) < std::numeric_limits<double>::epsilon() && fabs(beta) < std::numeric_limits<double>::epsilon();

    if (rtype < 0)
        rtype = type();
    else
        rtype = CV_MAKETYPE(CV_MAT_DEPTH(rtype), channels());

    int sdepth = depth();
    int ddepth = CV_MAT_DEPTH(rtype);
    if (sdepth == ddepth && noScale)
    {
        copyTo(dst);
        return;
    }

    GpuMat temp;
    const GpuMat* psrc = this;
    if (sdepth != ddepth && psrc == &dst)
    {
        temp = *this;
        psrc = &temp;
    }

    dst.create(size(), rtype);

    if (noScale)
        gpuFuncTable()->convert(*psrc, dst);
    else
        gpuFuncTable()->convert(*psrc, dst, alpha, beta);
}

GpuMat& cv::gpu::GpuMat::setTo(Scalar s, const GpuMat& mask)
{
    CV_Assert(mask.empty() || mask.type() == CV_8UC1);
    CV_DbgAssert(!empty());

    gpuFuncTable()->setTo(*this, s, mask);

    return *this;
}

void cv::gpu::GpuMat::create(int _rows, int _cols, int _type)
{
    _type &= TYPE_MASK;

    if (rows == _rows && cols == _cols && type() == _type && data)
        return;

    if (data)
        release();

    CV_DbgAssert(_rows >= 0 && _cols >= 0);

    if (_rows > 0 && _cols > 0)
    {
        flags = Mat::MAGIC_VAL + _type;
        rows = _rows;
        cols = _cols;

        size_t esz = elemSize();

        void* devPtr;
        gpuFuncTable()->mallocPitch(&devPtr, &step, esz * cols, rows);

        // Single row must be continuous
        if (rows == 1)
            step = esz * cols;

        if (esz * cols == step)
            flags |= Mat::CONTINUOUS_FLAG;

        int64 _nettosize = static_cast<int64>(step) * rows;
        size_t nettosize = static_cast<size_t>(_nettosize);

        datastart = data = static_cast<uchar*>(devPtr);
        dataend = data + nettosize;

        refcount = static_cast<int*>(fastMalloc(sizeof(*refcount)));
        *refcount = 1;
    }
}

void cv::gpu::GpuMat::release()
{
    if (refcount && CV_XADD(refcount, -1) == 1)
    {
        fastFree(refcount);

        gpuFuncTable()->free(datastart);
    }

    data = datastart = dataend = 0;
    step = rows = cols = 0;
    refcount = 0;
}

////////////////////////////////////////////////////////////////////////
// Error handling

void cv::gpu::error(const char *error_string, const char *file, const int line, const char *func)
{
    int code = CV_GpuApiCallError;

    if (std::uncaught_exception())
    {
        const char* errorStr = cvErrorStr(code);
        const char* function = func ? func : "unknown function";

        std::cerr << "OpenCV Error: " << errorStr << "(" << error_string << ") in " << function << ", file " << file << ", line " << line;
        std::cerr.flush();
    }
    else
        cv::error( cv::Exception(code, error_string, func, file, line) );
}
