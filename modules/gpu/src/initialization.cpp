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
using namespace cv::gpu;


namespace 
{
    // Compares value to set using the given comparator. Returns true if
    // there is at least one element x in the set satisfying to: x cmp value
    // predicate.
    template <typename Comparer>
    bool compareToSet(const std::string& set_as_str, int value, Comparer cmp)
    {
        if (set_as_str.find_first_not_of(" ") == string::npos)
            return false;

        std::stringstream stream(set_as_str);
        int cur_value;

        while (!stream.eof())
        {
            stream >> cur_value;
            if (cmp(cur_value, value))
                return true;
        }

        return false;
    }
}


bool cv::gpu::TargetArchs::builtWith(cv::gpu::FeatureSet feature_set)
{
#if defined (HAVE_CUDA)
    return ::compareToSet(CUDA_ARCH_FEATURES, feature_set, std::greater_equal<int>());
#else
	(void)feature_set;
	return false;
#endif
}


bool cv::gpu::TargetArchs::has(int major, int minor)
{
    return hasPtx(major, minor) || hasBin(major, minor);
}


bool cv::gpu::TargetArchs::hasPtx(int major, int minor)
{
#if defined (HAVE_CUDA)
    return ::compareToSet(CUDA_ARCH_PTX, major * 10 + minor, std::equal_to<int>());
#else
	(void)major;
	(void)minor;
	return false;
#endif
}


bool cv::gpu::TargetArchs::hasBin(int major, int minor)
{
#if defined (HAVE_CUDA)
    return ::compareToSet(CUDA_ARCH_BIN, major * 10 + minor, std::equal_to<int>());
#else
	(void)major;
	(void)minor;
	return false;
#endif
}


bool cv::gpu::TargetArchs::hasEqualOrLessPtx(int major, int minor)
{
#if defined (HAVE_CUDA)
    return ::compareToSet(CUDA_ARCH_PTX, major * 10 + minor, 
                     std::less_equal<int>());
#else
	(void)major;
	(void)minor;
	return false;
#endif
}


bool cv::gpu::TargetArchs::hasEqualOrGreater(int major, int minor)
{
    return hasEqualOrGreaterPtx(major, minor) ||
           hasEqualOrGreaterBin(major, minor);
}


bool cv::gpu::TargetArchs::hasEqualOrGreaterPtx(int major, int minor)
{
#if defined (HAVE_CUDA)
    return ::compareToSet(CUDA_ARCH_PTX, major * 10 + minor, 
                     std::greater_equal<int>());
#else
	(void)major;
	(void)minor;
	return false;
#endif
}


bool cv::gpu::TargetArchs::hasEqualOrGreaterBin(int major, int minor)
{
#if defined (HAVE_CUDA)
    return ::compareToSet(CUDA_ARCH_BIN, major * 10 + minor, 
                     std::greater_equal<int>());
#else
	(void)major;
	(void)minor;
	return false;
#endif
}


#if !defined (HAVE_CUDA)

int cv::gpu::getCudaEnabledDeviceCount() { return 0; }
void cv::gpu::setDevice(int) { throw_nogpu(); } 
int cv::gpu::getDevice() { throw_nogpu(); return 0; }
void cv::gpu::resetDevice() { throw_nogpu(); }
size_t cv::gpu::DeviceInfo::freeMemory() const { throw_nogpu(); return 0; }
size_t cv::gpu::DeviceInfo::totalMemory() const { throw_nogpu(); return 0; }
bool cv::gpu::DeviceInfo::supports(cv::gpu::FeatureSet) const { throw_nogpu(); return false; }
bool cv::gpu::DeviceInfo::isCompatible() const { throw_nogpu(); return false; }
void cv::gpu::DeviceInfo::query() { throw_nogpu(); }
void cv::gpu::DeviceInfo::queryMemory(size_t&, size_t&) const { throw_nogpu(); }
void cv::gpu::printCudaDeviceInfo(int) { throw_nogpu(); }
void cv::gpu::printShortCudaDeviceInfo(int) { throw_nogpu(); }

#else /* !defined (HAVE_CUDA) */

int cv::gpu::getCudaEnabledDeviceCount()
{
    int count;
    cudaError_t error = cudaGetDeviceCount( &count );

    if (error == cudaErrorInsufficientDriver)
        return -1;

    if (error == cudaErrorNoDevice)
        return 0;

    cudaSafeCall(error);
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


size_t cv::gpu::DeviceInfo::freeMemory() const
{
    size_t free_memory, total_memory;
    queryMemory(free_memory, total_memory);
    return free_memory;
}


size_t cv::gpu::DeviceInfo::totalMemory() const
{
    size_t free_memory, total_memory;
    queryMemory(free_memory, total_memory);
    return total_memory;
}


bool cv::gpu::DeviceInfo::supports(cv::gpu::FeatureSet feature_set) const
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
    cudaDeviceProp prop;
    cudaSafeCall(cudaGetDeviceProperties(&prop, device_id_));
    name_ = prop.name;
    multi_processor_count_ = prop.multiProcessorCount;
    majorVersion_ = prop.major;
    minorVersion_ = prop.minor;
}


void cv::gpu::DeviceInfo::queryMemory(size_t& free_memory, size_t& total_memory) const
{
    int prev_device_id = getDevice();
    if (prev_device_id != device_id_)
        setDevice(device_id_);

    cudaSafeCall(cudaMemGetInfo(&free_memory, &total_memory));

    if (prev_device_id != device_id_)
        setDevice(prev_device_id);
}

namespace
{
    template <class T> void getCudaAttribute(T *attribute, CUdevice_attribute device_attribute, int device)
    {
        *attribute = T();
        CUresult error = CUDA_SUCCESS;// = cuDeviceGetAttribute( attribute, device_attribute, device ); why link erros under ubuntu??
        if( CUDA_SUCCESS == error )
            return;       
 
        printf("Driver API error = %04d\n", error);
        cv::gpu::error("driver API error", __FILE__, __LINE__);   
    }
 
    int convertSMVer2Cores(int major, int minor)
    {
        // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
        typedef struct {
            int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
            int Cores;
        } SMtoCores;
 
        SMtoCores gpuArchCoresPerSM[] =  { { 0x10,  8 }, { 0x11,  8 }, { 0x12,  8 }, { 0x13,  8 }, { 0x20, 32 }, { 0x21, 48 }, { -1, -1 } };
 
        int index = 0;
        while (gpuArchCoresPerSM[index].SM != -1)
        {
            if (gpuArchCoresPerSM[index].SM == ((major << 4) + minor) )
                return gpuArchCoresPerSM[index].Cores;
            index++;
        }
        printf("MapSMtoCores undefined SMversion %d.%d!\n", major, minor);
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
        printf("  (%2d) Multiprocessors x (%2d) CUDA Cores/MP:     %d CUDA Cores\n",
            prop.multiProcessorCount, convertSMVer2Cores(prop.major, prop.minor),
            convertSMVer2Cores(prop.major, prop.minor) * prop.multiProcessorCount);
        printf("  GPU Clock Speed:                               %.2f GHz\n", prop.clockRate * 1e-6f);
 
        // This is not available in the CUDA Runtime API, so we make the necessary calls the driver API to support this for output
        int memoryClock, memBusWidth, L2CacheSize;
        getCudaAttribute<int>( &memoryClock, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, dev );       
        getCudaAttribute<int>( &memBusWidth, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, dev );               
        getCudaAttribute<int>( &L2CacheSize, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, dev );
 
        printf("  Memory Clock rate:                             %.2f Mhz\n", memoryClock * 1e-3f);
        printf("  Memory Bus Width:                              %d-bit\n", memBusWidth);
        if (L2CacheSize)
            printf("  L2 Cache Size:                                 %d bytes\n", L2CacheSize);
       
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
        printf(", sm_%d%d%s, %d cores", prop.major, prop.minor, arch_str, convertSMVer2Cores(prop.major, prop.minor) * prop.multiProcessorCount);               
        printf(", Driver/Runtime ver.%d.%d/%d.%d\n", driverVersion/1000, driverVersion%100, runtimeVersion/1000, runtimeVersion%100);
    }
    fflush(stdout);
}

#endif

