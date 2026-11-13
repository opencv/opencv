// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_CORE_SRC_ACCEL_GPU_RUNTIME_HPP
#define OPENCV_CORE_SRC_ACCEL_GPU_RUNTIME_HPP

// CUDA and HIP backends are one source: HIP when CV_ACCEL_GPU_HIP is defined, CUDA otherwise.
// CV_GPU(X) is cudaX or hipX; names that differ by more than the prefix are spelled out here.

#ifdef CV_ACCEL_GPU_HIP
#  include <hip/hip_runtime_api.h>
#  define CV_GPU(name) hip##name
#  define CV_GPU_NS hip_impl
#  define CV_GPU_RUNTIME "hip"
#  define CV_GPU_ERROR_OUT_OF_MEMORY hipErrorOutOfMemory
#  define CV_GPU_ATTR_MEMORY_POOLS hipDeviceAttributeMemoryPoolsSupported
#  define CV_GPU_HOST_ALLOC hipHostMalloc
#  define CV_GPU_HOST_FREE hipHostFree
#  define CV_GPU_HOST_ALLOC_PORTABLE hipHostMallocPortable
#  define CV_GPU_HOST_ALLOC_MAPPED hipHostMallocMapped
typedef hipDeviceProp_t cvGpuDeviceProp;
typedef hipDeviceAttribute_t cvGpuDeviceAttr;
#else
#  include <cuda_runtime.h>
#  define CV_GPU(name) cuda##name
#  define CV_GPU_NS cuda_impl
#  define CV_GPU_RUNTIME "cuda"
#  define CV_GPU_ERROR_OUT_OF_MEMORY cudaErrorMemoryAllocation
#  define CV_GPU_ATTR_MEMORY_POOLS cudaDevAttrMemoryPoolsSupported
#  define CV_GPU_HOST_ALLOC cudaHostAlloc
#  define CV_GPU_HOST_FREE cudaFreeHost
#  define CV_GPU_HOST_ALLOC_PORTABLE cudaHostAllocPortable
#  define CV_GPU_HOST_ALLOC_MAPPED cudaHostAllocMapped
typedef cudaDeviceProp cvGpuDeviceProp;
typedef cudaDeviceAttr cvGpuDeviceAttr;
#endif

#endif // OPENCV_CORE_SRC_ACCEL_GPU_RUNTIME_HPP
