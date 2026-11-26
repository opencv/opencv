/*
 * CUDA to HIP compatibility shim for the OpenCV CUDA modules.
 *
 * Force included into every HIP translation unit (see OpenCVDetectHIP.cmake).
 * It pulls in the HIP runtime and aliases the CUDA runtime spellings used by
 * the OpenCV device code to their HIP equivalents, so the .cu sources compile
 * unchanged under hipcc while the upstream NVIDIA CUDA build is left untouched
 * (this header is only included on the WITH_HIP path).
 *
 * Only the symbols the OpenCV CUDA modules actually use are mapped. The aliases
 * are the standard hipify cuda*->hip* substitutions; the handful that do not
 * follow the literal prefix swap (device attribute enums, descriptor and
 * property types) are listed explicitly.
 *
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 *
 * \author Jeff Daily <jeff.daily@amd.com>
 */

#pragma once

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Make the legacy __CUDA_ARCH__ predicate true in the device pass so the cudev
// shuffle and reduction fast paths (CV_CUDEV_ARCH >= 300) are selected rather
// than collapsing to empty no-op fallbacks. The PTX inline-asm SIMD paths that
// also key on CV_CUDEV_ARCH are guarded out separately for HIP at their use site.
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__CUDA_ARCH__)
#define __CUDA_ARCH__ 350
#endif

// --- runtime types ---
#define cudaError_t            hipError_t
#define cudaStream_t           hipStream_t
#define cudaEvent_t            hipEvent_t
#define cudaDeviceProp         hipDeviceProp_t
#define cudaFuncAttributes     hipFuncAttributes
#define cudaStreamCallback_t   hipStreamCallback_t

// --- error codes ---
#define cudaSuccess                 hipSuccess
#define cudaErrorNotReady           hipErrorNotReady
#define cudaErrorNoDevice           hipErrorNoDevice
#define cudaErrorInsufficientDriver hipErrorInsufficientDriver

// --- memcpy kinds / host alloc / register flags ---
#define cudaMemcpyHostToDevice     hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost     hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice   hipMemcpyDeviceToDevice
#define cudaHostAllocDefault       hipHostMallocDefault
#define cudaHostAllocMapped        hipHostMallocMapped
#define cudaHostAllocWriteCombined hipHostMallocWriteCombined
#define cudaHostRegisterPortable   hipHostRegisterPortable
#define cudaStreamNonBlocking      hipStreamNonBlocking

// --- device attribute enums (non literal swap) ---
#define cudaDevAttrClockRate         hipDeviceAttributeClockRate
#define cudaDevAttrComputeMode       hipDeviceAttributeComputeMode
#define cudaDevAttrKernelExecTimeout hipDeviceAttributeKernelExecTimeout
#define cudaDevAttrMemoryClockRate   hipDeviceAttributeMemoryClockRate

// --- runtime API ---
#define cudaMalloc                hipMalloc
#define cudaMallocPitch           hipMallocPitch
#define cudaMallocHost            hipHostMalloc
#define cudaFree                  hipFree
#define cudaFreeHost              hipHostFree
#define cudaHostAlloc             hipHostMalloc
#define cudaHostRegister          hipHostRegister
#define cudaHostUnregister        hipHostUnregister
#define cudaHostGetDevicePointer  hipHostGetDevicePointer
#define cudaMemcpy                hipMemcpy
#define cudaMemcpyAsync           hipMemcpyAsync
#define cudaMemcpy2D              hipMemcpy2D
#define cudaMemcpy2DAsync         hipMemcpy2DAsync
#define cudaMemset2D              hipMemset2D
#define cudaMemset2DAsync         hipMemset2DAsync
#define cudaMemset                hipMemset
#define cudaMemsetAsync           hipMemsetAsync
#define cudaMemGetInfo            hipMemGetInfo
#define cudaGetLastError          hipGetLastError
#define cudaGetErrorString        hipGetErrorString
#define cudaGetDevice             hipGetDevice
#define cudaSetDevice             hipSetDevice
#define cudaGetDeviceCount        hipGetDeviceCount
#define cudaGetDeviceProperties   hipGetDeviceProperties
#define cudaDeviceGetAttribute    hipDeviceGetAttribute
#define cudaDeviceReset           hipDeviceReset
#define cudaDeviceSynchronize     hipDeviceSynchronize
#define cudaDriverGetVersion      hipDriverGetVersion
#define cudaRuntimeGetVersion     hipRuntimeGetVersion
#define cudaFuncGetAttributes     hipFuncGetAttributes
// hipFuncSetCacheConfig takes a const void*; a __global__ function-template id
// (e.g. kernel<T>) does not implicitly convert to it the way nvcc accepts. A
// function-like macro cannot wrap the call because the template argument list
// carries commas, so route through a typed helper that accepts any kernel
// pointer. Cache config is advisory and ignored on AMD; keep it for parity.
namespace cv { namespace cuda { namespace detail {
template <typename Func>
static inline hipError_t hipFuncSetCacheConfigT(Func func, hipFuncCache_t config)
{
    return ::hipFuncSetCacheConfig(reinterpret_cast<const void*>(func), config);
}
template <typename Func>
static inline hipError_t hipFuncSetAttributeT(Func func, hipFuncAttribute attr, int value)
{
    return ::hipFuncSetAttribute(reinterpret_cast<const void*>(func), attr, value);
}
}}}
#define cudaFuncSetCacheConfig    cv::cuda::detail::hipFuncSetCacheConfigT
#define cudaFuncSetAttribute      cv::cuda::detail::hipFuncSetAttributeT
#define cudaFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize
#define cudaFuncCachePreferL1     hipFuncCachePreferL1
#define cudaFuncCachePreferShared hipFuncCachePreferShared
#define cudaDeviceGetTexture1DLinearMaxWidth hipDeviceGetTexture1DLinearMaxWidth

// --- streams / events ---
#define cudaStreamCreate          hipStreamCreate
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamDestroy         hipStreamDestroy
#define cudaStreamQuery           hipStreamQuery
#define cudaStreamSynchronize     hipStreamSynchronize
#define cudaStreamWaitEvent       hipStreamWaitEvent
#define cudaStreamAddCallback     hipStreamAddCallback
#define cudaEventCreate           hipEventCreate
#define cudaEventCreateWithFlags  hipEventCreateWithFlags
#define cudaEventDestroy          hipEventDestroy
#define cudaEventQuery            hipEventQuery
#define cudaEventRecord           hipEventRecord
#define cudaEventSynchronize      hipEventSynchronize
#define cudaEventElapsedTime      hipEventElapsedTime

// --- texture object API ---
#define cudaTextureObject_t       hipTextureObject_t
#define cudaResourceDesc          hipResourceDesc
#define cudaTextureDesc           hipTextureDesc
#define cudaChannelFormatDesc     hipChannelFormatDesc
#define cudaCreateChannelDesc     hipCreateChannelDesc
#define cudaCreateTextureObject   hipCreateTextureObject
#define cudaDestroyTextureObject  hipDestroyTextureObject
#define cudaResourceTypeLinear    hipResourceTypeLinear
#define cudaResourceTypePitch2D   hipResourceTypePitch2D
#define cudaTextureAddressMode    hipTextureAddressMode
#define cudaTextureFilterMode     hipTextureFilterMode
#define cudaTextureReadMode       hipTextureReadMode
#define cudaAddressModeClamp      hipAddressModeClamp
#define cudaAddressModeWrap       hipAddressModeWrap
#define cudaAddressModeMirror     hipAddressModeMirror
#define cudaAddressModeBorder     hipAddressModeBorder
#define cudaFilterModePoint       hipFilterModePoint
#define cudaFilterModeLinear      hipFilterModeLinear
#define cudaReadModeElementType   hipReadModeElementType
#define cudaReadModeNormalizedFloat hipReadModeNormalizedFloat

// --- constant memory ---
#define cudaMemcpyToSymbol        hipMemcpyToSymbol
#define cudaMemcpyToSymbolAsync   hipMemcpyToSymbolAsync
#define cudaMemcpyFromSymbol      hipMemcpyFromSymbol
#define cudaMemcpyFromSymbolAsync hipMemcpyFromSymbolAsync
#define cudaGetSymbolAddress      hipGetSymbolAddress

#endif // __HIP_PLATFORM_AMD__
