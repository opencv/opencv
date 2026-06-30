/*
 * cuBLAS/cuFFT to hipBLAS/hipFFT compatibility shim for the cudaarithm module.
 *
 * cudaarithm::gemm calls cuBLAS and cudaarithm::dft/convolve call cuFFT. On the
 * AMD ROCm path those map onto hipBLAS and hipFFT, whose APIs mirror the CUDA
 * libraries closely. This header is a host-only shim (included from the module
 * precompiled header on the WITH_HIP path) that aliases the cuBLAS/cuFFT
 * spellings the module uses to their hip* equivalents, so arithm.cpp compiles
 * unchanged while the upstream NVIDIA build is left untouched.
 *
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 *
 * \author Jeff Daily <jeff.daily@amd.com>
 */

#pragma once

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)

#include <hip/hip_complex.h>

// --- cuBLAS ---
#ifdef HAVE_CUBLAS
#include <hipblas/hipblas.h>

// types / handles
#define cublasHandle_t      hipblasHandle_t
#define cublasStatus_t      hipblasStatus_t
#define cublasOperation_t   hipblasOperation_t
#define cuComplex           hipComplex
#define cuDoubleComplex     hipDoubleComplex
#define make_cuComplex      make_hipComplex
#define make_cuDoubleComplex make_hipDoubleComplex

// status / op / pointer-mode enums
#define CUBLAS_STATUS_SUCCESS          HIPBLAS_STATUS_SUCCESS
#define CUBLAS_STATUS_NOT_INITIALIZED  HIPBLAS_STATUS_NOT_INITIALIZED
#define CUBLAS_STATUS_ALLOC_FAILED     HIPBLAS_STATUS_ALLOC_FAILED
#define CUBLAS_STATUS_INVALID_VALUE    HIPBLAS_STATUS_INVALID_VALUE
#define CUBLAS_STATUS_ARCH_MISMATCH    HIPBLAS_STATUS_ARCH_MISMATCH
#define CUBLAS_STATUS_MAPPING_ERROR    HIPBLAS_STATUS_MAPPING_ERROR
#define CUBLAS_STATUS_EXECUTION_FAILED HIPBLAS_STATUS_EXECUTION_FAILED
#define CUBLAS_STATUS_INTERNAL_ERROR   HIPBLAS_STATUS_INTERNAL_ERROR
#define CUBLAS_OP_N                    HIPBLAS_OP_N
#define CUBLAS_OP_T                    HIPBLAS_OP_T
#define CUBLAS_POINTER_MODE_HOST       HIPBLAS_POINTER_MODE_HOST

// runtime API (hipBLAS drops the _v2 suffix)
#define cublasCreate_v2          hipblasCreate
#define cublasDestroy_v2         hipblasDestroy
#define cublasSetStream_v2       hipblasSetStream
#define cublasSetPointerMode_v2  hipblasSetPointerMode
#define cublasSgemm_v2           hipblasSgemm
#define cublasDgemm_v2           hipblasDgemm
#define cublasCgemm_v2           hipblasCgemm
#define cublasZgemm_v2           hipblasZgemm
#endif // HAVE_CUBLAS

// --- cuFFT ---
#ifdef HAVE_CUFFT
#include <hipfft/hipfft.h>

#define cufftHandle    hipfftHandle
#define cufftType      hipfftType
#define cufftComplex   hipfftComplex
#define cufftReal      hipfftReal

#define CUFFT_SUCCESS         HIPFFT_SUCCESS
#define CUFFT_INVALID_PLAN    HIPFFT_INVALID_PLAN
#define CUFFT_ALLOC_FAILED    HIPFFT_ALLOC_FAILED
#define CUFFT_INVALID_TYPE    HIPFFT_INVALID_TYPE
#define CUFFT_INVALID_VALUE   HIPFFT_INVALID_VALUE
#define CUFFT_INTERNAL_ERROR  HIPFFT_INTERNAL_ERROR
#define CUFFT_EXEC_FAILED     HIPFFT_EXEC_FAILED
#define CUFFT_SETUP_FAILED    HIPFFT_SETUP_FAILED
#define CUFFT_INVALID_SIZE    HIPFFT_INVALID_SIZE
#define CUFFT_UNALIGNED_DATA  HIPFFT_UNALIGNED_DATA
#define CUFFT_FORWARD         HIPFFT_FORWARD
#define CUFFT_INVERSE         HIPFFT_BACKWARD
#define CUFFT_R2C             HIPFFT_R2C
#define CUFFT_C2R             HIPFFT_C2R
#define CUFFT_C2C             HIPFFT_C2C

#define cufftPlan1d    hipfftPlan1d
#define cufftPlan2d    hipfftPlan2d
#define cufftDestroy   hipfftDestroy
#define cufftSetStream hipfftSetStream
#define cufftExecC2C   hipfftExecC2C
#define cufftExecC2R   hipfftExecC2R
#define cufftExecR2C   hipfftExecR2C
#endif // HAVE_CUFFT

#endif // __HIP_PLATFORM_AMD__
