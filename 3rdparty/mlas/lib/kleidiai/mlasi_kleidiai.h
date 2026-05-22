//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: MIT
//

#pragma once

#include "../mlasi.h"
#include <iostream>

// Fix to ensure compatibility with MSVC build
#if defined(_MSC_VER)
  #define RESTRICT __restrict
#else
  #define RESTRICT __restrict__
#endif

// Logging macros.
#ifndef KLEIDIAI_DEBUG_LOGGING
#define KLEIDIAI_DEBUG_LOGGING 0
#endif
#ifndef KLEIDIAI_KERNEL_LOGGING
#define KLEIDIAI_KERNEL_LOGGING 0
#endif

#if KLEIDIAI_DEBUG_LOGGING ||KLEIDIAI_KERNEL_LOGGING
#define KLEIDIAI_LOG(tag, msg) \
    do { \
        std::cout << "[KLEIDIAI " << tag << "]: " << __FILE__ << " : " << __LINE__ << " : " << msg << std::endl; \
    } while(false)
#endif

// General logging. "tag" is expected to qualify the type of message.
#if KLEIDIAI_DEBUG_LOGGING
    // General debug messages.
    #define KLEIDIAI_DEBUG_LOG(msg) KLEIDIAI_LOG("DEBUG", msg)
#else
    #define KLEIDIAI_DEBUG_LOG(msg)
#endif

#if KLEIDIAI_KERNEL_LOGGING
    // Messages specifically written before a call to kai_run.
    // Note: In cases where a kernel is called in multiple threads, for example MlasTrySimpleParallel,
    // the output order can be inconsistient. The solution is to set the intra-node thread size to 1.
    // If using onnxruntime_perf_test this is done with "--x 1".
    #define KLEIDIAI_KERNEL_LOG(kernel_name) KLEIDIAI_LOG("KERNEL", kernel_name)
#else
    #define KLEIDIAI_KERNEL_LOG(msg)
#endif

namespace ArmKleidiAI {

// By default we should try for SME2 first before falling back to SME.
inline const bool UseSME2 = MLAS_CPUIDINFO::GetCPUIDInfo().HasArm_SME2();
inline const bool UseSME = MLAS_CPUIDINFO::GetCPUIDInfo().HasArm_SME();
inline const std::string_view vendor_name = MLAS_CPUIDINFO::GetCPUIDInfo().GetCPUVendor();

// Buffer packing routines.
//
size_t
MLASCALL
MlasGemmPackBSize(
    CBLAS_TRANSPOSE TransA,
    CBLAS_TRANSPOSE TransB,
    size_t N,
    size_t K
    );

bool
MLASCALL
MlasGemmPackB(
    CBLAS_TRANSPOSE TransA,
    CBLAS_TRANSPOSE TransB,
    size_t N,
    size_t K,
    const float* B,
    size_t ldb,
    void* PackedB
    );

bool
MLASCALL
MlasGemvBatch(
    CBLAS_TRANSPOSE TransA,
    CBLAS_TRANSPOSE TransB,
    size_t M,
    size_t N,
    size_t K,
    const MLAS_SGEMM_DATA_PARAMS* Data,
    size_t BatchSize
    );


bool
MLASCALL
MlasGemmBatch(
    CBLAS_TRANSPOSE TransA,
    CBLAS_TRANSPOSE TransB,
    size_t M,
    size_t N,
    size_t K,
    const MLAS_SGEMM_DATA_PARAMS* Data,
    size_t BatchSize,
    MLAS_THREADPOOL* ThreadPool
    );

#if defined(__aarch64__) && defined(__linux__)
size_t
MLASCALL
MlasSBGemmPackBSize(
    CBLAS_TRANSPOSE TransA,
    CBLAS_TRANSPOSE TransB,
    size_t N,
    size_t K
    );

bool
MLASCALL
MlasSBGemmPackB(
    CBLAS_TRANSPOSE TransA,
    CBLAS_TRANSPOSE TransB,
    size_t N,
    size_t K,
    const float* B,
    size_t ldb,
    void* PackedB
    );

bool
MLASCALL
MlasSBGemmBatch(
    CBLAS_TRANSPOSE TransA,
    CBLAS_TRANSPOSE TransB,
    size_t M,
    size_t N,
    size_t K,
    const MLAS_SBGEMM_DATA_PARAMS* Data,
    size_t BatchSize,
    MLAS_THREADPOOL* ThreadPool
    );
#endif

size_t
MLASCALL
MlasDynamicQGemmPackBSize(
    size_t N,
    size_t K
);

void
MLASCALL
MlasDynamicQGemmPackB(
    size_t N,
    size_t K,
    const int8_t* B,
    const float* Scales,
    const float* Bias,
    void* PackedB
);

//pack symmetric quantized B and dynamic quantized A
void
MLASCALL
MlasDynamicQGemmBatch(
    const MLAS_GEMM_DYN_QUANT_SHAPE_PARAMS& Shape,
    const MLAS_GEMM_DYN_QUANT_DATA_PARAMS* DataParams,
    const size_t BatchN,
    MLAS_THREADPOOL* ThreadPool
    );

bool
MLASCALL
MlasConvPrepare(MLAS_CONV_PARAMETERS* Parameters,
                size_t Dimensions,
                size_t BatchCount,
                size_t GroupCount,
                size_t InputChannels,
                const int64_t* InputShape,
                const int64_t* KernelShape,
                const int64_t* DilationShape,
                const int64_t* Padding,
                const int64_t* StrideShape,
                const int64_t* OutputShape,
                size_t FilterCount,
                const MLAS_ACTIVATION* Activation,
                size_t* WorkingBufferSize,
                float Beta,
                MLAS_THREADPOOL* ThreadPool);

bool
MLASCALL
MlasConv(
    const MLAS_CONV_PARAMETERS* Parameters,
    const float* Input,
    const float* Filter,
    const float* Bias,
    float* WorkingBuffer,
    float* Output,
    MLAS_THREADPOOL* ThreadPool
    );
}

/*++

Routine Description:

    This routine determines if a wraparound will occur when multiplying two size_t variables
    Uses __builtin_mul_overflow if available on the current system and if not falls back
    to a default implementation to check this wraparound.

Arguments:

    a - Supplies the first number to be muliplied.

    b - Supplies the second number to be muliplied.

    out - pointer to a size_t which acts as the return value in success cases.

Return Value:

    Returns false if the operation was successful
    Returns true if wraparound of size_t was detected

--*/
inline bool mul_overflow_size_t_builtin(size_t a, size_t b, size_t* out) {
#if defined(__has_builtin)
#  if __has_builtin(__builtin_mul_overflow)
    return __builtin_mul_overflow(a, b, out);
#  endif
#endif
    // Fallback to manual check if builtin not available
    if (b != 0 && a > SIZE_MAX / b) return true;
    if (out) *out = a * b;
    return false;
}
