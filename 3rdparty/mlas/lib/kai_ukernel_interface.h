//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: MIT
//

#pragma once

#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp_qsi4c32p_interface.h"

#include "kai/ukernels/matmul/matmul_clamp_f32_f32p_f32p/kai_matmul_clamp_f32_f32p_f32p_interface.h"

#include "kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p_interface.h"

#include "kai/ukernels/matmul/matmul_clamp_f32_bf16p_bf16p/kai_matmul_clamp_f32_bf16p_bf16p_interface.h"

#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp_qsi8cxp_interface.h"

#include "kai/ukernels/matmul/imatmul_clamp_f32_f32p_f32p/kai_imatmul_clamp_f32_f32p_f32p_interface.h"

// Wrapper type that carries a stable "name" alongside the KAI ukernel interface.
// This avoids needing to infer which underlying microkernel was selected from a function pointer.
template <typename UkernelFn>
struct KaiMatmulKernel {
    const char* name;
    UkernelFn ukernel;
};

// Wrapper for FP32 GEMM kernels where both LHS and RHS are pre-packed (common SGEMM path).
using KaiF32SgemmKernel = KaiMatmulKernel<kai_matmul_clamp_f32_f32p_f32p_ukernel>;

// Wrapper for FP32 kernels used for GEMV-style workloads (typically a single-row/skinny-M use case).
using KaiF32SgemvKernel = KaiMatmulKernel<kai_matmul_clamp_f32_f32_f32p_ukernel>;

// Wrapper for Qnbit GEMM kernels producing FP32 output.
using KaiQnbitGemmKernel = KaiMatmulKernel<kai_matmul_clamp_f32_qai8dxp_qsi4c32p_ukernel>;

// Wrapper for dynamic-quantized GEMM kernels producing FP32 output.
using KaiDynamicQGemmKernel = KaiMatmulKernel<kai_matmul_clamp_f32_qai8dxp_qsi8cxp_ukernel>;

// Wrapper for FP32 IMATMUL kernels used by the KleidiAI convolution implementation.
using KaiF32IMatmulKernel = KaiMatmulKernel<kai_imatmul_clamp_f32_f32p_f32p_ukernel>;

using KaiBF16SBgemmKernel = KaiMatmulKernel<kai_matmul_clamp_f32_bf16p_bf16p_ukernel>;

// Returns the selected Qnbit GEMM ukernel based on runtime CPU capabilities.
const KaiQnbitGemmKernel& GetKleidiAIGemmUKernel();

// Returns the selected Qnbit kernel used for GEMV-style workloads based on runtime CPU capabilities.
const KaiQnbitGemmKernel& GetKleidiAIGemvUKernel();

// Returns the selected dynamic-quantized GEMM ukernel based on runtime CPU capabilities and optional vendor selection.
const KaiDynamicQGemmKernel& GetKleidiAIQGemmUKernel();

// Returns the selected FP32 SGEMM ukernel based on runtime CPU capabilities and optional vendor selection.
const KaiF32SgemmKernel& GetKleidiAISGemmUKernel();

// Returns the selected FP32 kernel used for GEMV-style workloads based on runtime CPU capabilities.
const KaiF32SgemvKernel& GetKleidiAISGemvUKernel();

// Returns the selected FP32 IMATMUL ukernel used by the KleidiAI convolution implementation.
const KaiF32IMatmulKernel& GetKleidiAIF32IMatmulUKernel();

// Returns the selected BF16 SBGEMM ukernel used by the KleidiAI based on runtime CPU capabilities.
const KaiBF16SBgemmKernel& GetKleidiAISBGemmUKernel();