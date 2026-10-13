//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: MIT
//

// OpenCV: trimmed to the FP32 kernels we use, so quantized/BF16 kernels are not linked.

#include "mlasi.h"

#include "kleidiai/mlasi_kleidiai.h"

#include "kai_ukernel_interface.h"

// SME kernels
//   GEMM
#include "kai/ukernels/matmul/matmul_clamp_f32_f32p_f32p/kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa.h"
//   GEMV
#include "kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p2vlx1b_1x8vl_sme_mla.h"
//   IMATMUL
#include "kai/ukernels/matmul/imatmul_clamp_f32_f32p_f32p/kai_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa.h"

// SME2 kernels
//   GEMM
#include "kai/ukernels/matmul/matmul_clamp_f32_f32p_f32p/kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa.h"
//   GEMV
#include "kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla.h"
//   IMATMUL
#include "kai/ukernels/matmul/imatmul_clamp_f32_f32p_f32p/kai_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme2_mopa.h"

// Kernel table for GEMM (packed A and B).
#define KAI_WRAP_UKERNEL_RUN_MATMUL_11(STEM)                                                             \
    {                                                                                                    \
        "kai_run_" #STEM,                                                                                \
        {kai_get_m_step_##STEM,                                                                          \
         kai_get_n_step_##STEM,                                                                          \
         kai_get_mr_##STEM,                                                                              \
         kai_get_nr_##STEM,                                                                              \
         kai_get_kr_##STEM,                                                                              \
         kai_get_sr_##STEM,                                                                              \
         kai_get_lhs_packed_offset_##STEM,                                                               \
         kai_get_rhs_packed_offset_##STEM,                                                               \
         kai_get_dst_offset_##STEM,                                                                      \
         kai_get_dst_size_##STEM,                                                                        \
         kai_run_##STEM}                                                                                 \
    }

// Kernel table for indirect GEMM (convolution).
#define KAI_WRAP_UKERNEL_RUN_IMATMUL_PACKED_7(STEM)                                                      \
    {                                                                                                    \
        "kai_run_" #STEM,                                                                                \
        {kai_get_m_step_##STEM,                                                                          \
         kai_get_n_step_##STEM,                                                                          \
         kai_get_lhs_packed_offset_##STEM,                                                               \
         kai_get_rhs_packed_offset_##STEM,                                                               \
         kai_get_dst_offset_##STEM,                                                                      \
         kai_get_dst_size_##STEM,                                                                        \
         kai_run_##STEM}                                                                                 \
    }

// Kernel table for GEMV (unpacked A).
#define KAI_WRAP_UKERNEL_RUN_MATMUL_10_LHS_OFFSET(STEM)                                                   \
    {                                                                                                     \
        "kai_run_" #STEM,                                                                                 \
        {kai_get_m_step_##STEM,                                                                           \
         kai_get_n_step_##STEM,                                                                           \
         kai_get_nr_##STEM,                                                                               \
         kai_get_kr_##STEM,                                                                               \
         kai_get_sr_##STEM,                                                                               \
         kai_get_lhs_offset_##STEM,                                                                       \
         kai_get_rhs_packed_offset_##STEM,                                                                \
         kai_get_dst_offset_##STEM,                                                                       \
         kai_get_dst_size_##STEM,                                                                         \
         kai_run_##STEM}                                                                                  \
    }

const KaiF32SgemmKernel sgemm_gemm_sme =
    KAI_WRAP_UKERNEL_RUN_MATMUL_11(matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa);

const KaiF32SgemmKernel sgemm_gemm_sme2 =
    KAI_WRAP_UKERNEL_RUN_MATMUL_11(matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa);

const KaiF32SgemvKernel sgemm_gemv_sme =
    KAI_WRAP_UKERNEL_RUN_MATMUL_10_LHS_OFFSET(matmul_clamp_f32_f32_f32p2vlx1b_1x8vl_sme_mla);

const KaiF32SgemvKernel sgemm_gemv_sme2 =
    KAI_WRAP_UKERNEL_RUN_MATMUL_10_LHS_OFFSET(matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla);

const KaiF32IMatmulKernel imatmul_conv_sme =
    KAI_WRAP_UKERNEL_RUN_IMATMUL_PACKED_7(imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa);

const KaiF32IMatmulKernel imatmul_conv_sme2 =
    KAI_WRAP_UKERNEL_RUN_IMATMUL_PACKED_7(imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme2_mopa);

const KaiF32SgemmKernel& GetKleidiAISGemmUKernel() {
    if (MLAS_CPUIDINFO::GetCPUIDInfo().HasArm_SME2()) {
        return sgemm_gemm_sme2;
    } else {
        return sgemm_gemm_sme;
    }
}

const KaiF32SgemvKernel& GetKleidiAISGemvUKernel() {
    if (MLAS_CPUIDINFO::GetCPUIDInfo().HasArm_SME2()) {
        return sgemm_gemv_sme2;
    } else {
        return sgemm_gemv_sme;
    }
}

const KaiF32IMatmulKernel& GetKleidiAIF32IMatmulUKernel() {
    if (MLAS_CPUIDINFO::GetCPUIDInfo().HasArm_SME2()) {
        return imatmul_conv_sme2;
    } else {
        return imatmul_conv_sme;
    }
}
