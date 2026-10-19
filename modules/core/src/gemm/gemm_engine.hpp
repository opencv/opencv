// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_CORE_SRC_GEMM_ENGINE_HPP
#define OPENCV_CORE_SRC_GEMM_ENGINE_HPP

namespace cv { namespace gemm_engine {

// The engine kernels exist on AArch64 only for now; other targets keep the existing GEMM code.
#if CV_NEON && CV_NEON_AARCH64
#define CV_GEMM_ENGINE_NEON 1
#else
#define CV_GEMM_ENGINE_NEON 0
#endif

// Block sizes of the 8x12 NEON kernel.
enum
{
    GEMM_MR = 8,
    GEMM_NR = 12,
    GEMM_MC = 144,
    GEMM_NC = 72,
    GEMM_KC = 384,
    GEMM_TASK_MACS = 1 << 17,   // smallest useful amount of work per task
    GEMM_SMALL_TASK_MACS = 1 << 16
};

enum PackedFormat
{
    PACKED_PLAIN = 1,  // op(B) as a dense k x n matrix
    PACKED_PANELS = 2, // W-column panels, each k x W row-major (W = 12 for NEON, 2*VL for SME)
    PACKED_HAL = 3     // written by a HAL hook, read only by that hook
};

// Every packed B buffer starts with this header; the payload follows at PACKED_HEADER_SIZE.
struct PackedBHeader
{
    unsigned magic;
    unsigned format;
    int n;
    int k;
    int width;  // panel width W for PACKED_PANELS
    int sme;    // 1 if packed for the SME kernels
};

enum
{
    PACKED_HEADER_SIZE = 64,
    PACKED_MAGIC = 0x42504743 // "CGPB"
};

static inline int roundUp(int a, int b) { return (a + b - 1) / b * b; }

// B operand: either a strided matrix (B(p, j) = ptr[p*ld0 + j*ld1]) or W-column panels over the full k.
struct BMatrix
{
    const float* ptr;
    size_t ld0, ld1;
    int panelWidth;  // 0 for a strided matrix
};

// Element (i, p) of A is A[i*lda0 + p*lda1]. Computes C = alpha*A*B + beta*C for `batch`
// products placed by element offsets (B offsets are ignored for panel B).
void gemmBatch(int batch, const size_t* A_offsets, const size_t* B_offsets, const size_t* C_offsets,
               int m, int n, int k, float alpha, const float* A, size_t lda0, size_t lda1,
               const BMatrix& B, float beta, float* C, size_t ldc, bool multiThread, bool sme);

// Payload size and packing for the engine's own format (panels, or plain on other targets).
size_t packedPayloadSize(int n, int k);
void packB(int n, int k, const float* B, size_t ldb0, size_t ldb1, void* packedB, bool multiThread = true);

// Checks the header of a packed buffer against the expected sizes.
const PackedBHeader& checkPackedB(const void* packedB, int n, int k);

// True when the SME kernels are built, the CPU has SME and nobody turned it off.
bool smeEnabled();

// Floats per SME vector when SME is enabled, else 0.
int smeVectorLength();

// cv::gemm route for fp32; returns false when the old code path should run instead.
bool gemm32f(const float* src1, size_t src1_step, const float* src2, size_t src2_step,
             float alpha, const float* src3, size_t src3_step, float beta, float* dst, size_t dst_step,
             int m_a, int n_a, int n_d, int flags);

}} // cv::gemm_engine

#endif // OPENCV_CORE_SRC_GEMM_ENGINE_HPP
