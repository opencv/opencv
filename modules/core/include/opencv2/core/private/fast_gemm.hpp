// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// This file is modified from the ficus (https://github.com/vpisarev/ficus/blob/master/runtime/ficus/impl/gemm.impl.h).
// Here is the original license:
/*
    This file is a part of ficus language project.
    See ficus/LICENSE for the licensing terms
*/

#ifndef OPENCV_CORE_PRIVATE_FAST_GEMM_HPP
#define OPENCV_CORE_PRIVATE_FAST_GEMM_HPP

#ifndef __OPENCV_BUILD
#  error this is a private header which should not be used from outside of the OpenCV library
#endif

#include "opencv2/core.hpp"
#include <vector>

//! @cond IGNORED

namespace cv {

struct FastGemmOpt {
    bool use_avx;
    bool use_avx2;
    bool use_neon;
    bool use_lasx;
    bool use_rvv;
    bool multi_thread;

    FastGemmOpt() {
        use_avx = false;
        use_avx2 = false;
        use_neon = false;
        use_lasx = false;
        use_rvv = false;
        multi_thread = false;
    }

    void init() {
        use_avx = checkHardwareSupport(CPU_AVX);
        use_avx2 = checkHardwareSupport(CPU_AVX2);
        use_neon = checkHardwareSupport(CPU_NEON);
        use_lasx = checkHardwareSupport(CPU_LASX);
        use_rvv = checkHardwareSupport(CPU_RVV);
        multi_thread = true;
    }

    bool all() {
        return use_avx || use_avx2 || use_neon || use_lasx || use_rvv;
    }
};

namespace hal {

CV_EXPORTS int fastGemmMC(const FastGemmOpt &opt);
CV_EXPORTS int fastGemmNC(const FastGemmOpt &opt);
CV_EXPORTS int fastGemmKC(const FastGemmOpt &opt);
CV_EXPORTS int fastGemmNR(const FastGemmOpt &opt);

CV_EXPORTS size_t fastGemmPackBSize(size_t N, size_t K, const FastGemmOpt &opt);
CV_EXPORTS void fastGemmPackB(bool trans, size_t N, size_t K, const float *B, size_t ldb,
                              float *packed_B, const FastGemmOpt &opt);

CV_EXPORTS void fastGemm(bool trans_a, int M, int N, int K,
                         float alpha, const float *A, int lda,
                         const float *packed_B, float beta,
                         float *C, int ldc, const FastGemmOpt &opt);

CV_EXPORTS void fastGemm(bool trans_a, bool trans_b, int ma, int na, int mb, int nb,
                         float alpha, const float *A, int lda0, int lda1,
                         const float *B, int ldb0, int ldb1,
                         float beta, float *C, int ldc, const FastGemmOpt &opt);

CV_EXPORTS void fastGemmBatch(size_t batch, const size_t *A_offsets, const size_t *B_offsets, const size_t *C_offsets,
                              int M, int N, int K, float alpha, const float *A, int lda0, int lda1,
                              const float *B, int ldb0, int ldb1, float beta, float *C, int ldc,
                              const FastGemmOpt &opt);
CV_EXPORTS void fastGemmBatch(size_t batch, const size_t *A_offsets, const size_t *packed_B_offsets, const size_t *C_offsets,
                              int M, int N, int K, float alpha, const float *A, int lda0, int lda1,
                              const float *packed_B, float beta, float *C, int ldc, const FastGemmOpt &opt);

CV_EXPORTS bool fastGemmThinEligible(int M, int N, int K);
CV_EXPORTS size_t fastGemmThinPackBSize(int N, int K);
CV_EXPORTS void fastGemmThinPackB(int N, int K, const float* B, size_t ldb_K, size_t ldb_N, float* packed_B);
CV_EXPORTS void fastGemmThin(int M, int N, int K, float alpha,
                             const float* A, int lda0, int lda1,
                             const float* packed_B, float beta,
                             float* C, int ldc, bool multi_thread);

CV_EXPORTS void fastGemmPagedQK(const char *Q, const std::vector<const char *> &K, char *A,
                                size_t B, size_t T_q, size_t Nq, size_t N_k, size_t T_s, size_t D, size_t T_k,
                                float sm_scale, size_t esz, bool isQ3d, const FastGemmOpt &opt);
CV_EXPORTS void fastGemmPagedAV(const char *A, const std::vector<const char *> &V, char *Out,
                                size_t B, size_t T_a, size_t Nq, size_t N_k, size_t T_s, size_t D, size_t T_v,
                                size_t esz, bool canonical_layout, size_t packed_stride, const FastGemmOpt &opt);

CV_EXPORTS bool fastGemm32f(const float* src1, size_t src1_step, const float* src2, size_t src2_step,
                            float alpha, const float* src3, size_t src3_step, float beta,
                            float* dst, size_t dst_step, int m_a, int n_a, int n_d, int flags);

}} // cv::hal

//! @endcond

#endif // OPENCV_CORE_PRIVATE_FAST_GEMM_HPP
