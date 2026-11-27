/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Copyright (C) 2015, Itseez Inc., all rights reserved.
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

#ifndef OPENCV_HAL_HPP
#define OPENCV_HAL_HPP

#include "opencv2/core/cvdef.h"
#include "opencv2/core/cvstd.hpp"
#include "opencv2/core/hal/interface.h"

namespace cv { namespace hal {

//! @addtogroup core_hal_functions
//! @{

CV_EXPORTS int normHamming(const uchar* a, int n);
CV_EXPORTS int normHamming(const uchar* a, const uchar* b, int n);

CV_EXPORTS int normHamming(const uchar* a, int n, int cellSize);
CV_EXPORTS int normHamming(const uchar* a, const uchar* b, int n, int cellSize);

CV_EXPORTS int LU32f(float* A, size_t astep, int m, float* b, size_t bstep, int n);
CV_EXPORTS int LU64f(double* A, size_t astep, int m, double* b, size_t bstep, int n);
CV_EXPORTS bool Cholesky32f(float* A, size_t astep, int m, float* b, size_t bstep, int n);
CV_EXPORTS bool Cholesky64f(double* A, size_t astep, int m, double* b, size_t bstep, int n);
CV_EXPORTS void SVD32f(float* At, size_t astep, float* W, float* U, size_t ustep, float* Vt, size_t vstep, int m, int n, int flags);
CV_EXPORTS void SVD64f(double* At, size_t astep, double* W, double* U, size_t ustep, double* Vt, size_t vstep, int m, int n, int flags);
CV_EXPORTS int QR32f(float* A, size_t astep, int m, int n, int k, float* b, size_t bstep, float* hFactors);
CV_EXPORTS int QR64f(double* A, size_t astep, int m, int n, int k, double* b, size_t bstep, double* hFactors);

CV_EXPORTS void gemm32f(const float* src1, size_t src1_step, const float* src2, size_t src2_step,
                        float alpha, const float* src3, size_t src3_step, float beta, float* dst, size_t dst_step,
                        int m_a, int n_a, int n_d, int flags);
CV_EXPORTS void gemm64f(const double* src1, size_t src1_step, const double* src2, size_t src2_step,
                        double alpha, const double* src3, size_t src3_step, double beta, double* dst, size_t dst_step,
                        int m_a, int n_a, int n_d, int flags);
CV_EXPORTS void gemm32fc(const float* src1, size_t src1_step, const float* src2, size_t src2_step,
                        float alpha, const float* src3, size_t src3_step, float beta, float* dst, size_t dst_step,
                        int m_a, int n_a, int n_d, int flags);
CV_EXPORTS void gemm64fc(const double* src1, size_t src1_step, const double* src2, size_t src2_step,
                        double alpha, const double* src3, size_t src3_step, double beta, double* dst, size_t dst_step,
                        int m_a, int n_a, int n_d, int flags);

/** @brief Returns the size in bytes of a buffer for gemm32fPackB.
@param n number of columns of op(B)
@param k number of rows of op(B)
@param flags CV_HAL_GEMM_2_T if B is stored as an n x k matrix
*/
CV_EXPORTS size_t gemm32fPackBSize(int n, int k, int flags);

/** @brief Packs op(B) once so that gemm32fPacked can reuse it.
@param B matrix B, k x n (or n x k with CV_HAL_GEMM_2_T)
@param B_step bytes between rows of B
@param n number of columns of op(B)
@param k number of rows of op(B)
@param flags CV_HAL_GEMM_2_T if B is stored as an n x k matrix, CV_HAL_GEMM_SINGLE_THREAD
@param packedB output buffer of gemm32fPackBSize(n, k, flags) bytes
*/
CV_EXPORTS void gemm32fPackB(const float* B, size_t B_step, int n, int k, int flags, void* packedB);

/** @brief Computes C = alpha*op(A)*B + beta*C with B packed by gemm32fPackB.
@param A matrix A, m x k (or k x m with CV_HAL_GEMM_1_T)
@param A_step bytes between rows of A
@param packedB buffer filled by gemm32fPackB with the same n and k
@param C matrix C, m x n, read when beta != 0
@param C_step bytes between rows of C
@param m number of rows of C
@param n number of columns of C
@param k number of columns of op(A)
@param alpha multiplier of op(A)*B
@param beta multiplier of C
@param flags CV_HAL_GEMM_1_T, CV_HAL_GEMM_SINGLE_THREAD
*/
CV_EXPORTS void gemm32fPacked(const float* A, size_t A_step, const void* packedB,
                              float* C, size_t C_step, int m, int n, int k,
                              float alpha, float beta, int flags);

/** @brief Computes C_i = alpha*op(A_i)*op(B_i) + beta*C_i for i = 0..batch-1.

Matrix i starts at A + A_offsets[i], B + B_offsets[i] and C + C_offsets[i]. Offsets are in elements,
so a shared (broadcast) matrix simply repeats its offset.
@param batch number of products
@param A, A_offsets, A_step matrices A, m x k (or k x m with CV_HAL_GEMM_1_T), row step in bytes
@param B, B_offsets, B_step matrices B, k x n (or n x k with CV_HAL_GEMM_2_T), row step in bytes
@param C, C_offsets, C_step matrices C, m x n, row step in bytes
@param m, n, k sizes of the products
@param alpha multiplier of op(A)*op(B)
@param beta multiplier of C
@param flags CV_HAL_GEMM_1_T, CV_HAL_GEMM_2_T, CV_HAL_GEMM_SINGLE_THREAD
*/
CV_EXPORTS void gemm32fBatch(int batch,
                             const float* A, const size_t* A_offsets, size_t A_step,
                             const float* B, const size_t* B_offsets, size_t B_step,
                             float* C, const size_t* C_offsets, size_t C_step,
                             int m, int n, int k, float alpha, float beta, int flags);

CV_EXPORTS int normL1_(const uchar* a, const uchar* b, int n);
CV_EXPORTS float normL1_(const float* a, const float* b, int n);
CV_EXPORTS float normL2Sqr_(const float* a, const float* b, int n);

CV_EXPORTS void exp32f(const float* src, float* dst, int n);
CV_EXPORTS void exp64f(const double* src, double* dst, int n);
CV_EXPORTS void log32f(const float* src, float* dst, int n);
CV_EXPORTS void log64f(const double* src, double* dst, int n);

CV_EXPORTS void cartToPolar32f(const float* x, const float* y, float* mag, float* angle, int n, bool angleInDegrees);
CV_EXPORTS void cartToPolar64f(const double* x, const double* y, double* mag, double* angle, int n, bool angleInDegrees);
CV_EXPORTS void fastAtan32f(const float* y, const float* x, float* dst, int n, bool angleInDegrees);
CV_EXPORTS void fastAtan64f(const double* y, const double* x, double* dst, int n, bool angleInDegrees);
CV_EXPORTS void magnitude32f(const float* x, const float* y, float* dst, int n);
CV_EXPORTS void magnitude64f(const double* x, const double* y, double* dst, int n);
CV_EXPORTS void polarToCart32f(const float* mag, const float* angle, float* x, float* y, int n, bool angleInDegrees);
CV_EXPORTS void polarToCart64f(const double* mag, const double* angle, double* x, double* y, int n, bool angleInDegrees);
CV_EXPORTS void sqrt32f(const float* src, float* dst, int len);
CV_EXPORTS void sqrt64f(const double* src, double* dst, int len);
CV_EXPORTS void invSqrt32f(const float* src, float* dst, int len);
CV_EXPORTS void invSqrt64f(const double* src, double* dst, int len);

CV_EXPORTS void split8u(const uchar* src, uchar** dst, int len, int cn );
CV_EXPORTS void split16u(const ushort* src, ushort** dst, int len, int cn );
CV_EXPORTS void split32s(const int* src, int** dst, int len, int cn );
CV_EXPORTS void split64s(const int64* src, int64** dst, int len, int cn );

CV_EXPORTS void merge8u(const uchar** src, uchar* dst, int len, int cn );
CV_EXPORTS void merge16u(const ushort** src, ushort* dst, int len, int cn );
CV_EXPORTS void merge32s(const int** src, int* dst, int len, int cn );
CV_EXPORTS void merge64s(const int64** src, int64* dst, int len, int cn );

CV_EXPORTS void add8u( const uchar* src1, size_t step1, const uchar* src2, size_t step2, uchar* dst, size_t step, int width, int height, void* );
CV_EXPORTS void add8s( const schar* src1, size_t step1, const schar* src2, size_t step2, schar* dst, size_t step, int width, int height, void* );
CV_EXPORTS void add16u( const ushort* src1, size_t step1, const ushort* src2, size_t step2, ushort* dst, size_t step, int width, int height, void* );
CV_EXPORTS void add16s( const short* src1, size_t step1, const short* src2, size_t step2, short* dst, size_t step, int width, int height, void* );
CV_EXPORTS void add32s( const int* src1, size_t step1, const int* src2, size_t step2, int* dst, size_t step, int width, int height, void* );
CV_EXPORTS void add32f( const float* src1, size_t step1, const float* src2, size_t step2, float* dst, size_t step, int width, int height, void* );
CV_EXPORTS void add64f( const double* src1, size_t step1, const double* src2, size_t step2, double* dst, size_t step, int width, int height, void* );
CV_EXPORTS void add16f( const cv_hal_f16* src1, size_t step1, const cv_hal_f16* src2, size_t step2, cv_hal_f16* dst, size_t step, int width, int height, void* );
CV_EXPORTS void add16bf( const cv_hal_bf16* src1, size_t step1, const cv_hal_bf16* src2, size_t step2, cv_hal_bf16* dst, size_t step, int width, int height, void* );
CV_EXPORTS void add64u( const uint64* src1, size_t step1, const uint64* src2, size_t step2, uint64* dst, size_t step, int width, int height, void* );
CV_EXPORTS void add64s( const int64* src1, size_t step1, const int64* src2, size_t step2, int64* dst, size_t step, int width, int height, void* );
CV_EXPORTS void add32u( const unsigned* src1, size_t step1, const unsigned* src2, size_t step2, unsigned* dst, size_t step, int width, int height, void* );

CV_EXPORTS void sub8u( const uchar* src1, size_t step1, const uchar* src2, size_t step2, uchar* dst, size_t step, int width, int height, void* );
CV_EXPORTS void sub8s( const schar* src1, size_t step1, const schar* src2, size_t step2, schar* dst, size_t step, int width, int height, void* );
CV_EXPORTS void sub16u( const ushort* src1, size_t step1, const ushort* src2, size_t step2, ushort* dst, size_t step, int width, int height, void* );
CV_EXPORTS void sub16s( const short* src1, size_t step1, const short* src2, size_t step2, short* dst, size_t step, int width, int height, void* );
CV_EXPORTS void sub32s( const int* src1, size_t step1, const int* src2, size_t step2, int* dst, size_t step, int width, int height, void* );
CV_EXPORTS void sub32f( const float* src1, size_t step1, const float* src2, size_t step2, float* dst, size_t step, int width, int height, void* );
CV_EXPORTS void sub64f( const double* src1, size_t step1, const double* src2, size_t step2, double* dst, size_t step, int width, int height, void* );
CV_EXPORTS void sub16f( const cv_hal_f16* src1, size_t step1, const cv_hal_f16* src2, size_t step2, cv_hal_f16* dst, size_t step, int width, int height, void* );
CV_EXPORTS void sub16bf( const cv_hal_bf16* src1, size_t step1, const cv_hal_bf16* src2, size_t step2, cv_hal_bf16* dst, size_t step, int width, int height, void* );
CV_EXPORTS void sub64u( const uint64* src1, size_t step1, const uint64* src2, size_t step2, uint64* dst, size_t step, int width, int height, void* );
CV_EXPORTS void sub64s( const int64* src1, size_t step1, const int64* src2, size_t step2, int64* dst, size_t step, int width, int height, void* );
CV_EXPORTS void sub32u( const unsigned* src1, size_t step1, const unsigned* src2, size_t step2, unsigned* dst, size_t step, int width, int height, void* );

CV_EXPORTS void max8u( const uchar* src1, size_t step1, const uchar* src2, size_t step2, uchar* dst, size_t step, int width, int height, void* );
CV_EXPORTS void max8s( const schar* src1, size_t step1, const schar* src2, size_t step2, schar* dst, size_t step, int width, int height, void* );
CV_EXPORTS void max16u( const ushort* src1, size_t step1, const ushort* src2, size_t step2, ushort* dst, size_t step, int width, int height, void* );
CV_EXPORTS void max16s( const short* src1, size_t step1, const short* src2, size_t step2, short* dst, size_t step, int width, int height, void* );
CV_EXPORTS void max32s( const int* src1, size_t step1, const int* src2, size_t step2, int* dst, size_t step, int width, int height, void* );
CV_EXPORTS void max32f( const float* src1, size_t step1, const float* src2, size_t step2, float* dst, size_t step, int width, int height, void* );
CV_EXPORTS void max64f( const double* src1, size_t step1, const double* src2, size_t step2, double* dst, size_t step, int width, int height, void* );
CV_EXPORTS void max16f( const cv_hal_f16* src1, size_t step1, const cv_hal_f16* src2, size_t step2, cv_hal_f16* dst, size_t step, int width, int height, void* );
CV_EXPORTS void max16bf( const cv_hal_bf16* src1, size_t step1, const cv_hal_bf16* src2, size_t step2, cv_hal_bf16* dst, size_t step, int width, int height, void* );
CV_EXPORTS void max64u( const uint64* src1, size_t step1, const uint64* src2, size_t step2, uint64* dst, size_t step, int width, int height, void* );
CV_EXPORTS void max64s( const int64* src1, size_t step1, const int64* src2, size_t step2, int64* dst, size_t step, int width, int height, void* );
CV_EXPORTS void max32u( const unsigned* src1, size_t step1, const unsigned* src2, size_t step2, unsigned* dst, size_t step, int width, int height, void* );

CV_EXPORTS void min8u( const uchar* src1, size_t step1, const uchar* src2, size_t step2, uchar* dst, size_t step, int width, int height, void* );
CV_EXPORTS void min8s( const schar* src1, size_t step1, const schar* src2, size_t step2, schar* dst, size_t step, int width, int height, void* );
CV_EXPORTS void min16u( const ushort* src1, size_t step1, const ushort* src2, size_t step2, ushort* dst, size_t step, int width, int height, void* );
CV_EXPORTS void min16s( const short* src1, size_t step1, const short* src2, size_t step2, short* dst, size_t step, int width, int height, void* );
CV_EXPORTS void min32s( const int* src1, size_t step1, const int* src2, size_t step2, int* dst, size_t step, int width, int height, void* );
CV_EXPORTS void min32f( const float* src1, size_t step1, const float* src2, size_t step2, float* dst, size_t step, int width, int height, void* );
CV_EXPORTS void min64f( const double* src1, size_t step1, const double* src2, size_t step2, double* dst, size_t step, int width, int height, void* );
CV_EXPORTS void min16f( const cv_hal_f16* src1, size_t step1, const cv_hal_f16* src2, size_t step2, cv_hal_f16* dst, size_t step, int width, int height, void* );
CV_EXPORTS void min16bf( const cv_hal_bf16* src1, size_t step1, const cv_hal_bf16* src2, size_t step2, cv_hal_bf16* dst, size_t step, int width, int height, void* );
CV_EXPORTS void min64u( const uint64* src1, size_t step1, const uint64* src2, size_t step2, uint64* dst, size_t step, int width, int height, void* );
CV_EXPORTS void min64s( const int64* src1, size_t step1, const int64* src2, size_t step2, int64* dst, size_t step, int width, int height, void* );
CV_EXPORTS void min32u( const unsigned* src1, size_t step1, const unsigned* src2, size_t step2, unsigned* dst, size_t step, int width, int height, void* );

CV_EXPORTS void absdiff8u( const uchar* src1, size_t step1, const uchar* src2, size_t step2, uchar* dst, size_t step, int width, int height, void* );
CV_EXPORTS void absdiff8s( const schar* src1, size_t step1, const schar* src2, size_t step2, schar* dst, size_t step, int width, int height, void* );
CV_EXPORTS void absdiff16u( const ushort* src1, size_t step1, const ushort* src2, size_t step2, ushort* dst, size_t step, int width, int height, void* );
CV_EXPORTS void absdiff16s( const short* src1, size_t step1, const short* src2, size_t step2, short* dst, size_t step, int width, int height, void* );
CV_EXPORTS void absdiff32s( const int* src1, size_t step1, const int* src2, size_t step2, int* dst, size_t step, int width, int height, void* );
CV_EXPORTS void absdiff32f( const float* src1, size_t step1, const float* src2, size_t step2, float* dst, size_t step, int width, int height, void* );
CV_EXPORTS void absdiff64f( const double* src1, size_t step1, const double* src2, size_t step2, double* dst, size_t step, int width, int height, void* );
CV_EXPORTS void absdiff16f( const cv_hal_f16* src1, size_t step1, const cv_hal_f16* src2, size_t step2, cv_hal_f16* dst, size_t step, int width, int height, void* );
CV_EXPORTS void absdiff16bf( const cv_hal_bf16* src1, size_t step1, const cv_hal_bf16* src2, size_t step2, cv_hal_bf16* dst, size_t step, int width, int height, void* );
CV_EXPORTS void absdiff64u( const uint64* src1, size_t step1, const uint64* src2, size_t step2, uint64* dst, size_t step, int width, int height, void* );
CV_EXPORTS void absdiff64s( const int64* src1, size_t step1, const int64* src2, size_t step2, int64* dst, size_t step, int width, int height, void* );
CV_EXPORTS void absdiff32u( const unsigned* src1, size_t step1, const unsigned* src2, size_t step2, unsigned* dst, size_t step, int width, int height, void* );

CV_EXPORTS void and8u( const uchar* src1, size_t step1, const uchar* src2, size_t step2, uchar* dst, size_t step, int width, int height, void* );
CV_EXPORTS void or8u( const uchar* src1, size_t step1, const uchar* src2, size_t step2, uchar* dst, size_t step, int width, int height, void* );
CV_EXPORTS void xor8u( const uchar* src1, size_t step1, const uchar* src2, size_t step2, uchar* dst, size_t step, int width, int height, void* );
CV_EXPORTS void not8u( const uchar* src1, size_t step1, const uchar* src2, size_t step2, uchar* dst, size_t step, int width, int height, void* );

CV_EXPORTS void cmp8u(const uchar* src1, size_t step1, const uchar* src2, size_t step2, uchar* dst, size_t step, int width, int height, void* _cmpop);
CV_EXPORTS void cmp8s(const schar* src1, size_t step1, const schar* src2, size_t step2, uchar* dst, size_t step, int width, int height, void* _cmpop);
CV_EXPORTS void cmp16u(const ushort* src1, size_t step1, const ushort* src2, size_t step2, uchar* dst, size_t step, int width, int height, void* _cmpop);
CV_EXPORTS void cmp16s(const short* src1, size_t step1, const short* src2, size_t step2, uchar* dst, size_t step, int width, int height, void* _cmpop);
CV_EXPORTS void cmp32s(const int* src1, size_t step1, const int* src2, size_t step2, uchar* dst, size_t step, int width, int height, void* _cmpop);
CV_EXPORTS void cmp32f(const float* src1, size_t step1, const float* src2, size_t step2, uchar* dst, size_t step, int width, int height, void* _cmpop);
CV_EXPORTS void cmp64f(const double* src1, size_t step1, const double* src2, size_t step2, uchar* dst, size_t step, int width, int height, void* _cmpop);
CV_EXPORTS void cmp16f( const cv_hal_f16* src1, size_t step1, const cv_hal_f16* src2, size_t step2, uchar* dst, size_t step, int width, int height, void* _cmpop);
CV_EXPORTS void cmp16bf( const cv_hal_bf16* src1, size_t step1, const cv_hal_bf16* src2, size_t step2, uchar* dst, size_t step, int width, int height, void* _cmpop);
CV_EXPORTS void cmp64u( const uint64* src1, size_t step1, const uint64* src2, size_t step2, uchar* dst, size_t step, int width, int height, void* _cmpop);
CV_EXPORTS void cmp64s( const int64* src1, size_t step1, const int64* src2, size_t step2, uchar* dst, size_t step, int width, int height, void* _cmpop);
CV_EXPORTS void cmp32u( const unsigned* src1, size_t step1, const unsigned* src2, size_t step2, uchar* dst, size_t step, int width, int height, void* _cmpop);

CV_EXPORTS void mul8u( const uchar* src1, size_t step1, const uchar* src2, size_t step2, uchar* dst, size_t step, int width, int height, void* scale);
CV_EXPORTS void mul8s( const schar* src1, size_t step1, const schar* src2, size_t step2, schar* dst, size_t step, int width, int height, void* scale);
CV_EXPORTS void mul16u( const ushort* src1, size_t step1, const ushort* src2, size_t step2, ushort* dst, size_t step, int width, int height, void* scale);
CV_EXPORTS void mul16s( const short* src1, size_t step1, const short* src2, size_t step2, short* dst, size_t step, int width, int height, void* scale);
CV_EXPORTS void mul32s( const int* src1, size_t step1, const int* src2, size_t step2, int* dst, size_t step, int width, int height, void* scale);
CV_EXPORTS void mul32f( const float* src1, size_t step1, const float* src2, size_t step2, float* dst, size_t step, int width, int height, void* scale);
CV_EXPORTS void mul64f( const double* src1, size_t step1, const double* src2, size_t step2, double* dst, size_t step, int width, int height, void* scale);
CV_EXPORTS void mul16f( const cv_hal_f16* src1, size_t step1, const cv_hal_f16* src2, size_t step2, cv_hal_f16* dst, size_t step, int width, int height, void* scale);
CV_EXPORTS void mul16bf( const cv_hal_bf16* src1, size_t step1, const cv_hal_bf16* src2, size_t step2, cv_hal_bf16* dst, size_t step, int width, int height, void* scale);
CV_EXPORTS void mul64u( const uint64* src1, size_t step1, const uint64* src2, size_t step2, uint64* dst, size_t step, int width, int height, void* scale);
CV_EXPORTS void mul64s( const int64* src1, size_t step1, const int64* src2, size_t step2, int64* dst, size_t step, int width, int height, void* scale);
CV_EXPORTS void mul32u( const unsigned* src1, size_t step1, const unsigned* src2, size_t step2, unsigned* dst, size_t step, int width, int height, void* scale);

CV_EXPORTS void div8u( const uchar* src1, size_t step1, const uchar* src2, size_t step2, uchar* dst, size_t step, int width, int height, void* scale);
CV_EXPORTS void div8s( const schar* src1, size_t step1, const schar* src2, size_t step2, schar* dst, size_t step, int width, int height, void* scale);
CV_EXPORTS void div16u( const ushort* src1, size_t step1, const ushort* src2, size_t step2, ushort* dst, size_t step, int width, int height, void* scale);
CV_EXPORTS void div16s( const short* src1, size_t step1, const short* src2, size_t step2, short* dst, size_t step, int width, int height, void* scale);
CV_EXPORTS void div32s( const int* src1, size_t step1, const int* src2, size_t step2, int* dst, size_t step, int width, int height, void* scale);
CV_EXPORTS void div32f( const float* src1, size_t step1, const float* src2, size_t step2, float* dst, size_t step, int width, int height, void* scale);
CV_EXPORTS void div64f( const double* src1, size_t step1, const double* src2, size_t step2, double* dst, size_t step, int width, int height, void* scale);
CV_EXPORTS void div16f( const cv_hal_f16* src1, size_t step1, const cv_hal_f16* src2, size_t step2, cv_hal_f16* dst, size_t step, int width, int height, void* scale);
CV_EXPORTS void div16bf( const cv_hal_bf16* src1, size_t step1, const cv_hal_bf16* src2, size_t step2, cv_hal_bf16* dst, size_t step, int width, int height, void* scale);
CV_EXPORTS void div64u( const uint64* src1, size_t step1, const uint64* src2, size_t step2, uint64* dst, size_t step, int width, int height, void* scale);
CV_EXPORTS void div64s( const int64* src1, size_t step1, const int64* src2, size_t step2, int64* dst, size_t step, int width, int height, void* scale);
CV_EXPORTS void div32u( const unsigned* src1, size_t step1, const unsigned* src2, size_t step2, unsigned* dst, size_t step, int width, int height, void* scale);

CV_EXPORTS void recip8u( const uchar *, size_t, const uchar * src2, size_t step2, uchar* dst, size_t step, int width, int height, void* scale);
CV_EXPORTS void recip8s( const schar *, size_t, const schar * src2, size_t step2, schar* dst, size_t step, int width, int height, void* scale);
CV_EXPORTS void recip16u( const ushort *, size_t, const ushort * src2, size_t step2, ushort* dst, size_t step, int width, int height, void* scale);
CV_EXPORTS void recip16s( const short *, size_t, const short * src2, size_t step2, short* dst, size_t step, int width, int height, void* scale);
CV_EXPORTS void recip32s( const int *, size_t, const int * src2, size_t step2, int* dst, size_t step, int width, int height, void* scale);
CV_EXPORTS void recip32f( const float *, size_t, const float * src2, size_t step2, float* dst, size_t step, int width, int height, void* scale);
CV_EXPORTS void recip64f( const double *, size_t, const double * src2, size_t step2, double* dst, size_t step, int width, int height, void* scale);
CV_EXPORTS void recip16f( const cv_hal_f16 *, size_t, const cv_hal_f16* src2, size_t step2, cv_hal_f16* dst, size_t step, int width, int height, void* );
CV_EXPORTS void recip16bf( const cv_hal_bf16 *, size_t, const cv_hal_bf16* src2, size_t step2, cv_hal_bf16* dst, size_t step, int width, int height, void* );
CV_EXPORTS void recip64u( const uint64 *, size_t, const uint64* src2, size_t step2, uint64* dst, size_t step, int width, int height, void* );
CV_EXPORTS void recip64s( const int64 *, size_t, const int64* src2, size_t step2, int64* dst, size_t step, int width, int height, void* );
CV_EXPORTS void recip32u( const unsigned *, size_t, const unsigned* src2, size_t step2, unsigned* dst, size_t step, int width, int height, void* );

CV_EXPORTS void addWeighted8u( const uchar* src1, size_t step1, const uchar* src2, size_t step2, uchar* dst, size_t step, int width, int height, void* _scalars );
CV_EXPORTS void addWeighted8s( const schar* src1, size_t step1, const schar* src2, size_t step2, schar* dst, size_t step, int width, int height, void* scalars );
CV_EXPORTS void addWeighted16u( const ushort* src1, size_t step1, const ushort* src2, size_t step2, ushort* dst, size_t step, int width, int height, void* scalars );
CV_EXPORTS void addWeighted16s( const short* src1, size_t step1, const short* src2, size_t step2, short* dst, size_t step, int width, int height, void* scalars );
CV_EXPORTS void addWeighted32s( const int* src1, size_t step1, const int* src2, size_t step2, int* dst, size_t step, int width, int height, void* scalars );
CV_EXPORTS void addWeighted32f( const float* src1, size_t step1, const float* src2, size_t step2, float* dst, size_t step, int width, int height, void* scalars );
CV_EXPORTS void addWeighted64f( const double* src1, size_t step1, const double* src2, size_t step2, double* dst, size_t step, int width, int height, void* scalars );
CV_EXPORTS void addWeighted16f( const cv_hal_f16* src1, size_t step1, const cv_hal_f16* src2, size_t step2, cv_hal_f16* dst, size_t step, int width, int height, void* scalars );
CV_EXPORTS void addWeighted16bf( const cv_hal_bf16* src1, size_t step1, const cv_hal_bf16* src2, size_t step2, cv_hal_bf16* dst, size_t step, int width, int height, void* scalars );
CV_EXPORTS void addWeighted64u( const uint64* src1, size_t step1, const uint64* src2, size_t step2, uint64* dst, size_t step, int width, int height, void* scalars );
CV_EXPORTS void addWeighted64s( const int64* src1, size_t step1, const int64* src2, size_t step2, int64* dst, size_t step, int width, int height, void* scalars );
CV_EXPORTS void addWeighted32u( const unsigned* src1, size_t step1, const unsigned* src2, size_t step2, unsigned* dst, size_t step, int width, int height, void* scalars );

CV_EXPORTS void cvt16f32f( const hfloat* src, float* dst, int len );
CV_EXPORTS void cvt32f16f( const float* src, hfloat* dst, int len );
CV_EXPORTS void cvt16bf32f( const bfloat* src, float* dst, int len );
CV_EXPORTS void cvt32f16bf( const float* src, bfloat* dst, int len );

CV_EXPORTS void addRNGBias32f( float* arr, const float* scaleBiasPairs, int len, int cn );
CV_EXPORTS void addRNGBias64f( double* arr, const double* scaleBiasPairs, int len, int cn );

struct CV_EXPORTS DFT1D
{
    static Ptr<DFT1D> create(int len, int count, int depth, int flags, bool * useBuffer = 0);
    virtual void apply(const uchar *src, uchar *dst) = 0;
    virtual ~DFT1D() {}
};

struct CV_EXPORTS DFT2D
{
    static Ptr<DFT2D> create(int width, int height, int depth,
                             int src_channels, int dst_channels,
                             int flags, int nonzero_rows = 0);
    virtual void apply(const uchar *src_data, size_t src_step, uchar *dst_data, size_t dst_step) = 0;
    virtual ~DFT2D() {}
};

struct CV_EXPORTS DCT2D
{
    static Ptr<DCT2D> create(int width, int height, int depth, int flags);
    virtual void apply(const uchar *src_data, size_t src_step, uchar *dst_data, size_t dst_step) = 0;
    virtual ~DCT2D() {}
};

//! @} core_hal

//=============================================================================
// for binary compatibility with 3.0

//! @cond IGNORED

CV_EXPORTS int LU(float* A, size_t astep, int m, float* b, size_t bstep, int n);
CV_EXPORTS int LU(double* A, size_t astep, int m, double* b, size_t bstep, int n);
CV_EXPORTS bool Cholesky(float* A, size_t astep, int m, float* b, size_t bstep, int n);
CV_EXPORTS bool Cholesky(double* A, size_t astep, int m, double* b, size_t bstep, int n);

CV_EXPORTS void exp(const float* src, float* dst, int n);
CV_EXPORTS void exp(const double* src, double* dst, int n);
CV_EXPORTS void log(const float* src, float* dst, int n);
CV_EXPORTS void log(const double* src, double* dst, int n);

CV_EXPORTS void fastAtan2(const float* y, const float* x, float* dst, int n, bool angleInDegrees);
CV_EXPORTS void magnitude(const float* x, const float* y, float* dst, int n);
CV_EXPORTS void magnitude(const double* x, const double* y, double* dst, int n);
CV_EXPORTS void sqrt(const float* src, float* dst, int len);
CV_EXPORTS void sqrt(const double* src, double* dst, int len);
CV_EXPORTS void invSqrt(const float* src, float* dst, int len);
CV_EXPORTS void invSqrt(const double* src, double* dst, int len);

//! @endcond

}} //cv::hal

#endif //OPENCV_HAL_HPP
