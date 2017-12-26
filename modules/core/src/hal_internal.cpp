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

#include "hal_internal.hpp"

#ifdef HAVE_LAPACK

#include "opencv_lapack.h"

#include <cmath>
#include <algorithm>
#include <typeinfo>
#include <limits>
#include <complex>
#include <vector>

#define HAL_GEMM_SMALL_COMPLEX_MATRIX_THRESH 100
#define HAL_GEMM_SMALL_MATRIX_THRESH 100
#define HAL_SVD_SMALL_MATRIX_THRESH 25
#define HAL_QR_SMALL_MATRIX_THRESH 30
#define HAL_LU_SMALL_MATRIX_THRESH 100
#define HAL_CHOLESKY_SMALL_MATRIX_THRESH 100

//lapack stores matrices in column-major order so transposing is neded everywhere
template <typename fptype> static inline void
transpose_square_inplace(fptype *src, size_t src_ld, size_t m)
{
    for(size_t i = 0; i < m - 1; i++)
        for(size_t j = i + 1; j < m; j++)
            std::swap(src[j*src_ld + i], src[i*src_ld + j]);
}

template <typename fptype> static inline void
transpose(const fptype *src, size_t src_ld, fptype* dst, size_t dst_ld, size_t m, size_t n)
{
    for(size_t i = 0; i < m; i++)
        for(size_t j = 0; j < n; j++)
            dst[j*dst_ld + i] = src[i*src_ld + j];
}

template <typename fptype> static inline void
copy_matrix(const fptype *src, size_t src_ld, fptype* dst, size_t dst_ld, size_t m, size_t n)
{
    for(size_t i = 0; i < m; i++)
        for(size_t j = 0; j < n; j++)
            dst[i*dst_ld + j] = src[i*src_ld + j];
}

template <typename fptype> static inline void
set_value(fptype *dst, size_t dst_ld, fptype value, size_t m, size_t n)
{
    for(size_t i = 0; i < m; i++)
        for(size_t j = 0; j < n; j++)
            dst[i*dst_ld + j] = value;
}

template <typename fptype> static inline int
lapack_LU(fptype* a, size_t a_step, int m, fptype* b, size_t b_step, int n, int* info)
{
    int lda = (int)(a_step / sizeof(fptype)), sign = 0;
    int* piv = new int[m];

    transpose_square_inplace(a, lda, m);

    if(b)
    {
        if(n == 1 && b_step == sizeof(fptype))
        {
            if(typeid(fptype) == typeid(float))
                sgesv_(&m, &n, (float*)a, &lda, piv, (float*)b, &m, info);
            else if(typeid(fptype) == typeid(double))
                dgesv_(&m, &n, (double*)a, &lda, piv, (double*)b, &m, info);
        }
        else
        {
            int ldb = (int)(b_step / sizeof(fptype));
            fptype* tmpB = new fptype[m*n];

            transpose(b, ldb, tmpB, m, m, n);

            if(typeid(fptype) == typeid(float))
                sgesv_(&m, &n, (float*)a, &lda, piv, (float*)tmpB, &m, info);
            else if(typeid(fptype) == typeid(double))
                dgesv_(&m, &n, (double*)a, &lda, piv, (double*)tmpB, &m, info);

            transpose(tmpB, m, b, ldb, n, m);
            delete[] tmpB;
        }
    }
    else
    {
        if(typeid(fptype) == typeid(float))
            sgetrf_(&m, &m, (float*)a, &lda, piv, info);
        else if(typeid(fptype) == typeid(double))
            dgetrf_(&m, &m, (double*)a, &lda, piv, info);
    }

    if(*info == 0)
    {
        for(int i = 0; i < m; i++)
            sign ^= piv[i] != i + 1;
        *info = sign ? -1 : 1;
    }
    else
        *info = 0; //in opencv LU function zero means error

    delete[] piv;
    return CV_HAL_ERROR_OK;
}

template <typename fptype> static inline int
lapack_Cholesky(fptype* a, size_t a_step, int m, fptype* b, size_t b_step, int n, bool* info)
{
    int lapackStatus = 0;
    int lda = (int)(a_step / sizeof(fptype));
    char L[] = {'L', '\0'};

    if(b)
    {
        if(n == 1 && b_step == sizeof(fptype))
        {
            if(typeid(fptype) == typeid(float))
                sposv_(L, &m, &n, (float*)a, &lda, (float*)b, &m, &lapackStatus);
            else if(typeid(fptype) == typeid(double))
                dposv_(L, &m, &n, (double*)a, &lda, (double*)b, &m, &lapackStatus);
        }
        else
        {
            int ldb = (int)(b_step / sizeof(fptype));
            fptype* tmpB = new fptype[m*n];
            transpose(b, ldb, tmpB, m, m, n);

            if(typeid(fptype) == typeid(float))
                sposv_(L, &m, &n, (float*)a, &lda, (float*)tmpB, &m, &lapackStatus);
            else if(typeid(fptype) == typeid(double))
                dposv_(L, &m, &n, (double*)a, &lda, (double*)tmpB, &m, &lapackStatus);

            transpose(tmpB, m, b, ldb, n, m);
            delete[] tmpB;
        }
    }
    else
    {
        if(typeid(fptype) == typeid(float))
            spotrf_(L, &m, (float*)a, &lda, &lapackStatus);
        else if(typeid(fptype) == typeid(double))
            dpotrf_(L, &m, (double*)a, &lda, &lapackStatus);
    }

    if(lapackStatus == 0) *info = true;
    else *info = false; //in opencv Cholesky function false means error

    return CV_HAL_ERROR_OK;
}

template <typename fptype> static inline int
lapack_SVD(fptype* a, size_t a_step, fptype *w, fptype* u, size_t u_step, fptype* vt, size_t v_step, int m, int n, int flags, int* info)
{
    int lda = (int)(a_step / sizeof(fptype));
    int ldv = (int)(v_step / sizeof(fptype));
    int ldu = (int)(u_step / sizeof(fptype));
    int lwork = -1;
    int* iworkBuf = new int[8*std::min(m, n)];
    fptype work1 = 0;

    //A already transposed and m>=n
    char mode[] = { ' ', '\0'};
    if(flags & CV_HAL_SVD_NO_UV)
    {
        ldv = 1;
        mode[0] = 'N';
    }
    else if((flags & CV_HAL_SVD_SHORT_UV) && (flags & CV_HAL_SVD_MODIFY_A)) //short SVD, U stored in a
        mode[0] = 'O';
    else if((flags & CV_HAL_SVD_SHORT_UV) && !(flags & CV_HAL_SVD_MODIFY_A)) //short SVD, U stored in u if m>=n
        mode[0] = 'S';
    else if(flags & CV_HAL_SVD_FULL_UV) //full SVD, U stored in u or in a
        mode[0] = 'A';

    if((flags & CV_HAL_SVD_MODIFY_A) && (flags & CV_HAL_SVD_FULL_UV)) //U stored in a
    {
        u = new fptype[m*m];
        ldu = m;
    }

    if(typeid(fptype) == typeid(float))
        sgesdd_(mode, &m, &n, (float*)a, &lda, (float*)w, (float*)u, &ldu, (float*)vt, &ldv, (float*)&work1, &lwork, iworkBuf, info);
    else if(typeid(fptype) == typeid(double))
        dgesdd_(mode, &m, &n, (double*)a, &lda, (double*)w, (double*)u, &ldu, (double*)vt, &ldv, (double*)&work1, &lwork, iworkBuf, info);

    lwork = (int)round(work1); //optimal buffer size
    fptype* buffer = new fptype[lwork + 1];

    if(typeid(fptype) == typeid(float))
        sgesdd_(mode, &m, &n, (float*)a, &lda, (float*)w, (float*)u, &ldu, (float*)vt, &ldv, (float*)buffer, &lwork, iworkBuf, info);
    else if(typeid(fptype) == typeid(double))
        dgesdd_(mode, &m, &n, (double*)a, &lda, (double*)w, (double*)u, &ldu, (double*)vt, &ldv, (double*)buffer, &lwork, iworkBuf, info);

    if(!(flags & CV_HAL_SVD_NO_UV))
        transpose_square_inplace(vt, ldv, n);

    if((flags & CV_HAL_SVD_MODIFY_A) && (flags & CV_HAL_SVD_FULL_UV))
    {
        for(int i = 0; i < m; i++)
            for(int j = 0; j < m; j++)
                a[i*lda + j] = u[i*m + j];
        delete[] u;
    }

    delete[] iworkBuf;
    delete[] buffer;
    return CV_HAL_ERROR_OK;
}

template <typename fptype> static inline int
lapack_QR(fptype* a, size_t a_step, int m, int n, int k, fptype* b, size_t b_step, fptype* dst, int* info)
{
    int lda = (int)(a_step / sizeof(fptype));
    char mode[] = { 'N', '\0' };
    if(m < n)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    std::vector<fptype> tmpAMemHolder;
    fptype* tmpA;
    int ldtmpA;
    if (m == n)
    {
        transpose_square_inplace(a, lda, m);
        tmpA = a;
        ldtmpA = lda;
    }
    else
    {
        tmpAMemHolder.resize(m*n);
        tmpA = &tmpAMemHolder.front();
        ldtmpA = m;
        transpose(a, lda, tmpA, m, m, n);
    }

    int lwork = -1;
    fptype work1 = 0.;

    if (b)
    {
        if (k == 1 && b_step == sizeof(fptype))
        {
            if (typeid(fptype) == typeid(float))
                sgels_(mode, &m, &n, &k, (float*)tmpA, &ldtmpA, (float*)b, &m, (float*)&work1, &lwork, info);
            else if (typeid(fptype) == typeid(double))
                dgels_(mode, &m, &n, &k, (double*)tmpA, &ldtmpA, (double*)b, &m, (double*)&work1, &lwork, info);

            lwork = cvRound(work1); //optimal buffer size
            std::vector<fptype> workBufMemHolder(lwork + 1);
            fptype* buffer = &workBufMemHolder.front();

            if (typeid(fptype) == typeid(float))
                sgels_(mode, &m, &n, &k, (float*)tmpA, &ldtmpA, (float*)b, &m, (float*)buffer, &lwork, info);
            else if (typeid(fptype) == typeid(double))
                dgels_(mode, &m, &n, &k, (double*)tmpA, &ldtmpA, (double*)b, &m, (double*)buffer, &lwork, info);
        }
        else
        {
            std::vector<fptype> tmpBMemHolder(m*k);
            fptype* tmpB = &tmpBMemHolder.front();
            int ldb = (int)(b_step / sizeof(fptype));
            transpose(b, ldb, tmpB, m, m, k);

            if (typeid(fptype) == typeid(float))
                sgels_(mode, &m, &n, &k, (float*)tmpA, &ldtmpA, (float*)tmpB, &m, (float*)&work1, &lwork, info);
            else if (typeid(fptype) == typeid(double))
                dgels_(mode, &m, &n, &k, (double*)tmpA, &ldtmpA, (double*)tmpB, &m, (double*)&work1, &lwork, info);

            lwork = cvRound(work1); //optimal buffer size
            std::vector<fptype> workBufMemHolder(lwork + 1);
            fptype* buffer = &workBufMemHolder.front();

            if (typeid(fptype) == typeid(float))
                sgels_(mode, &m, &n, &k, (float*)tmpA, &ldtmpA, (float*)tmpB, &m, (float*)buffer, &lwork, info);
            else if (typeid(fptype) == typeid(double))
                dgels_(mode, &m, &n, &k, (double*)tmpA, &ldtmpA, (double*)tmpB, &m, (double*)buffer, &lwork, info);

            transpose(tmpB, m, b, ldb, k, m);
        }
    }
    else
    {
        if (typeid(fptype) == typeid(float))
            sgeqrf_(&m, &n, (float*)tmpA, &ldtmpA, (float*)dst, (float*)&work1, &lwork, info);
        else if (typeid(fptype) == typeid(double))
            dgeqrf_(&m, &n, (double*)tmpA, &ldtmpA, (double*)dst, (double*)&work1, &lwork, info);

        lwork = cvRound(work1); //optimal buffer size
        std::vector<fptype> workBufMemHolder(lwork + 1);
        fptype* buffer = &workBufMemHolder.front();

        if (typeid(fptype) == typeid(float))
            sgeqrf_(&m, &n, (float*)tmpA, &ldtmpA, (float*)dst, (float*)buffer, &lwork, info);
        else if (typeid(fptype) == typeid(double))
            dgeqrf_(&m, &n, (double*)tmpA, &ldtmpA, (double*)dst, (double*)buffer, &lwork, info);
    }

    if (m == n)
        transpose_square_inplace(a, lda, m);
    else
        transpose(tmpA, m, a, lda, n, m);

    if (*info != 0)
        *info = 0;
    else
        *info = 1;

    return CV_HAL_ERROR_OK;
}

template <typename fptype> static inline int
lapack_gemm(const fptype *src1, size_t src1_step, const fptype *src2, size_t src2_step, fptype alpha,
            const fptype *src3, size_t src3_step, fptype beta, fptype *dst, size_t dst_step, int a_m, int a_n, int d_n, int flags)
{
    int ldsrc1 = (int)(src1_step / sizeof(fptype));
    int ldsrc2 = (int)(src2_step / sizeof(fptype));
    int ldsrc3 = (int)(src3_step / sizeof(fptype));
    int lddst = (int)(dst_step / sizeof(fptype));
    int c_m, c_n, d_m;
    CBLAS_TRANSPOSE transA, transB;

    if(flags & CV_HAL_GEMM_2_T)
    {
        transB = CblasTrans;
        if(flags & CV_HAL_GEMM_1_T )
        {
            d_m = a_n;
        }
        else
        {
            d_m = a_m;
        }
    }
    else
    {
        transB = CblasNoTrans;
        if(flags & CV_HAL_GEMM_1_T )
        {
            d_m = a_n;
        }
        else
        {
            d_m = a_m;
        }
    }

    if(flags & CV_HAL_GEMM_3_T)
    {
        c_m = d_n;
        c_n = d_m;
    }
    else
    {
        c_m = d_m;
        c_n = d_n;
    }

    if(flags & CV_HAL_GEMM_1_T )
    {
        transA = CblasTrans;
        std::swap(a_n, a_m);
    }
    else
    {
        transA = CblasNoTrans;
    }

    if(src3 != dst && beta != 0.0 && src3_step != 0) {
        if(flags & CV_HAL_GEMM_3_T)
            transpose(src3, ldsrc3, dst, lddst, c_m, c_n);
        else
            copy_matrix(src3, ldsrc3, dst, lddst, c_m, c_n);
    }
    else if (src3 == dst && (flags & CV_HAL_GEMM_3_T)) //actually transposing C in this case done by openCV
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    else if(src3_step == 0 && beta != 0.0)
        set_value(dst, lddst, (fptype)0.0, d_m, d_n);

    if(typeid(fptype) == typeid(float))
        cblas_sgemm(CblasRowMajor, transA, transB, a_m, d_n, a_n, (float)alpha, (float*)src1, ldsrc1, (float*)src2, ldsrc2, (float)beta, (float*)dst, lddst);
    else if(typeid(fptype) == typeid(double))
        cblas_dgemm(CblasRowMajor, transA, transB, a_m, d_n, a_n, (double)alpha, (double*)src1, ldsrc1, (double*)src2, ldsrc2, (double)beta, (double*)dst, lddst);

    return CV_HAL_ERROR_OK;
}


template <typename fptype> static inline int
lapack_gemm_c(const fptype *src1, size_t src1_step, const fptype *src2, size_t src2_step, fptype alpha,
            const fptype *src3, size_t src3_step, fptype beta, fptype *dst, size_t dst_step, int a_m, int a_n, int d_n, int flags)
{
    int ldsrc1 = (int)(src1_step / sizeof(std::complex<fptype>));
    int ldsrc2 = (int)(src2_step / sizeof(std::complex<fptype>));
    int ldsrc3 = (int)(src3_step / sizeof(std::complex<fptype>));
    int lddst = (int)(dst_step / sizeof(std::complex<fptype>));
    int c_m, c_n, d_m;
    CBLAS_TRANSPOSE transA, transB;
    std::complex<fptype> cAlpha(alpha, 0.0);
    std::complex<fptype> cBeta(beta, 0.0);

    if(flags & CV_HAL_GEMM_2_T)
    {
        transB = CblasTrans;
        if(flags & CV_HAL_GEMM_1_T )
        {
            d_m = a_n;
        }
        else
        {
            d_m = a_m;
        }
    }
    else
    {
        transB = CblasNoTrans;
        if(flags & CV_HAL_GEMM_1_T )
        {
            d_m = a_n;
        }
        else
        {
            d_m = a_m;
        }
    }

    if(flags & CV_HAL_GEMM_3_T)
    {
        c_m = d_n;
        c_n = d_m;
    }
    else
    {
        c_m = d_m;
        c_n = d_n;
    }

    if(flags & CV_HAL_GEMM_1_T )
    {
        transA = CblasTrans;
        std::swap(a_n, a_m);
    }
    else
    {
        transA = CblasNoTrans;
    }

    if(src3 != dst && beta != 0.0 && src3_step != 0) {
        if(flags & CV_HAL_GEMM_3_T)
            transpose((std::complex<fptype>*)src3, ldsrc3, (std::complex<fptype>*)dst, lddst, c_m, c_n);
        else
            copy_matrix((std::complex<fptype>*)src3, ldsrc3, (std::complex<fptype>*)dst, lddst, c_m, c_n);
    }
    else if (src3 == dst && (flags & CV_HAL_GEMM_3_T)) //actually transposing C in this case done by openCV
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    else if(src3_step == 0 && beta != 0.0)
        set_value((std::complex<fptype>*)dst, lddst, std::complex<fptype>(0.0, 0.0), d_m, d_n);

    if(typeid(fptype) == typeid(float))
        cblas_cgemm(CblasRowMajor, transA, transB, a_m, d_n, a_n, (float*)reinterpret_cast<fptype(&)[2]>(cAlpha), (float*)src1, ldsrc1, (float*)src2, ldsrc2, (float*)reinterpret_cast<fptype(&)[2]>(cBeta), (float*)dst, lddst);
    else if(typeid(fptype) == typeid(double))
        cblas_zgemm(CblasRowMajor, transA, transB, a_m, d_n, a_n, (double*)reinterpret_cast<fptype(&)[2]>(cAlpha), (double*)src1, ldsrc1, (double*)src2, ldsrc2, (double*)reinterpret_cast<fptype(&)[2]>(cBeta), (double*)dst, lddst);

    return CV_HAL_ERROR_OK;
}
int lapack_LU32f(float* a, size_t a_step, int m, float* b, size_t b_step, int n, int* info)
{
    if(m < HAL_LU_SMALL_MATRIX_THRESH)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    return lapack_LU(a, a_step, m, b, b_step, n, info);
}

int lapack_LU64f(double* a, size_t a_step, int m, double* b, size_t b_step, int n, int* info)
{
    if(m < HAL_LU_SMALL_MATRIX_THRESH)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    return lapack_LU(a, a_step, m, b, b_step, n, info);
}

int lapack_Cholesky32f(float* a, size_t a_step, int m, float* b, size_t b_step, int n, bool *info)
{
    if(m < HAL_CHOLESKY_SMALL_MATRIX_THRESH)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    return lapack_Cholesky(a, a_step, m, b, b_step, n, info);
}

int lapack_Cholesky64f(double* a, size_t a_step, int m, double* b, size_t b_step, int n, bool *info)
{
    if(m < HAL_CHOLESKY_SMALL_MATRIX_THRESH)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    return lapack_Cholesky(a, a_step, m, b, b_step, n, info);
}

int lapack_SVD32f(float* a, size_t a_step, float *w, float* u, size_t u_step, float* vt, size_t v_step, int m, int n, int flags)
{

    if(m < HAL_SVD_SMALL_MATRIX_THRESH)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    int info;
    return lapack_SVD(a, a_step, w, u, u_step, vt, v_step, m, n, flags, &info);
}

int lapack_SVD64f(double* a, size_t a_step, double *w, double* u, size_t u_step, double* vt, size_t v_step, int m, int n, int flags)
{

    if(m < HAL_SVD_SMALL_MATRIX_THRESH)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    int info;
    return lapack_SVD(a, a_step, w, u, u_step, vt, v_step, m, n, flags, &info);
}

int lapack_QR32f(float* src1, size_t src1_step, int m, int n, int k, float* src2, size_t src2_step, float* dst, int* info)
{
    if (m < HAL_QR_SMALL_MATRIX_THRESH)
      return CV_HAL_ERROR_NOT_IMPLEMENTED;
    return lapack_QR(src1, src1_step, m, n, k, src2, src2_step, dst, info);
}

int lapack_QR64f(double* src1, size_t src1_step, int m, int n, int k, double* src2, size_t src2_step, double* dst, int* info)
{
    if (m < HAL_QR_SMALL_MATRIX_THRESH)
      return CV_HAL_ERROR_NOT_IMPLEMENTED;
    return lapack_QR(src1, src1_step, m, n, k, src2, src2_step, dst, info);
}

int lapack_gemm32f(const float *src1, size_t src1_step, const float *src2, size_t src2_step, float alpha,
                   const float *src3, size_t src3_step, float beta, float *dst, size_t dst_step, int m, int n, int k, int flags)
{
    if(m < HAL_GEMM_SMALL_MATRIX_THRESH)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    return lapack_gemm(src1, src1_step, src2, src2_step, alpha, src3, src3_step, beta, dst, dst_step, m, n, k, flags);
}

int lapack_gemm64f(const double *src1, size_t src1_step, const double *src2, size_t src2_step, double alpha,
                   const double *src3, size_t src3_step, double beta, double *dst, size_t dst_step, int m, int n, int k, int flags)
{
    if(m < HAL_GEMM_SMALL_MATRIX_THRESH)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    return lapack_gemm(src1, src1_step, src2, src2_step, alpha, src3, src3_step, beta, dst, dst_step, m, n, k, flags);
}

int lapack_gemm32fc(const float *src1, size_t src1_step, const float *src2, size_t src2_step, float alpha,
                   const float *src3, size_t src3_step, float beta, float *dst, size_t dst_step, int m, int n, int k, int flags)
{
    if(m < HAL_GEMM_SMALL_COMPLEX_MATRIX_THRESH)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    return lapack_gemm_c(src1, src1_step, src2, src2_step, alpha, src3, src3_step, beta, dst, dst_step, m, n, k, flags);
}
int lapack_gemm64fc(const double *src1, size_t src1_step, const double *src2, size_t src2_step, double alpha,
                   const double *src3, size_t src3_step, double beta, double *dst, size_t dst_step, int m, int n, int k, int flags)
{
    if(m < HAL_GEMM_SMALL_COMPLEX_MATRIX_THRESH)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    return lapack_gemm_c(src1, src1_step, src2, src2_step, alpha, src3, src3_step, beta, dst, dst_step, m, n, k, flags);
}

#endif //HAVE_LAPACK
