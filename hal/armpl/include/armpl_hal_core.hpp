#ifndef OPENCV_ARMPL_HAL_CORE_HPP
#define OPENCV_ARMPL_HAL_CORE_HPP

#ifdef HAVE_ARMPL

#include <stddef.h>
#include <fftw3.h>
#include <opencv2/core/base.hpp>
#include <opencv2/core/utility.hpp>

#ifndef cvhalDFT
struct cvhalDFT;
#endif

int armpl_hal_dftInit2D(cvhalDFT **context, int width, int height,
                        int depth, int src_channels, int dst_channels,
                        int flags, int nonzero_rows);

int armpl_hal_dft2D(cvhalDFT *context, const unsigned char *src_data,
                    size_t src_step, unsigned char *dst_data, size_t dst_step);

int armpl_hal_dftFree2D(cvhalDFT *context);

#undef  cv_hal_dftInit2D
#define cv_hal_dftInit2D armpl_hal_dftInit2D

#undef  cv_hal_dft2D
#define cv_hal_dft2D armpl_hal_dft2D

#undef  cv_hal_dftFree2D
#define cv_hal_dftFree2D armpl_hal_dftFree2D

struct ArmplDFTSpec_C_32fc {
    fftwf_plan plan;
    int        n;
    bool       isInverse;
};

struct ArmplDFTSpec_C_64fc {
    fftw_plan plan;
    int       n;
    bool      isInverse;
};

struct ArmplDFTSpec_R_32f {
    fftwf_plan plan;
    int        n;
    bool       isInverse;
    double     scale;
};

struct ArmplDFTSpec_R_64f {
    fftw_plan plan;
    int       n;
    bool      isInverse;
    double    scale;
};

int armpl_hal_dftInit1D(cvhalDFT **context, int len, int count,
                        int depth, int flags, bool *needBuffer);

int armpl_hal_dft1D(cvhalDFT *context,
                    const unsigned char *src, unsigned char *dst);

int armpl_hal_dftFree1D(cvhalDFT *context);

#undef  cv_hal_dftInit1D
#define cv_hal_dftInit1D armpl_hal_dftInit1D

#undef  cv_hal_dft1D
#define cv_hal_dft1D armpl_hal_dft1D

#undef  cv_hal_dftFree1D
#define cv_hal_dftFree1D armpl_hal_dftFree1D

int armpl_hal_dctInit2D(cvhalDFT **context, int width, int height,
                        int depth, int flags);
int armpl_hal_dct2D(cvhalDFT *context, const unsigned char *src_data,
                    size_t src_step, unsigned char *dst_data, size_t dst_step);
int armpl_hal_dctFree2D(cvhalDFT *context);

#undef  cv_hal_dctInit2D
#define cv_hal_dctInit2D armpl_hal_dctInit2D
#undef  cv_hal_dct2D
#define cv_hal_dct2D armpl_hal_dct2D
#undef  cv_hal_dctFree2D
#define cv_hal_dctFree2D armpl_hal_dctFree2D

int armpl_hal_gemm32f(const float* src1, size_t src1_step, const float* src2, size_t src2_step,
                       float alpha, const float* src3, size_t src3_step, float beta, float* dst, size_t dst_step,
                       int m, int n, int k, int flags);
int armpl_hal_gemm64f(const double* src1, size_t src1_step, const double* src2, size_t src2_step,
                       double alpha, const double* src3, size_t src3_step, double beta, double* dst, size_t dst_step,
                       int m, int n, int k, int flags);
int armpl_hal_gemm32fc(const float* src1, size_t src1_step, const float* src2, size_t src2_step,
                       float alpha, const float* src3, size_t src3_step, float beta, float* dst, size_t dst_step,
                       int m, int n, int k, int flags);
int armpl_hal_gemm64fc(const double* src1, size_t src1_step, const double* src2, size_t src2_step,
                       double alpha, const double* src3, size_t src3_step, double beta, double* dst, size_t dst_step,
                       int m, int n, int k, int flags);

#undef  cv_hal_gemm32f
#define cv_hal_gemm32f armpl_hal_gemm32f
#undef  cv_hal_gemm64f
#define cv_hal_gemm64f armpl_hal_gemm64f
#undef  cv_hal_gemm32fc
#define cv_hal_gemm32fc armpl_hal_gemm32fc
#undef  cv_hal_gemm64fc
#define cv_hal_gemm64fc armpl_hal_gemm64fc

#endif  // HAVE_ARMPL

#endif  // OPENCV_ARMPL_HAL_CORE_HPP
