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

int armplDFTFwd_RToPack(const float*  src, float*  dst,
                        const void* spec, unsigned char* buf);
int armplDFTFwd_RToPack(const double* src, double* dst,
                        const void* spec, unsigned char* buf);
int armplDFTInv_PackToR(const float*  src, float*  dst,
                        const void* spec, unsigned char* buf);
int armplDFTInv_PackToR(const double* src, double* dst,
                        const void* spec, unsigned char* buf);

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

#endif  // HAVE_ARMPL

#endif  // OPENCV_ARMPL_HAL_CORE_HPP
