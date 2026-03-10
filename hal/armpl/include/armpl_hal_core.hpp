#ifndef OPENCV_ARMPL_HAL_CORE_HPP
#define OPENCV_ARMPL_HAL_CORE_HPP

#ifdef HAVE_ARMPL

#include <stddef.h>
#include <fftw3.h>

#ifndef cvhalDFT
struct cvhalDFT;
#endif

#ifndef CV_HAL_ERROR_OK
#  define CV_HAL_ERROR_OK               0
#endif

#ifndef CV_HAL_ERROR_NOT_IMPLEMENTED
#  define CV_HAL_ERROR_NOT_IMPLEMENTED  1
#endif

#ifndef CV_HAL_DFT_INVERSE
#  define CV_HAL_DFT_INVERSE        (1 << 0)
#endif

#ifndef CV_HAL_DFT_SCALE
#  define CV_HAL_DFT_SCALE          (1 << 1)
#endif

#ifndef CV_HAL_DFT_ROWS
#  define CV_HAL_DFT_ROWS           (1 << 2)
#endif

#ifndef CV_HAL_DFT_COMPLEX_OUTPUT
#  define CV_HAL_DFT_COMPLEX_OUTPUT (1 << 3)
#endif

#ifndef CV_HAL_DFT_REAL_OUTPUT
#  define CV_HAL_DFT_REAL_OUTPUT    (1 << 4)
#endif

#ifndef CV_HAL_DFT_TWO_STAGE
#  define CV_HAL_DFT_TWO_STAGE      (1 << 5)
#endif

#ifndef CV_HAL_DFT_STAGE_COLS
#  define CV_HAL_DFT_STAGE_COLS     (1 << 6)
#endif

#ifndef CV_HAL_DFT_IS_CONTINUOUS
#  define CV_HAL_DFT_IS_CONTINUOUS  (1 << 7)
#endif

#ifndef CV_HAL_DFT_IS_INPLACE
#  define CV_HAL_DFT_IS_INPLACE     (1 << 8)
#endif

#ifndef CV_32F
#  define CV_32F 5
#endif

#ifndef CV_64F
#  define CV_64F 6
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

#endif  // HAVE_ARMPL

#endif  // OPENCV_ARMPL_HAL_CORE_HPP
