#ifndef __IPP_HAL_CORE_HPP__
#define __IPP_HAL_CORE_HPP__

#include <opencv2/core/base.hpp>
#include "ipp_utils.hpp"

#if (IPP_VERSION_X100 >= 700)
int ipp_hal_meanStdDev(const uchar* src_data, size_t src_step, int width, int height, int src_type,
                       double* mean_val, double* stddev_val, uchar* mask, size_t mask_step);

#undef cv_hal_meanStdDev
#define cv_hal_meanStdDev ipp_hal_meanStdDev

int ipp_hal_minMaxIdxMaskStep(const uchar* src_data, size_t src_step, int width, int height, int depth,
                              double* _minVal, double* _maxVal, int* _minIdx, int* _maxIdx, uchar* mask, size_t mask_step);

#undef cv_hal_minMaxIdxMaskStep
#define cv_hal_minMaxIdxMaskStep ipp_hal_minMaxIdxMaskStep

#define IPP_DISABLE_NORM_8U             1 // accuracy difference in perf test sanity check

int ipp_hal_norm(const uchar* src, size_t src_step, const uchar* mask, size_t mask_step,
                 int width, int height, int type, int norm_type, double* result);

#undef cv_hal_norm
#define cv_hal_norm ipp_hal_norm


int ipp_hal_normDiff(const uchar* src1, size_t src1_step, const uchar* src2, size_t src2_step, const uchar* mask,
                     size_t mask_step, int width, int height, int type, int norm_type, double* result);

#undef cv_hal_normDiff
#define cv_hal_normDiff ipp_hal_normDiff

#endif

int ipp_hal_polarToCart32f(const float* mag, const float* angle, float* x, float* y, int len, bool angleInDegrees);
int ipp_hal_polarToCart64f(const double* mag, const double* angle, double* x, double* y, int len, bool angleInDegrees);

#undef cv_hal_polarToCart32f
#define cv_hal_polarToCart32f ipp_hal_polarToCart32f
#undef cv_hal_polarToCart64f
#define cv_hal_polarToCart64f ipp_hal_polarToCart64f

#ifdef HAVE_IPP_IW
int ipp_hal_flip(int src_type, const uchar* src_data, size_t src_step, int src_width, int src_height,
                 uchar* dst_data, size_t dst_step, int flip_mode);

#undef cv_hal_flip
#define cv_hal_flip ipp_hal_flip
#endif

int ipp_hal_transpose2d(const uchar* src_data, size_t src_step, uchar* dst_data, size_t dst_step, int src_width,
                        int src_height, int element_size);

#undef cv_hal_transpose2d
#define cv_hal_transpose2d ipp_hal_transpose2d

#endif
