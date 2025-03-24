#ifndef __IPP_HAL_CORE_HPP__
#define __IPP_HAL_CORE_HPP__

#include <opencv2/core/base.hpp>
#include "ipp_utils.hpp"

#if (IPP_VERSION_X100 >= 700)
int ipp_hal_meanStdDev(const uchar* src_data, size_t src_step, int width, int height, int src_type,
                       double* mean_val, double* stddev_val, uchar* mask, size_t mask_step);

#undef cv_hal_meanStdDev
#define cv_hal_meanStdDev ipp_hal_meanStdDev

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

#endif
