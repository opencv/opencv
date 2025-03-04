#ifndef __IPP_HAL_CORE_HPP__
#define __IPP_HAL_CORE_HPP__

#include <opencv2/core/base.hpp>
#include "ipp_utils.hpp"

#if (IPP_VERSION_X100 >= 700)
int ipp_hal_meanStdDev(const uchar* src_data, size_t src_step, int width, int height, int src_type,
                       double* mean_val, double* stddev_val, uchar* mask, size_t mask_step);

#undef cv_hal_meanStdDev
#define cv_hal_meanStdDev ipp_hal_meanStdDev
#endif

#endif
