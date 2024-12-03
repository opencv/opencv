// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#ifndef OPENCV_HAL_RVV_MEANSTDDEV_HPP_INCLUDED
#define OPENCV_HAL_RVV_MEANSTDDEV_HPP_INCLUDED

#include <riscv_vector.h>

namespace cv { namespace cv_hal_rvv {

#undef cv_hal_meanStdDev
#define cv_hal_meanStdDev cv::cv_hal_rvv::meanStdDev

inline int meanStdDev(const uchar* src_data, size_t src_step, int width, int height,
                             int src_type, double* mean_val, double* stddev_val, uchar* mask, size_t mask_step) {
    if (src_type != CV_8UC1)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    double sum = 0.0, sqsum = 0.0;
    size_t total_count = 0;

    for (int i = 0; i < height; ++i)
    {
        const uchar* src_row = src_data + i * src_step;
        const uchar* mask_row = mask ? (mask + i * mask_step) : nullptr;

        for (int j = 0; j < width; ++j)
        {
            if (mask_row && mask_row[j] == 0)
                continue;

            double pixel_value = static_cast<double>(src_row[j]);

            sum += pixel_value;
            sqsum += pixel_value * pixel_value;
            ++total_count;
        }
    }

    if (total_count == 0)
    {
        if (mean_val) *mean_val = 0.0;
        if (stddev_val) *stddev_val = 0.0;
        return CV_HAL_ERROR_OK;
    }

    double mean = sum / total_count;
    double variance = std::max((sqsum / total_count) - (mean * mean), 0.0);
    double stddev = std::sqrt(variance);

    if (mean_val) *mean_val = mean;
    if (stddev_val) *stddev_val = stddev;
    return CV_HAL_ERROR_OK;
}


}}

#endif
