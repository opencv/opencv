// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "accelerate_hal_core.hpp"
#include "opencv2/core/utility.hpp"

#include <Accelerate/Accelerate.h>


int accelerate_hal_meanStdDev(const uchar* src_data, size_t src_step, int width, int height, int src_type,
                              double* mean_val, double* stddev_val, uchar* mask, size_t mask_step)
{
    (void)mask_step;
    if (src_type != CV_32F || mask != nullptr) {
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    const int step = src_step / 4;
    const float *src = reinterpret_cast<const float*>(src_data);
    float mean;
    cv::AutoBuffer<float> means(height);
    for (int i = 0; i < height; ++i, src += step) {
        vDSP_meanv(src, 1, means.data() + i, width);
    }
    vDSP_meanv(means.data(), 1, &mean, height);
    if (mean_val != nullptr) {
        *mean_val = mean;
    }
    if (stddev_val != nullptr) {
        float var;
        src = reinterpret_cast<const float*>(src_data);
        mean *= -1.f;
        cv::AutoBuffer<float> vars(height);
        cv::AutoBuffer<float> tmp(width);
        for (int i = 0; i < height; ++i, src += step) {
            vDSP_vsadd(src, 1, &mean, tmp.data(), 1, width);
            vDSP_measqv(tmp.data(), 1, vars.data() + i, width);
        }
        vDSP_meanv(vars.data(), 1, &var, height);
        *stddev_val = std::sqrt(var);
    }
    return CV_HAL_ERROR_OK;
}
