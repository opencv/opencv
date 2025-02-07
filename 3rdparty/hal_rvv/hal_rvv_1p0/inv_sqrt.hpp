// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#ifndef OPENCV_HAL_RVV_INV_SQRTF32_HPP_INCLUDED
#define OPENCV_HAL_RVV_INV_SQRTF32_HPP_INCLUDED

#include <riscv_vector.h>

namespace cv { namespace cv_hal_rvv {

#undef cv_hal_invSqrt32f
#define cv_hal_invSqrt32f cv::cv_hal_rvv::invSqrt32f

inline int invSqrt32f (const float *src, float *dst, const int len) {
    const size_t vl = __riscv_vsetvl_e32m8(len);
    const size_t remainings = len % vl;
    auto calc_fun = [&](const size_t i, const size_t vl) {
        vfloat32m8_t vsrc = __riscv_vle32_v_f32m8(&src[i], vl), 
                    vres;
        vres = __riscv_vfsqrt_v_f32m8(vsrc, vl);
        vres =  __riscv_vfrdiv_vf_f32m8(vres, 1., vl);
        __riscv_vse32_v_f32m8(&dst[i], vres, vl);
    };

    size_t i = 0;
    for (; i < len - remainings; i += vl)
        calc_fun(i, vl);
    if (remainings) {
        size_t tail_len = __riscv_vsetvl_e32m8(len - i);
        calc_fun(i, tail_len);
    }
    return CV_HAL_ERROR_OK;
}

}}
#endif
