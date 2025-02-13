// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#ifndef OPENCV_HAL_RVV_INV_SQRT_HPP_INCLUDED
#define OPENCV_HAL_RVV_INV_SQRT_HPP_INCLUDED

#include <riscv_vector.h>

namespace cv { namespace cv_hal_rvv {

#undef cv_hal_invSqrt32f
#define cv_hal_invSqrt32f cv::cv_hal_rvv::invSqrt32f

#undef cv_hal_invSqrt64f
#define cv_hal_invSqrt64f cv::cv_hal_rvv::invSqrt64f

inline int invSqrt32f (const float *src, float *dst, const int len)
{
    int vl = 0;
    for (int i = 0; i < len; i += vl)
    {
        vl = __riscv_vsetvl_e32m4(len - i);
        vfloat32m4_t vres, vsrc = __riscv_vle32_v_f32m4(src + i, vl);
        vres = __riscv_vfsqrt_v_f32m4(vsrc, vl);
        vres =  __riscv_vfrdiv_vf_f32m4(vres, 1.f, vl);
        __riscv_vse32_v_f32m4(dst + i, vres, vl);
    }
    return CV_HAL_ERROR_OK;
}

inline int invSqrt64f (const double *src, double *dst, const int len)
{
    int vl = 0;
    for (int i = 0; i < len; i += vl)
    {
        vl = __riscv_vsetvl_e64m4(len - i);
        vfloat64m4_t vres, vsrc = __riscv_vle64_v_f64m4(src + i, vl);
        vres = __riscv_vfsqrt_v_f64m4(vsrc, vl);
        vres =  __riscv_vfrdiv_vf_f64m4(vres, 1., vl);
        __riscv_vse64_v_f64m4(dst + i, vres, vl);
    }
    return CV_HAL_ERROR_OK;
}

}}
#endif
