// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#pragma once

#include <riscv_vector.h>

namespace cv { namespace cv_hal_rvv {

#undef cv_hal_magnitude32f
#define cv_hal_magnitude32f cv::cv_hal_rvv::magnitude32f
#undef cv_hal_magnitude64f
#define cv_hal_magnitude64f cv::cv_hal_rvv::magnitude64f

inline int magnitude32f(const float* x, const float* y, float* dst, int len)
{
    size_t vl;
    for (; len > 0; len -= (int)vl, x += vl, y += vl, dst += vl)
    {
        vl = __riscv_vsetvl_e32m4(len);

        auto vx = __riscv_vle32_v_f32m4(x, vl);
        auto vy = __riscv_vle32_v_f32m4(y, vl);

        auto vmag = __riscv_vfsqrt(__riscv_vfmadd(vx, vx, __riscv_vfmul(vy, vy, vl), vl), vl);
        __riscv_vse32(dst, vmag, vl);
    }

    return CV_HAL_ERROR_OK;
}

inline int magnitude64f(const double* x, const double* y, double* dst, int len)
{
    size_t vl;
    for (; len > 0; len -= (int)vl, x += vl, y += vl, dst += vl)
    {
        vl = __riscv_vsetvl_e64m8(len);

        auto vx = __riscv_vfncvt_f(__riscv_vle64_v_f64m8(x, vl), vl);
        auto vy = __riscv_vfncvt_f(__riscv_vle64_v_f64m8(y, vl), vl);

        auto vmag = __riscv_vfsqrt(__riscv_vfmadd(vx, vx, __riscv_vfmul(vy, vy, vl), vl), vl);
        __riscv_vse64(dst, __riscv_vfwcvt_f(vmag, vl), vl);
    }

    return CV_HAL_ERROR_OK;
}

}}  // namespace cv::cv_hal_rvv
