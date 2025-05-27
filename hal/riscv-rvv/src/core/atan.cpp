// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level
// directory of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2025, Institute of Software, Chinese Academy of Sciences.

#include "rvv_hal.hpp"
#include "common.hpp"

namespace cv { namespace rvv_hal { namespace core {

#if CV_HAL_RVV_1P0_ENABLED

int fast_atan_32(const float* y, const float* x, float* dst, size_t n, bool angle_in_deg)
{
    auto atan_params = angle_in_deg ? common::atan_params_deg : common::atan_params_rad;

    for (size_t vl = 0; n > 0; n -= vl)
    {
        vl = __riscv_vsetvl_e32m4(n);

        auto vy = __riscv_vle32_v_f32m4(y, vl);
        auto vx = __riscv_vle32_v_f32m4(x, vl);

        auto a = common::rvv_atan(vy, vx, vl, atan_params);

        __riscv_vse32(dst, a, vl);

        x += vl;
        y += vl;
        dst += vl;
    }

    return CV_HAL_ERROR_OK;
}

int fast_atan_64(const double* y, const double* x, double* dst, size_t n, bool angle_in_deg)
{
    // this also uses float32 version, ref: mathfuncs_core.simd.hpp

    auto atan_params = angle_in_deg ? common::atan_params_deg : common::atan_params_rad;

    for (size_t vl = 0; n > 0; n -= vl)
    {
        vl = __riscv_vsetvl_e64m8(n);

        auto vy = __riscv_vfncvt_f(__riscv_vle64_v_f64m8(y, vl), vl);
        auto vx = __riscv_vfncvt_f(__riscv_vle64_v_f64m8(x, vl), vl);

        auto a = common::rvv_atan(vy, vx, vl, atan_params);

        __riscv_vse64(dst, __riscv_vfwcvt_f(a, vl), vl);

        x += vl;
        y += vl;
        dst += vl;
    }

    return CV_HAL_ERROR_OK;
}

#endif // CV_HAL_RVV_1P0_ENABLED

}}} // cv::rvv_hal::core
