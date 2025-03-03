// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#pragma once

#include <riscv_vector.h>

#include "hal_rvv_1p0/atan.hpp"

namespace cv { namespace cv_hal_rvv {

#undef cv_hal_cartToPolar32f
#define cv_hal_cartToPolar32f cv::cv_hal_rvv::cartToPolar<cv::cv_hal_rvv::CartToPolar32f>
#undef cv_hal_cartToPolar64f
#define cv_hal_cartToPolar64f cv::cv_hal_rvv::cartToPolar<cv::cv_hal_rvv::CartToPolar64f>

struct CartToPolar32f
{
    using ElemType = float;
    static inline size_t setvl(size_t len) { return __riscv_vsetvl_e32m4(len); }
    static inline vfloat32m4_t vload(const float* ptr, size_t vl) { return __riscv_vle32_v_f32m4(ptr, vl); }
    static inline void vstore(float* ptr, vfloat32m4_t v, size_t vl) { __riscv_vse32(ptr, v, vl); }
};

struct CartToPolar64f
{
    using ElemType = double;
    static inline size_t setvl(size_t len) { return __riscv_vsetvl_e64m8(len); }
    static inline vfloat32m4_t vload(const double* ptr, size_t vl) { return __riscv_vfncvt_f(__riscv_vle64_v_f64m8(ptr, vl), vl); }
    static inline void vstore(double* ptr, vfloat32m4_t v, size_t vl) { __riscv_vse64(ptr, __riscv_vfwcvt_f(v, vl), vl); }
};

template <typename RVV_T, typename Elem = typename RVV_T::ElemType>
inline int
    cartToPolar(const Elem* x, const Elem* y, Elem* mag, Elem* angle, int len, bool angleInDegrees)
{
    auto atan_params = angleInDegrees ? detail::atan_params_deg : detail::atan_params_rad;
    size_t vl;
    for (; len > 0; len -= (int)vl, x += vl, y += vl, mag += vl, angle += vl)
    {
        vl = RVV_T::setvl(len);

        auto vx = RVV_T::vload(x, vl);
        auto vy = RVV_T::vload(y, vl);

        auto vmag = __riscv_vfsqrt(__riscv_vfmadd(vx, vx, __riscv_vfmul(vy, vy, vl), vl), vl);
        RVV_T::vstore(mag, vmag, vl);

        auto vangle = detail::rvv_atan(vy, vx, vl, atan_params);
        RVV_T::vstore(angle, vangle, vl);
    }

    return CV_HAL_ERROR_OK;
}

}}  // namespace cv::cv_hal_rvv
