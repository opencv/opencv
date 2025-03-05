// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level
// directory of this distribution and at http://opencv.org/license.html.
#pragma once

#include <riscv_vector.h>

namespace cv { namespace cv_hal_rvv {

#undef cv_hal_sqrt32f
#define cv_hal_sqrt32f cv::cv_hal_rvv::sqrt32f
#undef cv_hal_sqrt64f
#define cv_hal_sqrt64f cv::cv_hal_rvv::sqrt64f
#undef cv_hal_invSqrt32f
#define cv_hal_invSqrt32f cv::cv_hal_rvv::invSqrt32f
#undef cv_hal_invSqrt64f
#define cv_hal_invSqrt64f cv::cv_hal_rvv::invSqrt64f

namespace detail {

// Newton-Raphson method
template <size_t iter_times, typename VEC_T>
inline VEC_T invSqrt(VEC_T x, size_t vl)
{
    auto x2 = __riscv_vfmul(x, 0.5, vl);
    auto y = __riscv_vfrsqrt7(x, vl);
#pragma unroll
    for (size_t i = 0; i < iter_times; i++)
    {
        auto t = __riscv_vfmul(y, y, vl);
        t = __riscv_vfmul(t, x2, vl);
        t = __riscv_vfrsub(t, 1.5, vl);
        y = __riscv_vfmul(t, y, vl);
    }
    return y;
}

// Newton-Raphson method
template <size_t iter_times, typename VEC_T>
inline VEC_T invSqrt(VEC_T x, VEC_T half_three, size_t vl)
{
    auto x2 = __riscv_vfmul(x, 0.5, vl);
    auto y = __riscv_vfrsqrt7(x, vl);
#pragma unroll
    for (size_t i = 0; i < iter_times; i++)
    {
        auto t = __riscv_vfmul(y, y, vl);
        t = __riscv_vfnmsub(t, x2, half_three, vl);
        y = __riscv_vfmul(t, y, vl);
    }
    return y;
}

}  // namespace detail

inline int sqrt32f(const float* src, float* dst, int _len)
{
    size_t vl = __riscv_vsetvlmax_e32m8();
    for (size_t len = _len; len > 0; len -= vl, src += vl, dst += vl)
    {
        vl = __riscv_vsetvl_e32m8(len);
        auto x = __riscv_vle32_v_f32m8(src, vl);
        auto y = __riscv_vfmul(x, detail::invSqrt<2>(x, vl), vl);
        __riscv_vse32(dst, y, vl);
    }

    return CV_HAL_ERROR_OK;
}

inline int sqrt64f(const double* src, double* dst, int _len)
{
    size_t vl = __riscv_vsetvlmax_e64m8();
    for (size_t len = _len; len > 0; len -= vl, src += vl, dst += vl)
    {
        vl = __riscv_vsetvl_e64m8(len);
        auto x = __riscv_vle64_v_f64m8(src, vl);
        auto y = __riscv_vfmul(x, detail::invSqrt<3>(x, vl), vl);
        __riscv_vse64(dst, y, vl);
    }

    return CV_HAL_ERROR_OK;
}

inline int invSqrt32f(const float* src, float* dst, int _len)
{
    size_t vl = __riscv_vsetvlmax_e32m8();
    auto half_three = __riscv_vfmv_v_f_f32m8(1.5, vl);
    for (size_t len = _len; len > 0; len -= vl, src += vl, dst += vl)
    {
        vl = __riscv_vsetvl_e32m8(len);
        auto x = __riscv_vle32_v_f32m8(src, vl);
        __riscv_vse32(dst, detail::invSqrt<2>(x, half_three, vl), vl);
    }

    return CV_HAL_ERROR_OK;
}

// Newton-Raphson method
inline int invSqrt64f(const double* src, double* dst, int _len)
{
    size_t vl = __riscv_vsetvlmax_e64m8();
    auto half_three = __riscv_vfmv_v_f_f64m8(1.5, vl);
    for (size_t len = _len; len > 0; len -= vl, src += vl, dst += vl)
    {
        vl = __riscv_vsetvl_e64m8(len);
        auto x = __riscv_vle64_v_f64m8(src, vl);
        __riscv_vse64(dst, detail::invSqrt<3>(x, half_three, vl), vl);
    }

    return CV_HAL_ERROR_OK;
}

}}  // namespace cv::cv_hal_rvv
