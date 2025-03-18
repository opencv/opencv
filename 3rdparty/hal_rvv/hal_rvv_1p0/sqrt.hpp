// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level
// directory of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2025, Institute of Software, Chinese Academy of Sciences.

#ifndef OPENCV_HAL_RVV_SQRT_HPP_INCLUDED
#define OPENCV_HAL_RVV_SQRT_HPP_INCLUDED

#include <riscv_vector.h>
#include <cmath>
#include "hal_rvv_1p0/types.hpp"

namespace cv { namespace cv_hal_rvv {

#undef cv_hal_sqrt32f
#undef cv_hal_sqrt64f
#undef cv_hal_invSqrt32f
#undef cv_hal_invSqrt64f

#define cv_hal_sqrt32f cv::cv_hal_rvv::sqrt<cv::cv_hal_rvv::Sqrt32f<cv::cv_hal_rvv::RVV_F32M8>>
#define cv_hal_sqrt64f cv::cv_hal_rvv::sqrt<cv::cv_hal_rvv::Sqrt64f<cv::cv_hal_rvv::RVV_F64M8>>

#ifdef __clang__
// Strange bug in clang: invSqrt use 2 LMUL registers to store mask, which will cause memory access.
// So a smaller LMUL is used here.
#    define cv_hal_invSqrt32f cv::cv_hal_rvv::invSqrt<cv::cv_hal_rvv::Sqrt32f<cv::cv_hal_rvv::RVV_F32M4>>
#    define cv_hal_invSqrt64f cv::cv_hal_rvv::invSqrt<cv::cv_hal_rvv::Sqrt64f<cv::cv_hal_rvv::RVV_F64M4>>
#else
#    define cv_hal_invSqrt32f cv::cv_hal_rvv::invSqrt<cv::cv_hal_rvv::Sqrt32f<cv::cv_hal_rvv::RVV_F32M8>>
#    define cv_hal_invSqrt64f cv::cv_hal_rvv::invSqrt<cv::cv_hal_rvv::Sqrt64f<cv::cv_hal_rvv::RVV_F64M8>>
#endif

namespace detail {

// Newton-Raphson method
// Use 4 LMUL registers
template <size_t iter_times, typename VEC_T>
inline VEC_T sqrt(VEC_T x, size_t vl)
{
    auto x2 = __riscv_vfmul(x, 0.5, vl);
    auto y = __riscv_vfrsqrt7(x, vl);
#ifdef __clang__
#pragma unroll
#endif
    for (size_t i = 0; i < iter_times; i++)
    {
        auto t = __riscv_vfmul(y, y, vl);
        t = __riscv_vfmul(t, x2, vl);
        t = __riscv_vfrsub(t, 1.5, vl);
        y = __riscv_vfmul(t, y, vl);
    }
    // just to prevent the compiler from calculating mask before the iteration, which will run out
    // of registers and cause memory access.
    asm volatile("" ::: "memory");
    auto classified = __riscv_vfclass(x, vl);
    // block -0, +0, positive subnormal number, +inf
    auto mask = __riscv_vmseq(__riscv_vand(classified, 0b10111000, vl), 0, vl);
    return __riscv_vfmul_mu(mask, x, x, y, vl);
}

// Newton-Raphson method
// Use 3 LMUL registers and 1 mask register
template <size_t iter_times, typename VEC_T>
inline VEC_T invSqrt(VEC_T x, size_t vl)
{
    auto classified = __riscv_vfclass(x, vl);
    // block -0, +0, positive subnormal number, +inf
    auto mask = __riscv_vmseq(__riscv_vand(classified, 0b10111000, vl), 0, vl);
    auto x2 = __riscv_vfmul(x, 0.5, vl);
    auto y = __riscv_vfrsqrt7(x, vl);
#ifdef __clang__
#pragma unroll
#endif
    for (size_t i = 0; i < iter_times; i++)
    {
        auto t = __riscv_vfmul(y, y, vl);
        t = __riscv_vfmul(t, x2, vl);
        t = __riscv_vfrsub(t, 1.5, vl);
        y = __riscv_vfmul_mu(mask, y, t, y, vl);
    }
    return y;
}

}  // namespace detail

template <typename RVV_T>
struct Sqrt32f
{
    using T = RVV_T;
    static constexpr size_t iter_times = 2;
};

template <typename RVV_T>
struct Sqrt64f
{
    using T = RVV_T;
    static constexpr size_t iter_times = 3;
};

template <typename SQRT_T, typename Elem = typename SQRT_T::T::ElemType>
inline int sqrt(const Elem* src, Elem* dst, int _len)
{
    size_t vl;
    for (size_t len = _len; len > 0; len -= vl, src += vl, dst += vl)
    {
        vl = SQRT_T::T::setvl(len);
        auto x = SQRT_T::T::vload(src, vl);
        SQRT_T::T::vstore(dst, detail::sqrt<SQRT_T::iter_times>(x, vl), vl);
    }

    return CV_HAL_ERROR_OK;
}

template <typename SQRT_T, typename Elem = typename SQRT_T::T::ElemType>
inline int invSqrt(const Elem* src, Elem* dst, int _len)
{
    size_t vl;
    for (size_t len = _len; len > 0; len -= vl, src += vl, dst += vl)
    {
        vl = SQRT_T::T::setvl(len);
        auto x = SQRT_T::T::vload(src, vl);
        SQRT_T::T::vstore(dst, detail::invSqrt<SQRT_T::iter_times>(x, vl), vl);
    }

    return CV_HAL_ERROR_OK;
}

}}  // namespace cv::cv_hal_rvv

#endif  // OPENCV_HAL_RVV_SQRT_HPP_INCLUDED
