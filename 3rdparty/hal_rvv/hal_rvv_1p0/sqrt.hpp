// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level
// directory of this distribution and at http://opencv.org/license.html.
#pragma once

#include <riscv_vector.h>
#include <cmath>

namespace cv { namespace cv_hal_rvv {

#undef cv_hal_sqrt32f
#define cv_hal_sqrt32f cv::cv_hal_rvv::sqrt<cv::cv_hal_rvv::Sqrt32f>
#undef cv_hal_sqrt64f
#define cv_hal_sqrt64f cv::cv_hal_rvv::sqrt<cv::cv_hal_rvv::Sqrt64f>
#undef cv_hal_invSqrt32f
#define cv_hal_invSqrt32f cv::cv_hal_rvv::invSqrt<cv::cv_hal_rvv::InvSqrt32f>
#undef cv_hal_invSqrt64f
#define cv_hal_invSqrt64f cv::cv_hal_rvv::invSqrt<cv::cv_hal_rvv::InvSqrt64f>

namespace detail {

// Newton-Raphson method
// Use 4 LMUL registers
template <size_t iter_times, typename VEC_T>
inline VEC_T sqrt(VEC_T x, size_t vl)
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
    // just to prevent the compiler from calculating mask before the invSqrt, which will run out
    // of registers and cause memory access.
    asm volatile("" ::: "memory");
    auto mask = __riscv_vmfne(x, 0.0, vl);
    mask = __riscv_vmfne_mu(mask, mask, x, INFINITY, vl);
    return __riscv_vfmul_mu(mask, x, x, y, vl);
}

// Newton-Raphson method
// Use 3 LMUL registers and 1 mask register
template <size_t iter_times, typename VEC_T>
inline VEC_T invSqrt(VEC_T x, size_t vl)
{
    auto mask = __riscv_vmfne(x, 0.0, vl);
    mask = __riscv_vmfne_mu(mask, mask, x, INFINITY, vl);
    auto x2 = __riscv_vfmul(x, 0.5, vl);
    auto y = __riscv_vfrsqrt7(x, vl);
#pragma unroll
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

struct Sqrt_f32m4
{
    using ElemType = float;
    static constexpr size_t iter_times = 2;
    static inline size_t setvl(size_t len) { return __riscv_vsetvl_e32m4(len); }
    static inline vfloat32m4_t vload(const float* ptr, size_t vl) { return __riscv_vle32_v_f32m4(ptr, vl); }
    static inline void vstore(float* ptr, vfloat32m4_t v, size_t vl) { __riscv_vse32(ptr, v, vl); }
};

struct Sqrt_f32m8
{
    using ElemType = float;
    static constexpr size_t iter_times = 2;
    static inline size_t setvl(size_t len) { return __riscv_vsetvl_e32m8(len); }
    static inline vfloat32m8_t vload(const float* ptr, size_t vl) { return __riscv_vle32_v_f32m8(ptr, vl); }
    static inline void vstore(float* ptr, vfloat32m8_t v, size_t vl) { __riscv_vse32(ptr, v, vl); }
};

struct Sqrt_f64m4
{
    using ElemType = double;
    static constexpr size_t iter_times = 3;
    static inline size_t setvl(size_t len) { return __riscv_vsetvl_e64m4(len); }
    static inline vfloat64m4_t vload(const double* ptr, size_t vl) { return __riscv_vle64_v_f64m4(ptr, vl); }
    static inline void vstore(double* ptr, vfloat64m4_t v, size_t vl) { __riscv_vse64(ptr, v, vl); }
};

struct Sqrt_f64m8
{
    using ElemType = double;
    static constexpr size_t iter_times = 3;
    static inline size_t setvl(size_t len) { return __riscv_vsetvl_e64m8(len); }
    static inline vfloat64m8_t vload(const double* ptr, size_t vl) { return __riscv_vle64_v_f64m8(ptr, vl); }
    static inline void vstore(double* ptr, vfloat64m8_t v, size_t vl) { __riscv_vse64(ptr, v, vl); }
};

using Sqrt32f = Sqrt_f32m8;
using Sqrt64f = Sqrt_f64m8;
#ifdef __clang__
// Strange bug in clang: invSqrt use 2 LMUL registers to store mask, which will cause memory access.
using InvSqrt32f = Sqrt_f32m4;
using InvSqrt64f = Sqrt_f64m4;
#else
using InvSqrt32f = Sqrt_f32m8;
using InvSqrt64f = Sqrt_f64m8;
#endif

template <typename RVV_T, typename Elem = typename RVV_T::ElemType>
inline int sqrt(const Elem* src, Elem* dst, int _len)
{
    size_t vl;
    for (size_t len = _len; len > 0; len -= vl, src += vl, dst += vl)
    {
        vl = RVV_T::setvl(len);
        auto x = RVV_T::vload(src, vl);
        RVV_T::vstore(dst, detail::sqrt<RVV_T::iter_times>(x, vl), vl);
    }

    return CV_HAL_ERROR_OK;
}

template <typename RVV_T, typename Elem = typename RVV_T::ElemType>
inline int invSqrt(const Elem* src, Elem* dst, int _len)
{
    size_t vl;
    for (size_t len = _len; len > 0; len -= vl, src += vl, dst += vl)
    {
        vl = RVV_T::setvl(len);
        auto x = RVV_T::vload(src, vl);
        RVV_T::vstore(dst, detail::invSqrt<RVV_T::iter_times>(x, vl), vl);
    }

    return CV_HAL_ERROR_OK;
}

}}  // namespace cv::cv_hal_rvv
