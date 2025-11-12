// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2025, SpaceMIT Inc., all rights reserved.
// Copyright (C) 2025, Institute of Software, Chinese Academy of Sciences.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_HAL_RVV_CORE_COMMON_HPP_INCLUDED
#define OPENCV_HAL_RVV_CORE_COMMON_HPP_INCLUDED

#include <riscv_vector.h>
#include <cmath>
#include <cfloat>

namespace cv { namespace rvv_hal { namespace core { namespace common {

#if CV_HAL_RVV_1P0_ENABLED

#define CV_HAL_RVV_NOOP(a) (a)

// ############ abs ############

#define CV_HAL_RVV_COMMON_CUSTOM_INTRIN_ABS(_Tpvs, _Tpvd, shift, suffix) \
    inline _Tpvd __riscv_vabs(const _Tpvs& v, const int vl) { \
        _Tpvs mask = __riscv_vsra(v, shift, vl); \
        _Tpvs v_xor = __riscv_vxor(v, mask, vl); \
        return __riscv_vreinterpret_##suffix( \
            __riscv_vsub(v_xor, mask, vl) \
        ); \
    }

CV_HAL_RVV_COMMON_CUSTOM_INTRIN_ABS(vint8m2_t,  vuint8m2_t,  7,  u8m2)
CV_HAL_RVV_COMMON_CUSTOM_INTRIN_ABS(vint8m8_t,  vuint8m8_t,  7,  u8m8)
CV_HAL_RVV_COMMON_CUSTOM_INTRIN_ABS(vint16m4_t, vuint16m4_t, 15, u16m4)
CV_HAL_RVV_COMMON_CUSTOM_INTRIN_ABS(vint16m8_t, vuint16m8_t, 15, u16m8)
CV_HAL_RVV_COMMON_CUSTOM_INTRIN_ABS(vint32m4_t, vuint32m4_t, 31, u32m4)
CV_HAL_RVV_COMMON_CUSTOM_INTRIN_ABS(vint32m8_t, vuint32m8_t, 31, u32m8)

// ############ absdiff ############

#define CV_HAL_RVV_COMMON_CUSTOM_INTRIN_ABSDIFF(_Tpvs, _Tpvd, cast, sub, max, min) \
    inline _Tpvd __riscv_vabd(const _Tpvs& v1, const _Tpvs& v2, const int vl) { \
        return cast(__riscv_##sub(__riscv_##max(v1, v2, vl), __riscv_##min(v1, v2, vl), vl)); \
    }

CV_HAL_RVV_COMMON_CUSTOM_INTRIN_ABSDIFF(vuint8m4_t, vuint8m4_t, CV_HAL_RVV_NOOP, vsub, vmaxu, vminu)
CV_HAL_RVV_COMMON_CUSTOM_INTRIN_ABSDIFF(vuint8m8_t, vuint8m8_t, CV_HAL_RVV_NOOP, vsub, vmaxu, vminu)
CV_HAL_RVV_COMMON_CUSTOM_INTRIN_ABSDIFF(vuint16m2_t, vuint16m2_t, CV_HAL_RVV_NOOP, vsub, vmaxu, vminu)
CV_HAL_RVV_COMMON_CUSTOM_INTRIN_ABSDIFF(vuint16m8_t, vuint16m8_t, CV_HAL_RVV_NOOP, vsub, vmaxu, vminu)

CV_HAL_RVV_COMMON_CUSTOM_INTRIN_ABSDIFF(vint8m4_t, vuint8m4_t, __riscv_vreinterpret_u8m4, vsub, vmax, vmin)
CV_HAL_RVV_COMMON_CUSTOM_INTRIN_ABSDIFF(vint8m8_t, vuint8m8_t, __riscv_vreinterpret_u8m8, vsub, vmax, vmin)
CV_HAL_RVV_COMMON_CUSTOM_INTRIN_ABSDIFF(vint16m2_t, vuint16m2_t, __riscv_vreinterpret_u16m2, vsub, vmax, vmin)
CV_HAL_RVV_COMMON_CUSTOM_INTRIN_ABSDIFF(vint16m8_t, vuint16m8_t, __riscv_vreinterpret_u16m8, vsub, vmax, vmin)
CV_HAL_RVV_COMMON_CUSTOM_INTRIN_ABSDIFF(vint32m4_t, vuint32m4_t, __riscv_vreinterpret_u32m4, vsub, vmax, vmin)
CV_HAL_RVV_COMMON_CUSTOM_INTRIN_ABSDIFF(vint32m8_t, vuint32m8_t, __riscv_vreinterpret_u32m8, vsub, vmax, vmin)

// ############ reciprocal ############

inline vfloat32m4_t __riscv_vfrec(const vfloat32m4_t &x, const int vl) {
    auto rec = __riscv_vfrec7(x, vl);
    auto cls = __riscv_vfclass(rec, vl);
    auto m = __riscv_vmseq(__riscv_vand(cls, 0b10111000, vl), 0, vl);
    auto two = __riscv_vfmv_v_f_f32m4(2.f, vl);
    rec = __riscv_vfmul_mu(m, rec, __riscv_vfnmsac(two, x, rec, vl), rec, vl);
    rec = __riscv_vfmul_mu(m, rec, __riscv_vfnmsac(two, x, rec, vl), rec, vl);
    return rec;
}

// ############ atan ############

// ref: mathfuncs_core.simd.hpp
static constexpr float pi = CV_PI;

struct AtanParams
{
    float p1, p3, p5, p7, angle_90;
};

static constexpr AtanParams atan_params_rad {
    0.9997878412794807F,
    -0.3258083974640975F,
    0.1555786518463281F,
    -0.04432655554792128F,
    90.F * (pi / 180.F)};
static constexpr AtanParams atan_params_deg {
    atan_params_rad.p1 * (180 / pi),
    atan_params_rad.p3 * (180 / pi),
    atan_params_rad.p5 * (180 / pi),
    atan_params_rad.p7 * (180 / pi),
    90.F};

template <typename VEC_T>
__attribute__((always_inline)) inline VEC_T
    rvv_atan(VEC_T vy, VEC_T vx, size_t vl, const AtanParams& params)
{
    const auto ax = __riscv_vfabs(vx, vl);
    const auto ay = __riscv_vfabs(vy, vl);
    // Reciprocal Estimate (vfrec7) is not accurate enough to pass the test of cartToPolar.
    const auto c = __riscv_vfdiv(__riscv_vfmin(ax, ay, vl),
                                 __riscv_vfadd(__riscv_vfmax(ax, ay, vl), FLT_EPSILON, vl),
                                 vl);
    const auto c2 = __riscv_vfmul(c, c, vl);

    // Using vfmadd only results in about a 2% performance improvement, but it occupies 3 additional
    // M4 registers. (Performance test on phase32f::VectorLength::1048576: time decreased
    // from 5.952ms to 5.805ms on Muse Pi)
    // Additionally, when registers are nearly fully utilized (though not yet exhausted), the
    // compiler is likely to fail to optimize and may introduce slower memory access (e.g., in
    // cv::rvv_hal::fast_atan_64).
    // Saving registers can also make this function more reusable in other contexts.
    // Therefore, vfmadd is not used here.
    auto a = __riscv_vfadd(__riscv_vfmul(c2, params.p7, vl), params.p5, vl);
    a = __riscv_vfadd(__riscv_vfmul(c2, a, vl), params.p3, vl);
    a = __riscv_vfadd(__riscv_vfmul(c2, a, vl), params.p1, vl);
    a = __riscv_vfmul(a, c, vl);

    a = __riscv_vfrsub_mu(__riscv_vmflt(ax, ay, vl), a, a, params.angle_90, vl);
    a = __riscv_vfrsub_mu(__riscv_vmflt(vx, 0.F, vl), a, a, params.angle_90 * 2, vl);
    a = __riscv_vfrsub_mu(__riscv_vmflt(vy, 0.F, vl), a, a, params.angle_90 * 4, vl);

    return a;
}

// ############ sqrt ############

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

#endif // CV_HAL_RVV_1P0_ENABLED

}}}} // cv::rvv_hal::core::common

#endif // OPENCV_HAL_RVV_CORE_COMMON_HPP_INCLUDED
