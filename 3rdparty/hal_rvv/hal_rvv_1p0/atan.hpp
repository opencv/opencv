// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level
// directory of this distribution and at http://opencv.org/license.html.
#pragma once

#undef cv_hal_fastAtan32f
#define cv_hal_fastAtan32f cv::cv_hal_rvv::fast_atan_32

#undef cv_hal_fastAtan64f
#define cv_hal_fastAtan64f cv::cv_hal_rvv::fast_atan_64

#include <riscv_vector.h>

#include <cfloat>

namespace cv { namespace cv_hal_rvv {

namespace detail {
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
    // cv::cv_hal_rvv::fast_atan_64).
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

}  // namespace detail

inline int fast_atan_32(const float* y, const float* x, float* dst, size_t n, bool angle_in_deg)
{
    auto atan_params = angle_in_deg ? detail::atan_params_deg : detail::atan_params_rad;

    for (size_t vl = 0; n > 0; n -= vl)
    {
        vl = __riscv_vsetvl_e32m4(n);

        auto vy = __riscv_vle32_v_f32m4(y, vl);
        auto vx = __riscv_vle32_v_f32m4(x, vl);

        auto a = detail::rvv_atan(vy, vx, vl, atan_params);

        __riscv_vse32(dst, a, vl);

        x += vl;
        y += vl;
        dst += vl;
    }

    return CV_HAL_ERROR_OK;
}

inline int fast_atan_64(const double* y, const double* x, double* dst, size_t n, bool angle_in_deg)
{
    // this also uses float32 version, ref: mathfuncs_core.simd.hpp

    auto atan_params = angle_in_deg ? detail::atan_params_deg : detail::atan_params_rad;

    for (size_t vl = 0; n > 0; n -= vl)
    {
        vl = __riscv_vsetvl_e64m8(n);

        auto vy = __riscv_vfncvt_f(__riscv_vle64_v_f64m8(y, vl), vl);
        auto vx = __riscv_vfncvt_f(__riscv_vle64_v_f64m8(x, vl), vl);

        auto a = detail::rvv_atan(vy, vx, vl, atan_params);

        __riscv_vse64(dst, __riscv_vfwcvt_f(a, vl), vl);

        x += vl;
        y += vl;
        dst += vl;
    }

    return CV_HAL_ERROR_OK;
}

}}  // namespace cv::cv_hal_rvv
