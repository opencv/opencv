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

namespace cv::cv_hal_rvv {

namespace detail {
// ref: mathfuncs_core.simd.hpp
static constexpr float pi = CV_PI;
static constexpr float atan2_p1 = 0.9997878412794807F * (180 / pi);
static constexpr float atan2_p3 = -0.3258083974640975F * (180 / pi);
static constexpr float atan2_p5 = 0.1555786518463281F * (180 / pi);
static constexpr float atan2_p7 = -0.04432655554792128F * (180 / pi);

__attribute__((always_inline)) inline vfloat32m4_t
rvv_atan_f32(vfloat32m4_t vy, vfloat32m4_t vx, size_t vl, float p7,
             vfloat32m4_t vp5, vfloat32m4_t vp3, vfloat32m4_t vp1,
             float angle_90_deg) {
    const auto ax = __riscv_vfabs(vx, vl);
    const auto ay = __riscv_vfabs(vy, vl);
    const auto c = __riscv_vfdiv(
        __riscv_vfmin(ax, ay, vl),
        __riscv_vfadd(__riscv_vfmax(ax, ay, vl), FLT_EPSILON, vl), vl);
    const auto c2 = __riscv_vfmul(c, c, vl);

    auto a = __riscv_vfmadd(c2, p7, vp5, vl);
    a = __riscv_vfmadd(a, c2, vp3, vl);
    a = __riscv_vfmadd(a, c2, vp1, vl);
    a = __riscv_vfmul(a, c, vl);

    const auto mask = __riscv_vmflt(ax, ay, vl);
    a = __riscv_vfrsub_mu(mask, a, a, angle_90_deg, vl);

    a = __riscv_vfrsub_mu(__riscv_vmflt(vx, 0.F, vl), a, a, angle_90_deg * 2,
                          vl);
    a = __riscv_vfrsub_mu(__riscv_vmflt(vy, 0.F, vl), a, a, angle_90_deg * 4,
                          vl);

    return a;
}

} // namespace detail

inline int fast_atan_32(const float *y, const float *x, float *dst, size_t n,
                        bool angle_in_deg) {
    const float scale = angle_in_deg ? 1.f : CV_PI / 180.f;
    const float p1 = detail::atan2_p1 * scale;
    const float p3 = detail::atan2_p3 * scale;
    const float p5 = detail::atan2_p5 * scale;
    const float p7 = detail::atan2_p7 * scale;
    const float angle_90_deg = 90.F * scale;

    static size_t vlmax = __riscv_vsetvlmax_e32m4();
    auto vp1 = __riscv_vfmv_v_f_f32m4(p1, vlmax);
    auto vp3 = __riscv_vfmv_v_f_f32m4(p3, vlmax);
    auto vp5 = __riscv_vfmv_v_f_f32m4(p5, vlmax);

    for (size_t vl{}; n > 0; n -= vl) {
        vl = __riscv_vsetvl_e32m4(n);

        auto vy = __riscv_vle32_v_f32m4(y, vl);
        auto vx = __riscv_vle32_v_f32m4(x, vl);

        auto a =
            detail::rvv_atan_f32(vy, vx, vl, p7, vp5, vp3, vp1, angle_90_deg);

        __riscv_vse32(dst, a, vl);

        x += vl;
        y += vl;
        dst += vl;
    }

    return CV_HAL_ERROR_OK;
}

inline int fast_atan_64(const double *y, const double *x, double *dst, size_t n,
                        bool angle_in_deg) {
    // this also uses float32 version, ref: mathfuncs_core.simd.hpp

    const float scale = angle_in_deg ? 1.f : CV_PI / 180.f;
    const float p1 = detail::atan2_p1 * scale;
    const float p3 = detail::atan2_p3 * scale;
    const float p5 = detail::atan2_p5 * scale;
    const float p7 = detail::atan2_p7 * scale;
    const float angle_90_deg = 90.F * scale;

    static size_t vlmax = __riscv_vsetvlmax_e32m4();
    auto vp1 = __riscv_vfmv_v_f_f32m4(p1, vlmax);
    auto vp3 = __riscv_vfmv_v_f_f32m4(p3, vlmax);
    auto vp5 = __riscv_vfmv_v_f_f32m4(p5, vlmax);

    for (size_t vl{}; n > 0; n -= vl) {
        vl = __riscv_vsetvl_e64m8(n);

        auto wy = __riscv_vle64_v_f64m8(y, vl);
        auto wx = __riscv_vle64_v_f64m8(x, vl);

        auto vy = __riscv_vfncvt_f_f_w_f32m4(wy, vl);
        auto vx = __riscv_vfncvt_f_f_w_f32m4(wx, vl);

        auto a =
            detail::rvv_atan_f32(vy, vx, vl, p7, vp5, vp3, vp1, angle_90_deg);

        auto wa = __riscv_vfwcvt_f_f_v_f64m8(a, vl);

        __riscv_vse64(dst, wa, vl);

        x += vl;
        y += vl;
        dst += vl;
    }

    return CV_HAL_ERROR_OK;
}

} // namespace cv::cv_hal_rvv
