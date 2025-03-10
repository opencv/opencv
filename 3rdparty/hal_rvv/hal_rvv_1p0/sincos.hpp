// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level
// directory of this distribution and at http://opencv.org/license.html.
#ifndef OPENCV_HAL_RVV_SINCOS_HPP_INCLUDED
#define OPENCV_HAL_RVV_SINCOS_HPP_INCLUDED

#include <riscv_vector.h>
#include "hal_rvv_1p0/types.hpp"

namespace cv { namespace cv_hal_rvv { namespace detail {

static constexpr size_t sincos_mask = 0x3;

static constexpr float sincos_rad_scale = 2.f / CV_PI;
static constexpr float sincos_deg_scale = 2.f / 180.f;

// Taylor expansion coefficients for sin(x*pi/2) and cos(x*pi/2)
static constexpr double sincos_sin_p7 = -0.004681754135319;
static constexpr double sincos_sin_p5 = 0.079692626246167;
static constexpr double sincos_sin_p3 = -0.645964097506246;
static constexpr double sincos_sin_p1 = 1.570796326794897;

static constexpr double sincos_cos_p8 = 0.000919260274839;
static constexpr double sincos_cos_p6 = -0.020863480763353;
static constexpr double sincos_cos_p4 = 0.253669507901048;
static constexpr double sincos_cos_p2 = -1.233700550136170;
static constexpr double sincos_cos_p0 = 1.000000000000000;

// Taylor expansion and angle sum identity
// Use 7 LMUL registers (can be reduced to 5 by splitting fmadd to fadd and fmul)
template <typename RVV_T, typename T = typename RVV_T::VecType>
static inline void
    SinCos32f(T angle, T& sinval, T& cosval, float scale, T cos_p2, T cos_p0, size_t vl)
{
    angle = __riscv_vfmul(angle, scale, vl);
    auto round_angle = RVV_ToInt<RVV_T>::cast(angle, vl);
    auto delta_angle = __riscv_vfsub(angle, RVV_T::cast(round_angle, vl), vl);
    auto delta_angle2 = __riscv_vfmul(delta_angle, delta_angle, vl);

    auto sin = __riscv_vfadd(__riscv_vfmul(delta_angle2, sincos_sin_p7, vl), sincos_sin_p5, vl);
    sin = __riscv_vfadd(__riscv_vfmul(delta_angle2, sin, vl), sincos_sin_p3, vl);
    sin = __riscv_vfadd(__riscv_vfmul(delta_angle2, sin, vl), sincos_sin_p1, vl);
    sin = __riscv_vfmul(delta_angle, sin, vl);

    auto cos = __riscv_vfadd(__riscv_vfmul(delta_angle2, sincos_cos_p8, vl), sincos_cos_p6, vl);
    cos = __riscv_vfadd(__riscv_vfmul(delta_angle2, cos, vl), sincos_cos_p4, vl);
    cos = __riscv_vfmadd(cos, delta_angle2, cos_p2, vl);
    cos = __riscv_vfmadd(cos, delta_angle2, cos_p0, vl);

    // idx = 0: sinval =  sin, cosval =  cos
    // idx = 1: sinval =  cos, cosval = -sin
    // idx = 2: sinval = -sin, cosval = -cos
    // idx = 3: sinval = -cos, cosval =  sin
    auto idx = __riscv_vand(round_angle, sincos_mask, vl);
    auto idx1 = __riscv_vmseq(idx, 1, vl);
    auto idx2 = __riscv_vmseq(idx, 2, vl);
    auto idx3 = __riscv_vmseq(idx, 3, vl);

    auto idx13 = __riscv_vmor(idx1, idx3, vl);
    sinval = __riscv_vmerge(sin, cos, idx13, vl);
    cosval = __riscv_vmerge(cos, sin, idx13, vl);

    sinval = __riscv_vfneg_mu(__riscv_vmor(idx2, idx3, vl), sinval, sinval, vl);
    cosval = __riscv_vfneg_mu(__riscv_vmor(idx1, idx2, vl), cosval, cosval, vl);
}

}}}  // namespace cv::cv_hal_rvv::detail

#endif  // OPENCV_HAL_RVV_SINCOS_HPP_INCLUDED
