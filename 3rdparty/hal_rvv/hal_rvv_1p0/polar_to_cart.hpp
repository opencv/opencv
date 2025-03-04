// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#pragma once

#include <riscv_vector.h>
#include "hal_rvv_1p0/sincos.hpp"

namespace cv { namespace cv_hal_rvv {

#undef cv_hal_polarToCart32f
#define cv_hal_polarToCart32f cv::cv_hal_rvv::polarToCart<cv::cv_hal_rvv::PolarToCart32f>
#undef cv_hal_polarToCart64f
#define cv_hal_polarToCart64f cv::cv_hal_rvv::polarToCart<cv::cv_hal_rvv::PolarToCart64f>

struct PolarToCart32f
{
    using ElemType = float;
    static inline size_t setvl(size_t len) { return __riscv_vsetvl_e32m4(len); }
    static inline vfloat32m4_t vload(const float* ptr, size_t vl) { return __riscv_vle32_v_f32m4(ptr, vl); }
    static inline void vstore(float* ptr, vfloat32m4_t v, size_t vl) { __riscv_vse32(ptr, v, vl); }
};

struct PolarToCart64f
{
    using ElemType = double;
    static inline size_t setvl(size_t len) { return __riscv_vsetvl_e64m8(len); }
    static inline vfloat32m4_t vload(const double* ptr, size_t vl) { return __riscv_vfncvt_f(__riscv_vle64_v_f64m8(ptr, vl), vl); }
    static inline void vstore(double* ptr, vfloat32m4_t v, size_t vl) { __riscv_vse64(ptr, __riscv_vfwcvt_f(v, vl), vl); }
};

template <typename RVV_T, typename Vlen, typename Elem = typename RVV_T::ElemType>
inline void polarToCartLoop(const Elem* mag,
                            const Elem* angle,
                            Elem* x,
                            Elem* y,
                            int len,
                            bool angleInDegrees)
{
    const auto sincos_scale = angleInDegrees ? detail::sincos_deg_scale : detail::sincos_rad_scale;
    const auto sincos_table = detail::SinCosLoadTab<Vlen>();

    size_t vl;
    for (; len > 0; len -= (int)vl, angle += vl, x += vl, y += vl)
    {
        vl = RVV_T::setvl(len);
        vfloat32m4_t vangle = RVV_T::vload(angle, vl);
        vfloat32m4_t vsin, vcos;
        detail::SinCos32f<Vlen>(vangle, sincos_scale, sincos_table, vl, vsin, vcos);
        if (mag)
        {
            vfloat32m4_t vmag = RVV_T::vload(mag, vl);
            vsin = __riscv_vfmul(vsin, vmag, vl);
            vcos = __riscv_vfmul(vcos, vmag, vl);
            mag += vl;
        }
        RVV_T::vstore(x, vcos, vl);
        RVV_T::vstore(y, vsin, vl);
    }
}

template <typename RVV_T, typename Elem = typename RVV_T::ElemType>
inline int
    polarToCart(const Elem* mag, const Elem* angle, Elem* x, Elem* y, int len, bool angleInDegrees)
{
    size_t vlen = __riscv_vlenb() * 8;
    if (vlen >= 512)
        polarToCartLoop<RVV_T, detail::SinCosVlen512>(mag, angle, x, y, len, angleInDegrees);
    else if (vlen >= 256)
        polarToCartLoop<RVV_T, detail::SinCosVlen256>(mag, angle, x, y, len, angleInDegrees);
    else if (vlen >= 128)
        polarToCartLoop<RVV_T, detail::SinCosVlen128>(mag, angle, x, y, len, angleInDegrees);
    else
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    return CV_HAL_ERROR_OK;
}

}}  // namespace cv::cv_hal_rvv
