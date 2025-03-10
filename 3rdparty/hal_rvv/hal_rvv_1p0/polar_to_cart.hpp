// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#ifndef OPENCV_HAL_RVV_POLAR_TO_CART_HPP_INCLUDED
#define OPENCV_HAL_RVV_POLAR_TO_CART_HPP_INCLUDED

#include <riscv_vector.h>
#include "hal_rvv_1p0/sincos.hpp"
#include "hal_rvv_1p0/types.hpp"

namespace cv { namespace cv_hal_rvv {

#undef cv_hal_polarToCart32f
#define cv_hal_polarToCart32f cv::cv_hal_rvv::polarToCart<cv::cv_hal_rvv::RVV_F32M4>
#undef cv_hal_polarToCart64f
#define cv_hal_polarToCart64f cv::cv_hal_rvv::polarToCart<cv::cv_hal_rvv::RVV_F64M8>

template <typename RVV_T, typename Elem = typename RVV_T::ElemType>
inline int
    polarToCart(const Elem* mag, const Elem* angle, Elem* x, Elem* y, int len, bool angleInDegrees)
{
    using T = RVV_F32M4;
    const auto sincos_scale = angleInDegrees ? detail::sincos_deg_scale : detail::sincos_rad_scale;

    size_t vl = T::setvlmax();
    auto cos_p2 = T::vmv(detail::sincos_cos_p2, vl);
    auto cos_p0 = T::vmv(detail::sincos_cos_p0, vl);
    for (; len > 0; len -= (int)vl, angle += vl, x += vl, y += vl)
    {
        vl = RVV_T::setvl(len);
        auto vangle = T::cast(RVV_T::vload(angle, vl), vl);
        T::VecType vsin, vcos;
        detail::SinCos32f<T>(vangle, vsin, vcos, sincos_scale, cos_p2, cos_p0, vl);
        if (mag)
        {
            auto vmag = T::cast(RVV_T::vload(mag, vl), vl);
            vsin = __riscv_vfmul(vsin, vmag, vl);
            vcos = __riscv_vfmul(vcos, vmag, vl);
            mag += vl;
        }
        RVV_T::vstore(x, RVV_T::cast(vcos, vl), vl);
        RVV_T::vstore(y, RVV_T::cast(vsin, vl), vl);
    }

    return CV_HAL_ERROR_OK;
}

}}  // namespace cv::cv_hal_rvv

#endif  // OPENCV_HAL_RVV_POLAR_TO_CART_HPP_INCLUDED
