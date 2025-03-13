// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2025, Institute of Software, Chinese Academy of Sciences.

#ifndef OPENCV_HAL_RVV_CART_TO_POLAR_HPP_INCLUDED
#define OPENCV_HAL_RVV_CART_TO_POLAR_HPP_INCLUDED

#include <riscv_vector.h>

#include "hal_rvv_1p0/atan.hpp"
#include "hal_rvv_1p0/sqrt.hpp"
#include "hal_rvv_1p0/types.hpp"

namespace cv { namespace cv_hal_rvv {

#undef cv_hal_cartToPolar32f
#define cv_hal_cartToPolar32f cv::cv_hal_rvv::cartToPolar<cv::cv_hal_rvv::RVV_F32M4>
#undef cv_hal_cartToPolar64f
#define cv_hal_cartToPolar64f cv::cv_hal_rvv::cartToPolar<cv::cv_hal_rvv::RVV_F64M8>

template <typename RVV_T, typename T = typename RVV_T::ElemType>
inline int cartToPolar(const T* x, const T* y, T* mag, T* angle, int len, bool angleInDegrees)
{
    using CalType = RVV_SameLen<float, RVV_T>;
    auto atan_params = angleInDegrees ? detail::atan_params_deg : detail::atan_params_rad;
    size_t vl;
    for (; len > 0; len -= (int)vl, x += vl, y += vl, mag += vl, angle += vl)
    {
        vl = RVV_T::setvl(len);

        auto vx = CalType::cast(RVV_T::vload(x, vl), vl);
        auto vy = CalType::cast(RVV_T::vload(y, vl), vl);

        auto vmag = detail::sqrt<2>(__riscv_vfmadd(vx, vx, __riscv_vfmul(vy, vy, vl), vl), vl);
        RVV_T::vstore(mag, RVV_T::cast(vmag, vl), vl);

        auto vangle = detail::rvv_atan(vy, vx, vl, atan_params);
        RVV_T::vstore(angle, RVV_T::cast(vangle, vl), vl);
    }

    return CV_HAL_ERROR_OK;
}

}}  // namespace cv::cv_hal_rvv

#endif  // OPENCV_HAL_RVV_CART_TO_POLAR_HPP_INCLUDED
