// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2025, Institute of Software, Chinese Academy of Sciences.

#ifndef OPENCV_HAL_RVV_MAGNITUDE_HPP_INCLUDED
#define OPENCV_HAL_RVV_MAGNITUDE_HPP_INCLUDED

#include <riscv_vector.h>

#include "hal_rvv_1p0/sqrt.hpp"
#include "hal_rvv_1p0/types.hpp"

namespace cv { namespace cv_hal_rvv {

#undef cv_hal_magnitude32f
#define cv_hal_magnitude32f cv::cv_hal_rvv::magnitude<cv::cv_hal_rvv::Sqrt32f<cv::cv_hal_rvv::RVV_F32M8>>
#undef cv_hal_magnitude64f
#define cv_hal_magnitude64f cv::cv_hal_rvv::magnitude<cv::cv_hal_rvv::Sqrt64f<cv::cv_hal_rvv::RVV_F64M8>>

template <typename SQRT_T, typename T = typename SQRT_T::T::ElemType>
inline int magnitude(const T* x, const T* y, T* dst, int len)
{
    size_t vl;
    for (; len > 0; len -= (int)vl, x += vl, y += vl, dst += vl)
    {
        vl = SQRT_T::T::setvl(len);

        auto vx = SQRT_T::T::vload(x, vl);
        auto vy = SQRT_T::T::vload(y, vl);

        auto vmag = detail::sqrt<SQRT_T::iter_times>(__riscv_vfmadd(vx, vx, __riscv_vfmul(vy, vy, vl), vl), vl);
        SQRT_T::T::vstore(dst, vmag, vl);
    }

    return CV_HAL_ERROR_OK;
}

}}  // namespace cv::cv_hal_rvv

#endif  // OPENCV_HAL_RVV_MAGNITUDE_HPP_INCLUDED
