// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2025, Institute of Software, Chinese Academy of Sciences.

#include "rvv_hal.hpp"
#include "common.hpp"

namespace cv { namespace rvv_hal { namespace core {

#if CV_HAL_RVV_1P0_ENABLED

namespace {

template <typename RVV_T, typename T = typename RVV_T::ElemType>
inline int cartToPolar(const T* x, const T* y, T* mag, T* angle, int len, bool angleInDegrees)
{
    using CalType = RVV_SameLen<float, RVV_T>;
    auto atan_params = angleInDegrees ? common::atan_params_deg : common::atan_params_rad;
    size_t vl;
    for (; len > 0; len -= (int)vl, x += vl, y += vl, mag += vl, angle += vl)
    {
        vl = RVV_T::setvl(len);

        auto vx = CalType::cast(RVV_T::vload(x, vl), vl);
        auto vy = CalType::cast(RVV_T::vload(y, vl), vl);

        auto vmag = common::sqrt<2>(__riscv_vfmadd(vx, vx, __riscv_vfmul(vy, vy, vl), vl), vl);
        RVV_T::vstore(mag, RVV_T::cast(vmag, vl), vl);

        auto vangle = common::rvv_atan(vy, vx, vl, atan_params);
        RVV_T::vstore(angle, RVV_T::cast(vangle, vl), vl);
    }

    return CV_HAL_ERROR_OK;
}

} // anonymous

int cartToPolar32f(const float* x, const float* y, float* mag, float* angle, int len, bool angleInDegrees) {
    return cartToPolar<RVV_F32M4>(x, y, mag, angle, len, angleInDegrees);
}
int cartToPolar64f(const double* x, const double* y, double* mag, double* angle, int len, bool angleInDegrees) {
    return cartToPolar<RVV_F64M8>(x, y, mag, angle, len, angleInDegrees);
}

#endif // CV_HAL_RVV_1P0_ENABLED

}}}  // cv::rvv_hal::core
