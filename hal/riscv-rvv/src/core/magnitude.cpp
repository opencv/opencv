// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2025, Institute of Software, Chinese Academy of Sciences.

#include "rvv_hal.hpp"
#include "common.hpp"

namespace cv { namespace rvv_hal { namespace core {

#if CV_HAL_RVV_1P0_ENABLED

namespace {

template <typename SQRT_T, typename T = typename SQRT_T::T::ElemType>
inline int magnitude(const T* x, const T* y, T* dst, int len)
{
    size_t vl;
    for (; len > 0; len -= (int)vl, x += vl, y += vl, dst += vl)
    {
        vl = SQRT_T::T::setvl(len);

        auto vx = SQRT_T::T::vload(x, vl);
        auto vy = SQRT_T::T::vload(y, vl);

        auto vmag = common::sqrt<SQRT_T::iter_times>(__riscv_vfmadd(vx, vx, __riscv_vfmul(vy, vy, vl), vl), vl);
        SQRT_T::T::vstore(dst, vmag, vl);
    }

    return CV_HAL_ERROR_OK;
}

} // anonymous

int magnitude32f(const float *x, const float *y, float *dst, int len) {
    return magnitude<common::Sqrt32f<RVV_F32M8>>(x, y, dst, len);
}
int magnitude64f(const double *x, const double  *y, double *dst, int len) {
    return magnitude<common::Sqrt64f<RVV_F64M8>>(x, y, dst, len);
}

#endif // CV_HAL_RVV_1P0_ENABLED

}}}  // cv::rvv_hal::core
