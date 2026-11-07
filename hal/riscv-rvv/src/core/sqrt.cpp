// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level
// directory of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2025, Institute of Software, Chinese Academy of Sciences.

#include "rvv_hal.hpp"
#include "common.hpp"

namespace cv { namespace rvv_hal { namespace core {

#if CV_HAL_RVV_1P0_ENABLED

namespace {

template <typename SQRT_T, typename Elem = typename SQRT_T::T::ElemType>
inline int sqrt(const Elem* src, Elem* dst, int _len)
{
    size_t vl;
    for (size_t len = _len; len > 0; len -= vl, src += vl, dst += vl)
    {
        vl = SQRT_T::T::setvl(len);
        auto x = SQRT_T::T::vload(src, vl);
        SQRT_T::T::vstore(dst, common::sqrt<SQRT_T::iter_times>(x, vl), vl);
    }

    return CV_HAL_ERROR_OK;
}

template <typename SQRT_T, typename Elem = typename SQRT_T::T::ElemType>
inline int invSqrt(const Elem* src, Elem* dst, int _len)
{
    size_t vl;
    for (size_t len = _len; len > 0; len -= vl, src += vl, dst += vl)
    {
        vl = SQRT_T::T::setvl(len);
        auto x = SQRT_T::T::vload(src, vl);
        SQRT_T::T::vstore(dst, common::invSqrt<SQRT_T::iter_times>(x, vl), vl);
    }

    return CV_HAL_ERROR_OK;
}

} // anonymous

int sqrt32f(const float* src, float* dst, int len) {
    return sqrt<common::Sqrt32f<RVV_F32M8>>(src, dst, len);
}
int sqrt64f(const double* src, double* dst, int len) {
    return sqrt<common::Sqrt64f<RVV_F64M8>>(src, dst, len);
}

int invSqrt32f(const float* src, float* dst, int len) {
#ifdef __clang__
// Strange bug in clang: invSqrt use 2 LMUL registers to store mask, which will cause memory access.
// So a smaller LMUL is used here.
    return invSqrt<common::Sqrt32f<RVV_F32M4>>(src, dst, len);
#else
    return invSqrt<common::Sqrt32f<RVV_F32M8>>(src, dst, len);
#endif
}
int invSqrt64f(const double* src, double* dst, int len) {
#ifdef __clang__
// Strange bug in clang: invSqrt use 2 LMUL registers to store mask, which will cause memory access.
// So a smaller LMUL is used here.
    return invSqrt<common::Sqrt64f<RVV_F64M4>>(src, dst, len);
#else
    return invSqrt<common::Sqrt64f<RVV_F64M8>>(src, dst, len);
#endif
}

#endif // CV_HAL_RVV_1P0_ENABLED

}}}  // cv::rvv_hal::core
