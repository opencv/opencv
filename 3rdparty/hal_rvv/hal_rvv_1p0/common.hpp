// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2025, SpaceMIT Inc., all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_HAL_RVV_COMMON_HPP_INCLUDED
#define OPENCV_HAL_RVV_COMMON_HPP_INCLUDED

#include <riscv_vector.h>

namespace cv { namespace cv_hal_rvv { namespace custom_intrin {

#define CV_HAL_RVV_NOOP(a) (a)

#define CV_HAL_RVV_COMMON_CUSTOM_INTRIN_ABS(_Tpvs, _Tpvd, shift, suffix) \
    inline _Tpvd __riscv_vabs(const _Tpvs& v, const int vl) { \
        _Tpvs mask = __riscv_vsra(v, shift, vl); \
        _Tpvs v_xor = __riscv_vxor(v, mask, vl); \
        return __riscv_vreinterpret_##suffix( \
            __riscv_vsub(v_xor, mask, vl) \
        ); \
    }

CV_HAL_RVV_COMMON_CUSTOM_INTRIN_ABS(vint8m2_t,  vuint8m2_t,  7,  u8m2)
CV_HAL_RVV_COMMON_CUSTOM_INTRIN_ABS(vint8m8_t,  vuint8m8_t,  7,  u8m8)
CV_HAL_RVV_COMMON_CUSTOM_INTRIN_ABS(vint16m4_t, vuint16m4_t, 15, u16m4)
CV_HAL_RVV_COMMON_CUSTOM_INTRIN_ABS(vint16m8_t, vuint16m8_t, 15, u16m8)
CV_HAL_RVV_COMMON_CUSTOM_INTRIN_ABS(vint32m4_t, vuint32m4_t, 31, u32m4)
CV_HAL_RVV_COMMON_CUSTOM_INTRIN_ABS(vint32m8_t, vuint32m8_t, 31, u32m8)

#define CV_HAL_RVV_COMMON_CUSTOM_INTRIN_ABSDIFF(_Tpvs, _Tpvd, cast, sub, max, min) \
    inline _Tpvd __riscv_vabd(const _Tpvs& v1, const _Tpvs& v2, const int vl) { \
        return cast(__riscv_##sub(__riscv_##max(v1, v2, vl), __riscv_##min(v1, v2, vl), vl)); \
    }

CV_HAL_RVV_COMMON_CUSTOM_INTRIN_ABSDIFF(vuint8m4_t, vuint8m4_t, CV_HAL_RVV_NOOP, vsub, vmaxu, vminu)
CV_HAL_RVV_COMMON_CUSTOM_INTRIN_ABSDIFF(vuint8m8_t, vuint8m8_t, CV_HAL_RVV_NOOP, vsub, vmaxu, vminu)
CV_HAL_RVV_COMMON_CUSTOM_INTRIN_ABSDIFF(vuint16m2_t, vuint16m2_t, CV_HAL_RVV_NOOP, vsub, vmaxu, vminu)
CV_HAL_RVV_COMMON_CUSTOM_INTRIN_ABSDIFF(vuint16m8_t, vuint16m8_t, CV_HAL_RVV_NOOP, vsub, vmaxu, vminu)

CV_HAL_RVV_COMMON_CUSTOM_INTRIN_ABSDIFF(vint8m4_t, vuint8m4_t, __riscv_vreinterpret_u8m4, vsub, vmax, vmin)
CV_HAL_RVV_COMMON_CUSTOM_INTRIN_ABSDIFF(vint8m8_t, vuint8m8_t, __riscv_vreinterpret_u8m8, vsub, vmax, vmin)
CV_HAL_RVV_COMMON_CUSTOM_INTRIN_ABSDIFF(vint16m2_t, vuint16m2_t, __riscv_vreinterpret_u16m2, vsub, vmax, vmin)
CV_HAL_RVV_COMMON_CUSTOM_INTRIN_ABSDIFF(vint16m8_t, vuint16m8_t, __riscv_vreinterpret_u16m8, vsub, vmax, vmin)
CV_HAL_RVV_COMMON_CUSTOM_INTRIN_ABSDIFF(vint32m4_t, vuint32m4_t, __riscv_vreinterpret_u32m4, vsub, vmax, vmin)
CV_HAL_RVV_COMMON_CUSTOM_INTRIN_ABSDIFF(vint32m8_t, vuint32m8_t, __riscv_vreinterpret_u32m8, vsub, vmax, vmin)

#define CV_HAL_RVV_COMMON_CUSTOM_INTRIN_RECIPROCAL(_Tpv, suffix) \
    inline _Tpv __riscv_vfrecprocal(const _Tpv& a, const int vl) { \
        _Tpv recp = __riscv_vfrec7_v_##suffix(a, vl); \
        _Tpv two = __riscv_vfmv_v_f_##suffix(2.f, vl); \
        recp = __riscv_vfmul_vv_##suffix(__riscv_vfnmsac_vv_##suffix(two, a, recp, vl), \
                                      recp, vl); \
        recp = __riscv_vfmul_vv_##suffix(__riscv_vfnmsac_vv_##suffix(two, a, recp, vl), \
                                      recp, vl); \
        return recp; \
    }

CV_HAL_RVV_COMMON_CUSTOM_INTRIN_RECIPROCAL(vfloat32m4_t, f32m4)

}}} // cv::cv_hal_rvv::custom_intrin

#endif
