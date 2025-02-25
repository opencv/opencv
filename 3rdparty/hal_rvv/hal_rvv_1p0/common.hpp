// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_HAL_RVV_COMMON_HPP_INCLUDED
#define OPENCV_HAL_RVV_COMMON_HPP_INCLUDED

#include <riscv_vector.h>

namespace cv { namespace cv_hal_rvv { namespace custom_intrin {

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

}}} // cv::cv_hal_rvv::custom_intrin

#endif
