// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#ifndef OPENCV_HAL_RVV_SPLIT_HPP_INCLUDED
#define OPENCV_HAL_RVV_SPLIT_HPP_INCLUDED

#include <riscv_vector.h>

namespace cv { namespace cv_hal_rvv {

#undef cv_hal_split8u
#define cv_hal_split8u cv::cv_hal_rvv::split8u

inline int split8u(const uchar* src, uchar** dst, int len, int cn)
{
    if (cn > 8)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    int vl = 0;
    if (cn == 1)
    {
        uchar* dst0 = dst[0];
        for (int i = 0; i < len; i += vl)
        {
            vl = __riscv_vsetvl_e8m8(len - i);
            __riscv_vse8_v_u8m8(dst0 + i, __riscv_vle8_v_u8m8(src + i, vl), vl);
        }
    }
    else if (cn == 2)
    {
        uchar *dst0 = dst[0], *dst1 = dst[1];
        for (int i = 0; i < len; i += vl)
        {
            vl = __riscv_vsetvl_e8m4(len - i);
            vuint8m4x2_t seg = __riscv_vlseg2e8_v_u8m4x2(src + i * cn, vl);
            __riscv_vse8_v_u8m4(dst0 + i, __riscv_vget_v_u8m4x2_u8m4(seg, 0), vl);
            __riscv_vse8_v_u8m4(dst1 + i, __riscv_vget_v_u8m4x2_u8m4(seg, 1), vl);
        }
    }
    else if (cn == 3)
    {
        uchar *dst0 = dst[0], *dst1 = dst[1], *dst2 = dst[2];
        for (int i = 0; i < len; i += vl)
        {
            vl = __riscv_vsetvl_e8m2(len - i);
            vuint8m2x3_t seg = __riscv_vlseg3e8_v_u8m2x3(src + i * cn, vl);
            __riscv_vse8_v_u8m2(dst0 + i, __riscv_vget_v_u8m2x3_u8m2(seg, 0), vl);
            __riscv_vse8_v_u8m2(dst1 + i, __riscv_vget_v_u8m2x3_u8m2(seg, 1), vl);
            __riscv_vse8_v_u8m2(dst2 + i, __riscv_vget_v_u8m2x3_u8m2(seg, 2), vl);
        }
    }
    else if (cn == 4)
    {
        uchar *dst0 = dst[0], *dst1 = dst[1], *dst2 = dst[2], *dst3 = dst[3];
        for (int i = 0; i < len; i += vl)
        {
            vl = __riscv_vsetvl_e8m2(len - i);
            vuint8m2x4_t seg = __riscv_vlseg4e8_v_u8m2x4(src + i * cn, vl);
            __riscv_vse8_v_u8m2(dst0 + i, __riscv_vget_v_u8m2x4_u8m2(seg, 0), vl);
            __riscv_vse8_v_u8m2(dst1 + i, __riscv_vget_v_u8m2x4_u8m2(seg, 1), vl);
            __riscv_vse8_v_u8m2(dst2 + i, __riscv_vget_v_u8m2x4_u8m2(seg, 2), vl);
            __riscv_vse8_v_u8m2(dst3 + i, __riscv_vget_v_u8m2x4_u8m2(seg, 3), vl);
        }
    }
    else if ( cn == 5)
    {
        uchar *dst0 = dst[0], *dst1 = dst[1], *dst2 = dst[2], *dst3 = dst[3], *dst4 = dst[4];
        for (int i = 0; i < len; i += vl)
        {
            vl = __riscv_vsetvl_e8m1(len - i);
            vuint8m1x5_t seg = __riscv_vlseg5e8_v_u8m1x5(src + i * cn, vl);
            __riscv_vse8_v_u8m1(dst0 + i, __riscv_vget_v_u8m1x5_u8m1(seg, 0), vl);
            __riscv_vse8_v_u8m1(dst1 + i, __riscv_vget_v_u8m1x5_u8m1(seg, 1), vl);
            __riscv_vse8_v_u8m1(dst2 + i, __riscv_vget_v_u8m1x5_u8m1(seg, 2), vl);
            __riscv_vse8_v_u8m1(dst3 + i, __riscv_vget_v_u8m1x5_u8m1(seg, 3), vl);
            __riscv_vse8_v_u8m1(dst4 + i, __riscv_vget_v_u8m1x5_u8m1(seg, 4), vl);
        }
    }
    else if ( cn == 6)
    {
        uchar *dst0 = dst[0], *dst1 = dst[1], *dst2 = dst[2], *dst3 = dst[3], *dst4 = dst[4], *dst5 = dst[5];
        for (int i = 0; i < len; i += vl)
        {
            vl = __riscv_vsetvl_e8m1(len - i);
            vuint8m1x6_t seg = __riscv_vlseg6e8_v_u8m1x6(src + i * cn, vl);
            __riscv_vse8_v_u8m1(dst0 + i, __riscv_vget_v_u8m1x6_u8m1(seg, 0), vl);
            __riscv_vse8_v_u8m1(dst1 + i, __riscv_vget_v_u8m1x6_u8m1(seg, 1), vl);
            __riscv_vse8_v_u8m1(dst2 + i, __riscv_vget_v_u8m1x6_u8m1(seg, 2), vl);
            __riscv_vse8_v_u8m1(dst3 + i, __riscv_vget_v_u8m1x6_u8m1(seg, 3), vl);
            __riscv_vse8_v_u8m1(dst4 + i, __riscv_vget_v_u8m1x6_u8m1(seg, 4), vl);
            __riscv_vse8_v_u8m1(dst5 + i, __riscv_vget_v_u8m1x6_u8m1(seg, 5), vl);
        }
    }
    else if ( cn == 7)
    {
        uchar *dst0 = dst[0], *dst1 = dst[1], *dst2 = dst[2], *dst3 = dst[3], *dst4 = dst[4], *dst5 = dst[5], *dst6 = dst[6];
        for (int i = 0; i < len; i += vl)
        {
            vl = __riscv_vsetvl_e8m1(len - i);
            vuint8m1x7_t seg = __riscv_vlseg7e8_v_u8m1x7(src + i * cn, vl);
            __riscv_vse8_v_u8m1(dst0 + i, __riscv_vget_v_u8m1x7_u8m1(seg, 0), vl);
            __riscv_vse8_v_u8m1(dst1 + i, __riscv_vget_v_u8m1x7_u8m1(seg, 1), vl);
            __riscv_vse8_v_u8m1(dst2 + i, __riscv_vget_v_u8m1x7_u8m1(seg, 2), vl);
            __riscv_vse8_v_u8m1(dst3 + i, __riscv_vget_v_u8m1x7_u8m1(seg, 3), vl);
            __riscv_vse8_v_u8m1(dst4 + i, __riscv_vget_v_u8m1x7_u8m1(seg, 4), vl);
            __riscv_vse8_v_u8m1(dst5 + i, __riscv_vget_v_u8m1x7_u8m1(seg, 5), vl);
            __riscv_vse8_v_u8m1(dst6 + i, __riscv_vget_v_u8m1x7_u8m1(seg, 6), vl);
        }
    }
    else if ( cn == 8)
    {
        uchar *dst0 = dst[0], *dst1 = dst[1], *dst2 = dst[2], *dst3 = dst[3], *dst4 = dst[4], *dst5 = dst[5], *dst6 = dst[6], *dst7 = dst[7];
        for (int i = 0; i < len; i += vl)
        {
            vl = __riscv_vsetvl_e8m2(len - i);
            vuint8m2x4_t seg = __riscv_vlsseg4e8_v_u8m2x4(src + i * cn, cn, vl);
            __riscv_vse8_v_u8m2(dst0 + i, __riscv_vget_v_u8m2x4_u8m2(seg, 0), vl);
            __riscv_vse8_v_u8m2(dst1 + i, __riscv_vget_v_u8m2x4_u8m2(seg, 1), vl);
            __riscv_vse8_v_u8m2(dst2 + i, __riscv_vget_v_u8m2x4_u8m2(seg, 2), vl);
            __riscv_vse8_v_u8m2(dst3 + i, __riscv_vget_v_u8m2x4_u8m2(seg, 3), vl);
        }
        for (int i = 0; i < len; i += vl)
        {
            vl = __riscv_vsetvl_e8m2(len - i);
            vuint8m2x4_t seg = __riscv_vlsseg4e8_v_u8m2x4(src + 4 + i * cn, cn, vl);
            __riscv_vse8_v_u8m2(dst4 + i, __riscv_vget_v_u8m2x4_u8m2(seg, 0), vl);
            __riscv_vse8_v_u8m2(dst5 + i, __riscv_vget_v_u8m2x4_u8m2(seg, 1), vl);
            __riscv_vse8_v_u8m2(dst6 + i, __riscv_vget_v_u8m2x4_u8m2(seg, 2), vl);
            __riscv_vse8_v_u8m2(dst7 + i, __riscv_vget_v_u8m2x4_u8m2(seg, 3), vl);
        }
    }
    return CV_HAL_ERROR_OK;
}

}}
#endif
