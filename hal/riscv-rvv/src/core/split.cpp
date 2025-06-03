// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "rvv_hal.hpp"

namespace cv { namespace rvv_hal { namespace core {

#if CV_HAL_RVV_1P0_ENABLED

int split8u(const uchar* src, uchar** dst, int len, int cn)
{
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
    else
    {
        int k = 0;
        for (; k <= cn - 4; k += 4)
        {
            uchar *dst0 = dst[k], *dst1 = dst[k + 1], *dst2 = dst[k + 2], *dst3 = dst[k + 3];
            for (int i = 0; i < len; i += vl)
            {
                vl = __riscv_vsetvl_e8m2(len - i);
                vuint8m2x4_t seg = __riscv_vlsseg4e8_v_u8m2x4(src + k + i * cn, cn, vl);
                __riscv_vse8_v_u8m2(dst0 + i, __riscv_vget_v_u8m2x4_u8m2(seg, 0), vl);
                __riscv_vse8_v_u8m2(dst1 + i, __riscv_vget_v_u8m2x4_u8m2(seg, 1), vl);
                __riscv_vse8_v_u8m2(dst2 + i, __riscv_vget_v_u8m2x4_u8m2(seg, 2), vl);
                __riscv_vse8_v_u8m2(dst3 + i, __riscv_vget_v_u8m2x4_u8m2(seg, 3), vl);
            }
        }
        for (; k < cn; ++k)
        {
            uchar* dstK = dst[k];
            for (int i = 0; i < len; i += vl)
            {
                vl = __riscv_vsetvl_e8m2(len - i);
                vuint8m2_t seg = __riscv_vlse8_v_u8m2(src + k + i * cn, cn, vl);
                __riscv_vse8_v_u8m2(dstK + i, seg, vl);
            }
        }
    }
    return CV_HAL_ERROR_OK;
}

#endif // CV_HAL_RVV_1P0_ENABLED

}}} // cv::rvv_hal::core
