// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "rvv_hal.hpp"

namespace cv { namespace rvv_hal { namespace core {

#if CV_HAL_RVV_1P0_ENABLED

#define OPENCV_HAL_IMPL_RVV_SPLIT(func, T, vtype, suffix, width) \
int func(const T* src, T** dst, int len, int cn) \
{ \
    int vl = 0; \
    if (cn == 1) \
    { \
        T* dst0 = dst[0]; \
        for (int i = 0; i < len; i += vl) \
        { \
            vl = __riscv_vsetvl_e##width##m8(len - i); \
            __riscv_vse##width##_v_##suffix##m8(dst0 + i, __riscv_vle##width##_v_##suffix##m8(src + i, vl), vl); \
        } \
    } \
    else if (cn == 2) \
    { \
        T *dst0 = dst[0], *dst1 = dst[1]; \
        for (int i = 0; i < len; i += vl) \
        { \
            vl = __riscv_vsetvl_e##width##m4(len - i); \
            v##vtype##m4x2_t seg = __riscv_vlseg2e##width##_v_##suffix##m4x2(src + i * cn, vl); \
            __riscv_vse##width##_v_##suffix##m4(dst0 + i, __riscv_vget_v_##suffix##m4x2_##suffix##m4(seg, 0), vl); \
            __riscv_vse##width##_v_##suffix##m4(dst1 + i, __riscv_vget_v_##suffix##m4x2_##suffix##m4(seg, 1), vl); \
        } \
    } \
    else if (cn == 3) \
    { \
        T *dst0 = dst[0], *dst1 = dst[1], *dst2 = dst[2]; \
        for (int i = 0; i < len; i += vl) \
        { \
            vl = __riscv_vsetvl_e##width##m2(len - i); \
            v##vtype##m2x3_t seg = __riscv_vlseg3e##width##_v_##suffix##m2x3(src + i * cn, vl); \
            __riscv_vse##width##_v_##suffix##m2(dst0 + i, __riscv_vget_v_##suffix##m2x3_##suffix##m2(seg, 0), vl); \
            __riscv_vse##width##_v_##suffix##m2(dst1 + i, __riscv_vget_v_##suffix##m2x3_##suffix##m2(seg, 1), vl); \
            __riscv_vse##width##_v_##suffix##m2(dst2 + i, __riscv_vget_v_##suffix##m2x3_##suffix##m2(seg, 2), vl); \
        } \
    } \
    else if (cn == 4) \
    { \
        T *dst0 = dst[0], *dst1 = dst[1], *dst2 = dst[2], *dst3 = dst[3]; \
        for (int i = 0; i < len; i += vl) \
        { \
            vl = __riscv_vsetvl_e##width##m2(len - i); \
            v##vtype##m2x4_t seg = __riscv_vlseg4e##width##_v_##suffix##m2x4(src + i * cn, vl); \
            __riscv_vse##width##_v_##suffix##m2(dst0 + i, __riscv_vget_v_##suffix##m2x4_##suffix##m2(seg, 0), vl); \
            __riscv_vse##width##_v_##suffix##m2(dst1 + i, __riscv_vget_v_##suffix##m2x4_##suffix##m2(seg, 1), vl); \
            __riscv_vse##width##_v_##suffix##m2(dst2 + i, __riscv_vget_v_##suffix##m2x4_##suffix##m2(seg, 2), vl); \
            __riscv_vse##width##_v_##suffix##m2(dst3 + i, __riscv_vget_v_##suffix##m2x4_##suffix##m2(seg, 3), vl); \
        } \
    } \
    else \
    { \
        int k = 0; \
        for (; k <= cn - 4; k += 4) \
        { \
            T *dst0 = dst[k], *dst1 = dst[k + 1], *dst2 = dst[k + 2], *dst3 = dst[k + 3]; \
            for (int i = 0; i < len; i += vl) \
            { \
                vl = __riscv_vsetvl_e##width##m2(len - i); \
                v##vtype##m2x4_t seg = __riscv_vlsseg4e##width##_v_##suffix##m2x4(src + k + i * cn, cn * sizeof(T), vl); \
                __riscv_vse##width##_v_##suffix##m2(dst0 + i, __riscv_vget_v_##suffix##m2x4_##suffix##m2(seg, 0), vl); \
                __riscv_vse##width##_v_##suffix##m2(dst1 + i, __riscv_vget_v_##suffix##m2x4_##suffix##m2(seg, 1), vl); \
                __riscv_vse##width##_v_##suffix##m2(dst2 + i, __riscv_vget_v_##suffix##m2x4_##suffix##m2(seg, 2), vl); \
                __riscv_vse##width##_v_##suffix##m2(dst3 + i, __riscv_vget_v_##suffix##m2x4_##suffix##m2(seg, 3), vl); \
            } \
        } \
        for (; k < cn; ++k) \
        { \
            T* dstK = dst[k]; \
            for (int i = 0; i < len; i += vl) \
            { \
                vl = __riscv_vsetvl_e##width##m2(len - i); \
                v##vtype##m2_t seg = __riscv_vlse##width##_v_##suffix##m2(src + k + i * cn, cn * sizeof(T), vl); \
                __riscv_vse##width##_v_##suffix##m2(dstK + i, seg, vl); \
            } \
        } \
    } \
    return CV_HAL_ERROR_OK; \
}

OPENCV_HAL_IMPL_RVV_SPLIT(split8u, uchar, uint8, u8, 8)
OPENCV_HAL_IMPL_RVV_SPLIT(split16u, ushort, uint16, u16, 16)
OPENCV_HAL_IMPL_RVV_SPLIT(split32s, int, int32, i32, 32)
OPENCV_HAL_IMPL_RVV_SPLIT(split64s, int64, int64, i64, 64)

#undef OPENCV_HAL_IMPL_RVV_SPLIT

#endif // CV_HAL_RVV_1P0_ENABLED

}}} // cv::rvv_hal::core
