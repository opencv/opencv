// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#ifndef OPENCV_HAL_RVV_MERGE_HPP_INCLUDED
#define OPENCV_HAL_RVV_MERGE_HPP_INCLUDED

#include <riscv_vector.h>

namespace cv { namespace cv_hal_rvv {

#undef cv_hal_merge8u
#define cv_hal_merge8u cv::cv_hal_rvv::merge8u
#undef cv_hal_merge16u
#define cv_hal_merge16u cv::cv_hal_rvv::merge16u
#undef cv_hal_merge32s
#define cv_hal_merge32s cv::cv_hal_rvv::merge32s
#undef cv_hal_merge64s
#define cv_hal_merge64s cv::cv_hal_rvv::merge64s

inline int merge8u(const uchar** src, uchar* dst, int len, int cn ) {
    int vl = 0;
    if (cn == 1)
    {
        const uchar* src0 = src[0];
        for (int i = 0; i < len; i += vl)
        {
            vl = __riscv_vsetvl_e8m8(len - i);
            __riscv_vse8_v_u8m8(dst + i, __riscv_vle8_v_u8m8(src0 + i, vl), vl);
        }
    }
    else if (cn == 2)
    {
        const uchar *src0 = src[0], *src1 = src[1];
        for (int i = 0; i < len; i += vl)
        {
            vl = __riscv_vsetvl_e8m4(len - i);
            vuint8m4x2_t seg = __riscv_vcreate_v_u8m4x2(
                __riscv_vle8_v_u8m4(src0 + i, vl),
                __riscv_vle8_v_u8m4(src1 + i, vl)
            );
            __riscv_vsseg2e8_v_u8m4x2(dst + i * cn, seg, vl);
        }
    }
    else if (cn == 3)
    {
        const uchar *src0 = src[0], *src1 = src[1], *src2 = src[2];
        for (int i = 0; i < len; i += vl)
        {
            vl = __riscv_vsetvl_e8m2(len - i);
            vuint8m2x3_t seg = __riscv_vcreate_v_u8m2x3(
                __riscv_vle8_v_u8m2(src0 + i, vl),
                __riscv_vle8_v_u8m2(src1 + i, vl),
                __riscv_vle8_v_u8m2(src2 + i, vl)
            );
            __riscv_vsseg3e8_v_u8m2x3(dst + i * cn, seg, vl);
        }
    }
    else if (cn == 4)
    {
        const uchar *src0 = src[0], *src1 = src[1], *src2 = src[2], *src3 = src[3];
        for (int i = 0; i < len; i += vl)
        {
            vl = __riscv_vsetvl_e8m2(len - i);
            vuint8m2x4_t seg = __riscv_vcreate_v_u8m2x4(
                __riscv_vle8_v_u8m2(src0 + i, vl),
                __riscv_vle8_v_u8m2(src1 + i, vl),
                __riscv_vle8_v_u8m2(src2 + i, vl),
                __riscv_vle8_v_u8m2(src3 + i, vl)
            );
            __riscv_vsseg4e8_v_u8m2x4(dst + i * cn, seg, vl);
        }
    }
    else
    {
        int k = 0;
        for (; k <= cn - 4; k += 4)
        {
            const uchar *src0 = src[k], *src1 = src[k + 1], *src2 = src[k + 2], *src3 = src[k + 3];
            for (int i = 0; i < len; i += vl)
            {
                vl = __riscv_vsetvl_e8m2(len - i);
                vuint8m2x4_t seg = __riscv_vcreate_v_u8m2x4(
                    __riscv_vle8_v_u8m2(src0 + i, vl),
                    __riscv_vle8_v_u8m2(src1 + i, vl),
                    __riscv_vle8_v_u8m2(src2 + i, vl),
                    __riscv_vle8_v_u8m2(src3 + i, vl)
                );
                __riscv_vssseg4e8_v_u8m2x4(dst + k + i * cn, cn, seg, vl);
            }
        }
        for (; k < cn; ++k)
        {
            const uchar* srcK = src[k];
            for (int i = 0; i < len; i += vl)
            {
                vl = __riscv_vsetvl_e8m2(len - i);
                vuint8m2_t seg = __riscv_vle8_v_u8m2(srcK + i, vl);
                __riscv_vsse8_v_u8m2(dst + k + i * cn, cn, seg, vl);
            }
        }
    }
    return CV_HAL_ERROR_OK;
}

inline int merge16u(const ushort** src, ushort* dst, int len, int cn ) {
    int vl = 0;
    if (cn == 1)
    {
        const ushort* src0 = src[0];
        for (int i = 0; i < len; i += vl)
        {
            vl = __riscv_vsetvl_e16m8(len - i);
            __riscv_vse16_v_u16m8(dst + i, __riscv_vle16_v_u16m8(src0 + i, vl), vl);
        }
    }
    else if (cn == 2)
    {
        const ushort *src0 = src[0], *src1 = src[1];
        for (int i = 0; i < len; i += vl)
        {
            vl = __riscv_vsetvl_e16m4(len - i);
            vuint16m4x2_t seg = __riscv_vcreate_v_u16m4x2(
                __riscv_vle16_v_u16m4(src0 + i, vl),
                __riscv_vle16_v_u16m4(src1 + i, vl)
            );
            __riscv_vsseg2e16_v_u16m4x2(dst + i * cn, seg, vl);
        }
    }
    else if (cn == 3)
    {
        const ushort *src0 = src[0], *src1 = src[1], *src2 = src[2];
        for (int i = 0; i < len; i += vl)
        {
            vl = __riscv_vsetvl_e16m2(len - i);
            vuint16m2x3_t seg = __riscv_vcreate_v_u16m2x3(
                __riscv_vle16_v_u16m2(src0 + i, vl),
                __riscv_vle16_v_u16m2(src1 + i, vl),
                __riscv_vle16_v_u16m2(src2 + i, vl)
            );
            __riscv_vsseg3e16_v_u16m2x3(dst + i * cn, seg, vl);
        }
    }
    else if (cn == 4)
    {
        const ushort *src0 = src[0], *src1 = src[1], *src2 = src[2], *src3 = src[3];
        for (int i = 0; i < len; i += vl)
        {
            vl = __riscv_vsetvl_e16m2(len - i);
            vuint16m2x4_t seg = __riscv_vcreate_v_u16m2x4(
                __riscv_vle16_v_u16m2(src0 + i, vl),
                __riscv_vle16_v_u16m2(src1 + i, vl),
                __riscv_vle16_v_u16m2(src2 + i, vl),
                __riscv_vle16_v_u16m2(src3 + i, vl)
            );
            __riscv_vsseg4e16_v_u16m2x4(dst + i * cn, seg, vl);
        }
    }
    else
    {
        int k = 0;
        for (; k <= cn - 4; k += 4)
        {
            const ushort *src0 = src[k], *src1 = src[k + 1], *src2 = src[k + 2], *src3 = src[k + 3];
            for (int i = 0; i < len; i += vl)
            {
                vl = __riscv_vsetvl_e16m2(len - i);
                vuint16m2x4_t seg = __riscv_vcreate_v_u16m2x4(
                    __riscv_vle16_v_u16m2(src0 + i, vl),
                    __riscv_vle16_v_u16m2(src1 + i, vl),
                    __riscv_vle16_v_u16m2(src2 + i, vl),
                    __riscv_vle16_v_u16m2(src3 + i, vl)
                );
                __riscv_vssseg4e16_v_u16m2x4(dst + k + i * cn, cn * sizeof(ushort), seg, vl);
            }
        }
        for (; k < cn; ++k)
        {
            const ushort* srcK = src[k];
            for (int i = 0; i < len; i += vl)
            {
                vl = __riscv_vsetvl_e16m2(len - i);
                vuint16m2_t seg = __riscv_vle16_v_u16m2(srcK + i, vl);
                __riscv_vsse16_v_u16m2(dst + k + i * cn, cn * sizeof(ushort), seg, vl);
            }
        }
    }
    return CV_HAL_ERROR_OK;
}

#if defined __GNUC__
__attribute__((optimize("no-tree-vectorize")))
#endif
inline int merge32s(const int** src, int* dst, int len, int cn ) {
    int k = cn % 4 ? cn % 4 : 4;
    int i, j;
    if( k == 1 )
    {
        const int* src0 = src[0];
        #if defined(__clang__)
        #pragma clang loop vectorize(disable)
        #endif
        for( i = j = 0; i < len; i++, j += cn )
            dst[j] = src0[i];
    }
    else if( k == 2 )
    {
        const int *src0 = src[0], *src1 = src[1];
        i = j = 0;
        #if defined(__clang__)
        #pragma clang loop vectorize(disable)
        #endif
        for( ; i < len; i++, j += cn )
        {
            dst[j] = src0[i];
            dst[j+1] = src1[i];
        }
    }
    else if( k == 3 )
    {
        const int *src0 = src[0], *src1 = src[1], *src2 = src[2];
        i = j = 0;
        #if defined(__clang__)
        #pragma clang loop vectorize(disable)
        #endif
        for( ; i < len; i++, j += cn )
        {
            dst[j] = src0[i];
            dst[j+1] = src1[i];
            dst[j+2] = src2[i];
        }
    }
    else
    {
        const int *src0 = src[0], *src1 = src[1], *src2 = src[2], *src3 = src[3];
        i = j = 0;
        #if defined(__clang__)
        #pragma clang loop vectorize(disable)
        #endif
        for( ; i < len; i++, j += cn )
        {
            dst[j] = src0[i]; dst[j+1] = src1[i];
            dst[j+2] = src2[i]; dst[j+3] = src3[i];
        }
    }
    #if defined(__clang__)
    #pragma clang loop vectorize(disable)
    #endif
    for( ; k < cn; k += 4 )
    {
        const int *src0 = src[k], *src1 = src[k+1], *src2 = src[k+2], *src3 = src[k+3];
        for( i = 0, j = k; i < len; i++, j += cn )
        {
            dst[j] = src0[i]; dst[j+1] = src1[i];
            dst[j+2] = src2[i]; dst[j+3] = src3[i];
        }
    }
    return CV_HAL_ERROR_OK;
}

#if defined __GNUC__
__attribute__((optimize("no-tree-vectorize")))
#endif
inline int merge64s(const int64** src, int64* dst, int len, int cn ) {
    int k = cn % 4 ? cn % 4 : 4;
    int i, j;
    if( k == 1 )
    {
        const int64* src0 = src[0];
        #if defined(__clang__)
        #pragma clang loop vectorize(disable)
        #endif
        for( i = j = 0; i < len; i++, j += cn )
            dst[j] = src0[i];
    }
    else if( k == 2 )
    {
        const int64 *src0 = src[0], *src1 = src[1];
        i = j = 0;
        #if defined(__clang__)
        #pragma clang loop vectorize(disable)
        #endif
        for( ; i < len; i++, j += cn )
        {
            dst[j] = src0[i];
            dst[j+1] = src1[i];
        }
    }
    else if( k == 3 )
    {
        const int64 *src0 = src[0], *src1 = src[1], *src2 = src[2];
        i = j = 0;
        #if defined(__clang__)
        #pragma clang loop vectorize(disable)
        #endif
        for( ; i < len; i++, j += cn )
        {
            dst[j] = src0[i];
            dst[j+1] = src1[i];
            dst[j+2] = src2[i];
        }
    }
    else
    {
        const int64 *src0 = src[0], *src1 = src[1], *src2 = src[2], *src3 = src[3];
        i = j = 0;
        #if defined(__clang__)
        #pragma clang loop vectorize(disable)
        #endif
        for( ; i < len; i++, j += cn )
        {
            dst[j] = src0[i]; dst[j+1] = src1[i];
            dst[j+2] = src2[i]; dst[j+3] = src3[i];
        }
    }
    #if defined(__clang__)
    #pragma clang loop vectorize(disable)
    #endif
    for( ; k < cn; k += 4 )
    {
        const int64 *src0 = src[k], *src1 = src[k+1], *src2 = src[k+2], *src3 = src[k+3];
        for( i = 0, j = k; i < len; i++, j += cn )
        {
            dst[j] = src0[i]; dst[j+1] = src1[i];
            dst[j+2] = src2[i]; dst[j+3] = src3[i];
        }
    }
    return CV_HAL_ERROR_OK;
}

}}

#endif
