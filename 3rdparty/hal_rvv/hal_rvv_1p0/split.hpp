// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#ifndef OPENCV_HAL_RVV_SPLIT_HPP_INCLUDED
#define OPENCV_HAL_RVV_SPLIT_HPP_INCLUDED

#include <riscv_vector.h>
#include <stdio.h>
#include "../../../modules/core/src/precomp.hpp"
namespace cv { namespace cv_hal_rvv {

#undef cv_hal_split8u
#define cv_hal_split8u cv::cv_hal_rvv::split8u


inline int split8u(const uchar* src, uchar** dst, int len, int cn) {
    int k = cn % 4 ? cn % 4 : 4;
    int i = 0;
    int vl = __riscv_vsetvlmax_e8m1();

    if (k == 1) {
        uchar* dst0 = dst[0];
        if (cn == 1) {
            size_t n = len * sizeof(uchar);
            unsigned char* dst_tmp = dst0;
            const unsigned char *src_tmp = src;
            for(; n > 0; n -= vl, src_tmp += vl, dst_tmp += vl) {
                vl = __riscv_vsetvl_e8m8(n);
                vuint8m8_t vec_src = __riscv_vle8_v_u8m8(src_tmp, vl);
                __riscv_vse8_v_u8m8(dst_tmp, vec_src, vl);
            }
        } else {
            for (; i <= len - vl; i += vl) {
                auto vec = __riscv_vlse8_v_u8m1(src + i * cn, cn * sizeof(uchar), vl);
                __riscv_vse8_v_u8m1(dst0 + i, vec, vl);
            }
            #if defined(__clang__)
            #pragma clang loop vectorize(disable)
            #endif
            for (; i < len; i++)    
                dst0[i] = src[i * cn];
            return CV_HAL_ERROR_OK;
        }
    } else if (k == 2) {
        uchar *dst0 = dst[0], *dst1 = dst[1];
        for( i = 0; i < len; i += vl * 4 )
        {
            //VecT a, b;
            vuint8m1_t a1 = __riscv_vlse8_v_u8m1(src + i * cn, cn * sizeof(uchar), vl);
            vuint8m1_t b1 = __riscv_vlse8_v_u8m1(src + 1 + i * cn, cn * sizeof(uchar), vl);
            vuint8m1_t a2 = __riscv_vlse8_v_u8m1(src + i * cn, cn * sizeof(uchar), vl);
            vuint8m1_t b2 = __riscv_vlse8_v_u8m1(src + 1 + i * cn, cn * sizeof(uchar), vl);
            vuint8m1_t a3 = __riscv_vlse8_v_u8m1(src + i * cn, cn * sizeof(uchar), vl);
            vuint8m1_t b3 = __riscv_vlse8_v_u8m1(src + 1 + i * cn, cn * sizeof(uchar), vl);
            vuint8m1_t a4 = __riscv_vlse8_v_u8m1(src + i * cn, cn * sizeof(uchar), vl);
            vuint8m1_t b4 = __riscv_vlse8_v_u8m1(src + 1 + i * cn, cn * sizeof(uchar), vl);
            //v_load_deinterleave(src + i*cn, a, b);
            __riscv_vse8_v_u8m1(dst0 + i, a1, vl);
             __riscv_vse8_v_u8m1(dst1 + i, b1, vl);
            //return CV_HAL_ERROR_OK;
        }
        // uchar *dst0 = dst[0], *dst1 = dst[1];
        // for (; i <= len - vl; i += vl) {
        //     auto a = __riscv_vlse8_v_u8m1(src + i * cn, cn * sizeof(uchar), vl);
        //     auto b = __riscv_vlse8_v_u8m1(src + 1 + i * cn, cn * sizeof(uchar), vl);
        //     __riscv_vse8_v_u8m1(dst0 + i, a, vl);
        //     __riscv_vse8_v_u8m1(dst1 + i, b, vl);
        // }
        // return CV_HAL_ERROR_OK;
        // #if defined(__clang__)
        // #pragma clang loop vectorize(disable)
        // #endif
        // for (; i < len; i++) {
        //     dst0[i] = src[i * cn];
        //     dst1[i] = src[i * cn + 1];
        // }
        return CV_HAL_ERROR_OK;
    } else if (k == 3) {
        uchar *dst0 = dst[0], *dst1 = dst[1], *dst2 = dst[2];
        for (; i <= len - vl; i += vl) {
            auto a = __riscv_vlse8_v_u8m1(src + 0 + i * cn, cn * sizeof(uchar), vl);
            auto b = __riscv_vlse8_v_u8m1(src + 1 + i * cn, cn * sizeof(uchar), vl);
            auto c = __riscv_vlse8_v_u8m1(src + 2 + i * cn, cn * sizeof(uchar), vl);
            __riscv_vse8_v_u8m1(dst0 + i, a, vl);
            __riscv_vse8_v_u8m1(dst1 + i, b, vl);
            __riscv_vse8_v_u8m1(dst2 + i, c, vl);
        }
        #if defined(__clang__)
        #pragma clang loop vectorize(disable)
        #endif
        for (; i < len; i++) {
            //int j = i * cn;
            dst0[i] = src[i * cn];
            dst1[i] = src[i * cn + 1];
            dst2[i] = src[i * cn + 2];
        }
    } else {
        uchar *dst0 = dst[0], *dst1 = dst[1], *dst2 = dst[2], *dst3 = dst[3];
        for (; i <= len - vl; i += vl) {
            auto a = __riscv_vlse8_v_u8m1(src + 0 + i * cn, cn * sizeof(uchar), vl);
            auto b = __riscv_vlse8_v_u8m1(src + 1 + i * cn, cn * sizeof(uchar), vl);
            auto c = __riscv_vlse8_v_u8m1(src + 2 + i * cn, cn * sizeof(uchar), vl);
            auto d = __riscv_vlse8_v_u8m1(src + 3 + i * cn, cn * sizeof(uchar), vl);
            __riscv_vse8_v_u8m1(dst0 + i, a, vl);
            __riscv_vse8_v_u8m1(dst1 + i, b, vl);
            __riscv_vse8_v_u8m1(dst2 + i, c, vl);
            __riscv_vse8_v_u8m1(dst3 + i, d, vl);
        }
        #if defined(__clang__)
        #pragma clang loop vectorize(disable)
        #endif
        for (; i < len; i++) {
            int j = i * cn;
            dst0[i] = src[i * cn];
            dst1[i] = src[i * cn + 1];
            dst2[i] = src[i * cn + 2];
            dst3[i] = src[i * cn + 3];
        }
    }
    for (; k < cn; k += 4) {
        uchar *dst0 = dst[k], *dst1 = dst[k+1], *dst2 = dst[k+2], *dst3 = dst[k+3];
        i = 0;

        for (; i <= len - vl; i += vl) {
            auto a = __riscv_vlse8_v_u8m1(src + k + 0 + i * cn, cn * sizeof(uchar), vl);
            auto b = __riscv_vlse8_v_u8m1(src + k + 1 + i * cn, cn * sizeof(uchar), vl);
            auto c = __riscv_vlse8_v_u8m1(src + k + 2 + i * cn, cn * sizeof(uchar), vl);
            auto d = __riscv_vlse8_v_u8m1(src + k + 3 + i * cn, cn * sizeof(uchar), vl);
            __riscv_vse8_v_u8m1(dst0 + i, a, vl);
            __riscv_vse8_v_u8m1(dst1 + i, b, vl);
            __riscv_vse8_v_u8m1(dst2 + i, c, vl);
            __riscv_vse8_v_u8m1(dst3 + i, d, vl);
        }
        #if defined(__clang__)
        #pragma clang loop vectorize(disable)
        #endif
        for (; i < len; i++) {
            //int j = k + i * cn;
            dst0[i] = src[k + i * cn];
            dst1[i] = src[k + i * cn + 1];
            dst2[i] = src[k + i * cn + 2];
            dst3[i] = src[k + i * cn + 3];
        }
    }
    return CV_HAL_ERROR_OK;
}

}
}
#endif