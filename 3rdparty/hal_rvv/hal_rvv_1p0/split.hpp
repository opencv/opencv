// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#ifndef OPENCV_HAL_RVV_SPLIT_HPP_INCLUDED
#define OPENCV_HAL_RVV_SPLIT_HPP_INCLUDED 
#include <riscv_vector.h>
#include <stdio.h>
namespace cv { namespace cv_hal_rvv {

#undef cv_hal_split8u
#define cv_hal_split8u cv::cv_hal_rvv::split8u
#undef cv_hal_split16u
#define cv_hal_split8u cv::cv_hal_rvv::split16u

#if defined __GNUC__
__attribute__((optimize("no-tree-vectorize")))
#endif
inline int split8u(const uchar* src, uchar** dst, int len, int cn) {
    int k = cn % 4 ? cn % 4 : 4;
    int i = 0;
    int vl = __riscv_vsetvlmax_e8m1();

    if (k == 1) {
        uchar* dst0 = dst[0];
        if (cn == 1) {
            memcpy(dst0, src, len * sizeof(uchar));
        } else {
            // Векторная обработка
            for (; i <= len - vl; i += vl) {
                auto vec = __riscv_vlse8_v_u8m1(src + i * cn, cn * sizeof(uchar), vl);
                __riscv_vse8_v_u8m1(dst0 + i, vec, vl);
            }
            // Скалярный остаток
            #if defined(__clang__)
            #pragma clang loop vectorize(disable)
            #endif
            for (; i < len; i++)
                dst0[i] = src[i * cn];
        }
    } else if (k == 2) {
        uchar *dst0 = dst[0], *dst1 = dst[1];
        // Векторная обработка
        for (; i <= len - vl * 4; i += vl * 4) {
            auto a1 = __riscv_vlse8_v_u8m1(src + 0 + i * cn, cn * sizeof(uchar), vl);
            auto b1 = __riscv_vlse8_v_u8m1(src + 1 + i * cn, cn * sizeof(uchar), vl);
            auto a2 = __riscv_vlse8_v_u8m1(src + 0 + (i + vl) * cn, cn * sizeof(uchar), vl);
            auto b2 = __riscv_vlse8_v_u8m1(src + 1 + (i + vl) * cn, cn * sizeof(uchar), vl);
            auto a3 = __riscv_vlse8_v_u8m1(src + 0 + (i + vl + vl) * cn, cn * sizeof(uchar), vl);
            auto b3 = __riscv_vlse8_v_u8m1(src + 1 + (i + vl + vl) * cn, cn * sizeof(uchar), vl);
            auto a4 = __riscv_vlse8_v_u8m1(src + 0 + (i + vl + vl + vl) * cn, cn * sizeof(uchar), vl);
            auto b4 = __riscv_vlse8_v_u8m1(src + 1 + (i + vl + vl + vl) * cn, cn * sizeof(uchar), vl);
            __riscv_vse8_v_u8m1(dst0 + i, a1, vl);
            __riscv_vse8_v_u8m1(dst1 + i, b1, vl);
            __riscv_vse8_v_u8m1(dst0 + i + vl, a2, vl);
            __riscv_vse8_v_u8m1(dst1 + i + vl, b2, vl);
            __riscv_vse8_v_u8m1(dst0 + i + vl + vl, a3, vl);
            __riscv_vse8_v_u8m1(dst1 + i + vl + vl, b3, vl);
            __riscv_vse8_v_u8m1(dst0 + i + vl + vl + vl, a4, vl);
            __riscv_vse8_v_u8m1(dst1 + i + vl + vl + vl, b4, vl);
        }
        // Скалярный остаток
        #if defined(__clang__)
        #pragma clang loop vectorize(disable)
        #endif
        for (; i < len; i++) {
            dst0[i] = src[i * cn];
            dst1[i] = src[i * cn + 1];
        }
    } else if (k == 3) {
        uchar *dst0 = dst[0], *dst1 = dst[1], *dst2 = dst[2];
        // Векторная обработка
        for (; i <= len - vl * 4; i += vl * 4) {
            auto a = __riscv_vlse8_v_u8m1(src + 0 + i * cn, cn * sizeof(uchar), vl);
            auto b = __riscv_vlse8_v_u8m1(src + 1 + i * cn, cn * sizeof(uchar), vl);
            auto c = __riscv_vlse8_v_u8m1(src + 2 + i * cn, cn * sizeof(uchar), vl);
            __riscv_vse8_v_u8m1(dst0 + i, a, vl);
            __riscv_vse8_v_u8m1(dst1 + i, b, vl);
            __riscv_vse8_v_u8m1(dst2 + i, c, vl);
            auto a2 = __riscv_vlse8_v_u8m1(src + 0 + (i + vl) * cn, cn * sizeof(uchar), vl);
            auto b2 = __riscv_vlse8_v_u8m1(src + 1 + (i + vl) * cn, cn * sizeof(uchar), vl);
            auto c2 = __riscv_vlse8_v_u8m1(src + 2 + (i + vl) * cn, cn * sizeof(uchar), vl);
            __riscv_vse8_v_u8m1(dst0 + i + vl, a2, vl);
            __riscv_vse8_v_u8m1(dst1 + i + vl, b2, vl);
            __riscv_vse8_v_u8m1(dst2 + i + vl, c2, vl);
            auto a3 = __riscv_vlse8_v_u8m1(src + 0 + (i + vl + vl) * cn, cn * sizeof(uchar), vl);
            auto b3 = __riscv_vlse8_v_u8m1(src + 1 + (i + vl + vl) * cn, cn * sizeof(uchar), vl);
            auto c3 = __riscv_vlse8_v_u8m1(src + 2 + (i + vl + vl) * cn, cn * sizeof(uchar), vl);
            __riscv_vse8_v_u8m1(dst0 + i + vl + vl, a3, vl);
            __riscv_vse8_v_u8m1(dst1 + i + vl + vl, b3, vl);
            __riscv_vse8_v_u8m1(dst2 + i + vl + vl, c3, vl);
            auto a4 = __riscv_vlse8_v_u8m1(src + 0 + (i + vl + vl + vl) * cn, cn * sizeof(uchar), vl);
            auto b4 = __riscv_vlse8_v_u8m1(src + 1 + (i + vl + vl + vl) * cn, cn * sizeof(uchar), vl);
            auto c4 = __riscv_vlse8_v_u8m1(src + 2 + (i + vl + vl + vl) * cn, cn * sizeof(uchar), vl);
            __riscv_vse8_v_u8m1(dst0 + i + vl + vl + vl, a4, vl);
            __riscv_vse8_v_u8m1(dst1 + i + vl + vl + vl, b4, vl);
            __riscv_vse8_v_u8m1(dst2 + i + vl + vl + vl, c4, vl);
        }
        // Скалярный остаток
        #if defined(__clang__)
        #pragma clang loop vectorize(disable)
        #endif
        for (; i < len; i++) {
            dst0[i] = src[i * cn];
            dst1[i] = src[i * cn + 1];
            dst2[i] = src[i * cn + 2];
        }
    } else {
        uchar *dst0 = dst[0], *dst1 = dst[1], *dst2 = dst[2], *dst3 = dst[3];
        // Векторная обработка
        for (; i <= len - vl * 4; i += vl * 4) {
            auto a = __riscv_vlse8_v_u8m1(src + 0 + i * cn, cn * sizeof(uchar), vl);
            auto b = __riscv_vlse8_v_u8m1(src + 1 + i * cn, cn * sizeof(uchar), vl);
            auto c = __riscv_vlse8_v_u8m1(src + 2 + i * cn, cn * sizeof(uchar), vl);
            auto d = __riscv_vlse8_v_u8m1(src + 3 + i * cn, cn * sizeof(uchar), vl);
            __riscv_vse8_v_u8m1(dst0 + i, a, vl);
            __riscv_vse8_v_u8m1(dst1 + i, b, vl);
            __riscv_vse8_v_u8m1(dst2 + i, c, vl);
            __riscv_vse8_v_u8m1(dst3 + i, d, vl);

            auto a2 = __riscv_vlse8_v_u8m1(src + 0 + (i + vl) * cn, cn * sizeof(uchar), vl);
            auto b2 = __riscv_vlse8_v_u8m1(src + 1 + (i + vl) * cn, cn * sizeof(uchar), vl);
            auto c2 = __riscv_vlse8_v_u8m1(src + 2 + (i + vl) * cn, cn * sizeof(uchar), vl);
            auto d2 = __riscv_vlse8_v_u8m1(src + 3 + (i + vl) * cn, cn * sizeof(uchar), vl);
            __riscv_vse8_v_u8m1(dst0 + i + vl, a2, vl);
            __riscv_vse8_v_u8m1(dst1 + i + vl, b2, vl);
            __riscv_vse8_v_u8m1(dst2 + i + vl, c2, vl);
            __riscv_vse8_v_u8m1(dst3 + i + vl, d2, vl);

            auto a3 = __riscv_vlse8_v_u8m1(src + 0 + (i + vl + vl) * cn, cn * sizeof(uchar), vl);
            auto b3 = __riscv_vlse8_v_u8m1(src + 1 + (i + vl + vl) * cn, cn * sizeof(uchar), vl);
            auto c3 = __riscv_vlse8_v_u8m1(src + 2 + (i + vl + vl) * cn, cn * sizeof(uchar), vl);
            auto d3 = __riscv_vlse8_v_u8m1(src + 3 + (i + vl + vl) * cn, cn * sizeof(uchar), vl);
            __riscv_vse8_v_u8m1(dst0 + i + vl + vl, a3, vl);
            __riscv_vse8_v_u8m1(dst1 + i + vl + vl, b3, vl);
            __riscv_vse8_v_u8m1(dst2 + i + vl + vl, c3, vl);
            __riscv_vse8_v_u8m1(dst3 + i + vl + vl, d3, vl);

            auto a4 = __riscv_vlse8_v_u8m1(src + 0 + (i + vl + vl + vl) * cn, cn * sizeof(uchar), vl);
            auto b4 = __riscv_vlse8_v_u8m1(src + 1 + (i + vl + vl + vl) * cn, cn * sizeof(uchar), vl);
            auto c4 = __riscv_vlse8_v_u8m1(src + 2 + (i + vl + vl + vl) * cn, cn * sizeof(uchar), vl);
            auto d4 = __riscv_vlse8_v_u8m1(src + 3 + (i + vl + vl + vl) * cn, cn * sizeof(uchar), vl);
            __riscv_vse8_v_u8m1(dst0 + i + vl + vl + vl, a4, vl);
            __riscv_vse8_v_u8m1(dst1 + i + vl + vl + vl, b4, vl);
            __riscv_vse8_v_u8m1(dst2 + i + vl + vl + vl, c4, vl);
            __riscv_vse8_v_u8m1(dst3 + i + vl + vl + vl, d4, vl);
        }
        // Скалярный остаток
        #if defined(__clang__)
        #pragma clang loop vectorize(disable)
        #endif
        for (; i < len; i++) {
            dst0[i] = src[i * cn];
            dst1[i] = src[i * cn + 1];
            dst2[i] = src[i * cn + 2];
            dst3[i] = src[i * cn + 3];
        }
    }

    // Обработка оставшихся каналов (блоками по 4)
    for (; k < cn; k += 4) {
        uchar *dst0 = dst[k], *dst1 = dst[k+1], *dst2 = dst[k+2], *dst3 = dst[k+3];
        i = 0;
        // Векторная обработка
        for (; i <= len - vl * 4; i += vl * 4) {
            auto a = __riscv_vlse8_v_u8m1(src + k + 0 + i * cn, cn * sizeof(uchar), vl);
            auto b = __riscv_vlse8_v_u8m1(src + k + 1 + i * cn, cn * sizeof(uchar), vl);
            auto c = __riscv_vlse8_v_u8m1(src + k + 2 + i * cn, cn * sizeof(uchar), vl);
            auto d = __riscv_vlse8_v_u8m1(src + k + 3 + i * cn, cn * sizeof(uchar), vl);
            __riscv_vse8_v_u8m1(dst0 + i, a, vl);
            __riscv_vse8_v_u8m1(dst1 + i, b, vl);
            __riscv_vse8_v_u8m1(dst2 + i, c, vl);
            __riscv_vse8_v_u8m1(dst3 + i, d, vl);

            auto a2 = __riscv_vlse8_v_u8m1(src + k + 0 + (i + vl) * cn, cn * sizeof(uchar), vl);
            auto b2 = __riscv_vlse8_v_u8m1(src + k + 1 + (i + vl) * cn, cn * sizeof(uchar), vl);
            auto c2 = __riscv_vlse8_v_u8m1(src + k + 2 + (i + vl) * cn, cn * sizeof(uchar), vl);
            auto d2 = __riscv_vlse8_v_u8m1(src + k + 3 + (i + vl) * cn, cn * sizeof(uchar), vl);
            __riscv_vse8_v_u8m1(dst0 + i + vl, a2, vl);
            __riscv_vse8_v_u8m1(dst1 + i + vl, b2, vl);
            __riscv_vse8_v_u8m1(dst2 + i + vl, c2, vl);
            __riscv_vse8_v_u8m1(dst3 + i + vl, d2, vl);

            auto a3 = __riscv_vlse8_v_u8m1(src + k + 0 + (i + vl + vl) * cn, cn * sizeof(uchar), vl);
            auto b3 = __riscv_vlse8_v_u8m1(src + k + 1 + (i + vl + vl) * cn, cn * sizeof(uchar), vl);
            auto c3 = __riscv_vlse8_v_u8m1(src + k + 2 + (i + vl + vl) * cn, cn * sizeof(uchar), vl);
            auto d3 = __riscv_vlse8_v_u8m1(src + k + 3 + (i + vl + vl) * cn, cn * sizeof(uchar), vl);
            __riscv_vse8_v_u8m1(dst0 + i + vl + vl, a3, vl);
            __riscv_vse8_v_u8m1(dst1 + i + vl + vl, b3, vl);
            __riscv_vse8_v_u8m1(dst2 + i + vl + vl, c3, vl);
            __riscv_vse8_v_u8m1(dst3 + i + vl + vl, d3, vl);

            auto a4 = __riscv_vlse8_v_u8m1(src + k + 0 + (i + vl + vl + vl) * cn, cn * sizeof(uchar), vl);
            auto b4 = __riscv_vlse8_v_u8m1(src + k + 1 + (i + vl + vl + vl) * cn, cn * sizeof(uchar), vl);
            auto c4 = __riscv_vlse8_v_u8m1(src + k + 2 + (i + vl + vl + vl) * cn, cn * sizeof(uchar), vl);
            auto d4 = __riscv_vlse8_v_u8m1(src + k + 3 + (i + vl + vl + vl) * cn, cn * sizeof(uchar), vl);
            __riscv_vse8_v_u8m1(dst0 + i + vl + vl + vl, a3, vl);
            __riscv_vse8_v_u8m1(dst1 + i + vl + vl + vl, b3, vl);
            __riscv_vse8_v_u8m1(dst2 + i + vl + vl + vl, c3, vl);
            __riscv_vse8_v_u8m1(dst3 + i + vl + vl + vl, d3, vl);
        }
        // Скалярный остаток
        #if defined(__clang__)
        #pragma clang loop vectorize(disable)
        #endif
        for (; i < len; i++) {
            dst0[i] = src[k + i * cn];
            dst1[i] = src[k + i * cn + 1];
            dst2[i] = src[k + i * cn + 2];
            dst3[i] = src[k + i * cn + 3];
        }
    }
    return CV_HAL_ERROR_OK;
}
inline int split16u(const uchar* src, uchar** dst, int len, int cn) {
    int k = cn % 4 ? cn % 4 : 4;
    int i = 0;
    int vl = __riscv_vsetvlmax_e16m1();

    if (k == 1) {
        uchar* dst0 = dst[0];
        if (cn == 1) {
            memcpy(dst0, src, len * sizeof(uchar));
        }
        else {
            // Векторная обработка
            for (; i <= len - vl; i += vl) {
                auto vec = __riscv_vlse16_v_u16m1(src + i * cn, cn * sizeof(uchar), vl);
                __riscv_vse16_v_u16m1(dst0 + i, vec, vl);
            }
            // Скалярный остаток
#if defined(__clang__)
#pragma clang loop vectorize(disable)
#endif
            for (; i < len; i++)
                dst0[i] = src[i * cn];
        }
    }
    else if (k == 2) {
        uchar* dst0 = dst[0], * dst1 = dst[1];
        // Векторная обработка
        for (; i <= len - vl * 4; i += vl * 4) {
            auto a1 = __riscv_vlse16_v_u16m1(src + 0 + i * cn, cn * sizeof(uchar), vl);
            auto b1 = __riscv_vlse16_v_u16m1(src + 1 + i * cn, cn * sizeof(uchar), vl);
            auto a2 = __riscv_vlse16_v_u16m1(src + 0 + (i + vl) * cn, cn * sizeof(uchar), vl);
            auto b2 = __riscv_vlse16_v_u16m1(src + 1 + (i + vl) * cn, cn * sizeof(uchar), vl);
            auto a3 = __riscv_vlse16_v_u16m1(src + 0 + (i + vl + vl) * cn, cn * sizeof(uchar), vl);
            auto b3 = __riscv_vlse16_v_u16m1(src + 1 + (i + vl + vl) * cn, cn * sizeof(uchar), vl);
            auto a4 = __riscv_vlse16_v_u16m1(src + 0 + (i + vl + vl + vl) * cn, cn * sizeof(uchar), vl);
            auto b4 = __riscv_vlse16_v_u16m1(src + 1 + (i + vl + vl + vl) * cn, cn * sizeof(uchar), vl);
            __riscv_vse16_v_u16m1(dst0 + i, a1, vl);
            __riscv_vse16_v_u16m1(dst1 + i, b1, vl);
            __riscv_vse16_v_u16m1(dst0 + i + vl, a2, vl);
            __riscv_vse16_v_u16m1(dst1 + i + vl, b2, vl);
            __riscv_vse16_v_u16m1(dst0 + i + vl + vl, a3, vl);
            __riscv_vse16_v_u16m1(dst1 + i + vl + vl, b3, vl);
            __riscv_vse16_v_u16m1(dst0 + i + vl + vl + vl, a4, vl);
            __riscv_vse16_v_u16m1(dst1 + i + vl + vl + vl, b4, vl);
        }
        // Скалярный остаток
#if defined(__clang__)
#pragma clang loop vectorize(disable)
#endif
        for (; i < len; i++) {
            dst0[i] = src[i * cn];
            dst1[i] = src[i * cn + 1];
        }
    }
    else if (k == 3) {
        uchar* dst0 = dst[0], * dst1 = dst[1], * dst2 = dst[2];
        // Векторная обработка
        for (; i <= len - vl * 4; i += vl * 4) {
            auto a = __riscv_vlse16_v_u16m1(src + 0 + i * cn, cn * sizeof(uchar), vl);
            auto b = __riscv_vlse16_v_u16m1(src + 1 + i * cn, cn * sizeof(uchar), vl);
            auto c = __riscv_vlse16_v_u16m1(src + 2 + i * cn, cn * sizeof(uchar), vl);
            __riscv_vse16_v_u16m1(dst0 + i, a, vl);
            __riscv_vse16_v_u16m1(dst1 + i, b, vl);
            __riscv_vse16_v_u16m1(dst2 + i, c, vl);
            auto a2 = __riscv_vlse16_v_u16m1(src + 0 + (i + vl) * cn, cn * sizeof(uchar), vl);
            auto b2 = __riscv_vlse16_v_u16m1(src + 1 + (i + vl) * cn, cn * sizeof(uchar), vl);
            auto c2 = __riscv_vlse16_v_u16m1(src + 2 + (i + vl) * cn, cn * sizeof(uchar), vl);
            __riscv_vse16_v_u16m1(dst0 + i + vl, a2, vl);
            __riscv_vse16_v_u16m1(dst1 + i + vl, b2, vl);
            __riscv_vse16_v_u16m1(dst2 + i + vl, c2, vl);
            auto a3 = __riscv_vlse16_v_u16m1(src + 0 + (i + vl + vl) * cn, cn * sizeof(uchar), vl);
            auto b3 = __riscv_vlse16_v_u16m1(src + 1 + (i + vl + vl) * cn, cn * sizeof(uchar), vl);
            auto c3 = __riscv_vlse16_v_u16m1(src + 2 + (i + vl + vl) * cn, cn * sizeof(uchar), vl);
            __riscv_vse16_v_u16m1(dst0 + i + vl + vl, a3, vl);
            __riscv_vse16_v_u16m1(dst1 + i + vl + vl, b3, vl);
            __riscv_vse16_v_u16m1(dst2 + i + vl + vl, c3, vl);
            auto a4 = __riscv_vlse16_v_u16m1(src + 0 + (i + vl + vl + vl) * cn, cn * sizeof(uchar), vl);
            auto b4 = __riscv_vlse16_v_u16m1(src + 1 + (i + vl + vl + vl) * cn, cn * sizeof(uchar), vl);
            auto c4 = __riscv_vlse16_v_u16m1(src + 2 + (i + vl + vl + vl) * cn, cn * sizeof(uchar), vl);
            __riscv_vse16_v_u16m1(dst0 + i + vl + vl + vl, a4, vl);
            __riscv_vse16_v_u16m1(dst1 + i + vl + vl + vl, b4, vl);
            __riscv_vse16_v_u16m1(dst2 + i + vl + vl + vl, c4, vl);
        }
        // Скалярный остаток
#if defined(__clang__)
#pragma clang loop vectorize(disable)
#endif
        for (; i < len; i++) {
            dst0[i] = src[i * cn];
            dst1[i] = src[i * cn + 1];
            dst2[i] = src[i * cn + 2];
        }
    }
    else {
        uchar* dst0 = dst[0], * dst1 = dst[1], * dst2 = dst[2], * dst3 = dst[3];
        // Векторная обработка
        for (; i <= len - vl * 4; i += vl * 4) {
            auto a = __riscv_vlse16_v_u16m1(src + 0 + i * cn, cn * sizeof(uchar), vl);
            auto b = __riscv_vlse16_v_u16m1(src + 1 + i * cn, cn * sizeof(uchar), vl);
            auto c = __riscv_vlse16_v_u16m1(src + 2 + i * cn, cn * sizeof(uchar), vl);
            auto d = __riscv_vlse16_v_u16m1(src + 3 + i * cn, cn * sizeof(uchar), vl);
            __riscv_vse16_v_u16m1(dst0 + i, a, vl);
            __riscv_vse16_v_u16m1(dst1 + i, b, vl);
            __riscv_vse16_v_u16m1(dst2 + i, c, vl);
            __riscv_vse16_v_u16m1(dst3 + i, d, vl);

            auto a2 = __riscv_vlse16_v_u16m1(src + 0 + (i + vl) * cn, cn * sizeof(uchar), vl);
            auto b2 = __riscv_vlse16_v_u16m1(src + 1 + (i + vl) * cn, cn * sizeof(uchar), vl);
            auto c2 = __riscv_vlse16_v_u16m1(src + 2 + (i + vl) * cn, cn * sizeof(uchar), vl);
            auto d2 = __riscv_vlse16_v_u16m1(src + 3 + (i + vl) * cn, cn * sizeof(uchar), vl);
            __riscv_vse16_v_u16m1(dst0 + i + vl, a2, vl);
            __riscv_vse16_v_u16m1(dst1 + i + vl, b2, vl);
            __riscv_vse16_v_u16m1(dst2 + i + vl, c2, vl);
            __riscv_vse16_v_u16m1(dst3 + i + vl, d2, vl);

            auto a3 = __riscv_vlse16_v_u16m1(src + 0 + (i + vl + vl) * cn, cn * sizeof(uchar), vl);
            auto b3 = __riscv_vlse16_v_u16m1(src + 1 + (i + vl + vl) * cn, cn * sizeof(uchar), vl);
            auto c3 = __riscv_vlse16_v_u16m1(src + 2 + (i + vl + vl) * cn, cn * sizeof(uchar), vl);
            auto d3 = __riscv_vlse16_v_u16m1(src + 3 + (i + vl + vl) * cn, cn * sizeof(uchar), vl);
            __riscv_vse16_v_u16m1(dst0 + i + vl + vl, a3, vl);
            __riscv_vse16_v_u16m1(dst1 + i + vl + vl, b3, vl);
            __riscv_vse16_v_u16m1(dst2 + i + vl + vl, c3, vl);
            __riscv_vse16_v_u16m1(dst3 + i + vl + vl, d3, vl);

            auto a4 = __riscv_vlse16_v_u16m1(src + 0 + (i + vl + vl + vl) * cn, cn * sizeof(uchar), vl);
            auto b4 = __riscv_vlse16_v_u16m1(src + 1 + (i + vl + vl + vl) * cn, cn * sizeof(uchar), vl);
            auto c4 = __riscv_vlse16_v_u16m1(src + 2 + (i + vl + vl + vl) * cn, cn * sizeof(uchar), vl);
            auto d4 = __riscv_vlse16_v_u16m1(src + 3 + (i + vl + vl + vl) * cn, cn * sizeof(uchar), vl);
            __riscv_vse16_v_u16m1(dst0 + i + vl + vl + vl, a4, vl);
            __riscv_vse16_v_u16m1(dst1 + i + vl + vl + vl, b4, vl);
            __riscv_vse16_v_u16m1(dst2 + i + vl + vl + vl, c4, vl);
            __riscv_vse16_v_u16m1(dst3 + i + vl + vl + vl, d4, vl);
        }
        // Скалярный остаток
#if defined(__clang__)
#pragma clang loop vectorize(disable)
#endif
        for (; i < len; i++) {
            dst0[i] = src[i * cn];
            dst1[i] = src[i * cn + 1];
            dst2[i] = src[i * cn + 2];
            dst3[i] = src[i * cn + 3];
        }
    }

    // Обработка оставшихся каналов (блоками по 4)
    for (; k < cn; k += 4) {
        uchar* dst0 = dst[k], * dst1 = dst[k + 1], * dst2 = dst[k + 2], * dst3 = dst[k + 3];
        i = 0;
        // Векторная обработка
        for (; i <= len - vl * 4; i += vl * 4) {
            auto a = __riscv_vlse16_v_u16m1(src + k + 0 + i * cn, cn * sizeof(uchar), vl);
            auto b = __riscv_vlse16_v_u16m1(src + k + 1 + i * cn, cn * sizeof(uchar), vl);
            auto c = __riscv_vlse16_v_u16m1(src + k + 2 + i * cn, cn * sizeof(uchar), vl);
            auto d = __riscv_vlse16_v_u16m1(src + k + 3 + i * cn, cn * sizeof(uchar), vl);
            __riscv_vse16_v_u16m1(dst0 + i, a, vl);
            __riscv_vse16_v_u16m1(dst1 + i, b, vl);
            __riscv_vse16_v_u16m1(dst2 + i, c, vl);
            __riscv_vse16_v_u16m1(dst3 + i, d, vl);

            auto a2 = __riscv_vlse16_v_u16m1(src + k + 0 + (i + vl) * cn, cn * sizeof(uchar), vl);
            auto b2 = __riscv_vlse16_v_u16m1(src + k + 1 + (i + vl) * cn, cn * sizeof(uchar), vl);
            auto c2 = __riscv_vlse16_v_u16m1(src + k + 2 + (i + vl) * cn, cn * sizeof(uchar), vl);
            auto d2 = __riscv_vlse16_v_u16m1(src + k + 3 + (i + vl) * cn, cn * sizeof(uchar), vl);
            __riscv_vse16_v_u16m1(dst0 + i + vl, a2, vl);
            __riscv_vse16_v_u16m1(dst1 + i + vl, b2, vl);
            __riscv_vse16_v_u16m1(dst2 + i + vl, c2, vl);
            __riscv_vse16_v_u16m1(dst3 + i + vl, d2, vl);

            auto a3 = __riscv_vlse16_v_u16m1(src + k + 0 + (i + vl + vl) * cn, cn * sizeof(uchar), vl);
            auto b3 = __riscv_vlse16_v_u16m1(src + k + 1 + (i + vl + vl) * cn, cn * sizeof(uchar), vl);
            auto c3 = __riscv_vlse16_v_u16m1(src + k + 2 + (i + vl + vl) * cn, cn * sizeof(uchar), vl);
            auto d3 = __riscv_vlse16_v_u16m1(src + k + 3 + (i + vl + vl) * cn, cn * sizeof(uchar), vl);
            __riscv_vse16_v_u16m1(dst0 + i + vl + vl, a3, vl);
            __riscv_vse16_v_u16m1(dst1 + i + vl + vl, b3, vl);
            __riscv_vse16_v_u16m1(dst2 + i + vl + vl, c3, vl);
            __riscv_vse16_v_u16m1(dst3 + i + vl + vl, d3, vl);

            auto a4 = __riscv_vlse16_v_u16m1(src + k + 0 + (i + vl + vl + vl) * cn, cn * sizeof(uchar), vl);
            auto b4 = __riscv_vlse16_v_u16m1(src + k + 1 + (i + vl + vl + vl) * cn, cn * sizeof(uchar), vl);
            auto c4 = __riscv_vlse16_v_u16m1(src + k + 2 + (i + vl + vl + vl) * cn, cn * sizeof(uchar), vl);
            auto d4 = __riscv_vlse16_v_u16m1(src + k + 3 + (i + vl + vl + vl) * cn, cn * sizeof(uchar), vl);
            __riscv_vse16_v_u16m1(dst0 + i + vl + vl + vl, a3, vl);
            __riscv_vse16_v_u16m1(dst1 + i + vl + vl + vl, b3, vl);
            __riscv_vse16_v_u16m1(dst2 + i + vl + vl + vl, c3, vl);
            __riscv_vse16_v_u16m1(dst3 + i + vl + vl + vl, d3, vl);
        }
        // Скалярный остаток
#if defined(__clang__)
#pragma clang loop vectorize(disable)
#endif
        for (; i < len; i++) {
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
