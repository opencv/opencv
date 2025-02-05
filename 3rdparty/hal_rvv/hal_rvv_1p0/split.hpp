// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//#ifndef OPENCV_HAL_RVV_SPLIT_HPP_INCLUDED
//#define OPENCV_HAL_RVV_SPLIT_HPP_INCLUDED

#include <riscv_vector.h>
#include <stdio.h>
namespace cv { namespace cv_hal_rvv {

#undef cv_hal_split8u
#define cv_hal_split8u cv::cv_hal_rvv::split8u

#if defined __GNUC__
__attribute__((optimize("no-tree-vectorize")))
#endif
      /*
inline int split8u(const uchar* src, uchar** dst, int len, int cn ) {
 
    printf("split8u\n");
    printf("split8u\n");
    printf("split8u\n");
    printf("split8u\n");
    printf("split8u\n");
    
    int k = cn % 4 ? cn % 4 : 4;
    int i = 0;
    int vl = __riscv_vsetvlmax_e8m1();
    if( k == 1 )
    {
        uchar* dst0 = dst[0];
        if(cn == 1) {
            memcpy(dst0, src, len * sizeof(uchar)); // векторизировать
        }
        else {
            for( ; i <= len - vl; i += vl)
            {
                auto a = __riscv_vle8_v_u8m1(src + i, vl);
                __riscv_vsse8_v_u8m1(dst0 + i*cn, sizeof(uchar)*cn, a, vl);
            }
            #if defined(__clang__)
            #pragma clang loop vectorize(disable)
            #endif
            for( ; i < len; i++)
                dst0[i*cn] = src[i];
        }
    }
        else if(k == 2) {
            uchar *dst0 = dst[0], *dst1 = dst[1];
            for( ; i <= len - vl; i += vl)
            {
                auto a = __riscv_vle8_v_u8m1(src + i, vl);
                auto b = __riscv_vle8_v_u8m1(src + i, vl);
                __riscv_vsse8_v_u8m1(dst0 + i*cn, sizeof(uchar)*cn, a, vl);
                __riscv_vsse8_v_u8m1(dst1 + i*cn + 1, sizeof(uchar)*cn, b, vl);
            }
            #if defined(__clang__)
            #pragma clang loop vectorize(disable)
            #endif
            for( ; i < len; i++ )
            {
                dst0[i*cn] = src[i];
                dst1[i*cn] = src[i + 1];
            }
        }
        else if(k == 3) {
            uchar *dst0 = dst[0], *dst1 = dst[1], *dst2 = dst[2];
            for( ; i <= len - vl; i += vl)
            {
                auto a = __riscv_vle8_v_u8m1(src + i, vl);
                auto b = __riscv_vle8_v_u8m1(src + i, vl);
                auto c = __riscv_vle8_v_u8m1(src + i, vl);
                __riscv_vsse8_v_u8m1(dst0 + i*cn, sizeof(uchar)*cn, a, vl);
                __riscv_vsse8_v_u8m1(dst1 + i*cn + 1, sizeof(uchar)*cn, b, vl);
                __riscv_vsse8_v_u8m1(dst2 + i*cn + 2, sizeof(uchar)*cn, c, vl);
            }
            #if defined(__clang__)
            #pragma clang loop vectorize(disable)
            #endif
            for( ; i < len; i++ )
            {
                dst0[i*cn] = src[i];
                dst1[i*cn] = src[i + 1];
                dst2[i*cn] = src[i + 2];
            }
        }
        else {
            uchar *dst0 = dst[0], *dst1 = dst[1], *dst2 = dst[2], *dst3 = dst[3];
            for( ; i <= len - vl; i += vl)
            {
                auto a = __riscv_vle8_v_u8m1(src + i, vl);
                auto b = __riscv_vle8_v_u8m1(src + i, vl);
                auto c = __riscv_vle8_v_u8m1(src + i, vl);
                auto d = __riscv_vle8_v_u8m1(src + i, vl);
                __riscv_vsse8_v_u8m1(dst0 + i*cn, sizeof(uchar)*cn, a, vl);
                __riscv_vsse8_v_u8m1(dst1 + i*cn + 1, sizeof(uchar)*cn, b, vl);
                __riscv_vsse8_v_u8m1(dst2 + i*cn + 2, sizeof(uchar)*cn, c, vl);
                __riscv_vsse8_v_u8m1(dst3 + i*cn + 3, sizeof(uchar)*cn, d, vl);
            }
            #if defined(__clang__)
            #pragma clang loop vectorize(disable)
            #endif
            for( ; i < len; i++ )
            {
                dst0[i*cn] = src[i];
                dst1[i*cn] = src[i + 1];
                dst2[i*cn] = src[i + 2];
                dst3[i*cn] = src[i + 3];
            }
        }
        #if defined(__clang__)
        #pragma clang loop vectorize(disable)
        #endif
        for( ; k < cn; k += 4 )
        {
            uchar *dst0 = dst[k], *dst1 = dst[k+1], *dst2 = dst[k+2], *dst3 = dst[k+3];
            i = 0;
            for( ; i <= len - vl; i += vl)
            {
                auto a = __riscv_vle8_v_u8m1(src + i, vl);
                auto b = __riscv_vle8_v_u8m1(src + i, vl);
                auto c = __riscv_vle8_v_u8m1(src + i, vl);
                auto d = __riscv_vle8_v_u8m1(src + i, vl);
                __riscv_vsse8_v_u8m1(dst0 + k+i*cn, sizeof(uchar)*cn, a, vl);
                __riscv_vsse8_v_u8m1(dst1 + k+i*cn + 1, sizeof(uchar)*cn, b, vl);
                __riscv_vsse8_v_u8m1(dst2 + k+i*cn + 2, sizeof(uchar)*cn, c, vl);
                __riscv_vsse8_v_u8m1(dst3 + k+i*cn + 3, sizeof(uchar)*cn, d, vl);
            }
            #if defined(__clang__)
            #pragma clang loop vectorize(disable)
            #endif
            for( ; i < len; i++ )
            {
                dst0[k+i*cn] = src[i];
                dst1[k+i*cn] = src[i + 1];
                dst2[k+i*cn] = src[i + 2];
                dst3[k+i*cn] = src[i + 3];
            }
        }
        return CV_HAL_ERROR_OK;
    }
    */

    /*
    inline int split8u(const uchar* src, uchar** dst, int len, int cn ) {
        int k = cn % 4 ? cn % 4 : 4;
        //printf("\n%\n", src);
        //printf("\n %i\n", k);
        int i = 0;
        int j;
        int vl = __riscv_vsetvlmax_e8m1();
        if( k == 1 )
        {
            uchar* dst0 = dst[0];
            if(cn == 1)
            {
                memcpy(dst0, src, len * sizeof(uchar));
            }
            else {
                for( ; i <= len - vl; i += vl)
                {
                     auto a = __riscv_vle8_v_u8m1(src + i, vl);
                     __riscv_vsse8_v_u8m1(dst0 + i* cn, sizeof(uchar)* cn, a, vl);
                }
                #if defined(__clang__)
                #pragma clang loop vectorize(disable)
                #endif
                for( ; i < len; i++)
                    dst0[i] = src[i * cn];
            }
            // else
            // {
            //     for( i = 0, j = 0 ; i < len; i++, j += cn )
            //         dst0[i] = src[j];
            // }
        }
        else if( k == 2 )
        {
            uchar *dst0 = dst[0], *dst1 = dst[1];
            i = j = 0;

            for( ; i < len; i++, j += cn )
            {
                dst0[i] = src[j];
                dst1[i] = src[j+1];
            }
        }
        else if( k == 3 )
        {
            uchar *dst0 = dst[0], *dst1 = dst[1], *dst2 = dst[2];
            i = j = 0;

            for( ; i < len; i++, j += cn )
            {
                dst0[i] = src[j];
                dst1[i] = src[j+1];
                dst2[i] = src[j+2];
            }
        }
        else
        {
            uchar *dst0 = dst[0], *dst1 = dst[1], *dst2 = dst[2], *dst3 = dst[3];
            i = j = 0;

            for( ; i < len; i++, j += cn )
            {
                dst0[i] = src[j]; dst1[i] = src[j+1];
                dst2[i] = src[j+2]; dst3[i] = src[j+3];
            }
        }

        for( ; k < cn; k += 4 )
        {
            uchar *dst0 = dst[k], *dst1 = dst[k+1], *dst2 = dst[k+2], *dst3 = dst[k+3];
            for( i = 0, j = k; i < len; i++, j += cn )
            {
                dst0[i] = src[j]; dst1[i] = src[j+1];
                dst2[i] = src[j+2]; dst3[i] = src[j+3];
            }
        }
        return CV_HAL_ERROR_OK;
    } */
    
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
        for (; i <= len - vl; i += vl) {
            auto a = __riscv_vlse8_v_u8m1(src + 0 + i * cn, cn * sizeof(uchar), vl);
            auto b = __riscv_vlse8_v_u8m1(src + 1 + i * cn, cn * sizeof(uchar), vl);
            __riscv_vse8_v_u8m1(dst0 + i, a, vl);
            __riscv_vse8_v_u8m1(dst1 + i, b, vl);
        }
        // Скалярный остаток
        #if defined(__clang__)
        #pragma clang loop vectorize(disable)
        #endif
        for (; i < len; i++) {
            int j = i * cn;
            dst0[i] = src[j];
            dst1[i] = src[j + 1];
        }
    } else if (k == 3) {
        uchar *dst0 = dst[0], *dst1 = dst[1], *dst2 = dst[2];
        // Векторная обработка
        for (; i <= len - vl; i += vl) {
            auto a = __riscv_vlse8_v_u8m1(src + 0 + i * cn, cn * sizeof(uchar), vl);
            auto b = __riscv_vlse8_v_u8m1(src + 1 + i * cn, cn * sizeof(uchar), vl);
            auto c = __riscv_vlse8_v_u8m1(src + 2 + i * cn, cn * sizeof(uchar), vl);
            __riscv_vse8_v_u8m1(dst0 + i, a, vl);
            __riscv_vse8_v_u8m1(dst1 + i, b, vl);
            __riscv_vse8_v_u8m1(dst2 + i, c, vl);
        }
        // Скалярный остаток
        #if defined(__clang__)
        #pragma clang loop vectorize(disable)
        #endif
        for (; i < len; i++) {
            int j = i * cn;
            dst0[i] = src[j];
            dst1[i] = src[j + 1];
            dst2[i] = src[j + 2];
        }
    } else {
        uchar *dst0 = dst[0], *dst1 = dst[1], *dst2 = dst[2], *dst3 = dst[3];
        // Векторная обработка
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
        // Скалярный остаток
        #if defined(__clang__)
        #pragma clang loop vectorize(disable)
        #endif
        for (; i < len; i++) {
            int j = i * cn;
            dst0[i] = src[j];
            dst1[i] = src[j + 1];
            dst2[i] = src[j + 2];
            dst3[i] = src[j + 3];
        }
    }

    // Обработка оставшихся каналов (блоками по 4)
    for (; k < cn; k += 4) {
        uchar *dst0 = dst[k], *dst1 = dst[k+1], *dst2 = dst[k+2], *dst3 = dst[k+3];
        i = 0;
        // Векторная обработка
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
        // Скалярный остаток
        #if defined(__clang__)
        #pragma clang loop vectorize(disable)
        #endif
        for (; i < len; i++) {
            int j = k + i * cn;
            dst0[i] = src[j];
            dst1[i] = src[j + 1];
            dst2[i] = src[j + 2];
            dst3[i] = src[j + 3];
        }
    }
    return CV_HAL_ERROR_OK;
}
}
}