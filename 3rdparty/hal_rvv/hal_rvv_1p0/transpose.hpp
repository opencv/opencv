// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2025, SpaceMIT Inc., all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_HAL_RVV_TRANSPOSE_HPP_INCLUDED
#define OPENCV_HAL_RVV_TRANSPOSE_HPP_INCLUDED

#include <riscv_vector.h>

namespace cv { namespace cv_hal_rvv { namespace transpose {

/*
static void transpose_8u(const uchar *src_data, size_t src_step, uchar *dst_data, size_t dst_step, int src_width, int src_height) {
    for (int h = 0; h < src_height; h++) {
        const uchar *src = src_data + h * src_step;
        uchar *dst = dst_data + h;
        int vl;
        for (int w = 0; w < src_width; w += vl) {
            vl = __riscv_vsetvl_e8m8(src_width - w);

            auto v = __riscv_vle8_v_u8m8(src + w, vl);
            __riscv_vsse8(dst + w * dst_step, dst_step, v, vl);
        }
    }
}
*/

static void transpose_16u(const uchar *src_data, size_t src_step, uchar *dst_data, size_t dst_step, int src_width, int src_height) {
    auto transpose_16u_8x8 = [](const ushort *src, size_t src_step, ushort *dst, size_t dst_step) {
        auto v =__riscv_vlsseg8e16_v_u16m1x8(src, src_step, 8);

        auto v0 = __riscv_vget_v_u16m1x8_u16m1(v, 0);
        auto v1 = __riscv_vget_v_u16m1x8_u16m1(v, 1);
        auto v2 = __riscv_vget_v_u16m1x8_u16m1(v, 2);
        auto v3 = __riscv_vget_v_u16m1x8_u16m1(v, 3);
        auto v4 = __riscv_vget_v_u16m1x8_u16m1(v, 4);
        auto v5 = __riscv_vget_v_u16m1x8_u16m1(v, 5);
        auto v6 = __riscv_vget_v_u16m1x8_u16m1(v, 6);
        auto v7 = __riscv_vget_v_u16m1x8_u16m1(v, 7);

        __riscv_vse16(dst, v0, 8);
        __riscv_vse16(dst + dst_step, v1, 8);
        __riscv_vse16(dst + 2 * dst_step, v2, 8);
        __riscv_vse16(dst + 3 * dst_step, v3, 8);
        __riscv_vse16(dst + 4 * dst_step, v4, 8);
        __riscv_vse16(dst + 5 * dst_step, v5, 8);
        __riscv_vse16(dst + 6 * dst_step, v6, 8);
        __riscv_vse16(dst + 7 * dst_step, v7, 8);
    };

    constexpr int block_size = 128; // multiple of 8
    size_t src_step_base = src_step / sizeof(ushort);
    size_t dst_step_base = dst_step / sizeof(ushort);

    int h = 0, w = 0;
    for (; h <= src_height - block_size; h += block_size) {
        const ushort *_src = (const ushort*)(src_data) + h * src_step_base;
        ushort *_dst = (ushort*)(dst_data) + h;
        for (w = 0; w <= src_width - block_size; w += block_size) {
            const auto *src = _src + w;
            auto *dst = _dst + w * dst_step_base;
            for (int i = 0; i < block_size; i += 8) {
                for (int j = 0; j < block_size; j += 8) {
                    transpose_16u_8x8(src + j, src_step, dst + j * dst_step_base, dst_step_base);
                }
                src += 8 * src_step_base;
                dst += 8;
            }
        }

        int vl;
        for (int j = w; j < src_width; j += vl) {
            const auto *src = _src + j;
            auto *dst = _dst + j * dst_step_base;
            vl = __riscv_vsetvl_e16m2(src_width - j);
            for (int i = 0; i < block_size; i++) {
                auto v = __riscv_vle16_v_u16m2(src, vl);
                __riscv_vsse16(dst, dst_step, v, vl);
                src += src_step_base;
                dst += 1;
            }
        }
    }
    for (; h < src_height; h++) {
        const ushort *src = (const ushort*)(src_data) + h * src_step_base;
        ushort *dst = (ushort*)(dst_data) + h;
        int vl;
        for (w = 0; w < src_width; w += vl) {
            vl = __riscv_vsetvl_e16m8(src_width - w);
            auto v = __riscv_vle16_v_u16m8(src + w, vl);
            __riscv_vsse16(dst + w * dst_step_base, dst_step, v, vl);
        }
    }
}

static void transpose_32s(const uchar *src_data, size_t src_step, uchar *dst_data, size_t dst_step, int src_width, int src_height) {
    auto transpose_32s_4x4 = [](const int *src, size_t src_step, int *dst, size_t dst_step) {
        auto v =__riscv_vlsseg4e32_v_i32m1x4(src, src_step, 4);

        auto v0 = __riscv_vget_v_i32m1x4_i32m1(v, 0);
        auto v1 = __riscv_vget_v_i32m1x4_i32m1(v, 1);
        auto v2 = __riscv_vget_v_i32m1x4_i32m1(v, 2);
        auto v3 = __riscv_vget_v_i32m1x4_i32m1(v, 3);

        __riscv_vse32(dst, v0, 4);
        __riscv_vse32(dst + dst_step, v1, 4);
        __riscv_vse32(dst + 2 * dst_step, v2, 4);
        __riscv_vse32(dst + 3 * dst_step, v3, 4);
    };

    constexpr int block_size = 88; // multiple of 4
    size_t src_step_base = src_step / sizeof(int);
    size_t dst_step_base = dst_step / sizeof(int);

    int h = 0, w = 0;
    for (; h <= src_height - block_size; h += block_size) {
        const int *_src = (const int*)(src_data) + h * src_step_base;
        int *_dst = (int*)(dst_data) + h;
        for (w = 0; w <= src_width - block_size; w += block_size) {
            const auto *src = _src + w;
            auto *dst = _dst + w * dst_step_base;
            for (int i = 0; i < block_size; i += 4) {
                for (int j = 0; j < block_size; j += 4) {
                    transpose_32s_4x4(src + j, src_step, dst + j * dst_step_base, dst_step_base);
                }
                src += 4 * src_step_base;
                dst += 4;
            }
        }

        int vl;
        for (; w < src_width; w += vl) {
            const auto *src = _src + w;
            auto *dst = _dst + w * dst_step_base;
            vl = __riscv_vsetvl_e32m2(src_width - w);
            for (int i = 0; i < block_size; i++) {
                auto v = __riscv_vle32_v_i32m2(src, vl);
                __riscv_vsse32(dst, dst_step, v, vl);
                src += src_step_base;
                dst += 1;
            }
        }
    }
    for (; h < src_height; h++) {
        const int *src = (const int*)(src_data) + h * src_step_base;
        int *dst = (int*)(dst_data) + h;
        int vl;
        for (w = 0; w < src_width; w += vl) {
            vl = __riscv_vsetvl_e32m8(src_width - w);
            auto v = __riscv_vle32_v_i32m8(src + w, vl);
            __riscv_vsse32(dst + w * dst_step_base, dst_step, v, vl);
        }
    }
}

#undef cv_hal_transpose2d
#define cv_hal_transpose2d cv::cv_hal_rvv::transpose::transpose2d

using Transpose2dFunc = void (*)(const uchar*, size_t, uchar*, size_t, int, int);
inline int transpose2d(const uchar* src_data, size_t src_step, uchar* dst_data, size_t dst_step,
                       int src_width, int src_height, int element_size) {
    if (src_data == dst_data) {
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    // static Transpose2dFunc tab[] = {
    //     0, transpose_8u, transpose_16u, transpose_8uC3, transpose_32s, 0, transpose_16uC3, 0,
    //     transpose_32sC2, 0, 0, 0, transpose_32sC3, 0, 0, 0, transpose_32sC4,
    //     0, 0, 0, 0, 0, 0, 0, transpose_32sC6, 0, 0, 0, 0, 0, 0, 0, transpose_32sC8
    // };
    static Transpose2dFunc tab[] = {
        0, 0, transpose_16u, 0,
        transpose_32s, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0
    };
    Transpose2dFunc func = tab[element_size];
    if (!func) {
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    func(src_data, src_step, dst_data, dst_step, src_width, src_height);

    return CV_HAL_ERROR_OK;
}

}}} // cv::cv_hal_rvv::transpose

#endif // OPENCV_HAL_RVV_TRANSPOSE_HPP_INCLUDED
