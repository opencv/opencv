// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2025, SpaceMIT Inc., all rights reserved.
// Third party copyrights are property of their respective owners.

#include "rvv_hal.hpp"

#if defined (__clang__) && __clang_major__ < 18
#define OPENCV_HAL_IMPL_RVV_VCREATE_x4(suffix, width, v0, v1, v2, v3) \
    __riscv_vset_v_##suffix##m##width##_##suffix##m##width##x4(v, 0, v0); \
    v = __riscv_vset(v, 1, v1); \
    v = __riscv_vset(v, 2, v2); \
    v = __riscv_vset(v, 3, v3);

#define OPENCV_HAL_IMPL_RVV_VCREATE_x8(suffix, width, v0, v1, v2, v3, v4, v5, v6, v7) \
    __riscv_vset_v_##suffix##m##width##_##suffix##m##width##x8(v, 0, v0); \
    v = __riscv_vset(v, 1, v1); \
    v = __riscv_vset(v, 2, v2); \
    v = __riscv_vset(v, 3, v3); \
    v = __riscv_vset(v, 4, v4); \
    v = __riscv_vset(v, 5, v5); \
    v = __riscv_vset(v, 6, v6); \
    v = __riscv_vset(v, 7, v7);

#define __riscv_vcreate_v_u8m1x8(v0, v1, v2, v3, v4, v5, v6, v7) OPENCV_HAL_IMPL_RVV_VCREATE_x8(u8, 1, v0, v1, v2, v3, v4, v5, v6, v7)
#define __riscv_vcreate_v_u16m1x8(v0, v1, v2, v3, v4, v5, v6, v7) OPENCV_HAL_IMPL_RVV_VCREATE_x8(u16, 1, v0, v1, v2, v3, v4, v5, v6, v7)
#define __riscv_vcreate_v_i32m1x4(v0, v1, v2, v3) OPENCV_HAL_IMPL_RVV_VCREATE_x4(i32, 1, v0, v1, v2, v3)
#define __riscv_vcreate_v_i64m1x8(v0, v1, v2, v3, v4, v5, v6, v7) OPENCV_HAL_IMPL_RVV_VCREATE_x8(i64, 1, v0, v1, v2, v3, v4, v5, v6, v7)
#endif

namespace cv { namespace rvv_hal { namespace core {

#if CV_HAL_RVV_1P0_ENABLED

static void transpose2d_8u(const uchar *src_data, size_t src_step, uchar *dst_data, size_t dst_step, int src_width, int src_height) {
    auto transpose_8u_8xVl = [](const uchar *src, size_t src_step, uchar *dst, size_t dst_step, const int vl) {
        auto v0 = __riscv_vle8_v_u8m1(src, vl);
        auto v1 = __riscv_vle8_v_u8m1(src + src_step, vl);
        auto v2 = __riscv_vle8_v_u8m1(src + 2 * src_step, vl);
        auto v3 = __riscv_vle8_v_u8m1(src + 3 * src_step, vl);
        auto v4 = __riscv_vle8_v_u8m1(src + 4 * src_step, vl);
        auto v5 = __riscv_vle8_v_u8m1(src + 5 * src_step, vl);
        auto v6 = __riscv_vle8_v_u8m1(src + 6 * src_step, vl);
        auto v7 = __riscv_vle8_v_u8m1(src + 7 * src_step, vl);
        vuint8m1x8_t v = __riscv_vcreate_v_u8m1x8(v0, v1, v2, v3, v4, v5, v6, v7);
        __riscv_vssseg8e8(dst, dst_step, v, vl);
    };

    int h = 0, w = 0;
    for (; h <= src_height - 8; h += 8) {
        const uchar *src = src_data + h * src_step;
        uchar *dst = dst_data + h;
        int vl;
        for (w = 0; w < src_width; w += vl) {
            vl = __riscv_vsetvl_e8m1(src_width - w);
            transpose_8u_8xVl(src + w, src_step, dst + w * dst_step, dst_step, vl);
        }
    }
    for (; h < src_height; h++) {
        const uchar *src = src_data + h * src_step;
        uchar *dst = dst_data + h;
        int vl;
        for (w = 0; w < src_width; w += vl) {
            vl = __riscv_vsetvl_e8m8(src_width - w);
            auto v = __riscv_vle8_v_u8m8(src + w, vl);
            __riscv_vsse8(dst + w * dst_step, dst_step, v, vl);
        }
    }
}

static void transpose2d_16u(const uchar *src_data, size_t src_step, uchar *dst_data, size_t dst_step, int src_width, int src_height) {
    auto transpose_16u_8xVl = [](const ushort *src, size_t src_step, ushort *dst, size_t dst_step, const int vl) {
        auto v0 = __riscv_vle16_v_u16m1(src, vl);
        auto v1 = __riscv_vle16_v_u16m1(src + src_step, vl);
        auto v2 = __riscv_vle16_v_u16m1(src + 2 * src_step, vl);
        auto v3 = __riscv_vle16_v_u16m1(src + 3 * src_step, vl);
        auto v4 = __riscv_vle16_v_u16m1(src + 4 * src_step, vl);
        auto v5 = __riscv_vle16_v_u16m1(src + 5 * src_step, vl);
        auto v6 = __riscv_vle16_v_u16m1(src + 6 * src_step, vl);
        auto v7 = __riscv_vle16_v_u16m1(src + 7 * src_step, vl);
        vuint16m1x8_t v = __riscv_vcreate_v_u16m1x8(v0, v1, v2, v3, v4, v5, v6, v7);
        __riscv_vssseg8e16(dst, dst_step, v, vl);
    };

    size_t src_step_base = src_step / sizeof(ushort);
    size_t dst_step_base = dst_step / sizeof(ushort);

    int h = 0, w = 0;
    for (; h <= src_height - 8; h += 8) {
        const ushort *src = (const ushort*)(src_data) + h * src_step_base;
        ushort *dst = (ushort*)(dst_data) + h;
        int vl;
        for (w = 0; w < src_width; w += vl) {
            vl = __riscv_vsetvl_e16m1(src_width - w);
            transpose_16u_8xVl(src + w, src_step_base, dst + w * dst_step_base, dst_step, vl);
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

static void transpose2d_32s(const uchar *src_data, size_t src_step, uchar *dst_data, size_t dst_step, int src_width, int src_height) {
    auto transpose_32s_4xVl = [](const int *src, size_t src_step, int *dst, size_t dst_step, const int vl) {
        auto v0 = __riscv_vle32_v_i32m1(src, vl);
        auto v1 = __riscv_vle32_v_i32m1(src + src_step, vl);
        auto v2 = __riscv_vle32_v_i32m1(src + 2 * src_step, vl);
        auto v3 = __riscv_vle32_v_i32m1(src + 3 * src_step, vl);
        vint32m1x4_t v = __riscv_vcreate_v_i32m1x4(v0, v1, v2, v3);
        __riscv_vssseg4e32(dst, dst_step, v, vl);
    };

    size_t src_step_base = src_step / sizeof(int);
    size_t dst_step_base = dst_step / sizeof(int);

    int h = 0, w = 0;
    for (; h <= src_height - 4; h += 4) {
        const int *src = (const int*)(src_data) + h * src_step_base;
        int *dst = (int*)(dst_data) + h;
        int vl;
        for (w = 0; w < src_width; w += vl) {
            vl = __riscv_vsetvl_e32m1(src_width - w);
            transpose_32s_4xVl(src + w, src_step_base, dst + w * dst_step_base, dst_step, vl);
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

static void transpose2d_32sC2(const uchar *src_data, size_t src_step, uchar *dst_data, size_t dst_step, int src_width, int src_height) {
    auto transpose_64s_8xVl = [](const int64_t *src, size_t src_step, int64_t *dst, size_t dst_step, const int vl) {
        auto v0 = __riscv_vle64_v_i64m1(src, vl);
        auto v1 = __riscv_vle64_v_i64m1(src + src_step, vl);
        auto v2 = __riscv_vle64_v_i64m1(src + 2 * src_step, vl);
        auto v3 = __riscv_vle64_v_i64m1(src + 3 * src_step, vl);
        auto v4 = __riscv_vle64_v_i64m1(src + 4 * src_step, vl);
        auto v5 = __riscv_vle64_v_i64m1(src + 5 * src_step, vl);
        auto v6 = __riscv_vle64_v_i64m1(src + 6 * src_step, vl);
        auto v7 = __riscv_vle64_v_i64m1(src + 7 * src_step, vl);
        vint64m1x8_t v = __riscv_vcreate_v_i64m1x8(v0, v1, v2, v3, v4, v5, v6, v7);
        __riscv_vssseg8e64(dst, dst_step, v, vl);
    };

    size_t src_step_base = src_step / sizeof(int64_t);
    size_t dst_step_base = dst_step / sizeof(int64_t);

    int h = 0, w = 0;
    for (; h <= src_height - 8; h += 8) {
        const int64_t *src = (const int64_t*)(src_data) + h * src_step_base;
        int64_t *dst = (int64_t*)(dst_data) + h;
        int vl;
        for (w = 0; w < src_width; w += vl) {
            vl = __riscv_vsetvl_e64m1(src_width - w);
            transpose_64s_8xVl(src + w, src_step_base, dst + w * dst_step_base, dst_step, vl);
        }
    }
    for (; h < src_height; h++) {
        const int64_t *src = (const int64_t*)(src_data) + h * src_step_base;
        int64_t *dst = (int64_t*)(dst_data) + h;
        int vl;
        for (w = 0; w < src_width; w += vl) {
            vl = __riscv_vsetvl_e64m8(src_width - w);
            auto v = __riscv_vle64_v_i64m8(src + w, vl);
            __riscv_vsse64(dst + w * dst_step_base, dst_step, v, vl);
        }
    }
}

using Transpose2dFunc = void (*)(const uchar*, size_t, uchar*, size_t, int, int);
int transpose2d(const uchar* src_data, size_t src_step, uchar* dst_data, size_t dst_step,
                int src_width, int src_height, int element_size) {
    if (src_data == dst_data) {
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    static Transpose2dFunc tab[] = {
        0, transpose2d_8u, transpose2d_16u, 0,
        transpose2d_32s, 0, 0, 0,
        transpose2d_32sC2, 0, 0, 0,
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

#endif // CV_HAL_RVV_1P0_ENABLED

}}} // cv::rvv_hal::core
