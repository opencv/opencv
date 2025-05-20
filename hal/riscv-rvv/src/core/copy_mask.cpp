// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2025, SpaceMIT Inc., all rights reserved.
// Third party copyrights are property of their respective owners.

#include "rvv_hal.hpp"

namespace cv { namespace rvv_hal { namespace core {

#if CV_HAL_RVV_1P0_ENABLED

namespace {

#define CV_HAL_RVV_COPY_MASK_eXc1(X, mask_lmul) \
static int copyToMasked_e##X##c1(const uchar *src_data, size_t src_step, const uchar *mask_data, size_t mask_step, \
                                 uchar *dst_data, size_t dst_step, int width, int height) { \
    for (; height--; mask_data += mask_step, src_data += src_step, dst_data += dst_step) { \
        const uint##X##_t *src = (const uint##X##_t*)src_data; \
        uint##X##_t *dst = (uint##X##_t*)dst_data; \
        int vl; \
        for (int i = 0; i < width; i += vl) { \
            vl = __riscv_vsetvl_e8m##mask_lmul(width - i); \
            auto m = __riscv_vmsne(__riscv_vle8_v_u8m##mask_lmul(mask_data + i, vl), 0, vl); \
            auto v = __riscv_vle##X##_v_u##X##m8_m(m, src + i, vl); \
            __riscv_vse##X##_v_u##X##m8_m(m, dst + i, v, vl); \
        } \
    } \
    return CV_HAL_ERROR_OK; \
}

CV_HAL_RVV_COPY_MASK_eXc1(8,  8)
CV_HAL_RVV_COPY_MASK_eXc1(16, 4)
CV_HAL_RVV_COPY_MASK_eXc1(32, 2)
CV_HAL_RVV_COPY_MASK_eXc1(64, 1)

#define CV_HAL_RVV_COPY_MASK_eXc3(X, mask_lmul) \
static int copyToMasked_e##X##c3(const uchar *src_data, size_t src_step, const uchar *mask_data, size_t mask_step, \
                                 uchar *dst_data, size_t dst_step, int width, int height) { \
    for (; height--; mask_data += mask_step, src_data += src_step, dst_data += dst_step) { \
        const uint##X##_t *src = (const uint##X##_t*)src_data; \
        uint##X##_t *dst = (uint##X##_t*)dst_data; \
        int vl; \
        for (int i = 0; i < width; i += vl) { \
            vl = __riscv_vsetvl_e8m##mask_lmul(width - i); \
            auto m = __riscv_vmsne(__riscv_vle8_v_u8m##mask_lmul(mask_data + i, vl), 0, vl); \
            auto v = __riscv_vlseg3e##X##_v_u##X##m2x3_m(m, src + 3 * i, vl); \
            __riscv_vsseg3e##X##_v_u##X##m2x3_m(m, dst + 3 * i, v, vl); \
        } \
    } \
    return CV_HAL_ERROR_OK; \
}

CV_HAL_RVV_COPY_MASK_eXc3(8,  2)
CV_HAL_RVV_COPY_MASK_eXc3(16, 1)
CV_HAL_RVV_COPY_MASK_eXc3(32, f2)
CV_HAL_RVV_COPY_MASK_eXc3(64, f4)

static int copyToMasked_e64c2(const uchar *src_data, size_t src_step,
                              const uchar *mask_data, size_t mask_step,
                              uchar *dst_data, size_t dst_step, int width,
                              int height) {
    for (; height--; mask_data += mask_step, src_data += src_step, dst_data += dst_step) {
        const uint64_t *src = (const uint64_t *)src_data;
        uint64_t *dst = (uint64_t *)dst_data;
        int vl;
        for (int i = 0; i < width; i += vl) {
            vl = __riscv_vsetvl_e8mf2(width - i);
            auto m = __riscv_vmsne(__riscv_vle8_v_u8mf2(mask_data + i, vl), 0, vl);
            auto v = __riscv_vlseg2e64_v_u64m4x2_m(m, src + 2 * i, vl);
            __riscv_vsseg2e64_v_u64m4x2_m(m, dst + 2 * i, v, vl);
        }
    }
    return CV_HAL_ERROR_OK;
}

static int copyToMasked_e64c4(const uchar *src_data, size_t src_step,
                              const uchar *mask_data, size_t mask_step,
                              uchar *dst_data, size_t dst_step, int width,
                              int height) {
    for (; height--; mask_data += mask_step, src_data += src_step, dst_data += dst_step) {
        const uint64_t *src = (const uint64_t *)src_data;
        uint64_t *dst = (uint64_t *)dst_data;
        int vl;
        for (int i = 0; i < width; i += vl) {
            vl = __riscv_vsetvl_e8mf4(width - i);
            auto m = __riscv_vmsne(__riscv_vle8_v_u8mf4(mask_data + i, vl), 0, vl);
            auto v = __riscv_vlseg4e64_v_u64m2x4_m(m, src + 4 * i, vl);
            __riscv_vsseg4e64_v_u64m2x4_m(m, dst + 4 * i, v, vl);
        }
    }
    return CV_HAL_ERROR_OK;
}

} // anonymous

using CopyToMaskedFunc = int (*)(const uchar*, size_t, const uchar*, size_t, uchar*, size_t, int, int);
int copyToMasked(const uchar *src_data, size_t src_step, uchar *dst_data, size_t dst_step, int width, int height,
                 int type, const uchar *mask_data, size_t mask_step, int mask_type) {
    int cn = CV_MAT_CN(type);
    int mdepth = CV_MAT_DEPTH(mask_type), mcn = CV_MAT_CN(mask_type);

    if (mcn > 1 || mdepth != CV_8U) {
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    static CopyToMaskedFunc tab[] = {
        0, copyToMasked_e8c1, copyToMasked_e16c1, copyToMasked_e8c3,
        copyToMasked_e32c1, 0, copyToMasked_e16c3, 0,
        copyToMasked_e64c1, 0, 0, 0,
        copyToMasked_e32c3, 0, 0, 0,
        copyToMasked_e64c2, 0, 0, 0,
        0, 0, 0, 0,
        copyToMasked_e64c3, 0, 0, 0,
        0, 0, 0, 0,
        copyToMasked_e64c4
    };
    CopyToMaskedFunc func = tab[CV_ELEM_SIZE(type)];
    if (func == nullptr) {
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    size_t elem_size1 = static_cast<size_t>(CV_ELEM_SIZE1(type));
    bool src_continuous = (src_step == width * elem_size1 * cn || (src_step != width * elem_size1 * cn && height == 1));
    bool dst_continuous = (dst_step == width * elem_size1 * cn || (dst_step != width * elem_size1 * cn && height == 1));
    bool mask_continuous = (mask_step == static_cast<size_t>(width));
    size_t nplanes = 1;
    int _width = width, _height = height;
    if (!src_continuous || !dst_continuous || !mask_continuous) {
        nplanes = height;
        _width = width * mcn;
        _height = 1;
    }

    auto _src = src_data;
    auto _mask = mask_data;
    auto _dst = dst_data;
    for (size_t i = 0; i < nplanes; i++) {
        if (!src_continuous || !dst_continuous || !mask_continuous) {
            _src = src_data + src_step * i;
            _mask = mask_data + mask_step * i;
            _dst = dst_data + dst_step * i;
        }
        func(_src, src_step, _mask, mask_step, _dst, dst_step, _width, _height);
    }

    return CV_HAL_ERROR_OK;
}

#endif // CV_HAL_RVV_1P0_ENABLED

}}} // cv::rvv_hal::core
