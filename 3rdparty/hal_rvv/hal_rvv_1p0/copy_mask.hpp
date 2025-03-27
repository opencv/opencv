// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2025, SpaceMIT Inc., all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_HAL_RVV_COPY_MASK_HPP_INCLUDED
#define OPENCV_HAL_RVV_COPY_MASK_HPP_INCLUDED

#include <riscv_vector.h>

namespace cv { namespace cv_hal_rvv {

#undef cv_hal_copyToMask
#define cv_hal_copyToMask cv::cv_hal_rvv::copyToMask

namespace {

#define CV_HAL_RVV_COPY_MASK_eXc1(X, mask_lmul) \
static inline int copyToMask_e##X##c1(const uchar *src_data, size_t src_step, const uchar *mask_data, size_t mask_step, \
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
static inline int copyToMask_e##X##c3(const uchar *src_data, size_t src_step, const uchar *mask_data, size_t mask_step, \
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

static inline int copyToMask_e64c2(const uchar *src_data, size_t src_step,
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

static inline int copyToMask_e64c4(const uchar *src_data, size_t src_step,
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

inline int copyToMask(const uchar *src_data, size_t src_step, uchar *dst_data, size_t dst_step, int width, int height,
                      int type, const uchar *mask_data, size_t mask_step, int mask_type) {
    int depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);
    int mdepth = CV_MAT_DEPTH(mask_type), mcn = CV_MAT_CN(mask_type);

    if (mcn > 1 || mdepth != CV_8U) {
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    switch (depth) {
        case CV_8U: {}
        case CV_8S: switch (cn) {
            case 1: return copyToMask_e8c1(src_data, src_step, mask_data, mask_step, dst_data, dst_step, width, height);
            case 2: return copyToMask_e16c1(src_data, src_step, mask_data, mask_step, dst_data, dst_step, width, height);
            case 3: return copyToMask_e8c3(src_data, src_step, mask_data, mask_step, dst_data, dst_step, width, height);
            case 4: return copyToMask_e32c1(src_data, src_step, mask_data, mask_step, dst_data, dst_step, width, height);
            case 6: return copyToMask_e16c3(src_data, src_step, mask_data, mask_step, dst_data, dst_step, width, height);
            case 8: return copyToMask_e64c1(src_data, src_step, mask_data, mask_step, dst_data, dst_step, width, height);
            default: return CV_HAL_ERROR_NOT_IMPLEMENTED;
        }
        case CV_16U: {}
        case CV_16S: switch (cn) {
            case 1: return copyToMask_e16c1(src_data, src_step, mask_data, mask_step, dst_data, dst_step, width, height);
            case 2: return copyToMask_e32c1(src_data, src_step, mask_data, mask_step, dst_data, dst_step, width, height);
            case 3: return copyToMask_e16c3(src_data, src_step, mask_data, mask_step, dst_data, dst_step, width, height);
            case 4: return copyToMask_e64c1(src_data, src_step, mask_data, mask_step, dst_data, dst_step, width, height);
            case 6: return copyToMask_e32c3(src_data, src_step, mask_data, mask_step, dst_data, dst_step, width, height);
            case 8: return copyToMask_e64c2(src_data, src_step, mask_data, mask_step, dst_data, dst_step, width, height);
            default: return CV_HAL_ERROR_NOT_IMPLEMENTED;
        }
        case CV_32S: {}
        case CV_32F: switch (cn) {
            case 1: return copyToMask_e32c1(src_data, src_step, mask_data, mask_step, dst_data, dst_step, width, height);
            case 2: return copyToMask_e64c1(src_data, src_step, mask_data, mask_step, dst_data, dst_step, width, height);
            case 3: return copyToMask_e32c3(src_data, src_step, mask_data, mask_step, dst_data, dst_step, width, height);
            case 4: return copyToMask_e64c2(src_data, src_step, mask_data, mask_step, dst_data, dst_step, width, height);
            case 6: return copyToMask_e64c3(src_data, src_step, mask_data, mask_step, dst_data, dst_step, width, height);
            case 8: return copyToMask_e64c4(src_data, src_step, mask_data, mask_step, dst_data, dst_step, width, height);
            default: return CV_HAL_ERROR_NOT_IMPLEMENTED;
        }
        case CV_64F: switch (cn) {
            case 1: return copyToMask_e64c1(src_data, src_step, mask_data, mask_step, dst_data, dst_step, width, height);
            case 2: return copyToMask_e64c2(src_data, src_step, mask_data, mask_step, dst_data, dst_step, width, height);
            case 3: return copyToMask_e64c3(src_data, src_step, mask_data, mask_step, dst_data, dst_step, width, height);
            case 4: return copyToMask_e64c4(src_data, src_step, mask_data, mask_step, dst_data, dst_step, width, height);
            default: return CV_HAL_ERROR_NOT_IMPLEMENTED;
        }
        default: return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }
}

}} // cv::cv_hal_rvv

#endif
