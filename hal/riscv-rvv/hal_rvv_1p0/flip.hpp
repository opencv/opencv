// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2025, Institute of Software, Chinese Academy of Sciences.
// Copyright (C) 2025, SpaceMIT Inc., all rights reserved.

#ifndef OPENCV_HAL_RVV_FLIP_HPP_INCLUDED
#define OPENCV_HAL_RVV_FLIP_HPP_INCLUDED


#include <riscv_vector.h>
#include <opencv2/core/base.hpp>
#include "hal_rvv_1p0/types.hpp"

namespace cv { namespace cv_hal_rvv {

#undef cv_hal_flip
#define cv_hal_flip cv::cv_hal_rvv::flip

namespace {

#define CV_HAL_RVV_FLIPY_C1(name, _Tps, RVV) \
inline void flip_##name(const uchar* src_data, size_t src_step, uchar* dst_data, size_t dst_step, int src_width, int src_height, int flip_mode) { \
    for (int h = 0; h < src_height; h++) { \
        const _Tps* src_row = (const _Tps*)(src_data + src_step * h); \
        _Tps* dst_row = (_Tps*)(dst_data + dst_step * (flip_mode < 0 ? (src_height - h) : (h + 1))); \
        int vl; \
        for (int w = 0; w < src_width; w += vl) { \
            vl = RVV::setvl(src_width - w); \
            RVV::VecType indices = __riscv_vrsub(RVV::vid(vl), vl - 1, vl); \
            auto v = RVV::vload(src_row + w, vl); \
            RVV::vstore(dst_row - w - vl, __riscv_vrgather(v, indices, vl), vl); \
        } \
    } \
}
CV_HAL_RVV_FLIPY_C1(8UC1, uchar, RVV_U8M8)
CV_HAL_RVV_FLIPY_C1(16UC1, ushort, RVV_U16M8)
CV_HAL_RVV_FLIPY_C1(32UC1, unsigned, RVV_U32M8)
CV_HAL_RVV_FLIPY_C1(64UC1, uint64_t, RVV_U64M8)

#if defined (__clang__) && __clang_major__ < 18
#define OPENCV_HAL_IMPL_RVV_VCREATE_x3(suffix, width, v0, v1, v2) \
    __riscv_vset_v_##suffix##m##width##_##suffix##m##width##x3(v, 0, v0); \
    v = __riscv_vset(v, 1, v1); \
    v = __riscv_vset(v, 2, v2);
#define __riscv_vcreate_v_u8m2x3(v0, v1, v2)  OPENCV_HAL_IMPL_RVV_VCREATE_x3(u8, 2, v0, v1, v2)
#define __riscv_vcreate_v_u16m2x3(v0, v1, v2) OPENCV_HAL_IMPL_RVV_VCREATE_x3(u16, 2, v0, v1, v2)
#define __riscv_vcreate_v_u32m2x3(v0, v1, v2) OPENCV_HAL_IMPL_RVV_VCREATE_x3(u32, 2, v0, v1, v2)
#define __riscv_vcreate_v_u64m2x3(v0, v1, v2) OPENCV_HAL_IMPL_RVV_VCREATE_x3(u64, 2, v0, v1, v2)
#endif

#define CV_HAL_RVV_FLIPY_C3_TYPES(width) \
struct RVV_C3_U##width##M2 : RVV_U##width##M2 { \
    static inline vuint##width##m2x3_t vload3(const uint##width##_t *base, size_t vl) { return __riscv_vlseg3e##width##_v_u##width##m2x3(base, vl); } \
    static inline vuint##width##m2x3_t vflip3(const vuint##width##m2x3_t &v_tuple, const vuint##width##m2_t &indices, size_t vl) { \
        auto v0 = __riscv_vrgather(__riscv_vget_u##width##m2(v_tuple, 0), indices, vl); \
        auto v1 = __riscv_vrgather(__riscv_vget_u##width##m2(v_tuple, 1), indices, vl); \
        auto v2 = __riscv_vrgather(__riscv_vget_u##width##m2(v_tuple, 2), indices, vl); \
        vuint##width##m2x3_t v = __riscv_vcreate_v_u##width##m2x3(v0, v1, v2); \
        return v; \
    } \
    static inline void vstore3(uint##width##_t *base, const vuint##width##m2x3_t &v_tuple, size_t vl) { __riscv_vsseg3e##width(base, v_tuple, vl); } \
};
CV_HAL_RVV_FLIPY_C3_TYPES(8)
CV_HAL_RVV_FLIPY_C3_TYPES(16)
CV_HAL_RVV_FLIPY_C3_TYPES(32)
CV_HAL_RVV_FLIPY_C3_TYPES(64)

#define CV_HAL_RVV_FLIPY_C3(name, _Tps, RVV) \
inline void flip_##name(const uchar* src_data, size_t src_step, uchar* dst_data, size_t dst_step, int src_width, int src_height, int flip_mode) { \
    for (int h = 0; h < src_height; h++) { \
        const _Tps* src_row = (const _Tps*)(src_data + src_step * h); \
        _Tps* dst_row = (_Tps*)(dst_data + dst_step * (flip_mode < 0 ? (src_height - h) : (h + 1))); \
        int vl; \
        for (int w = 0; w < src_width; w += vl) { \
            vl = RVV::setvl(src_width - w); \
            RVV::VecType indices = __riscv_vrsub(RVV::vid(vl), vl - 1, vl); \
            auto v = RVV::vload3(src_row + 3 * w, vl); \
            auto flipped = RVV::vflip3(v, indices, vl); \
            RVV::vstore3(dst_row - 3 * (w + vl), flipped, vl); \
        } \
    } \
}
CV_HAL_RVV_FLIPY_C3(8UC3, uchar, RVV_C3_U8M2)
CV_HAL_RVV_FLIPY_C3(16UC3, ushort, RVV_C3_U16M2)
CV_HAL_RVV_FLIPY_C3(32UC3, unsigned, RVV_C3_U32M2)
CV_HAL_RVV_FLIPY_C3(64UC3, uint64_t, RVV_C3_U64M2)

struct FlipVlen256
{
    using SrcType = RVV_U8M8;
    using TabType = RVV_U8M8;
    using TabVecType = typename TabType::VecType;

    static inline void gather(const uchar* src, TabVecType tab, uchar* dst, size_t vl)
    {
        auto src_v = SrcType::vload(src, vl);
        SrcType::vstore(dst, __riscv_vrgather(src_v, tab, vl), vl);
    }
};

struct FlipVlen512 : RVV_U8M8
{
    using SrcType = RVV_U8M4;
    using TabType = RVV_U16M8;
    using TabVecType = typename TabType::VecType;

    static inline void gather(const uchar* src, TabVecType tab, uchar* dst, size_t vl)
    {
        auto src_v = SrcType::vload(src, vl);
        SrcType::vstore(dst, __riscv_vrgatherei16(src_v, tab, vl), vl);
    }
};

template <typename T>
inline void flipFillBuffer(T* buf, size_t len, int esz)
{
    for (int i = (int)len - esz; i >= 0; i -= esz, buf += esz)
        for (int j = 0; j < esz; j++)
            buf[j] = (T)(i + j);
}

template <typename FlipVlen,
          typename SrcType = typename FlipVlen::SrcType,
          typename TabType = typename FlipVlen::TabType>
inline void flipY(int esz,
                  const uchar* src_data,
                  size_t src_step,
                  int src_width,
                  int src_height,
                  uchar* dst_data,
                  size_t dst_step)
{
    size_t w = (size_t)src_width * esz;
    size_t vl = std::min(SrcType::setvlmax() / esz * esz, w);
    typename TabType::VecType tab_v;
    if (esz == 1)
        tab_v = __riscv_vrsub(TabType::vid(vl), vl - 1, vl);
    else
    {
        // max vlen supported is 1024 (vlmax of u8m4 for vlen 1024 is 512)
        typename TabType::ElemType buf[512];
        flipFillBuffer(buf, vl, esz);
        tab_v = TabType::vload(buf, vl);
    }
    if (vl == w)
        for (; src_height; src_height--, src_data += src_step, dst_data += dst_step)
            FlipVlen::gather(src_data, tab_v, dst_data, vl);
    else
        for (; src_height; src_height--, src_data += src_step, dst_data += dst_step)
        {
            auto src0 = src_data, src1 = src_data + w - vl;
            auto dst0 = dst_data, dst1 = dst_data + w - vl;
            for (; src0 < src1 + vl; src0 += vl, src1 -= vl, dst0 += vl, dst1 -= vl)
            {
                FlipVlen::gather(src0, tab_v, dst1, vl);
                FlipVlen::gather(src1, tab_v, dst0, vl);
            }
        }
}

template <typename FlipVlen,
          typename SrcType = typename FlipVlen::SrcType,
          typename TabType = typename FlipVlen::TabType>
inline void flipXY(int esz,
                   const uchar* src_data,
                   size_t src_step,
                   int src_width,
                   int src_height,
                   uchar* dst_data,
                   size_t dst_step)
{
    size_t w = (size_t)src_width * esz;
    size_t vl = std::min(SrcType::setvlmax() / esz * esz, w);
    typename TabType::VecType tab_v;
    if (esz == 1)
        tab_v = __riscv_vrsub(TabType::vid(vl), vl - 1, vl);
    else
    {
        // max vlen supported is 1024 (vlmax of u8m4 for vlen 1024 is 512)
        typename TabType::ElemType buf[512];
        flipFillBuffer(buf, vl, esz);
        tab_v = TabType::vload(buf, vl);
    }
    auto src0 = src_data, src1 = src_data + src_step * (src_height - 1);
    auto dst0 = dst_data, dst1 = dst_data + dst_step * (src_height - 1);
    if (vl == w)
    {
        for (src_height -= 2; src_height >= 0;
             src_height -= 2,
             src0 += src_step,
             dst0 += dst_step,
             src1 -= src_step,
             dst1 -= dst_step)
        {
            FlipVlen::gather(src0, tab_v, dst1, vl);
            FlipVlen::gather(src1, tab_v, dst0, vl);
        }
        if (src_height == -1)
        {
            FlipVlen::gather(src1, tab_v, dst0, vl);
        }
    }
    else
    {
        for (src_height -= 2; src_height >= 0;
             src_height -= 2,
             src0 += src_step,
             dst0 += dst_step,
             src1 -= src_step,
             dst1 -= dst_step)
        {
            for (size_t i = 0; 2 * i < w; i += vl)
            {
                FlipVlen::gather(src0 + i, tab_v, dst1 + w - i - vl, vl);
                FlipVlen::gather(src0 + w - i - vl, tab_v, dst1 + i, vl);
                FlipVlen::gather(src1 + i, tab_v, dst0 + w - i - vl, vl);
                FlipVlen::gather(src1 + w - i - vl, tab_v, dst0 + i, vl);
            }
        }
        if (src_height == -1)
        {
            for (size_t i = 0; 2 * i < w; i += vl)
            {
                FlipVlen::gather(src1 + i, tab_v, dst0 + w - i - vl, vl);
                FlipVlen::gather(src1 + w - i - vl, tab_v, dst0 + i, vl);
            }
        }
    }
}

} // namespace anonymous

inline int flip(int src_type, const uchar* src_data, size_t src_step, int src_width, int src_height,
                uchar* dst_data, size_t dst_step, int flip_mode)
{
    int esz = CV_ELEM_SIZE(src_type);
    if (src_width < 0 || src_height < 0 || src_data == dst_data || esz > 32)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    if (flip_mode == 0)
    {
        for (int h = 0; h < src_height; h++) {
            const uchar* src_row = src_data + src_step * h;
            uchar* dst_row = dst_data + dst_step * (src_height - h - 1);
            std::memcpy(dst_row, src_row, esz * src_width);
        }
        return CV_HAL_ERROR_OK;
    }

    using FlipFunc = void (*)(const uchar*, size_t, uchar*, size_t, int, int, int);
    static FlipFunc flip_func_tab[] = {
        0, flip_8UC1, flip_16UC1, flip_8UC3,
        flip_32UC1, 0, flip_16UC3, 0,
        flip_64UC1, 0, 0, 0,
        flip_32UC3, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        flip_64UC3, 0, 0, 0,
        0, 0, 0, 0,
        0
    };
    FlipFunc func = flip_func_tab[esz];
    if (func) {
        func(src_data, src_step, dst_data, dst_step, src_width, src_height, flip_mode);
        return CV_HAL_ERROR_OK;
    }

    if (flip_mode > 0)
    {
        if (__riscv_vlenb() * 8 <= 256)
            flipY<FlipVlen256>(esz, src_data, src_step, src_width, src_height, dst_data, dst_step);
        else
            flipY<FlipVlen512>(esz, src_data, src_step, src_width, src_height, dst_data, dst_step);
    }
    else
    {
        if (__riscv_vlenb() * 8 <= 256)
            flipXY<FlipVlen256>(esz, src_data, src_step, src_width, src_height, dst_data, dst_step);
        else
            flipXY<FlipVlen512>(esz, src_data, src_step, src_width, src_height, dst_data, dst_step);
    }

    return CV_HAL_ERROR_OK;
}

}}  // namespace cv::cv_hal_rvv

#endif //OPENCV_HAL_RVV_FLIP_HPP_INCLUDED
