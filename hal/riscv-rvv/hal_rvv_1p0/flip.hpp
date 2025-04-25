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

inline void flipX(const uchar* src_data, size_t src_step, uchar* dst_data, size_t dst_step,
                  int src_width, int src_height, int esz) {
    for (int h = 0; h < src_height; h++) {
        const uchar* src_row = src_data + src_step * h;
        uchar* dst_row = dst_data + dst_step * (src_height - h - 1);
        std::memcpy(dst_row, src_row, esz * src_width);
    }
}

#define CV_HAL_RVV_FLIPY_C1(name, _Tps, RVV) \
inline void flipY_##name(const uchar* src_data, size_t src_step, uchar* dst_data, size_t dst_step, int src_width, int src_height) { \
    for (int h = 0; h < src_height; h++) { \
        const _Tps* src_row = (const _Tps*)(src_data + src_step * h); \
        _Tps* dst_row = (_Tps*)(dst_data + dst_step * (h + 1)); \
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

struct RVV_C3_U8M2 : RVV_U8M2 {
    static inline vuint8m2x3_t vload3(const uint8_t *base, size_t vl) { return __riscv_vlseg3e8_v_u8m2x3(base, vl); }
    static inline vuint8m2x3_t vflip3(const vuint8m2x3_t &v_tuple, const vuint8m2_t &indices, size_t vl) {
        return __riscv_vcreate_v_u8m2x3(__riscv_vrgather(__riscv_vget_u8m2(v_tuple, 0), indices, vl),
                                        __riscv_vrgather(__riscv_vget_u8m2(v_tuple, 1), indices, vl),
                                        __riscv_vrgather(__riscv_vget_u8m2(v_tuple, 2), indices, vl));
    }
    static inline void vstore3(uint8_t *base, const vuint8m2x3_t &v_tuple, size_t vl) { __riscv_vsseg3e8(base, v_tuple, vl); }
};
struct RVV_C3_U16M2 : RVV_U16M2 {
    static inline vuint16m2x3_t vload3(const uint16_t *base, size_t vl) { return __riscv_vlseg3e16_v_u16m2x3(base, vl); }
    static inline vuint16m2x3_t vflip3(const vuint16m2x3_t &v_tuple, const vuint16m2_t &indices, size_t vl) {
        return __riscv_vcreate_v_u16m2x3(__riscv_vrgather(__riscv_vget_u16m2(v_tuple, 0), indices, vl),
                                         __riscv_vrgather(__riscv_vget_u16m2(v_tuple, 1), indices, vl),
                                         __riscv_vrgather(__riscv_vget_u16m2(v_tuple, 2), indices, vl));
    }
    static inline void vstore3(uint16_t *base, const vuint16m2x3_t &v_tuple, size_t vl) { __riscv_vsseg3e16(base, v_tuple, vl); }
};
struct RVV_C3_U32M2 : RVV_U32M2 {
    static inline vuint32m2x3_t vload3(const uint32_t *base, size_t vl) { return __riscv_vlseg3e32_v_u32m2x3(base, vl); }
    static inline vuint32m2x3_t vflip3(const vuint32m2x3_t &v_tuple, const vuint32m2_t &indices, size_t vl) {
        return __riscv_vcreate_v_u32m2x3(__riscv_vrgather(__riscv_vget_u32m2(v_tuple, 0), indices, vl),
                                         __riscv_vrgather(__riscv_vget_u32m2(v_tuple, 1), indices, vl),
                                         __riscv_vrgather(__riscv_vget_u32m2(v_tuple, 2), indices, vl));
    }
    static inline void vstore3(uint32_t *base, const vuint32m2x3_t &v_tuple, size_t vl) { __riscv_vsseg3e32(base, v_tuple, vl); }
};
struct RVV_C3_U64M2 : RVV_U64M2 {
    static inline vuint64m2x3_t vload3(const uint64_t *base, size_t vl) { return __riscv_vlseg3e64_v_u64m2x3(base, vl); }
    static inline vuint64m2x3_t vflip3(const vuint64m2x3_t &v_tuple, const vuint64m2_t &indices, size_t vl) {
        return __riscv_vcreate_v_u64m2x3(__riscv_vrgather(__riscv_vget_u64m2(v_tuple, 0), indices, vl),
                                         __riscv_vrgather(__riscv_vget_u64m2(v_tuple, 1), indices, vl),
                                         __riscv_vrgather(__riscv_vget_u64m2(v_tuple, 2), indices, vl));
    }
    static inline void vstore3(uint64_t *base, const vuint64m2x3_t &v_tuple, size_t vl) { __riscv_vsseg3e64(base, v_tuple, vl); }
};
#define CV_HAL_RVV_FLIPY_C3(name, _Tps, RVV) \
inline void flipY_##name(const uchar* src_data, size_t src_step, uchar* dst_data, size_t dst_step, int src_width, int src_height) { \
    int vlmax = RVV::setvlmax(); \
    for (int h = 0; h < src_height; h++) { \
        const _Tps* src_row = (const _Tps*)(src_data + src_step * h); \
        _Tps* dst_row = (_Tps*)(dst_data + dst_step * (h + 1)); \
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

inline int flip(int src_type,
                const uchar* src_data,
                size_t src_step,
                int src_width,
                int src_height,
                uchar* dst_data,
                size_t dst_step,
                int flip_mode)
{
    if (src_width < 0 || src_height < 0 || src_data == dst_data)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    int esz = CV_ELEM_SIZE(src_type);
    if (flip_mode == 0)
    {
        flipX(src_data, src_step, dst_data, dst_step, src_width, src_height, esz);
        return CV_HAL_ERROR_OK;
    }

    if (flip_mode > 0) {
        switch (esz) {
            case 1: {
                flipY_8UC1(src_data, src_step, dst_data, dst_step, src_width, src_height);
                return CV_HAL_ERROR_OK;
            }
            case 2: {
                flipY_16UC1(src_data, src_step, dst_data, dst_step, src_width, src_height);
                return CV_HAL_ERROR_OK;
            }
            case 3: {
                flipY_8UC3(src_data, src_step, dst_data, dst_step, src_width, src_height);
                return CV_HAL_ERROR_OK;
            }
            case 4: {
                flipY_32UC1(src_data, src_step, dst_data, dst_step, src_width, src_height);
                return CV_HAL_ERROR_OK;
            }
            case 6: {
                flipY_16UC3(src_data, src_step, dst_data, dst_step, src_width, src_height);
                return CV_HAL_ERROR_OK;
            }
            case 8: {
                flipY_64UC1(src_data, src_step, dst_data, dst_step, src_width, src_height);
                return CV_HAL_ERROR_OK;
            }
            case 12: {
                flipY_32UC3(src_data, src_step, dst_data, dst_step, src_width, src_height);
                return CV_HAL_ERROR_OK;
            }
            case 24: {
                flipY_64UC3(src_data, src_step, dst_data, dst_step, src_width, src_height);
                return CV_HAL_ERROR_OK;
            }
            // no default
        }
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
