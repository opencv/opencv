// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#pragma once

#include <riscv_vector.h>
#include <opencv2/core/base.hpp>
#include "hal_rvv_1p0/types.hpp"

namespace cv { namespace cv_hal_rvv {

#undef cv_hal_flip
#define cv_hal_flip cv::cv_hal_rvv::flip

struct FlipVlen256
{
    using SrcType = RVVU8M8;
    using TabType = RVVU8M8;
    using TabVecType = typename TabType::VecType;

    static inline void gather(const uchar* src, TabVecType tab, uchar* dst, size_t vl)
    {
        auto src_v = SrcType::vload(src, vl);
        SrcType::vstore(dst, __riscv_vrgather(src_v, tab, vl), vl);
    }
};

struct FlipVlen512 : RVVU8M8
{
    using SrcType = RVVU8M4;
    using TabType = RVVU16M8;
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

inline void flipX(int esz,
                  const uchar* src_data,
                  size_t src_step,
                  int src_width,
                  int src_height,
                  uchar* dst_data,
                  size_t dst_step)
{
    size_t w = (size_t)src_width * esz;
    auto src0 = src_data, src1 = src_data + src_step * (src_height - 1);
    auto dst0 = dst_data, dst1 = dst_data + dst_step * (src_height - 1);
    size_t vl;
    for (src_height -= 2; src_height >= 0;
         src_height -= 2, src0 += src_step, dst0 += dst_step, src1 -= src_step, dst1 -= dst_step)
    {
        for (size_t i = 0; i < w; i += vl)
        {
            vl = __riscv_vsetvl_e8m8(w - i);
            __riscv_vse8(dst1 + i, __riscv_vle8_v_u8m8(src0 + i, vl), vl);
            __riscv_vse8(dst0 + i, __riscv_vle8_v_u8m8(src1 + i, vl), vl);
        }
    }
    if (src_height == -1)
    {
        for (size_t i = 0; i < w; i += (int)vl)
        {
            vl = __riscv_vsetvl_e8m8(w - i);
            __riscv_vse8(dst0 + i, __riscv_vle8_v_u8m8(src1 + i, vl), vl);
        }
    }
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
        flipX(esz, src_data, src_step, src_width, src_height, dst_data, dst_step);
    }
    else if (flip_mode > 0)
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
