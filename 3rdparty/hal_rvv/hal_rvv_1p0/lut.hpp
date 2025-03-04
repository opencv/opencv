// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#pragma once

#include <riscv_vector.h>
#include <opencv2/core/base.hpp>
#include <opencv2/core/utility.hpp>

namespace cv { namespace cv_hal_rvv {

#undef cv_hal_lut
#define cv_hal_lut cv::cv_hal_rvv::lut

// need vlen >= 256
struct LUTCacheU8
{
    using ElemType = uchar;
    constexpr static size_t elem_size = sizeof(ElemType);

    static inline vuint8m8_t loadTable(const ElemType* lut_data)
    {
        return __riscv_vle8_v_u8m8(lut_data, 256);
    }

    static inline size_t setvl(size_t len)
    {
        return __riscv_vsetvl_e8m8(len);
    }

    static inline size_t setvlmax()
    {
        return __riscv_vsetvlmax_e8m8();
    }

    static inline void lut(const uchar* src, vuint8m8_t lut_v, ElemType* dst, size_t vl)
    {
        auto src_v = __riscv_vle8_v_u8m8(src, vl);
        auto dst_v = __riscv_vrgather(lut_v, src_v, vl);
        __riscv_vse8(dst, dst_v, vl);
    }
};

// need vlen >= 512
struct LUTCacheU16
{
    using ElemType = uint16_t;
    constexpr static size_t elem_size = sizeof(ElemType);

    static inline vuint16m8_t loadTable(const ElemType* lut_data)
    {
        return __riscv_vle16_v_u16m8(lut_data, 256);
    }

    static inline size_t setvl(size_t len)
    {
        return __riscv_vsetvl_e16m8(len);
    }

    static inline size_t setvlmax()
    {
        return __riscv_vsetvlmax_e16m8();
    }

    static inline void lut(const uchar* src, vuint16m8_t lut_v, ElemType* dst, size_t vl)
    {
        auto src_v = __riscv_vzext_vf2(__riscv_vle8_v_u8m4(src, vl), vl);
        auto dst_v = __riscv_vrgather(lut_v, src_v, vl);
        __riscv_vse16(dst, dst_v, vl);
    }
};

// need vlen >= 1024
struct LUTCacheU32
{
    using ElemType = uint32_t;
    constexpr static size_t elem_size = sizeof(ElemType);

    static inline vuint32m8_t loadTable(const ElemType* lut_data)
    {
        return __riscv_vle32_v_u32m8(lut_data, 256);
    }

    static inline size_t setvl(size_t len)
    {
        return __riscv_vsetvl_e32m8(len);
    }

    static inline size_t setvlmax()
    {
        return __riscv_vsetvlmax_e32m8();
    }

    static inline void lut(const uchar* src, vuint32m8_t lut_v, ElemType* dst, size_t vl)
    {
        auto src_v = __riscv_vzext_vf2(__riscv_vle8_v_u8m2(src, vl), vl);
        auto dst_v = __riscv_vrgatherei16(lut_v, src_v, vl);
        __riscv_vse32(dst, dst_v, vl);
    }
};

template <typename LUT_TYPE>
struct LUTParallelBody
{
    using ElemType = typename LUT_TYPE::ElemType;

    const uchar* src_data;
    const uchar* lut_data;
    uchar* dst_data;
    size_t src_step;
    size_t dst_step;
    size_t width;

    LUTParallelBody(const uchar* src_data,
                    size_t src_step,
                    const uchar* lut_data,
                    uchar* dst_data,
                    size_t dst_step,
                    size_t width) :
        src_data(src_data), lut_data(lut_data), dst_data(dst_data), src_step(src_step),
        dst_step(dst_step), width(width)
    {
    }
};

template <typename LUT_TYPE>
void parallel_for_body_fn(int start, int end, void* data)
{
    typedef LUTParallelBody<LUT_TYPE> BodyType;
    BodyType *inst = (BodyType*)data;

    auto src = inst->src_data + start * inst->src_step;
    auto dst = inst->dst_data + start * inst->dst_step;
    size_t h = end - start;
    size_t w = inst->width;
    if (w == inst->src_step && w * LUT_TYPE::elem_size == inst->dst_step)
    {
        w = w * h;
        h = 1;
    }
    auto lut = LUT_TYPE::loadTable((typename BodyType::ElemType*)inst->lut_data);
    size_t maxlv = LUT_TYPE::setvlmax();
    for (; h; h--, src += inst->src_step, dst += inst->dst_step)
    {
        size_t vl = maxlv;
        size_t l = w;
        auto s = src;
        auto d = (typename BodyType::ElemType*)dst;
        for (; l >= vl; l -= vl, s += vl, d += vl)
        {
            LUT_TYPE::lut(s, lut, d, vl);
        }
        for (; l > 0; l -= vl, s += vl, d += vl)
        {
            vl = LUT_TYPE::setvl(l);
            LUT_TYPE::lut(s, lut, d, vl);
        }
    }
}

inline int lut(const uchar* src_data,
               size_t src_step,
               size_t src_type,
               const uchar* lut_data,
               size_t lut_channel_size,
               size_t lut_channels,
               uchar* dst_data,
               size_t dst_step,
               int width,
               int height,
               HAL_Context * ctx)
{
    if (width <= 0 || height <= 0)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    size_t w = width;
    size_t h = height;
    size_t vlen = __riscv_vsetvlmax_e8m8();
    if (lut_channels == 1)
    {
        w *= CV_MAT_CN(src_type);

        // Actually, vlen 128 can use four u8m4 vectors to load the whole table, gather and merge
        // the result, but the performance is almost the same as the scalar.
        if (lut_channel_size == 1 && vlen >= 256)
        {
            LUTParallelBody<LUTCacheU8> body(src_data, src_step, lut_data, dst_data, dst_step, w);
            if (w * h >= (1 << 18))
                ctx->parallel_for_fn(height, &parallel_for_body_fn<LUTCacheU8>, &body);
            else
                parallel_for_body_fn<LUTCacheU8>(0, height, &body);
            return CV_HAL_ERROR_OK;
        }
        else if (lut_channel_size == 2 && vlen >= 512)
        {
            LUTParallelBody<LUTCacheU16> body(src_data, src_step, lut_data, dst_data, dst_step, w);
            if (w * h >= (1 << 18))
                ctx->parallel_for_fn(height, &parallel_for_body_fn<LUTCacheU16>, &body);
            else
                parallel_for_body_fn<LUTCacheU16>(0, height, &body);
            return CV_HAL_ERROR_OK;
        }
        else if (lut_channel_size == 4 && vlen >= 1024)
        {
            LUTParallelBody<LUTCacheU32> body(src_data, src_step, lut_data, dst_data, dst_step, w);
            if (w * h >= (1 << 18))
                ctx->parallel_for_fn(height, &parallel_for_body_fn<LUTCacheU32>, &body);
            else
                parallel_for_body_fn<LUTCacheU32>(0, height, &body);
            return CV_HAL_ERROR_OK;
        }
    }
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

}}  // namespace cv::cv_hal_rvv
