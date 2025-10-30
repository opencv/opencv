// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2025, Institute of Software, Chinese Academy of Sciences.

#include "rvv_hal.hpp"

namespace cv { namespace rvv_hal { namespace core {

#if CV_HAL_RVV_1P0_ENABLED

// need vlen >= 256
struct LUTCacheU8 : RVV_U8M8
{
    using TabType = RVV_U8M8;
    using SrcType = RVV_SameLen<uint8_t, TabType>;
    using IdxType = RVV_SameLen<uint8_t, TabType>;

    using ElemType = typename TabType::ElemType;
    constexpr static size_t elem_size = sizeof(ElemType);

    static inline typename TabType::VecType
        gather(typename IdxType::VecType src_v, typename TabType::VecType lut_v, size_t vl)
    {
        return __riscv_vrgather(lut_v, src_v, vl);
    }
};

// need vlen >= 512
struct LUTCacheU16 : RVV_U16M8
{
    using TabType = RVV_U16M8;
    using SrcType = RVV_SameLen<uint8_t, TabType>;
    using IdxType = RVV_SameLen<uint16_t, TabType>;

    using ElemType = typename TabType::ElemType;
    constexpr static size_t elem_size = sizeof(ElemType);

    static inline typename TabType::VecType
        gather(typename IdxType::VecType src_v, typename TabType::VecType lut_v, size_t vl)
    {
        return __riscv_vrgather(lut_v, src_v, vl);
    }
};

// need vlen >= 1024
struct LUTCacheU32 : RVV_U32M8
{
    using TabType = RVV_U32M8;
    using SrcType = RVV_SameLen<uint8_t, TabType>;
    using IdxType = RVV_SameLen<uint16_t, TabType>;

    using ElemType = typename TabType::ElemType;
    constexpr static size_t elem_size = sizeof(ElemType);

    static inline typename TabType::VecType
        gather(typename IdxType::VecType src_v, typename TabType::VecType lut_v, size_t vl)
    {
        return __riscv_vrgatherei16(lut_v, src_v, vl);
    }
};

template <typename LUT_TYPE>
class LUTParallelBody : public cv::ParallelLoopBody
{
    using ElemType = typename LUT_TYPE::ElemType;

public:
    const uchar* src_data;
    const uchar* lut_data;
    uchar* dst_data;
    size_t src_step;
    size_t dst_step;
    size_t width;

    LUTParallelBody(const uchar* _src_data,
                    size_t _src_step,
                    const uchar* _lut_data,
                    uchar* _dst_data,
                    size_t _dst_step,
                    size_t _width) :
        src_data(_src_data), lut_data(_lut_data), dst_data(_dst_data), src_step(_src_step),
        dst_step(_dst_step), width(_width)
    {
    }

    void operator()(const cv::Range& range) const CV_OVERRIDE
    {
        auto src = src_data + range.start * src_step;
        auto dst = dst_data + range.start * dst_step;
        size_t h = range.size();
        size_t w = width;
        if (w == src_step && w * LUT_TYPE::elem_size == dst_step)
        {
            w = w * h;
            h = 1;
        }
        auto lut = LUT_TYPE::vload((ElemType*)lut_data, 256);
        size_t maxlv = LUT_TYPE::setvlmax();
        for (; h; h--, src += src_step, dst += dst_step)
        {
            size_t vl = maxlv;
            size_t l = w;
            auto s = src;
            auto d = (ElemType*)dst;
            for (; l >= vl; l -= vl, s += vl, d += vl)
            {
                auto src_v = LUT_TYPE::SrcType::vload(s, vl);
                auto idx_v = LUT_TYPE::IdxType::cast(src_v, vl);
                auto dst_v = LUT_TYPE::gather(idx_v, lut, vl);
                LUT_TYPE::vstore(d, dst_v, vl);
            }
            for (; l > 0; l -= vl, s += vl, d += vl)
            {
                vl = LUT_TYPE::setvl(l);
                auto src_v = LUT_TYPE::SrcType::vload(s, vl);
                auto idx_v = LUT_TYPE::IdxType::cast(src_v, vl);
                auto dst_v = LUT_TYPE::gather(idx_v, lut, vl);
                LUT_TYPE::vstore(d, dst_v, vl);
            }
        }
    }

private:
    LUTParallelBody(const LUTParallelBody&);
    LUTParallelBody& operator=(const LUTParallelBody&);
};

int lut(const uchar* src_data,
               size_t src_step,
               size_t src_type,
               const uchar* lut_data,
               size_t lut_channel_size,
               size_t lut_channels,
               uchar* dst_data,
               size_t dst_step,
               int width,
               int height)
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
            Range all(0, height);
            if (w * h >= (1 << 18))
                cv::parallel_for_(all, body);
            else
                body(all);
            return CV_HAL_ERROR_OK;
        }
        else if (lut_channel_size == 2 && vlen >= 512)
        {
            LUTParallelBody<LUTCacheU16> body(src_data, src_step, lut_data, dst_data, dst_step, w);
            Range all(0, height);
            if (w * h >= (1 << 18))
                cv::parallel_for_(all, body);
            else
                body(all);
            return CV_HAL_ERROR_OK;
        }
        else if (lut_channel_size == 4 && vlen >= 1024)
        {
            LUTParallelBody<LUTCacheU32> body(src_data, src_step, lut_data, dst_data, dst_step, w);
            Range all(0, height);
            if (w * h >= (1 << 18))
                cv::parallel_for_(all, body);
            else
                body(all);
            return CV_HAL_ERROR_OK;
        }
    }
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

#endif // CV_HAL_RVV_1P0_ENABLED

}}}  // cv::rvv_hal::core
