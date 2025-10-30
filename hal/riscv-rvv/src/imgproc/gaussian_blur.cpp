// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2025, Institute of Software, Chinese Academy of Sciences.

#include "rvv_hal.hpp"
#include "common.hpp"

namespace cv { namespace rvv_hal { namespace imgproc {

#if CV_HAL_RVV_1P0_ENABLED

namespace {

// the algorithm is same as cv_hal_sepFilter
template<int ksize, typename helperT, typename helperWT>
static inline int gaussianBlurC1(int start, int end, const uchar* src_data, size_t src_step, uchar* dst_data, size_t dst_step, int width, int full_width, int full_height, int offset_x, int offset_y, int border_type)
{
    using T = typename helperT::ElemType;
    using WT = typename helperWT::ElemType;

    constexpr int noval = std::numeric_limits<int>::max();
    auto accessX = [&](int x) {
        int pi = common::borderInterpolate(offset_y + x - ksize / 2, full_height, border_type); // [TODO] fix dependencies
        return pi < 0 ? noval : pi - offset_y;
    };
    auto accessY = [&](int y) {
        int pj = common::borderInterpolate(offset_x + y - ksize / 2, full_width, border_type);
        return pj < 0 ? noval : pj - offset_x;
    };
    auto p2idx = [&](int x, int y){ return (x + ksize) % ksize * width + y; };

    constexpr uint kernel[2][5] = {{1, 2, 1}, {1, 4, 6, 4, 1}};
    std::vector<WT> res(width * ksize);
    auto process = [&](int x, int y) {
        WT sum = 0;
        for (int i = 0; i < ksize; i++)
        {
            int p = accessY(y + i);
            if (p != noval)
            {
                sum += kernel[ksize == 5][i] * static_cast<WT>(reinterpret_cast<const T*>(src_data + x * src_step)[p]);
            }
        }
        res[p2idx(x, y)] = sum;
    };

    const int left = ksize / 2, right = width - ksize / 2;
    for (int i = start - ksize / 2; i < end + ksize / 2; i++)
    {
        if (i + offset_y >= 0 && i + offset_y < full_height)
        {
            if (left >= right)
            {
                for (int j = 0; j < width; j++)
                    process(i, j);
            }
            else
            {
                for (int j = 0; j < left; j++)
                    process(i, j);
                for (int j = right; j < width; j++)
                    process(i, j);

                int vl;
                for (int j = left; j < right; j += vl)
                {
                    vl = helperT::setvl(right - j);
                    const T* extra = reinterpret_cast<const T*>(src_data + i * src_step) + j - ksize / 2;
                    auto src = __riscv_vzext_vf2(helperT::vload(extra, vl), vl);

                    extra += vl;
                    auto sum = src;
                    if (ksize == 3)
                    {
                        src = __riscv_vslide1down(src, extra[0], vl);
                        sum = __riscv_vadd(sum, __riscv_vsll(src, 1, vl), vl);
                        src = __riscv_vslide1down(src, extra[1], vl);
                        sum = __riscv_vadd(sum, src, vl);
                    }
                    else
                    {
                        src = __riscv_vslide1down(src, extra[0], vl);
                        sum = __riscv_vadd(sum, __riscv_vsll(src, 2, vl), vl);
                        src = __riscv_vslide1down(src, extra[1], vl);
                        sum = __riscv_vadd(sum, __riscv_vadd(__riscv_vsll(src, 1, vl), __riscv_vsll(src, 2, vl), vl), vl);
                        src = __riscv_vslide1down(src, extra[2], vl);
                        sum = __riscv_vadd(sum, __riscv_vsll(src, 2, vl), vl);
                        src = __riscv_vslide1down(src, extra[3], vl);
                        sum = __riscv_vadd(sum, src, vl);
                    }
                    helperWT::vstore(res.data() + p2idx(i, j), sum, vl);
                }
            }
        }

        int cur = i - ksize / 2;
        if (cur >= start)
        {
            const WT* row0 = accessX(cur    ) == noval ? nullptr : res.data() + p2idx(accessX(cur    ), 0);
            const WT* row1 = accessX(cur + 1) == noval ? nullptr : res.data() + p2idx(accessX(cur + 1), 0);
            const WT* row2 = accessX(cur + 2) == noval ? nullptr : res.data() + p2idx(accessX(cur + 2), 0);
            const WT* row3 = nullptr, *row4 = nullptr;
            if (ksize == 5)
            {
                row3 = accessX(cur + 3) == noval ? nullptr : res.data() + p2idx(accessX(cur + 3), 0);
                row4 = accessX(cur + 4) == noval ? nullptr : res.data() + p2idx(accessX(cur + 4), 0);
            }

            int vl;
            for (int j = 0; j < width; j += vl)
            {
                vl = helperWT::setvl(width - j);
                auto v0 = row0 ? helperWT::vload(row0 + j, vl) : helperWT::vmv(0, vl);
                auto v1 = row1 ? helperWT::vload(row1 + j, vl) : helperWT::vmv(0, vl);
                auto v2 = row2 ? helperWT::vload(row2 + j, vl) : helperWT::vmv(0, vl);
                typename helperWT::VecType sum;
                if (ksize == 3)
                {
                    sum = __riscv_vadd(__riscv_vadd(v0, v2, vl), __riscv_vsll(v1, 1, vl), vl);
                }
                else
                {
                    sum = __riscv_vadd(v0, __riscv_vadd(__riscv_vsll(v2, 1, vl), __riscv_vsll(v2, 2, vl), vl), vl);
                    auto v3 = row3 ? helperWT::vload(row3 + j, vl) : helperWT::vmv(0, vl);
                    sum = __riscv_vadd(sum, __riscv_vsll(__riscv_vadd(v1, v3, vl), 2, vl), vl);
                    auto v4 = row4 ? helperWT::vload(row4 + j, vl) : helperWT::vmv(0, vl);
                    sum = __riscv_vadd(sum, v4, vl);
                }
                helperT::vstore(reinterpret_cast<T*>(dst_data + cur * dst_step) + j, __riscv_vnclipu(sum, ksize == 5 ? 8 : 4, __RISCV_VXRM_RNU, vl), vl);
            }
        }
    }

    return CV_HAL_ERROR_OK;
}

template<int ksize>
static inline int gaussianBlurC4(int start, int end, const uchar* src_data, size_t src_step, uchar* dst_data, size_t dst_step, int width, int full_width, int full_height, int offset_x, int offset_y, int border_type)
{
    constexpr int noval = std::numeric_limits<int>::max();
    auto accessX = [&](int x) {
        int pi = common::borderInterpolate(offset_y + x - ksize / 2, full_height, border_type);
        return pi < 0 ? noval : pi - offset_y;
    };
    auto accessY = [&](int y) {
        int pj = common::borderInterpolate(offset_x + y - ksize / 2, full_width, border_type);
        return pj < 0 ? noval : pj - offset_x;
    };
    auto p2idx = [&](int x, int y){ return ((x + ksize) % ksize * width + y) * 4; };

    constexpr uint kernel[2][5] = {{1, 2, 1}, {1, 4, 6, 4, 1}};
    std::vector<ushort> res(width * ksize * 4);
    auto process = [&](int x, int y) {
        ushort sum0, sum1, sum2, sum3;
        sum0 = sum1 = sum2 = sum3 = 0;
        for (int i = 0; i < ksize; i++)
        {
            int p = accessY(y + i);
            if (p != noval)
            {
                sum0 += kernel[ksize == 5][i] * static_cast<ushort>((src_data + x * src_step)[p * 4    ]);
                sum1 += kernel[ksize == 5][i] * static_cast<ushort>((src_data + x * src_step)[p * 4 + 1]);
                sum2 += kernel[ksize == 5][i] * static_cast<ushort>((src_data + x * src_step)[p * 4 + 2]);
                sum3 += kernel[ksize == 5][i] * static_cast<ushort>((src_data + x * src_step)[p * 4 + 3]);
            }
        }
        res[p2idx(x, y)    ] = sum0;
        res[p2idx(x, y) + 1] = sum1;
        res[p2idx(x, y) + 2] = sum2;
        res[p2idx(x, y) + 3] = sum3;
    };

    const int left = ksize / 2, right = width - ksize / 2;
    for (int i = start - ksize / 2; i < end + ksize / 2; i++)
    {
        if (i + offset_y >= 0 && i + offset_y < full_height)
        {
            if (left >= right)
            {
                for (int j = 0; j < width; j++)
                    process(i, j);
            }
            else
            {
                for (int j = 0; j < left; j++)
                    process(i, j);
                for (int j = right; j < width; j++)
                    process(i, j);

                int vl;
                for (int j = left; j < right; j += vl)
                {
                    vl = __riscv_vsetvl_e8m1(right - j);
                    const uchar* extra = src_data + i * src_step + (j - ksize / 2) * 4;
                    auto src = __riscv_vlseg4e8_v_u8m1x4(extra, vl);
                    auto src0 = __riscv_vzext_vf2(__riscv_vget_v_u8m1x4_u8m1(src, 0), vl);
                    auto src1 = __riscv_vzext_vf2(__riscv_vget_v_u8m1x4_u8m1(src, 1), vl);
                    auto src2 = __riscv_vzext_vf2(__riscv_vget_v_u8m1x4_u8m1(src, 2), vl);
                    auto src3 = __riscv_vzext_vf2(__riscv_vget_v_u8m1x4_u8m1(src, 3), vl);

                    extra += vl * 4;
                    auto sum0 = src0, sum1 = src1, sum2 = src2, sum3 = src3;
                    if (ksize == 3)
                    {
                        src0 = __riscv_vslide1down(src0, extra[0], vl);
                        src1 = __riscv_vslide1down(src1, extra[1], vl);
                        src2 = __riscv_vslide1down(src2, extra[2], vl);
                        src3 = __riscv_vslide1down(src3, extra[3], vl);
                        sum0 = __riscv_vadd(sum0, __riscv_vsll(src0, 1, vl), vl);
                        sum1 = __riscv_vadd(sum1, __riscv_vsll(src1, 1, vl), vl);
                        sum2 = __riscv_vadd(sum2, __riscv_vsll(src2, 1, vl), vl);
                        sum3 = __riscv_vadd(sum3, __riscv_vsll(src3, 1, vl), vl);
                        src0 = __riscv_vslide1down(src0, extra[4], vl);
                        src1 = __riscv_vslide1down(src1, extra[5], vl);
                        src2 = __riscv_vslide1down(src2, extra[6], vl);
                        src3 = __riscv_vslide1down(src3, extra[7], vl);
                        sum0 = __riscv_vadd(sum0, src0, vl);
                        sum1 = __riscv_vadd(sum1, src1, vl);
                        sum2 = __riscv_vadd(sum2, src2, vl);
                        sum3 = __riscv_vadd(sum3, src3, vl);
                    }
                    else
                    {
                        src0 = __riscv_vslide1down(src0, extra[0], vl);
                        src1 = __riscv_vslide1down(src1, extra[1], vl);
                        src2 = __riscv_vslide1down(src2, extra[2], vl);
                        src3 = __riscv_vslide1down(src3, extra[3], vl);
                        sum0 = __riscv_vadd(sum0, __riscv_vsll(src0, 2, vl), vl);
                        sum1 = __riscv_vadd(sum1, __riscv_vsll(src1, 2, vl), vl);
                        sum2 = __riscv_vadd(sum2, __riscv_vsll(src2, 2, vl), vl);
                        sum3 = __riscv_vadd(sum3, __riscv_vsll(src3, 2, vl), vl);
                        src0 = __riscv_vslide1down(src0, extra[4], vl);
                        src1 = __riscv_vslide1down(src1, extra[5], vl);
                        src2 = __riscv_vslide1down(src2, extra[6], vl);
                        src3 = __riscv_vslide1down(src3, extra[7], vl);
                        sum0 = __riscv_vadd(sum0, __riscv_vadd(__riscv_vsll(src0, 1, vl), __riscv_vsll(src0, 2, vl), vl), vl);
                        sum1 = __riscv_vadd(sum1, __riscv_vadd(__riscv_vsll(src1, 1, vl), __riscv_vsll(src1, 2, vl), vl), vl);
                        sum2 = __riscv_vadd(sum2, __riscv_vadd(__riscv_vsll(src2, 1, vl), __riscv_vsll(src2, 2, vl), vl), vl);
                        sum3 = __riscv_vadd(sum3, __riscv_vadd(__riscv_vsll(src3, 1, vl), __riscv_vsll(src3, 2, vl), vl), vl);
                        src0 = __riscv_vslide1down(src0, extra[ 8], vl);
                        src1 = __riscv_vslide1down(src1, extra[ 9], vl);
                        src2 = __riscv_vslide1down(src2, extra[10], vl);
                        src3 = __riscv_vslide1down(src3, extra[11], vl);
                        sum0 = __riscv_vadd(sum0, __riscv_vsll(src0, 2, vl), vl);
                        sum1 = __riscv_vadd(sum1, __riscv_vsll(src1, 2, vl), vl);
                        sum2 = __riscv_vadd(sum2, __riscv_vsll(src2, 2, vl), vl);
                        sum3 = __riscv_vadd(sum3, __riscv_vsll(src3, 2, vl), vl);
                        src0 = __riscv_vslide1down(src0, extra[12], vl);
                        src1 = __riscv_vslide1down(src1, extra[13], vl);
                        src2 = __riscv_vslide1down(src2, extra[14], vl);
                        src3 = __riscv_vslide1down(src3, extra[15], vl);
                        sum0 = __riscv_vadd(sum0, src0, vl);
                        sum1 = __riscv_vadd(sum1, src1, vl);
                        sum2 = __riscv_vadd(sum2, src2, vl);
                        sum3 = __riscv_vadd(sum3, src3, vl);
                    }

                    vuint16m2x4_t dst{};
                    dst = __riscv_vset_v_u16m2_u16m2x4(dst, 0, sum0);
                    dst = __riscv_vset_v_u16m2_u16m2x4(dst, 1, sum1);
                    dst = __riscv_vset_v_u16m2_u16m2x4(dst, 2, sum2);
                    dst = __riscv_vset_v_u16m2_u16m2x4(dst, 3, sum3);
                    __riscv_vsseg4e16(res.data() + p2idx(i, j), dst, vl);
                }
            }
        }

        int cur = i - ksize / 2;
        if (cur >= start)
        {
            const ushort* row0 = accessX(cur    ) == noval ? nullptr : res.data() + p2idx(accessX(cur    ), 0);
            const ushort* row1 = accessX(cur + 1) == noval ? nullptr : res.data() + p2idx(accessX(cur + 1), 0);
            const ushort* row2 = accessX(cur + 2) == noval ? nullptr : res.data() + p2idx(accessX(cur + 2), 0);
            const ushort* row3 = nullptr, *row4 = nullptr;
            if (ksize == 5)
            {
                row3 = accessX(cur + 3) == noval ? nullptr : res.data() + p2idx(accessX(cur + 3), 0);
                row4 = accessX(cur + 4) == noval ? nullptr : res.data() + p2idx(accessX(cur + 4), 0);
            }

            int vl;
            for (int j = 0; j < width; j += vl)
            {
                vl = __riscv_vsetvl_e16m2(width - j);
                vuint16m2_t sum0, sum1, sum2, sum3, src0{}, src1{}, src2{}, src3{};
                sum0 = sum1 = sum2 = sum3 = __riscv_vmv_v_x_u16m2(0, vl);

                auto loadres = [&](const ushort* row) {
                    auto src = __riscv_vlseg4e16_v_u16m2x4(row + j * 4, vl);
                    src0 = __riscv_vget_v_u16m2x4_u16m2(src, 0);
                    src1 = __riscv_vget_v_u16m2x4_u16m2(src, 1);
                    src2 = __riscv_vget_v_u16m2x4_u16m2(src, 2);
                    src3 = __riscv_vget_v_u16m2x4_u16m2(src, 3);
                };
                if (row0)
                {
                    loadres(row0);
                    sum0 = src0;
                    sum1 = src1;
                    sum2 = src2;
                    sum3 = src3;
                }
                if (row1)
                {
                    loadres(row1);
                    sum0 = __riscv_vadd(sum0, __riscv_vsll(src0, ksize == 5 ? 2 : 1, vl), vl);
                    sum1 = __riscv_vadd(sum1, __riscv_vsll(src1, ksize == 5 ? 2 : 1, vl), vl);
                    sum2 = __riscv_vadd(sum2, __riscv_vsll(src2, ksize == 5 ? 2 : 1, vl), vl);
                    sum3 = __riscv_vadd(sum3, __riscv_vsll(src3, ksize == 5 ? 2 : 1, vl), vl);
                }
                if (row2)
                {
                    loadres(row2);
                    if (ksize == 5)
                    {
                        src0 = __riscv_vadd(__riscv_vsll(src0, 1, vl), __riscv_vsll(src0, 2, vl), vl);
                        src1 = __riscv_vadd(__riscv_vsll(src1, 1, vl), __riscv_vsll(src1, 2, vl), vl);
                        src2 = __riscv_vadd(__riscv_vsll(src2, 1, vl), __riscv_vsll(src2, 2, vl), vl);
                        src3 = __riscv_vadd(__riscv_vsll(src3, 1, vl), __riscv_vsll(src3, 2, vl), vl);
                    }
                    sum0 = __riscv_vadd(sum0, src0, vl);
                    sum1 = __riscv_vadd(sum1, src1, vl);
                    sum2 = __riscv_vadd(sum2, src2, vl);
                    sum3 = __riscv_vadd(sum3, src3, vl);
                }
                if (row3)
                {
                    loadres(row3);
                    sum0 = __riscv_vadd(sum0, __riscv_vsll(src0, 2, vl), vl);
                    sum1 = __riscv_vadd(sum1, __riscv_vsll(src1, 2, vl), vl);
                    sum2 = __riscv_vadd(sum2, __riscv_vsll(src2, 2, vl), vl);
                    sum3 = __riscv_vadd(sum3, __riscv_vsll(src3, 2, vl), vl);
                }
                if (row4)
                {
                    loadres(row4);
                    sum0 = __riscv_vadd(sum0, src0, vl);
                    sum1 = __riscv_vadd(sum1, src1, vl);
                    sum2 = __riscv_vadd(sum2, src2, vl);
                    sum3 = __riscv_vadd(sum3, src3, vl);
                }

                vuint8m1x4_t dst{};
                dst = __riscv_vset_v_u8m1_u8m1x4(dst, 0, __riscv_vnclipu(sum0, ksize == 5 ? 8 : 4, __RISCV_VXRM_RNU, vl));
                dst = __riscv_vset_v_u8m1_u8m1x4(dst, 1, __riscv_vnclipu(sum1, ksize == 5 ? 8 : 4, __RISCV_VXRM_RNU, vl));
                dst = __riscv_vset_v_u8m1_u8m1x4(dst, 2, __riscv_vnclipu(sum2, ksize == 5 ? 8 : 4, __RISCV_VXRM_RNU, vl));
                dst = __riscv_vset_v_u8m1_u8m1x4(dst, 3, __riscv_vnclipu(sum3, ksize == 5 ? 8 : 4, __RISCV_VXRM_RNU, vl));
                __riscv_vsseg4e8(dst_data + cur * dst_step + j * 4, dst, vl);
            }
        }
    }

    return CV_HAL_ERROR_OK;
}

} // anonymous

int gaussianBlurBinomial(const uchar* src_data, size_t src_step, uchar* dst_data, size_t dst_step, int width, int height, int depth, int cn, size_t margin_left, size_t margin_top, size_t margin_right, size_t margin_bottom, size_t ksize, int border_type)
{
    const int type = CV_MAKETYPE(depth, cn);
    if ((type != CV_8UC1 && type != CV_8UC4 && type != CV_16UC1) || src_data == dst_data)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if ((ksize != 3 && ksize != 5) || border_type & BORDER_ISOLATED || border_type == BORDER_WRAP)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    switch (ksize*100 + type)
    {
    case 300 + CV_8UC1:
        return common::invoke(height, {gaussianBlurC1<3, RVV_U8M4, RVV_U16M8>}, src_data, src_step, dst_data, dst_step, width, margin_left + width + margin_right, margin_top + height + margin_bottom, margin_left, margin_top, border_type);
    case 500 + CV_8UC1:
        return common::invoke(height, {gaussianBlurC1<5, RVV_U8M4, RVV_U16M8>}, src_data, src_step, dst_data, dst_step, width, margin_left + width + margin_right, margin_top + height + margin_bottom, margin_left, margin_top, border_type);
    case 300 + CV_16UC1:
        return common::invoke(height, {gaussianBlurC1<3, RVV_U16M4, RVV_U32M8>}, src_data, src_step, dst_data, dst_step, width, margin_left + width + margin_right, margin_top + height + margin_bottom, margin_left, margin_top, border_type);
    case 500 + CV_16UC1:
        return common::invoke(height, {gaussianBlurC1<5, RVV_U16M4, RVV_U32M8>}, src_data, src_step, dst_data, dst_step, width, margin_left + width + margin_right, margin_top + height + margin_bottom, margin_left, margin_top, border_type);
    case 300 + CV_8UC4:
        return common::invoke(height, {gaussianBlurC4<3>}, src_data, src_step, dst_data, dst_step, width, margin_left + width + margin_right, margin_top + height + margin_bottom, margin_left, margin_top, border_type);
    case 500 + CV_8UC4:
        return common::invoke(height, {gaussianBlurC4<5>}, src_data, src_step, dst_data, dst_step, width, margin_left + width + margin_right, margin_top + height + margin_bottom, margin_left, margin_top, border_type);
    }

    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

#endif // CV_HAL_RVV_1P0_ENABLED

}}} // cv::rvv_hal::imgproc
