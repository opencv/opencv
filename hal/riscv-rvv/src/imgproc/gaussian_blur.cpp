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

// Channel-blind kernel for interleaved 8U data: a row is treated as width*cn
// flat u8 elements whose horizontal taps sit at strides of cn elements, so no
// segment loads or slide chains are needed; the vertical pass is a single
// contiguous u16 stream and the output is a plain store. Weighted taps are
// combined with shifts; u16 is exact (two 16x passes: 16*16*255 = 65280).
template<int ksize, int cn>
static inline int gaussianBlurFlat8U(int start, int end, const uchar* src_data, size_t src_step, uchar* dst_data, size_t dst_step, int width, int full_width, int full_height, int offset_x, int offset_y, int border_type)
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
    const int W = width * cn;
    auto p2idx = [&](int x){ return (x + ksize) % ksize * W; };

    constexpr uint kernel[2][5] = {{1, 2, 1}, {1, 4, 6, 4, 1}};
    std::vector<ushort> res(W * ksize);
    auto process = [&](int x, int y) {
        ushort sum[cn];
        for (int c = 0; c < cn; c++)
            sum[c] = 0;
        for (int i = 0; i < ksize; i++)
        {
            int p = accessY(y + i);
            if (p != noval)
            {
                for (int c = 0; c < cn; c++)
                    sum[c] += kernel[ksize == 5][i] * static_cast<ushort>((src_data + x * src_step)[p * cn + c]);
            }
        }
        for (int c = 0; c < cn; c++)
            res[p2idx(x) + y * cn + c] = sum[c];
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

                ushort* row = res.data() + p2idx(i);
                int vl;
                for (int fj = left * cn; fj < right * cn; fj += vl)
                {
                    vl = __riscv_vsetvl_e8m4(right * cn - fj);
                    const uchar* p = src_data + i * src_step + fj - (ksize / 2) * cn;
                    vuint16m8_t acc;
                    if (ksize == 3)
                    {
                        // 1 2 1
                        acc = __riscv_vwaddu_vv(__riscv_vle8_v_u8m4(p, vl), __riscv_vle8_v_u8m4(p + 2 * cn, vl), vl);
                        auto mid = __riscv_vle8_v_u8m4(p + cn, vl);
                        acc = __riscv_vwaddu_wv(acc, mid, vl);
                        acc = __riscv_vwaddu_wv(acc, mid, vl);
                    }
                    else
                    {
                        // 1 4 6 4 1
                        acc = __riscv_vwaddu_vv(__riscv_vle8_v_u8m4(p, vl), __riscv_vle8_v_u8m4(p + 4 * cn, vl), vl);
                        auto t13 = __riscv_vwaddu_vv(__riscv_vle8_v_u8m4(p + cn, vl), __riscv_vle8_v_u8m4(p + 3 * cn, vl), vl);
                        acc = __riscv_vadd(acc, __riscv_vsll(t13, 2, vl), vl);
                        auto mid = __riscv_vzext_vf2(__riscv_vle8_v_u8m4(p + 2 * cn, vl), vl);
                        acc = __riscv_vadd(acc, __riscv_vadd(__riscv_vsll(mid, 2, vl), __riscv_vsll(mid, 1, vl), vl), vl);
                    }
                    __riscv_vse16(row + fj, acc, vl);
                }
            }
        }

        int cur = i - ksize / 2;
        if (cur >= start)
        {
            const ushort* rows[ksize];
            for (int k = 0; k < ksize; k++)
                rows[k] = accessX(cur + k) == noval ? nullptr : res.data() + p2idx(accessX(cur + k));

            int vl;
            for (int fj = 0; fj < W; fj += vl)
            {
                vl = __riscv_vsetvl_e16m8(W - fj);
                auto vzero = __riscv_vmv_v_x_u16m8(0, vl);
                auto load = [&](int k) { return rows[k] ? __riscv_vle16_v_u16m8(rows[k] + fj, vl) : vzero; };
                vuint16m8_t sum;
                if (ksize == 3)
                {
                    // 1 2 1
                    sum = __riscv_vadd(__riscv_vadd(load(0), load(2), vl), __riscv_vsll(load(1), 1, vl), vl);
                }
                else
                {
                    // 1 4 6 4 1
                    sum = __riscv_vadd(load(0), load(4), vl);
                    sum = __riscv_vadd(sum, __riscv_vsll(__riscv_vadd(load(1), load(3), vl), 2, vl), vl);
                    auto mid = load(2);
                    sum = __riscv_vadd(sum, __riscv_vadd(__riscv_vsll(mid, 2, vl), __riscv_vsll(mid, 1, vl), vl), vl);
                }
                __riscv_vse8(dst_data + cur * dst_step + fj, __riscv_vnclipu(sum, ksize == 5 ? 8 : 4, __RISCV_VXRM_RNU, vl), vl);
            }
        }
    }

    return CV_HAL_ERROR_OK;
}

} // anonymous

int gaussianBlurBinomial(const uchar* src_data, size_t src_step, uchar* dst_data, size_t dst_step, int width, int height, int depth, int cn, size_t margin_left, size_t margin_top, size_t margin_right, size_t margin_bottom, size_t ksize, int border_type)
{
    const int type = CV_MAKETYPE(depth, cn);
    if ((type != CV_8UC1 && type != CV_8UC3 && type != CV_8UC4 && type != CV_16UC1) || src_data == dst_data)
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
    case 300 + CV_8UC3:
        return common::invoke(height, {gaussianBlurFlat8U<3, 3>}, src_data, src_step, dst_data, dst_step, width, margin_left + width + margin_right, margin_top + height + margin_bottom, margin_left, margin_top, border_type);
    case 500 + CV_8UC3:
        return common::invoke(height, {gaussianBlurFlat8U<5, 3>}, src_data, src_step, dst_data, dst_step, width, margin_left + width + margin_right, margin_top + height + margin_bottom, margin_left, margin_top, border_type);
    case 300 + CV_8UC4:
        return common::invoke(height, {gaussianBlurFlat8U<3, 4>}, src_data, src_step, dst_data, dst_step, width, margin_left + width + margin_right, margin_top + height + margin_bottom, margin_left, margin_top, border_type);
    case 500 + CV_8UC4:
        return common::invoke(height, {gaussianBlurFlat8U<5, 4>}, src_data, src_step, dst_data, dst_step, width, margin_left + width + margin_right, margin_top + height + margin_bottom, margin_left, margin_top, border_type);
    }

    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

#endif // CV_HAL_RVV_1P0_ENABLED

}}} // cv::rvv_hal::imgproc
