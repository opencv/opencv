// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2025, Institute of Software, Chinese Academy of Sciences.

#include "rvv_hal.hpp"
#include "common.hpp"

namespace cv { namespace rvv_hal { namespace imgproc {

#if CV_HAL_RVV_1P0_ENABLED

namespace {

// Exact (a + b/2) / b for b in {9, 25, 49} via multiply-high and shift
// (Granlund-Montgomery, m = ceil(2^18 / b)); exact while a + b/2 < 32768,
// the maximum here is 49*255 + 24 = 12519
template<typename VecType>
static inline VecType vdiv_box_u16(VecType a, ushort b, size_t vl)
{
    const ushort m = b == 9 ? 29128 : b == 25 ? 10486 : 5350;
    return __riscv_vsrl(__riscv_vmulhu(__riscv_vadd(a, b / 2, vl), m, vl), 2, vl);
}

template<typename T> struct rvv;
template<> struct rvv<uchar>
{
    static inline vuint16m8_t vcvt0(vuint8m4_t a, size_t b) { return __riscv_vzext_vf2(a, b); }
    static inline vuint8m4_t vcvt1(vuint16m8_t a, size_t b) { return __riscv_vnclipu(a, 0, __RISCV_VXRM_RNU, b); }
    static inline vuint16m8_t vdiv(vuint16m8_t a, ushort b, size_t c) { return vdiv_box_u16(a, b, c); }
};
template<> struct rvv<short>
{
    static inline vint32m8_t vcvt0(vint16m4_t a, size_t b) { return __riscv_vsext_vf2(a, b); }
    static inline vint16m4_t vcvt1(vint32m8_t a, size_t b) { return __riscv_vnclip(a, 0, __RISCV_VXRM_RNU, b); }
    static inline vint32m8_t vdiv(vint32m8_t a, int b, size_t c) { return __riscv_vdiv(__riscv_vadd(a, b / 2, c), b, c); }
};
template<> struct rvv<int>
{
    static inline vint32m8_t vcvt0(vint32m8_t a, size_t) { return a; }
    static inline vint32m8_t vcvt1(vint32m8_t a, size_t) { return a; }
    static inline vint32m8_t vdiv(vint32m8_t a, int b, size_t c) { return __riscv_vdiv(__riscv_vadd(a, b / 2, c), b, c); }
};
template<> struct rvv<float>
{
    static inline vfloat32m8_t vcvt0(vfloat32m8_t a, size_t) { return a; }
    static inline vfloat32m8_t vcvt1(vfloat32m8_t a, size_t) { return a; }
    static inline vfloat32m8_t vdiv(vfloat32m8_t a, float b, size_t c) { return __riscv_vfdiv(a, b, c); }
};

// the algorithm is same as cv_hal_sepFilter
template<int ksize, typename helperT, typename helperWT, bool cast>
static inline int boxFilterC1(int start, int end, const uchar* src_data, size_t src_step, uchar* dst_data, size_t dst_step, int width, int full_width, int full_height, int offset_x, int offset_y, int anchor_x, int anchor_y, bool normalize, int border_type)
{
    using T = typename helperT::ElemType;
    using WT = typename helperWT::ElemType;

    constexpr int noval = std::numeric_limits<int>::max();
    auto accessX = [&](int x) {
        int pi = common::borderInterpolate(offset_y + x - anchor_y, full_height, border_type);
        return pi < 0 ? noval : pi - offset_y;
    };
    auto accessY = [&](int y) {
        int pj = common::borderInterpolate(offset_x + y - anchor_x, full_width, border_type);
        return pj < 0 ? noval : pj - offset_x;
    };
    auto p2idx = [&](int x, int y){ return (x + ksize) % ksize * width + y; };

    std::vector<WT> res(width * ksize);
    auto process = [&](int x, int y) {
        WT sum = 0;
        for (int i = 0; i < ksize; i++)
        {
            int p = accessY(y + i);
            if (p != noval)
            {
                sum += reinterpret_cast<const T*>(src_data + x * src_step)[p];
            }
        }
        res[p2idx(x, y)] = sum;
    };

    const int left = anchor_x, right = width - (ksize - 1 - anchor_x);
    for (int i = start - anchor_y; i < end + (ksize - 1 - anchor_y); i++)
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
                    const T* extra = reinterpret_cast<const T*>(src_data + i * src_step) + j - anchor_x;
                    auto src = rvv<T>::vcvt0(helperT::vload(extra, vl), vl);

                    extra += vl;
                    auto sum = src;
                    src = helperWT::vslide1down(src, extra[0], vl);
                    sum = helperWT::vadd(sum, src, vl);
                    src = helperWT::vslide1down(src, extra[1], vl);
                    sum = helperWT::vadd(sum, src, vl);
                    if (ksize >= 5)
                    {
                        src = helperWT::vslide1down(src, extra[2], vl);
                        sum = helperWT::vadd(sum, src, vl);
                        src = helperWT::vslide1down(src, extra[3], vl);
                        sum = helperWT::vadd(sum, src, vl);
                    }
                    if (ksize == 7)
                    {
                        src = helperWT::vslide1down(src, extra[4], vl);
                        sum = helperWT::vadd(sum, src, vl);
                        src = helperWT::vslide1down(src, extra[5], vl);
                        sum = helperWT::vadd(sum, src, vl);
                    }
                    helperWT::vstore(res.data() + p2idx(i, j), sum, vl);
                }
            }
        }

        int cur = i - (ksize - 1 - anchor_y);
        if (cur >= start)
        {
            const WT* row0 = accessX(cur    ) == noval ? nullptr : res.data() + p2idx(accessX(cur    ), 0);
            const WT* row1 = accessX(cur + 1) == noval ? nullptr : res.data() + p2idx(accessX(cur + 1), 0);
            const WT* row2 = accessX(cur + 2) == noval ? nullptr : res.data() + p2idx(accessX(cur + 2), 0);
            const WT* row3 = nullptr, *row4 = nullptr, *row5 = nullptr, *row6 = nullptr;
            if (ksize >= 5)
            {
                row3 = accessX(cur + 3) == noval ? nullptr : res.data() + p2idx(accessX(cur + 3), 0);
                row4 = accessX(cur + 4) == noval ? nullptr : res.data() + p2idx(accessX(cur + 4), 0);
            }
            if (ksize == 7)
            {
                row5 = accessX(cur + 5) == noval ? nullptr : res.data() + p2idx(accessX(cur + 5), 0);
                row6 = accessX(cur + 6) == noval ? nullptr : res.data() + p2idx(accessX(cur + 6), 0);
            }

            int vl;
            for (int j = 0; j < width; j += vl)
            {
                vl = helperWT::setvl(width - j);
                auto sum = row0 ? helperWT::vload(row0 + j, vl) : helperWT::vmv(0, vl);
                if (row1) sum = helperWT::vadd(sum, helperWT::vload(row1 + j, vl), vl);
                if (row2) sum = helperWT::vadd(sum, helperWT::vload(row2 + j, vl), vl);
                if (row3) sum = helperWT::vadd(sum, helperWT::vload(row3 + j, vl), vl);
                if (row4) sum = helperWT::vadd(sum, helperWT::vload(row4 + j, vl), vl);
                if (row5) sum = helperWT::vadd(sum, helperWT::vload(row5 + j, vl), vl);
                if (row6) sum = helperWT::vadd(sum, helperWT::vload(row6 + j, vl), vl);
                if (normalize) sum = rvv<T>::vdiv(sum, ksize * ksize, vl);

                if (cast)
                {
                    helperT::vstore(reinterpret_cast<T*>(dst_data + cur * dst_step) + j, rvv<T>::vcvt1(sum, vl), vl);
                }
                else
                {
                    helperWT::vstore(reinterpret_cast<WT*>(dst_data + cur * dst_step) + j, sum, vl);
                }
            }
        }
    }

    return CV_HAL_ERROR_OK;
}

template<int ksize>
static inline int boxFilterC3(int start, int end, const uchar* src_data, size_t src_step, uchar* dst_data, size_t dst_step, int width, int full_width, int full_height, int offset_x, int offset_y, int anchor_x, int anchor_y, bool normalize, int border_type)
{
    constexpr int noval = std::numeric_limits<int>::max();
    auto accessX = [&](int x) {
        int pi = common::borderInterpolate(offset_y + x - anchor_y, full_height, border_type);
        return pi < 0 ? noval : pi - offset_y;
    };
    auto accessY = [&](int y) {
        int pj = common::borderInterpolate(offset_x + y - anchor_x, full_width, border_type);
        return pj < 0 ? noval : pj - offset_x;
    };
    auto p2idx = [&](int x, int y){ return ((x + ksize) % ksize * width + y) * 3; };

    std::vector<float> res(width * ksize * 3);
    auto process = [&](int x, int y) {
        float sum0, sum1, sum2;
        sum0 = sum1 = sum2 = 0;
        for (int i = 0; i < ksize; i++)
        {
            int p = accessY(y + i);
            if (p != noval)
            {
                sum0 += reinterpret_cast<const float*>(src_data + x * src_step)[p * 3    ];
                sum1 += reinterpret_cast<const float*>(src_data + x * src_step)[p * 3 + 1];
                sum2 += reinterpret_cast<const float*>(src_data + x * src_step)[p * 3 + 2];
            }
        }
        res[p2idx(x, y)    ] = sum0;
        res[p2idx(x, y) + 1] = sum1;
        res[p2idx(x, y) + 2] = sum2;
    };

    const int left = anchor_x, right = width - (ksize - 1 - anchor_x);
    for (int i = start - anchor_y; i < end + (ksize - 1 - anchor_y); i++)
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
                    vl = __riscv_vsetvl_e32m2(right - j);
                    const float* extra = reinterpret_cast<const float*>(src_data + i * src_step) + (j - anchor_x) * 3;
                    auto src = __riscv_vlseg3e32_v_f32m2x3(extra, vl);
                    auto src0 = __riscv_vget_v_f32m2x3_f32m2(src, 0);
                    auto src1 = __riscv_vget_v_f32m2x3_f32m2(src, 1);
                    auto src2 = __riscv_vget_v_f32m2x3_f32m2(src, 2);

                    extra += vl * 3;
                    auto sum0 = src0, sum1 = src1, sum2 = src2;
                    src0 = __riscv_vfslide1down(src0, extra[0], vl);
                    src1 = __riscv_vfslide1down(src1, extra[1], vl);
                    src2 = __riscv_vfslide1down(src2, extra[2], vl);
                    sum0 = __riscv_vfadd(sum0, src0, vl);
                    sum1 = __riscv_vfadd(sum1, src1, vl);
                    sum2 = __riscv_vfadd(sum2, src2, vl);
                    src0 = __riscv_vfslide1down(src0, extra[3], vl);
                    src1 = __riscv_vfslide1down(src1, extra[4], vl);
                    src2 = __riscv_vfslide1down(src2, extra[5], vl);
                    sum0 = __riscv_vfadd(sum0, src0, vl);
                    sum1 = __riscv_vfadd(sum1, src1, vl);
                    sum2 = __riscv_vfadd(sum2, src2, vl);
                    if (ksize >= 5)
                    {
                        src0 = __riscv_vfslide1down(src0, extra[6], vl);
                        src1 = __riscv_vfslide1down(src1, extra[7], vl);
                        src2 = __riscv_vfslide1down(src2, extra[8], vl);
                        sum0 = __riscv_vfadd(sum0, src0, vl);
                        sum1 = __riscv_vfadd(sum1, src1, vl);
                        sum2 = __riscv_vfadd(sum2, src2, vl);
                        src0 = __riscv_vfslide1down(src0, extra[ 9], vl);
                        src1 = __riscv_vfslide1down(src1, extra[10], vl);
                        src2 = __riscv_vfslide1down(src2, extra[11], vl);
                        sum0 = __riscv_vfadd(sum0, src0, vl);
                        sum1 = __riscv_vfadd(sum1, src1, vl);
                        sum2 = __riscv_vfadd(sum2, src2, vl);
                    }
                    if (ksize == 7)
                    {
                        src0 = __riscv_vfslide1down(src0, extra[12], vl);
                        src1 = __riscv_vfslide1down(src1, extra[13], vl);
                        src2 = __riscv_vfslide1down(src2, extra[14], vl);
                        sum0 = __riscv_vfadd(sum0, src0, vl);
                        sum1 = __riscv_vfadd(sum1, src1, vl);
                        sum2 = __riscv_vfadd(sum2, src2, vl);
                        src0 = __riscv_vfslide1down(src0, extra[15], vl);
                        src1 = __riscv_vfslide1down(src1, extra[16], vl);
                        src2 = __riscv_vfslide1down(src2, extra[17], vl);
                        sum0 = __riscv_vfadd(sum0, src0, vl);
                        sum1 = __riscv_vfadd(sum1, src1, vl);
                        sum2 = __riscv_vfadd(sum2, src2, vl);
                    }

                    vfloat32m2x3_t dst{};
                    dst = __riscv_vset_v_f32m2_f32m2x3(dst, 0, sum0);
                    dst = __riscv_vset_v_f32m2_f32m2x3(dst, 1, sum1);
                    dst = __riscv_vset_v_f32m2_f32m2x3(dst, 2, sum2);
                    __riscv_vsseg3e32(res.data() + p2idx(i, j), dst, vl);
                }
            }
        }

        int cur = i - (ksize - 1 - anchor_y);
        if (cur >= start)
        {
            const float* row0 = accessX(cur    ) == noval ? nullptr : res.data() + p2idx(accessX(cur    ), 0);
            const float* row1 = accessX(cur + 1) == noval ? nullptr : res.data() + p2idx(accessX(cur + 1), 0);
            const float* row2 = accessX(cur + 2) == noval ? nullptr : res.data() + p2idx(accessX(cur + 2), 0);
            const float* row3 = nullptr, *row4 = nullptr, *row5 = nullptr, *row6 = nullptr;
            if (ksize >= 5)
            {
                row3 = accessX(cur + 3) == noval ? nullptr : res.data() + p2idx(accessX(cur + 3), 0);
                row4 = accessX(cur + 4) == noval ? nullptr : res.data() + p2idx(accessX(cur + 4), 0);
            }
            if (ksize == 7)
            {
                row5 = accessX(cur + 5) == noval ? nullptr : res.data() + p2idx(accessX(cur + 5), 0);
                row6 = accessX(cur + 6) == noval ? nullptr : res.data() + p2idx(accessX(cur + 6), 0);
            }

            int vl;
            for (int j = 0; j < width; j += vl)
            {
                vl = __riscv_vsetvl_e32m2(width - j);
                vfloat32m2_t sum0, sum1, sum2;
                sum0 = sum1 = sum2 = __riscv_vfmv_v_f_f32m2(0, vl);
                auto loadres = [&](const float* row) {
                    if (!row) return;
                    auto src = __riscv_vlseg3e32_v_f32m2x3(row + j * 3, vl);
                    sum0 = __riscv_vfadd(sum0, __riscv_vget_v_f32m2x3_f32m2(src, 0), vl);
                    sum1 = __riscv_vfadd(sum1, __riscv_vget_v_f32m2x3_f32m2(src, 1), vl);
                    sum2 = __riscv_vfadd(sum2, __riscv_vget_v_f32m2x3_f32m2(src, 2), vl);
                };
                loadres(row0);
                loadres(row1);
                loadres(row2);
                loadres(row3);
                loadres(row4);
                loadres(row5);
                loadres(row6);
                if (normalize)
                {
                    sum0 = __riscv_vfdiv(sum0, ksize * ksize, vl);
                    sum1 = __riscv_vfdiv(sum1, ksize * ksize, vl);
                    sum2 = __riscv_vfdiv(sum2, ksize * ksize, vl);
                }

                vfloat32m2x3_t dst{};
                dst = __riscv_vset_v_f32m2_f32m2x3(dst, 0, sum0);
                dst = __riscv_vset_v_f32m2_f32m2x3(dst, 1, sum1);
                dst = __riscv_vset_v_f32m2_f32m2x3(dst, 2, sum2);
                __riscv_vsseg3e32(reinterpret_cast<float*>(dst_data + cur * dst_step) + j * 3, dst, vl);
            }
        }
    }

    return CV_HAL_ERROR_OK;
}

// Channel-blind kernel for interleaved 8U data: a row is treated as width*cn
// flat u8 elements, whose horizontal taps sit at strides of cn elements
// (flat[i] sums flat[i], flat[i+cn], ..., flat[i+(ksize-1)*cn], which are
// exactly element i's own-channel taps). This removes all segment loads and
// slide chains: the horizontal pass is ksize independent unaligned loads with
// widening adds, the vertical pass is a single contiguous u16 stream, and the
// output is a plain store. u16 column sums are exact for ksize <= 7
// (49*255 = 12495 < 65535).
template<int ksize, int cn>
static inline int boxFilterFlat8U(int start, int end, const uchar* src_data, size_t src_step, uchar* dst_data, size_t dst_step, int width, int full_width, int full_height, int offset_x, int offset_y, int anchor_x, int anchor_y, bool normalize, int border_type)
{
    constexpr int noval = std::numeric_limits<int>::max();
    auto accessX = [&](int x) {
        int pi = common::borderInterpolate(offset_y + x - anchor_y, full_height, border_type);
        return pi < 0 ? noval : pi - offset_y;
    };
    auto accessY = [&](int y) {
        int pj = common::borderInterpolate(offset_x + y - anchor_x, full_width, border_type);
        return pj < 0 ? noval : pj - offset_x;
    };
    const int W = width * cn;
    // ring keeps one extra row so the row leaving the window is still present
    // for the incremental vertical update (ksize > 3)
    constexpr int ring = ksize > 3 ? ksize + 1 : ksize;
    auto p2idx = [&](int x){ return (x + ring) % ring * W; };

    std::vector<ushort> res(W * ring);
    std::vector<ushort> vsum(ksize > 3 ? W : 0);
    bool vsum_valid = false;
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
                    sum[c] += (src_data + x * src_step)[p * cn + c];
            }
        }
        for (int c = 0; c < cn; c++)
            res[p2idx(x) + y * cn + c] = sum[c];
    };

    const int left = anchor_x, right = width - (ksize - 1 - anchor_x);
    for (int i = start - anchor_y; i < end + (ksize - 1 - anchor_y); i++)
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
                    const uchar* p = src_data + i * src_step + fj - anchor_x * cn;
                    auto acc = __riscv_vwaddu_vv(__riscv_vle8_v_u8m4(p, vl), __riscv_vle8_v_u8m4(p + cn, vl), vl);
                    for (int t = 2; t < ksize; t++)
                        acc = __riscv_vwaddu_wv(acc, __riscv_vle8_v_u8m4(p + t * cn, vl), vl);
                    __riscv_vse16(row + fj, acc, vl);
                }
            }
        }

        int cur = i - (ksize - 1 - anchor_y);
        if (cur >= start)
        {
            if (ksize > 3 && vsum_valid)
            {
                // slide the window down one row: add the entering row's column
                // sums, subtract the leaving row's (u16 stays exact: the window
                // sum and the intermediate sum+entering both fit)
                int add_r = accessX(cur + ksize - 1);
                int sub_r = accessX(cur - 1);
                const ushort* addp = add_r == noval ? nullptr : res.data() + p2idx(add_r);
                const ushort* subp = sub_r == noval ? nullptr : res.data() + p2idx(sub_r);

                int vl;
                for (int fj = 0; fj < W; fj += vl)
                {
                    vl = __riscv_vsetvl_e16m8(W - fj);
                    auto sum = __riscv_vle16_v_u16m8(vsum.data() + fj, vl);
                    if (addp) sum = __riscv_vadd(sum, __riscv_vle16_v_u16m8(addp + fj, vl), vl);
                    if (subp) sum = __riscv_vsub(sum, __riscv_vle16_v_u16m8(subp + fj, vl), vl);
                    __riscv_vse16(vsum.data() + fj, sum, vl);
                    if (normalize) sum = vdiv_box_u16(sum, ksize * ksize, vl);
                    __riscv_vse8(dst_data + cur * dst_step + fj, __riscv_vnclipu(sum, 0, __RISCV_VXRM_RNU, vl), vl);
                }
            }
            else
            {
                const ushort* rows[ksize];
                for (int k = 0; k < ksize; k++)
                    rows[k] = accessX(cur + k) == noval ? nullptr : res.data() + p2idx(accessX(cur + k));

                int vl;
                for (int fj = 0; fj < W; fj += vl)
                {
                    vl = __riscv_vsetvl_e16m8(W - fj);
                    auto sum = rows[0] ? __riscv_vle16_v_u16m8(rows[0] + fj, vl) : __riscv_vmv_v_x_u16m8(0, vl);
                    for (int k = 1; k < ksize; k++)
                        if (rows[k]) sum = __riscv_vadd(sum, __riscv_vle16_v_u16m8(rows[k] + fj, vl), vl);
                    if (ksize > 3) __riscv_vse16(vsum.data() + fj, sum, vl);
                    if (normalize) sum = vdiv_box_u16(sum, ksize * ksize, vl);
                    __riscv_vse8(dst_data + cur * dst_step + fj, __riscv_vnclipu(sum, 0, __RISCV_VXRM_RNU, vl), vl);
                }
                vsum_valid = true;
            }
        }
    }

    return CV_HAL_ERROR_OK;
}

} // anonymous

int boxFilter(const uchar* src_data, size_t src_step, uchar* dst_data, size_t dst_step, int width, int height, int src_depth, int dst_depth, int cn, int margin_left, int margin_top, int margin_right, int margin_bottom, size_t ksize_width, size_t ksize_height, int anchor_x, int anchor_y, bool normalize, int border_type)
{
    const int src_type = CV_MAKETYPE(src_depth, cn), dst_type = CV_MAKETYPE(dst_depth, cn);
    if (ksize_width != ksize_height || (ksize_width != 3 && ksize_width != 5 && ksize_width != 7))
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if (border_type & BORDER_ISOLATED || border_type == BORDER_WRAP)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    uchar* _dst_data = dst_data;
    size_t _dst_step = dst_step;
    const size_t size = CV_ELEM_SIZE(dst_type);
    std::vector<uchar> dst;
    if (src_data == _dst_data)
    {
        dst = std::vector<uchar>(width * height * size);
        dst_data = dst.data();
        dst_step = width * size;
    }

    int res = CV_HAL_ERROR_NOT_IMPLEMENTED;
    anchor_x = anchor_x < 0 ? ksize_width  / 2 : anchor_x;
    anchor_y = anchor_y < 0 ? ksize_height / 2 : anchor_y;
    if (src_type != dst_type)
    {
        if (src_type == CV_8UC1 && dst_type == CV_16UC1)
        {
            if (ksize_width == 3)
            {
                res = common::invoke(height, {boxFilterC1<3, RVV_U8M4, RVV_U16M8, false>}, src_data, src_step, dst_data, dst_step, width, margin_left + width + margin_right, margin_top + height + margin_bottom, margin_left, margin_top, anchor_x, anchor_y, normalize, border_type);
            }
            if (ksize_width == 5)
            {
                res = common::invoke(height, {boxFilterC1<5, RVV_U8M4, RVV_U16M8, false>}, src_data, src_step, dst_data, dst_step, width, margin_left + width + margin_right, margin_top + height + margin_bottom, margin_left, margin_top, anchor_x, anchor_y, normalize, border_type);
            }
            if (ksize_width == 7)
            {
                res = common::invoke(height, {boxFilterC1<7, RVV_U8M4, RVV_U16M8, false>}, src_data, src_step, dst_data, dst_step, width, margin_left + width + margin_right, margin_top + height + margin_bottom, margin_left, margin_top, anchor_x, anchor_y, normalize, border_type);
            }
        }
    }
    else
    {
        switch (ksize_width*100 + src_type)
        {
        case 300 + CV_8UC1:
            res = common::invoke(height, {boxFilterC1<3, RVV_U8M4, RVV_U16M8, true>}, src_data, src_step, dst_data, dst_step, width, margin_left + width + margin_right, margin_top + height + margin_bottom, margin_left, margin_top, anchor_x, anchor_y, normalize, border_type);
            break;
        case 500 + CV_8UC1:
            res = common::invoke(height, {boxFilterC1<5, RVV_U8M4, RVV_U16M8, true>}, src_data, src_step, dst_data, dst_step, width, margin_left + width + margin_right, margin_top + height + margin_bottom, margin_left, margin_top, anchor_x, anchor_y, normalize, border_type);
            break;
        case 700 + CV_8UC1:
            res = common::invoke(height, {boxFilterFlat8U<7, 1>}, src_data, src_step, dst_data, dst_step, width, margin_left + width + margin_right, margin_top + height + margin_bottom, margin_left, margin_top, anchor_x, anchor_y, normalize, border_type);
            break;
        case 300 + CV_8UC3:
            res = common::invoke(height, {boxFilterFlat8U<3, 3>}, src_data, src_step, dst_data, dst_step, width, margin_left + width + margin_right, margin_top + height + margin_bottom, margin_left, margin_top, anchor_x, anchor_y, normalize, border_type);
            break;
        case 500 + CV_8UC3:
            res = common::invoke(height, {boxFilterFlat8U<5, 3>}, src_data, src_step, dst_data, dst_step, width, margin_left + width + margin_right, margin_top + height + margin_bottom, margin_left, margin_top, anchor_x, anchor_y, normalize, border_type);
            break;
        case 700 + CV_8UC3:
            res = common::invoke(height, {boxFilterFlat8U<7, 3>}, src_data, src_step, dst_data, dst_step, width, margin_left + width + margin_right, margin_top + height + margin_bottom, margin_left, margin_top, anchor_x, anchor_y, normalize, border_type);
            break;
        case 300 + CV_8UC4:
            res = common::invoke(height, {boxFilterFlat8U<3, 4>}, src_data, src_step, dst_data, dst_step, width, margin_left + width + margin_right, margin_top + height + margin_bottom, margin_left, margin_top, anchor_x, anchor_y, normalize, border_type);
            break;
        case 500 + CV_8UC4:
            res = common::invoke(height, {boxFilterFlat8U<5, 4>}, src_data, src_step, dst_data, dst_step, width, margin_left + width + margin_right, margin_top + height + margin_bottom, margin_left, margin_top, anchor_x, anchor_y, normalize, border_type);
            break;
        case 700 + CV_8UC4:
            res = common::invoke(height, {boxFilterFlat8U<7, 4>}, src_data, src_step, dst_data, dst_step, width, margin_left + width + margin_right, margin_top + height + margin_bottom, margin_left, margin_top, anchor_x, anchor_y, normalize, border_type);
            break;
        case 300 + CV_16SC1:
            res = common::invoke(height, {boxFilterC1<3, RVV_I16M4, RVV_I32M8, true>}, src_data, src_step, dst_data, dst_step, width, margin_left + width + margin_right, margin_top + height + margin_bottom, margin_left, margin_top, anchor_x, anchor_y, normalize, border_type);
            break;
        case 500 + CV_16SC1:
            res = common::invoke(height, {boxFilterC1<5, RVV_I16M4, RVV_I32M8, true>}, src_data, src_step, dst_data, dst_step, width, margin_left + width + margin_right, margin_top + height + margin_bottom, margin_left, margin_top, anchor_x, anchor_y, normalize, border_type);
            break;
        case 700 + CV_16SC1:
            res = common::invoke(height, {boxFilterC1<7, RVV_I16M4, RVV_I32M8, true>}, src_data, src_step, dst_data, dst_step, width, margin_left + width + margin_right, margin_top + height + margin_bottom, margin_left, margin_top, anchor_x, anchor_y, normalize, border_type);
            break;
        case 300 + CV_32SC1:
            res = common::invoke(height, {boxFilterC1<3, RVV_I32M8, RVV_I32M8, true>}, src_data, src_step, dst_data, dst_step, width, margin_left + width + margin_right, margin_top + height + margin_bottom, margin_left, margin_top, anchor_x, anchor_y, normalize, border_type);
            break;
        case 500 + CV_32SC1:
            res = common::invoke(height, {boxFilterC1<5, RVV_I32M8, RVV_I32M8, true>}, src_data, src_step, dst_data, dst_step, width, margin_left + width + margin_right, margin_top + height + margin_bottom, margin_left, margin_top, anchor_x, anchor_y, normalize, border_type);
            break;
        case 300 + CV_32FC1:
            res = common::invoke(height, {boxFilterC1<3, RVV_F32M8, RVV_F32M8, true>}, src_data, src_step, dst_data, dst_step, width, margin_left + width + margin_right, margin_top + height + margin_bottom, margin_left, margin_top, anchor_x, anchor_y, normalize, border_type);
            break;
        case 500 + CV_32FC1:
            res = common::invoke(height, {boxFilterC1<5, RVV_F32M8, RVV_F32M8, true>}, src_data, src_step, dst_data, dst_step, width, margin_left + width + margin_right, margin_top + height + margin_bottom, margin_left, margin_top, anchor_x, anchor_y, normalize, border_type);
            break;
        case 700 + CV_32FC1:
            res = common::invoke(height, {boxFilterC1<7, RVV_F32M8, RVV_F32M8, true>}, src_data, src_step, dst_data, dst_step, width, margin_left + width + margin_right, margin_top + height + margin_bottom, margin_left, margin_top, anchor_x, anchor_y, normalize, border_type);
            break;
        case 300 + CV_32FC3:
            res = common::invoke(height, {boxFilterC3<3>}, src_data, src_step, dst_data, dst_step, width, margin_left + width + margin_right, margin_top + height + margin_bottom, margin_left, margin_top, anchor_x, anchor_y, normalize, border_type);
            break;
        case 500 + CV_32FC3:
            res = common::invoke(height, {boxFilterC3<5>}, src_data, src_step, dst_data, dst_step, width, margin_left + width + margin_right, margin_top + height + margin_bottom, margin_left, margin_top, anchor_x, anchor_y, normalize, border_type);
            break;
        case 700 + CV_32FC3:
            res = common::invoke(height, {boxFilterC3<7>}, src_data, src_step, dst_data, dst_step, width, margin_left + width + margin_right, margin_top + height + margin_bottom, margin_left, margin_top, anchor_x, anchor_y, normalize, border_type);
            break;
        }
    }
    if (res == CV_HAL_ERROR_NOT_IMPLEMENTED)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    if (src_data == _dst_data)
    {
        for (int i = 0; i < height; i++)
            memcpy(_dst_data + i * _dst_step, dst.data() + i * dst_step, dst_step);
    }

    return res;
}

#endif // CV_HAL_RVV_1P0_ENABLED

}}} // cv::rvv_hal::imgproc
