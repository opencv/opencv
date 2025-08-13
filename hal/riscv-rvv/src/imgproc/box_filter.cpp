// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2025, Institute of Software, Chinese Academy of Sciences.

#include "rvv_hal.hpp"
#include "common.hpp"

namespace cv { namespace rvv_hal { namespace imgproc {

#if CV_HAL_RVV_1P0_ENABLED

namespace {

template<typename T> struct rvv;
template<> struct rvv<uchar>
{
    static inline vuint16m8_t vcvt0(vuint8m4_t a, size_t b) { return __riscv_vzext_vf2(a, b); }
    static inline vuint8m4_t vcvt1(vuint16m8_t a, size_t b) { return __riscv_vnclipu(a, 0, __RISCV_VXRM_RNU, b); }
    static inline vuint16m8_t vdiv(vuint16m8_t a, ushort b, size_t c) { return __riscv_vdivu(__riscv_vadd(a, b / 2, c), b, c); }
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
                    if (ksize == 5)
                    {
                        src = helperWT::vslide1down(src, extra[2], vl);
                        sum = helperWT::vadd(sum, src, vl);
                        src = helperWT::vslide1down(src, extra[3], vl);
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
                auto sum = row0 ? helperWT::vload(row0 + j, vl) : helperWT::vmv(0, vl);
                if (row1) sum = helperWT::vadd(sum, helperWT::vload(row1 + j, vl), vl);
                if (row2) sum = helperWT::vadd(sum, helperWT::vload(row2 + j, vl), vl);
                if (row3) sum = helperWT::vadd(sum, helperWT::vload(row3 + j, vl), vl);
                if (row4) sum = helperWT::vadd(sum, helperWT::vload(row4 + j, vl), vl);
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
                    if (ksize == 5)
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
            const float* row3 = nullptr, *row4 = nullptr;
            if (ksize == 5)
            {
                row3 = accessX(cur + 3) == noval ? nullptr : res.data() + p2idx(accessX(cur + 3), 0);
                row4 = accessX(cur + 4) == noval ? nullptr : res.data() + p2idx(accessX(cur + 4), 0);
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

} // anonymous

int boxFilter(const uchar* src_data, size_t src_step, uchar* dst_data, size_t dst_step, int width, int height, int src_depth, int dst_depth, int cn, int margin_left, int margin_top, int margin_right, int margin_bottom, size_t ksize_width, size_t ksize_height, int anchor_x, int anchor_y, bool normalize, int border_type)
{
    const int src_type = CV_MAKETYPE(src_depth, cn), dst_type = CV_MAKETYPE(dst_depth, cn);
    if (ksize_width != ksize_height || (ksize_width != 3 && ksize_width != 5))
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
        case 300 + CV_16SC1:
            res = common::invoke(height, {boxFilterC1<3, RVV_I16M4, RVV_I32M8, true>}, src_data, src_step, dst_data, dst_step, width, margin_left + width + margin_right, margin_top + height + margin_bottom, margin_left, margin_top, anchor_x, anchor_y, normalize, border_type);
            break;
        case 500 + CV_16SC1:
            res = common::invoke(height, {boxFilterC1<5, RVV_I16M4, RVV_I32M8, true>}, src_data, src_step, dst_data, dst_step, width, margin_left + width + margin_right, margin_top + height + margin_bottom, margin_left, margin_top, anchor_x, anchor_y, normalize, border_type);
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
        case 300 + CV_32FC3:
            res = common::invoke(height, {boxFilterC3<3>}, src_data, src_step, dst_data, dst_step, width, margin_left + width + margin_right, margin_top + height + margin_bottom, margin_left, margin_top, anchor_x, anchor_y, normalize, border_type);
            break;
        case 500 + CV_32FC3:
            res = common::invoke(height, {boxFilterC3<5>}, src_data, src_step, dst_data, dst_step, width, margin_left + width + margin_right, margin_top + height + margin_bottom, margin_left, margin_top, anchor_x, anchor_y, normalize, border_type);
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
