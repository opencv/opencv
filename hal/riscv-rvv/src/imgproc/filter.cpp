// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2025, Institute of Software, Chinese Academy of Sciences.

#include "rvv_hal.hpp"
#include "common.hpp"

namespace cv { namespace rvv_hal { namespace imgproc {

#if CV_HAL_RVV_1P0_ENABLED

namespace {

struct Filter2D
{
    const uchar* kernel_data;
    size_t kernel_step;
    int kernel_type;
    int kernel_width;
    int kernel_height;
    int src_type;
    int dst_type;
    int borderType;
    double delta;
    int anchor_x;
    int anchor_y;
};

static void process3(int anchor, int left, int right, float delta, const float* kernel, const uchar* row0, const uchar* row1, const uchar* row2, uchar* dst)
{
    int vl;
    for (int i = left; i < right; i += vl)
    {
        vl = __riscv_vsetvl_e8m1(right - i);
        auto s0 = __riscv_vfmv_v_f_f32m4(delta, vl);
        auto s1 = __riscv_vfmv_v_f_f32m4(delta, vl);
        auto s2 = __riscv_vfmv_v_f_f32m4(delta, vl);
        auto s3 = __riscv_vfmv_v_f_f32m4(delta, vl);

        auto addshift = [&](vfloat32m4_t a, vfloat32m4_t b, float k0, float k1, float k2, float r1, float r2) {
            a = __riscv_vfmacc(a, k0, b, vl);
            b = __riscv_vfslide1down(b, r1, vl);
            a = __riscv_vfmacc(a, k1, b, vl);
            b = __riscv_vfslide1down(b, r2, vl);
            return __riscv_vfmacc(a, k2, b, vl);
        };
        auto loadsrc = [&](const uchar* row, float k0, float k1, float k2) {
            if (!row) return;

            const uchar* extra = row + (i - anchor) * 4;
            auto src = __riscv_vlseg4e8_v_u8m1x4(extra, vl);
            auto v0 = __riscv_vfwcvt_f(__riscv_vwcvtu_x(__riscv_vget_v_u8m1x4_u8m1(src, 0), vl), vl);
            auto v1 = __riscv_vfwcvt_f(__riscv_vwcvtu_x(__riscv_vget_v_u8m1x4_u8m1(src, 1), vl), vl);
            auto v2 = __riscv_vfwcvt_f(__riscv_vwcvtu_x(__riscv_vget_v_u8m1x4_u8m1(src, 2), vl), vl);
            auto v3 = __riscv_vfwcvt_f(__riscv_vwcvtu_x(__riscv_vget_v_u8m1x4_u8m1(src, 3), vl), vl);

            extra += vl * 4;
            s0 = addshift(s0, v0, k0, k1, k2, extra[0], extra[4]);
            s1 = addshift(s1, v1, k0, k1, k2, extra[1], extra[5]);
            s2 = addshift(s2, v2, k0, k1, k2, extra[2], extra[6]);
            s3 = addshift(s3, v3, k0, k1, k2, extra[3], extra[7]);
        };

        loadsrc(row0, kernel[0], kernel[1], kernel[2]);
        loadsrc(row1, kernel[3], kernel[4], kernel[5]);
        loadsrc(row2, kernel[6], kernel[7], kernel[8]);
        vuint8m1x4_t val{};
        val = __riscv_vset_v_u8m1_u8m1x4(val, 0, __riscv_vnclipu(__riscv_vfncvt_xu(s0, vl), 0, __RISCV_VXRM_RNU, vl));
        val = __riscv_vset_v_u8m1_u8m1x4(val, 1, __riscv_vnclipu(__riscv_vfncvt_xu(s1, vl), 0, __RISCV_VXRM_RNU, vl));
        val = __riscv_vset_v_u8m1_u8m1x4(val, 2, __riscv_vnclipu(__riscv_vfncvt_xu(s2, vl), 0, __RISCV_VXRM_RNU, vl));
        val = __riscv_vset_v_u8m1_u8m1x4(val, 3, __riscv_vnclipu(__riscv_vfncvt_xu(s3, vl), 0, __RISCV_VXRM_RNU, vl));
        __riscv_vsseg4e8(dst + i * 4, val, vl);
    }
}

static void process5(int anchor, int left, int right, float delta, const float* kernel, const uchar* row0, const uchar* row1, const uchar* row2, const uchar* row3, const uchar* row4, uchar* dst)
{
    int vl;
    for (int i = left; i < right; i += vl)
    {
        vl = __riscv_vsetvl_e8m1(right - i);
        auto s0 = __riscv_vfmv_v_f_f32m4(delta, vl);
        auto s1 = __riscv_vfmv_v_f_f32m4(delta, vl);
        auto s2 = __riscv_vfmv_v_f_f32m4(delta, vl);
        auto s3 = __riscv_vfmv_v_f_f32m4(delta, vl);

        auto addshift = [&](vfloat32m4_t a, vfloat32m4_t b, float k0, float k1, float k2, float k3, float k4, float r1, float r2, float r3, float r4) {
            a = __riscv_vfmacc(a, k0, b, vl);
            b = __riscv_vfslide1down(b, r1, vl);
            a = __riscv_vfmacc(a, k1, b, vl);
            b = __riscv_vfslide1down(b, r2, vl);
            a = __riscv_vfmacc(a, k2, b, vl);
            b = __riscv_vfslide1down(b, r3, vl);
            a = __riscv_vfmacc(a, k3, b, vl);
            b = __riscv_vfslide1down(b, r4, vl);
            return __riscv_vfmacc(a, k4, b, vl);
        };
        auto loadsrc = [&](const uchar* row, float k0, float k1, float k2, float k3, float k4) {
            if (!row) return;

            const uchar* extra = row + (i - anchor) * 4;
            auto src = __riscv_vlseg4e8_v_u8m1x4(extra, vl);
            auto v0 = __riscv_vfwcvt_f(__riscv_vwcvtu_x(__riscv_vget_v_u8m1x4_u8m1(src, 0), vl), vl);
            auto v1 = __riscv_vfwcvt_f(__riscv_vwcvtu_x(__riscv_vget_v_u8m1x4_u8m1(src, 1), vl), vl);
            auto v2 = __riscv_vfwcvt_f(__riscv_vwcvtu_x(__riscv_vget_v_u8m1x4_u8m1(src, 2), vl), vl);
            auto v3 = __riscv_vfwcvt_f(__riscv_vwcvtu_x(__riscv_vget_v_u8m1x4_u8m1(src, 3), vl), vl);

            extra += vl * 4;
            s0 = addshift(s0, v0, k0, k1, k2, k3, k4, extra[0], extra[4], extra[ 8], extra[12]);
            s1 = addshift(s1, v1, k0, k1, k2, k3, k4, extra[1], extra[5], extra[ 9], extra[13]);
            s2 = addshift(s2, v2, k0, k1, k2, k3, k4, extra[2], extra[6], extra[10], extra[14]);
            s3 = addshift(s3, v3, k0, k1, k2, k3, k4, extra[3], extra[7], extra[11], extra[15]);
        };

        loadsrc(row0, kernel[ 0], kernel[ 1], kernel[ 2], kernel[ 3], kernel[ 4]);
        loadsrc(row1, kernel[ 5], kernel[ 6], kernel[ 7], kernel[ 8], kernel[ 9]);
        loadsrc(row2, kernel[10], kernel[11], kernel[12], kernel[13], kernel[14]);
        loadsrc(row3, kernel[15], kernel[16], kernel[17], kernel[18], kernel[19]);
        loadsrc(row4, kernel[20], kernel[21], kernel[22], kernel[23], kernel[24]);
        vuint8m1x4_t val{};
        val = __riscv_vset_v_u8m1_u8m1x4(val, 0, __riscv_vnclipu(__riscv_vfncvt_xu(s0, vl), 0, __RISCV_VXRM_RNU, vl));
        val = __riscv_vset_v_u8m1_u8m1x4(val, 1, __riscv_vnclipu(__riscv_vfncvt_xu(s1, vl), 0, __RISCV_VXRM_RNU, vl));
        val = __riscv_vset_v_u8m1_u8m1x4(val, 2, __riscv_vnclipu(__riscv_vfncvt_xu(s2, vl), 0, __RISCV_VXRM_RNU, vl));
        val = __riscv_vset_v_u8m1_u8m1x4(val, 3, __riscv_vnclipu(__riscv_vfncvt_xu(s3, vl), 0, __RISCV_VXRM_RNU, vl));
        __riscv_vsseg4e8(dst + i * 4, val, vl);
    }
}

// the algorithm is copied from 3rdparty/carotene/src/convolution.cpp,
// in the function void CAROTENE_NS::convolution
template<int ksize>
static inline int filter(int start, int end, Filter2D* data, const uchar* src_data, size_t src_step, uchar* dst_data, int width, int height, int full_width, int full_height, int offset_x, int offset_y)
{
    float kernel[ksize * ksize];
    for (int i = 0; i < ksize * ksize; i++)
    {
        kernel[i] = reinterpret_cast<const float*>(data->kernel_data + (i / ksize) * data->kernel_step)[i % ksize];
    }

    constexpr int noval = std::numeric_limits<int>::max();
    auto access = [&](int x, int y) {
        int pi, pj;
        if (data->borderType & BORDER_ISOLATED)
        {
            pi = common::borderInterpolate(x - data->anchor_y, height, data->borderType & ~BORDER_ISOLATED);
            pj = common::borderInterpolate(y - data->anchor_x, width , data->borderType & ~BORDER_ISOLATED);
            pi = pi < 0 ? noval : pi;
            pj = pj < 0 ? noval : pj;
        }
        else
        {
            pi = common::borderInterpolate(offset_y + x - data->anchor_y, full_height, data->borderType);
            pj = common::borderInterpolate(offset_x + y - data->anchor_x, full_width , data->borderType);
            pi = pi < 0 ? noval : pi - offset_y;
            pj = pj < 0 ? noval : pj - offset_x;
        }
        return std::make_pair(pi, pj);
    };

    auto process = [&](int x, int y) {
        float sum0, sum1, sum2, sum3;
        sum0 = sum1 = sum2 = sum3 = data->delta;
        for (int i = 0; i < ksize * ksize; i++)
        {
            auto p = access(x + i / ksize, y + i % ksize);
            if (p.first != noval && p.second != noval)
            {
                sum0 += kernel[i] * src_data[p.first * src_step + p.second * 4    ];
                sum1 += kernel[i] * src_data[p.first * src_step + p.second * 4 + 1];
                sum2 += kernel[i] * src_data[p.first * src_step + p.second * 4 + 2];
                sum3 += kernel[i] * src_data[p.first * src_step + p.second * 4 + 3];
            }
        }
        dst_data[(x * width + y) * 4    ] = std::max(0, std::min((int)std::round(sum0), (int)std::numeric_limits<uchar>::max()));
        dst_data[(x * width + y) * 4 + 1] = std::max(0, std::min((int)std::round(sum1), (int)std::numeric_limits<uchar>::max()));
        dst_data[(x * width + y) * 4 + 2] = std::max(0, std::min((int)std::round(sum2), (int)std::numeric_limits<uchar>::max()));
        dst_data[(x * width + y) * 4 + 3] = std::max(0, std::min((int)std::round(sum3), (int)std::numeric_limits<uchar>::max()));
    };

    const int left = data->anchor_x, right = width - (ksize - 1 - data->anchor_x);
    for (int i = start; i < end; i++)
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

            const uchar* row0 = access(i    , 0).first == noval ? nullptr : src_data + access(i    , 0).first * src_step;
            const uchar* row1 = access(i + 1, 0).first == noval ? nullptr : src_data + access(i + 1, 0).first * src_step;
            const uchar* row2 = access(i + 2, 0).first == noval ? nullptr : src_data + access(i + 2, 0).first * src_step;
            if (ksize == 3)
            {
                process3(data->anchor_x, left, right, data->delta, kernel, row0, row1, row2, dst_data + i * width * 4);
            }
            else
            {
                const uchar* row3 = access(i + 3, 0).first == noval ? nullptr : src_data + access(i + 3, 0).first * src_step;
                const uchar* row4 = access(i + 4, 0).first == noval ? nullptr : src_data + access(i + 4, 0).first * src_step;
                process5(data->anchor_x, left, right, data->delta, kernel, row0, row1, row2, row3, row4, dst_data + i * width * 4);
            }
        }
    }

    return CV_HAL_ERROR_OK;
}

} // anonymous

int filterInit(cvhalFilter2D** context, uchar* kernel_data, size_t kernel_step, int kernel_type, int kernel_width, int kernel_height, int /*max_width*/, int /*max_height*/, int src_type, int dst_type, int borderType, double delta, int anchor_x, int anchor_y, bool /*allowSubmatrix*/, bool /*allowInplace*/)
{
    if (kernel_type != CV_32FC1 || src_type != CV_8UC4 || dst_type != CV_8UC4)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if (kernel_width != kernel_height)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if (kernel_width != 3 && kernel_width != 5)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if ((borderType & ~BORDER_ISOLATED) == BORDER_WRAP)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    anchor_x = anchor_x < 0 ? kernel_width  / 2 : anchor_x;
    anchor_y = anchor_y < 0 ? kernel_height / 2 : anchor_y;
    *context = reinterpret_cast<cvhalFilter2D*>(new Filter2D{kernel_data, kernel_step, kernel_type, kernel_width, kernel_height, src_type, dst_type, borderType, delta, anchor_x, anchor_y});
    return CV_HAL_ERROR_OK;
}

int filter(cvhalFilter2D* context, uchar* src_data, size_t src_step, uchar* dst_data, size_t dst_step, int width, int height, int full_width, int full_height, int offset_x, int offset_y)
{
    Filter2D* data = reinterpret_cast<Filter2D*>(context);
    std::vector<uchar> dst(width * height * 4);

    int res = CV_HAL_ERROR_NOT_IMPLEMENTED;
    switch (data->kernel_width)
    {
    case 3:
        res = common::invoke(height, {filter<3>}, data, src_data, src_step, dst.data(), width, height, full_width, full_height, offset_x, offset_y);
        break;
    case 5:
        res = common::invoke(height, {filter<5>}, data, src_data, src_step, dst.data(), width, height, full_width, full_height, offset_x, offset_y);
        break;
    }

    for (int i = 0; i < height; i++)
        memcpy(dst_data + i * dst_step, dst.data() + i * width * 4, width * 4);
    return res;
}

int filterFree(cvhalFilter2D* context)
{
    delete reinterpret_cast<Filter2D*>(context);
    return CV_HAL_ERROR_OK;
}

#endif // CV_HAL_RVV_1P0_ENABLED

}}} // cv::rvv_hal::imgproc
