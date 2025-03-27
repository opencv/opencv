// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2025, Institute of Software, Chinese Academy of Sciences.

#ifndef OPENCV_HAL_RVV_FILTER_HPP_INCLUDED
#define OPENCV_HAL_RVV_FILTER_HPP_INCLUDED

#include <riscv_vector.h>

struct cvhalFilter2D;

namespace cv { namespace cv_hal_rvv {

namespace filter {
#undef cv_hal_filterInit
#undef cv_hal_filter
#undef cv_hal_filterFree
#define cv_hal_filterInit cv::cv_hal_rvv::filter::filterInit
#define cv_hal_filter cv::cv_hal_rvv::filter::filter
#define cv_hal_filterFree cv::cv_hal_rvv::filter::filterFree

class FilterInvoker : public ParallelLoopBody
{
public:
    template<typename... Args>
    FilterInvoker(std::function<int(int, int, Args...)> _func, Args&&... args)
    {
        func = std::bind(_func, std::placeholders::_1, std::placeholders::_2, std::forward<Args>(args)...);
    }

    virtual void operator()(const Range& range) const override
    {
        func(range.start, range.end);
    }

private:
    std::function<int(int, int)> func;
};

template<typename... Args>
static inline int invoke(int height, std::function<int(int, int, Args...)> func, Args&&... args)
{
    cv::parallel_for_(Range(1, height), FilterInvoker(func, std::forward<Args>(args)...), cv::getNumThreads());
    return func(0, 1, std::forward<Args>(args)...);
}

static inline int borderInterpolate( int p, int len, int borderType )
{
    if ((unsigned)p < (unsigned)len)
        ;
    else if (borderType == BORDER_REPLICATE)
        p = p < 0 ? 0 : len - 1;
    else if (borderType == BORDER_REFLECT || borderType == BORDER_REFLECT_101)
    {
        int delta = borderType == BORDER_REFLECT_101;
        if (len == 1)
            return 0;
        do
        {
            if (p < 0)
                p = -p - 1 + delta;
            else
                p = len - 1 - (p - len) - delta;
        }
        while( (unsigned)p >= (unsigned)len );
    }
    else if (borderType == BORDER_WRAP)
    {
        if (p < 0)
            p -= ((p-len+1)/len)*len;
        if (p >= len)
            p %= len;
    }
    else if (borderType == BORDER_CONSTANT)
        p = -1;
    return p;
}

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

inline int filterInit(cvhalFilter2D** context, uchar* kernel_data, size_t kernel_step, int kernel_type, int kernel_width, int kernel_height, int /*max_width*/, int /*max_height*/, int src_type, int dst_type, int borderType, double delta, int anchor_x, int anchor_y, bool /*allowSubmatrix*/, bool /*allowInplace*/)
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
            pi = borderInterpolate(x - data->anchor_y, height, data->borderType & ~BORDER_ISOLATED);
            pj = borderInterpolate(y - data->anchor_x, width , data->borderType & ~BORDER_ISOLATED);
            pi = pi < 0 ? noval : pi;
            pj = pj < 0 ? noval : pj;
        }
        else
        {
            pi = borderInterpolate(offset_y + x - data->anchor_y, full_height, data->borderType);
            pj = borderInterpolate(offset_x + y - data->anchor_x, full_width , data->borderType);
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

inline int filter(cvhalFilter2D* context, uchar* src_data, size_t src_step, uchar* dst_data, size_t dst_step, int width, int height, int full_width, int full_height, int offset_x, int offset_y)
{
    Filter2D* data = reinterpret_cast<Filter2D*>(context);
    std::vector<uchar> dst(width * height * 4);

    int res = CV_HAL_ERROR_NOT_IMPLEMENTED;
    switch (data->kernel_width)
    {
    case 3:
        res = invoke(height, {filter<3>}, data, src_data, src_step, dst.data(), width, height, full_width, full_height, offset_x, offset_y);
        break;
    case 5:
        res = invoke(height, {filter<5>}, data, src_data, src_step, dst.data(), width, height, full_width, full_height, offset_x, offset_y);
        break;
    }

    for (int i = 0; i < height; i++)
        memcpy(dst_data + i * dst_step, dst.data() + i * width * 4, width * 4);
    return res;
}

inline int filterFree(cvhalFilter2D* context)
{
    delete reinterpret_cast<Filter2D*>(context);
    return CV_HAL_ERROR_OK;
}
} // cv::cv_hal_rvv::filter

namespace sepFilter {
#undef cv_hal_sepFilterInit
#undef cv_hal_sepFilter
#undef cv_hal_sepFilterFree
#define cv_hal_sepFilterInit cv::cv_hal_rvv::sepFilter::sepFilterInit
#define cv_hal_sepFilter cv::cv_hal_rvv::sepFilter::sepFilter
#define cv_hal_sepFilterFree cv::cv_hal_rvv::sepFilter::sepFilterFree

struct sepFilter2D
{
    int src_type;
    int dst_type;
    int kernel_type;
    const uchar* kernelx_data;
    int kernelx_length;
    const uchar* kernely_data;
    int kernely_length;
    int anchor_x;
    int anchor_y;
    double delta;
    int borderType;
};

inline int sepFilterInit(cvhalFilter2D **context, int src_type, int dst_type, int kernel_type, uchar* kernelx_data, int kernelx_length, uchar* kernely_data, int kernely_length, int anchor_x, int anchor_y, double delta, int borderType)
{
    if (kernel_type != CV_32FC1)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if (src_type != CV_8UC1 && src_type != CV_16SC1 && src_type != CV_32FC1)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if (dst_type != CV_16SC1 && dst_type != CV_32FC1)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if ((kernelx_length != 3 && kernelx_length != 5) || kernelx_length != kernely_length)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if ((borderType & ~BORDER_ISOLATED) == BORDER_WRAP)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    anchor_x = anchor_x < 0 ? kernelx_length / 2 : anchor_x;
    anchor_y = anchor_y < 0 ? kernely_length / 2 : anchor_y;
    *context = reinterpret_cast<cvhalFilter2D*>(new sepFilter2D{src_type, dst_type, kernel_type, kernelx_data, kernelx_length, kernely_data, kernely_length, anchor_x, anchor_y, delta, borderType & ~BORDER_ISOLATED});
    return CV_HAL_ERROR_OK;
}

// the algorithm is copied from 3rdparty/carotene/src/separable_filter.hpp,
// in the functor RowFilter3x3S16Generic and ColFilter3x3S16Generic
template<int ksize, typename T>
static inline int sepFilter(int start, int end, sepFilter2D* data, const uchar* src_data, size_t src_step, uchar* dst_data, size_t dst_step, int width, int height, int full_width, int full_height, int offset_x, int offset_y)
{
    constexpr int noval = std::numeric_limits<int>::max();
    auto accessX = [&](int x) {
        int pi;
        if (data->borderType & BORDER_ISOLATED)
        {
            pi = filter::borderInterpolate(x - data->anchor_y, height, data->borderType & ~BORDER_ISOLATED);
            pi = pi < 0 ? noval : pi;
        }
        else
        {
            pi = filter::borderInterpolate(offset_y + x - data->anchor_y, full_height, data->borderType);
            pi = pi < 0 ? noval : pi - offset_y;
        }
        return pi;
    };
    auto accessY = [&](int y) {
        int pj;
        if (data->borderType & BORDER_ISOLATED)
        {
            pj = filter::borderInterpolate(y - data->anchor_x, width, data->borderType & ~BORDER_ISOLATED);
            pj = pj < 0 ? noval : pj;
        }
        else
        {
            pj = filter::borderInterpolate(offset_x + y - data->anchor_x, full_width, data->borderType);
            pj = pj < 0 ? noval : pj - offset_x;
        }
        return pj;
    };
    auto p2idx = [&](int x, int y){ return (x + ksize) % ksize * width + y; };

    const float* kx = reinterpret_cast<const float*>(data->kernelx_data);
    const float* ky = reinterpret_cast<const float*>(data->kernely_data);
    std::vector<float> res(width * ksize);
    auto process = [&](int x, int y) {
        float sum = 0;
        for (int i = 0; i < ksize; i++)
        {
            int p = accessY(y + i);
            if (p != noval)
            {
                sum += kx[i] * reinterpret_cast<const T*>(src_data + x * src_step)[p];
            }
        }
        res[p2idx(x, y)] = sum;
    };

    const int left = data->anchor_x, right = width - (ksize - 1 - data->anchor_x);
    for (int i = start - data->anchor_y; i < end + (ksize - 1 - data->anchor_y); i++)
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
                    vl = __riscv_vsetvl_e8m2(right - j);
                    const T* extra = reinterpret_cast<const T*>(src_data + i * src_step) + j - data->anchor_x;
                    vfloat32m8_t src;
                    if (std::is_same<T, uchar>::value)
                    {
                        src = __riscv_vfwcvt_f(__riscv_vwcvtu_x(__riscv_vle8_v_u8m2(reinterpret_cast<const uchar*>(extra), vl), vl), vl);
                    }
                    else if (std::is_same<T, short>::value)
                    {
                        src = __riscv_vfwcvt_f(__riscv_vle16_v_i16m4(reinterpret_cast<const short*>(extra), vl), vl);
                    }
                    else
                    {
                        src = __riscv_vle32_v_f32m8(reinterpret_cast<const float*>(extra), vl);
                    }

                    extra += vl;
                    auto sum = __riscv_vfmul(src, kx[0], vl);
                    src = __riscv_vfslide1down(src, extra[0], vl);
                    sum = __riscv_vfmacc(sum, kx[1], src, vl);
                    src = __riscv_vfslide1down(src, extra[1], vl);
                    sum = __riscv_vfmacc(sum, kx[2], src, vl);
                    if (ksize == 5)
                    {
                        src = __riscv_vfslide1down(src, extra[2], vl);
                        sum = __riscv_vfmacc(sum, kx[3], src, vl);
                        src = __riscv_vfslide1down(src, extra[3], vl);
                        sum = __riscv_vfmacc(sum, kx[4], src, vl);
                    }
                    __riscv_vse32(res.data() + p2idx(i, j), sum, vl);
                }
            }
        }

        int cur = i - (ksize - 1 - data->anchor_y);
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
                vl = __riscv_vsetvl_e32m4(width - j);
                auto v0 = row0 ? __riscv_vle32_v_f32m4(row0 + j, vl) : __riscv_vfmv_v_f_f32m4(0, vl);
                auto v1 = row1 ? __riscv_vle32_v_f32m4(row1 + j, vl) : __riscv_vfmv_v_f_f32m4(0, vl);
                auto v2 = row2 ? __riscv_vle32_v_f32m4(row2 + j, vl) : __riscv_vfmv_v_f_f32m4(0, vl);
                auto sum = __riscv_vfmacc(__riscv_vfmacc(__riscv_vfmacc(__riscv_vfmv_v_f_f32m4(data->delta, vl), ky[0], v0, vl), ky[1], v1, vl), ky[2], v2, vl);

                if (ksize == 5)
                {
                    auto v3 = row3 ? __riscv_vle32_v_f32m4(row3 + j, vl) : __riscv_vfmv_v_f_f32m4(0, vl);
                    auto v4 = row4 ? __riscv_vle32_v_f32m4(row4 + j, vl) : __riscv_vfmv_v_f_f32m4(0, vl);
                    sum = __riscv_vfmacc(__riscv_vfmacc(sum, ky[3], v3, vl), ky[4], v4, vl);
                }

                if (data->dst_type == CV_16SC1)
                {
                    __riscv_vse16(reinterpret_cast<short*>(dst_data + cur * dst_step) + j, __riscv_vfncvt_x(sum, vl), vl);
                }
                else
                {
                    __riscv_vse32(reinterpret_cast<float*>(dst_data + cur * dst_step) + j, sum, vl);
                }
            }
        }
    }

    return CV_HAL_ERROR_OK;
}

inline int sepFilter(cvhalFilter2D *context, uchar* src_data, size_t src_step, uchar* dst_data, size_t dst_step, int width, int height, int full_width, int full_height, int offset_x, int offset_y)
{
    sepFilter2D* data = reinterpret_cast<sepFilter2D*>(context);

    uchar* _dst_data = dst_data;
    size_t _dst_step = dst_step;
    const size_t size = CV_ELEM_SIZE(data->dst_type);
    std::vector<uchar> dst;
    if (src_data == _dst_data)
    {
        dst = std::vector<uchar>(width * height * size);
        dst_data = dst.data();
        dst_step = width * size;
    }

    int res = CV_HAL_ERROR_NOT_IMPLEMENTED;
    switch (data->kernelx_length*100 + data->src_type)
    {
    case 300 + CV_8UC1:
        res = filter::invoke(height, {sepFilter<3, uchar>}, data, src_data, src_step, dst_data, dst_step, width, height, full_width, full_height, offset_x, offset_y);
        break;
    case 500 + CV_8UC1:
        res = filter::invoke(height, {sepFilter<5, uchar>}, data, src_data, src_step, dst_data, dst_step, width, height, full_width, full_height, offset_x, offset_y);
        break;
    case 300 + CV_16SC1:
        res = filter::invoke(height, {sepFilter<3, short>}, data, src_data, src_step, dst_data, dst_step, width, height, full_width, full_height, offset_x, offset_y);
        break;
    case 500 + CV_16SC1:
        res = filter::invoke(height, {sepFilter<5, short>}, data, src_data, src_step, dst_data, dst_step, width, height, full_width, full_height, offset_x, offset_y);
        break;
    case 300 + CV_32FC1:
        res = filter::invoke(height, {sepFilter<3, float>}, data, src_data, src_step, dst_data, dst_step, width, height, full_width, full_height, offset_x, offset_y);
        break;
    case 500 + CV_32FC1:
        res = filter::invoke(height, {sepFilter<5, float>}, data, src_data, src_step, dst_data, dst_step, width, height, full_width, full_height, offset_x, offset_y);
        break;
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

inline int sepFilterFree(cvhalFilter2D* context)
{
    delete reinterpret_cast<sepFilter2D*>(context);
    return CV_HAL_ERROR_OK;
}
} // cv::cv_hal_rvv::sepFilter

namespace morph {
#undef cv_hal_morphInit
#undef cv_hal_morph
#undef cv_hal_morphFree
#define cv_hal_morphInit cv::cv_hal_rvv::morph::morphInit
#define cv_hal_morph cv::cv_hal_rvv::morph::morph
#define cv_hal_morphFree cv::cv_hal_rvv::morph::morphFree

struct Morph2D
{
    int operation;
    int src_type;
    int dst_type;
    int kernel_type;
    uchar* kernel_data;
    size_t kernel_step;
    int kernel_width;
    int kernel_height;
    int anchor_x;
    int anchor_y;
    int borderType;
    const uchar* borderValue;
};

inline int morphInit(cvhalFilter2D** context, int operation, int src_type, int dst_type, int /*max_width*/, int /*max_height*/, int kernel_type, uchar* kernel_data, size_t kernel_step, int kernel_width, int kernel_height, int anchor_x, int anchor_y, int borderType, const double borderValue[4], int iterations, bool /*allowSubmatrix*/, bool /*allowInplace*/)
{
    if (kernel_type != CV_8UC1 || src_type != dst_type)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if (src_type != CV_8UC1 && src_type != CV_8UC4)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if (kernel_width != kernel_height || kernel_width != 3)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if (iterations != 1)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if (operation != CV_HAL_MORPH_ERODE && operation != CV_HAL_MORPH_DILATE)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if ((borderType & ~BORDER_ISOLATED) == BORDER_WRAP)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    uchar* borderV;
    if (src_type == CV_8UC1)
    {
        borderV = new uchar{static_cast<uchar>(borderValue[0])};
        if (operation == CV_HAL_MORPH_DILATE && borderValue[0] == DBL_MAX)
            borderV[0] = 0;
    }
    else
    {
        borderV = new uchar[4]{static_cast<uchar>(borderValue[0]), static_cast<uchar>(borderValue[1]), static_cast<uchar>(borderValue[2]), static_cast<uchar>(borderValue[3])};
        if (operation == CV_HAL_MORPH_DILATE)
        {
            if (borderValue[0] == DBL_MAX)
                borderV[0] = 0;
            if (borderValue[1] == DBL_MAX)
                borderV[1] = 0;
            if (borderValue[2] == DBL_MAX)
                borderV[2] = 0;
            if (borderValue[3] == DBL_MAX)
                borderV[3] = 0;
        }
    }

    anchor_x = anchor_x < 0 ? kernel_width  / 2 : anchor_x;
    anchor_y = anchor_y < 0 ? kernel_height / 2 : anchor_y;
    *context = reinterpret_cast<cvhalFilter2D*>(new Morph2D{operation, src_type, dst_type, kernel_type, kernel_data, kernel_step, kernel_width, kernel_height, anchor_x, anchor_y, borderType, borderV});
    return CV_HAL_ERROR_OK;
}

template<int op> struct rvv;
template<> struct rvv<CV_HAL_MORPH_ERODE>
{
    static inline uchar init() { return std::numeric_limits<uchar>::max(); }
    static inline uchar mop(uchar a, uchar b) { return a < b ? a : b; }
    static inline vuint8m4_t vop(vuint8m4_t a, vuint8m4_t b, size_t c) { return __riscv_vminu(a, b, c); }
    static inline vuint8m4_t vop(vuint8m4_t a, uchar b, size_t c) { return __riscv_vminu(a, b, c); }
};
template<> struct rvv<CV_HAL_MORPH_DILATE>
{
    static inline uchar init() { return std::numeric_limits<uchar>::min(); }
    static inline uchar mop(uchar a, uchar b) { return a > b ? a : b; }
    static inline vuint8m4_t vop(vuint8m4_t a, vuint8m4_t b, size_t c) { return __riscv_vmaxu(a, b, c); }
    static inline vuint8m4_t vop(vuint8m4_t a, uchar b, size_t c) { return __riscv_vmaxu(a, b, c); }
};

// the algorithm is copied from 3rdparty/carotene/src/morph.cpp,
// in the function template void morph3x3
template<int op>
static inline int morph(int start, int end, Morph2D* data, const uchar* src_data, size_t src_step, uchar* dst_data, int width, int height, int full_width, int full_height, int offset_x, int offset_y)
{
    bool kernel[9];
    for (int i = 0; i < 9; i++)
    {
        kernel[i] = data->kernel_data[(i / 3) * data->kernel_step + i % 3] != 0;
    }

    constexpr int noval = std::numeric_limits<int>::max();
    auto access = [&](int x, int y) {
        int pi, pj;
        if (data->borderType & BORDER_ISOLATED)
        {
            pi = filter::borderInterpolate(x - data->anchor_y, height, data->borderType & ~BORDER_ISOLATED);
            pj = filter::borderInterpolate(y - data->anchor_x, width , data->borderType & ~BORDER_ISOLATED);
            pi = pi < 0 ? noval : pi;
            pj = pj < 0 ? noval : pj;
        }
        else
        {
            pi = filter::borderInterpolate(offset_y + x - data->anchor_y, full_height, data->borderType);
            pj = filter::borderInterpolate(offset_x + y - data->anchor_x, full_width , data->borderType);
            pi = pi < 0 ? noval : pi - offset_y;
            pj = pj < 0 ? noval : pj - offset_x;
        }
        return std::make_pair(pi, pj);
    };

    auto process = [&](int x, int y) {
        if (data->src_type == CV_8UC1)
        {
            uchar val = rvv<op>::init();
            for (int i = 0; i < 9; i++)
            {
                if (kernel[i])
                {
                    auto p = access(x + i / 3, y + i % 3);
                    if (p.first != noval && p.second != noval)
                    {
                        val = rvv<op>::mop(val, src_data[p.first * src_step + p.second]);
                    }
                    else
                    {
                        val = rvv<op>::mop(val, data->borderValue[0]);
                    }
                }
            }
            dst_data[x * width + y] = val;
        }
        else
        {
            uchar val0, val1, val2, val3;
            val0 = val1 = val2 = val3 = rvv<op>::init();
            for (int i = 0; i < 9; i++)
            {
                if (kernel[i])
                {
                    auto p = access(x + i / 3, y + i % 3);
                    if (p.first != noval && p.second != noval)
                    {
                        val0 = rvv<op>::mop(val0, src_data[p.first * src_step + p.second * 4    ]);
                        val1 = rvv<op>::mop(val1, src_data[p.first * src_step + p.second * 4 + 1]);
                        val2 = rvv<op>::mop(val2, src_data[p.first * src_step + p.second * 4 + 2]);
                        val3 = rvv<op>::mop(val3, src_data[p.first * src_step + p.second * 4 + 3]);
                    }
                    else
                    {
                        val0 = rvv<op>::mop(val0, data->borderValue[0]);
                        val1 = rvv<op>::mop(val1, data->borderValue[1]);
                        val2 = rvv<op>::mop(val2, data->borderValue[2]);
                        val3 = rvv<op>::mop(val3, data->borderValue[3]);
                    }
                }
            }
            dst_data[(x * width + y) * 4    ] = val0;
            dst_data[(x * width + y) * 4 + 1] = val1;
            dst_data[(x * width + y) * 4 + 2] = val2;
            dst_data[(x * width + y) * 4 + 3] = val3;
        }
    };

    const int left = data->anchor_x, right = width - (2 - data->anchor_x);
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
            if (data->src_type == CV_8UC1)
            {
                int vl;
                for (int j = left; j < right; j += vl)
                {
                    vl = __riscv_vsetvl_e8m4(right - j);
                    auto m0 = __riscv_vmv_v_x_u8m4(rvv<op>::init(), vl);
                    auto loadsrc = [&](const uchar* row, bool k0, bool k1, bool k2) {
                        if (!row)
                        {
                            m0 = rvv<op>::vop(m0, data->borderValue[0], vl);
                            return;
                        }

                        const uchar* extra = row + j - data->anchor_x;
                        auto v0 = __riscv_vle8_v_u8m4(extra, vl);

                        if (k0) m0 = rvv<op>::vop(m0, v0, vl);
                        v0 = __riscv_vslide1down(v0, extra[vl], vl);
                        if (k1) m0 = rvv<op>::vop(m0, v0, vl);
                        if (!k2) return;
                        v0 = __riscv_vslide1down(v0, extra[vl + 1], vl);
                        m0 = rvv<op>::vop(m0, v0, vl);
                    };

                    loadsrc(row0, kernel[0], kernel[1], kernel[2]);
                    loadsrc(row1, kernel[3], kernel[4], kernel[5]);
                    loadsrc(row2, kernel[6], kernel[7], kernel[8]);
                    __riscv_vse8(dst_data + i * width + j, m0, vl);
                }
            }
            else
            {
                int vl, vl0, vl1;
                for (int j = left; j < right; j += vl)
                {
                    vl = __riscv_vsetvl_e8m4(right - j);
                    vl0 = std::min(vl, (int)__riscv_vlenb() * 2);
                    vl1 = vl - vl0;
                    auto m0 = __riscv_vmv_v_x_u8m4(rvv<op>::init(), vl);
                    auto m1 = __riscv_vmv_v_x_u8m4(rvv<op>::init(), vl);
                    auto m2 = __riscv_vmv_v_x_u8m4(rvv<op>::init(), vl);
                    auto m3 = __riscv_vmv_v_x_u8m4(rvv<op>::init(), vl);

                    auto opshift = [&](vuint8m4_t a, vuint8m4_t b, bool k0, bool k1, bool k2, uchar r1, uchar r2) {
                        if (k0) a = rvv<op>::vop(a, b, vl);
                        b = __riscv_vslide1down(b, r1, vl);
                        if (k1) a = rvv<op>::vop(a, b, vl);
                        if (!k2) return a;
                        b = __riscv_vslide1down(b, r2, vl);
                        return rvv<op>::vop(a, b, vl);
                    };
                    auto loadsrc = [&](const uchar* row, bool k0, bool k1, bool k2) {
                        if (!row)
                        {
                            m0 = rvv<op>::vop(m0, data->borderValue[0], vl);
                            m1 = rvv<op>::vop(m1, data->borderValue[1], vl);
                            m2 = rvv<op>::vop(m2, data->borderValue[2], vl);
                            m3 = rvv<op>::vop(m3, data->borderValue[3], vl);
                            return;
                        }

                        vuint8m4_t v0{}, v1{}, v2{}, v3{};
                        const uchar* extra = row + (j - data->anchor_x) * 4;
                        auto src = __riscv_vlseg4e8_v_u8m2x4(extra, vl0);
                        v0 = __riscv_vset_v_u8m2_u8m4(v0, 0, __riscv_vget_v_u8m2x4_u8m2(src, 0));
                        v1 = __riscv_vset_v_u8m2_u8m4(v1, 0, __riscv_vget_v_u8m2x4_u8m2(src, 1));
                        v2 = __riscv_vset_v_u8m2_u8m4(v2, 0, __riscv_vget_v_u8m2x4_u8m2(src, 2));
                        v3 = __riscv_vset_v_u8m2_u8m4(v3, 0, __riscv_vget_v_u8m2x4_u8m2(src, 3));
                        src = __riscv_vlseg4e8_v_u8m2x4(extra + vl0 * 4, vl1);
                        v0 = __riscv_vset_v_u8m2_u8m4(v0, 1, __riscv_vget_v_u8m2x4_u8m2(src, 0));
                        v1 = __riscv_vset_v_u8m2_u8m4(v1, 1, __riscv_vget_v_u8m2x4_u8m2(src, 1));
                        v2 = __riscv_vset_v_u8m2_u8m4(v2, 1, __riscv_vget_v_u8m2x4_u8m2(src, 2));
                        v3 = __riscv_vset_v_u8m2_u8m4(v3, 1, __riscv_vget_v_u8m2x4_u8m2(src, 3));

                        extra += vl * 4;
                        m0 = opshift(m0, v0, k0, k1, k2, extra[0], extra[4]);
                        m1 = opshift(m1, v1, k0, k1, k2, extra[1], extra[5]);
                        m2 = opshift(m2, v2, k0, k1, k2, extra[2], extra[6]);
                        m3 = opshift(m3, v3, k0, k1, k2, extra[3], extra[7]);
                    };

                    loadsrc(row0, kernel[0], kernel[1], kernel[2]);
                    loadsrc(row1, kernel[3], kernel[4], kernel[5]);
                    loadsrc(row2, kernel[6], kernel[7], kernel[8]);
                    vuint8m2x4_t val{};
                    val = __riscv_vset_v_u8m2_u8m2x4(val, 0, __riscv_vget_v_u8m4_u8m2(m0, 0));
                    val = __riscv_vset_v_u8m2_u8m2x4(val, 1, __riscv_vget_v_u8m4_u8m2(m1, 0));
                    val = __riscv_vset_v_u8m2_u8m2x4(val, 2, __riscv_vget_v_u8m4_u8m2(m2, 0));
                    val = __riscv_vset_v_u8m2_u8m2x4(val, 3, __riscv_vget_v_u8m4_u8m2(m3, 0));
                    __riscv_vsseg4e8(dst_data + (i * width + j) * 4, val, vl0);
                    val = __riscv_vset_v_u8m2_u8m2x4(val, 0, __riscv_vget_v_u8m4_u8m2(m0, 1));
                    val = __riscv_vset_v_u8m2_u8m2x4(val, 1, __riscv_vget_v_u8m4_u8m2(m1, 1));
                    val = __riscv_vset_v_u8m2_u8m2x4(val, 2, __riscv_vget_v_u8m4_u8m2(m2, 1));
                    val = __riscv_vset_v_u8m2_u8m2x4(val, 3, __riscv_vget_v_u8m4_u8m2(m3, 1));
                    __riscv_vsseg4e8(dst_data + (i * width + j + vl0) * 4, val, vl1);
                }
            }
        }
    }

    return CV_HAL_ERROR_OK;
}

inline int morph(cvhalFilter2D* context, uchar* src_data, size_t src_step, uchar* dst_data, size_t dst_step, int width, int height, int src_full_width, int src_full_height, int src_roi_x, int src_roi_y, int /*dst_full_width*/, int /*dst_full_height*/, int /*dst_roi_x*/, int /*dst_roi_y*/)
{
    Morph2D* data = reinterpret_cast<Morph2D*>(context);
    int cn = data->src_type == CV_8UC1 ? 1 : 4;
    std::vector<uchar> dst(width * height * cn);

    int res = CV_HAL_ERROR_NOT_IMPLEMENTED;
    switch (data->operation)
    {
    case CV_HAL_MORPH_ERODE:
        res = filter::invoke(height, {morph<CV_HAL_MORPH_ERODE>}, data, src_data, src_step, dst.data(), width, height, src_full_width, src_full_height, src_roi_x, src_roi_y);
        break;
    case CV_HAL_MORPH_DILATE:
        res = filter::invoke(height, {morph<CV_HAL_MORPH_DILATE>}, data, src_data, src_step, dst.data(), width, height, src_full_width, src_full_height, src_roi_x, src_roi_y);
        break;
    }

    for (int i = 0; i < height; i++)
        memcpy(dst_data + i * dst_step, dst.data() + i * width * cn, width * cn);
    return res;
}

inline int morphFree(cvhalFilter2D* context)
{
    delete reinterpret_cast<Morph2D*>(context)->borderValue;
    delete reinterpret_cast<Morph2D*>(context);
    return CV_HAL_ERROR_OK;
}
} // cv::cv_hal_rvv::morph

namespace gaussianBlurBinomial {
#undef cv_hal_gaussianBlurBinomial
#define cv_hal_gaussianBlurBinomial cv::cv_hal_rvv::gaussianBlurBinomial::gaussianBlurBinomial

// the algorithm is same as cv_hal_sepFilter
template<int ksize, typename helperT, typename helperWT>
static inline int gaussianBlurC1(int start, int end, const uchar* src_data, size_t src_step, uchar* dst_data, size_t dst_step, int width, int full_width, int full_height, int offset_x, int offset_y, int border_type)
{
    using T = typename helperT::ElemType;
    using WT = typename helperWT::ElemType;

    constexpr int noval = std::numeric_limits<int>::max();
    auto accessX = [&](int x) {
        int pi = filter::borderInterpolate(offset_y + x - ksize / 2, full_height, border_type);
        return pi < 0 ? noval : pi - offset_y;
    };
    auto accessY = [&](int y) {
        int pj = filter::borderInterpolate(offset_x + y - ksize / 2, full_width, border_type);
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
        int pi = filter::borderInterpolate(offset_y + x - ksize / 2, full_height, border_type);
        return pi < 0 ? noval : pi - offset_y;
    };
    auto accessY = [&](int y) {
        int pj = filter::borderInterpolate(offset_x + y - ksize / 2, full_width, border_type);
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

inline int gaussianBlurBinomial(const uchar* src_data, size_t src_step, uchar* dst_data, size_t dst_step, int width, int height, int depth, int cn, size_t margin_left, size_t margin_top, size_t margin_right, size_t margin_bottom, size_t ksize, int border_type)
{
    const int type = CV_MAKETYPE(depth, cn);
    if ((type != CV_8UC1 && type != CV_8UC4 && type != CV_16UC1) || src_data == dst_data)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if ((ksize != 3 && ksize != 5) || border_type & BORDER_ISOLATED || border_type == BORDER_WRAP)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    switch (ksize*100 + type)
    {
    case 300 + CV_8UC1:
        return filter::invoke(height, {gaussianBlurC1<3, RVV_U8M4, RVV_U16M8>}, src_data, src_step, dst_data, dst_step, width, margin_left + width + margin_right, margin_top + height + margin_bottom, margin_left, margin_top, border_type);
    case 500 + CV_8UC1:
        return filter::invoke(height, {gaussianBlurC1<5, RVV_U8M4, RVV_U16M8>}, src_data, src_step, dst_data, dst_step, width, margin_left + width + margin_right, margin_top + height + margin_bottom, margin_left, margin_top, border_type);
    case 300 + CV_16UC1:
        return filter::invoke(height, {gaussianBlurC1<3, RVV_U16M4, RVV_U32M8>}, src_data, src_step, dst_data, dst_step, width, margin_left + width + margin_right, margin_top + height + margin_bottom, margin_left, margin_top, border_type);
    case 500 + CV_16UC1:
        return filter::invoke(height, {gaussianBlurC1<5, RVV_U16M4, RVV_U32M8>}, src_data, src_step, dst_data, dst_step, width, margin_left + width + margin_right, margin_top + height + margin_bottom, margin_left, margin_top, border_type);
    case 300 + CV_8UC4:
        return filter::invoke(height, {gaussianBlurC4<3>}, src_data, src_step, dst_data, dst_step, width, margin_left + width + margin_right, margin_top + height + margin_bottom, margin_left, margin_top, border_type);
    case 500 + CV_8UC4:
        return filter::invoke(height, {gaussianBlurC4<5>}, src_data, src_step, dst_data, dst_step, width, margin_left + width + margin_right, margin_top + height + margin_bottom, margin_left, margin_top, border_type);
    }

    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}
} // cv::cv_hal_rvv::gaussianBlurBinomial

namespace medianBlur {
#undef cv_hal_medianBlur
#define cv_hal_medianBlur cv::cv_hal_rvv::medianBlur::medianBlur

// the algorithm is copied from imgproc/src/median_blur.simd.cpp
// in the function template static void medianBlur_SortNet
template<int ksize, typename helper>
static inline int medianBlurC1(int start, int end, const uchar* src_data, size_t src_step, uchar* dst_data, size_t dst_step, int width, int height)
{
    using T = typename helper::ElemType;
    using VT = typename helper::VecType;

    for (int i = start; i < end; i++)
    {
        const T* row0 = reinterpret_cast<const T*>(src_data + std::min(std::max(i     - ksize / 2, 0), height - 1) * src_step);
        const T* row1 = reinterpret_cast<const T*>(src_data + std::min(std::max(i + 1 - ksize / 2, 0), height - 1) * src_step);
        const T* row2 = reinterpret_cast<const T*>(src_data + std::min(std::max(i + 2 - ksize / 2, 0), height - 1) * src_step);
        const T* row3 = reinterpret_cast<const T*>(src_data + std::min(std::max(i + 3 - ksize / 2, 0), height - 1) * src_step);
        const T* row4 = reinterpret_cast<const T*>(src_data + std::min(std::max(i + 4 - ksize / 2, 0), height - 1) * src_step);
        int vl;
        auto vop = [&vl](VT& a, VT& b) {
            auto t = a;
            a = helper::vmin(a, b, vl);
            b = helper::vmax(t, b, vl);
        };

        for (int j = 0; j < width; j += vl)
        {
            vl = helper::setvl(width - j);
            if (ksize == 3)
            {
                VT p0, p1, p2;
                VT p3, p4, p5;
                VT p6, p7, p8;
                if (j != 0)
                {
                    p0 = helper::vload(row0 + j - 1, vl);
                    p3 = helper::vload(row1 + j - 1, vl);
                    p6 = helper::vload(row2 + j - 1, vl);
                }
                else
                {
                    p0 = helper::vslide1up(helper::vload(row0, vl), row0[0], vl);
                    p3 = helper::vslide1up(helper::vload(row1, vl), row1[0], vl);
                    p6 = helper::vslide1up(helper::vload(row2, vl), row2[0], vl);
                }
                p1 = helper::vslide1down(p0, row0[j + vl - 1], vl);
                p4 = helper::vslide1down(p3, row1[j + vl - 1], vl);
                p7 = helper::vslide1down(p6, row2[j + vl - 1], vl);
                p2 = helper::vslide1down(p1, row0[std::min(width - 1, j + vl)], vl);
                p5 = helper::vslide1down(p4, row1[std::min(width - 1, j + vl)], vl);
                p8 = helper::vslide1down(p7, row2[std::min(width - 1, j + vl)], vl);

                vop(p1, p2); vop(p4, p5); vop(p7, p8); vop(p0, p1);
                vop(p3, p4); vop(p6, p7); vop(p1, p2); vop(p4, p5);
                vop(p7, p8); vop(p0, p3); vop(p5, p8); vop(p4, p7);
                vop(p3, p6); vop(p1, p4); vop(p2, p5); vop(p4, p7);
                vop(p4, p2); vop(p6, p4); vop(p4, p2);
                helper::vstore(reinterpret_cast<T*>(dst_data + i * dst_step) + j, p4, vl);
            }
            else
            {
                VT p0, p1, p2, p3, p4;
                VT p5, p6, p7, p8, p9;
                VT p10, p11, p12, p13, p14;
                VT p15, p16, p17, p18, p19;
                VT p20, p21, p22, p23, p24;
                if (j >= 2)
                {
                    p0 = helper::vload(row0 + j - 2, vl);
                    p5 = helper::vload(row1 + j - 2, vl);
                    p10 = helper::vload(row2 + j - 2, vl);
                    p15 = helper::vload(row3 + j - 2, vl);
                    p20 = helper::vload(row4 + j - 2, vl);
                }
                else
                {
                    p0 = helper::vslide1up(helper::vload(row0, vl), row0[0], vl);
                    p5 = helper::vslide1up(helper::vload(row1, vl), row1[0], vl);
                    p10 = helper::vslide1up(helper::vload(row2, vl), row2[0], vl);
                    p15 = helper::vslide1up(helper::vload(row3, vl), row3[0], vl);
                    p20 = helper::vslide1up(helper::vload(row4, vl), row4[0], vl);
                    if (j == 0)
                    {
                        p0 = helper::vslide1up(p0, row0[0], vl);
                        p5 = helper::vslide1up(p5, row1[0], vl);
                        p10 = helper::vslide1up(p10, row2[0], vl);
                        p15 = helper::vslide1up(p15, row3[0], vl);
                        p20 = helper::vslide1up(p20, row4[0], vl);
                    }
                }
                p1 = helper::vslide1down(p0, row0[j + vl - 2], vl);
                p6 = helper::vslide1down(p5, row1[j + vl - 2], vl);
                p11 = helper::vslide1down(p10, row2[j + vl - 2], vl);
                p16 = helper::vslide1down(p15, row3[j + vl - 2], vl);
                p21 = helper::vslide1down(p20, row4[j + vl - 2], vl);
                p2 = helper::vslide1down(p1, row0[j + vl - 1], vl);
                p7 = helper::vslide1down(p6, row1[j + vl - 1], vl);
                p12 = helper::vslide1down(p11, row2[j + vl - 1], vl);
                p17 = helper::vslide1down(p16, row3[j + vl - 1], vl);
                p22 = helper::vslide1down(p21, row4[j + vl - 1], vl);
                p3 = helper::vslide1down(p2, row0[std::min(width - 1, j + vl)], vl);
                p8 = helper::vslide1down(p7, row1[std::min(width - 1, j + vl)], vl);
                p13 = helper::vslide1down(p12, row2[std::min(width - 1, j + vl)], vl);
                p18 = helper::vslide1down(p17, row3[std::min(width - 1, j + vl)], vl);
                p23 = helper::vslide1down(p22, row4[std::min(width - 1, j + vl)], vl);
                p4 = helper::vslide1down(p3, row0[std::min(width - 1, j + vl + 1)], vl);
                p9 = helper::vslide1down(p8, row1[std::min(width - 1, j + vl + 1)], vl);
                p14 = helper::vslide1down(p13, row2[std::min(width - 1, j + vl + 1)], vl);
                p19 = helper::vslide1down(p18, row3[std::min(width - 1, j + vl + 1)], vl);
                p24 = helper::vslide1down(p23, row4[std::min(width - 1, j + vl + 1)], vl);

                vop(p1, p2); vop(p0, p1); vop(p1, p2); vop(p4, p5); vop(p3, p4);
                vop(p4, p5); vop(p0, p3); vop(p2, p5); vop(p2, p3); vop(p1, p4);
                vop(p1, p2); vop(p3, p4); vop(p7, p8); vop(p6, p7); vop(p7, p8);
                vop(p10, p11); vop(p9, p10); vop(p10, p11); vop(p6, p9); vop(p8, p11);
                vop(p8, p9); vop(p7, p10); vop(p7, p8); vop(p9, p10); vop(p0, p6);
                vop(p4, p10); vop(p4, p6); vop(p2, p8); vop(p2, p4); vop(p6, p8);
                vop(p1, p7); vop(p5, p11); vop(p5, p7); vop(p3, p9); vop(p3, p5);
                vop(p7, p9); vop(p1, p2); vop(p3, p4); vop(p5, p6); vop(p7, p8);
                vop(p9, p10); vop(p13, p14); vop(p12, p13); vop(p13, p14); vop(p16, p17);
                vop(p15, p16); vop(p16, p17); vop(p12, p15); vop(p14, p17); vop(p14, p15);
                vop(p13, p16); vop(p13, p14); vop(p15, p16); vop(p19, p20); vop(p18, p19);
                vop(p19, p20); vop(p21, p22); vop(p23, p24); vop(p21, p23); vop(p22, p24);
                vop(p22, p23); vop(p18, p21); vop(p20, p23); vop(p20, p21); vop(p19, p22);
                vop(p22, p24); vop(p19, p20); vop(p21, p22); vop(p23, p24); vop(p12, p18);
                vop(p16, p22); vop(p16, p18); vop(p14, p20); vop(p20, p24); vop(p14, p16);
                vop(p18, p20); vop(p22, p24); vop(p13, p19); vop(p17, p23); vop(p17, p19);
                vop(p15, p21); vop(p15, p17); vop(p19, p21); vop(p13, p14); vop(p15, p16);
                vop(p17, p18); vop(p19, p20); vop(p21, p22); vop(p23, p24); vop(p0, p12);
                vop(p8, p20); vop(p8, p12); vop(p4, p16); vop(p16, p24); vop(p12, p16);
                vop(p2, p14); vop(p10, p22); vop(p10, p14); vop(p6, p18); vop(p6, p10);
                vop(p10, p12); vop(p1, p13); vop(p9, p21); vop(p9, p13); vop(p5, p17);
                vop(p13, p17); vop(p3, p15); vop(p11, p23); vop(p11, p15); vop(p7, p19);
                vop(p7, p11); vop(p11, p13); vop(p11, p12);
                helper::vstore(reinterpret_cast<T*>(dst_data + i * dst_step) + j, p12, vl);
            }
        }
    }

    return CV_HAL_ERROR_OK;
}

template<int ksize>
static inline int medianBlurC4(int start, int end, const uchar* src_data, size_t src_step, uchar* dst_data, size_t dst_step, int width, int height)
{
    for (int i = start; i < end; i++)
    {
        const uchar* row0 = src_data + std::min(std::max(i     - ksize / 2, 0), height - 1) * src_step;
        const uchar* row1 = src_data + std::min(std::max(i + 1 - ksize / 2, 0), height - 1) * src_step;
        const uchar* row2 = src_data + std::min(std::max(i + 2 - ksize / 2, 0), height - 1) * src_step;
        const uchar* row3 = src_data + std::min(std::max(i + 3 - ksize / 2, 0), height - 1) * src_step;
        const uchar* row4 = src_data + std::min(std::max(i + 4 - ksize / 2, 0), height - 1) * src_step;
        int vl;
        for (int j = 0; j < width; j += vl)
        {
            if (ksize == 3)
            {
                vl = __riscv_vsetvl_e8m1(width - j);
                vuint8m1_t p00, p01, p02;
                vuint8m1_t p03, p04, p05;
                vuint8m1_t p06, p07, p08;
                vuint8m1_t p10, p11, p12;
                vuint8m1_t p13, p14, p15;
                vuint8m1_t p16, p17, p18;
                vuint8m1_t p20, p21, p22;
                vuint8m1_t p23, p24, p25;
                vuint8m1_t p26, p27, p28;
                vuint8m1_t p30, p31, p32;
                vuint8m1_t p33, p34, p35;
                vuint8m1_t p36, p37, p38;
                auto loadsrc = [&vl](const uchar* row, vuint8m1_t& p0, vuint8m1_t& p1, vuint8m1_t& p2, vuint8m1_t& p3) {
                    auto src = __riscv_vlseg4e8_v_u8m1x4(row, vl);
                    p0 = __riscv_vget_v_u8m1x4_u8m1(src, 0);
                    p1 = __riscv_vget_v_u8m1x4_u8m1(src, 1);
                    p2 = __riscv_vget_v_u8m1x4_u8m1(src, 2);
                    p3 = __riscv_vget_v_u8m1x4_u8m1(src, 3);
                };
                if (j != 0)
                {
                    loadsrc(row0 + (j - 1) * 4, p00, p10, p20, p30);
                    loadsrc(row1 + (j - 1) * 4, p03, p13, p23, p33);
                    loadsrc(row2 + (j - 1) * 4, p06, p16, p26, p36);
                }
                else
                {
                    loadsrc(row0, p00, p10, p20, p30);
                    loadsrc(row1, p03, p13, p23, p33);
                    loadsrc(row2, p06, p16, p26, p36);
                    p00 = __riscv_vslide1up(p00, row0[0], vl);
                    p10 = __riscv_vslide1up(p10, row0[1], vl);
                    p20 = __riscv_vslide1up(p20, row0[2], vl);
                    p30 = __riscv_vslide1up(p30, row0[3], vl);
                    p03 = __riscv_vslide1up(p03, row1[0], vl);
                    p13 = __riscv_vslide1up(p13, row1[1], vl);
                    p23 = __riscv_vslide1up(p23, row1[2], vl);
                    p33 = __riscv_vslide1up(p33, row1[3], vl);
                    p06 = __riscv_vslide1up(p06, row2[0], vl);
                    p16 = __riscv_vslide1up(p16, row2[1], vl);
                    p26 = __riscv_vslide1up(p26, row2[2], vl);
                    p36 = __riscv_vslide1up(p36, row2[3], vl);
                }
                p01 = __riscv_vslide1down(p00, row0[(j + vl - 1) * 4    ], vl);
                p11 = __riscv_vslide1down(p10, row0[(j + vl - 1) * 4 + 1], vl);
                p21 = __riscv_vslide1down(p20, row0[(j + vl - 1) * 4 + 2], vl);
                p31 = __riscv_vslide1down(p30, row0[(j + vl - 1) * 4 + 3], vl);
                p04 = __riscv_vslide1down(p03, row1[(j + vl - 1) * 4    ], vl);
                p14 = __riscv_vslide1down(p13, row1[(j + vl - 1) * 4 + 1], vl);
                p24 = __riscv_vslide1down(p23, row1[(j + vl - 1) * 4 + 2], vl);
                p34 = __riscv_vslide1down(p33, row1[(j + vl - 1) * 4 + 3], vl);
                p07 = __riscv_vslide1down(p06, row2[(j + vl - 1) * 4    ], vl);
                p17 = __riscv_vslide1down(p16, row2[(j + vl - 1) * 4 + 1], vl);
                p27 = __riscv_vslide1down(p26, row2[(j + vl - 1) * 4 + 2], vl);
                p37 = __riscv_vslide1down(p36, row2[(j + vl - 1) * 4 + 3], vl);
                p02 = __riscv_vslide1down(p01, row0[std::min(width - 1, j + vl) * 4    ], vl);
                p12 = __riscv_vslide1down(p11, row0[std::min(width - 1, j + vl) * 4 + 1], vl);
                p22 = __riscv_vslide1down(p21, row0[std::min(width - 1, j + vl) * 4 + 2], vl);
                p32 = __riscv_vslide1down(p31, row0[std::min(width - 1, j + vl) * 4 + 3], vl);
                p05 = __riscv_vslide1down(p04, row1[std::min(width - 1, j + vl) * 4    ], vl);
                p15 = __riscv_vslide1down(p14, row1[std::min(width - 1, j + vl) * 4 + 1], vl);
                p25 = __riscv_vslide1down(p24, row1[std::min(width - 1, j + vl) * 4 + 2], vl);
                p35 = __riscv_vslide1down(p34, row1[std::min(width - 1, j + vl) * 4 + 3], vl);
                p08 = __riscv_vslide1down(p07, row2[std::min(width - 1, j + vl) * 4    ], vl);
                p18 = __riscv_vslide1down(p17, row2[std::min(width - 1, j + vl) * 4 + 1], vl);
                p28 = __riscv_vslide1down(p27, row2[std::min(width - 1, j + vl) * 4 + 2], vl);
                p38 = __riscv_vslide1down(p37, row2[std::min(width - 1, j + vl) * 4 + 3], vl);

                auto vop = [&vl](vuint8m1_t& a, vuint8m1_t& b) {
                    auto t = a;
                    a = __riscv_vminu(a, b, vl);
                    b = __riscv_vmaxu(t, b, vl);
                };
                vuint8m1x4_t dst{};
                vop(p01, p02); vop(p04, p05); vop(p07, p08); vop(p00, p01);
                vop(p03, p04); vop(p06, p07); vop(p01, p02); vop(p04, p05);
                vop(p07, p08); vop(p00, p03); vop(p05, p08); vop(p04, p07);
                vop(p03, p06); vop(p01, p04); vop(p02, p05); vop(p04, p07);
                vop(p04, p02); vop(p06, p04); vop(p04, p02);
                dst = __riscv_vset_v_u8m1_u8m1x4(dst, 0, p04);
                vop(p11, p12); vop(p14, p15); vop(p17, p18); vop(p10, p11);
                vop(p13, p14); vop(p16, p17); vop(p11, p12); vop(p14, p15);
                vop(p17, p18); vop(p10, p13); vop(p15, p18); vop(p14, p17);
                vop(p13, p16); vop(p11, p14); vop(p12, p15); vop(p14, p17);
                vop(p14, p12); vop(p16, p14); vop(p14, p12);
                dst = __riscv_vset_v_u8m1_u8m1x4(dst, 1, p14);
                vop(p21, p22); vop(p24, p25); vop(p27, p28); vop(p20, p21);
                vop(p23, p24); vop(p26, p27); vop(p21, p22); vop(p24, p25);
                vop(p27, p28); vop(p20, p23); vop(p25, p28); vop(p24, p27);
                vop(p23, p26); vop(p21, p24); vop(p22, p25); vop(p24, p27);
                vop(p24, p22); vop(p26, p24); vop(p24, p22);
                dst = __riscv_vset_v_u8m1_u8m1x4(dst, 2, p24);
                vop(p31, p32); vop(p34, p35); vop(p37, p38); vop(p30, p31);
                vop(p33, p34); vop(p36, p37); vop(p31, p32); vop(p34, p35);
                vop(p37, p38); vop(p30, p33); vop(p35, p38); vop(p34, p37);
                vop(p33, p36); vop(p31, p34); vop(p32, p35); vop(p34, p37);
                vop(p34, p32); vop(p36, p34); vop(p34, p32);
                dst = __riscv_vset_v_u8m1_u8m1x4(dst, 3, p34);
                __riscv_vsseg4e8(dst_data + i * dst_step + j * 4, dst, vl);
            }
            else
            {
                vl = __riscv_vsetvl_e8m2(width - j);
                vuint8m2_t p00, p01, p02, p03, p04;
                vuint8m2_t p05, p06, p07, p08, p09;
                vuint8m2_t p010, p011, p012, p013, p014;
                vuint8m2_t p015, p016, p017, p018, p019;
                vuint8m2_t p020, p021, p022, p023, p024;
                vuint8m2_t p10, p11, p12, p13, p14;
                vuint8m2_t p15, p16, p17, p18, p19;
                vuint8m2_t p110, p111, p112, p113, p114;
                vuint8m2_t p115, p116, p117, p118, p119;
                vuint8m2_t p120, p121, p122, p123, p124;
                vuint8m2_t p20, p21, p22, p23, p24;
                vuint8m2_t p25, p26, p27, p28, p29;
                vuint8m2_t p210, p211, p212, p213, p214;
                vuint8m2_t p215, p216, p217, p218, p219;
                vuint8m2_t p220, p221, p222, p223, p224;
                vuint8m2_t p30, p31, p32, p33, p34;
                vuint8m2_t p35, p36, p37, p38, p39;
                vuint8m2_t p310, p311, p312, p313, p314;
                vuint8m2_t p315, p316, p317, p318, p319;
                vuint8m2_t p320, p321, p322, p323, p324;
                auto loadsrc = [&vl](const uchar* row, vuint8m2_t& p0, vuint8m2_t& p1, vuint8m2_t& p2, vuint8m2_t& p3) {
                    auto src = __riscv_vlseg4e8_v_u8m2x4(row, vl);
                    p0 = __riscv_vget_v_u8m2x4_u8m2(src, 0);
                    p1 = __riscv_vget_v_u8m2x4_u8m2(src, 1);
                    p2 = __riscv_vget_v_u8m2x4_u8m2(src, 2);
                    p3 = __riscv_vget_v_u8m2x4_u8m2(src, 3);
                };
                if (j >= 2)
                {
                    loadsrc(row0 + (j - 2) * 4, p00, p10, p20, p30);
                    loadsrc(row1 + (j - 2) * 4, p05, p15, p25, p35);
                    loadsrc(row2 + (j - 2) * 4, p010, p110, p210, p310);
                    loadsrc(row3 + (j - 2) * 4, p015, p115, p215, p315);
                    loadsrc(row4 + (j - 2) * 4, p020, p120, p220, p320);
                }
                else
                {
                    loadsrc(row0, p00, p10, p20, p30);
                    loadsrc(row1, p05, p15, p25, p35);
                    loadsrc(row2, p010, p110, p210, p310);
                    loadsrc(row3, p015, p115, p215, p315);
                    loadsrc(row4, p020, p120, p220, p320);
                    auto slideup = [&] {
                        p00 = __riscv_vslide1up(p00, row0[0], vl);
                        p10 = __riscv_vslide1up(p10, row0[1], vl);
                        p20 = __riscv_vslide1up(p20, row0[2], vl);
                        p30 = __riscv_vslide1up(p30, row0[3], vl);
                        p05 = __riscv_vslide1up(p05, row1[0], vl);
                        p15 = __riscv_vslide1up(p15, row1[1], vl);
                        p25 = __riscv_vslide1up(p25, row1[2], vl);
                        p35 = __riscv_vslide1up(p35, row1[3], vl);
                        p010 = __riscv_vslide1up(p010, row2[0], vl);
                        p110 = __riscv_vslide1up(p110, row2[1], vl);
                        p210 = __riscv_vslide1up(p210, row2[2], vl);
                        p310 = __riscv_vslide1up(p310, row2[3], vl);
                        p015 = __riscv_vslide1up(p015, row3[0], vl);
                        p115 = __riscv_vslide1up(p115, row3[1], vl);
                        p215 = __riscv_vslide1up(p215, row3[2], vl);
                        p315 = __riscv_vslide1up(p315, row3[3], vl);
                        p020 = __riscv_vslide1up(p020, row4[0], vl);
                        p120 = __riscv_vslide1up(p120, row4[1], vl);
                        p220 = __riscv_vslide1up(p220, row4[2], vl);
                        p320 = __riscv_vslide1up(p320, row4[3], vl);
                    };
                    slideup();
                    if (j == 0)
                    {
                        slideup();
                    }
                }
                p01 = __riscv_vslide1down(p00, row0[(j + vl - 2) * 4    ], vl);
                p11 = __riscv_vslide1down(p10, row0[(j + vl - 2) * 4 + 1], vl);
                p21 = __riscv_vslide1down(p20, row0[(j + vl - 2) * 4 + 2], vl);
                p31 = __riscv_vslide1down(p30, row0[(j + vl - 2) * 4 + 3], vl);
                p06 = __riscv_vslide1down(p05, row1[(j + vl - 2) * 4    ], vl);
                p16 = __riscv_vslide1down(p15, row1[(j + vl - 2) * 4 + 1], vl);
                p26 = __riscv_vslide1down(p25, row1[(j + vl - 2) * 4 + 2], vl);
                p36 = __riscv_vslide1down(p35, row1[(j + vl - 2) * 4 + 3], vl);
                p011 = __riscv_vslide1down(p010, row2[(j + vl - 2) * 4    ], vl);
                p111 = __riscv_vslide1down(p110, row2[(j + vl - 2) * 4 + 1], vl);
                p211 = __riscv_vslide1down(p210, row2[(j + vl - 2) * 4 + 2], vl);
                p311 = __riscv_vslide1down(p310, row2[(j + vl - 2) * 4 + 3], vl);
                p016 = __riscv_vslide1down(p015, row3[(j + vl - 2) * 4    ], vl);
                p116 = __riscv_vslide1down(p115, row3[(j + vl - 2) * 4 + 1], vl);
                p216 = __riscv_vslide1down(p215, row3[(j + vl - 2) * 4 + 2], vl);
                p316 = __riscv_vslide1down(p315, row3[(j + vl - 2) * 4 + 3], vl);
                p021 = __riscv_vslide1down(p020, row4[(j + vl - 2) * 4    ], vl);
                p121 = __riscv_vslide1down(p120, row4[(j + vl - 2) * 4 + 1], vl);
                p221 = __riscv_vslide1down(p220, row4[(j + vl - 2) * 4 + 2], vl);
                p321 = __riscv_vslide1down(p320, row4[(j + vl - 2) * 4 + 3], vl);
                p02 = __riscv_vslide1down(p01, row0[(j + vl - 1) * 4    ], vl);
                p12 = __riscv_vslide1down(p11, row0[(j + vl - 1) * 4 + 1], vl);
                p22 = __riscv_vslide1down(p21, row0[(j + vl - 1) * 4 + 2], vl);
                p32 = __riscv_vslide1down(p31, row0[(j + vl - 1) * 4 + 3], vl);
                p07 = __riscv_vslide1down(p06, row1[(j + vl - 1) * 4    ], vl);
                p17 = __riscv_vslide1down(p16, row1[(j + vl - 1) * 4 + 1], vl);
                p27 = __riscv_vslide1down(p26, row1[(j + vl - 1) * 4 + 2], vl);
                p37 = __riscv_vslide1down(p36, row1[(j + vl - 1) * 4 + 3], vl);
                p012 = __riscv_vslide1down(p011, row2[(j + vl - 1) * 4    ], vl);
                p112 = __riscv_vslide1down(p111, row2[(j + vl - 1) * 4 + 1], vl);
                p212 = __riscv_vslide1down(p211, row2[(j + vl - 1) * 4 + 2], vl);
                p312 = __riscv_vslide1down(p311, row2[(j + vl - 1) * 4 + 3], vl);
                p017 = __riscv_vslide1down(p016, row3[(j + vl - 1) * 4    ], vl);
                p117 = __riscv_vslide1down(p116, row3[(j + vl - 1) * 4 + 1], vl);
                p217 = __riscv_vslide1down(p216, row3[(j + vl - 1) * 4 + 2], vl);
                p317 = __riscv_vslide1down(p316, row3[(j + vl - 1) * 4 + 3], vl);
                p022 = __riscv_vslide1down(p021, row4[(j + vl - 1) * 4    ], vl);
                p122 = __riscv_vslide1down(p121, row4[(j + vl - 1) * 4 + 1], vl);
                p222 = __riscv_vslide1down(p221, row4[(j + vl - 1) * 4 + 2], vl);
                p322 = __riscv_vslide1down(p321, row4[(j + vl - 1) * 4 + 3], vl);
                p03 = __riscv_vslide1down(p02, row0[std::min(width - 1, j + vl) * 4    ], vl);
                p13 = __riscv_vslide1down(p12, row0[std::min(width - 1, j + vl) * 4 + 1], vl);
                p23 = __riscv_vslide1down(p22, row0[std::min(width - 1, j + vl) * 4 + 2], vl);
                p33 = __riscv_vslide1down(p32, row0[std::min(width - 1, j + vl) * 4 + 3], vl);
                p08 = __riscv_vslide1down(p07, row1[std::min(width - 1, j + vl) * 4    ], vl);
                p18 = __riscv_vslide1down(p17, row1[std::min(width - 1, j + vl) * 4 + 1], vl);
                p28 = __riscv_vslide1down(p27, row1[std::min(width - 1, j + vl) * 4 + 2], vl);
                p38 = __riscv_vslide1down(p37, row1[std::min(width - 1, j + vl) * 4 + 3], vl);
                p013 = __riscv_vslide1down(p012, row2[std::min(width - 1, j + vl) * 4    ], vl);
                p113 = __riscv_vslide1down(p112, row2[std::min(width - 1, j + vl) * 4 + 1], vl);
                p213 = __riscv_vslide1down(p212, row2[std::min(width - 1, j + vl) * 4 + 2], vl);
                p313 = __riscv_vslide1down(p312, row2[std::min(width - 1, j + vl) * 4 + 3], vl);
                p018 = __riscv_vslide1down(p017, row3[std::min(width - 1, j + vl) * 4    ], vl);
                p118 = __riscv_vslide1down(p117, row3[std::min(width - 1, j + vl) * 4 + 1], vl);
                p218 = __riscv_vslide1down(p217, row3[std::min(width - 1, j + vl) * 4 + 2], vl);
                p318 = __riscv_vslide1down(p317, row3[std::min(width - 1, j + vl) * 4 + 3], vl);
                p023 = __riscv_vslide1down(p022, row4[std::min(width - 1, j + vl) * 4    ], vl);
                p123 = __riscv_vslide1down(p122, row4[std::min(width - 1, j + vl) * 4 + 1], vl);
                p223 = __riscv_vslide1down(p222, row4[std::min(width - 1, j + vl) * 4 + 2], vl);
                p323 = __riscv_vslide1down(p322, row4[std::min(width - 1, j + vl) * 4 + 3], vl);
                p04 = __riscv_vslide1down(p03, row0[std::min(width - 1, j + vl + 1) * 4    ], vl);
                p14 = __riscv_vslide1down(p13, row0[std::min(width - 1, j + vl + 1) * 4 + 1], vl);
                p24 = __riscv_vslide1down(p23, row0[std::min(width - 1, j + vl + 1) * 4 + 2], vl);
                p34 = __riscv_vslide1down(p33, row0[std::min(width - 1, j + vl + 1) * 4 + 3], vl);
                p09 = __riscv_vslide1down(p08, row1[std::min(width - 1, j + vl + 1) * 4    ], vl);
                p19 = __riscv_vslide1down(p18, row1[std::min(width - 1, j + vl + 1) * 4 + 1], vl);
                p29 = __riscv_vslide1down(p28, row1[std::min(width - 1, j + vl + 1) * 4 + 2], vl);
                p39 = __riscv_vslide1down(p38, row1[std::min(width - 1, j + vl + 1) * 4 + 3], vl);
                p014 = __riscv_vslide1down(p013, row2[std::min(width - 1, j + vl + 1) * 4    ], vl);
                p114 = __riscv_vslide1down(p113, row2[std::min(width - 1, j + vl + 1) * 4 + 1], vl);
                p214 = __riscv_vslide1down(p213, row2[std::min(width - 1, j + vl + 1) * 4 + 2], vl);
                p314 = __riscv_vslide1down(p313, row2[std::min(width - 1, j + vl + 1) * 4 + 3], vl);
                p019 = __riscv_vslide1down(p018, row3[std::min(width - 1, j + vl + 1) * 4    ], vl);
                p119 = __riscv_vslide1down(p118, row3[std::min(width - 1, j + vl + 1) * 4 + 1], vl);
                p219 = __riscv_vslide1down(p218, row3[std::min(width - 1, j + vl + 1) * 4 + 2], vl);
                p319 = __riscv_vslide1down(p318, row3[std::min(width - 1, j + vl + 1) * 4 + 3], vl);
                p024 = __riscv_vslide1down(p023, row4[std::min(width - 1, j + vl + 1) * 4    ], vl);
                p124 = __riscv_vslide1down(p123, row4[std::min(width - 1, j + vl + 1) * 4 + 1], vl);
                p224 = __riscv_vslide1down(p223, row4[std::min(width - 1, j + vl + 1) * 4 + 2], vl);
                p324 = __riscv_vslide1down(p323, row4[std::min(width - 1, j + vl + 1) * 4 + 3], vl);

                auto vop = [&vl](vuint8m2_t& a, vuint8m2_t& b) {
                    auto t = a;
                    a = __riscv_vminu(a, b, vl);
                    b = __riscv_vmaxu(t, b, vl);
                };
                vuint8m2x4_t dst{};
                vop(p01, p02); vop(p00, p01); vop(p01, p02); vop(p04, p05); vop(p03, p04);
                vop(p04, p05); vop(p00, p03); vop(p02, p05); vop(p02, p03); vop(p01, p04);
                vop(p01, p02); vop(p03, p04); vop(p07, p08); vop(p06, p07); vop(p07, p08);
                vop(p010, p011); vop(p09, p010); vop(p010, p011); vop(p06, p09); vop(p08, p011);
                vop(p08, p09); vop(p07, p010); vop(p07, p08); vop(p09, p010); vop(p00, p06);
                vop(p04, p010); vop(p04, p06); vop(p02, p08); vop(p02, p04); vop(p06, p08);
                vop(p01, p07); vop(p05, p011); vop(p05, p07); vop(p03, p09); vop(p03, p05);
                vop(p07, p09); vop(p01, p02); vop(p03, p04); vop(p05, p06); vop(p07, p08);
                vop(p09, p010); vop(p013, p014); vop(p012, p013); vop(p013, p014); vop(p016, p017);
                vop(p015, p016); vop(p016, p017); vop(p012, p015); vop(p014, p017); vop(p014, p015);
                vop(p013, p016); vop(p013, p014); vop(p015, p016); vop(p019, p020); vop(p018, p019);
                vop(p019, p020); vop(p021, p022); vop(p023, p024); vop(p021, p023); vop(p022, p024);
                vop(p022, p023); vop(p018, p021); vop(p020, p023); vop(p020, p021); vop(p019, p022);
                vop(p022, p024); vop(p019, p020); vop(p021, p022); vop(p023, p024); vop(p012, p018);
                vop(p016, p022); vop(p016, p018); vop(p014, p020); vop(p020, p024); vop(p014, p016);
                vop(p018, p020); vop(p022, p024); vop(p013, p019); vop(p017, p023); vop(p017, p019);
                vop(p015, p021); vop(p015, p017); vop(p019, p021); vop(p013, p014); vop(p015, p016);
                vop(p017, p018); vop(p019, p020); vop(p021, p022); vop(p023, p024); vop(p00, p012);
                vop(p08, p020); vop(p08, p012); vop(p04, p016); vop(p016, p024); vop(p012, p016);
                vop(p02, p014); vop(p010, p022); vop(p010, p014); vop(p06, p018); vop(p06, p010);
                vop(p010, p012); vop(p01, p013); vop(p09, p021); vop(p09, p013); vop(p05, p017);
                vop(p013, p017); vop(p03, p015); vop(p011, p023); vop(p011, p015); vop(p07, p019);
                vop(p07, p011); vop(p011, p013); vop(p011, p012);
                dst = __riscv_vset_v_u8m2_u8m2x4(dst, 0, p012);
                vop(p11, p12); vop(p10, p11); vop(p11, p12); vop(p14, p15); vop(p13, p14);
                vop(p14, p15); vop(p10, p13); vop(p12, p15); vop(p12, p13); vop(p11, p14);
                vop(p11, p12); vop(p13, p14); vop(p17, p18); vop(p16, p17); vop(p17, p18);
                vop(p110, p111); vop(p19, p110); vop(p110, p111); vop(p16, p19); vop(p18, p111);
                vop(p18, p19); vop(p17, p110); vop(p17, p18); vop(p19, p110); vop(p10, p16);
                vop(p14, p110); vop(p14, p16); vop(p12, p18); vop(p12, p14); vop(p16, p18);
                vop(p11, p17); vop(p15, p111); vop(p15, p17); vop(p13, p19); vop(p13, p15);
                vop(p17, p19); vop(p11, p12); vop(p13, p14); vop(p15, p16); vop(p17, p18);
                vop(p19, p110); vop(p113, p114); vop(p112, p113); vop(p113, p114); vop(p116, p117);
                vop(p115, p116); vop(p116, p117); vop(p112, p115); vop(p114, p117); vop(p114, p115);
                vop(p113, p116); vop(p113, p114); vop(p115, p116); vop(p119, p120); vop(p118, p119);
                vop(p119, p120); vop(p121, p122); vop(p123, p124); vop(p121, p123); vop(p122, p124);
                vop(p122, p123); vop(p118, p121); vop(p120, p123); vop(p120, p121); vop(p119, p122);
                vop(p122, p124); vop(p119, p120); vop(p121, p122); vop(p123, p124); vop(p112, p118);
                vop(p116, p122); vop(p116, p118); vop(p114, p120); vop(p120, p124); vop(p114, p116);
                vop(p118, p120); vop(p122, p124); vop(p113, p119); vop(p117, p123); vop(p117, p119);
                vop(p115, p121); vop(p115, p117); vop(p119, p121); vop(p113, p114); vop(p115, p116);
                vop(p117, p118); vop(p119, p120); vop(p121, p122); vop(p123, p124); vop(p10, p112);
                vop(p18, p120); vop(p18, p112); vop(p14, p116); vop(p116, p124); vop(p112, p116);
                vop(p12, p114); vop(p110, p122); vop(p110, p114); vop(p16, p118); vop(p16, p110);
                vop(p110, p112); vop(p11, p113); vop(p19, p121); vop(p19, p113); vop(p15, p117);
                vop(p113, p117); vop(p13, p115); vop(p111, p123); vop(p111, p115); vop(p17, p119);
                vop(p17, p111); vop(p111, p113); vop(p111, p112);
                dst = __riscv_vset_v_u8m2_u8m2x4(dst, 1, p112);
                vop(p21, p22); vop(p20, p21); vop(p21, p22); vop(p24, p25); vop(p23, p24);
                vop(p24, p25); vop(p20, p23); vop(p22, p25); vop(p22, p23); vop(p21, p24);
                vop(p21, p22); vop(p23, p24); vop(p27, p28); vop(p26, p27); vop(p27, p28);
                vop(p210, p211); vop(p29, p210); vop(p210, p211); vop(p26, p29); vop(p28, p211);
                vop(p28, p29); vop(p27, p210); vop(p27, p28); vop(p29, p210); vop(p20, p26);
                vop(p24, p210); vop(p24, p26); vop(p22, p28); vop(p22, p24); vop(p26, p28);
                vop(p21, p27); vop(p25, p211); vop(p25, p27); vop(p23, p29); vop(p23, p25);
                vop(p27, p29); vop(p21, p22); vop(p23, p24); vop(p25, p26); vop(p27, p28);
                vop(p29, p210); vop(p213, p214); vop(p212, p213); vop(p213, p214); vop(p216, p217);
                vop(p215, p216); vop(p216, p217); vop(p212, p215); vop(p214, p217); vop(p214, p215);
                vop(p213, p216); vop(p213, p214); vop(p215, p216); vop(p219, p220); vop(p218, p219);
                vop(p219, p220); vop(p221, p222); vop(p223, p224); vop(p221, p223); vop(p222, p224);
                vop(p222, p223); vop(p218, p221); vop(p220, p223); vop(p220, p221); vop(p219, p222);
                vop(p222, p224); vop(p219, p220); vop(p221, p222); vop(p223, p224); vop(p212, p218);
                vop(p216, p222); vop(p216, p218); vop(p214, p220); vop(p220, p224); vop(p214, p216);
                vop(p218, p220); vop(p222, p224); vop(p213, p219); vop(p217, p223); vop(p217, p219);
                vop(p215, p221); vop(p215, p217); vop(p219, p221); vop(p213, p214); vop(p215, p216);
                vop(p217, p218); vop(p219, p220); vop(p221, p222); vop(p223, p224); vop(p20, p212);
                vop(p28, p220); vop(p28, p212); vop(p24, p216); vop(p216, p224); vop(p212, p216);
                vop(p22, p214); vop(p210, p222); vop(p210, p214); vop(p26, p218); vop(p26, p210);
                vop(p210, p212); vop(p21, p213); vop(p29, p221); vop(p29, p213); vop(p25, p217);
                vop(p213, p217); vop(p23, p215); vop(p211, p223); vop(p211, p215); vop(p27, p219);
                vop(p27, p211); vop(p211, p213); vop(p211, p212);
                dst = __riscv_vset_v_u8m2_u8m2x4(dst, 2, p212);
                vop(p31, p32); vop(p30, p31); vop(p31, p32); vop(p34, p35); vop(p33, p34);
                vop(p34, p35); vop(p30, p33); vop(p32, p35); vop(p32, p33); vop(p31, p34);
                vop(p31, p32); vop(p33, p34); vop(p37, p38); vop(p36, p37); vop(p37, p38);
                vop(p310, p311); vop(p39, p310); vop(p310, p311); vop(p36, p39); vop(p38, p311);
                vop(p38, p39); vop(p37, p310); vop(p37, p38); vop(p39, p310); vop(p30, p36);
                vop(p34, p310); vop(p34, p36); vop(p32, p38); vop(p32, p34); vop(p36, p38);
                vop(p31, p37); vop(p35, p311); vop(p35, p37); vop(p33, p39); vop(p33, p35);
                vop(p37, p39); vop(p31, p32); vop(p33, p34); vop(p35, p36); vop(p37, p38);
                vop(p39, p310); vop(p313, p314); vop(p312, p313); vop(p313, p314); vop(p316, p317);
                vop(p315, p316); vop(p316, p317); vop(p312, p315); vop(p314, p317); vop(p314, p315);
                vop(p313, p316); vop(p313, p314); vop(p315, p316); vop(p319, p320); vop(p318, p319);
                vop(p319, p320); vop(p321, p322); vop(p323, p324); vop(p321, p323); vop(p322, p324);
                vop(p322, p323); vop(p318, p321); vop(p320, p323); vop(p320, p321); vop(p319, p322);
                vop(p322, p324); vop(p319, p320); vop(p321, p322); vop(p323, p324); vop(p312, p318);
                vop(p316, p322); vop(p316, p318); vop(p314, p320); vop(p320, p324); vop(p314, p316);
                vop(p318, p320); vop(p322, p324); vop(p313, p319); vop(p317, p323); vop(p317, p319);
                vop(p315, p321); vop(p315, p317); vop(p319, p321); vop(p313, p314); vop(p315, p316);
                vop(p317, p318); vop(p319, p320); vop(p321, p322); vop(p323, p324); vop(p30, p312);
                vop(p38, p320); vop(p38, p312); vop(p34, p316); vop(p316, p324); vop(p312, p316);
                vop(p32, p314); vop(p310, p322); vop(p310, p314); vop(p36, p318); vop(p36, p310);
                vop(p310, p312); vop(p31, p313); vop(p39, p321); vop(p39, p313); vop(p35, p317);
                vop(p313, p317); vop(p33, p315); vop(p311, p323); vop(p311, p315); vop(p37, p319);
                vop(p37, p311); vop(p311, p313); vop(p311, p312);
                dst = __riscv_vset_v_u8m2_u8m2x4(dst, 3, p312);
                __riscv_vsseg4e8(dst_data + i * dst_step + j * 4, dst, vl);
            }
        }
    }

    return CV_HAL_ERROR_OK;
}

inline int medianBlur(const uchar* src_data, size_t src_step, uchar* dst_data, size_t dst_step, int width, int height, int depth, int cn, int ksize)
{
    const int type = CV_MAKETYPE(depth, cn);
    if (type != CV_8UC1 && type != CV_8UC4 && type != CV_16UC1 && type != CV_16SC1 && type != CV_32FC1)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if ((ksize != 3 && ksize != 5) || src_data == dst_data)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    switch (ksize*100 + type)
    {
    case 300 + CV_8UC1:
        return filter::invoke(height, {medianBlurC1<3, RVV_U8M4>}, src_data, src_step, dst_data, dst_step, width, height);
    case 300 + CV_16UC1:
        return filter::invoke(height, {medianBlurC1<3, RVV_U16M4>}, src_data, src_step, dst_data, dst_step, width, height);
    case 300 + CV_16SC1:
        return filter::invoke(height, {medianBlurC1<3, RVV_I16M4>}, src_data, src_step, dst_data, dst_step, width, height);
    case 300 + CV_32FC1:
        return filter::invoke(height, {medianBlurC1<3, RVV_F32M4>}, src_data, src_step, dst_data, dst_step, width, height);
    case 500 + CV_8UC1:
        return filter::invoke(height, {medianBlurC1<5, RVV_U8M1>}, src_data, src_step, dst_data, dst_step, width, height);
    case 500 + CV_16UC1:
        return filter::invoke(height, {medianBlurC1<5, RVV_U16M1>}, src_data, src_step, dst_data, dst_step, width, height);
    case 500 + CV_16SC1:
        return filter::invoke(height, {medianBlurC1<5, RVV_I16M1>}, src_data, src_step, dst_data, dst_step, width, height);
    case 500 + CV_32FC1:
        return filter::invoke(height, {medianBlurC1<5, RVV_F32M1>}, src_data, src_step, dst_data, dst_step, width, height);

    case 300 + CV_8UC4:
        return filter::invoke(height, {medianBlurC4<3>}, src_data, src_step, dst_data, dst_step, width, height);
    case 500 + CV_8UC4:
        return filter::invoke(height, {medianBlurC4<5>}, src_data, src_step, dst_data, dst_step, width, height);
    }

    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}
} // cv::cv_hal_rvv::medianBlur

namespace boxFilter {
#undef cv_hal_boxFilter
#define cv_hal_boxFilter cv::cv_hal_rvv::boxFilter::boxFilter

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
        int pi = filter::borderInterpolate(offset_y + x - anchor_y, full_height, border_type);
        return pi < 0 ? noval : pi - offset_y;
    };
    auto accessY = [&](int y) {
        int pj = filter::borderInterpolate(offset_x + y - anchor_x, full_width, border_type);
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
        int pi = filter::borderInterpolate(offset_y + x - anchor_y, full_height, border_type);
        return pi < 0 ? noval : pi - offset_y;
    };
    auto accessY = [&](int y) {
        int pj = filter::borderInterpolate(offset_x + y - anchor_x, full_width, border_type);
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

inline int boxFilter(const uchar* src_data, size_t src_step, uchar* dst_data, size_t dst_step, int width, int height, int src_depth, int dst_depth, int cn, int margin_left, int margin_top, int margin_right, int margin_bottom, size_t ksize_width, size_t ksize_height, int anchor_x, int anchor_y, bool normalize, int border_type)
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
                res = filter::invoke(height, {boxFilterC1<3, RVV_U8M4, RVV_U16M8, false>}, src_data, src_step, dst_data, dst_step, width, margin_left + width + margin_right, margin_top + height + margin_bottom, margin_left, margin_top, anchor_x, anchor_y, normalize, border_type);
            }
            if (ksize_width == 5)
            {
                res = filter::invoke(height, {boxFilterC1<5, RVV_U8M4, RVV_U16M8, false>}, src_data, src_step, dst_data, dst_step, width, margin_left + width + margin_right, margin_top + height + margin_bottom, margin_left, margin_top, anchor_x, anchor_y, normalize, border_type);
            }
        }
    }
    else
    {
        switch (ksize_width*100 + src_type)
        {
        case 300 + CV_8UC1:
            res = filter::invoke(height, {boxFilterC1<3, RVV_U8M4, RVV_U16M8, true>}, src_data, src_step, dst_data, dst_step, width, margin_left + width + margin_right, margin_top + height + margin_bottom, margin_left, margin_top, anchor_x, anchor_y, normalize, border_type);
            break;
        case 500 + CV_8UC1:
            res = filter::invoke(height, {boxFilterC1<5, RVV_U8M4, RVV_U16M8, true>}, src_data, src_step, dst_data, dst_step, width, margin_left + width + margin_right, margin_top + height + margin_bottom, margin_left, margin_top, anchor_x, anchor_y, normalize, border_type);
            break;
        case 300 + CV_16SC1:
            res = filter::invoke(height, {boxFilterC1<3, RVV_I16M4, RVV_I32M8, true>}, src_data, src_step, dst_data, dst_step, width, margin_left + width + margin_right, margin_top + height + margin_bottom, margin_left, margin_top, anchor_x, anchor_y, normalize, border_type);
            break;
        case 500 + CV_16SC1:
            res = filter::invoke(height, {boxFilterC1<5, RVV_I16M4, RVV_I32M8, true>}, src_data, src_step, dst_data, dst_step, width, margin_left + width + margin_right, margin_top + height + margin_bottom, margin_left, margin_top, anchor_x, anchor_y, normalize, border_type);
            break;
        case 300 + CV_32SC1:
            res = filter::invoke(height, {boxFilterC1<3, RVV_I32M8, RVV_I32M8, true>}, src_data, src_step, dst_data, dst_step, width, margin_left + width + margin_right, margin_top + height + margin_bottom, margin_left, margin_top, anchor_x, anchor_y, normalize, border_type);
            break;
        case 500 + CV_32SC1:
            res = filter::invoke(height, {boxFilterC1<5, RVV_I32M8, RVV_I32M8, true>}, src_data, src_step, dst_data, dst_step, width, margin_left + width + margin_right, margin_top + height + margin_bottom, margin_left, margin_top, anchor_x, anchor_y, normalize, border_type);
            break;
        case 300 + CV_32FC1:
            res = filter::invoke(height, {boxFilterC1<3, RVV_F32M8, RVV_F32M8, true>}, src_data, src_step, dst_data, dst_step, width, margin_left + width + margin_right, margin_top + height + margin_bottom, margin_left, margin_top, anchor_x, anchor_y, normalize, border_type);
            break;
        case 500 + CV_32FC1:
            res = filter::invoke(height, {boxFilterC1<5, RVV_F32M8, RVV_F32M8, true>}, src_data, src_step, dst_data, dst_step, width, margin_left + width + margin_right, margin_top + height + margin_bottom, margin_left, margin_top, anchor_x, anchor_y, normalize, border_type);
            break;
        case 300 + CV_32FC3:
            res = filter::invoke(height, {boxFilterC3<3>}, src_data, src_step, dst_data, dst_step, width, margin_left + width + margin_right, margin_top + height + margin_bottom, margin_left, margin_top, anchor_x, anchor_y, normalize, border_type);
            break;
        case 500 + CV_32FC3:
            res = filter::invoke(height, {boxFilterC3<5>}, src_data, src_step, dst_data, dst_step, width, margin_left + width + margin_right, margin_top + height + margin_bottom, margin_left, margin_top, anchor_x, anchor_y, normalize, border_type);
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
} // cv::cv_hal_rvv::boxFilter

namespace bilateralFilter {
#undef cv_hal_bilateralFilter
#define cv_hal_bilateralFilter cv::cv_hal_rvv::bilateralFilter::bilateralFilter

// the algorithm is copied from imgproc/src/bilateral_filter.simd.cpp
// in the functor BilateralFilter_8u_Invoker
static inline int bilateralFilter8UC1(int start, int end, const uchar* src_data, size_t src_step, uchar* dst_data, size_t dst_step, int width, int radius, int maxk, const int* space_ofs, const float* space_weight, const float* color_weight)
{
    constexpr int align = 31;
    std::vector<float> _sum(width + align), _wsum(width + align);
    float* sum = reinterpret_cast<float*>(((size_t)_sum.data() + align) & ~align);
    float* wsum = reinterpret_cast<float*>(((size_t)_wsum.data() + align) & ~align);

    for (int i = start; i < end; i++)
    {
        const uchar* sptr = src_data + (i+radius) * src_step + radius;
        memset(sum, 0, sizeof(float) * width);
        memset(wsum, 0, sizeof(float) * width);
        for(int k = 0; k < maxk; k++)
        {
            const uchar* ksptr = sptr + space_ofs[k];
            int vl;
            for (int j = 0; j < width; j += vl)
            {
                vl = __riscv_vsetvl_e8m2(width - j);
                auto src = __riscv_vle8_v_u8m2(sptr + j, vl);
                auto ksrc = __riscv_vle8_v_u8m2(ksptr + j, vl);
                auto diff = __riscv_vsub(__riscv_vmaxu(src, ksrc, vl), __riscv_vminu(src, ksrc, vl), vl);
                auto w = __riscv_vloxei16_v_f32m8(color_weight, __riscv_vmul(__riscv_vzext_vf2(diff, vl), sizeof(float), vl), vl);
                w = __riscv_vfmul(w, space_weight[k], vl);

                __riscv_vse32(wsum + j, __riscv_vfadd(w, __riscv_vle32_v_f32m8(wsum + j, vl), vl), vl);
                __riscv_vse32(sum + j, __riscv_vfmadd(w, __riscv_vfwcvt_f(__riscv_vzext_vf2(ksrc, vl), vl), __riscv_vle32_v_f32m8(sum + j, vl), vl), vl);
            }
        }

        int vl;
        for (int j = 0; j < width; j += vl)
        {
            vl = __riscv_vsetvl_e8m2(width - j);
            auto dst = __riscv_vfncvt_xu(__riscv_vfdiv(__riscv_vle32_v_f32m8(sum + j, vl), __riscv_vle32_v_f32m8(wsum + j, vl), vl), vl);
            __riscv_vse8(dst_data + i * dst_step + j, __riscv_vncvt_x(dst, vl), vl);
        }
    }

    return CV_HAL_ERROR_OK;
}

static inline int bilateralFilter8UC3(int start, int end, const uchar* src_data, size_t src_step, uchar* dst_data, size_t dst_step, int width, int radius, int maxk, const int* space_ofs, const float* space_weight, const float* color_weight)
{
    constexpr int align = 31;
    std::vector<float> _sum_b(width + align), _sum_g(width + align), _sum_r(width + align), _wsum(width + align);
    float* sum_b = reinterpret_cast<float*>(((size_t)_sum_b.data() + align) & ~align);
    float* sum_g = reinterpret_cast<float*>(((size_t)_sum_g.data() + align) & ~align);
    float* sum_r = reinterpret_cast<float*>(((size_t)_sum_r.data() + align) & ~align);
    float* wsum = reinterpret_cast<float*>(((size_t)_wsum.data() + align) & ~align);

    for (int i = start; i < end; i++)
    {
        const uchar* sptr = src_data + (i+radius) * src_step + radius*3;
        memset(sum_b, 0, sizeof(float) * width);
        memset(sum_g, 0, sizeof(float) * width);
        memset(sum_r, 0, sizeof(float) * width);
        memset(wsum, 0, sizeof(float) * width);
        for(int k = 0; k < maxk; k++)
        {
            const uchar* ksptr = sptr + space_ofs[k];
            int vl;
            for (int j = 0; j < width; j += vl)
            {
                vl = __riscv_vsetvl_e8m2(width - j);
                auto src = __riscv_vlseg3e8_v_u8m2x3(sptr + j * 3, vl);
                auto src0 = __riscv_vget_v_u8m2x3_u8m2(src, 0);
                auto src1 = __riscv_vget_v_u8m2x3_u8m2(src, 1);
                auto src2 = __riscv_vget_v_u8m2x3_u8m2(src, 2);
                src = __riscv_vlseg3e8_v_u8m2x3(ksptr + j * 3, vl);
                auto ksrc0 = __riscv_vget_v_u8m2x3_u8m2(src, 0);
                auto ksrc1 = __riscv_vget_v_u8m2x3_u8m2(src, 1);
                auto ksrc2 = __riscv_vget_v_u8m2x3_u8m2(src, 2);

                auto diff0 = __riscv_vsub(__riscv_vmaxu(src0, ksrc0, vl), __riscv_vminu(src0, ksrc0, vl), vl);
                auto diff1 = __riscv_vsub(__riscv_vmaxu(src1, ksrc1, vl), __riscv_vminu(src1, ksrc1, vl), vl);
                auto diff2 = __riscv_vsub(__riscv_vmaxu(src2, ksrc2, vl), __riscv_vminu(src2, ksrc2, vl), vl);
                auto w = __riscv_vloxei16_v_f32m8(color_weight, __riscv_vmul(__riscv_vadd(__riscv_vadd(__riscv_vzext_vf2(diff0, vl), __riscv_vzext_vf2(diff1, vl), vl), __riscv_vzext_vf2(diff2, vl), vl), sizeof(float), vl), vl);
                w = __riscv_vfmul(w, space_weight[k], vl);

                __riscv_vse32(wsum + j, __riscv_vfadd(w, __riscv_vle32_v_f32m8(wsum + j, vl), vl), vl);
                __riscv_vse32(sum_b + j, __riscv_vfmadd(w, __riscv_vfwcvt_f(__riscv_vzext_vf2(ksrc0, vl), vl), __riscv_vle32_v_f32m8(sum_b + j, vl), vl), vl);
                __riscv_vse32(sum_g + j, __riscv_vfmadd(w, __riscv_vfwcvt_f(__riscv_vzext_vf2(ksrc1, vl), vl), __riscv_vle32_v_f32m8(sum_g + j, vl), vl), vl);
                __riscv_vse32(sum_r + j, __riscv_vfmadd(w, __riscv_vfwcvt_f(__riscv_vzext_vf2(ksrc2, vl), vl), __riscv_vle32_v_f32m8(sum_r + j, vl), vl), vl);
            }
        }

        int vl;
        for (int j = 0; j < width; j += vl)
        {
            vl = __riscv_vsetvl_e8m2(width - j);
            auto w = __riscv_vfrdiv(__riscv_vle32_v_f32m8(wsum + j, vl), 1.0f, vl);
            vuint8m2x3_t dst{};
            dst = __riscv_vset_v_u8m2_u8m2x3(dst, 0,__riscv_vncvt_x(__riscv_vfncvt_xu(__riscv_vfmul(__riscv_vle32_v_f32m8(sum_b + j, vl), w, vl), vl), vl));
            dst = __riscv_vset_v_u8m2_u8m2x3(dst, 1,__riscv_vncvt_x(__riscv_vfncvt_xu(__riscv_vfmul(__riscv_vle32_v_f32m8(sum_g + j, vl), w, vl), vl), vl));
            dst = __riscv_vset_v_u8m2_u8m2x3(dst, 2,__riscv_vncvt_x(__riscv_vfncvt_xu(__riscv_vfmul(__riscv_vle32_v_f32m8(sum_r + j, vl), w, vl), vl), vl));
            __riscv_vsseg3e8(dst_data + i * dst_step + j * 3, dst, vl);
        }
    }

    return CV_HAL_ERROR_OK;
}

// the algorithm is copied from imgproc/src/bilateral_filter.simd.cpp
// in the functor BilateralFilter_32f_Invoker
static inline int bilateralFilter32FC1(int start, int end, const uchar* src_data, size_t src_step, uchar* dst_data, size_t dst_step, int width, int radius, int maxk, const int* space_ofs, const float* space_weight, const float* expLUT, float scale_index)
{
    constexpr int align = 31;
    std::vector<float> _sum(width + align), _wsum(width + align);
    float* sum = reinterpret_cast<float*>(((size_t)_sum.data() + align) & ~align);
    float* wsum = reinterpret_cast<float*>(((size_t)_wsum.data() + align) & ~align);

    for (int i = start; i < end; i++)
    {
        const float* sptr = reinterpret_cast<const float*>(src_data + (i+radius) * src_step) + radius;
        memset(sum, 0, sizeof(float) * width);
        memset(wsum, 0, sizeof(float) * width);
        for(int k = 0; k < maxk; k++)
        {
            const float* ksptr = sptr + space_ofs[k];
            int vl;
            for (int j = 0; j < width; j += vl)
            {
                vl = __riscv_vsetvl_e32m4(width - j);
                auto src = __riscv_vle32_v_f32m4(sptr + j, vl);
                auto ksrc = __riscv_vle32_v_f32m4(ksptr + j, vl);
                auto diff = __riscv_vfmul(__riscv_vfabs(__riscv_vfsub(src, ksrc, vl), vl), scale_index, vl);
                auto idx = __riscv_vfcvt_rtz_x(diff, vl);
                auto alpha = __riscv_vfsub(diff, __riscv_vfcvt_f(idx, vl), vl);

                auto exp = __riscv_vloxseg2ei32_v_f32m4x2(expLUT, __riscv_vreinterpret_v_i32m4_u32m4(__riscv_vmul(idx, sizeof(float), vl)), vl);
                auto w = __riscv_vfmadd(alpha, __riscv_vfsub(__riscv_vget_v_f32m4x2_f32m4(exp, 1), __riscv_vget_v_f32m4x2_f32m4(exp, 0), vl), __riscv_vget_v_f32m4x2_f32m4(exp, 0), vl);
                w = __riscv_vfmul(w, space_weight[k], vl);

                __riscv_vse32(wsum + j, __riscv_vfadd(w, __riscv_vle32_v_f32m4(wsum + j, vl), vl), vl);
                __riscv_vse32(sum + j, __riscv_vfmadd(w, ksrc, __riscv_vle32_v_f32m4(sum + j, vl), vl), vl);
            }
        }

        int vl;
        for (int j = 0; j < width; j += vl)
        {
            vl = __riscv_vsetvl_e32m4(width - j);
            auto src = __riscv_vle32_v_f32m4(sptr + j, vl);
            auto dst = __riscv_vfdiv(__riscv_vfadd(__riscv_vle32_v_f32m4(sum + j, vl), src, vl), __riscv_vfadd(__riscv_vle32_v_f32m4(wsum + j, vl), 1, vl), vl);
            __riscv_vse32(reinterpret_cast<float*>(dst_data + i * dst_step) + j, dst, vl);
        }
    }

    return CV_HAL_ERROR_OK;
}

static inline int bilateralFilter32FC3(int start, int end, const uchar* src_data, size_t src_step, uchar* dst_data, size_t dst_step, int width, int radius, int maxk, const int* space_ofs, const float* space_weight, const float* expLUT, float scale_index)
{
    constexpr int align = 31;
    std::vector<float> _sum_b(width + align), _sum_g(width + align), _sum_r(width + align), _wsum(width + align);
    float* sum_b = reinterpret_cast<float*>(((size_t)_sum_b.data() + align) & ~align);
    float* sum_g = reinterpret_cast<float*>(((size_t)_sum_g.data() + align) & ~align);
    float* sum_r = reinterpret_cast<float*>(((size_t)_sum_r.data() + align) & ~align);
    float* wsum = reinterpret_cast<float*>(((size_t)_wsum.data() + align) & ~align);

    for (int i = start; i < end; i++)
    {
        const float* sptr = reinterpret_cast<const float*>(src_data + (i+radius) * src_step) + radius*3;
        memset(sum_b, 0, sizeof(float) * width);
        memset(sum_g, 0, sizeof(float) * width);
        memset(sum_r, 0, sizeof(float) * width);
        memset(wsum, 0, sizeof(float) * width);
        for(int k = 0; k < maxk; k++)
        {
            const float* ksptr = sptr + space_ofs[k];
            int vl;
            for (int j = 0; j < width; j += vl)
            {
                vl = __riscv_vsetvl_e32m2(width - j);
                auto src = __riscv_vlseg3e32_v_f32m2x3(sptr + j * 3, vl);
                auto src0 = __riscv_vget_v_f32m2x3_f32m2(src, 0);
                auto src1 = __riscv_vget_v_f32m2x3_f32m2(src, 1);
                auto src2 = __riscv_vget_v_f32m2x3_f32m2(src, 2);
                src = __riscv_vlseg3e32_v_f32m2x3(ksptr + j * 3, vl);
                auto ksrc0 = __riscv_vget_v_f32m2x3_f32m2(src, 0);
                auto ksrc1 = __riscv_vget_v_f32m2x3_f32m2(src, 1);
                auto ksrc2 = __riscv_vget_v_f32m2x3_f32m2(src, 2);

                auto diff = __riscv_vfmul(__riscv_vfadd(__riscv_vfadd(__riscv_vfabs(__riscv_vfsub(src0, ksrc0, vl), vl), __riscv_vfabs(__riscv_vfsub(src1, ksrc1, vl), vl), vl), __riscv_vfabs(__riscv_vfsub(src2, ksrc2, vl), vl), vl), scale_index, vl);
                auto idx = __riscv_vfcvt_rtz_x(diff, vl);
                auto alpha = __riscv_vfsub(diff, __riscv_vfcvt_f(idx, vl), vl);

                auto exp = __riscv_vloxseg2ei32_v_f32m2x2(expLUT, __riscv_vreinterpret_v_i32m2_u32m2(__riscv_vmul(idx, sizeof(float), vl)), vl);
                auto w = __riscv_vfmadd(alpha, __riscv_vfsub(__riscv_vget_v_f32m2x2_f32m2(exp, 1), __riscv_vget_v_f32m2x2_f32m2(exp, 0), vl), __riscv_vget_v_f32m2x2_f32m2(exp, 0), vl);
                w = __riscv_vfmul(w, space_weight[k], vl);

                __riscv_vse32(wsum + j, __riscv_vfadd(w, __riscv_vle32_v_f32m2(wsum + j, vl), vl), vl);
                __riscv_vse32(sum_b + j, __riscv_vfmadd(w, ksrc0, __riscv_vle32_v_f32m2(sum_b + j, vl), vl), vl);
                __riscv_vse32(sum_g + j, __riscv_vfmadd(w, ksrc1, __riscv_vle32_v_f32m2(sum_g + j, vl), vl), vl);
                __riscv_vse32(sum_r + j, __riscv_vfmadd(w, ksrc2, __riscv_vle32_v_f32m2(sum_r + j, vl), vl), vl);
            }
        }

        int vl;
        for (int j = 0; j < width; j += vl)
        {
            vl = __riscv_vsetvl_e32m2(width - j);
            auto w = __riscv_vfrdiv(__riscv_vfadd(__riscv_vle32_v_f32m2(wsum + j, vl), 1, vl), 1, vl);
            auto src = __riscv_vlseg3e32_v_f32m2x3(sptr + j * 3, vl);
            auto src0 = __riscv_vget_v_f32m2x3_f32m2(src, 0);
            auto src1 = __riscv_vget_v_f32m2x3_f32m2(src, 1);
            auto src2 = __riscv_vget_v_f32m2x3_f32m2(src, 2);

            vfloat32m2x3_t dst{};
            dst = __riscv_vset_v_f32m2_f32m2x3(dst, 0, __riscv_vfmul(w, __riscv_vfadd(__riscv_vle32_v_f32m2(sum_b + j, vl), src0, vl), vl));
            dst = __riscv_vset_v_f32m2_f32m2x3(dst, 1, __riscv_vfmul(w, __riscv_vfadd(__riscv_vle32_v_f32m2(sum_g + j, vl), src1, vl), vl));
            dst = __riscv_vset_v_f32m2_f32m2x3(dst, 2, __riscv_vfmul(w, __riscv_vfadd(__riscv_vle32_v_f32m2(sum_r + j, vl), src2, vl), vl));
            __riscv_vsseg3e32(reinterpret_cast<float*>(dst_data + i * dst_step) + j * 3, dst, vl);
        }
    }

    return CV_HAL_ERROR_OK;
}

// the algorithm is copied from imgproc/src/bilateral_filter.dispatch.cpp
// in the function static void bilateralFilter_8u and bilateralFilter_32f
inline int bilateralFilter(const uchar* src_data, size_t src_step, uchar* dst_data, size_t dst_step,
                           int width, int height, int depth, int cn, int d, double sigma_color, double sigma_space, int border_type)
{
    const int type = CV_MAKETYPE(depth, cn);
    if (type != CV_8UC1 && type != CV_8UC3 && type != CV_32FC1 && type != CV_32FC3)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if (type == CV_32FC1 && width * height > 1 << 20)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if (src_data == dst_data || border_type & BORDER_ISOLATED)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    sigma_color = sigma_color <= 0 ? 1 : sigma_color;
    sigma_space = sigma_space <= 0 ? 1 : sigma_space;
    double gauss_color_coeff = -0.5/(sigma_color*sigma_color);
    double gauss_space_coeff = -0.5/(sigma_space*sigma_space);
    int radius = d <= 0 ? std::round(sigma_space*1.5) : d/2;
    radius = std::max(radius, 1);
    d = radius*2 + 1;

    const int size = depth == CV_32F ? cn * sizeof(float) : cn;
    const int temp_step = (width + radius * 2) * size;
    std::vector<uchar> _temp((width + radius * 2) * (height + radius * 2) * size, 0);
    uchar* temp = _temp.data();
    std::vector<int> width_interpolate(radius * 2);
    for (int j = 0; j < radius; j++)
    {
        width_interpolate[j] = filter::borderInterpolate(j - radius, width, border_type);
        width_interpolate[j + radius] = filter::borderInterpolate(width + j, width, border_type);
    }
    for (int i = 0; i < height + radius * 2; i++)
    {
        int x = filter::borderInterpolate(i - radius, height, border_type);
        if (x != -1)
        {
            for (int j = 0; j < radius; j++)
            {
                int y = width_interpolate[j];
                if (y != -1)
                    memcpy(temp + i * temp_step + j * size, src_data + x * src_step + y * size, size);
                y = width_interpolate[j + radius];
                if (y != -1)
                    memcpy(temp + i * temp_step + (width + j + radius) * size, src_data + x * src_step + y * size, size);
            }
            memcpy(temp + i * temp_step + radius * size, src_data + x * src_step, width * size);
        }
    }

    std::vector<float> _space_weight(d*d);
    std::vector<int> _space_ofs(d*d);
    float* space_weight = _space_weight.data();
    int* space_ofs = _space_ofs.data();
    int maxk = 0;
    for (int i = -radius; i <= radius; i++)
    {
        for (int j = -radius; j <= radius; j++)
        {
            double r = std::sqrt((double)i*i + (double)j*j);
            if (r <= radius && (depth == CV_8U || i != 0 || j != 0))
            {
                space_weight[maxk] = static_cast<float>(r*r*gauss_space_coeff);
                space_ofs[maxk++] = (i * (temp_step / size) + j) * cn;
            }
        }
    }
    cv::cv_hal_rvv::exp32f(space_weight, space_weight, maxk);

    if (depth == CV_8U)
    {
        std::vector<float> _color_weight(cn*256);
        float* color_weight = _color_weight.data();
        for (int i = 0; i < 256*cn; i++)
            color_weight[i] = static_cast<float>(i*i*gauss_color_coeff);
        cv::cv_hal_rvv::exp32f(color_weight, color_weight, 256*cn);

        switch (cn)
        {
        case 1:
            return filter::invoke(height, {bilateralFilter8UC1}, temp, temp_step, dst_data, dst_step, width, radius, maxk, space_ofs, space_weight, color_weight);
        case 3:
            return filter::invoke(height, {bilateralFilter8UC3}, temp, temp_step, dst_data, dst_step, width, radius, maxk, space_ofs, space_weight, color_weight);
        }
    }
    else
    {
        double minValSrc = -1, maxValSrc = 1;
        cv::cv_hal_rvv::minmax::minMaxIdx(src_data, src_step, width * cn, height, CV_32F, &minValSrc, &maxValSrc, nullptr, nullptr, nullptr);
        if(std::abs(minValSrc - maxValSrc) < FLT_EPSILON)
        {
            for (int i = 0; i < width; i++)
                memcpy(dst_data + i * dst_step, src_data + i * src_step, width * size);
            return CV_HAL_ERROR_OK;
        }

        const int kExpNumBinsPerChannel = 1 << 12;
        const int kExpNumBins = kExpNumBinsPerChannel * cn;
        const float scale_index = kExpNumBins / static_cast<float>((maxValSrc - minValSrc) * cn);
        std::vector<float> _expLUT(kExpNumBins+2, 0);
        float* expLUT = _expLUT.data();
        for (int i = 0; i < kExpNumBins+2; i++)
        {
            double val = i / scale_index;
            expLUT[i] = static_cast<float>(val * val * gauss_color_coeff);
        }
        cv::cv_hal_rvv::exp32f(expLUT, expLUT, kExpNumBins+2);

        switch (cn)
        {
        case 1:
            return filter::invoke(height, {bilateralFilter32FC1}, temp, temp_step, dst_data, dst_step, width, radius, maxk, space_ofs, space_weight, expLUT, scale_index);
        case 3:
            return filter::invoke(height, {bilateralFilter32FC3}, temp, temp_step, dst_data, dst_step, width, radius, maxk, space_ofs, space_weight, expLUT, scale_index);
        }
    }

    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}
} // cv::cv_hal_rvv::bilateralFilter

}}

#endif
