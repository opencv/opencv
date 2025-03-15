// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#ifndef OPENCV_HAL_RVV_FILTER_HPP_INCLUDED
#define OPENCV_HAL_RVV_FILTER_HPP_INCLUDED

#include "../../imgproc/include/opencv2/imgproc/hal/interface.h"
#include "hal_rvv_1p0/types.hpp"

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
    if( (unsigned)p < (unsigned)len )
        ;
    else if( borderType == BORDER_REPLICATE )
        p = p < 0 ? 0 : len - 1;
    else if( borderType == BORDER_REFLECT || borderType == BORDER_REFLECT_101 )
    {
        int delta = borderType == BORDER_REFLECT_101;
        if( len == 1 )
            return 0;
        do
        {
            if( p < 0 )
                p = -p - 1 + delta;
            else
                p = len - 1 - (p - len) - delta;
        }
        while( (unsigned)p >= (unsigned)len );
    }
    else if( borderType == BORDER_CONSTANT )
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
        std::copy(dst.data() + i * width * 4, dst.data() + (i + 1) * width * 4, dst_data + i * dst_step);
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

inline int sepFilterInit(cvhalFilter2D **context, int src_type, int dst_type, int kernel_type, uchar *kernelx_data, int kernelx_length, uchar *kernely_data, int kernely_length, int anchor_x, int anchor_y, double delta, int borderType)
{
    if (kernel_type != CV_32FC1)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if (src_type != CV_8UC1 && src_type != CV_16SC1 && src_type != CV_32FC1)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if (dst_type != CV_16SC1 && dst_type != CV_32FC1)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if (kernelx_length != kernely_length)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if (kernelx_length != 3 && kernelx_length != 5)
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

inline int sepFilter(cvhalFilter2D *context, uchar *src_data, size_t src_step, uchar* dst_data, size_t dst_step, int width, int height, int full_width, int full_height, int offset_x, int offset_y)
{
    sepFilter2D* data = reinterpret_cast<sepFilter2D*>(context);

    switch (data->kernelx_length*100 + data->src_type)
    {
    case 300 + CV_8UC1:
        return filter::invoke(height, {sepFilter<3, uchar>}, data, src_data, src_step, dst_data, dst_step, width, height, full_width, full_height, offset_x, offset_y);
    case 500 + CV_8UC1:
        return filter::invoke(height, {sepFilter<5, uchar>}, data, src_data, src_step, dst_data, dst_step, width, height, full_width, full_height, offset_x, offset_y);
    case 300 + CV_16SC1:
        return filter::invoke(height, {sepFilter<3, short>}, data, src_data, src_step, dst_data, dst_step, width, height, full_width, full_height, offset_x, offset_y);
    case 500 + CV_16SC1:
        return filter::invoke(height, {sepFilter<5, short>}, data, src_data, src_step, dst_data, dst_step, width, height, full_width, full_height, offset_x, offset_y);
    case 300 + CV_32FC1:
        return filter::invoke(height, {sepFilter<3, float>}, data, src_data, src_step, dst_data, dst_step, width, height, full_width, full_height, offset_x, offset_y);
    case 500 + CV_32FC1:
        return filter::invoke(height, {sepFilter<5, float>}, data, src_data, src_step, dst_data, dst_step, width, height, full_width, full_height, offset_x, offset_y);
    }

    return CV_HAL_ERROR_NOT_IMPLEMENTED;
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
    uchar *kernel_data;
    size_t kernel_step;
    int kernel_width;
    int kernel_height;
    int anchor_x;
    int anchor_y;
    int borderType;
    const uchar* borderValue;
};

inline int morphInit(cvhalFilter2D** context, int operation, int src_type, int dst_type, int /*max_width*/, int /*max_height*/, int kernel_type, uchar *kernel_data, size_t kernel_step, int kernel_width, int kernel_height, int anchor_x, int anchor_y, int borderType, const double borderValue[4], int iterations, bool /*allowSubmatrix*/, bool /*allowInplace*/)
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

inline int morph(cvhalFilter2D* context, uchar *src_data, size_t src_step, uchar *dst_data, size_t dst_step, int width, int height, int src_full_width, int src_full_height, int src_roi_x, int src_roi_y, int /*dst_full_width*/, int /*dst_full_height*/, int /*dst_roi_x*/, int /*dst_roi_y*/)
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
        std::copy(dst.data() + i * width * cn, dst.data() + (i + 1) * width * cn, dst_data + i * dst_step);
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
    if (type != CV_8UC1 && type != CV_8UC4 && type != CV_16UC1)
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

}}

#endif
