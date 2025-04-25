// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2025, Institute of Software, Chinese Academy of Sciences.

#ifndef OPENCV_HAL_RVV_THRESH_HPP_INCLUDED
#define OPENCV_HAL_RVV_THRESH_HPP_INCLUDED

#include <riscv_vector.h>
#include <atomic>

namespace cv { namespace cv_hal_rvv {

namespace threshold {
// disabled since UI is fast enough, only called in threshold_otsu
// #undef cv_hal_threshold
// #define cv_hal_threshold cv::cv_hal_rvv::threshold::threshold

class ThresholdInvoker : public ParallelLoopBody
{
public:
    template<typename... Args>
    ThresholdInvoker(std::function<int(int, int, Args...)> _func, Args&&... args)
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
static inline int invoke(int width, int height, std::function<int(int, int, Args...)> func, Args&&... args)
{
    cv::parallel_for_(Range(1, height), ThresholdInvoker(func, std::forward<Args>(args)...), static_cast<double>((width - 1) * height) / (1 << 15));
    return func(0, 1, std::forward<Args>(args)...);
}

template<typename T> struct rvv;
template<> struct rvv<uchar>
{
    static inline vuint8m4_t vmerge(vuint8m4_t a, uchar b, vbool2_t c, size_t d) { return __riscv_vmerge(a, b, c, d); }
};
template<> struct rvv<short>
{
    static inline vint16m4_t vmerge(vint16m4_t a, short b, vbool4_t c, size_t d) { return __riscv_vmerge(a, b, c, d); }
};
template<> struct rvv<float>
{
    static inline vfloat32m4_t vmerge(vfloat32m4_t a, float b, vbool8_t c, size_t d) { return __riscv_vfmerge(a, b, c, d); }
};
template<> struct rvv<double>
{
    static inline vfloat64m4_t vmerge(vfloat64m4_t a, double b, vbool16_t c, size_t d) { return __riscv_vfmerge(a, b, c, d); }
};

// the algorithm is copied from imgproc/src/thresh.cpp,
// in the functor ThresholdRunner
template<typename helper, int type, typename T = typename helper::ElemType>
static inline int threshold(int start, int end, const uchar* src_data, size_t src_step, uchar* dst_data, size_t dst_step, int width, T tval, T mval)
{
    auto zero = helper::vmv(0, helper::setvlmax());
    for (int i = start; i < end; i++)
    {
        const T* src = reinterpret_cast<const T*>(src_data + i * src_step);
        T* dst = reinterpret_cast<T*>(dst_data + i * dst_step);
        int vl0, vl1;
        for (int j = 0; j < width; j += vl0 + vl1)
        {
            vl0 = helper::setvl(width - j);
            vl1 = helper::setvl(width - j - vl0);
            auto src0 = helper::vload(src + j, vl0);
            auto src1 = helper::vload(src + j + vl0, vl1);

            typename helper::VecType dst0, dst1;
            switch (type)
            {
            case CV_HAL_THRESH_BINARY:
                dst0 = rvv<T>::vmerge(zero, mval, helper::vmgt(src0, tval, vl0), vl0);
                dst1 = rvv<T>::vmerge(zero, mval, helper::vmgt(src1, tval, vl1), vl1);
                break;
            case CV_HAL_THRESH_BINARY_INV:
                dst0 = rvv<T>::vmerge(zero, mval, helper::vmle(src0, tval, vl0), vl0);
                dst1 = rvv<T>::vmerge(zero, mval, helper::vmle(src1, tval, vl1), vl1);
                break;
            case CV_HAL_THRESH_TRUNC:
                dst0 = rvv<T>::vmerge(src0, tval, helper::vmgt(src0, tval, vl0), vl0);
                dst1 = rvv<T>::vmerge(src1, tval, helper::vmgt(src1, tval, vl1), vl1);
                break;
            case CV_HAL_THRESH_TOZERO:
                dst0 = rvv<T>::vmerge(src0, 0, helper::vmle(src0, tval, vl0), vl0);
                dst1 = rvv<T>::vmerge(src1, 0, helper::vmle(src1, tval, vl1), vl1);
                break;
            case CV_HAL_THRESH_TOZERO_INV:
                dst0 = rvv<T>::vmerge(src0, 0, helper::vmgt(src0, tval, vl0), vl0);
                dst1 = rvv<T>::vmerge(src1, 0, helper::vmgt(src1, tval, vl1), vl1);
                break;
            }
            helper::vstore(dst + j, dst0, vl0);
            helper::vstore(dst + j + vl0, dst1, vl1);
        }
    }

    return CV_HAL_ERROR_OK;
}

static inline int threshold_range(int start, int end, const uchar* src_data, size_t src_step, uchar* dst_data, size_t dst_step, int width, int depth, int cn, double thresh, double maxValue, int thresholdType)
{
    auto saturate_8u = [](double x){ return static_cast<uchar>(std::min(std::max(x, static_cast<double>(std::numeric_limits<uchar>::lowest())), static_cast<double>(std::numeric_limits<uchar>::max()))); };
    auto saturate_16s = [](double x){ return static_cast<short>(std::min(std::max(x, static_cast<double>(std::numeric_limits<short>::lowest())), static_cast<double>(std::numeric_limits<short>::max()))); };

    width *= cn;
    switch (depth)
    {
    case CV_8U:
        switch (thresholdType)
        {
        case CV_HAL_THRESH_BINARY:
            return threshold<RVV_U8M4, CV_HAL_THRESH_BINARY>(start, end, src_data, src_step, dst_data, dst_step, width, saturate_8u(std::floor(thresh)), saturate_8u(std::round(maxValue)));
        case CV_HAL_THRESH_BINARY_INV:
            return threshold<RVV_U8M4, CV_HAL_THRESH_BINARY_INV>(start, end, src_data, src_step, dst_data, dst_step, width, saturate_8u(std::floor(thresh)), saturate_8u(std::round(maxValue)));
        case CV_HAL_THRESH_TRUNC:
            return threshold<RVV_U8M4, CV_HAL_THRESH_TRUNC>(start, end, src_data, src_step, dst_data, dst_step, width, saturate_8u(std::floor(thresh)), saturate_8u(std::round(maxValue)));
        case CV_HAL_THRESH_TOZERO:
            return threshold<RVV_U8M4, CV_HAL_THRESH_TOZERO>(start, end, src_data, src_step, dst_data, dst_step, width, saturate_8u(std::floor(thresh)), saturate_8u(std::round(maxValue)));
        case CV_HAL_THRESH_TOZERO_INV:
            return threshold<RVV_U8M4, CV_HAL_THRESH_TOZERO_INV>(start, end, src_data, src_step, dst_data, dst_step, width, saturate_8u(std::floor(thresh)), saturate_8u(std::round(maxValue)));
        }
        break;
    case CV_16S:
        switch (thresholdType)
        {
        case CV_HAL_THRESH_BINARY:
            return threshold<RVV_I16M4, CV_HAL_THRESH_BINARY>(start, end, src_data, src_step, dst_data, dst_step, width, saturate_16s(std::floor(thresh)), saturate_16s(std::round(maxValue)));
        case CV_HAL_THRESH_BINARY_INV:
            return threshold<RVV_I16M4, CV_HAL_THRESH_BINARY_INV>(start, end, src_data, src_step, dst_data, dst_step, width, saturate_16s(std::floor(thresh)), saturate_16s(std::round(maxValue)));
        case CV_HAL_THRESH_TRUNC:
            return threshold<RVV_I16M4, CV_HAL_THRESH_TRUNC>(start, end, src_data, src_step, dst_data, dst_step, width, saturate_16s(std::floor(thresh)), saturate_16s(std::round(maxValue)));
        case CV_HAL_THRESH_TOZERO:
            return threshold<RVV_I16M4, CV_HAL_THRESH_TOZERO>(start, end, src_data, src_step, dst_data, dst_step, width, saturate_16s(std::floor(thresh)), saturate_16s(std::round(maxValue)));
        case CV_HAL_THRESH_TOZERO_INV:
            return threshold<RVV_I16M4, CV_HAL_THRESH_TOZERO_INV>(start, end, src_data, src_step, dst_data, dst_step, width, saturate_16s(std::floor(thresh)), saturate_16s(std::round(maxValue)));
        }
        break;
    case CV_32F:
        switch (thresholdType)
        {
        case CV_HAL_THRESH_BINARY:
            return threshold<RVV_F32M4, CV_HAL_THRESH_BINARY>(start, end, src_data, src_step, dst_data, dst_step, width, static_cast<float>(thresh), static_cast<float>(maxValue));
        case CV_HAL_THRESH_BINARY_INV:
            return threshold<RVV_F32M4, CV_HAL_THRESH_BINARY_INV>(start, end, src_data, src_step, dst_data, dst_step, width, static_cast<float>(thresh), static_cast<float>(maxValue));
        case CV_HAL_THRESH_TRUNC:
            return threshold<RVV_F32M4, CV_HAL_THRESH_TRUNC>(start, end, src_data, src_step, dst_data, dst_step, width, static_cast<float>(thresh), static_cast<float>(maxValue));
        case CV_HAL_THRESH_TOZERO:
            return threshold<RVV_F32M4, CV_HAL_THRESH_TOZERO>(start, end, src_data, src_step, dst_data, dst_step, width, static_cast<float>(thresh), static_cast<float>(maxValue));
        case CV_HAL_THRESH_TOZERO_INV:
            return threshold<RVV_F32M4, CV_HAL_THRESH_TOZERO_INV>(start, end, src_data, src_step, dst_data, dst_step, width, static_cast<float>(thresh), static_cast<float>(maxValue));
        }
        break;
    case CV_64F:
        switch (thresholdType)
        {
        case CV_HAL_THRESH_BINARY:
            return threshold<RVV_F64M4, CV_HAL_THRESH_BINARY>(start, end, src_data, src_step, dst_data, dst_step, width, thresh, maxValue);
        case CV_HAL_THRESH_BINARY_INV:
            return threshold<RVV_F64M4, CV_HAL_THRESH_BINARY_INV>(start, end, src_data, src_step, dst_data, dst_step, width, thresh, maxValue);
        case CV_HAL_THRESH_TRUNC:
            return threshold<RVV_F64M4, CV_HAL_THRESH_TRUNC>(start, end, src_data, src_step, dst_data, dst_step, width, thresh, maxValue);
        case CV_HAL_THRESH_TOZERO:
            return threshold<RVV_F64M4, CV_HAL_THRESH_TOZERO>(start, end, src_data, src_step, dst_data, dst_step, width, thresh, maxValue);
        case CV_HAL_THRESH_TOZERO_INV:
            return threshold<RVV_F64M4, CV_HAL_THRESH_TOZERO_INV>(start, end, src_data, src_step, dst_data, dst_step, width, thresh, maxValue);
        }
        break;
    }
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

inline int threshold(const uchar* src_data, size_t src_step, uchar* dst_data, size_t dst_step, int width, int height, int depth, int cn, double thresh, double maxValue, int thresholdType)
{
    return threshold_range(0, height, src_data, src_step, dst_data, dst_step, width, depth, cn, thresh, maxValue, thresholdType);
}
} // cv::cv_hal_rvv::threshold

namespace threshold_otsu {
#undef cv_hal_threshold_otsu
#define cv_hal_threshold_otsu cv::cv_hal_rvv::threshold_otsu::threshold_otsu

static inline int otsu(int start, int end, const uchar* src_data, size_t src_step, int width, std::atomic<int>* cnt, int N, int* h)
{
    const int c = cnt->fetch_add(1) % cv::getNumThreads();
    h += c * N;

    for (int i = start; i < end; i++)
    {
        for (int j = 0; j < width; j++)
            h[src_data[i * src_step + j]]++;
    }
    return CV_HAL_ERROR_OK;
}

// the algorithm is copied from imgproc/src/thresh.cpp,
// in the function template static double getThreshVal_Otsu
inline int threshold_otsu(const uchar* src_data, size_t src_step, uchar* dst_data, size_t dst_step, int width, int height, int depth, double maxValue, int thresholdType, double* thresh)
{
    if (depth != CV_8UC1 || width * height < (1 << 15))
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    const int N = std::numeric_limits<uchar>::max() + 1;
    const int nums = cv::getNumThreads();
    std::vector<int> _h(N * nums, 0);
    int* h = _h.data();

    std::atomic<int> cnt(0);
    cv::parallel_for_(Range(0, height), threshold::ThresholdInvoker({otsu}, src_data, src_step, width, &cnt, N, h), nums);
    for (int i = N; i < nums * N; i++)
    {
        h[i % N] += h[i];
    }

    double mu = 0, scale = 1. / (width*height);
    for (int i = 0; i < N; i++)
    {
        mu += i*(double)h[i];
    }

    mu *= scale;
    double mu1 = 0, q1 = 0;
    double max_sigma = 0, max_val = 0;

    for (int i = 0; i < N; i++)
    {
        double p_i, q2, mu2, sigma;

        p_i = h[i]*scale;
        mu1 *= q1;
        q1 += p_i;
        q2 = 1. - q1;

        if (std::min(q1,q2) < FLT_EPSILON || std::max(q1,q2) > 1. - FLT_EPSILON)
            continue;

        mu1 = (mu1 + i*p_i)/q1;
        mu2 = (mu - q1*mu1)/q2;
        sigma = q1*q2*(mu1 - mu2)*(mu1 - mu2);
        if (sigma > max_sigma)
        {
            max_sigma = sigma;
            max_val = i;
        }
    }

    *thresh = max_val;
    if (dst_data == nullptr)
        return CV_HAL_ERROR_OK;

    return threshold::invoke(width, height, {threshold::threshold_range}, src_data, src_step, dst_data, dst_step, width, depth, 1, max_val, maxValue, thresholdType);
}
} // cv::cv_hal_rvv::threshold_otsu

namespace adaptiveThreshold {
#undef cv_hal_adaptiveThreshold
#define cv_hal_adaptiveThreshold cv::cv_hal_rvv::adaptiveThreshold::adaptiveThreshold

// the algorithm is copied from imgproc/src/thresh.cpp,
// in the function void cv::adaptiveThreshold
template<int ksize, int method, int type>
static inline int adaptiveThreshold(int start, int end, const uchar* src_data, size_t src_step, uchar* dst_data, size_t dst_step, int width, int height, double maxValue, double C)
{
    auto saturate = [](double x){ return static_cast<uchar>(std::min(std::max(x, static_cast<double>(std::numeric_limits<uchar>::lowest())), static_cast<double>(std::numeric_limits<uchar>::max()))); };
    uchar mval = saturate(std::round(maxValue));

    if (method == CV_HAL_ADAPTIVE_THRESH_MEAN_C)
    {
        int cval = static_cast<int>(std::round(C));
        if (cval != C)
            return CV_HAL_ERROR_NOT_IMPLEMENTED;

        std::vector<short> res(width * ksize);
        auto process = [&](int x, int y) {
            int sum = 0;
            for (int i = 0; i < ksize; i++)
            {
                int q = std::min(std::max(y + i - ksize / 2, 0), width - 1);
                sum += src_data[x * src_step + q];
            }
            res[(x % ksize) * width + y] = sum;
        };

        const int left = ksize - 1, right = width - (ksize - 1);
        for (int i = start - ksize / 2; i < end + ksize / 2; i++)
        {
            if (i >= 0 && i < height)
            {
                for (int j = 0; j < left; j++)
                    process(i, j);
                for (int j = right; j < width; j++)
                    process(i, j);

                int vl;
                for (int j = left; j < right; j += vl)
                {
                    vl = __riscv_vsetvl_e8m4(right - j);
                    const uchar* row = src_data + i * src_step + j - ksize / 2;
                    auto src = __riscv_vreinterpret_v_u16m8_i16m8(__riscv_vzext_vf2(__riscv_vle8_v_u8m4(row, vl), vl));
                    auto sum = src;
                    src = __riscv_vslide1down(src, row[vl], vl);
                    sum = __riscv_vadd(sum, src, vl);
                    src = __riscv_vslide1down(src, row[vl + 1], vl);
                    sum = __riscv_vadd(sum, src, vl);
                    if (ksize == 5)
                    {
                        src = __riscv_vslide1down(src, row[vl + 2], vl);
                        sum = __riscv_vadd(sum, src, vl);
                        src = __riscv_vslide1down(src, row[vl + 3], vl);
                        sum = __riscv_vadd(sum, src, vl);
                    }
                    __riscv_vse16(res.data() + (i % ksize) * width + j, sum, vl);
                }
            }

            int cur = i - ksize / 2;
            if (cur >= start)
            {
                const short* row0 = res.data() + std::min(std::max(cur     - ksize / 2, 0), height - 1) % ksize * width;
                const short* row1 = res.data() + std::min(std::max(cur + 1 - ksize / 2, 0), height - 1) % ksize * width;
                const short* row2 = res.data() + std::min(std::max(cur + 2 - ksize / 2, 0), height - 1) % ksize * width;
                const short* row3 = res.data() + std::min(std::max(cur + 3 - ksize / 2, 0), height - 1) % ksize * width;
                const short* row4 = res.data() + std::min(std::max(cur + 4 - ksize / 2, 0), height - 1) % ksize * width;
                int vl;
                for (int j = 0; j < width; j += vl)
                {
                    vl = __riscv_vsetvl_e16m8(width - j);
                    auto sum = __riscv_vle16_v_i16m8(row0 + j, vl);
                    sum = __riscv_vadd(sum, __riscv_vle16_v_i16m8(row1 + j, vl), vl);
                    sum = __riscv_vadd(sum, __riscv_vle16_v_i16m8(row2 + j, vl), vl);
                    if (ksize == 5)
                    {
                        sum = __riscv_vadd(sum, __riscv_vle16_v_i16m8(row3 + j, vl), vl);
                        sum = __riscv_vadd(sum, __riscv_vle16_v_i16m8(row4 + j, vl), vl);
                    }
                    auto mean = __riscv_vsub(__riscv_vdiv(sum, ksize * ksize, vl), cval, vl);
                    auto cmp = __riscv_vmsgt(__riscv_vreinterpret_v_u16m8_i16m8(__riscv_vzext_vf2(__riscv_vle8_v_u8m4(src_data + cur * src_step + j, vl), vl)), mean, vl);
                    if (type == CV_HAL_THRESH_BINARY)
                    {
                        __riscv_vse8(dst_data + cur * dst_step + j, __riscv_vmerge(__riscv_vmv_v_x_u8m4(0, vl), mval, cmp, vl), vl);
                    }
                    else
                    {
                        __riscv_vse8(dst_data + cur * dst_step + j, __riscv_vmerge(__riscv_vmv_v_x_u8m4(mval, vl), 0, cmp, vl), vl);
                    }
                }
            }
        }
    }
    else
    {
        constexpr float kernel[2][5] = {{0.25f, 0.5f, 0.25f}, {0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f}};
        std::vector<float> res(width * ksize);
        auto process = [&](int x, int y) {
            float sum = 0;
            for (int i = 0; i < ksize; i++)
            {
                int q = std::min(std::max(y + i - ksize / 2, 0), width - 1);
                sum += kernel[ksize == 5][i] * src_data[x * src_step + q];
            }
            res[(x % ksize) * width + y] = sum;
        };

        const int left = ksize - 1, right = width - (ksize - 1);
        for (int i = start - ksize / 2; i < end + ksize / 2; i++)
        {
            if (i >= 0 && i < height)
            {
                for (int j = 0; j < left; j++)
                    process(i, j);
                for (int j = right; j < width; j++)
                    process(i, j);

                int vl;
                for (int j = left; j < right; j += vl)
                {
                    vl = __riscv_vsetvl_e8m2(right - j);
                    const uchar* row = src_data + i * src_step + j - ksize / 2;
                    auto src = __riscv_vfwcvt_f(__riscv_vzext_vf2(__riscv_vle8_v_u8m2(row, vl), vl), vl);
                    auto sum = __riscv_vfmul(src, kernel[ksize == 5][0], vl);
                    src = __riscv_vfslide1down(src, row[vl], vl);
                    sum = __riscv_vfmacc(sum, kernel[ksize == 5][1], src, vl);
                    src = __riscv_vfslide1down(src, row[vl + 1], vl);
                    sum = __riscv_vfmacc(sum, kernel[ksize == 5][2], src, vl);
                    if (ksize == 5)
                    {
                        src = __riscv_vfslide1down(src, row[vl + 2], vl);
                        sum = __riscv_vfmacc(sum, kernel[1][3], src, vl);
                        src = __riscv_vfslide1down(src, row[vl + 3], vl);
                        sum = __riscv_vfmacc(sum, kernel[1][4], src, vl);
                    }
                    __riscv_vse32(res.data() + (i % ksize) * width + j, sum, vl);
                }
            }

            int cur = i - ksize / 2;
            if (cur >= start)
            {
                const float* row0 = res.data() + std::min(std::max(cur     - ksize / 2, 0), height - 1) % ksize * width;
                const float* row1 = res.data() + std::min(std::max(cur + 1 - ksize / 2, 0), height - 1) % ksize * width;
                const float* row2 = res.data() + std::min(std::max(cur + 2 - ksize / 2, 0), height - 1) % ksize * width;
                const float* row3 = res.data() + std::min(std::max(cur + 3 - ksize / 2, 0), height - 1) % ksize * width;
                const float* row4 = res.data() + std::min(std::max(cur + 4 - ksize / 2, 0), height - 1) % ksize * width;
                int vl;
                for (int j = 0; j < width; j += vl)
                {
                    vl = __riscv_vsetvl_e32m8(width - j);
                    auto sum = __riscv_vfmv_v_f_f32m8(-C, vl);
                    sum = __riscv_vfmacc(sum, kernel[ksize == 5][0], __riscv_vle32_v_f32m8(row0 + j, vl), vl);
                    sum = __riscv_vfmacc(sum, kernel[ksize == 5][1], __riscv_vle32_v_f32m8(row1 + j, vl), vl);
                    sum = __riscv_vfmacc(sum, kernel[ksize == 5][2], __riscv_vle32_v_f32m8(row2 + j, vl), vl);
                    if (ksize == 5)
                    {
                        sum = __riscv_vfmacc(sum, kernel[1][3], __riscv_vle32_v_f32m8(row3 + j, vl), vl);
                        sum = __riscv_vfmacc(sum, kernel[1][4], __riscv_vle32_v_f32m8(row4 + j, vl), vl);
                    }
                    auto mean = __riscv_vnclipu(__riscv_vfncvt_rtz_xu(sum, vl), 0, __RISCV_VXRM_RNU, vl);
                    auto cmp = __riscv_vmsgtu(__riscv_vle8_v_u8m2(src_data + cur * src_step + j, vl), mean, vl);
                    if (type == CV_HAL_THRESH_BINARY)
                    {
                        __riscv_vse8(dst_data + cur * dst_step + j, __riscv_vmerge(__riscv_vmv_v_x_u8m2(0, vl), mval, cmp, vl), vl);
                    }
                    else
                    {
                        __riscv_vse8(dst_data + cur * dst_step + j, __riscv_vmerge(__riscv_vmv_v_x_u8m2(mval, vl), 0, cmp, vl), vl);
                    }
                }
            }
        }
    }

    return CV_HAL_ERROR_OK;
}

inline int adaptiveThreshold(const uchar* src_data, size_t src_step, uchar* dst_data, size_t dst_step, int width, int height, double maxValue, int adaptiveMethod, int thresholdType, int blockSize, double C)
{
    if (thresholdType != CV_HAL_THRESH_BINARY && thresholdType != CV_HAL_THRESH_BINARY_INV)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if (adaptiveMethod != CV_HAL_ADAPTIVE_THRESH_MEAN_C && adaptiveMethod != CV_HAL_ADAPTIVE_THRESH_GAUSSIAN_C)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if ((blockSize != 3 && blockSize != 5) || width < blockSize * 2)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    switch (blockSize*100 + adaptiveMethod*10 + thresholdType)
    {
    case 300 + CV_HAL_ADAPTIVE_THRESH_MEAN_C*10 + CV_HAL_THRESH_BINARY:
        return threshold::invoke(width, height, {adaptiveThreshold<3, CV_HAL_ADAPTIVE_THRESH_MEAN_C, CV_HAL_THRESH_BINARY>}, src_data, src_step, dst_data, dst_step, width, height, maxValue, C);
    case 300 + CV_HAL_ADAPTIVE_THRESH_MEAN_C*10 + CV_HAL_THRESH_BINARY_INV:
        return threshold::invoke(width, height, {adaptiveThreshold<3, CV_HAL_ADAPTIVE_THRESH_MEAN_C, CV_HAL_THRESH_BINARY_INV>}, src_data, src_step, dst_data, dst_step, width, height, maxValue, C);
    case 500 + CV_HAL_ADAPTIVE_THRESH_MEAN_C*10 + CV_HAL_THRESH_BINARY:
        return threshold::invoke(width, height, {adaptiveThreshold<5, CV_HAL_ADAPTIVE_THRESH_MEAN_C, CV_HAL_THRESH_BINARY>}, src_data, src_step, dst_data, dst_step, width, height, maxValue, C);
    case 500 + CV_HAL_ADAPTIVE_THRESH_MEAN_C*10 + CV_HAL_THRESH_BINARY_INV:
        return threshold::invoke(width, height, {adaptiveThreshold<5, CV_HAL_ADAPTIVE_THRESH_MEAN_C, CV_HAL_THRESH_BINARY_INV>}, src_data, src_step, dst_data, dst_step, width, height, maxValue, C);
    case 300 + CV_HAL_ADAPTIVE_THRESH_GAUSSIAN_C*10 + CV_HAL_THRESH_BINARY:
        return threshold::invoke(width, height, {adaptiveThreshold<3, CV_HAL_ADAPTIVE_THRESH_GAUSSIAN_C, CV_HAL_THRESH_BINARY>}, src_data, src_step, dst_data, dst_step, width, height, maxValue, C);
    case 300 + CV_HAL_ADAPTIVE_THRESH_GAUSSIAN_C*10 + CV_HAL_THRESH_BINARY_INV:
        return threshold::invoke(width, height, {adaptiveThreshold<3, CV_HAL_ADAPTIVE_THRESH_GAUSSIAN_C, CV_HAL_THRESH_BINARY_INV>}, src_data, src_step, dst_data, dst_step, width, height, maxValue, C);
    case 500 + CV_HAL_ADAPTIVE_THRESH_GAUSSIAN_C*10 + CV_HAL_THRESH_BINARY:
        return threshold::invoke(width, height, {adaptiveThreshold<5, CV_HAL_ADAPTIVE_THRESH_GAUSSIAN_C, CV_HAL_THRESH_BINARY>}, src_data, src_step, dst_data, dst_step, width, height, maxValue, C);
    case 500 + CV_HAL_ADAPTIVE_THRESH_GAUSSIAN_C*10 + CV_HAL_THRESH_BINARY_INV:
        return threshold::invoke(width, height, {adaptiveThreshold<5, CV_HAL_ADAPTIVE_THRESH_GAUSSIAN_C, CV_HAL_THRESH_BINARY_INV>}, src_data, src_step, dst_data, dst_step, width, height, maxValue, C);
    }

    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}
} // cv::cv_hal_rvv::adaptiveThreshold

}}

#endif
