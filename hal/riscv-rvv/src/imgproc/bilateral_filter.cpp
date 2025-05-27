// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2025, Institute of Software, Chinese Academy of Sciences.

#include "rvv_hal.hpp"
#include "common.hpp"

namespace cv { namespace rvv_hal { namespace imgproc {

#if CV_HAL_RVV_1P0_ENABLED

namespace {

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

} // anonymous

// the algorithm is copied from imgproc/src/bilateral_filter.dispatch.cpp
// in the function static void bilateralFilter_8u and bilateralFilter_32f
int bilateralFilter(const uchar* src_data, size_t src_step, uchar* dst_data, size_t dst_step,
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
        width_interpolate[j] = common::borderInterpolate(j - radius, width, border_type);
        width_interpolate[j + radius] = common::borderInterpolate(width + j, width, border_type);
    }
    for (int i = 0; i < height + radius * 2; i++)
    {
        int x = common::borderInterpolate(i - radius, height, border_type);
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
    cv::rvv_hal::core::exp32f(space_weight, space_weight, maxk);

    if (depth == CV_8U)
    {
        std::vector<float> _color_weight(cn*256);
        float* color_weight = _color_weight.data();
        for (int i = 0; i < 256*cn; i++)
            color_weight[i] = static_cast<float>(i*i*gauss_color_coeff);
        cv::rvv_hal::core::exp32f(color_weight, color_weight, 256*cn);

        switch (cn)
        {
        case 1:
            return common::invoke(height, {bilateralFilter8UC1}, temp, temp_step, dst_data, dst_step, width, radius, maxk, space_ofs, space_weight, color_weight);
        case 3:
            return common::invoke(height, {bilateralFilter8UC3}, temp, temp_step, dst_data, dst_step, width, radius, maxk, space_ofs, space_weight, color_weight);
        }
    }
    else
    {
        double minValSrc = -1, maxValSrc = 1;
        cv::rvv_hal::core::minMaxIdx(src_data, src_step, width * cn, height, CV_32F, &minValSrc, &maxValSrc, nullptr, nullptr, nullptr);
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
        cv::rvv_hal::core::exp32f(expLUT, expLUT, kExpNumBins+2);

        switch (cn)
        {
        case 1:
            return common::invoke(height, {bilateralFilter32FC1}, temp, temp_step, dst_data, dst_step, width, radius, maxk, space_ofs, space_weight, expLUT, scale_index);
        case 3:
            return common::invoke(height, {bilateralFilter32FC3}, temp, temp_step, dst_data, dst_step, width, radius, maxk, space_ofs, space_weight, expLUT, scale_index);
        }
    }

    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

#endif // CV_HAL_RVV_1P0_ENABLED

}}} // cv::rvv_hal::imgproc
