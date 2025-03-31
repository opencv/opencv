// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2025, SpaceMIT Inc., all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_HAL_RVV_DIV_HPP_INCLUDED
#define OPENCV_HAL_RVV_DIV_HPP_INCLUDED

#include <riscv_vector.h>

#include <limits>

namespace cv { namespace cv_hal_rvv { namespace div {

namespace {

    template <typename ST> inline size_t setvl(int);
    template<> inline size_t setvl<uint8_t>(int l)  { return __riscv_vsetvl_e8m1(l); }
    template<> inline size_t setvl<int8_t>(int l)   { return __riscv_vsetvl_e8m1(l); }
    template<> inline size_t setvl<uint16_t>(int l) { return __riscv_vsetvl_e16m2(l); }
    template<> inline size_t setvl<int16_t>(int l)  { return __riscv_vsetvl_e16m2(l); }
    template<> inline size_t setvl<int>(int l)      { return __riscv_vsetvl_e32m4(l); }
    template<> inline size_t setvl<float>(int l)    { return __riscv_vsetvl_e32m4(l); }
    // template<> inline size_t setvl<double>(int l)   { return __riscv_vsetvl_e64m1(l); }

    inline vuint8m1_t   vle(const uint8_t  *p, int vl) { return __riscv_vle8_v_u8m1(p, vl); }
    inline vint8m1_t    vle(const int8_t   *p, int vl) { return __riscv_vle8_v_i8m1(p, vl); }
    inline vuint16m2_t  vle(const uint16_t *p, int vl) { return __riscv_vle16_v_u16m2(p, vl); }
    inline vint16m2_t   vle(const int16_t  *p, int vl) { return __riscv_vle16_v_i16m2(p, vl); }
    inline vint32m4_t   vle(const int      *p, int vl) { return __riscv_vle32_v_i32m4(p, vl); }
    inline vfloat32m4_t vle(const float    *p, int vl) { return __riscv_vle32_v_f32m4(p, vl); }
    // inline vfloat64m1_t vle(const double   *p, int vl) { return __riscv_vle64_v_f64m1(p, vl); }

    inline void vse(uint8_t  *p, const vuint8m1_t   &v, int vl) { __riscv_vse8_v_u8m1(p, v, vl); }
    inline void vse(int8_t   *p, const vint8m1_t    &v, int vl) { __riscv_vse8_v_i8m1(p, v, vl); }
    inline void vse(uint16_t *p, const vuint16m2_t  &v, int vl) { __riscv_vse16_v_u16m2(p, v, vl); }
    inline void vse(int16_t  *p, const vint16m2_t   &v, int vl) { __riscv_vse16_v_i16m2(p, v, vl); }
    inline void vse(int      *p, const vint32m4_t   &v, int vl) { __riscv_vse32_v_i32m4(p, v, vl); }
    inline void vse(float    *p, const vfloat32m4_t &v, int vl) { __riscv_vse32_v_f32m4(p, v, vl); }
    // inline void vse(double   *p, const vfloat64m1_t &v, int vl) { __riscv_vse64_v_f64m1(p, v, vl); }

    inline vuint16m2_t ext(const vuint8m1_t  &v, const int vl) { return __riscv_vzext_vf2(v, vl); }
    inline vint16m2_t  ext(const vint8m1_t   &v, const int vl) { return __riscv_vsext_vf2(v, vl); }
    inline vuint32m4_t ext(const vuint16m2_t &v, const int vl) { return __riscv_vzext_vf2(v, vl); }
    inline vint32m4_t  ext(const vint16m2_t  &v, const int vl) { return __riscv_vsext_vf2(v, vl); }

    inline vuint8m1_t  nclip(const vuint16m2_t &v, const int vl) { return __riscv_vnclipu(v, 0, __RISCV_VXRM_RDN, vl); }
    inline vint8m1_t   nclip(const vint16m2_t  &v, const int vl) { return __riscv_vnclip(v, 0, __RISCV_VXRM_RDN, vl); }
    inline vuint16m2_t nclip(const vuint32m4_t &v, const int vl) { return __riscv_vnclipu(v, 0, __RISCV_VXRM_RDN, vl); }
    inline vint16m2_t  nclip(const vint32m4_t  &v, const int vl) { return __riscv_vnclip(v, 0, __RISCV_VXRM_RDN, vl); }

    template <typename VT> inline
    VT div_sat(const VT &v1, const VT &v2, const float scale, const int vl) {
        return nclip(div_sat(ext(v1, vl), ext(v2, vl), scale, vl), vl);
    }
    template <> inline
    vint32m4_t div_sat(const vint32m4_t &v1, const vint32m4_t &v2, const float scale, const int vl) {
        auto f1 = __riscv_vfcvt_f(v1, vl);
        auto f2 = __riscv_vfcvt_f(v2, vl);
        auto res = __riscv_vfmul(__riscv_vfmul(f1, scale, vl), custom_intrin::__riscv_vfrecprocal(f2, vl), vl);
        return __riscv_vfcvt_x(res, vl);
    }
    template <> inline
    vuint32m4_t div_sat(const vuint32m4_t &v1, const vuint32m4_t &v2, const float scale, const int vl) {
        auto f1 = __riscv_vfcvt_f(v1, vl);
        auto f2 = __riscv_vfcvt_f(v2, vl);
        auto res = __riscv_vfmul(__riscv_vfmul(f1, scale, vl), custom_intrin::__riscv_vfrecprocal(f2, vl), vl);
        return __riscv_vfcvt_xu(res, vl);
    }

    template <typename VT> inline
    VT recip_sat(const VT &v, const float scale, const int vl) {
        return nclip(recip_sat(ext(v, vl), scale, vl), vl);
    }
    template <> inline
    vint32m4_t recip_sat(const vint32m4_t &v, const float scale, const int vl) {
        auto f = __riscv_vfcvt_f(v, vl);
        auto res = __riscv_vfmul(custom_intrin::__riscv_vfrecprocal(f, vl), scale, vl);
        return __riscv_vfcvt_x(res, vl);
    }
    template <> inline
    vuint32m4_t recip_sat(const vuint32m4_t &v, const float scale, const int vl) {
        auto f = __riscv_vfcvt_f(v, vl);
        auto res = __riscv_vfmul(custom_intrin::__riscv_vfrecprocal(f, vl), scale, vl);
        return __riscv_vfcvt_xu(res, vl);
    }

} // anonymous

#undef cv_hal_div8u
#define cv_hal_div8u cv::cv_hal_rvv::div::div<uint8_t>
#undef cv_hal_div8s
#define cv_hal_div8s cv::cv_hal_rvv::div::div<int8_t>
#undef cv_hal_div16u
#define cv_hal_div16u cv::cv_hal_rvv::div::div<uint16_t>
#undef cv_hal_div16s
#define cv_hal_div16s cv::cv_hal_rvv::div::div<int16_t>
#undef cv_hal_div32s
#define cv_hal_div32s cv::cv_hal_rvv::div::div<int>
#undef cv_hal_div32f
#define cv_hal_div32f cv::cv_hal_rvv::div::div<float>
// #undef cv_hal_div64f
// #define cv_hal_div64f cv::cv_hal_rvv::div::div<double>

template <typename ST> inline
int div(const ST *src1, size_t step1, const ST *src2, size_t step2,
         ST *dst, size_t step, int width, int height, float scale) {
    if (scale == 0.f ||
        (scale * static_cast<float>(std::numeric_limits<ST>::max())) <  1.f &&
        (scale * static_cast<float>(std::numeric_limits<ST>::max())) > -1.f) {
        for (int h = 0; h < height; h++) {
            ST *dst_h = reinterpret_cast<ST*>((uchar*)dst + h * step);
            std::memset(dst_h, 0, sizeof(ST) * width);
        }
        return CV_HAL_ERROR_OK;
    }

    for (int h = 0; h < height; h++) {
        const ST *src1_h = reinterpret_cast<const ST*>((const uchar*)src1 + h * step1);
        const ST *src2_h = reinterpret_cast<const ST*>((const uchar*)src2 + h * step2);
        ST *dst_h = reinterpret_cast<ST*>((uchar*)dst + h * step);

        int vl;
        for (int w = 0; w < width; w += vl) {
            vl = setvl<ST>(width - w);

            auto v1 = vle(src1_h + w, vl);
            auto v2 = vle(src2_h + w, vl);

            vse(dst_h + w, div_sat(v1, v2, scale, vl), vl);
        }
    }

    return CV_HAL_ERROR_OK;
}

template <> inline
int div(const float *src1, size_t step1, const float *src2, size_t step2,
        float *dst, size_t step, int width, int height, float scale) {
    if (scale == 0.f) {
        for (int h = 0; h < height; h++) {
            float *dst_h = reinterpret_cast<float*>((uchar*)dst + h * step);
            std::memset(dst_h, 0, sizeof(float) * width);
        }
        return CV_HAL_ERROR_OK;
    }

    if (std::fabs(scale - 1.f) < FLT_EPSILON) {
        for (int h = 0; h < height; h++) {
            const float *src1_h = reinterpret_cast<const float*>((const uchar*)src1 + h * step1);
            const float *src2_h = reinterpret_cast<const float*>((const uchar*)src2 + h * step2);
            float *dst_h = reinterpret_cast<float*>((uchar*)dst + h * step);

            int vl;
            for (int w = 0; w < width; w += vl) {
                vl = setvl<float>(width - w);

                auto v1 = vle(src1_h + w, vl);
                auto v2 = vle(src2_h + w, vl);

                vse(dst_h + w, __riscv_vfmul(v1, custom_intrin::__riscv_vfrecprocal(v2, vl), vl), vl);
            }
        }
    } else {
        for (int h = 0; h < height; h++) {
            const float *src1_h = reinterpret_cast<const float*>((const uchar*)src1 + h * step1);
            const float *src2_h = reinterpret_cast<const float*>((const uchar*)src2 + h * step2);
            float *dst_h = reinterpret_cast<float*>((uchar*)dst + h * step);

            int vl;
            for (int w = 0; w < width; w += vl) {
                vl = setvl<float>(width - w);

                auto v1 = vle(src1_h + w, vl);
                auto v2 = vle(src2_h + w, vl);

                vse(dst_h + w, __riscv_vfmul(__riscv_vfmul(v1, scale, vl), custom_intrin::__riscv_vfrecprocal(v2, vl), vl), vl);
            }
        }
    }

    return CV_HAL_ERROR_OK;
}

#undef cv_hal_recip8u
#define cv_hal_recip8u cv::cv_hal_rvv::div::recip<uint8_t>
#undef cv_hal_recip8s
#define cv_hal_recip8s cv::cv_hal_rvv::div::recip<int8_t>
#undef cv_hal_recip16u
#define cv_hal_recip16u cv::cv_hal_rvv::div::recip<uint16_t>
#undef cv_hal_recip16s
#define cv_hal_recip16s cv::cv_hal_rvv::div::recip<int16_t>
#undef cv_hal_recip32s
#define cv_hal_recip32s cv::cv_hal_rvv::div::recip<int>
#undef cv_hal_recip32f
#define cv_hal_recip32f cv::cv_hal_rvv::div::recip<float>
// #undef cv_hal_recip64f
// #define cv_hal_recip64f cv::cv_hal_rvv::div::recip<double>

template <typename ST> inline
int recip(const ST *src_data, size_t src_step, ST *dst_data, size_t dst_step,
          int width, int height, float scale) {
    if (scale == 0.f || scale < 1.f && scale > -1.f) {
        for (int h = 0; h < height; h++) {
            ST *dst_h = reinterpret_cast<ST*>((uchar*)dst_data + h * dst_step);
            std::memset(dst_h, 0, sizeof(ST) * width);
        }
        return CV_HAL_ERROR_OK;
    }

    for (int h = 0; h < height; h++) {
        const ST *src_h = reinterpret_cast<const ST*>((const uchar*)src_data + h * src_step);
        ST *dst_h = reinterpret_cast<ST*>((uchar*)dst_data + h * dst_step);

        int vl;
        for (int w = 0; w < width; w += vl) {
            vl = setvl<ST>(width - w);

            auto v = vle(src_h + w, vl);

            vse(dst_h + w, recip_sat(v, scale, vl), vl);
        }
    }

    return CV_HAL_ERROR_OK;
}

template <> inline
int recip(const float *src_data, size_t src_step, float *dst_data, size_t dst_step,
          int width, int height, float scale) {
    if (scale == 0.f) {
        for (int h = 0; h < height; h++) {
            float *dst_h = reinterpret_cast<float*>((uchar*)dst_data + h * dst_step);
            std::memset(dst_h, 0, sizeof(float) * width);
        }
        return CV_HAL_ERROR_OK;
    }

    if (std::fabs(scale - 1.f) < FLT_EPSILON) {
        for (int h = 0; h < height; h++) {
            const float *src_h = reinterpret_cast<const float*>((const uchar*)src_data + h * src_step);
            float *dst_h = reinterpret_cast<float*>((uchar*)dst_data + h * dst_step);

            int vl;
            for (int w = 0; w < width; w += vl) {
                vl = setvl<float>(width - w);

                auto v = vle(src_h + w, vl);

                vse(dst_h + w, custom_intrin::__riscv_vfrecprocal(v, vl), vl);
            }
        }
    } else {
        for (int h = 0; h < height; h++) {
            const float *src_h = reinterpret_cast<const float*>((const uchar*)src_data + h * src_step);
            float *dst_h = reinterpret_cast<float*>((uchar*)dst_data + h * dst_step);

            int vl;
            for (int w = 0; w < width; w += vl) {
                vl = setvl<float>(width - w);

                auto v = vle(src_h + w, vl);

                vse(dst_h + w, __riscv_vfmul(custom_intrin::__riscv_vfrecprocal(v, vl), scale, vl), vl);
            }
        }
    }

    return CV_HAL_ERROR_OK;
}

}}} // cv::cv_hal_rvv::div

#endif // OPENCV_HAL_RVV_DIV_HPP_INCLUDED
