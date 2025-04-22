// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2025, Institute of Software, Chinese Academy of Sciences.
// Copyright (C) 2025, SpaceMIT Inc., all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_HAL_RVV_NORM_DIFF_HPP_INCLUDED
#define OPENCV_HAL_RVV_NORM_DIFF_HPP_INCLUDED

#include "common.hpp"

namespace cv { namespace cv_hal_rvv { namespace norm_diff {

#undef cv_hal_normDiff
#define cv_hal_normDiff cv::cv_hal_rvv::norm_diff::normDiff

namespace {

template <typename T, typename ST>
struct NormDiffInf_RVV {
    inline ST operator() (const T* src1, const T* src2, int n) const {
        ST s = 0;
        for (int i = 0; i < n; i++) {
            s = std::max(s, (ST)std::abs(src1[i] - src2[i]));
        }
        return s;
    }
};

template <typename T, typename ST>
struct NormDiffL1_RVV {
    inline ST operator() (const T* src1, const T* src2, int n) const {
        ST s = 0;
        for (int i = 0; i < n; i++) {
            s += std::abs(src1[i] - src2[i]);
        }
        return s;
    }
};

template <typename T, typename ST>
struct NormDiffL2_RVV {
    inline ST operator() (const T* src1, const T* src2, int n) const {
        ST s = 0;
        for (int i = 0; i < n; i++) {
            ST v = (ST)src1[i] - (ST)src2[i];
            s += v * v;
        }
        return s;
    }
};

template<>
struct NormDiffInf_RVV<uchar, int> {
    int operator() (const uchar* src1, const uchar* src2, int n) const {
        int vlmax = __riscv_vsetvlmax_e8m8();
        auto s = __riscv_vmv_v_x_u8m8(0, vlmax);
        int vl;
        for (int i = 0; i < n; i += vl) {
            vl = __riscv_vsetvl_e8m8(n - i);
            auto v1 = __riscv_vle8_v_u8m8(src1 + i, vl);
            auto v2 = __riscv_vle8_v_u8m8(src2 + i, vl);
            auto v = custom_intrin::__riscv_vabd(v1, v2, vl);
            s = __riscv_vmaxu_tu(s, s, v, vl);
        }
        return __riscv_vmv_x(__riscv_vredmaxu(s, __riscv_vmv_s_x_u8m1(0, __riscv_vsetvlmax_e8m1()), vlmax));
    }
};

template<>
struct NormDiffInf_RVV<schar, int> {
    int operator() (const schar* src1, const schar* src2, int n) const {
        int vlmax = __riscv_vsetvlmax_e8m8();
        auto s = __riscv_vmv_v_x_u8m8(0, vlmax);
        int vl;
        for (int i = 0; i < n; i += vl) {
            vl = __riscv_vsetvl_e8m8(n - i);
            auto v1 = __riscv_vle8_v_i8m8(src1 + i, vl);
            auto v2 = __riscv_vle8_v_i8m8(src2 + i, vl);
            auto v = custom_intrin::__riscv_vabd(v1, v2, vl);
            s = __riscv_vmaxu_tu(s, s, v, vl);
        }
        return __riscv_vmv_x(__riscv_vredmaxu(s, __riscv_vmv_s_x_u8m1(0, __riscv_vsetvlmax_e8m1()), vlmax));
    }
};

template<>
struct NormDiffInf_RVV<ushort, int> {
    int operator() (const ushort* src1, const ushort* src2, int n) const {
        int vlmax = __riscv_vsetvlmax_e16m8();
        auto s = __riscv_vmv_v_x_u16m8(0, vlmax);
        int vl;
        for (int i = 0; i < n; i += vl) {
            vl = __riscv_vsetvl_e16m8(n - i);
            auto v1 = __riscv_vle16_v_u16m8(src1 + i, vl);
            auto v2 = __riscv_vle16_v_u16m8(src2 + i, vl);
            auto v = custom_intrin::__riscv_vabd(v1, v2, vl);
            s = __riscv_vmaxu_tu(s, s, v, vl);
        }
        return __riscv_vmv_x(__riscv_vredmaxu(s, __riscv_vmv_s_x_u16m1(0, __riscv_vsetvlmax_e16m1()), vlmax));
    }
};

template<>
struct NormDiffInf_RVV<short, int> {
    int operator() (const short* src1, const short* src2, int n) const {
        int vlmax = __riscv_vsetvlmax_e16m8();
        auto s = __riscv_vmv_v_x_u16m8(0, vlmax);
        int vl;
        for (int i = 0; i < n; i += vl) {
            vl = __riscv_vsetvl_e16m8(n - i);
            auto v1 = __riscv_vle16_v_i16m8(src1 + i, vl);
            auto v2 = __riscv_vle16_v_i16m8(src2 + i, vl);
            auto v = custom_intrin::__riscv_vabd(v1, v2, vl);
            s = __riscv_vmaxu_tu(s, s, v, vl);
        }
        return __riscv_vmv_x(__riscv_vredmaxu(s, __riscv_vmv_s_x_u16m1(0, __riscv_vsetvlmax_e16m1()), vlmax));
    }
};

template<>
struct NormDiffInf_RVV<int, int> {
    int operator() (const int* src1, const int* src2, int n) const {
        int vlmax = __riscv_vsetvlmax_e32m8();
        auto s = __riscv_vmv_v_x_u32m8(0, vlmax);
        int vl;
        for (int i = 0; i < n; i += vl) {
            vl = __riscv_vsetvl_e32m8(n - i);
            auto v1 = __riscv_vle32_v_i32m8(src1 + i, vl);
            auto v2 = __riscv_vle32_v_i32m8(src2 + i, vl);
            // auto v = custom_intrin::__riscv_vabd(v1, v2, vl); // 5.x
            auto v = custom_intrin::__riscv_vabs(__riscv_vsub(v1, v2, vl), vl); // 4.x
            s = __riscv_vmaxu_tu(s, s, v, vl);
        }
        return __riscv_vmv_x(__riscv_vredmaxu(s, __riscv_vmv_s_x_u32m1(0, __riscv_vsetvlmax_e32m1()), vlmax));
    }
};

template<>
struct NormDiffInf_RVV<float, float> {
    float operator() (const float* src1, const float* src2, int n) const {
        int vlmax = __riscv_vsetvlmax_e32m8();
        auto s = __riscv_vfmv_v_f_f32m8(0, vlmax);
        int vl;
        for (int i = 0; i < n; i += vl) {
            vl = __riscv_vsetvl_e32m8(n - i);
            auto v1 = __riscv_vle32_v_f32m8(src1 + i, vl);
            auto v2 = __riscv_vle32_v_f32m8(src2 + i, vl);
            auto v = __riscv_vfabs(__riscv_vfsub(v1, v2, vl), vl);
            s = __riscv_vfmax_tu(s, s, v, vl);
        }
        return __riscv_vfmv_f(__riscv_vfredmax(s, __riscv_vfmv_s_f_f32m1(0, __riscv_vsetvlmax_e32m1()), vlmax));
    }
};

template<>
struct NormDiffInf_RVV<double, double> {
    double operator() (const double* src1, const double* src2, int n) const {
        int vlmax = __riscv_vsetvlmax_e64m8();
        auto s = __riscv_vfmv_v_f_f64m8(0, vlmax);
        int vl;
        for (int i = 0; i < n; i += vl) {
            vl = __riscv_vsetvl_e64m8(n - i);
            auto v1 = __riscv_vle64_v_f64m8(src1 + i, vl);
            auto v2 = __riscv_vle64_v_f64m8(src2 + i, vl);
            auto v = __riscv_vfabs(__riscv_vfsub(v1, v2, vl), vl);
            s = __riscv_vfmax_tu(s, s, v, vl);
        }
        return __riscv_vfmv_f(__riscv_vfredmax(s, __riscv_vfmv_s_f_f64m1(0, __riscv_vsetvlmax_e64m1()), vlmax));
    }
};

template<>
struct NormDiffL1_RVV<uchar, int> {
    int operator() (const uchar* src1, const uchar* src2, int n) const {
        auto s = __riscv_vmv_v_x_u32m1(0, __riscv_vsetvlmax_e32m1());
        auto zero = __riscv_vmv_v_x_u16m1(0, __riscv_vsetvlmax_e16m1());
        int vl;
        for (int i = 0; i < n; i += vl) {
            vl = __riscv_vsetvl_e8m8(n - i);
            auto v1 = __riscv_vle8_v_u8m8(src1 + i, vl);
            auto v2 = __riscv_vle8_v_u8m8(src2 + i, vl);
            auto v = custom_intrin::__riscv_vabd(v1, v2, vl);
            s = __riscv_vwredsumu(__riscv_vwredsumu_tu(zero, v, zero, vl), s, __riscv_vsetvlmax_e16m1());
        }
        return __riscv_vmv_x(s);
    }
};

template<>
struct NormDiffL1_RVV<schar, int> {
    int operator() (const schar* src1, const schar* src2, int n) const {
        auto s = __riscv_vmv_v_x_u32m1(0, __riscv_vsetvlmax_e32m1());
        auto zero = __riscv_vmv_v_x_u16m1(0, __riscv_vsetvlmax_e16m1());
        int vl;
        for (int i = 0; i < n; i += vl) {
            vl = __riscv_vsetvl_e8m8(n - i);
            auto v1 = __riscv_vle8_v_i8m8(src1 + i, vl);
            auto v2 = __riscv_vle8_v_i8m8(src2 + i, vl);
            auto v = custom_intrin::__riscv_vabd(v1, v2, vl);
            s = __riscv_vwredsumu(__riscv_vwredsumu_tu(zero, v, zero, vl), s, __riscv_vsetvlmax_e16m1());
        }
        return __riscv_vmv_x(s);
    }
};

template<>
struct NormDiffL1_RVV<ushort, int> {
    int operator() (const ushort* src1, const ushort* src2, int n) const {
        auto s = __riscv_vmv_v_x_u32m1(0, __riscv_vsetvlmax_e32m1());
        int vl;
        for (int i = 0; i < n; i += vl) {
            vl = __riscv_vsetvl_e16m8(n - i);
            auto v1 = __riscv_vle16_v_u16m8(src1 + i, vl);
            auto v2 = __riscv_vle16_v_u16m8(src2 + i, vl);
            auto v = custom_intrin::__riscv_vabd(v1, v2, vl);
            s = __riscv_vwredsumu(v, s, vl);
        }
        return __riscv_vmv_x(s);
    }
};

template<>
struct NormDiffL1_RVV<short, int> {
    int operator() (const short* src1, const short* src2, int n) const {
        auto s = __riscv_vmv_v_x_u32m1(0, __riscv_vsetvlmax_e32m1());
        int vl;
        for (int i = 0; i < n; i += vl) {
            vl = __riscv_vsetvl_e16m8(n - i);
            auto v1 = __riscv_vle16_v_i16m8(src1 + i, vl);
            auto v2 = __riscv_vle16_v_i16m8(src2 + i, vl);
            auto v = custom_intrin::__riscv_vabd(v1, v2, vl);
            s = __riscv_vwredsumu(v, s, vl);
        }
        return __riscv_vmv_x(s);
    }
};

template<>
struct NormDiffL1_RVV<int, double> {
    double operator() (const int* src1, const int* src2, int n) const {
        int vlmax = __riscv_vsetvlmax_e32m4();
        auto s = __riscv_vfmv_v_f_f64m8(0, vlmax);
        int vl;
        for (int i = 0; i < n; i += vl) {
            vl = __riscv_vsetvl_e32m4(n - i);
            auto v1 = __riscv_vle32_v_i32m4(src1 + i, vl);
            auto v2 = __riscv_vle32_v_i32m4(src2 + i, vl);
            // auto v = custom_intrin::__riscv_vabd(v1, v2, vl); // 5.x
            auto v = custom_intrin::__riscv_vabs(__riscv_vsub(v1, v2, vl), vl); // 4.x
            s = __riscv_vfadd_tu(s, s, __riscv_vfwcvt_f(v, vl), vl);
        }
        return __riscv_vfmv_f(__riscv_vfredosum(s, __riscv_vfmv_s_f_f64m1(0, __riscv_vsetvlmax_e64m1()), vlmax));
    }
};

template<>
struct NormDiffL1_RVV<float, double> {
    double operator() (const float* src1, const float* src2, int n) const {
        int vlmax = __riscv_vsetvlmax_e32m4();
        auto s = __riscv_vfmv_v_f_f64m8(0, vlmax);
        int vl;
        for (int i = 0; i < n; i += vl) {
            vl = __riscv_vsetvl_e32m4(n - i);
            auto v1 = __riscv_vle32_v_f32m4(src1 + i, vl);
            auto v2 = __riscv_vle32_v_f32m4(src2 + i, vl);
            auto v = __riscv_vfabs(__riscv_vfsub(v1, v2, vl), vl);
            s = __riscv_vfadd_tu(s, s, __riscv_vfwcvt_f(v, vl), vl);
        }
        return __riscv_vfmv_f(__riscv_vfredosum(s, __riscv_vfmv_s_f_f64m1(0, __riscv_vsetvlmax_e64m1()), vlmax));
    }
};

template<>
struct NormDiffL1_RVV<double, double> {
    double operator() (const double* src1, const double* src2, int n) const {
        int vlmax = __riscv_vsetvlmax_e64m8();
        auto s = __riscv_vfmv_v_f_f64m8(0, vlmax);
        int vl;
        for (int i = 0; i < n; i += vl) {
            vl = __riscv_vsetvl_e64m8(n - i);
            auto v1 = __riscv_vle64_v_f64m8(src1 + i, vl);
            auto v2 = __riscv_vle64_v_f64m8(src2 + i, vl);
            auto v = __riscv_vfabs(__riscv_vfsub(v1, v2, vl), vl);
            s = __riscv_vfadd_tu(s, s, v, vl);
        }
        return __riscv_vfmv_f(__riscv_vfredosum(s, __riscv_vfmv_s_f_f64m1(0, __riscv_vsetvlmax_e64m1()), vlmax));
    }
};

template<>
struct NormDiffL2_RVV<uchar, int> {
    int operator() (const uchar* src1, const uchar* src2, int n) const {
        auto s = __riscv_vmv_v_x_u32m1(0, __riscv_vsetvlmax_e32m1());
        int vl;
        for (int i = 0; i < n; i += vl) {
            vl = __riscv_vsetvl_e8m4(n - i);
            auto v1 = __riscv_vle8_v_u8m4(src1 + i, vl);
            auto v2 = __riscv_vle8_v_u8m4(src2 + i, vl);
            auto v = custom_intrin::__riscv_vabd(v1, v2, vl);
            s = __riscv_vwredsumu(__riscv_vwmulu(v, v, vl), s, vl);
        }
        return __riscv_vmv_x(s);
    }
};

template<>
struct NormDiffL2_RVV<schar, int> {
    int operator() (const schar* src1, const schar* src2, int n) const {
        auto s = __riscv_vmv_v_x_u32m1(0, __riscv_vsetvlmax_e32m1());
        int vl;
        for (int i = 0; i < n; i += vl) {
            vl = __riscv_vsetvl_e8m4(n - i);
            auto v1 = __riscv_vle8_v_i8m4(src1 + i, vl);
            auto v2 = __riscv_vle8_v_i8m4(src2 + i, vl);
            auto v = custom_intrin::__riscv_vabd(v1, v2, vl);
            s = __riscv_vwredsumu(__riscv_vwmulu(v, v, vl), s, vl);
        }
        return __riscv_vmv_x(s);
    }
};

template<>
struct NormDiffL2_RVV<ushort, double> {
    double operator() (const ushort* src1, const ushort* src2, int n) const {
        int vlmax = __riscv_vsetvlmax_e16m2();
        auto s = __riscv_vfmv_v_f_f64m8(0, vlmax);
        int vl;
        for (int i = 0; i < n; i += vl) {
            vl = __riscv_vsetvl_e16m2(n - i);
            auto v1 = __riscv_vle16_v_u16m2(src1 + i, vl);
            auto v2 = __riscv_vle16_v_u16m2(src2 + i, vl);
            auto v = custom_intrin::__riscv_vabd(v1, v2, vl);
            auto v_mul = __riscv_vwmulu(v, v, vl);
            s = __riscv_vfadd_tu(s, s, __riscv_vfwcvt_f(v_mul, vl), vl);
        }
        return __riscv_vfmv_f(__riscv_vfredosum(s, __riscv_vfmv_s_f_f64m1(0, __riscv_vsetvlmax_e64m1()), vlmax));
    }
};

template<>
struct NormDiffL2_RVV<short, double> {
    double operator() (const short* src1, const short* src2, int n) const {
        int vlmax = __riscv_vsetvlmax_e16m2();
        auto s = __riscv_vfmv_v_f_f64m8(0, vlmax);
        int vl;
        for (int i = 0; i < n; i += vl) {
            vl = __riscv_vsetvl_e16m2(n - i);
            auto v1 = __riscv_vle16_v_i16m2(src1 + i, vl);
            auto v2 = __riscv_vle16_v_i16m2(src2 + i, vl);
            auto v = custom_intrin::__riscv_vabd(v1, v2, vl);
            auto v_mul = __riscv_vwmulu(v, v, vl);
            s = __riscv_vfadd_tu(s, s, __riscv_vfwcvt_f(v_mul, vl), vl);
        }
        return __riscv_vfmv_f(__riscv_vfredosum(s, __riscv_vfmv_s_f_f64m1(0, __riscv_vsetvlmax_e64m1()), vlmax));
    }
};

template<>
struct NormDiffL2_RVV<int, double> {
    double operator() (const int* src1, const int* src2, int n) const {
        int vlmax = __riscv_vsetvlmax_e32m4();
        auto s = __riscv_vfmv_v_f_f64m8(0, vlmax);
        int vl;
        for (int i = 0; i < n; i += vl) {
            vl = __riscv_vsetvl_e32m4(n - i);
            auto v1 = __riscv_vle32_v_i32m4(src1 + i, vl);
            auto v2 = __riscv_vle32_v_i32m4(src2 + i, vl);
            auto v = custom_intrin::__riscv_vabd(v1, v2, vl);
            auto v_mul = __riscv_vwmulu(v, v, vl);
            s = __riscv_vfadd_tu(s, s, __riscv_vfcvt_f(v_mul, vl), vl);
        }
        return __riscv_vfmv_f(__riscv_vfredosum(s, __riscv_vfmv_s_f_f64m1(0, __riscv_vsetvlmax_e64m1()), vlmax));
    }
};

template<>
struct NormDiffL2_RVV<float, double> {
    double operator() (const float* src1, const float* src2, int n) const {
        int vlmax = __riscv_vsetvlmax_e32m4();
        auto s = __riscv_vfmv_v_f_f64m8(0, vlmax);
        int vl;
        for (int i = 0; i < n; i += vl) {
            vl = __riscv_vsetvl_e32m4(n - i);
            auto v1 = __riscv_vle32_v_f32m4(src1 + i, vl);
            auto v2 = __riscv_vle32_v_f32m4(src2 + i, vl);
            auto v = __riscv_vfsub(v1, v2, vl);
            auto v_mul = __riscv_vfwmul(v, v, vl);
            s = __riscv_vfadd_tu(s, s, v_mul, vl);
        }
        return __riscv_vfmv_f(__riscv_vfredosum(s, __riscv_vfmv_s_f_f64m1(0, __riscv_vsetvlmax_e64m1()), vlmax));
    }
};

template<>
struct NormDiffL2_RVV<double, double> {
    double operator() (const double* src1, const double* src2, int n) const {
        int vlmax = __riscv_vsetvlmax_e64m8();
        auto s = __riscv_vfmv_v_f_f64m8(0, vlmax);
        int vl;
        for (int i = 0; i < n; i += vl) {
            vl = __riscv_vsetvl_e64m8(n - i);
            auto v1 = __riscv_vle64_v_f64m8(src1 + i, vl);
            auto v2 = __riscv_vle64_v_f64m8(src2 + i, vl);
            auto v = __riscv_vfsub(v1, v2, vl);
            auto v_mul = __riscv_vfmul(v, v, vl);
            s = __riscv_vfadd_tu(s, s, v_mul, vl);
        }
        return __riscv_vfmv_f(__riscv_vfredosum(s, __riscv_vfmv_s_f_f64m1(0, __riscv_vsetvlmax_e64m1()), vlmax));
    }
};

// Norm with mask

template <typename T, typename ST>
struct MaskedNormDiffInf_RVV {
    inline ST operator() (const T* src1, const T* src2, const uchar* mask, int len, int cn) const {
        ST s = 0;
        for( int i = 0; i < len; i++, src1 += cn, src2 += cn ) {
            if( mask[i] ) {
                for( int k = 0; k < cn; k++ ) {
                    s = std::max(s, (ST)std::abs(src1[k] - src2[k]));
                }
            }
        }
        return s;
    }
};

template <typename T, typename ST>
struct MaskedNormDiffL1_RVV {
    inline ST operator() (const T* src1, const T* src2, const uchar* mask, int len, int cn) const {
        ST s = 0;
        for( int i = 0; i < len; i++, src1 += cn, src2 += cn ) {
            if( mask[i] ) {
                for( int k = 0; k < cn; k++ ) {
                    s += std::abs(src1[k] - src2[k]);
                }
            }
        }
        return s;
    }
};

template <typename T, typename ST>
struct MaskedNormDiffL2_RVV {
    inline ST operator() (const T* src1, const T* src2, const uchar* mask, int len, int cn) const {
        ST s = 0;
        for( int i = 0; i < len; i++, src1 += cn, src2 += cn ) {
            if( mask[i] ) {
                for( int k = 0; k < cn; k++ ) {
                    ST v = (ST)src1[k] - (ST)src2[k];
                    s += v * v;
                }
            }
        }
        return s;
    }
};

template<>
struct MaskedNormDiffInf_RVV<uchar, int> {
    int operator() (const uchar* src1, const uchar* src2, const uchar* mask, int len, int cn) const {
        int vlmax = __riscv_vsetvlmax_e8m8();
        auto s = __riscv_vmv_v_x_u8m8(0, vlmax);
        if (cn == 1) {
            int vl;
            for (int i = 0; i < len; i += vl) {
                vl = __riscv_vsetvl_e8m8(len - i);
                auto v1 = __riscv_vle8_v_u8m8(src1 + i, vl);
                auto v2 = __riscv_vle8_v_u8m8(src2 + i, vl);
                auto v = custom_intrin::__riscv_vabd(v1, v2, vl);
                auto m = __riscv_vle8_v_u8m8(mask + i, vl);
                auto b = __riscv_vmsne(m, 0, vl);
                s = __riscv_vmaxu_tumu(b, s, s, v, vl);
            }
        } else if (cn == 4) {
            int vl;
            for (int i = 0; i < len; i += vl) {
                vl = __riscv_vsetvl_e8m2(len - i);
                auto v1 = __riscv_vle8_v_u8m8(src1 + i * 4, vl * 4);
                auto v2 = __riscv_vle8_v_u8m8(src2 + i * 4, vl * 4);
                auto v = custom_intrin::__riscv_vabd(v1, v2, vl * 4);
                auto m = __riscv_vle8_v_u8m2(mask + i, vl);
                auto b = __riscv_vmsne(__riscv_vreinterpret_u8m8(__riscv_vmul(__riscv_vzext_vf4(__riscv_vminu(m, 1, vl), vl), 0x01010101, vl)), 0, vl * 4);
                s = __riscv_vmaxu_tumu(b, s, s, v, vl * 4);
            }
        } else {
            for (int cn_index = 0; cn_index < cn; cn_index++) {
                int vl;
                for (int i = 0; i < len; i += vl) {
                    vl = __riscv_vsetvl_e8m8(len - i);
                    auto v1 = __riscv_vlse8_v_u8m8(src1 + cn * i + cn_index, sizeof(uchar) * cn, vl);
                    auto v2 = __riscv_vlse8_v_u8m8(src2 + cn * i + cn_index, sizeof(uchar) * cn, vl);
                    auto v = custom_intrin::__riscv_vabd(v1, v2, vl);
                    auto m = __riscv_vle8_v_u8m8(mask + i, vl);
                    auto b = __riscv_vmsne(m, 0, vl);
                    s = __riscv_vmaxu_tumu(b, s, s, v, vl);
                }
            }
        }
        return __riscv_vmv_x(__riscv_vredmaxu(s, __riscv_vmv_s_x_u8m1(0, __riscv_vsetvlmax_e8m1()), vlmax));
    }
};

template<>
struct MaskedNormDiffInf_RVV<schar, int> {
    int operator() (const schar* src1, const schar* src2, const uchar* mask, int len, int cn) const {
        int vlmax = __riscv_vsetvlmax_e8m8();
        auto s = __riscv_vmv_v_x_u8m8(0, vlmax);
        for (int cn_index = 0; cn_index < cn; cn_index++) {
            int vl;
            for (int i = 0; i < len; i += vl) {
                vl = __riscv_vsetvl_e8m8(len - i);
                auto v1 = __riscv_vlse8_v_i8m8(src1 + cn * i + cn_index, sizeof(schar) * cn, vl);
                auto v2 = __riscv_vlse8_v_i8m8(src2 + cn * i + cn_index, sizeof(schar) * cn, vl);
                auto v = custom_intrin::__riscv_vabd(v1, v2, vl);
                auto m = __riscv_vle8_v_u8m8(mask + i, vl);
                auto b = __riscv_vmsne(m, 0, vl);
                s = __riscv_vmaxu_tumu(b, s, s, v, vl);
            }
        }
        return __riscv_vmv_x(__riscv_vredmaxu(s, __riscv_vmv_s_x_u8m1(0, __riscv_vsetvlmax_e8m1()), vlmax));
    }
};

template<>
struct MaskedNormDiffInf_RVV<ushort, int> {
    int operator() (const ushort* src1, const ushort* src2, const uchar* mask, int len, int cn) const {
        int vlmax = __riscv_vsetvlmax_e16m8();
        auto s = __riscv_vmv_v_x_u16m8(0, vlmax);
        for (int cn_index = 0; cn_index < cn; cn_index++) {
            int vl;
            for (int i = 0; i < len; i += vl) {
                vl = __riscv_vsetvl_e16m8(len - i);
                auto v1 = __riscv_vlse16_v_u16m8(src1 + cn * i + cn_index, sizeof(ushort) * cn, vl);
                auto v2 = __riscv_vlse16_v_u16m8(src2 + cn * i + cn_index, sizeof(ushort) * cn, vl);
                auto v = custom_intrin::__riscv_vabd(v1, v2, vl);
                auto m = __riscv_vle8_v_u8m4(mask + i, vl);
                auto b = __riscv_vmsne(m, 0, vl);
                s = __riscv_vmaxu_tumu(b, s, s, v, vl);
            }
        }
        return __riscv_vmv_x(__riscv_vredmaxu(s, __riscv_vmv_s_x_u16m1(0, __riscv_vsetvlmax_e16m1()), vlmax));
    }
};

template<>
struct MaskedNormDiffInf_RVV<short, int> {
    int operator() (const short* src1, const short* src2, const uchar* mask, int len, int cn) const {
        int vlmax = __riscv_vsetvlmax_e16m8();
        auto s = __riscv_vmv_v_x_u16m8(0, vlmax);
        for (int cn_index = 0; cn_index < cn; cn_index++) {
            int vl;
            for (int i = 0; i < len; i += vl) {
                vl = __riscv_vsetvl_e16m8(len - i);
                auto v1 = __riscv_vlse16_v_i16m8(src1 + cn * i + cn_index, sizeof(short) * cn, vl);
                auto v2 = __riscv_vlse16_v_i16m8(src2 + cn * i + cn_index, sizeof(short) * cn, vl);
                auto v = custom_intrin::__riscv_vabd(v1, v2, vl);
                auto m = __riscv_vle8_v_u8m4(mask + i, vl);
                auto b = __riscv_vmsne(m, 0, vl);
                s = __riscv_vmaxu_tumu(b, s, s, v, vl);
            }
        }
        return __riscv_vmv_x(__riscv_vredmaxu(s, __riscv_vmv_s_x_u16m1(0, __riscv_vsetvlmax_e16m1()), vlmax));
    }
};

template<>
struct MaskedNormDiffInf_RVV<int, int> {
    int operator() (const int* src1, const int* src2, const uchar* mask, int len, int cn) const {
        int vlmax = __riscv_vsetvlmax_e32m8();
        auto s = __riscv_vmv_v_x_u32m8(0, vlmax);
        for (int cn_index = 0; cn_index < cn; cn_index++) {
            int vl;
            for (int i = 0; i < len; i += vl) {
                vl = __riscv_vsetvl_e32m8(len - i);
                auto v1 = __riscv_vlse32_v_i32m8(src1 + cn * i + cn_index, sizeof(int) * cn, vl);
                auto v2 = __riscv_vlse32_v_i32m8(src2 + cn * i + cn_index, sizeof(int) * cn, vl);
                // auto v = custom_intrin::__riscv_vabd(v1, v2, vl); // 5.x
                auto v = custom_intrin::__riscv_vabs(__riscv_vsub(v1, v2, vl), vl); // 4.x
                auto m = __riscv_vle8_v_u8m2(mask + i, vl);
                auto b = __riscv_vmsne(m, 0, vl);
                s = __riscv_vmaxu_tumu(b, s, s, v, vl);
            }
        }
        return __riscv_vmv_x(__riscv_vredmaxu(s, __riscv_vmv_s_x_u32m1(0, __riscv_vsetvlmax_e32m1()), vlmax));
    }
};

template<>
struct MaskedNormDiffInf_RVV<float, float> {
    float operator() (const float* src1, const float* src2, const uchar* mask, int len, int cn) const {
        int vlmax = __riscv_vsetvlmax_e32m8();
        auto s = __riscv_vfmv_v_f_f32m8(0, vlmax);
        if (cn == 1) {
            int vl;
            for (int i = 0; i < len; i += vl) {
                vl = __riscv_vsetvl_e32m8(len - i);
                auto v1 = __riscv_vle32_v_f32m8(src1 + i, vl);
                auto v2 = __riscv_vle32_v_f32m8(src2 + i, vl);
                auto v = __riscv_vfabs(__riscv_vfsub(v1, v2, vl), vl);
                auto m = __riscv_vle8_v_u8m2(mask + i, vl);
                auto b = __riscv_vmsne(m, 0, vl);
                s = __riscv_vfmax_tumu(b, s, s, v, vl);
            }
        } else {
            for (int cn_index = 0; cn_index < cn; cn_index++) {
                int vl;
                for (int i = 0; i < len; i += vl) {
                    vl = __riscv_vsetvl_e32m8(len - i);
                    auto v1 = __riscv_vlse32_v_f32m8(src1 + cn * i + cn_index, sizeof(float) * cn, vl);
                    auto v2 = __riscv_vlse32_v_f32m8(src2 + cn * i + cn_index, sizeof(float) * cn, vl);
                    auto v = __riscv_vfabs(__riscv_vfsub(v1, v2, vl), vl);
                    auto m = __riscv_vle8_v_u8m2(mask + i, vl);
                    auto b = __riscv_vmsne(m, 0, vl);
                    s = __riscv_vfmax_tumu(b, s, s, v, vl);
                }
            }
        }
        return __riscv_vfmv_f(__riscv_vfredmax(s, __riscv_vfmv_s_f_f32m1(0, __riscv_vsetvlmax_e32m1()), vlmax));
    }
};

template<>
struct MaskedNormDiffInf_RVV<double, double> {
    double operator() (const double* src1, const double* src2, const uchar* mask, int len, int cn) const {
        int vlmax = __riscv_vsetvlmax_e64m8();
        auto s = __riscv_vfmv_v_f_f64m8(0, vlmax);
        for (int cn_index = 0; cn_index < cn; cn_index++) {
            int vl;
            for (int i = 0; i < len; i += vl) {
                vl = __riscv_vsetvl_e64m8(len - i);
                auto v1 = __riscv_vlse64_v_f64m8(src1 + cn * i + cn_index, sizeof(double) * cn, vl);
                auto v2 = __riscv_vlse64_v_f64m8(src2 + cn * i + cn_index, sizeof(double) * cn, vl);
                auto v = __riscv_vfabs(__riscv_vfsub(v1, v2, vl), vl);
                auto m = __riscv_vle8_v_u8m1(mask + i, vl);
                auto b = __riscv_vmsne(m, 0, vl);
                s = __riscv_vfmax_tumu(b, s, s, __riscv_vfabs(v, vl), vl);
            }
        }
        return __riscv_vfmv_f(__riscv_vfredmax(s, __riscv_vfmv_s_f_f64m1(0, __riscv_vsetvlmax_e64m1()), vlmax));
    }
};

template<>
struct MaskedNormDiffL1_RVV<uchar, int> {
    int operator() (const uchar* src1, const uchar* src2, const uchar* mask, int len, int cn) const {
        auto s = __riscv_vmv_v_x_u32m1(0, __riscv_vsetvlmax_e32m1());
        auto zero = __riscv_vmv_v_x_u16m1(0, __riscv_vsetvlmax_e16m1());
        if (cn == 1) {
            int vl;
            for (int i = 0; i < len; i += vl) {
                vl = __riscv_vsetvl_e8m8(len - i);
                auto v1 = __riscv_vle8_v_u8m8(src1 + i, vl);
                auto v2 = __riscv_vle8_v_u8m8(src2 + i, vl);
                auto v = custom_intrin::__riscv_vabd(v1, v2, vl);
                auto m = __riscv_vle8_v_u8m8(mask + i, vl);
                auto b = __riscv_vmsne(m, 0, vl);
                s = __riscv_vwredsumu(__riscv_vwredsumu_tum(b, zero, v, zero, vl), s, __riscv_vsetvlmax_e16m1());
            }
        } else if (cn == 4) {
            int vl;
            for (int i = 0; i < len; i += vl) {
                vl = __riscv_vsetvl_e8m2(len - i);
                auto v1 = __riscv_vle8_v_u8m8(src1 + i * 4, vl * 4);
                auto v2 = __riscv_vle8_v_u8m8(src2 + i * 4, vl * 4);
                auto v = custom_intrin::__riscv_vabd(v1, v2, vl * 4);
                auto m = __riscv_vle8_v_u8m2(mask + i, vl);
                auto b = __riscv_vmsne(__riscv_vreinterpret_u8m8(__riscv_vmul(__riscv_vzext_vf4(__riscv_vminu(m, 1, vl), vl), 0x01010101, vl)), 0, vl * 4);
                s = __riscv_vwredsumu(__riscv_vwredsumu_tum(b, zero, v, zero, vl * 4), s, __riscv_vsetvlmax_e16m1());
            }
        } else {
            for (int cn_index = 0; cn_index < cn; cn_index++) {
                int vl;
                for (int i = 0; i < len; i += vl) {
                    vl = __riscv_vsetvl_e8m8(len - i);
                    auto v1 = __riscv_vlse8_v_u8m8(src1 + cn * i + cn_index, sizeof(uchar) * cn, vl);
                    auto v2 = __riscv_vlse8_v_u8m8(src2 + cn * i + cn_index, sizeof(uchar) * cn, vl);
                    auto v = custom_intrin::__riscv_vabd(v1, v2, vl);
                    auto m = __riscv_vle8_v_u8m8(mask + i, vl);
                    auto b = __riscv_vmsne(m, 0, vl);
                    s = __riscv_vwredsumu(__riscv_vwredsumu_tum(b, zero, v, zero, vl), s, __riscv_vsetvlmax_e16m1());
                }
            }
        }
        return __riscv_vmv_x(s);
    }
};

template<>
struct MaskedNormDiffL1_RVV<schar, int> {
    int operator() (const schar* src1, const schar* src2, const uchar* mask, int len, int cn) const {
        auto s = __riscv_vmv_v_x_u32m1(0, __riscv_vsetvlmax_e32m1());
        auto zero = __riscv_vmv_v_x_u16m1(0, __riscv_vsetvlmax_e16m1());
        for (int cn_index = 0; cn_index < cn; cn_index++) {
            int vl;
            for (int i = 0; i < len; i += vl) {
                vl = __riscv_vsetvl_e8m8(len - i);
                auto v1 = __riscv_vlse8_v_i8m8(src1 + cn * i + cn_index, sizeof(schar) * cn, vl);
                auto v2 = __riscv_vlse8_v_i8m8(src2 + cn * i + cn_index, sizeof(schar) * cn, vl);
                auto v = custom_intrin::__riscv_vabd(v1, v2, vl);
                auto m = __riscv_vle8_v_u8m8(mask + i, vl);
                auto b = __riscv_vmsne(m, 0, vl);
                s = __riscv_vwredsumu(__riscv_vwredsumu_tum(b, zero, v, zero, vl), s, __riscv_vsetvlmax_e16m1());
            }
        }
        return __riscv_vmv_x(s);
    }
};

template<>
struct MaskedNormDiffL1_RVV<ushort, int> {
    int operator() (const ushort* src1, const ushort* src2, const uchar* mask, int len, int cn) const {
        auto s = __riscv_vmv_v_x_u32m1(0, __riscv_vsetvlmax_e32m1());
        for (int cn_index = 0; cn_index < cn; cn_index++) {
            int vl;
            for (int i = 0; i < len; i += vl) {
                vl = __riscv_vsetvl_e8m4(len - i);
                auto v1 = __riscv_vlse16_v_u16m8(src1 + cn * i + cn_index, sizeof(ushort) * cn, vl);
                auto v2 = __riscv_vlse16_v_u16m8(src2 + cn * i + cn_index, sizeof(ushort) * cn, vl);
                auto v = custom_intrin::__riscv_vabd(v1, v2, vl);
                auto m = __riscv_vle8_v_u8m4(mask + i, vl);
                auto b = __riscv_vmsne(m, 0, vl);
                s = __riscv_vwredsumu_tum(b, s, v, s, vl);
            }
        }
        return __riscv_vmv_x(s);
    }
};

template<>
struct MaskedNormDiffL1_RVV<short, int> {
    int operator() (const short* src1, const short* src2, const uchar* mask, int len, int cn) const {
        auto s = __riscv_vmv_v_x_u32m1(0, __riscv_vsetvlmax_e32m1());
        for (int cn_index = 0; cn_index < cn; cn_index++) {
            int vl;
            for (int i = 0; i < len; i += vl) {
                vl = __riscv_vsetvl_e8m4(len - i);
                auto v1 = __riscv_vlse16_v_i16m8(src1 + cn * i + cn_index, sizeof(short) * cn, vl);
                auto v2 = __riscv_vlse16_v_i16m8(src2 + cn * i + cn_index, sizeof(short) * cn, vl);
                auto v = custom_intrin::__riscv_vabd(v1, v2, vl);
                auto m = __riscv_vle8_v_u8m4(mask + i, vl);
                auto b = __riscv_vmsne(m, 0, vl);
                s = __riscv_vwredsumu_tum(b, s, v, s, vl);
            }
        }
        return __riscv_vmv_x(s);
    }
};

template<>
struct MaskedNormDiffL1_RVV<int, double> {
    double operator() (const int* src1, const int* src2, const uchar* mask, int len, int cn) const {
        int vlmax = __riscv_vsetvlmax_e32m4();
        auto s = __riscv_vfmv_v_f_f64m8(0, vlmax);
        for (int cn_index = 0; cn_index < cn; cn_index++) {
            int vl;
            for (int i = 0; i < len; i += vl) {
                vl = __riscv_vsetvl_e32m4(len - i);
                auto v1 = __riscv_vlse32_v_i32m4(src1 + cn * i + cn_index, sizeof(int) * cn, vl);
                auto v2 = __riscv_vlse32_v_i32m4(src2 + cn * i + cn_index, sizeof(int) * cn, vl);
                // auto v = custom_intrin::__riscv_vabd(v1, v2, vl); // 5.x
                auto v = custom_intrin::__riscv_vabs(__riscv_vsub(v1, v2, vl), vl); // 4.x
                auto m = __riscv_vle8_v_u8m1(mask + i, vl);
                auto b = __riscv_vmsne(m, 0, vl);
                s = __riscv_vfadd_tumu(b, s, s, __riscv_vfwcvt_f(b, v, vl), vl);
            }
        }
        return __riscv_vfmv_f(__riscv_vfredosum(s, __riscv_vfmv_s_f_f64m1(0, __riscv_vsetvlmax_e64m1()), vlmax));
    }
};

template<>
struct MaskedNormDiffL1_RVV<float, double> {
    double operator() (const float* src1, const float* src2, const uchar* mask, int len, int cn) const {
        int vlmax = __riscv_vsetvlmax_e32m4();
        auto s = __riscv_vfmv_v_f_f64m8(0, vlmax);
        if (cn == 1) {
            int vl;
            for (int i = 0; i < len; i += vl) {
                vl = __riscv_vsetvl_e32m4(len - i);
                auto v1 = __riscv_vle32_v_f32m4(src1 + i, vl);
                auto v2 = __riscv_vle32_v_f32m4(src2 + i, vl);
                auto v = __riscv_vfabs(__riscv_vfsub(v1, v2, vl), vl);
                auto m = __riscv_vle8_v_u8m1(mask + i, vl);
                auto b = __riscv_vmsne(m, 0, vl);
                s = __riscv_vfadd_tumu(b, s, s, __riscv_vfwcvt_f(b, v, vl), vl);
            }
        } else {
            for (int cn_index = 0; cn_index < cn; cn_index++) {
                int vl;
                for (int i = 0; i < len; i += vl) {
                    vl = __riscv_vsetvl_e32m4(len - i);
                    auto v1 = __riscv_vlse32_v_f32m4(src1 + cn * i + cn_index, sizeof(float) * cn, vl);
                    auto v2 = __riscv_vlse32_v_f32m4(src2 + cn * i + cn_index, sizeof(float) * cn, vl);
                    auto v = __riscv_vfabs(__riscv_vfsub(v1, v2, vl), vl);
                    auto m = __riscv_vle8_v_u8m1(mask + i, vl);
                    auto b = __riscv_vmsne(m, 0, vl);
                    s = __riscv_vfadd_tumu(b, s, s, __riscv_vfwcvt_f(b, v, vl), vl);
                }
            }
        }
        return __riscv_vfmv_f(__riscv_vfredosum(s, __riscv_vfmv_s_f_f64m1(0, __riscv_vsetvlmax_e64m1()), vlmax));
    }
};

template<>
struct MaskedNormDiffL1_RVV<double, double> {
    double operator() (const double* src1, const double* src2, const uchar* mask, int len, int cn) const {
        int vlmax = __riscv_vsetvlmax_e64m8();
        auto s = __riscv_vfmv_v_f_f64m8(0, vlmax);
        for (int cn_index = 0; cn_index < cn; cn_index++) {
            int vl;
            for (int i = 0; i < len; i += vl) {
                vl = __riscv_vsetvl_e64m8(len - i);
                auto v1 = __riscv_vlse64_v_f64m8(src1 + cn * i + cn_index, sizeof(double) * cn, vl);
                auto v2 = __riscv_vlse64_v_f64m8(src2 + cn * i + cn_index, sizeof(double) * cn, vl);
                auto v = __riscv_vfabs(__riscv_vfsub(v1, v2, vl), vl);
                auto m = __riscv_vle8_v_u8m1(mask + i, vl);
                auto b = __riscv_vmsne(m, 0, vl);
                s = __riscv_vfadd_tumu(b, s, s, __riscv_vfabs(v, vl), vl);
            }
        }
        return __riscv_vfmv_f(__riscv_vfredosum(s, __riscv_vfmv_s_f_f64m1(0, __riscv_vsetvlmax_e64m1()), vlmax));
    }
};

template<>
struct MaskedNormDiffL2_RVV<uchar, int> {
    int operator() (const uchar* src1, const uchar* src2, const uchar* mask, int len, int cn) const {
        auto s = __riscv_vmv_v_x_u32m1(0, __riscv_vsetvlmax_e32m1());
        if (cn == 1) {
            int vl;
            for (int i = 0; i < len; i += vl) {
                vl = __riscv_vsetvl_e8m4(len - i);
                auto v1 = __riscv_vle8_v_u8m4(src1 + i, vl);
                auto v2 = __riscv_vle8_v_u8m4(src2 + i, vl);
                auto v = custom_intrin::__riscv_vabd(v1, v2, vl);
                auto m = __riscv_vle8_v_u8m4(mask + i, vl);
                auto b = __riscv_vmsne(m, 0, vl);
                s = __riscv_vwredsumu(b, __riscv_vwmulu(b, v, v, vl), s, vl);
            }
        } else if (cn == 4) {
            int vl;
            for (int i = 0; i < len; i += vl) {
                vl = __riscv_vsetvl_e8m1(len - i);
                auto v1 = __riscv_vle8_v_u8m4(src1 + i * 4, vl * 4);
                auto v2 = __riscv_vle8_v_u8m4(src2 + i * 4, vl * 4);
                auto v = custom_intrin::__riscv_vabd(v1, v2, vl * 4);
                auto m = __riscv_vle8_v_u8m1(mask + i, vl);
                auto b = __riscv_vmsne(__riscv_vreinterpret_u8m4(__riscv_vmul(__riscv_vzext_vf4(__riscv_vminu(m, 1, vl), vl), 0x01010101, vl)), 0, vl * 4);
                s = __riscv_vwredsumu(b, __riscv_vwmulu(b, v, v, vl * 4), s, vl * 4);
            }
        } else {
            for (int cn_index = 0; cn_index < cn; cn_index++) {
                int vl;
                for (int i = 0; i < len; i += vl) {
                    vl = __riscv_vsetvl_e8m4(len - i);
                    auto v1 = __riscv_vlse8_v_u8m4(src1 + cn * i + cn_index, sizeof(uchar) * cn, vl);
                    auto v2 = __riscv_vlse8_v_u8m4(src2 + cn * i + cn_index, sizeof(uchar) * cn, vl);
                    auto v = custom_intrin::__riscv_vabd(v1, v2, vl);
                    auto m = __riscv_vle8_v_u8m4(mask + i, vl);
                    auto b = __riscv_vmsne(m, 0, vl);
                    s = __riscv_vwredsumu(b, __riscv_vwmulu(b, v, v, vl), s, vl);
                }
            }
        }
        return __riscv_vmv_x(s);
    }
};

template<>
struct MaskedNormDiffL2_RVV<schar, int> {
    int operator() (const schar* src1, const schar* src2, const uchar* mask, int len, int cn) const {
        auto s = __riscv_vmv_v_x_u32m1(0, __riscv_vsetvlmax_e32m1());
        for (int cn_index = 0; cn_index < cn; cn_index++) {
            int vl;
            for (int i = 0; i < len; i += vl) {
                vl = __riscv_vsetvl_e8m4(len - i);
                auto v1 = __riscv_vlse8_v_i8m4(src1 + cn * i + cn_index, sizeof(schar) * cn, vl);
                auto v2 = __riscv_vlse8_v_i8m4(src2 + cn * i + cn_index, sizeof(schar) * cn, vl);
                auto v = custom_intrin::__riscv_vabd(v1, v2, vl);
                auto m = __riscv_vle8_v_u8m4(mask + i, vl);
                auto b = __riscv_vmsne(m, 0, vl);
                s = __riscv_vwredsumu(b, __riscv_vwmulu(b, v, v, vl), s, vl);
            }
        }
        return __riscv_vmv_x(s);
    }
};

template<>
struct MaskedNormDiffL2_RVV<ushort, double> {
    double operator() (const ushort* src1, const ushort* src2, const uchar* mask, int len, int cn) const {
        int vlmax = __riscv_vsetvlmax_e16m2();
        auto s = __riscv_vfmv_v_f_f64m8(0, vlmax);
        for (int cn_index = 0; cn_index < cn; cn_index++) {
            int vl;
            for (int i = 0; i < len; i += vl) {
                vl = __riscv_vsetvl_e16m2(len - i);
                auto v1 = __riscv_vlse16_v_u16m2(src1 + cn * i + cn_index, sizeof(ushort) * cn, vl);
                auto v2 = __riscv_vlse16_v_u16m2(src2 + cn * i + cn_index, sizeof(ushort) * cn, vl);
                auto v = custom_intrin::__riscv_vabd(v1, v2, vl);
                auto m = __riscv_vle8_v_u8m1(mask + i, vl);
                auto b = __riscv_vmsne(m, 0, vl);
                auto v_mul = __riscv_vwmulu(b, v, v, vl);
                s = __riscv_vfadd_tumu(b, s, s, __riscv_vfwcvt_f(b, v_mul, vl), vl);
            }
        }
        return __riscv_vfmv_f(__riscv_vfredosum(s, __riscv_vfmv_s_f_f64m1(0, __riscv_vsetvlmax_e64m1()), vlmax));
    }
};

template<>
struct MaskedNormDiffL2_RVV<short, double> {
    double operator() (const short* src1, const short* src2, const uchar* mask, int len, int cn) const {
        int vlmax = __riscv_vsetvlmax_e16m2();
        auto s = __riscv_vfmv_v_f_f64m8(0, vlmax);
        for (int cn_index = 0; cn_index < cn; cn_index++) {
            int vl;
            for (int i = 0; i < len; i += vl) {
                vl = __riscv_vsetvl_e16m2(len - i);
                auto v1 = __riscv_vlse16_v_i16m2(src1 + cn * i + cn_index, sizeof(short) * cn, vl);
                auto v2 = __riscv_vlse16_v_i16m2(src2 + cn * i + cn_index, sizeof(short) * cn, vl);
                auto v = custom_intrin::__riscv_vabd(v1, v2, vl);
                auto m = __riscv_vle8_v_u8m1(mask + i, vl);
                auto b = __riscv_vmsne(m, 0, vl);
                auto v_mul = __riscv_vwmulu(b, v, v, vl);
                s = __riscv_vfadd_tumu(b, s, s, __riscv_vfwcvt_f(b, v_mul, vl), vl);
            }
        }
        return __riscv_vfmv_f(__riscv_vfredosum(s, __riscv_vfmv_s_f_f64m1(0, __riscv_vsetvlmax_e64m1()), vlmax));
    }
};

template<>
struct MaskedNormDiffL2_RVV<int, double> {
    double operator() (const int* src1, const int* src2, const uchar* mask, int len, int cn) const {
        int vlmax = __riscv_vsetvlmax_e32m4();
        auto s = __riscv_vfmv_v_f_f64m8(0, vlmax);
        for (int cn_index = 0; cn_index < cn; cn_index++) {
            int vl;
            for (int i = 0; i < len; i += vl) {
                vl = __riscv_vsetvl_e16m2(len - i);
                auto v1 = __riscv_vlse32_v_i32m4(src1 + cn * i + cn_index, sizeof(int) * cn, vl);
                auto v2 = __riscv_vlse32_v_i32m4(src2 + cn * i + cn_index, sizeof(int) * cn, vl);
                auto v = custom_intrin::__riscv_vabd(v1, v2, vl);
                auto m = __riscv_vle8_v_u8m1(mask + i, vl);
                auto b = __riscv_vmsne(m, 0, vl);
                auto v_mul = __riscv_vwmulu(b, v, v, vl);
                s = __riscv_vfadd_tumu(b, s, s, __riscv_vfcvt_f(b, v_mul, vl), vl);
            }
        }
        return __riscv_vfmv_f(__riscv_vfredosum(s, __riscv_vfmv_s_f_f64m1(0, __riscv_vsetvlmax_e64m1()), vlmax));
    }
};

template<>
struct MaskedNormDiffL2_RVV<float, double> {
    double operator() (const float* src1, const float* src2, const uchar* mask, int len, int cn) const {
        int vlmax = __riscv_vsetvlmax_e32m4();
        auto s = __riscv_vfmv_v_f_f64m8(0, vlmax);
        if (cn == 1) {
            int vl;
            for (int i = 0; i < len; i += vl) {
                vl = __riscv_vsetvl_e32m4(len - i);
                auto v1 = __riscv_vle32_v_f32m4(src1 + i, vl);
                auto v2 = __riscv_vle32_v_f32m4(src2 + i, vl);
                auto v = __riscv_vfsub(v1, v2, vl);
                auto m = __riscv_vle8_v_u8m1(mask + i, vl);
                auto b = __riscv_vmsne(m, 0, vl);
                auto v_mul = __riscv_vfwmul(b, v, v, vl);
                s = __riscv_vfadd_tumu(b, s, s, v_mul, vl);
            }
        } else {
            for (int cn_index = 0; cn_index < cn; cn_index++) {
                int vl;
                for (int i = 0; i < len; i += vl) {
                    vl = __riscv_vsetvl_e32m4(len - i);
                    auto v1 = __riscv_vlse32_v_f32m4(src1 + cn * i + cn_index, sizeof(float) * cn, vl);
                    auto v2 = __riscv_vlse32_v_f32m4(src2 + cn * i + cn_index, sizeof(float) * cn, vl);
                    auto v = __riscv_vfsub(v1, v2, vl);
                    auto m = __riscv_vle8_v_u8m1(mask + i, vl);
                    auto b = __riscv_vmsne(m, 0, vl);
                    auto v_mul = __riscv_vfwmul(b, v, v, vl);
                    s = __riscv_vfadd_tumu(b, s, s, v_mul, vl);
                }
            }
        }
        return __riscv_vfmv_f(__riscv_vfredosum(s, __riscv_vfmv_s_f_f64m1(0, __riscv_vsetvlmax_e64m1()), vlmax));
    }
};

template<>
struct MaskedNormDiffL2_RVV<double, double> {
    double operator() (const double* src1, const double* src2, const uchar* mask, int len, int cn) const {
        int vlmax = __riscv_vsetvlmax_e64m8();
        auto s = __riscv_vfmv_v_f_f64m8(0, vlmax);
        for (int cn_index = 0; cn_index < cn; cn_index++) {
            int vl;
            for (int i = 0; i < len; i += vl) {
                vl = __riscv_vsetvl_e64m8(len - i);
                auto v1 = __riscv_vlse64_v_f64m8(src1 + cn * i + cn_index, sizeof(double) * cn, vl);
                auto v2 = __riscv_vlse64_v_f64m8(src2 + cn * i + cn_index, sizeof(double) * cn, vl);
                auto v = __riscv_vfsub(v1, v2, vl);
                auto m = __riscv_vle8_v_u8m1(mask + i, vl);
                auto b = __riscv_vmsne(m, 0, vl);
                auto v_mul = __riscv_vfmul(b, v, v, vl);
                s = __riscv_vfadd_tumu(b, s, s, v_mul, vl);
            }
        }
        return __riscv_vfmv_f(__riscv_vfredosum(s, __riscv_vfmv_s_f_f64m1(0, __riscv_vsetvlmax_e64m1()), vlmax));
    }
};

template<typename T, typename ST> int
normDiffInf_(const T* src1, const T* src2, const uchar* mask, ST* _result, int len, int cn) {
    ST result = *_result;
    if( !mask ) {
        NormDiffInf_RVV<T, ST> op;
        result = std::max(result, op(src1, src2, len*cn));
    } else {
        MaskedNormDiffInf_RVV<T, ST> op;
        result = std::max(result, op(src1, src2, mask, len, cn));
    }
    *_result = result;
    return 0;
}

template<typename T, typename ST> int
normDiffL1_(const T* src1, const T* src2, const uchar* mask, ST* _result, int len, int cn) {
    ST result = *_result;
    if( !mask ) {
        NormDiffL1_RVV<T, ST> op;
        result += op(src1, src2, len*cn);
    } else {
        MaskedNormDiffL1_RVV<T, ST> op;
        result += op(src1, src2, mask, len, cn);
    }
    *_result = result;
    return 0;
}

template<typename T, typename ST> int
normDiffL2_(const T* src1, const T* src2, const uchar* mask, ST* _result, int len, int cn) {
    ST result = *_result;
    if( !mask ) {
        NormDiffL2_RVV<T, ST> op;
        result += op(src1, src2, len*cn);
    } else {
        MaskedNormDiffL2_RVV<T, ST> op;
        result += op(src1, src2, mask, len, cn);
    }
    *_result = result;
    return 0;
}

#define CV_HAL_RVV_DEF_NORM_DIFF_FUNC(L, suffix, type, ntype) \
    static int normDiff##L##_##suffix(const type* src1, const type* src2, const uchar* mask, ntype* r, int len, int cn) \
    { return normDiff##L##_(src1, src2, mask, r, len, cn); }

#define CV_HAL_RVV_DEF_NORM_DIFF_ALL(suffix, type, inftype, l1type, l2type) \
    CV_HAL_RVV_DEF_NORM_DIFF_FUNC(Inf, suffix, type, inftype) \
    CV_HAL_RVV_DEF_NORM_DIFF_FUNC(L1, suffix, type, l1type) \
    CV_HAL_RVV_DEF_NORM_DIFF_FUNC(L2, suffix, type, l2type)

CV_HAL_RVV_DEF_NORM_DIFF_ALL(8u, uchar, int, int, int)
CV_HAL_RVV_DEF_NORM_DIFF_ALL(8s, schar, int, int, int)
CV_HAL_RVV_DEF_NORM_DIFF_ALL(16u, ushort, int, int, double)
CV_HAL_RVV_DEF_NORM_DIFF_ALL(16s, short, int, int, double)
CV_HAL_RVV_DEF_NORM_DIFF_ALL(32s, int, int, double, double)
CV_HAL_RVV_DEF_NORM_DIFF_ALL(32f, float, float, double, double)
CV_HAL_RVV_DEF_NORM_DIFF_ALL(64f, double, double, double, double)

#undef CV_HAL_RVV_DEF_NORM_DIFF_ALL
#undef CV_HAL_RVV_DEF_NORM_DIFF_FUNC

}

using NormDiffFunc = int (*)(const uchar*, const uchar*, const uchar*, uchar*, int, int);
inline int normDiff(const uchar* src1, size_t src1_step, const uchar* src2, size_t src2_step, const uchar* mask,
                    size_t mask_step, int width, int height, int type, int norm_type, double* result)
{
    int depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);

    bool relative = norm_type & NORM_RELATIVE;
    norm_type &= ~NORM_RELATIVE;

    if (result == nullptr || depth == CV_16F || (norm_type > NORM_L2SQR && !relative)) {
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    // [FIXME] append 0's when merging to 5.x
    static NormDiffFunc norm_diff_tab[3][CV_DEPTH_MAX] = {
        {
            (NormDiffFunc)(normDiffInf_8u),  (NormDiffFunc)(normDiffInf_8s),
            (NormDiffFunc)(normDiffInf_16u), (NormDiffFunc)(normDiffInf_16s),
            (NormDiffFunc)(normDiffInf_32s), (NormDiffFunc)(normDiffInf_32f),
            (NormDiffFunc)(normDiffInf_64f), 0,
        },
        {
            (NormDiffFunc)(normDiffL1_8u),  (NormDiffFunc)(normDiffL1_8s),
            (NormDiffFunc)(normDiffL1_16u), (NormDiffFunc)(normDiffL1_16s),
            (NormDiffFunc)(normDiffL1_32s), (NormDiffFunc)(normDiffL1_32f),
            (NormDiffFunc)(normDiffL1_64f), 0,
        },
        {
            (NormDiffFunc)(normDiffL2_8u),  (NormDiffFunc)(normDiffL2_8s),
            (NormDiffFunc)(normDiffL2_16u), (NormDiffFunc)(normDiffL2_16s),
            (NormDiffFunc)(normDiffL2_32s), (NormDiffFunc)(normDiffL2_32f),
            (NormDiffFunc)(normDiffL2_64f), 0,
        },
    };

    static const size_t elem_size_tab[CV_DEPTH_MAX] = {
        sizeof(uchar),   sizeof(schar),
        sizeof(ushort),  sizeof(short),
        sizeof(int),     sizeof(float),
        sizeof(int64_t), 0,
    };
    CV_Assert(elem_size_tab[depth]);

    bool src_continuous = (src1_step == width * elem_size_tab[depth] * cn || (src1_step != width * elem_size_tab[depth] * cn && height == 1));
    src_continuous &= (src2_step == width * elem_size_tab[depth] * cn || (src2_step != width * elem_size_tab[depth] * cn && height == 1));
    bool mask_continuous = (mask_step == static_cast<size_t>(width));
    size_t nplanes = 1;
    size_t size = width * height;
    if ((mask && (!src_continuous || !mask_continuous)) || !src_continuous) {
        nplanes = height;
        size = width;
    }

    NormDiffFunc func = norm_diff_tab[norm_type >> 1][depth];
    if (func == nullptr) {
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    // Handle overflow
    union {
        double d;
        float f;
        unsigned u;
    } res;
    res.d = 0;
    if ((norm_type == NORM_L1 && depth <= CV_16S) ||
        ((norm_type == NORM_L2 || norm_type == NORM_L2SQR) && depth <= CV_8S)) {
        const size_t esz = elem_size_tab[depth] * cn;
        const int total = (int)size;
        const int intSumBlockSize = (norm_type == NORM_L1 && depth <= CV_8S ? (1 << 23) : (1 << 15))/cn;
        const int blockSize = std::min(total, intSumBlockSize);
        int isum = 0;
        int count = 0;
        auto _src1 = src1, _src2 = src2;
        auto _mask = mask;
        for (size_t i = 0; i < nplanes; i++) {
            if ((mask && (!src_continuous || !mask_continuous)) || !src_continuous) {
                _src1 = src1 + src1_step * i;
                _src2 = src2 + src2_step * i;
                _mask = mask + mask_step * i;
            }
            for (int j = 0; j < total; j += blockSize) {
                int bsz = std::min(total - j, blockSize);
                func(_src1, _src2, _mask, (uchar*)&isum, bsz, cn);
                count += bsz;
                if (count + blockSize >= intSumBlockSize || (i + 1 >= nplanes && j + bsz >= total)) {
                    res.d += isum;
                    isum = 0;
                    count = 0;
                }
                _src1 += bsz * esz;
                _src2 += bsz * esz;
                if (mask) {
                    _mask += bsz;
                }
            }
        }
    } else {
        auto _src1 = src1, _src2 = src2;
        auto _mask = mask;
        for (size_t i = 0; i < nplanes; i++) {
            if ((mask && (!src_continuous || !mask_continuous)) || !src_continuous) {
                _src1 = src1 + src1_step * i;
                _src2 = src2 + src2_step * i;
                _mask = mask + mask_step * i;
            }
            func(_src1, _src2, _mask, (uchar*)&res, (int)size, cn);
        }
    }

    if (norm_type == NORM_INF) {
        if (depth == CV_64F) {
            *result = res.d;
        } else if (depth == CV_32F) {
            *result = res.f;
        } else {
            *result = res.u;
        }
    } else if (norm_type == NORM_L2) {
        *result = std::sqrt(res.d);
    } else {
        *result = res.d;
    }

    if(relative)
    {
        double result_;
        int ret = cv::cv_hal_rvv::norm::norm(src2, src2_step, mask, mask_step, width, height, type, norm_type, &result_);
        if(ret == CV_HAL_ERROR_OK)
        {
            *result /= result_ + DBL_EPSILON;
        }
    }

    return CV_HAL_ERROR_OK;
}

}}}

#endif
