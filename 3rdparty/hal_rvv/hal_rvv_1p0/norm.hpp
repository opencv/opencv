// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2025, Institute of Software, Chinese Academy of Sciences.

#ifndef OPENCV_HAL_RVV_NORM_HPP_INCLUDED
#define OPENCV_HAL_RVV_NORM_HPP_INCLUDED

#include "common.hpp"

namespace cv { namespace cv_hal_rvv {

#undef cv_hal_getNormFunc
#define cv_hal_getNormFunc cv::cv_hal_rvv::get_norm_func

namespace {

template <typename T, typename ST>
struct NormInf_RVV {
    inline ST operator() (const T* src, int n) const { return 0; }
};

template <typename T, typename ST>
struct NormL1_RVV {
    inline ST operator() (const T* src, int n) const { return 0; }
};

template <typename T, typename ST>
struct NormL2_RVV {
    inline ST operator() (const T* src, int n) const { return 0; }
};

template<>
struct NormInf_RVV<uchar, int> {
    int operator() (const uchar* src, int n) const {
        int vlmax = __riscv_vsetvlmax_e8m8();
        auto s = __riscv_vmv_v_x_u8m8(0, vlmax);
        int vl;
        for (int i = 0; i < n; i += vl) {
            vl = __riscv_vsetvl_e8m8(n - i);
            auto v = __riscv_vle8_v_u8m8(src + i, vl);
            s = __riscv_vmaxu_tu(s, s, v, vl);
        }
        return __riscv_vmv_x(__riscv_vredmaxu(s, __riscv_vmv_s_x_u8m1(0, __riscv_vsetvlmax_e8m1()), vlmax));
    }
};

template<>
struct NormInf_RVV<schar, int> {
    int operator() (const schar* src, int n) const {
        int vlmax = __riscv_vsetvlmax_e8m8();
        auto s = __riscv_vmv_v_x_u8m8(0, vlmax);
        int vl;
        for (int i = 0; i < n; i += vl) {
            vl = __riscv_vsetvl_e8m8(n - i);
            auto v = __riscv_vle8_v_i8m8(src + i, vl);
            s = __riscv_vmaxu_tu(s, s, custom_intrin::__riscv_vabs(v, vl), vl);
        }
        return __riscv_vmv_x(__riscv_vredmaxu(s, __riscv_vmv_s_x_u8m1(0, __riscv_vsetvlmax_e8m1()), vlmax));
    }
};

template<>
struct NormInf_RVV<ushort, int> {
    int operator() (const ushort* src, int n) const {
        int vlmax = __riscv_vsetvlmax_e16m8();
        auto s = __riscv_vmv_v_x_u16m8(0, vlmax);
        int vl;
        for (int i = 0; i < n; i += vl) {
            vl = __riscv_vsetvl_e16m8(n - i);
            auto v = __riscv_vle16_v_u16m8(src + i, vl);
            s = __riscv_vmaxu_tu(s, s, v, vl);
        }
        return __riscv_vmv_x(__riscv_vredmaxu(s, __riscv_vmv_s_x_u16m1(0, __riscv_vsetvlmax_e16m1()), vlmax));
    }
};

template<>
struct NormInf_RVV<short, int> {
    int operator() (const short* src, int n) const {
        int vlmax = __riscv_vsetvlmax_e16m8();
        auto s = __riscv_vmv_v_x_u16m8(0, vlmax);
        int vl;
        for (int i = 0; i < n; i += vl) {
            vl = __riscv_vsetvl_e16m8(n - i);
            auto v = __riscv_vle16_v_i16m8(src + i, vl);
            s = __riscv_vmaxu_tu(s, s, custom_intrin::__riscv_vabs(v, vl), vl);
        }
        return __riscv_vmv_x(__riscv_vredmaxu(s, __riscv_vmv_s_x_u16m1(0, __riscv_vsetvlmax_e16m1()), vlmax));
    }
};

template<>
struct NormInf_RVV<int, int> {
    int operator() (const int* src, int n) const {
        int vlmax = __riscv_vsetvlmax_e32m8();
        auto s = __riscv_vmv_v_x_u32m8(0, vlmax);
        int vl;
        for (int i = 0; i < n; i += vl) {
            vl = __riscv_vsetvl_e32m8(n - i);
            auto v = __riscv_vle32_v_i32m8(src + i, vl);
            s = __riscv_vmaxu_tu(s, s, custom_intrin::__riscv_vabs(v, vl), vl);
        }
        return __riscv_vmv_x(__riscv_vredmaxu(s, __riscv_vmv_s_x_u32m1(0, __riscv_vsetvlmax_e32m1()), vlmax));
    }
};

template<>
struct NormInf_RVV<float, float> {
    float operator() (const float* src, int n) const {
        int vlmax = __riscv_vsetvlmax_e32m8();
        auto s = __riscv_vfmv_v_f_f32m8(0, vlmax);
        int vl;
        for (int i = 0; i < n; i += vl) {
            vl = __riscv_vsetvl_e32m8(n - i);
            auto v = __riscv_vle32_v_f32m8(src + i, vl);
            s = __riscv_vfmax_tu(s, s, __riscv_vfabs(v, vl), vl);
        }
        return __riscv_vfmv_f(__riscv_vfredmax(s, __riscv_vfmv_s_f_f32m1(0, __riscv_vsetvlmax_e32m1()), vlmax));
    }
};

template<>
struct NormInf_RVV<double, double> {
    double operator() (const double* src, int n) const {
        int vlmax = __riscv_vsetvlmax_e64m8();
        auto s = __riscv_vfmv_v_f_f64m8(0, vlmax);
        int vl;
        for (int i = 0; i < n; i += vl) {
            vl = __riscv_vsetvl_e64m8(n - i);
            auto v = __riscv_vle64_v_f64m8(src + i, vl);
            s = __riscv_vfmax_tu(s, s, __riscv_vfabs(v, vl), vl);
        }
        return __riscv_vfmv_f(__riscv_vfredmax(s, __riscv_vfmv_s_f_f64m1(0, __riscv_vsetvlmax_e64m1()), vlmax));
    }
};

template<>
struct NormL1_RVV<uchar, int> {
    int operator() (const uchar* src, int n) const {
        auto s = __riscv_vmv_v_x_u32m1(0, __riscv_vsetvlmax_e32m1());
        auto zero = __riscv_vmv_v_x_u16m1(0, __riscv_vsetvlmax_e16m1());
        int vl;
        for (int i = 0; i < n; i += vl) {
            vl = __riscv_vsetvl_e8m8(n - i);
            auto v = __riscv_vle8_v_u8m8(src + i, vl);
            s = __riscv_vwredsumu(__riscv_vwredsumu_tu(zero, v, zero, vl), s, vl);
        }
        return __riscv_vmv_x(s);
    }
};

template<>
struct NormL1_RVV<schar, int> {
    int operator() (const schar* src, int n) const {
        auto s = __riscv_vmv_v_x_u32m1(0, __riscv_vsetvlmax_e32m1());
        auto zero = __riscv_vmv_v_x_u16m1(0, __riscv_vsetvlmax_e16m1());
        int vl;
        for (int i = 0; i < n; i += vl) {
            vl = __riscv_vsetvl_e8m8(n - i);
            auto v = custom_intrin::__riscv_vabs(__riscv_vle8_v_i8m8(src + i, vl), vl);
            s = __riscv_vwredsumu(__riscv_vwredsumu_tu(zero, v, zero, vl), s, vl);
        }
        return __riscv_vmv_x(s);
    }
};

template<>
struct NormL1_RVV<ushort, int> {
    int operator() (const ushort* src, int n) const {
        auto s = __riscv_vmv_v_x_u32m1(0, __riscv_vsetvlmax_e32m1());
        int vl;
        for (int i = 0; i < n; i += vl) {
            vl = __riscv_vsetvl_e16m8(n - i);
            auto v = __riscv_vle16_v_u16m8(src + i, vl);
            s = __riscv_vwredsumu(v, s, vl);
        }
        return __riscv_vmv_x(s);
    }
};

template<>
struct NormL1_RVV<short, int> {
    int operator() (const short* src, int n) const {
        auto s = __riscv_vmv_v_x_u32m1(0, __riscv_vsetvlmax_e32m1());
        int vl;
        for (int i = 0; i < n; i += vl) {
            vl = __riscv_vsetvl_e16m8(n - i);
            auto v = custom_intrin::__riscv_vabs(__riscv_vle16_v_i16m8(src + i, vl), vl);
            s = __riscv_vwredsumu(v, s, vl);
        }
        return __riscv_vmv_x(s);
    }
};

template<>
struct NormL1_RVV<int, double> {
    double operator() (const int* src, int n) const {
        int vlmax = __riscv_vsetvlmax_e32m4();
        auto s = __riscv_vfmv_v_f_f64m8(0, vlmax);
        int vl;
        for (int i = 0; i < n; i += vl) {
            vl = __riscv_vsetvl_e32m4(n - i);
            auto v = custom_intrin::__riscv_vabs(__riscv_vle32_v_i32m4(src + i, vl), vl);
            s = __riscv_vfadd_tu(s, s, __riscv_vfwcvt_f(v, vl), vl);
        }
        return __riscv_vfmv_f(__riscv_vfredosum(s, __riscv_vfmv_s_f_f64m1(0, __riscv_vsetvlmax_e64m1()), vlmax));
    }
};

template<>
struct NormL1_RVV<float, double> {
    double operator() (const float* src, int n) const {
        int vlmax = __riscv_vsetvlmax_e32m4();
        auto s = __riscv_vfmv_v_f_f64m8(0, vlmax);
        int vl;
        for (int i = 0; i < n; i += vl) {
            vl = __riscv_vsetvl_e32m4(n - i);
            auto v = __riscv_vfabs(__riscv_vle32_v_f32m4(src + i, vl), vl);
            s = __riscv_vfadd_tu(s, s, __riscv_vfwcvt_f(v, vl), vl);
        }
        return __riscv_vfmv_f(__riscv_vfredosum(s, __riscv_vfmv_s_f_f64m1(0, __riscv_vsetvlmax_e64m1()), vlmax));
    }
};

template<>
struct NormL1_RVV<double, double> {
    double operator() (const double* src, int n) const {
        int vlmax = __riscv_vsetvlmax_e64m8();
        auto s = __riscv_vfmv_v_f_f64m8(0, vlmax);
        int vl;
        for (int i = 0; i < n; i += vl) {
            vl = __riscv_vsetvl_e64m8(n - i);
            auto v = __riscv_vle64_v_f64m8(src + i, vl);
            s = __riscv_vfadd_tu(s, s, __riscv_vfabs(v, vl), vl);
        }
        return __riscv_vfmv_f(__riscv_vfredosum(s, __riscv_vfmv_s_f_f64m1(0, __riscv_vsetvlmax_e64m1()), vlmax));
    }
};

template<>
struct NormL2_RVV<uchar, int> {
    int operator() (const uchar* src, int n) const {
        auto s = __riscv_vmv_v_x_u32m1(0, __riscv_vsetvlmax_e32m1());
        int vl;
        for (int i = 0; i < n; i += vl) {
            vl = __riscv_vsetvl_e8m4(n - i);
            auto v = __riscv_vle8_v_u8m4(src + i, vl);
            s = __riscv_vwredsumu(__riscv_vwmulu(v, v, vl), s, vl);
        }
        return __riscv_vmv_x(s);
    }
};

template<>
struct NormL2_RVV<schar, int> {
    int operator() (const schar* src, int n) const {
        auto s = __riscv_vmv_v_x_i32m1(0, __riscv_vsetvlmax_e32m1());
        int vl;
        for (int i = 0; i < n; i += vl) {
            vl = __riscv_vsetvl_e8m4(n - i);
            auto v = __riscv_vle8_v_i8m4(src + i, vl);
            s = __riscv_vwredsum(__riscv_vwmul(v, v, vl), s, vl);
        }
        return __riscv_vmv_x(s);
    }
};

template<>
struct NormL2_RVV<ushort, double> {
    double operator() (const ushort* src, int n) const {
        int vlmax = __riscv_vsetvlmax_e16m2();
        auto s = __riscv_vfmv_v_f_f64m8(0, vlmax);
        int vl;
        for (int i = 0; i < n; i += vl) {
            vl = __riscv_vsetvl_e16m2(n - i);
            auto v = __riscv_vle16_v_u16m2(src + i, vl);
            auto v_mul = __riscv_vwmulu(v, v, vl);
            s = __riscv_vfadd_tu(s, s, __riscv_vfwcvt_f(v_mul, vl), vl);
        }
        return __riscv_vfmv_f(__riscv_vfredosum(s, __riscv_vfmv_s_f_f64m1(0, __riscv_vsetvlmax_e64m1()), vlmax));
    }
};

template<>
struct NormL2_RVV<short, double> {
    double operator() (const short* src, int n) const {
        int vlmax = __riscv_vsetvlmax_e16m2();
        auto s = __riscv_vfmv_v_f_f64m8(0, vlmax);
        int vl;
        for (int i = 0; i < n; i += vl) {
            vl = __riscv_vsetvl_e16m2(n - i);
            auto v = __riscv_vle16_v_i16m2(src + i, vl);
            auto v_mul = __riscv_vwmul(v, v, vl);
            s = __riscv_vfadd_tu(s, s, __riscv_vfwcvt_f(v_mul, vl), vl);
        }
        return __riscv_vfmv_f(__riscv_vfredosum(s, __riscv_vfmv_s_f_f64m1(0, __riscv_vsetvlmax_e32m1()), vlmax));
    }
};

template<>
struct NormL2_RVV<int, double> {
    double operator() (const int* src, int n) const {
        int vlmax = __riscv_vsetvlmax_e32m4();
        auto s = __riscv_vfmv_v_f_f64m8(0, vlmax);
        int vl;
        for (int i = 0; i < n; i += vl) {
            vl = __riscv_vsetvl_e32m4(n - i);
            auto v = __riscv_vle32_v_i32m4(src + i, vl);
            auto v_mul = __riscv_vwmul(v, v, vl);
            s = __riscv_vfadd_tu(s, s, __riscv_vfcvt_f(v_mul, vl), vl);
        }
        return __riscv_vfmv_f(__riscv_vfredosum(s, __riscv_vfmv_s_f_f64m1(0, __riscv_vsetvlmax_e64m1()), vlmax));
    }
};

template<>
struct NormL2_RVV<float, double> {
    double operator() (const float* src, int n) const {
        int vlmax = __riscv_vsetvlmax_e32m4();
        auto s = __riscv_vfmv_v_f_f64m8(0, vlmax);
        int vl;
        for (int i = 0; i < n; i += vl) {
            vl = __riscv_vsetvl_e32m4(n - i);
            auto v = __riscv_vle32_v_f32m4(src + i, vl);
            auto v_mul = __riscv_vfwmul(v, v, vl);
            s = __riscv_vfadd_tu(s, s, v_mul, vl);
        }
        return __riscv_vfmv_f(__riscv_vfredosum(s, __riscv_vfmv_s_f_f64m1(0, __riscv_vsetvlmax_e64m1()), vlmax));
    }
};

template<>
struct NormL2_RVV<double, double> {
    double operator() (const double* src, int n) const {
        int vlmax = __riscv_vsetvlmax_e64m8();
        auto s = __riscv_vfmv_v_f_f64m8(0, vlmax);
        int vl;
        for (int i = 0; i < n; i += vl) {
            vl = __riscv_vsetvl_e64m8(n - i);
            auto v = __riscv_vle64_v_f64m8(src + i, vl);
            auto v_mul = __riscv_vfmul(v, v, vl);
            s = __riscv_vfadd_tu(s, s, v_mul, vl);
        }
        return __riscv_vfmv_f(__riscv_vfredosum(s, __riscv_vfmv_s_f_f64m1(0, __riscv_vsetvlmax_e64m1()), vlmax));
    }
};

// Norm with mask

template <typename T, typename ST>
struct MaskedNormInf_RVV {
    inline ST operator() (const T* src, const uchar* mask, int len, int cn) const {
        ST s = 0;
        for( int i = 0; i < len; i++, src += cn ) {
            if( mask[i] ) {
                for( int k = 0; k < cn; k++ ) {
                    s = std::max(s, ST(cv_abs(src[k])));
                }
            }
        }
        return s;
    }
};

template <typename T, typename ST>
struct MaskedNormL1_RVV {
    inline ST operator() (const T* src, const uchar* mask, int len, int cn) const {
        ST s = 0;
        for( int i = 0; i < len; i++, src += cn ) {
            if( mask[i] ) {
                for( int k = 0; k < cn; k++ ) {
                    s += cv_abs(src[k]);
                }
            }
        }
        return s;
    }
};

template <typename T, typename ST>
struct MaskedNormL2_RVV {
    inline ST operator() (const T* src, const uchar* mask, int len, int cn) const {
        ST s = 0;
        for( int i = 0; i < len; i++, src += cn ) {
            if( mask[i] ) {
                for( int k = 0; k < cn; k++ ) {
                    T v = src[k];
                    s += (ST)v*v;
                }
            }
        }
        return s;
    }
};

template<>
struct MaskedNormInf_RVV<uchar, int> {
    int operator() (const uchar* src, const uchar* mask, int len, int cn) const {
        int vlmax = __riscv_vsetvlmax_e8m8();
        auto s = __riscv_vmv_v_x_u8m8(0, vlmax);
        if (cn == 1) {
            int vl;
            for (int i = 0; i < len; i += vl) {
                vl = __riscv_vsetvl_e8m8(len - i);
                auto v = __riscv_vle8_v_u8m8(src + i, vl);
                auto m = __riscv_vle8_v_u8m8(mask + i, vl);
                auto b = __riscv_vmsne(m, 0, vl);
                s = __riscv_vmaxu_tumu(b, s, s, v, vl);
            }
        } else if (cn == 4) {
            int vl;
            for (int i = 0; i < len; i += vl) {
                vl = __riscv_vsetvl_e8m2(len - i);
                auto v = __riscv_vle8_v_u8m8(src + i * 4, vl * 4);
                auto m = __riscv_vle8_v_u8m2(mask + i, vl);
                auto b = __riscv_vmsne(__riscv_vreinterpret_u8m8(__riscv_vmul(__riscv_vzext_vf4(__riscv_vminu(m, 1, vl), vl), 0x01010101, vl)), 0, vl * 4);
                s = __riscv_vmaxu_tumu(b, s, s, v, vl * 4);
            }
        } else {
            for (int cn_index = 0; cn_index < cn; cn_index++) {
                int vl;
                for (int i = 0; i < len; i += vl) {
                    vl = __riscv_vsetvl_e8m8(len - i);
                    auto v = __riscv_vlse8_v_u8m8(src + cn * i + cn_index, sizeof(uchar) * cn, vl);
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
struct MaskedNormL1_RVV<uchar, int> {
    int operator() (const uchar* src, const uchar* mask, int len, int cn) const {
        auto s = __riscv_vmv_v_x_u32m1(0, __riscv_vsetvlmax_e32m1());
        auto zero = __riscv_vmv_v_x_u16m1(0, __riscv_vsetvlmax_e16m1());
        if (cn == 1) {
            int vl;
            for (int i = 0; i < len; i += vl) {
                vl = __riscv_vsetvl_e8m8(len - i);
                auto v = __riscv_vle8_v_u8m8(src + i, vl);
                auto m = __riscv_vle8_v_u8m8(mask + i, vl);
                auto b = __riscv_vmsne(m, 0, vl);
                s = __riscv_vwredsumu(__riscv_vwredsumu_tum(b, zero, v, zero, vl), s, __riscv_vsetvlmax_e16m1());
            }
        } else if (cn == 4) {
            int vl;
            for (int i = 0; i < len; i += vl) {
                vl = __riscv_vsetvl_e8m2(len - i);
                auto v = __riscv_vle8_v_u8m8(src + i * 4, vl * 4);
                auto m = __riscv_vle8_v_u8m2(mask + i, vl);
                auto b = __riscv_vmsne(__riscv_vreinterpret_u8m8(__riscv_vmul(__riscv_vzext_vf4(__riscv_vminu(m, 1, vl), vl), 0x01010101, vl)), 0, vl * 4);
                s = __riscv_vwredsumu(__riscv_vwredsumu_tum(b, zero, v, zero, vl * 4), s, __riscv_vsetvlmax_e16m1());
            }
        } else {
            for (int cn_index = 0; cn_index < cn; cn_index++) {
                int vl;
                for (int i = 0; i < len; i += vl) {
                    vl = __riscv_vsetvl_e8m8(len - i);
                    auto v = __riscv_vlse8_v_u8m8(src + cn * i + cn_index, sizeof(uchar) * cn, vl);
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
struct MaskedNormL2_RVV<uchar, int> {
    int operator() (const uchar* src, const uchar* mask, int len, int cn) const {
        int vlmax = __riscv_vsetvlmax_e8m2();
        auto s = __riscv_vmv_v_x_u32m8(0, vlmax);
        if (cn == 1) {
            int vl;
            for (int i = 0; i < len; i += vl) {
                vl = __riscv_vsetvl_e8m2(len - i);
                auto v = __riscv_vle8_v_u8m2(src + i, vl);
                auto m = __riscv_vle8_v_u8m2(mask + i, vl);
                auto b = __riscv_vmsne(m, 0, vl);
                auto v_mul = __riscv_vwmulu(b, v, v, vl);
                s = __riscv_vadd_tumu(b, s, s, __riscv_vzext_vf2(b, v_mul, vl), vl);
            }
        } else if (cn == 4) {
            int vl;
            for (int i = 0; i < len; i += vl) {
                vl = __riscv_vsetvl_e8mf2(len - i);
                auto v = __riscv_vle8_v_u8m2(src + i * 4, vl * 4);
                auto m = __riscv_vle8_v_u8mf2(mask + i, vl);
                auto b = __riscv_vmsne(__riscv_vreinterpret_u8m2(__riscv_vmul(__riscv_vzext_vf4(__riscv_vminu(m, 1, vl), vl), 0x01010101, vl)), 0, vl * 4);
                auto v_mul = __riscv_vwmulu(b, v, v, vl * 4);
                s = __riscv_vadd_tumu(b, s, s, __riscv_vzext_vf2(b, v_mul, vl * 4), vl * 4);
            }
        } else {
            for (int cn_index = 0; cn_index < cn; cn_index++) {
                int vl;
                for (int i = 0; i < len; i += vl) {
                    vl = __riscv_vsetvl_e8m2(len - i);
                    auto v = __riscv_vlse8_v_u8m2(src + cn * i + cn_index, sizeof(uchar) * cn, vl);
                    auto m = __riscv_vle8_v_u8m2(mask + i, vl);
                    auto b = __riscv_vmsne(m, 0, vl);
                    auto v_mul = __riscv_vwmulu(b, v, v, vl);
                    s = __riscv_vadd_tumu(b, s, s, __riscv_vzext_vf2(b, v_mul, vl), vl);
                }
            }
        }
        return __riscv_vmv_x(__riscv_vredsum(s, __riscv_vmv_s_x_u32m1(0, __riscv_vsetvlmax_e32m1()), vlmax));
    }
};

template<>
struct MaskedNormInf_RVV<schar, int> {
    int operator() (const schar* src, const uchar* mask, int len, int cn) const {
        int vlmax = __riscv_vsetvlmax_e8m8();
        auto s = __riscv_vmv_v_x_u8m8(0, vlmax);
        for (int cn_index = 0; cn_index < cn; cn_index++) {
            int vl;
            for (int i = 0; i < len; i += vl) {
                vl = __riscv_vsetvl_e8m8(len - i);
                auto v = __riscv_vlse8_v_i8m8(src + cn * i + cn_index, sizeof(schar) * cn, vl);
                auto m = __riscv_vle8_v_u8m8(mask + i, vl);
                auto b = __riscv_vmsne(m, 0, vl);
                s = __riscv_vmaxu_tumu(b, s, s, custom_intrin::__riscv_vabs(v, vl), vl);
            }
        }
        return __riscv_vmv_x(__riscv_vredmaxu(s, __riscv_vmv_s_x_u8m1(0, __riscv_vsetvlmax_e8m1()), vlmax));
    }
};

template<>
struct MaskedNormL1_RVV<schar, int> {
    int operator() (const schar* src, const uchar* mask, int len, int cn) const {
        auto s = __riscv_vmv_v_x_u32m1(0, __riscv_vsetvlmax_e32m1());
        auto zero = __riscv_vmv_v_x_u16m1(0, __riscv_vsetvlmax_e16m1());
        for (int cn_index = 0; cn_index < cn; cn_index++) {
            int vl;
            for (int i = 0; i < len; i += vl) {
                vl = __riscv_vsetvl_e8m8(len - i);
                auto v = custom_intrin::__riscv_vabs(__riscv_vlse8_v_i8m8(src + cn * i + cn_index, sizeof(schar) * cn, vl), vl);
                auto m = __riscv_vle8_v_u8m8(mask + i, vl);
                auto b = __riscv_vmsne(m, 0, vl);
                s = __riscv_vwredsumu(__riscv_vwredsumu_tum(b, zero, v, zero, vl), s, __riscv_vsetvlmax_e16m1());
            }
        }
        return __riscv_vmv_x(s);
    }
};

template<>
struct MaskedNormL2_RVV<schar, int> {
    int operator() (const schar* src, const uchar* mask, int len, int cn) const {
        int vlmax = __riscv_vsetvlmax_e8m2();
        auto s = __riscv_vmv_v_x_i32m8(0, vlmax);
        for (int cn_index = 0; cn_index < cn; cn_index++) {
            int vl;
            for (int i = 0; i < len; i += vl) {
                vl = __riscv_vsetvl_e8m2(len - i);
                auto v = __riscv_vlse8_v_i8m2(src + cn * i + cn_index, sizeof(schar) * cn, vl);
                auto m = __riscv_vle8_v_u8m2(mask + i, vl);
                auto b = __riscv_vmsne(m, 0, vl);
                auto v_mul = __riscv_vwmul(b, v, v, vl);
                s = __riscv_vadd_tumu(b, s, s, __riscv_vsext_vf2(b, v_mul, vl), vl);
            }
        }
        return __riscv_vmv_x(__riscv_vredsum(s, __riscv_vmv_s_x_i32m1(0, __riscv_vsetvlmax_e32m1()), vlmax));
    }
};

template<>
struct MaskedNormInf_RVV<ushort, int> {
    int operator() (const ushort* src, const uchar* mask, int len, int cn) const {
        int vlmax = __riscv_vsetvlmax_e16m8();
        auto s = __riscv_vmv_v_x_u16m8(0, vlmax);
        for (int cn_index = 0; cn_index < cn; cn_index++) {
            int vl;
            for (int i = 0; i < len; i += vl) {
                vl = __riscv_vsetvl_e16m8(len - i);
                auto v = __riscv_vlse16_v_u16m8(src + cn * i + cn_index, sizeof(ushort) * cn, vl);
                auto m = __riscv_vle8_v_u8m4(mask + i, vl);
                auto b = __riscv_vmsne(m, 0, vl);
                s = __riscv_vmaxu_tumu(b, s, s, v, vl);
            }
        }
        return __riscv_vmv_x(__riscv_vredmaxu(s, __riscv_vmv_s_x_u16m1(0, __riscv_vsetvlmax_e16m1()), vlmax));
    }
};

template<>
struct MaskedNormL1_RVV<ushort, int> {
    int operator() (const ushort* src, const uchar* mask, int len, int cn) const {
        auto s = __riscv_vmv_v_x_u32m1(0, __riscv_vsetvlmax_e32m1());
        for (int cn_index = 0; cn_index < cn; cn_index++) {
            int vl;
            for (int i = 0; i < len; i += vl) {
                vl = __riscv_vsetvl_e8m4(len - i);
                auto v = __riscv_vlse16_v_u16m8(src + cn * i + cn_index, sizeof(ushort) * cn, vl);
                auto m = __riscv_vle8_v_u8m4(mask + i, vl);
                auto b = __riscv_vmsne(m, 0, vl);
                s = __riscv_vwredsumu_tum(b, s, v, s, vl);
            }
        }
        return __riscv_vmv_x(s);
    }
};

template<>
struct MaskedNormL2_RVV<ushort, double> {
    double operator() (const ushort* src, const uchar* mask, int len, int cn) const {
        int vlmax = __riscv_vsetvlmax_e16m2();
        auto s = __riscv_vfmv_v_f_f64m8(0, vlmax);
        for (int cn_index = 0; cn_index < cn; cn_index++) {
            int vl;
            for (int i = 0; i < len; i += vl) {
                vl = __riscv_vsetvl_e16m2(len - i);
                auto v = __riscv_vlse16_v_u16m2(src + cn * i + cn_index, sizeof(ushort) * cn, vl);
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
struct MaskedNormInf_RVV<short, int> {
    int operator() (const short* src, const uchar* mask, int len, int cn) const {
        int vlmax = __riscv_vsetvlmax_e16m8();
        auto s = __riscv_vmv_v_x_u16m8(0, vlmax);
        for (int cn_index = 0; cn_index < cn; cn_index++) {
            int vl;
            for (int i = 0; i < len; i += vl) {
                vl = __riscv_vsetvl_e16m8(len - i);
                auto v = __riscv_vlse16_v_i16m8(src + cn * i + cn_index, sizeof(short) * cn, vl);
                auto m = __riscv_vle8_v_u8m4(mask + i, vl);
                auto b = __riscv_vmsne(m, 0, vl);
                s = __riscv_vmaxu_tumu(b, s, s, custom_intrin::__riscv_vabs(v, vl), vl);
            }
        }
        return __riscv_vmv_x(__riscv_vredmaxu(s, __riscv_vmv_s_x_u16m1(0, __riscv_vsetvlmax_e16m1()), vlmax));
    }
};

template<>
struct MaskedNormL1_RVV<short, int> {
    int operator() (const short* src, const uchar* mask, int len, int cn) const {
        auto s = __riscv_vmv_v_x_u32m1(0, __riscv_vsetvlmax_e32m1());
        for (int cn_index = 0; cn_index < cn; cn_index++) {
            int vl;
            for (int i = 0; i < len; i += vl) {
                vl = __riscv_vsetvl_e8m4(len - i);
                auto v = custom_intrin::__riscv_vabs(__riscv_vlse16_v_i16m8(src + cn * i + cn_index, sizeof(short) * cn, vl), vl);
                auto m = __riscv_vle8_v_u8m4(mask + i, vl);
                auto b = __riscv_vmsne(m, 0, vl);
                s = __riscv_vwredsumu_tum(b, s, v, s, vl);
            }
        }
        return __riscv_vmv_x(s);
    }
};

template<>
struct MaskedNormL2_RVV<short, double> {
    double operator() (const short* src, const uchar* mask, int len, int cn) const {
        int vlmax = __riscv_vsetvlmax_e16m2();
        auto s = __riscv_vfmv_v_f_f64m8(0, vlmax);
        for (int cn_index = 0; cn_index < cn; cn_index++) {
            int vl;
            for (int i = 0; i < len; i += vl) {
                vl = __riscv_vsetvl_e16m2(len - i);
                auto v = __riscv_vlse16_v_i16m2(src + cn * i + cn_index, sizeof(short) * cn, vl);
                auto m = __riscv_vle8_v_u8m1(mask + i, vl);
                auto b = __riscv_vmsne(m, 0, vl);
                auto v_mul = __riscv_vwmul(b, v, v, vl);
                s = __riscv_vfadd_tumu(b, s, s, __riscv_vfwcvt_f(b, v_mul, vl), vl);
            }
        }
        return __riscv_vfmv_f(__riscv_vfredosum(s, __riscv_vfmv_s_f_f64m1(0, __riscv_vsetvlmax_e32m1()), vlmax));
    }
};

template<>
struct MaskedNormInf_RVV<int, int> {
    int operator() (const int* src, const uchar* mask, int len, int cn) const {
        int vlmax = __riscv_vsetvlmax_e32m8();
        auto s = __riscv_vmv_v_x_u32m8(0, vlmax);
        for (int cn_index = 0; cn_index < cn; cn_index++) {
            int vl;
            for (int i = 0; i < len; i += vl) {
                vl = __riscv_vsetvl_e32m8(len - i);
                auto v = __riscv_vlse32_v_i32m8(src + cn * i + cn_index, sizeof(int) * cn, vl);
                auto m = __riscv_vle8_v_u8m2(mask + i, vl);
                auto b = __riscv_vmsne(m, 0, vl);
                s = __riscv_vmaxu_tumu(b, s, s, custom_intrin::__riscv_vabs(v, vl), vl);
            }
        }
        return __riscv_vmv_x(__riscv_vredmaxu(s, __riscv_vmv_s_x_u32m1(0, __riscv_vsetvlmax_e32m1()), vlmax));
    }
};

template<>
struct MaskedNormL1_RVV<int, double> {
    double operator() (const int* src, const uchar* mask, int len, int cn) const {
        int vlmax = __riscv_vsetvlmax_e32m4();
        auto s = __riscv_vfmv_v_f_f64m8(0, vlmax);
        for (int cn_index = 0; cn_index < cn; cn_index++) {
            int vl;
            for (int i = 0; i < len; i += vl) {
                vl = __riscv_vsetvl_e32m4(len - i);
                auto v = __riscv_vlse32_v_i32m4(src + cn * i + cn_index, sizeof(int) * cn, vl);
                auto m = __riscv_vle8_v_u8m1(mask + i, vl);
                auto b = __riscv_vmsne(m, 0, vl);
                s = __riscv_vfadd_tumu(b, s, s, __riscv_vfwcvt_f(b, custom_intrin::__riscv_vabs(v, vl), vl), vl);
            }
        }
        return __riscv_vfmv_f(__riscv_vfredosum(s, __riscv_vfmv_s_f_f64m1(0, __riscv_vsetvlmax_e64m1()), vlmax));
    }
};

template<>
struct MaskedNormL2_RVV<int, double> {
    double operator() (const int* src, const uchar* mask, int len, int cn) const {
        int vlmax = __riscv_vsetvlmax_e32m4();
        auto s = __riscv_vfmv_v_f_f64m8(0, vlmax);
        for (int cn_index = 0; cn_index < cn; cn_index++) {
            int vl;
            for (int i = 0; i < len; i += vl) {
                vl = __riscv_vsetvl_e16m2(len - i);
                auto v = __riscv_vlse32_v_i32m4(src + cn * i + cn_index, sizeof(int) * cn, vl);
                auto m = __riscv_vle8_v_u8m1(mask + i, vl);
                auto b = __riscv_vmsne(m, 0, vl);
                auto v_mul = __riscv_vwmul(b, v, v, vl);
                s = __riscv_vfadd_tumu(b, s, s, __riscv_vfcvt_f(b, v_mul, vl), vl);
            }
        }
        return __riscv_vfmv_f(__riscv_vfredosum(s, __riscv_vfmv_s_f_f64m1(0, __riscv_vsetvlmax_e64m1()), vlmax));
    }
};

template<>
struct MaskedNormInf_RVV<float, float> {
    float operator() (const float* src, const uchar* mask, int len, int cn) const {
        int vlmax = __riscv_vsetvlmax_e32m8();
        auto s = __riscv_vfmv_v_f_f32m8(0, vlmax);
        if (cn == 1) {
            int vl;
            for (int i = 0; i < len; i += vl) {
                vl = __riscv_vsetvl_e32m8(len - i);
                auto v = __riscv_vle32_v_f32m8(src + i, vl);
                auto m = __riscv_vle8_v_u8m2(mask + i, vl);
                auto b = __riscv_vmsne(m, 0, vl);
                s = __riscv_vfmax_tumu(b, s, s, __riscv_vfabs(v, vl), vl);
            }
        } else {
            for (int cn_index = 0; cn_index < cn; cn_index++) {
                int vl;
                for (int i = 0; i < len; i += vl) {
                    vl = __riscv_vsetvl_e32m8(len - i);
                    auto v = __riscv_vlse32_v_f32m8(src + cn * i + cn_index, sizeof(float) * cn, vl);
                    auto m = __riscv_vle8_v_u8m2(mask + i, vl);
                    auto b = __riscv_vmsne(m, 0, vl);
                    s = __riscv_vfmax_tumu(b, s, s, __riscv_vfabs(v, vl), vl);
                }
            }
        }
        return __riscv_vfmv_f(__riscv_vfredmax(s, __riscv_vfmv_s_f_f32m1(0, __riscv_vsetvlmax_e32m1()), vlmax));
    }
};

template<>
struct MaskedNormL1_RVV<float, double> {
    double operator() (const float* src, const uchar* mask, int len, int cn) const {
        int vlmax = __riscv_vsetvlmax_e32m4();
        auto s = __riscv_vfmv_v_f_f64m8(0, vlmax);
        if (cn == 1) {
            int vl;
            for (int i = 0; i < len; i += vl) {
                vl = __riscv_vsetvl_e32m4(len - i);
                auto v = __riscv_vle32_v_f32m4(src + i, vl);
                auto m = __riscv_vle8_v_u8m1(mask + i, vl);
                auto b = __riscv_vmsne(m, 0, vl);
                s = __riscv_vfadd_tumu(b, s, s, __riscv_vfwcvt_f(b, __riscv_vfabs(v, vl), vl), vl);
            }
        } else {
            for (int cn_index = 0; cn_index < cn; cn_index++) {
                int vl;
                for (int i = 0; i < len; i += vl) {
                    vl = __riscv_vsetvl_e32m4(len - i);
                    auto v = __riscv_vlse32_v_f32m4(src + cn * i + cn_index, sizeof(float) * cn, vl);
                    auto m = __riscv_vle8_v_u8m1(mask + i, vl);
                    auto b = __riscv_vmsne(m, 0, vl);
                    s = __riscv_vfadd_tumu(b, s, s, __riscv_vfwcvt_f(b, __riscv_vfabs(v, vl), vl), vl);
                }
            }
        }
        return __riscv_vfmv_f(__riscv_vfredosum(s, __riscv_vfmv_s_f_f64m1(0, __riscv_vsetvlmax_e64m1()), vlmax));
    }
};

template<>
struct MaskedNormL2_RVV<float, double> {
    double operator() (const float* src, const uchar* mask, int len, int cn) const {
        int vlmax = __riscv_vsetvlmax_e32m4();
        auto s = __riscv_vfmv_v_f_f64m8(0, vlmax);
        if (cn == 1) {
            int vl;
            for (int i = 0; i < len; i += vl) {
                vl = __riscv_vsetvl_e32m4(len - i);
                auto v = __riscv_vle32_v_f32m4(src + i, vl);
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
                    auto v = __riscv_vlse32_v_f32m4(src + cn * i + cn_index, sizeof(float) * cn, vl);
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
struct MaskedNormInf_RVV<double, double> {
    double operator() (const double* src, const uchar* mask, int len, int cn) const {
        int vlmax = __riscv_vsetvlmax_e64m8();
        auto s = __riscv_vfmv_v_f_f64m8(0, vlmax);
        for (int cn_index = 0; cn_index < cn; cn_index++) {
            int vl;
            for (int i = 0; i < len; i += vl) {
                vl = __riscv_vsetvl_e64m8(len - i);
                auto v = __riscv_vlse64_v_f64m8(src + cn * i + cn_index, sizeof(double) * cn, vl);
                auto m = __riscv_vle8_v_u8m1(mask + i, vl);
                auto b = __riscv_vmsne(m, 0, vl);
                s = __riscv_vfmax_tumu(b, s, s, __riscv_vfabs(v, vl), vl);
            }
        }
        return __riscv_vfmv_f(__riscv_vfredmax(s, __riscv_vfmv_s_f_f64m1(0, __riscv_vsetvlmax_e64m1()), vlmax));
    }
};

template<>
struct MaskedNormL1_RVV<double, double> {
    double operator() (const double* src, const uchar* mask, int len, int cn) const {
        int vlmax = __riscv_vsetvlmax_e64m8();
        auto s = __riscv_vfmv_v_f_f64m8(0, vlmax);
        for (int cn_index = 0; cn_index < cn; cn_index++) {
            int vl;
            for (int i = 0; i < len; i += vl) {
                vl = __riscv_vsetvl_e64m8(len - i);
                auto v = __riscv_vlse64_v_f64m8(src + cn * i + cn_index, sizeof(double) * cn, vl);
                auto m = __riscv_vle8_v_u8m1(mask + i, vl);
                auto b = __riscv_vmsne(m, 0, vl);
                s = __riscv_vfadd_tumu(b, s, s, __riscv_vfabs(v, vl), vl);
            }
        }
        return __riscv_vfmv_f(__riscv_vfredosum(s, __riscv_vfmv_s_f_f64m1(0, __riscv_vsetvlmax_e64m1()), vlmax));
    }
};

template<>
struct MaskedNormL2_RVV<double, double> {
    double operator() (const double* src, const uchar* mask, int len, int cn) const {
        int vlmax = __riscv_vsetvlmax_e64m8();
        auto s = __riscv_vfmv_v_f_f64m8(0, vlmax);
        for (int cn_index = 0; cn_index < cn; cn_index++) {
            int vl;
            for (int i = 0; i < len; i += vl) {
                vl = __riscv_vsetvl_e64m8(len - i);
                auto v = __riscv_vlse64_v_f64m8(src + cn * i + cn_index, sizeof(double) * cn, vl);
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
normInf_(const T* src, const uchar* mask, ST* _result, int len, int cn) {
    ST result = *_result;
    if( !mask ) {
        NormInf_RVV<T, ST> op;
        result = std::max(result, op(src, len*cn));
    } else {
        MaskedNormInf_RVV<T, ST> op;
        result = std::max(result, op(src, mask, len, cn));
    }
    *_result = result;
    return 0;
}

template<typename T, typename ST> int
normL1_(const T* src, const uchar* mask, ST* _result, int len, int cn) {
    ST result = *_result;
    if( !mask ) {
        NormL1_RVV<T, ST> op;
        result += op(src, len*cn);
    } else {
        MaskedNormL1_RVV<T, ST> op;
        result += op(src, mask, len, cn);
    }
    *_result = result;
    return 0;
}

template<typename T, typename ST> int
normL2_(const T* src, const uchar* mask, ST* _result, int len, int cn) {
    ST result = *_result;
    if( !mask ) {
        NormL2_RVV<T, ST> op;
        result += op(src, len*cn);
    } else {
        MaskedNormL2_RVV<T, ST> op;
        result += op(src, mask, len, cn);
    }
    *_result = result;
    return 0;
}

#define CV_HAL_RVV_DEF_NORM_FUNC(L, suffix, type, ntype) \
    static int norm##L##_##suffix(const type* src, const uchar* mask, ntype* r, int len, int cn) \
    { return norm##L##_(src, mask, r, len, cn); }

#define CV_HAL_RVV_DEF_NORM_ALL(suffix, type, inftype, l1type, l2type) \
    CV_HAL_RVV_DEF_NORM_FUNC(Inf, suffix, type, inftype) \
    CV_HAL_RVV_DEF_NORM_FUNC(L1, suffix, type, l1type) \
    CV_HAL_RVV_DEF_NORM_FUNC(L2, suffix, type, l2type)

CV_HAL_RVV_DEF_NORM_ALL(8u, uchar, int, int, int)
CV_HAL_RVV_DEF_NORM_ALL(8s, schar, int, int, int)
CV_HAL_RVV_DEF_NORM_ALL(16u, ushort, int, int, double)
CV_HAL_RVV_DEF_NORM_ALL(16s, short, int, int, double)
CV_HAL_RVV_DEF_NORM_ALL(32s, int, int, double, double)
CV_HAL_RVV_DEF_NORM_ALL(32f, float, float, double, double)
CV_HAL_RVV_DEF_NORM_ALL(64f, double, double, double, double)

}

using NormFunc = int (*)(const uchar*, const uchar*, uchar*, int, int);
inline int get_norm_func(int normType, int depth, NormFunc *fn)
{
    // [FIXME] append 0's when merging to 5.x
    static NormFunc norm_tab[3][CV_DEPTH_MAX] =
    {
        {
            (NormFunc)(normInf_8u), (NormFunc)(normInf_8s),
            (NormFunc)(normInf_16u), (NormFunc)(normInf_16s),
            (NormFunc)(normInf_32s), (NormFunc)(normInf_32f),
            (NormFunc)(normInf_64f), 0,
        },
        {
            (NormFunc)(normL1_8u), (NormFunc)(normL1_8s),
            (NormFunc)(normL1_16u), (NormFunc)(normL1_16s),
            (NormFunc)(normL1_32s), (NormFunc)(normL1_32f),
            (NormFunc)(normL1_64f), 0,
        },
        {
            (NormFunc)(normL2_8u), (NormFunc)(normL2_8s),
            (NormFunc)(normL2_16u), (NormFunc)(normL2_16s),
            (NormFunc)(normL2_32s), (NormFunc)(normL2_32f),
            (NormFunc)(normL2_64f), 0,
        },
    };
    *fn = norm_tab[normType][depth];
    if (*fn) {
        return CV_HAL_ERROR_OK;
    }
    else {
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }
}

}}

#endif
