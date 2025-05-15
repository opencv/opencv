// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2025, Institute of Software, Chinese Academy of Sciences.
// Copyright (C) 2025, SpaceMIT Inc., all rights reserved.
// Third party copyrights are property of their respective owners.

#include "rvv_hal.hpp"
#include "common.hpp"

namespace cv { namespace rvv_hal { namespace core {

#if CV_HAL_RVV_1P0_ENABLED

namespace {

template <typename T, typename ST>
struct NormInf_RVV {
    inline ST operator() (const T* src, int n) const {
        ST s = 0;
        for (int i = 0; i < n; i++) {
            s = std::max(s, (ST)cv_abs(src[i]));
        }
        return s;
    }
};

template <typename T, typename ST>
struct NormL1_RVV {
    inline ST operator() (const T* src, int n) const {
        ST s = 0;
        for (int i = 0; i < n; i++) {
            s += cv_abs(src[i]);
        }
        return s;
    }
};

template <typename T, typename ST>
struct NormL2_RVV {
    inline ST operator() (const T* src, int n) const {
        ST s = 0;
        for (int i = 0; i < n; i++) {
            ST v = src[i];
            s += v * v;
        }
        return s;
    }
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
            s = __riscv_vmaxu_tu(s, s, common::__riscv_vabs(v, vl), vl);
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
            s = __riscv_vmaxu_tu(s, s, common::__riscv_vabs(v, vl), vl);
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
            s = __riscv_vmaxu_tu(s, s, common::__riscv_vabs(v, vl), vl);
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
            s = __riscv_vwredsumu(__riscv_vwredsumu_tu(zero, v, zero, vl), s, __riscv_vsetvlmax_e16m1());
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
            auto v = common::__riscv_vabs(__riscv_vle8_v_i8m8(src + i, vl), vl);
            s = __riscv_vwredsumu(__riscv_vwredsumu_tu(zero, v, zero, vl), s, __riscv_vsetvlmax_e16m1());
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
            auto v = common::__riscv_vabs(__riscv_vle16_v_i16m8(src + i, vl), vl);
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
            auto v = common::__riscv_vabs(__riscv_vle32_v_i32m4(src + i, vl), vl);
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
        auto s = __riscv_vmv_v_x_u32m1(0, __riscv_vsetvlmax_e32m1());
        if (cn == 1) {
            int vl;
            for (int i = 0; i < len; i += vl) {
                vl = __riscv_vsetvl_e8m4(len - i);
                auto v = __riscv_vle8_v_u8m4(src + i, vl);
                auto m = __riscv_vle8_v_u8m4(mask + i, vl);
                auto b = __riscv_vmsne(m, 0, vl);
                s = __riscv_vwredsumu(b, __riscv_vwmulu(b, v, v, vl), s, vl);
            }
        } else if (cn == 4) {
            int vl;
            for (int i = 0; i < len; i += vl) {
                vl = __riscv_vsetvl_e8m1(len - i);
                auto v = __riscv_vle8_v_u8m4(src + i * 4, vl * 4);
                auto m = __riscv_vle8_v_u8m1(mask + i, vl);
                auto b = __riscv_vmsne(__riscv_vreinterpret_u8m4(__riscv_vmul(__riscv_vzext_vf4(__riscv_vminu(m, 1, vl), vl), 0x01010101, vl)), 0, vl * 4);
                s = __riscv_vwredsumu(b, __riscv_vwmulu(b, v, v, vl * 4), s, vl * 4);
            }
        } else {
            for (int cn_index = 0; cn_index < cn; cn_index++) {
                int vl;
                for (int i = 0; i < len; i += vl) {
                    vl = __riscv_vsetvl_e8m4(len - i);
                    auto v = __riscv_vlse8_v_u8m4(src + cn * i + cn_index, sizeof(uchar) * cn, vl);
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
                s = __riscv_vmaxu_tumu(b, s, s, common::__riscv_vabs(v, vl), vl);
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
                auto v = common::__riscv_vabs(__riscv_vlse8_v_i8m8(src + cn * i + cn_index, sizeof(schar) * cn, vl), vl);
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
        auto s = __riscv_vmv_v_x_i32m1(0, __riscv_vsetvlmax_e32m1());
        for (int cn_index = 0; cn_index < cn; cn_index++) {
            int vl;
            for (int i = 0; i < len; i += vl) {
                vl = __riscv_vsetvl_e8m4(len - i);
                auto v = __riscv_vlse8_v_i8m4(src + cn * i + cn_index, sizeof(schar) * cn, vl);
                auto m = __riscv_vle8_v_u8m4(mask + i, vl);
                auto b = __riscv_vmsne(m, 0, vl);
                s = __riscv_vwredsum(b, __riscv_vwmul(b, v, v, vl), s, vl);
            }
        }
        return __riscv_vmv_x(s);
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
                s = __riscv_vmaxu_tumu(b, s, s, common::__riscv_vabs(v, vl), vl);
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
                auto v = common::__riscv_vabs(__riscv_vlse16_v_i16m8(src + cn * i + cn_index, sizeof(short) * cn, vl), vl);
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
                s = __riscv_vmaxu_tumu(b, s, s, common::__riscv_vabs(v, vl), vl);
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
                s = __riscv_vfadd_tumu(b, s, s, __riscv_vfwcvt_f(b, common::__riscv_vabs(v, vl), vl), vl);
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
int norm(const uchar* src, size_t src_step, const uchar* mask, size_t mask_step,
         int width, int height, int type, int norm_type, double* result) {
    int depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);

    if (result == nullptr || depth == CV_16F || norm_type > NORM_L2SQR) {
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    // [FIXME] append 0's when merging to 5.x
    static NormFunc norm_tab[3][CV_DEPTH_MAX] = {
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

    static const size_t elem_size_tab[CV_DEPTH_MAX] = {
        sizeof(uchar),   sizeof(schar),
        sizeof(ushort),  sizeof(short),
        sizeof(int),     sizeof(float),
        sizeof(int64_t), 0,
    };
    CV_Assert(elem_size_tab[depth]);

    bool src_continuous = (src_step == width * elem_size_tab[depth] * cn || (src_step != width * elem_size_tab[depth] * cn && height == 1));
    bool mask_continuous = (mask_step == static_cast<size_t>(width));
    size_t nplanes = 1;
    size_t size = width * height;
    if ((mask && (!src_continuous || !mask_continuous)) || !src_continuous) {
        nplanes = height;
        size = width;
    }

    NormFunc func = norm_tab[norm_type >> 1][depth];
    if (func == nullptr) {
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    // Handle overflow
    union {
        double d;
        int i;
        float f;
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
        auto _src = src;
        auto _mask = mask;
        for (size_t i = 0; i < nplanes; i++) {
            if ((mask && (!src_continuous || !mask_continuous)) || !src_continuous) {
                _src = src + src_step * i;
                _mask = mask + mask_step * i;
            }
            for (int j = 0; j < total; j += blockSize) {
                int bsz = std::min(total - j, blockSize);
                func(_src, _mask, (uchar*)&isum, bsz, cn);
                count += bsz;
                if (count + blockSize >= intSumBlockSize || (i + 1 >= nplanes && j + bsz >= total)) {
                    res.d += isum;
                    isum = 0;
                    count = 0;
                }
                _src += bsz * esz;
                if (mask) {
                    _mask += bsz;
                }
            }
        }
    } else {
        auto _src = src;
        auto _mask = mask;
        for (size_t i = 0; i < nplanes; i++) {
            if ((mask && (!src_continuous || !mask_continuous)) || !src_continuous) {
                _src = src + src_step * i;
                _mask = mask + mask_step * i;
            }
            func(_src, _mask, (uchar*)&res, (int)size, cn);
        }
    }

    if (norm_type == NORM_INF) {
        if (depth == CV_64F) {
            *result = res.d;
        } else if (depth == CV_32F) {
            *result = res.f;
        } else {
            *result = res.i;
        }
    } else if (norm_type == NORM_L2) {
        *result = std::sqrt(res.d);
    } else {
        *result = res.d;
    }

    return CV_HAL_ERROR_OK;
}

#endif // CV_HAL_RVV_1P0_ENABLED

}}} // cv::rvv_hal::core
