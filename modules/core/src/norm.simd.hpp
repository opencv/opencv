// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "precomp.hpp"

#if CV_RVV
#include "norm.rvv1p0.hpp"
#endif

namespace cv {

using NormFunc = int (*)(const uchar*, const uchar*, uchar*, int, int);

CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

NormFunc getNormFunc(int normType, int depth);

#ifndef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

template <typename T, typename ST>
struct NormInf_SIMD {
    inline ST operator() (const T* src, int n) const {
        ST s = 0;
        for (int i = 0; i < n; i++) {
            s = std::max(s, (ST)cv_abs(src[i]));
        }
        return s;
    }
};

template <typename T, typename ST>
struct NormL1_SIMD {
    inline ST operator() (const T* src, int n) const {
        ST s = 0;
        for (int i = 0; i < n; i++) {
            s += cv_abs(src[i]);
        }
        return s;
    }
};

template <typename T, typename ST>
struct NormL2_SIMD {
    inline ST operator() (const T* src, int n) const {
        ST s = 0;
        for (int i = 0; i < n; i++) {
            ST v = src[i];
            s += v * v;
        }
        return s;
    }
};

#if (CV_SIMD || CV_SIMD_SCALABLE)

template<>
struct NormInf_SIMD<uchar, int> {
    int operator() (const uchar* src, int n) const {
        int j = 0;
        int s = 0;
        v_uint8 r0 = vx_setzero_u8(), r1 = vx_setzero_u8();
        v_uint8 r2 = vx_setzero_u8(), r3 = vx_setzero_u8();
        for (; j <= n - 4 * VTraits<v_uint8>::vlanes(); j += 4 * VTraits<v_uint8>::vlanes()) {
            r0 = v_max(r0, vx_load(src + j                                 ));
            r1 = v_max(r1, vx_load(src + j +     VTraits<v_uint8>::vlanes()));
            r2 = v_max(r2, vx_load(src + j + 2 * VTraits<v_uint8>::vlanes()));
            r3 = v_max(r3, vx_load(src + j + 3 * VTraits<v_uint8>::vlanes()));
        }
        r0 = v_max(r0, v_max(r1, v_max(r2, r3)));
        for (; j < n; j++) {
            s = std::max(s, (int)src[j]);
        }
        return std::max(s, (int)v_reduce_max(r0));
    }
};

template<>
struct NormInf_SIMD<schar, int> {
    int operator() (const schar* src, int n) const {
        int j = 0;
        int s = 0;
        v_uint8 r0 = vx_setzero_u8(), r1 = vx_setzero_u8();
        v_uint8 r2 = vx_setzero_u8(), r3 = vx_setzero_u8();
        for (; j <= n - 4 * VTraits<v_int8>::vlanes(); j += 4 * VTraits<v_int8>::vlanes()) {
            r0 = v_max(r0, v_abs(vx_load(src + j                                )));
            r1 = v_max(r1, v_abs(vx_load(src + j +     VTraits<v_int8>::vlanes())));
            r2 = v_max(r2, v_abs(vx_load(src + j + 2 * VTraits<v_int8>::vlanes())));
            r3 = v_max(r3, v_abs(vx_load(src + j + 3 * VTraits<v_int8>::vlanes())));
        }
        r0 = v_max(r0, v_max(r1, v_max(r2, r3)));
        for (; j < n; j++) {
            s = std::max(s, cv_abs(src[j]));
        }
        return std::max(s, saturate_cast<int>(v_reduce_max(r0)));
    }
};

template<>
struct NormInf_SIMD<ushort, int> {
    int operator() (const ushort* src, int n) const {
        int j = 0;
        int s = 0;
        v_uint16 d0 = vx_setzero_u16(), d1 = vx_setzero_u16();
        v_uint16 d2 = vx_setzero_u16(), d3 = vx_setzero_u16();
        for (; j <= n - 4 * VTraits<v_uint16>::vlanes(); j += 4 * VTraits<v_uint16>::vlanes()) {
            d0 = v_max(d0, vx_load(src + j                                  ));
            d1 = v_max(d1, vx_load(src + j +     VTraits<v_uint16>::vlanes()));
            d2 = v_max(d2, vx_load(src + j + 2 * VTraits<v_uint16>::vlanes()));
            d3 = v_max(d3, vx_load(src + j + 3 * VTraits<v_uint16>::vlanes()));
        }
        d0 = v_max(d0, v_max(d1, v_max(d2, d3)));
        for (; j < n; j++) {
            s = std::max(s, (int)src[j]);
        }
        return std::max(s, (int)v_reduce_max(d0));
    }
};

template<>
struct NormInf_SIMD<short, int> {
    int operator() (const short* src, int n) const {
        int j = 0;
        int s = 0;
        v_uint16 d0 = vx_setzero_u16(), d1 = vx_setzero_u16();
        v_uint16 d2 = vx_setzero_u16(), d3 = vx_setzero_u16();
        for (; j <= n - 4 * VTraits<v_int16>::vlanes(); j += 4 * VTraits<v_int16>::vlanes()) {
            d0 = v_max(d0, v_abs(vx_load(src + j                                  )));
            d1 = v_max(d1, v_abs(vx_load(src + j +     VTraits<v_int16>::vlanes())));
            d2 = v_max(d2, v_abs(vx_load(src + j + 2 * VTraits<v_int16>::vlanes())));
            d3 = v_max(d3, v_abs(vx_load(src + j + 3 * VTraits<v_int16>::vlanes())));
        }
        d0 = v_max(d0, v_max(d1, v_max(d2, d3)));
        for (; j < n; j++) {
            s = std::max(s, saturate_cast<int>(cv_abs(src[j])));
        }
        return std::max(s, saturate_cast<int>(v_reduce_max(d0)));
    }
};

template<>
struct NormInf_SIMD<int, int> {
    int operator() (const int* src, int n) const {
        int j = 0;
        int s = 0;
        v_uint32 r0 = vx_setzero_u32(), r1 = vx_setzero_u32();
        v_uint32 r2 = vx_setzero_u32(), r3 = vx_setzero_u32();
        for (; j <= n - 4 * VTraits<v_int32>::vlanes(); j += 4 * VTraits<v_int32>::vlanes()) {
            r0 = v_max(r0, v_abs(vx_load(src + j                                 )));
            r1 = v_max(r1, v_abs(vx_load(src + j +     VTraits<v_int32>::vlanes())));
            r2 = v_max(r2, v_abs(vx_load(src + j + 2 * VTraits<v_int32>::vlanes())));
            r3 = v_max(r3, v_abs(vx_load(src + j + 3 * VTraits<v_int32>::vlanes())));
        }
        r0 = v_max(r0, v_max(r1, v_max(r2, r3)));
        for (; j < n; j++) {
            s = std::max(s, cv_abs(src[j]));
        }
        return std::max(s, saturate_cast<int>(v_reduce_max(r0)));
    }
};

template<>
struct NormInf_SIMD<float, float> {
    float operator() (const float* src, int n) const {
        int j = 0;
        float s = 0.f;
        v_float32 r0 = vx_setzero_f32(), r1 = vx_setzero_f32();
        v_float32 r2 = vx_setzero_f32(), r3 = vx_setzero_f32();
        for (; j <= n - 4 * VTraits<v_float32>::vlanes(); j += 4 * VTraits<v_float32>::vlanes()) {
            r0 = v_max(r0, v_abs(vx_load(src + j                                   )));
            r1 = v_max(r1, v_abs(vx_load(src + j +     VTraits<v_float32>::vlanes())));
            r2 = v_max(r2, v_abs(vx_load(src + j + 2 * VTraits<v_float32>::vlanes())));
            r3 = v_max(r3, v_abs(vx_load(src + j + 3 * VTraits<v_float32>::vlanes())));
        }
        r0 = v_max(r0, v_max(r1, v_max(r2, r3)));
        for (; j < n; j++) {
            s = std::max(s, cv_abs(src[j]));
        }
        return std::max(s, v_reduce_max(r0));
    }
};

template<>
struct NormL1_SIMD<uchar, int> {
    int operator() (const uchar* src, int n) const {
        int j = 0;
        int s = 0;
        v_uint32 r0 = vx_setzero_u32(), r1 = vx_setzero_u32();
        v_uint8  one = vx_setall_u8(1);
        for (; j<= n - 2 * VTraits<v_uint8>::vlanes(); j += 2 * VTraits<v_uint8>::vlanes()) {
            v_uint8 v0 = vx_load(src + j);
            r0 = v_dotprod_expand_fast(v0, one, r0);

            v_uint8 v1 = vx_load(src + j + VTraits<v_uint8>::vlanes());
            r1 = v_dotprod_expand_fast(v1, one, r1);
        }
        s += v_reduce_sum(v_add(r0, r1));
        for (; j < n; j++) {
            s += src[j];
        }
        return s;
    }
};

template<>
struct NormL1_SIMD<schar, int> {
    int operator() (const schar* src, int n) const {
        int j = 0;
        int s = 0;
        v_uint32 r0 = vx_setzero_u32(), r1 = vx_setzero_u32();
        v_uint8  one = vx_setall_u8(1);
        for (; j<= n - 2 * VTraits<v_int8>::vlanes(); j += 2 * VTraits<v_int8>::vlanes()) {
            v_uint8 v0 = v_abs(vx_load(src + j));
            r0 = v_dotprod_expand_fast(v0, one, r0);

            v_uint8 v1 = v_abs(vx_load(src + j + VTraits<v_int8>::vlanes()));
            r1 = v_dotprod_expand_fast(v1, one, r1);
        }
        s += v_reduce_sum(v_add(r0, r1));
        for (; j < n; j++) {
            s += saturate_cast<int>(cv_abs(src[j]));
        }
        return s;
    }
};

template<>
struct NormL1_SIMD<ushort, int> {
    int operator() (const ushort* src, int n) const {
        int j = 0;
        int s = 0;
        v_uint64 r0 = vx_setzero_u64(), r1 = vx_setzero_u64();
        v_uint16 one = vx_setall_u16(1);
        for (; j<= n - 2 * VTraits<v_uint16>::vlanes(); j += 2 * VTraits<v_uint16>::vlanes()) {
            v_uint16 v0 = vx_load(src + j);
            r0 = v_dotprod_expand_fast(v0, one, r0);

            v_uint16 v1 = vx_load(src + j + VTraits<v_uint16>::vlanes());
            r1 = v_dotprod_expand_fast(v1, one, r1);
        }
        s += v_reduce_sum(v_add(r0, r1));
        for (; j < n; j++) {
            s += src[j];
        }
        return s;
    }
};

template<>
struct NormL1_SIMD<short, int> {
    int operator() (const short* src, int n) const {
        int j = 0;
        int s = 0;
        v_uint64 r0 = vx_setzero_u64(), r1 = vx_setzero_u64();
        v_uint16 one = vx_setall_u16(1);
        for (; j<= n - 2 * VTraits<v_int16>::vlanes(); j += 2 * VTraits<v_int16>::vlanes()) {
            v_uint16 v0 = v_abs(vx_load(src + j));
            r0 = v_dotprod_expand_fast(v0, one, r0);

            v_uint16 v1 = v_abs(vx_load(src + j + VTraits<v_int16>::vlanes()));
            r1 = v_dotprod_expand_fast(v1, one, r1);
        }
        s += v_reduce_sum(v_add(r0, r1));
        for (; j < n; j++) {
            s += saturate_cast<int>(cv_abs(src[j]));
        }
        return s;
    }
};

template<>
struct NormL2_SIMD<uchar, int> {
    int operator() (const uchar* src, int n) const {
        int j = 0;
        int s = 0;
        v_uint32 r0 = vx_setzero_u32(), r1 = vx_setzero_u32();
        for (; j <= n - 2 * VTraits<v_uint8>::vlanes(); j += 2 * VTraits<v_uint8>::vlanes()) {
            v_uint8 v0 = vx_load(src + j);
            r0 = v_dotprod_expand_fast(v0, v0, r0);

            v_uint8 v1 = vx_load(src + j + VTraits<v_uint8>::vlanes());
            r1 = v_dotprod_expand_fast(v1, v1, r1);
        }
        s += v_reduce_sum(v_add(r0, r1));
        for (; j < n; j++) {
            int v = saturate_cast<int>(src[j]);
            s += v * v;
        }
        return s;
    }
};

template<>
struct NormL2_SIMD<schar, int> {
    int operator() (const schar* src, int n) const {
        int j = 0;
        int s = 0;
        v_int32 r0 = vx_setzero_s32(), r1 = vx_setzero_s32();
        for (; j <= n - 2 * VTraits<v_int8>::vlanes(); j += 2 * VTraits<v_int8>::vlanes()) {
            v_int8 v0 = vx_load(src + j);
            r0 = v_dotprod_expand_fast(v0, v0, r0);
            v_int8 v1 = vx_load(src + j + VTraits<v_int8>::vlanes());
            r1 = v_dotprod_expand_fast(v1, v1, r1);
        }
        s += v_reduce_sum(v_add(r0, r1));
        for (; j < n; j++) {
            int v = saturate_cast<int>(src[j]);
            s += v * v;
        }
        return s;
    }
};

#endif

#if (CV_SIMD_64F || CV_SIMD_SCALABLE_64F)

template<>
struct NormInf_SIMD<double, double> {
    double operator() (const double* src, int n) const {
        int j = 0;
        double s = 0.f;
        v_float64 r0 = vx_setzero_f64(), r1 = vx_setzero_f64();
        v_float64 r2 = vx_setzero_f64(), r3 = vx_setzero_f64();
        for (; j <= n - 4 * VTraits<v_float64>::vlanes(); j += 4 * VTraits<v_float64>::vlanes()) {
            r0 = v_max(r0, v_abs(vx_load(src + j                                   )));
            r1 = v_max(r1, v_abs(vx_load(src + j +     VTraits<v_float64>::vlanes())));
            r2 = v_max(r2, v_abs(vx_load(src + j + 2 * VTraits<v_float64>::vlanes())));
            r3 = v_max(r3, v_abs(vx_load(src + j + 3 * VTraits<v_float64>::vlanes())));
        }
        r0 = v_max(r0, v_max(r1, v_max(r2, r3)));
        for (; j < n; j++) {
            s = std::max(s, cv_abs(src[j]));
        }
        // [TODO]: use v_reduce_max when it supports float64
        double t[VTraits<v_float64>::max_nlanes];
        vx_store(t, r0);
        for (int i = 0; i < VTraits<v_float64>::vlanes(); i++) {
            s = std::max(s, cv_abs(t[i]));
        }
        return s;
    }
};

template<>
struct NormL1_SIMD<int, double> {
    double operator() (const int* src, int n) const {
        int j = 0;
        double s = 0.f;
        v_float64 r00 = vx_setzero_f64(), r01 = vx_setzero_f64();
        v_float64 r10 = vx_setzero_f64(), r11 = vx_setzero_f64();
        for (; j <= n - 2 * VTraits<v_int32>::vlanes(); j += 2 * VTraits<v_int32>::vlanes()) {
            v_float32 v0 = v_abs(v_cvt_f32(vx_load(src + j))), v1 = v_abs(v_cvt_f32(vx_load(src + j + VTraits<v_int32>::vlanes())));
            r00 = v_add(r00, v_cvt_f64(v0)); r01 = v_add(r01, v_cvt_f64_high(v0));
            r10 = v_add(r10, v_cvt_f64(v1)); r11 = v_add(r11, v_cvt_f64_high(v1));
        }
        s += v_reduce_sum(v_add(v_add(v_add(r00, r01), r10), r11));
        for (; j < n; j++) {
            s += cv_abs(src[j]);
        }
        return s;
    }
};

template<>
struct NormL1_SIMD<float, double> {
    double operator() (const float* src, int n) const {
        int j = 0;
        double s = 0.f;
        v_float64 r00 = vx_setzero_f64(), r01 = vx_setzero_f64();
        v_float64 r10 = vx_setzero_f64(), r11 = vx_setzero_f64();
        v_float64 r20 = vx_setzero_f64(), r21 = vx_setzero_f64();
        v_float64 r30 = vx_setzero_f64(), r31 = vx_setzero_f64();
        for (; j <= n - 4 * VTraits<v_float32>::vlanes(); j += 4 * VTraits<v_float32>::vlanes()) {
            v_float32 v0 = v_abs(vx_load(src + j)), v1 = v_abs(vx_load(src + j + VTraits<v_float32>::vlanes()));
            r00 = v_add(r00, v_cvt_f64(v0)); r01 = v_add(r01, v_cvt_f64_high(v0));
            r10 = v_add(r10, v_cvt_f64(v1)); r11 = v_add(r11, v_cvt_f64_high(v1));

            v_float32 v2 = v_abs(vx_load(src + j + 2 * VTraits<v_float32>::vlanes())), v3 = v_abs(vx_load(src + j + 3 * VTraits<v_float32>::vlanes()));
            r20 = v_add(r20, v_cvt_f64(v2)); r21 = v_add(r21, v_cvt_f64_high(v2));
            r30 = v_add(r30, v_cvt_f64(v3)); r31 = v_add(r31, v_cvt_f64_high(v3));
        }
        s += v_reduce_sum(v_add(v_add(v_add(r00, r01), r10), r11));
        s += v_reduce_sum(v_add(v_add(v_add(r20, r21), r30), r31));
        for (; j < n; j++) {
            s += cv_abs(src[j]);
        }
        return s;
    }
};

template<>
struct NormL1_SIMD<double, double> {
    double operator() (const double* src, int n) const {
        int j = 0;
        double s = 0.f;
#if CV_RVV // This is introduced to workaround the accuracy issue on ci
        s = normL1_rvv<double, double>(src, n, j);
#else
        v_float64 r00 = vx_setzero_f64(), r01 = vx_setzero_f64();
        v_float64 r10 = vx_setzero_f64(), r11 = vx_setzero_f64();
        for (; j <= n - 4 * VTraits<v_float64>::vlanes(); j += 4 * VTraits<v_float64>::vlanes()) {
            r00 = v_add(r00, v_abs(vx_load(src + j                                   )));
            r01 = v_add(r01, v_abs(vx_load(src + j +     VTraits<v_float64>::vlanes())));
            r10 = v_add(r10, v_abs(vx_load(src + j + 2 * VTraits<v_float64>::vlanes())));
            r11 = v_add(r11, v_abs(vx_load(src + j + 3 * VTraits<v_float64>::vlanes())));
        }
        s += v_reduce_sum(v_add(v_add(v_add(r00, r01), r10), r11));
#endif
        for (; j < n; j++) {
            s += cv_abs(src[j]);
        }
        return s;
    }
};

template<>
struct NormL2_SIMD<ushort, double> {
    double operator() (const ushort* src, int n) const {
        int j = 0;
        double s = 0.f;
        v_float64 r0 = vx_setzero_f64(), r1 = vx_setzero_f64();
        for (; j <= n - 2 * VTraits<v_uint16>::vlanes(); j += 2 * VTraits<v_uint16>::vlanes()) {
            v_uint16 v0 = vx_load(src + j);
            v_uint64 u0 = v_dotprod_expand_fast(v0, v0);
            r0 = v_add(r0, v_cvt_f64(v_reinterpret_as_s64(u0)));

            v_uint16 v1 = vx_load(src + j + VTraits<v_uint16>::vlanes());
            v_uint64 u1 = v_dotprod_expand_fast(v1, v1);
            r1 = v_add(r1, v_cvt_f64(v_reinterpret_as_s64(u1)));
        }
        s += v_reduce_sum(v_add(r0, r1));
        for (; j < n; j++) {
            double v = saturate_cast<double>(src[j]);
            s += v * v;
        }
        return s;
    }
};

template<>
struct NormL2_SIMD<short, double> {
    double operator() (const short* src, int n) const {
        int j = 0;
        double s = 0.f;
        v_float64 r0 = vx_setzero_f64(), r1 = vx_setzero_f64();
        for (; j <= n - 2 * VTraits<v_int16>::vlanes(); j += 2 * VTraits<v_int16>::vlanes()) {
            v_int16 v0 = vx_load(src + j);
            r0 = v_add(r0, v_cvt_f64(v_dotprod_expand_fast(v0, v0)));

            v_int16 v1 = vx_load(src + j + VTraits<v_int16>::vlanes());
            r1 = v_add(r1, v_cvt_f64(v_dotprod_expand_fast(v1, v1)));
        }
        s += v_reduce_sum(v_add(r0, r1));
        for (; j < n; j++) {
            double v = saturate_cast<double>(src[j]);
            s += v * v;
        }
        return s;
    }
};

template<>
struct NormL2_SIMD<int, double> {
    double operator() (const int* src, int n) const {
        int j = 0;
        double s = 0.f;
        v_float64 r0 = vx_setzero_f64(), r1 = vx_setzero_f64();
        for (; j <= n - 2 * VTraits<v_int32>::vlanes(); j += 2 * VTraits<v_int32>::vlanes()) {
            v_int32 v0 = vx_load(src + j);
            r0 = v_dotprod_expand_fast(v0, v0, r0);

            v_int32 v1 = vx_load(src + j + VTraits<v_int32>::vlanes());
            r1 = v_dotprod_expand_fast(v1, v1, r1);
        }
        s += v_reduce_sum(v_add(r0, r1));
        for (; j < n; j++) {
            double v = src[j];
            s += v * v;
        }
        return s;
    }
};

template<>
struct NormL2_SIMD<float, double> {
    double operator() (const float* src, int n) const {
        int j = 0;
        double s = 0.f;
        v_float64 r00 = vx_setzero_f64(), r01 = vx_setzero_f64();
        v_float64 r10 = vx_setzero_f64(), r11 = vx_setzero_f64();
        for (; j <= n - 2 * VTraits<v_float32>::vlanes(); j += 2 * VTraits<v_float32>::vlanes()) {
            v_float32 v0 = vx_load(src + j), v1 = vx_load(src + j + VTraits<v_float32>::vlanes());
            v_float64 v00 = v_cvt_f64(v0), v01 = v_cvt_f64_high(v0);
            v_float64 v10 = v_cvt_f64(v1), v11 = v_cvt_f64_high(v1);
            r00 = v_fma(v00, v00, r00); r01 = v_fma(v01, v01, r01);
            r10 = v_fma(v10, v10, r10); r11 = v_fma(v11, v11, r11);
        }
        s += v_reduce_sum(v_add(v_add(v_add(r00, r01), r10), r11));
        for (; j < n; j++) {
            double v = src[j];
            s += v * v;
        }
        return s;
    }
};

template<>
struct NormL2_SIMD<double, double> {
    double operator() (const double* src, int n) const {
        int j = 0;
        double s = 0.f;
#if CV_RVV // This is introduced to workaround the accuracy issue on ci
        s = normL2_rvv<double, double>(src, n, j);
#else
        v_float64 r00 = vx_setzero_f64(), r01 = vx_setzero_f64();
        v_float64 r10 = vx_setzero_f64(), r11 = vx_setzero_f64();
        for (; j <= n - 4 * VTraits<v_float64>::vlanes(); j += 4 * VTraits<v_float64>::vlanes()) {
            v_float64 v00 = vx_load(src + j                                   );
            v_float64 v01 = vx_load(src + j +     VTraits<v_float64>::vlanes());
            v_float64 v10 = vx_load(src + j + 2 * VTraits<v_float64>::vlanes());
            v_float64 v11 = vx_load(src + j + 3 * VTraits<v_float64>::vlanes());
            r00 = v_fma(v00, v00, r00); r01 = v_fma(v01, v01, r01);
            r10 = v_fma(v10, v10, r10); r11 = v_fma(v11, v11, r11);
        }
        s += v_reduce_sum(v_add(v_add(v_add(r00, r01), r10), r11));
#endif
        for (; j < n; j++) {
            double v = src[j];
            s += v * v;
        }
        return s;
    }
};

#endif

template<typename T, typename ST> int
normInf_(const T* src, const uchar* mask, ST* _result, int len, int cn) {
    ST result = *_result;
    if( !mask ) {
        NormInf_SIMD<T, ST> op;
        result = std::max(result, op(src, len*cn));
    } else {
        for( int i = 0; i < len; i++, src += cn ) {
            if( mask[i] ) {
                for( int k = 0; k < cn; k++ ) {
                    result = std::max(result, ST(cv_abs(src[k])));
                }
            }
        }
    }
    *_result = result;
    return 0;
}

template<typename T, typename ST> int
normL1_(const T* src, const uchar* mask, ST* _result, int len, int cn) {
    ST result = *_result;
    if( !mask ) {
        NormL1_SIMD<T, ST> op;
        result += op(src, len*cn);
    } else {
        for( int i = 0; i < len; i++, src += cn ) {
            if( mask[i] ) {
                for( int k = 0; k < cn; k++ ) {
                    result += cv_abs(src[k]);
                }
            }
        }
    }
    *_result = result;
    return 0;
}

template<typename T, typename ST> int
normL2_(const T* src, const uchar* mask, ST* _result, int len, int cn) {
    ST result = *_result;
    if( !mask ) {
        NormL2_SIMD<T, ST> op;
        result += op(src, len*cn);
    } else {
        for( int i = 0; i < len; i++, src += cn ) {
            if( mask[i] ) {
                for( int k = 0; k < cn; k++ ) {
                    T v = src[k];
                    result += (ST)v*v;
                }
            }
        }
    }
    *_result = result;
    return 0;
}

#define CV_DEF_NORM_FUNC(L, suffix, type, ntype) \
    static int norm##L##_##suffix(const type* src, const uchar* mask, ntype* r, int len, int cn) \
{ CV_INSTRUMENT_REGION(); return norm##L##_(src, mask, r, len, cn); } \

#define CV_DEF_NORM_ALL(suffix, type, inftype, l1type, l2type) \
    CV_DEF_NORM_FUNC(Inf, suffix, type, inftype) \
    CV_DEF_NORM_FUNC(L1, suffix, type, l1type) \
    CV_DEF_NORM_FUNC(L2, suffix, type, l2type)

CV_DEF_NORM_ALL(8u, uchar, int, int, int)
CV_DEF_NORM_ALL(8s, schar, int, int, int)
CV_DEF_NORM_ALL(16u, ushort, int, int, double)
CV_DEF_NORM_ALL(16s, short, int, int, double)
CV_DEF_NORM_ALL(32s, int, int, double, double)
CV_DEF_NORM_ALL(32f, float, float, double, double)
CV_DEF_NORM_ALL(64f, double, double, double, double)

NormFunc getNormFunc(int normType, int depth)
{
    CV_INSTRUMENT_REGION();
    static NormFunc normTab[3][8] =
    {
        {
            (NormFunc)GET_OPTIMIZED(normInf_8u), (NormFunc)GET_OPTIMIZED(normInf_8s), (NormFunc)GET_OPTIMIZED(normInf_16u), (NormFunc)GET_OPTIMIZED(normInf_16s),
            (NormFunc)GET_OPTIMIZED(normInf_32s), (NormFunc)GET_OPTIMIZED(normInf_32f), (NormFunc)normInf_64f, 0
        },
        {
            (NormFunc)GET_OPTIMIZED(normL1_8u), (NormFunc)GET_OPTIMIZED(normL1_8s), (NormFunc)GET_OPTIMIZED(normL1_16u), (NormFunc)GET_OPTIMIZED(normL1_16s),
            (NormFunc)GET_OPTIMIZED(normL1_32s), (NormFunc)GET_OPTIMIZED(normL1_32f), (NormFunc)normL1_64f, 0
        },
        {
            (NormFunc)GET_OPTIMIZED(normL2_8u), (NormFunc)GET_OPTIMIZED(normL2_8s), (NormFunc)GET_OPTIMIZED(normL2_16u), (NormFunc)GET_OPTIMIZED(normL2_16s),
            (NormFunc)GET_OPTIMIZED(normL2_32s), (NormFunc)GET_OPTIMIZED(normL2_32f), (NormFunc)normL2_64f, 0
        }
    };

    return normTab[normType][depth];
}

#endif // CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

CV_CPU_OPTIMIZATION_NAMESPACE_END

} // cv::
