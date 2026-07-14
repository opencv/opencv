// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
//
// Copyright (C) 2025, SpaceMIT Inc., all rights reserved.
// Copyright (C) 2026, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.

#include "precomp.hpp"

namespace cv {

using NormFunc = int (*)(const uchar*, const uchar*, uchar*, int, int);
using NormDiffFunc = int (*)(const uchar*, const uchar*, const uchar*, uchar*, int, int);

CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

NormFunc getNormFunc(int normType, int depth);
NormDiffFunc getNormDiffFunc(int normType, int depth);

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

template <typename T, typename ST>
struct NormDiffInf_SIMD {
    inline ST operator() (const T* src1, const T* src2, int n) const {
        ST s = 0;
        for (int i = 0; i < n; i++) {
            s = std::max(s, (ST)std::abs(src1[i] - src2[i]));
        }
        return s;
    }
};

template <typename T, typename ST>
struct NormDiffL1_SIMD {
    inline ST operator() (const T* src1, const T* src2, int n) const {
        ST s = 0;
        for (int i = 0; i < n; i++) {
            s += std::abs(src1[i] - src2[i]);
        }
        return s;
    }
};

template <typename T, typename ST>
struct NormDiffL2_SIMD {
    inline ST operator() (const T* src1, const T* src2, int n) const {
        ST s = 0;
        for (int i = 0; i < n; i++) {
            ST v = (ST)src1[i] - (ST)src2[i];
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
        s = std::max(s, saturate_cast<int>(v_reduce_max(r0)));
        for (; j < n; j++) {
            s = std::max(s, cv_abs(src[j]));
        }
        return s;
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
        v_uint32 r00 = vx_setzero_u32(), r01 = vx_setzero_u32();
        v_uint32 r10 = vx_setzero_u32(), r11 = vx_setzero_u32();
        for (; j<= n - 2 * VTraits<v_uint16>::vlanes(); j += 2 * VTraits<v_uint16>::vlanes()) {
            v_uint16 v0 = vx_load(src + j);
            v_uint32 v00, v01;
            v_expand(v0, v00, v01);
            r00 = v_add(r00, v00);
            r01 = v_add(r01, v01);

            v_uint16 v1 = vx_load(src + j + VTraits<v_uint16>::vlanes());
            v_uint32 v10, v11;
            v_expand(v1, v10, v11);
            r10 = v_add(r10, v10);
            r11 = v_add(r11, v11);
        }
        s += (int)v_reduce_sum(v_add(v_add(v_add(r00, r01), r10), r11));
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
        v_uint32 r00 = vx_setzero_u32(), r01 = vx_setzero_u32();
        v_uint32 r10 = vx_setzero_u32(), r11 = vx_setzero_u32();
        for (; j<= n - 2 * VTraits<v_int16>::vlanes(); j += 2 * VTraits<v_int16>::vlanes()) {
            v_uint16 v0 = v_abs(vx_load(src + j));
            v_uint32 v00, v01;
            v_expand(v0, v00, v01);
            r00 = v_add(r00, v00);
            r01 = v_add(r01, v01);

            v_uint16 v1 = v_abs(vx_load(src + j + VTraits<v_int16>::vlanes()));
            v_uint32 v10, v11;
            v_expand(v1, v10, v11);
            r10 = v_add(r10, v10);
            r11 = v_add(r11, v11);
        }
        s += (int)v_reduce_sum(v_add(v_add(v_add(r00, r01), r10), r11));
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

template<>
struct NormDiffInf_SIMD<uchar, int> {
    int operator() (const uchar* src1, const uchar* src2, int n) const {
        int j = 0;
        int s = 0;
        v_uint8 r0 = vx_setzero_u8(), r1 = vx_setzero_u8();
        v_uint8 r2 = vx_setzero_u8(), r3 = vx_setzero_u8();
        for (; j <= n - 4 * VTraits<v_uint8>::vlanes(); j += 4 * VTraits<v_uint8>::vlanes()) {
            v_uint8 v01 = vx_load(src1 + j), v02 = vx_load(src2 + j);
            r0 = v_max(r0, v_absdiff(v01, v02));

            v_uint8 v11 = vx_load(src1 + j + VTraits<v_uint8>::vlanes()),
                    v12 = vx_load(src2 + j + VTraits<v_uint8>::vlanes());
            r1 = v_max(r1, v_absdiff(v11, v12));

            v_uint8 v21 = vx_load(src1 + j + 2 * VTraits<v_uint8>::vlanes()),
                    v22 = vx_load(src2 + j + 2 * VTraits<v_uint8>::vlanes());
            r2 = v_max(r2, v_absdiff(v21, v22));

            v_uint8 v31 = vx_load(src1 + j + 3 * VTraits<v_uint8>::vlanes()),
                    v32 = vx_load(src2 + j + 3 * VTraits<v_uint8>::vlanes());
            r3 = v_max(r3, v_absdiff(v31, v32));
        }
        s = (int)v_reduce_max(v_max(v_max(v_max(r0, r1), r2), r3));
        for (; j < n; j++) {
            s = std::max(s, (int)std::abs(src1[j] - src2[j]));
        }
        return s;
    }
};

template<>
struct NormDiffInf_SIMD<schar, int> {
    int operator() (const schar* src1, const schar* src2, int n) const {
        int j = 0;
        int s = 0;
        v_uint8 r0 = vx_setzero_u8(), r1 = vx_setzero_u8();
        v_uint8 r2 = vx_setzero_u8(), r3 = vx_setzero_u8();
        for (; j <= n - 4 * VTraits<v_int8>::vlanes(); j += 4 * VTraits<v_int8>::vlanes()) {
            v_int8 v01 = vx_load(src1 + j), v02 = vx_load(src2 + j);
            r0 = v_max(r0, v_absdiff(v01, v02));

            v_int8 v11 = vx_load(src1 + j + VTraits<v_int8>::vlanes()),
                   v12 = vx_load(src2 + j + VTraits<v_int8>::vlanes());
            r1 = v_max(r1, v_absdiff(v11, v12));

            v_int8 v21 = vx_load(src1 + j + 2 * VTraits<v_int8>::vlanes()),
                   v22 = vx_load(src2 + j + 2 * VTraits<v_int8>::vlanes());
            r2 = v_max(r2, v_absdiff(v21, v22));

            v_int8 v31 = vx_load(src1 + j + 3 * VTraits<v_int8>::vlanes()),
                   v32 = vx_load(src2 + j + 3 * VTraits<v_int8>::vlanes());
            r3 = v_max(r3, v_absdiff(v31, v32));
        }
        s = (int)v_reduce_max(v_max(v_max(v_max(r0, r1), r2), r3));
        for (; j < n; j++) {
            s = std::max(s, (int)std::abs(src1[j] - src2[j]));
        }
        return s;
    }
};

template<>
struct NormDiffInf_SIMD<ushort, int> {
    int operator() (const ushort* src1, const ushort* src2, int n) const {
        int j = 0;
        int s = 0;
        v_uint16 r0 = vx_setzero_u16(), r1 = vx_setzero_u16();
        v_uint16 r2 = vx_setzero_u16(), r3 = vx_setzero_u16();
        for (; j <= n - 4 * VTraits<v_uint16>::vlanes(); j += 4 * VTraits<v_uint16>::vlanes()) {
            v_uint16 v01 = vx_load(src1 + j), v02 = vx_load(src2 + j);
            r0 = v_max(r0, v_absdiff(v01, v02));

            v_uint16 v11 = vx_load(src1 + j + VTraits<v_uint16>::vlanes()),
                     v12 = vx_load(src2 + j + VTraits<v_uint16>::vlanes());
            r1 = v_max(r1, v_absdiff(v11, v12));

            v_uint16 v21 = vx_load(src1 + j + 2 * VTraits<v_uint16>::vlanes()),
                     v22 = vx_load(src2 + j + 2 * VTraits<v_uint16>::vlanes());
            r2 = v_max(r2, v_absdiff(v21, v22));

            v_uint16 v31 = vx_load(src1 + j + 3 * VTraits<v_uint16>::vlanes()),
                     v32 = vx_load(src2 + j + 3 * VTraits<v_uint16>::vlanes());
            r3 = v_max(r3, v_absdiff(v31, v32));
        }
        s = (int)v_reduce_max(v_max(v_max(v_max(r0, r1), r2), r3));
        for (; j < n; j++) {
            s = std::max(s, (int)std::abs(src1[j] - src2[j]));
        }
        return s;
    }
};

template<>
struct NormDiffInf_SIMD<short, int> {
    int operator() (const short* src1, const short* src2, int n) const {
        int j = 0;
        int s = 0;
        v_uint16 r0 = vx_setzero_u16(), r1 = vx_setzero_u16();
        v_uint16 r2 = vx_setzero_u16(), r3 = vx_setzero_u16();
        for (; j <= n - 4 * VTraits<v_int16>::vlanes(); j += 4 * VTraits<v_int16>::vlanes()) {
            v_int16 v01 = vx_load(src1 + j), v02 = vx_load(src2 + j);
            r0 = v_max(r0, v_absdiff(v01, v02));

            v_int16 v11 = vx_load(src1 + j + VTraits<v_int16>::vlanes()),
                    v12 = vx_load(src2 + j + VTraits<v_int16>::vlanes());
            r1 = v_max(r1, v_absdiff(v11, v12));

            v_int16 v21 = vx_load(src1 + j + 2 * VTraits<v_int16>::vlanes()),
                    v22 = vx_load(src2 + j + 2 * VTraits<v_int16>::vlanes());
            r2 = v_max(r2, v_absdiff(v21, v22));

            v_int16 v31 = vx_load(src1 + j + 3 * VTraits<v_int16>::vlanes()),
                    v32 = vx_load(src2 + j + 3 * VTraits<v_int16>::vlanes());
            r3 = v_max(r3, v_absdiff(v31, v32));
        }
        s = (int)v_reduce_max(v_max(v_max(v_max(r0, r1), r2), r3));
        for (; j < n; j++) {
            s = std::max(s, (int)std::abs(src1[j] - src2[j]));
        }
        return s;
    }
};

template<>
struct NormDiffInf_SIMD<int, int> {
    int operator() (const int* src1, const int* src2, int n) const {
        int j = 0;
        int s = 0;
        v_uint32 r0 = vx_setzero_u32(), r1 = vx_setzero_u32();
        v_uint32 r2 = vx_setzero_u32(), r3 = vx_setzero_u32();
        for (; j <= n - 4 * VTraits<v_int32>::vlanes(); j += 4 * VTraits<v_int32>::vlanes()) {
            v_int32 v01 = vx_load(src1 + j), v02 = vx_load(src2 + j);
            r0 = v_max(r0, v_abs(v_sub(v01, v02)));

            v_int32 v11 = vx_load(src1 + j + VTraits<v_int32>::vlanes()),
                    v12 = vx_load(src2 + j + VTraits<v_int32>::vlanes());
            r1 = v_max(r1, v_abs(v_sub(v11, v12)));

            v_int32 v21 = vx_load(src1 + j + 2 * VTraits<v_int32>::vlanes()),
                    v22 = vx_load(src2 + j + 2 * VTraits<v_int32>::vlanes());
            r2 = v_max(r2, v_abs(v_sub(v21, v22)));

            v_int32 v31 = vx_load(src1 + j + 3 * VTraits<v_int32>::vlanes()),
                    v32 = vx_load(src2 + j + 3 * VTraits<v_int32>::vlanes());
            r3 = v_max(r3, v_abs(v_sub(v31, v32)));
        }
        s = (int)v_reduce_max(v_max(v_max(v_max(r0, r1), r2), r3));
        for (; j < n; j++) {
            s = std::max(s, (int)std::abs(src1[j] - src2[j]));
        }
        return s;
    }
};

template<>
struct NormDiffInf_SIMD<float, float> {
    float operator() (const float* src1, const float* src2, int n) const {
        int j = 0;
        float s = 0;
        v_float32 r0 = vx_setzero_f32(), r1 = vx_setzero_f32();
        for (; j <= n - 2 * VTraits<v_float32>::vlanes(); j += 2 * VTraits<v_float32>::vlanes()) {
            v_float32 v01 = vx_load(src1 + j), v02 = vx_load(src2 + j);
            r0 = v_max(r0, v_absdiff(v01, v02));

            v_float32 v11 = vx_load(src1 + j + VTraits<v_float32>::vlanes()),
                      v12 = vx_load(src2 + j + VTraits<v_float32>::vlanes());
            r1 = v_max(r1, v_absdiff(v11, v12));
        }
        s = v_reduce_max(v_max(r0, r1));
        for (; j < n; j++) {
            s = std::max(s, (float)std::abs(src1[j] - src2[j]));
        }
        return s;
    }
};

template<>
struct NormDiffL1_SIMD<uchar, int> {
    int operator() (const uchar* src1, const uchar* src2, int n) const {
        int j = 0;
        int s = 0;
        v_uint32 r0 = vx_setzero_u32(), r1 = vx_setzero_u32();
        v_uint8  one = vx_setall_u8(1);
        for (; j<= n - 2 * VTraits<v_uint8>::vlanes(); j += 2 * VTraits<v_uint8>::vlanes()) {
            v_uint8 v01 = vx_load(src1 + j), v02 = vx_load(src2 + j);
            r0 = v_dotprod_expand_fast(v_absdiff(v01, v02), one, r0);

            v_uint8 v11 = vx_load(src1 + j + VTraits<v_uint8>::vlanes()),
                    v12 = vx_load(src2 + j + VTraits<v_uint8>::vlanes());
            r1 = v_dotprod_expand_fast(v_absdiff(v11, v12), one, r1);
        }
        s += v_reduce_sum(v_add(r0, r1));
        for (; j < n; j++) {
            s += std::abs(src1[j] - src2[j]);
        }
        return s;
    }
};

template<>
struct NormDiffL1_SIMD<schar, int> {
    int operator() (const schar* src1, const schar* src2, int n) const {
        int j = 0;
        int s = 0;
        v_uint32 r0 = vx_setzero_u32(), r1 = vx_setzero_u32();
        v_uint8  one = vx_setall_u8(1);
        for (; j<= n - 2 * VTraits<v_int8>::vlanes(); j += 2 * VTraits<v_int8>::vlanes()) {
            v_int8 v01 = vx_load(src1 + j), v02 = vx_load(src2 + j);
            r0 = v_dotprod_expand_fast(v_absdiff(v01, v02), one, r0);

            v_int8 v11 = vx_load(src1 + j + VTraits<v_int8>::vlanes()),
                   v12 = vx_load(src2 + j + VTraits<v_int8>::vlanes());
            r1 = v_dotprod_expand_fast(v_absdiff(v11, v12), one, r1);
        }
        s += v_reduce_sum(v_add(r0, r1));
        for (; j < n; j++) {
            s += std::abs(src1[j] - src2[j]);
        }
        return s;
    }
};

template<>
struct NormDiffL1_SIMD<ushort, int> {
    int operator() (const ushort* src1, const ushort* src2, int n) const {
        int j = 0;
        int s = 0;
        v_uint32 r0 = vx_setzero_u32(), r1 = vx_setzero_u32();
        v_uint32 r2 = vx_setzero_u32(), r3 = vx_setzero_u32();
        for (; j<= n - 4 * VTraits<v_uint16>::vlanes(); j += 4 * VTraits<v_uint16>::vlanes()) {
            v_uint16 v01 = vx_load(src1 + j), v02 = vx_load(src2 + j);
            v_uint32 u00, u01;
            v_expand(v_absdiff(v01, v02), u00, u01);
            r0 = v_add(r0, v_add(u00, u01));

            v_uint16 v11 = vx_load(src1 + j + VTraits<v_uint16>::vlanes()),
                     v12 = vx_load(src2 + j + VTraits<v_uint16>::vlanes());
            v_uint32 u10, u11;
            v_expand(v_absdiff(v11, v12), u10, u11);
            r1 = v_add(r1, v_add(u10, u11));

            v_uint16 v21 = vx_load(src1 + j + 2 * VTraits<v_uint16>::vlanes()),
                     v22 = vx_load(src2 + j + 2 * VTraits<v_uint16>::vlanes());
            v_uint32 u20, u21;
            v_expand(v_absdiff(v21, v22), u20, u21);
            r2 = v_add(r2, v_add(u20, u21));

            v_uint16 v31 = vx_load(src1 + j + 3 * VTraits<v_uint16>::vlanes()),
                     v32 = vx_load(src2 + j + 3 * VTraits<v_uint16>::vlanes());
            v_uint32 u30, u31;
            v_expand(v_absdiff(v31, v32), u30, u31);
            r3 = v_add(r3, v_add(u30, u31));
        }
        s += (int)v_reduce_sum(v_add(v_add(v_add(r0, r1), r2), r3));
        for (; j < n; j++) {
            s += std::abs(src1[j] - src2[j]);
        }
        return s;
    }
};

template<>
struct NormDiffL1_SIMD<short, int> {
    int operator() (const short* src1, const short* src2, int n) const {
        int j = 0;
        int s = 0;
        v_uint32 r0 = vx_setzero_u32(), r1 = vx_setzero_u32();
        v_uint32 r2 = vx_setzero_u32(), r3 = vx_setzero_u32();
        for (; j<= n - 4 * VTraits<v_int16>::vlanes(); j += 4 * VTraits<v_int16>::vlanes()) {
            v_int16 v01 = vx_load(src1 + j), v02 = vx_load(src2 + j);
            v_uint32 u00, u01;
            v_expand(v_absdiff(v01, v02), u00, u01);
            r0 = v_add(r0, v_add(u00, u01));

            v_int16 v11 = vx_load(src1 + j + VTraits<v_int16>::vlanes()),
                    v12 = vx_load(src2 + j + VTraits<v_int16>::vlanes());
            v_uint32 u10, u11;
            v_expand(v_absdiff(v11, v12), u10, u11);
            r1 = v_add(r1, v_add(u10, u11));

            v_int16 v21 = vx_load(src1 + j + 2 * VTraits<v_int16>::vlanes()),
                    v22 = vx_load(src2 + j + 2 * VTraits<v_int16>::vlanes());
            v_uint32 u20, u21;
            v_expand(v_absdiff(v21, v22), u20, u21);
            r2 = v_add(r2, v_add(u20, u21));

            v_int16 v31 = vx_load(src1 + j + 3 * VTraits<v_int16>::vlanes()),
                    v32 = vx_load(src2 + j + 3 * VTraits<v_int16>::vlanes());
            v_uint32 u30, u31;
            v_expand(v_absdiff(v31, v32), u30, u31);
            r3 = v_add(r3, v_add(u30, u31));
        }
        s += (int)v_reduce_sum(v_add(v_add(v_add(r0, r1), r2), r3));
        for (; j < n; j++) {
            s += std::abs(src1[j] - src2[j]);
        }
        return s;
    }
};

template<>
struct NormDiffL2_SIMD<uchar, int> {
    int operator() (const uchar* src1, const uchar* src2, int n) const {
        int j = 0;
        int s = 0;
        v_uint32 r0 = vx_setzero_u32(), r1 = vx_setzero_u32();
        for (; j <= n - 2 * VTraits<v_uint8>::vlanes(); j += 2 * VTraits<v_uint8>::vlanes()) {
            v_uint8 v01 = vx_load(src1 + j), v02 = vx_load(src2 + j);
            v_uint8 v0 = v_absdiff(v01, v02);
            r0 = v_dotprod_expand_fast(v0, v0, r0);

            v_uint8 v11 = vx_load(src1 + j + VTraits<v_uint8>::vlanes()),
                    v12 = vx_load(src2 + j + VTraits<v_uint8>::vlanes());
            v_uint8 v1 = v_absdiff(v11, v12);
            r1 = v_dotprod_expand_fast(v1, v1, r1);
        }
        s += v_reduce_sum(v_add(r0, r1));
        for (; j < n; j++) {
            int v = (int)src1[j] - (int)src2[j];
            s += v * v;
        }
        return s;
    }
};

template<>
struct NormDiffL2_SIMD<schar, int> {
    int operator() (const schar* src1, const schar* src2, int n) const {
        int j = 0;
        int s = 0;
        v_uint32 r0 = vx_setzero_u32(), r1 = vx_setzero_u32();
        for (; j <= n - 2 * VTraits<v_int8>::vlanes(); j += 2 * VTraits<v_int8>::vlanes()) {
            v_int8 v01 = vx_load(src1 + j), v02 = vx_load(src2 + j);
            v_uint8 v0 = v_absdiff(v01, v02);
            r0 = v_dotprod_expand_fast(v0, v0, r0);

            v_int8 v11 = vx_load(src1 + j + VTraits<v_int8>::vlanes()),
                   v12 = vx_load(src2 + j + VTraits<v_int8>::vlanes());
            v_uint8 v1 = v_absdiff(v11, v12);
            r1 = v_dotprod_expand_fast(v1, v1, r1);
        }
        s += v_reduce_sum(v_add(r0, r1));
        for (; j < n; j++) {
            int v = (int)src1[j] - (int)src2[j];
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

#endif

#if CV_SIMD_64F // CV_SIMD_SCALABLE_64F has accuracy problem with the following kernels on ci

template<>
struct NormL1_SIMD<double, double> {
    double operator() (const double* src, int n) const {
        int j = 0;
        double s = 0.f;
        v_float64 r00 = vx_setzero_f64(), r01 = vx_setzero_f64();
        v_float64 r10 = vx_setzero_f64(), r11 = vx_setzero_f64();
        for (; j <= n - 4 * VTraits<v_float64>::vlanes(); j += 4 * VTraits<v_float64>::vlanes()) {
            r00 = v_add(r00, v_abs(vx_load(src + j                                   )));
            r01 = v_add(r01, v_abs(vx_load(src + j +     VTraits<v_float64>::vlanes())));
            r10 = v_add(r10, v_abs(vx_load(src + j + 2 * VTraits<v_float64>::vlanes())));
            r11 = v_add(r11, v_abs(vx_load(src + j + 3 * VTraits<v_float64>::vlanes())));
        }
        s += v_reduce_sum(v_add(v_add(v_add(r00, r01), r10), r11));
        for (; j < n; j++) {
            s += cv_abs(src[j]);
        }
        return s;
    }
};

template<>
struct NormL2_SIMD<double, double> {
    double operator() (const double* src, int n) const {
        int j = 0;
        double s = 0.f;
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
        for (; j < n; j++) {
            double v = src[j];
            s += v * v;
        }
        return s;
    }
};

template<>
struct NormDiffInf_SIMD<double, double> {
    double operator() (const double* src1, const double* src2, int n) const {
        int j = 0;
        double s = 0;
        v_float64 r0 = vx_setzero_f64(), r1 = vx_setzero_f64();
        for (; j <= n - 2 * VTraits<v_float64>::vlanes(); j += 2 * VTraits<v_float64>::vlanes()) {
            v_float64 v01 = vx_load(src1 + j), v02 = vx_load(src2 + j);
            r0 = v_max(r0, v_absdiff(v01, v02));

            v_float64 v11 = vx_load(src1 + j + VTraits<v_float64>::vlanes()),
                      v12 = vx_load(src2 + j + VTraits<v_float64>::vlanes());
            r1 = v_max(r1, v_absdiff(v11, v12));
        }
        // [TODO]: use v_reduce_max when it supports float64
        double t[VTraits<v_float64>::max_nlanes];
        vx_store(t, v_max(r0, r1));
        for (int i = 0; i < VTraits<v_float64>::vlanes(); i++) {
            s = std::max(s, std::abs(t[i]));
        }
        for (; j < n; j++) {
            s = std::max(s, (double)std::abs(src1[j] - src2[j]));
        }
        return s;
    }
};

template<>
struct NormDiffL1_SIMD<int, double> {
    double operator() (const int* src1, const int* src2, int n) const {
        int j = 0;
        double s = 0.f;
        v_float64 r0 = vx_setzero_f64(), r1 = vx_setzero_f64();
        for (; j <= n - VTraits<v_int32>::vlanes(); j += VTraits<v_int32>::vlanes()) {
            v_int32 v01 = vx_load(src1 + j), v02 = vx_load(src2 + j);
            v_uint32 v0 = v_abs(v_sub(v01, v02));
            v_uint64 ev0, ev1;
            v_expand(v0, ev0, ev1);
            r0 = v_add(r0, v_cvt_f64(v_reinterpret_as_s64(ev0)));
            r1 = v_add(r1, v_cvt_f64(v_reinterpret_as_s64(ev1)));
        }
        s += v_reduce_sum(v_add(r0, r1));
        for (; j < n; j++) {
            s += std::abs(src1[j] - src2[j]);
        }
        return s;
    }
};

template<>
struct NormDiffL1_SIMD<float, double> {
    double operator() (const float* src1, const float* src2, int n) const {
        int j = 0;
        double s = 0.f;
        v_float64 r0 = vx_setzero_f64(), r1 = vx_setzero_f64();
        v_float64 r2 = vx_setzero_f64(), r3 = vx_setzero_f64();
        for (; j <= n - 2 * VTraits<v_float32>::vlanes(); j += 2 * VTraits<v_float32>::vlanes()) {
            v_float32 v01 = vx_load(src1 + j), v02 = vx_load(src2 + j);
            v_float32 v0 = v_absdiff(v01, v02);
            r0 = v_add(r0, v_cvt_f64(v0)); r1 = v_add(r1, v_cvt_f64_high(v0));

            v_float32 v11 = vx_load(src1 + j + VTraits<v_float32>::vlanes()),
                      v12 = vx_load(src2 + j + VTraits<v_float32>::vlanes());
            v_float32 v1 = v_absdiff(v11, v12);
            r2 = v_add(r2, v_cvt_f64(v1)); r3 = v_add(r3, v_cvt_f64_high(v1));
        }
        s += v_reduce_sum(v_add(v_add(v_add(r0, r1), r2), r3));
        for (; j < n; j++) {
            s += std::abs(src1[j] - src2[j]);
        }
        return s;
    }
};

template<>
struct NormDiffL1_SIMD<double, double> {
    double operator() (const double* src1, const double* src2, int n) const {
        int j = 0;
        double s = 0.f;
        v_float64 r0 = vx_setzero_f64(), r1 = vx_setzero_f64();
        for (; j <= n - 2 * VTraits<v_float64>::vlanes(); j += 2 * VTraits<v_float64>::vlanes()) {
            v_float64 v01 = vx_load(src1 + j), v02 = vx_load(src2 + j);
            r0 = v_add(r0, v_absdiff(v01, v02));

            v_float64 v11 = vx_load(src1 + j + VTraits<v_float64>::vlanes()),
                      v12 = vx_load(src2 + j + VTraits<v_float64>::vlanes());
            r1 = v_add(r1, v_absdiff(v11, v12));
        }
        s += v_reduce_sum(v_add(r0, r1));
        for (; j < n; j++) {
            s += std::abs(src1[j] - src2[j]);
        }
        return s;
    }
};

template<>
struct NormDiffL2_SIMD<ushort, double> {
    double operator() (const ushort* src1, const ushort* src2, int n) const {
        int j = 0;
        double s = 0.f;
        v_float64 r0 = vx_setzero_f64(), r1 = vx_setzero_f64();
        for (; j <= n - 2 * VTraits<v_uint16>::vlanes(); j += 2 * VTraits<v_uint16>::vlanes()) {
            v_uint16 v01 = vx_load(src1 + j), v02 = vx_load(src2 + j);
            v_uint16 v0 = v_absdiff(v01, v02);
            v_uint64 u0 = v_dotprod_expand_fast(v0, v0);
            r0 = v_add(r0, v_cvt_f64(v_reinterpret_as_s64(u0)));

            v_uint16 v11 = vx_load(src1 + j + VTraits<v_uint16>::vlanes()),
                     v12 = vx_load(src2 + j + VTraits<v_uint16>::vlanes());
            v_uint16 v1 = v_absdiff(v11, v12);
            v_uint64 u1 = v_dotprod_expand_fast(v1, v1);
            r1 = v_add(r1, v_cvt_f64(v_reinterpret_as_s64(u1)));
        }
        s += v_reduce_sum(v_add(r0, r1));
        for (; j < n; j++) {
            double v = (double)src1[j] - (double)src2[j];
            s += v * v;
        }
        return s;
    }
};

template<>
struct NormDiffL2_SIMD<short, double> {
    double operator() (const short* src1, const short* src2, int n) const {
        int j = 0;
        double s = 0.f;
        v_float64 r0 = vx_setzero_f64(), r1 = vx_setzero_f64();
        for (; j <= n - 2 * VTraits<v_int16>::vlanes(); j += 2 * VTraits<v_int16>::vlanes()) {
            v_int16 v01 = vx_load(src1 + j), v02 = vx_load(src2 + j);
            v_uint16 v0 = v_absdiff(v01, v02);
            v_uint64 u0 = v_dotprod_expand_fast(v0, v0);
            r0 = v_add(r0, v_cvt_f64(v_reinterpret_as_s64(u0)));

            v_int16 v11 = vx_load(src1 + j + VTraits<v_uint16>::vlanes()),
                    v12 = vx_load(src2 + j + VTraits<v_uint16>::vlanes());
            v_uint16 v1 = v_absdiff(v11, v12);
            v_uint64 u1 = v_dotprod_expand_fast(v1, v1);
            r1 = v_add(r1, v_cvt_f64(v_reinterpret_as_s64(u1)));
        }
        s += v_reduce_sum(v_add(r0, r1));
        for (; j < n; j++) {
            double v = (double)src1[j] - (double)src2[j];
            s += v * v;
        }
        return s;
    }
};

template<>
struct NormDiffL2_SIMD<int, double> {
    double operator() (const int* src1, const int* src2, int n) const {
        int j = 0;
        double s = 0.f;
        v_float64 r0 = vx_setzero_f64(), r1 = vx_setzero_f64();
        for (; j <= n - VTraits<v_int32>::vlanes(); j += VTraits<v_int32>::vlanes()) {
            v_int32 v01 = vx_load(src1 + j), v02 = vx_load(src2 + j);
            v_uint32 v0 = v_absdiff(v01, v02);
            v_uint64 ev0, ev1;
            v_expand(v0, ev0, ev1);
            v_float64 f0 = v_cvt_f64(v_reinterpret_as_s64(ev0)), f1 = v_cvt_f64(v_reinterpret_as_s64(ev1));
            r0 = v_fma(f0, f0, r0); r1 = v_fma(f1, f1, r1);
        }
        s += v_reduce_sum(v_add(r0, r1));
        for (; j < n; j++) {
            double v = (double)src1[j] - (double)src2[j];
            s += v * v;
        }
        return s;
    }
};

template<>
struct NormDiffL2_SIMD<float, double> {
    double operator() (const float* src1, const float* src2, int n) const {
        int j = 0;
        double s = 0.f;
        v_float64 r0 = vx_setzero_f64(), r1 = vx_setzero_f64();
        v_float64 r2 = vx_setzero_f64(), r3 = vx_setzero_f64();
        for (; j <= n - 2 * VTraits<v_float32>::vlanes(); j += 2 * VTraits<v_float32>::vlanes()) {
            v_float32 v01 = vx_load(src1 + j), v02 = vx_load(src2 + j);
            v_float32 v0 = v_absdiff(v01, v02);
            v_float64 f01 = v_cvt_f64(v0), f02 = v_cvt_f64_high(v0);
            r0 = v_fma(f01, f01, r0); r1 = v_fma(f02, f02, r1);

            v_float32 v11 = vx_load(src1 + j + VTraits<v_float32>::vlanes()),
                      v12 = vx_load(src2 + j + VTraits<v_float32>::vlanes());
            v_float32 v1 = v_absdiff(v11, v12);
            v_float64 f11 = v_cvt_f64(v1), f12 = v_cvt_f64_high(v1);
            r2 = v_fma(f11, f11, r2); r3 = v_fma(f12, f12, r3);
        }
        s += v_reduce_sum(v_add(v_add(v_add(r0, r1), r2), r3));
        for (; j < n; j++) {
            double v = (double)src1[j] - (double)src2[j];
            s += v * v;
        }
        return s;
    }
};

template<>
struct NormDiffL2_SIMD<double, double> {
    double operator() (const double* src1, const double* src2, int n) const {
        int j = 0;
        double s = 0.f;
        v_float64 r0 = vx_setzero_f64(), r1 = vx_setzero_f64();
        v_float64 r2 = vx_setzero_f64(), r3 = vx_setzero_f64();
        for (; j <= n - 4 * VTraits<v_float64>::vlanes(); j += 4 * VTraits<v_float64>::vlanes()) {
            v_float64 v01 = vx_load(src1 + j), v02 = vx_load(src2 + j);
            v_float64 v0 = v_absdiff(v01, v02);
            r0 = v_fma(v0, v0, r0);

            v_float64 v11 = vx_load(src1 + j + VTraits<v_float64>::vlanes()),
                      v12 = vx_load(src2 + j + VTraits<v_float64>::vlanes());
            v_float64 v1 = v_absdiff(v11, v12);
            r1 = v_fma(v1, v1, r1);

            v_float64 v21 = vx_load(src1 + j + 2 * VTraits<v_float64>::vlanes()),
                      v22 = vx_load(src2 + j + 2 * VTraits<v_float64>::vlanes());
            v_float64 v2 = v_absdiff(v21, v22);
            r2 = v_fma(v2, v2, r2);

            v_float64 v31 = vx_load(src1 + j + 3 * VTraits<v_float64>::vlanes()),
                      v32 = vx_load(src2 + j + 3 * VTraits<v_float64>::vlanes());
            v_float64 v3 = v_absdiff(v31, v32);
            r3 = v_fma(v3, v3, r3);
        }
        s += v_reduce_sum(v_add(v_add(v_add(r0, r1), r2), r3));
        for (; j < n; j++) {
            double v = (double)src1[j] - (double)src2[j];
            s += v * v;
        }
        return s;
    }
};

#endif

template <typename T, typename ST>
struct MaskedNormInf_SIMD {
    inline ST operator() (const T* src, const uchar* mask, int len, int cn) const {
        ST s = 0;
        if (cn == 1) {
            for (int i = 0; i < len; i++) {
                if (mask[i]) {
                    s = std::max(s, (ST)cv_abs(src[i]));
                }
            }
        }
        else {
            for (int i = 0; i < len; i++) {
                if (mask[i]) {
                    const T* elem = src + i * cn;
                    int k = 0;
                #if CV_ENABLE_UNROLLED
                    for (; k <= cn - 4; k += 4) {
                        s = std::max(s, (ST)cv_abs(elem[k]));
                        s = std::max(s, (ST)cv_abs(elem[k + 1]));
                        s = std::max(s, (ST)cv_abs(elem[k + 2]));
                        s = std::max(s, (ST)cv_abs(elem[k + 3]));
                    }
                #endif
                    for (; k < cn; k++) {
                        s = std::max(s, (ST)cv_abs(elem[k]));
                    }
                }
            }
        }
        return s;
    }
};

template <typename T, typename ST>
struct MaskedNormL1_SIMD {
    inline ST operator() (const T* src, const uchar* mask, int len, int cn) const {
        ST s = 0;
        if (cn == 1) {
            for (int i = 0; i < len; i++) {
                if (mask[i]) {
                    s += (ST)cv_abs(src[i]);
                }
            }
        }
        else {
            for (int i = 0; i < len; i++) {
                if (mask[i]) {
                    const T* elem = src + i * cn;
                    int k = 0;
                #if CV_ENABLE_UNROLLED
                    for (; k <= cn - 4; k += 4) {
                        s += (ST)cv_abs(elem[k]);
                        s += (ST)cv_abs(elem[k + 1]);
                        s += (ST)cv_abs(elem[k + 2]);
                        s += (ST)cv_abs(elem[k + 3]);
                    }
                #endif
                    for (; k < cn; k++) {
                        s += (ST)cv_abs(elem[k]);
                    }
                }
            }
        }
        return s;
    }
};

template <typename T, typename ST>
struct MaskedNormL2_SIMD {
    inline ST operator() (const T* src, const uchar* mask, int len, int cn) const {
        ST s = 0;
        if (cn == 1) {
            int i = 0;
        #if CV_ENABLE_UNROLLED
            for (; i <= len - 4; i += 4) {
                if (mask[i])     { T v0 = src[i];     s += (ST)v0 * v0; }
                if (mask[i + 1]) { T v1 = src[i + 1]; s += (ST)v1 * v1; }
                if (mask[i + 2]) { T v2 = src[i + 2]; s += (ST)v2 * v2; }
                if (mask[i + 3]) { T v3 = src[i + 3]; s += (ST)v3 * v3; }
            }
        #endif
            for (; i < len; i++) {
                if (mask[i]) {
                    T v = src[i];
                    s += (ST)v * v;
                }
            }
        }
        else {
            for (int i = 0; i < len; i++) {
                if (mask[i]) {
                    const T* elem = src + i * cn;
                    int k = 0;
                #if CV_ENABLE_UNROLLED
                    for (; k <= cn - 4; k += 4) {
                        T v0 = elem[k];     s += (ST)v0 * v0;
                        T v1 = elem[k + 1]; s += (ST)v1 * v1;
                        T v2 = elem[k + 2]; s += (ST)v2 * v2;
                        T v3 = elem[k + 3]; s += (ST)v3 * v3;
                    }
                #endif
                    for (; k < cn; k++) {
                        T v = elem[k];
                        s += (ST)v * v;
                    }
                }
            }
        }
        return s;
    }
};

template <>
struct MaskedNormInf_SIMD<float, float> {
    inline float operator()(const float* src, const uchar* mask, int len, int cn) const {
        float result = 0.0f;
        if (cn == 1) {
            int i = 0;
            const int vstep = VTraits<v_float32>::vlanes();
            v_float32 acc = vx_setzero_f32();

            for (; i <= len - vstep; i += vstep) {
                v_uint32 m = vx_load_expand_q(mask + i);
                v_uint32 cmp = v_gt(m, vx_setzero_u32());
                v_float32 s = vx_load(src + i);
                s = v_abs(s);
                s = v_reinterpret_as_f32(v_and(v_reinterpret_as_u32(s), cmp));
                acc = v_max(acc, s);
            }
            result = v_reduce_max(acc);

            for (; i < len; i++) {
                if (mask[i])
                    result = std::max(result, std::abs(src[i]));
            }
        }
        else {
            for (int i = 0; i < len; i++) {
                if (mask[i]) {
                    const float* elem = src + i * cn;
                    int k = 0;
                    const int vstep = VTraits<v_float32>::vlanes();
                    v_float32 acc = vx_setzero_f32();

                    for (; k <= cn - vstep; k += vstep) {
                        v_float32 s = vx_load(elem + k);
                        acc = v_max(acc, v_abs(s));
                    }

                    result = std::max(result, v_reduce_max(acc));

                    for (; k < cn; k++)
                        result = std::max(result, std::abs(elem[k]));
                }
            }
        }
        return result;
    }
};

#if CV_SIMD_64F
template <>
struct MaskedNormL1_SIMD<float, double> {
    inline double operator()(const float* src, const uchar* mask, int len, int cn) const {
        double result = 0.0;
        if (cn == 1) {
            int i = 0;
            const int vstep = VTraits<v_float32>::vlanes();
            v_float64 acc = vx_setzero_f64();

            for (; i <= len - vstep; i += vstep) {
                v_uint32 cmp = v_gt(vx_load_expand_q(mask + i), vx_setzero_u32());
                v_float32 s  = v_reinterpret_as_f32(v_and(v_reinterpret_as_u32(v_abs(vx_load(src + i))), cmp));
                acc = v_add(acc, v_cvt_f64(s));
                acc = v_add(acc, v_cvt_f64_high(s));
            }
            result = v_reduce_sum(acc);

            for (; i < len; i++) {
                if (mask[i])
                    result += std::abs(src[i]);
            }
        }
        else {
            for (int i = 0; i < len; i++) {
                if (mask[i]) {
                    const float* elem = src + i * cn;
                    int k = 0;
                    const int vstep = VTraits<v_float32>::vlanes();
                    v_float64 acc = vx_setzero_f64();

                    for (; k <= cn - vstep; k += vstep) {
                        v_float32 s = v_abs(vx_load(elem + k));
                        acc = v_add(acc, v_cvt_f64(s));
                        acc = v_add(acc, v_cvt_f64_high(s));
                    }

                    result += v_reduce_sum(acc);

                    for (; k < cn; k++)
                        result += std::abs(elem[k]);
                }
            }
        }
        return result;
    }
};

template <>
struct MaskedNormL2_SIMD<float, double> {
    inline double operator()(const float* src, const uchar* mask, int len, int cn) const {
        double result = 0.0;
        if (cn == 1) {
            int i = 0;
            const int vstep = VTraits<v_float32>::vlanes();
            v_float32 facc = vx_setzero_f32();
            v_float64 dacc = vx_setzero_f64();
            int flush = 0;

            for (; i <= len - vstep; i += vstep, flush += vstep) {
                if (flush >= 64) {
                    dacc = v_add(dacc, v_cvt_f64(facc));
                    dacc = v_add(dacc, v_cvt_f64_high(facc));
                    facc = vx_setzero_f32();
                    flush = 0;
                }
                v_uint32 cmp = v_gt(vx_load_expand_q(mask + i), vx_setzero_u32());
                v_float32 s  = v_reinterpret_as_f32(v_and(v_reinterpret_as_u32(vx_load(src + i)), cmp));
                facc = v_add(facc, v_mul(s, s));
            }

            dacc = v_add(dacc, v_cvt_f64(facc));
            dacc = v_add(dacc, v_cvt_f64_high(facc));
            result = v_reduce_sum(dacc);

            for (; i < len; i++) {
                if (mask[i]) {
                    double v = src[i];
                    result += v * v;
                }
            }
        }
        else {
            for (int i = 0; i < len; i++) {
                if (mask[i]) {
                    const float* elem = src + i * cn;
                    int k = 0;
                    const int vstep = VTraits<v_float32>::vlanes();
                    v_float32 facc = vx_setzero_f32();

                    for (; k <= cn - vstep; k += vstep) {
                        v_float32 s = vx_load(elem + k);
                        facc = v_add(facc, v_mul(s, s));
                    }

                    v_float64 dacc = v_add(v_cvt_f64(facc), v_cvt_f64_high(facc));
                    result += v_reduce_sum(dacc);

                    for (; k < cn; k++) {
                        double v = elem[k];
                        result += v * v;
                    }
                }
            }
        }
        return result;
    }
};

#endif

template <>
struct MaskedNormInf_SIMD<uchar, int> {
    inline int operator()(const uchar* src, const uchar* mask, int len, int cn) const {
        int result = 0;
        if (cn == 1) {
            int i = 0;
            const int vstep = VTraits<v_uint8>::vlanes();
            v_uint8 acc = vx_setzero_u8();

            for (; i <= len - vstep; i += vstep) {
                v_uint8 m   = vx_load(mask + i);
                v_uint8 s   = vx_load(src + i);
                v_uint8 sel = v_and(s, v_gt(m, vx_setzero_u8()));
                acc = v_max(acc, sel);
            }

            result = (int)v_reduce_max(acc);

            for (; i < len; i++) {
                if (mask[i])
                    result = std::max(result, (int)src[i]);
            }
        }
        else if (cn == 4 && len >= VTraits<v_uint8>::vlanes()) {
            const int vstep = VTraits<v_uint8>::vlanes();
            v_uint8 acc = vx_setzero_u8();
            int i = 0;
            for (;;) {
                if (i > len - vstep)
                    i = len - vstep;
                v_uint8 c0, c1, c2, c3;
                v_load_deinterleave(src + i * 4, c0, c1, c2, c3);
                v_uint8 pmax = v_max(v_max(c0, c1), v_max(c2, c3));
                v_uint8 m = v_gt(vx_load(mask + i), vx_setzero_u8());
                acc = v_max(acc, v_and(pmax, m));
                if (i >= len - vstep)
                    break;
                i += vstep;
            }
            result = (int)v_reduce_max(acc);
        }
        else if (cn == 4) {
            // len < one vector: pure scalar
            for (int i = 0; i < len; i++) {
                if (mask[i]) {
                    const uchar* elem = src + i * 4;
                    result = std::max(result, (int)std::max(std::max(elem[0], elem[1]),
                                                            std::max(elem[2], elem[3])));
                }
            }
        }
        else {
            for (int i = 0; i < len; i++) {
                if (mask[i]) {
                    const uchar* elem = src + i * cn;
                    for (int k = 0; k < cn; k++)
                        result = std::max(result, (int)elem[k]);
                }
            }
        }
        return result;
    }
};

template <>
struct MaskedNormL1_SIMD<uchar, int> {
    inline int operator()(const uchar* src, const uchar* mask, int len, int cn) const {
        int result = 0;
        if (cn == 1) {
            int i = 0;
            const int vstep = VTraits<v_uint8>::vlanes() / 4;
            v_uint32 acc = vx_setzero_u32();
            for (; i <= len - vstep; i += vstep) {
                v_uint32 m   = vx_load_expand_q(mask + i);
                v_uint32 s = vx_load_expand_q(src + i);
                v_uint32 sel = v_and(s, v_gt(m, vx_setzero_u32()));
                acc = v_add(acc, sel);
            }

            result = (int)v_reduce_sum(acc);

            for (; i < len; i++) {
                if (mask[i])
                    result += src[i];
            }
        }
        else {
            const int vstep = VTraits<v_uint8>::vlanes() / 4;
            if (cn >= vstep) {
                for (int i = 0; i < len; i++) {
                    if (mask[i]) {
                        const uchar* elem = src + i * cn;
                        int k = 0;
                        v_uint32 acc = vx_setzero_u32();
                        for (; k <= cn - vstep; k += vstep) {
                            v_uint32 s = vx_load_expand_q(elem + k);
                            acc = v_add(acc, s);
                        }
                        result += (int)v_reduce_sum(acc);
                        for (; k < cn; k++)
                            result += elem[k];
                    }
                }
            }
            else {
                for (int i = 0; i < len; i++) {
                    if (mask[i]) {
                        const uchar* elem = src + i * cn;
                        for (int k = 0; k < cn; k++)
                            result += elem[k];
                    }
                }
            }
        }
        return result;
    }
};

template <>
struct MaskedNormInf_SIMD<ushort, int> {
    inline int operator()(const ushort* src, const uchar* mask, int len, int cn) const {
        int result = 0;
        if (cn == 1) {
            int i = 0;
            const int vstep = VTraits<v_uint16>::vlanes();
            v_uint16 acc = vx_setzero_u16();

            for (; i <= len - vstep; i += vstep) {
                v_uint16 m   = vx_load_expand(mask + i);
                v_uint16 cmp = v_gt(m, vx_setzero_u16());
                v_uint16 s   = vx_load(src + i);
                v_uint16 sel = v_and(s, cmp);
                acc = v_max(acc, sel);
            }

            result = (int)v_reduce_max(acc);

            for (; i < len; i++) {
                if (mask[i])
                    result = std::max(result, (int)src[i]);
            }
        }
        else {
            for (int i = 0; i < len; i++) {
                if (mask[i]) {
                    const ushort* elem = src + i * cn;
                    int k = 0;
                    const int vstep = VTraits<v_uint16>::vlanes();
                    v_uint16 acc = vx_setzero_u16();

                    for (; k <= cn - vstep; k += vstep) {
                        acc = v_max(acc, vx_load(elem + k));
                    }

                    result = std::max(result, (int)v_reduce_max(acc));

                    for (; k < cn; k++)
                        result = std::max(result, (int)elem[k]);
                }
            }
        }
        return result;
    }
};

template <>
struct MaskedNormL1_SIMD<ushort, int> {
    inline int operator()(const ushort* src, const uchar* mask, int len, int cn) const {
        int result = 0;
        if (cn == 1) {
            int i = 0;
            const int vstep = VTraits<v_uint16>::vlanes();
            v_uint32 acc32 = vx_setzero_u32();
            v_uint64 acc64 = vx_setzero_u64();
            int acc32_elems = 0;

            for (; i <= len - vstep; i += vstep, acc32_elems += vstep) {
                if (acc32_elems >= 512) {
                    v_uint64 lo64, hi64;
                    v_expand(acc32, lo64, hi64);
                    acc64 = v_add(acc64, v_add(lo64, hi64));
                    acc32 = vx_setzero_u32();
                    acc32_elems = 0;
                }
                v_uint16 m   = vx_load_expand(mask + i);
                v_uint16 cmp = v_gt(m, vx_setzero_u16());
                v_uint16 s   = v_and(vx_load(src + i), cmp);
                v_uint32 lo32, hi32;
                v_expand(s, lo32, hi32);
                acc32 = v_add(acc32, v_add(lo32, hi32));
            }

            v_uint64 lo64, hi64;
            v_expand(acc32, lo64, hi64);
            acc64 = v_add(acc64, v_add(lo64, hi64));
            result = (int)v_reduce_sum(acc64);

            for (; i < len; i++) {
                if (mask[i])
                    result += src[i];
            }
        }
        else {
            for (int i = 0; i < len; i++) {
                if (mask[i]) {
                    const ushort* elem = src + i * cn;
                    int k = 0;
                    const int vstep = VTraits<v_uint16>::vlanes();
                    v_uint32 acc = vx_setzero_u32();

                    for (; k <= cn - vstep; k += vstep) {
                        v_uint32 lo32, hi32;
                        v_expand(vx_load(elem + k), lo32, hi32);
                        acc = v_add(acc, v_add(lo32, hi32));
                    }

                    result += (int)v_reduce_sum(acc);

                    for (; k < cn; k++)
                        result += elem[k];
                }
            }
        }
        return result;
    }
};

template <>
struct MaskedNormL2_SIMD<ushort, double> {
    inline double operator()(const ushort* src, const uchar* mask, int len, int cn) const {
        double result = 0.0;
        if (cn == 1) {
            int i = 0;
            const int vstep = VTraits<v_uint16>::vlanes();
            v_uint64 acc = vx_setzero_u64();
            for (; i <= len - vstep; i += vstep) {
                v_uint16 m   = vx_load_expand(mask + i);
                v_uint16 cmp = v_gt(m, vx_setzero_u16());
                v_uint16 s   = v_and(vx_load(src + i), cmp);
                v_uint32 lo32, hi32;
                v_expand(s, lo32, hi32);
                v_uint64 lo64a, lo64b, hi64a, hi64b;
                v_expand(v_mul(lo32, lo32), lo64a, lo64b);
                v_expand(v_mul(hi32, hi32), hi64a, hi64b);
                acc = v_add(acc, v_add(v_add(lo64a, lo64b), v_add(hi64a, hi64b)));
            }
            result = (double)v_reduce_sum(acc);
            for (; i < len; i++) {
                if (mask[i]) {
                    double v = src[i];
                    result += v * v;
                }
            }
        }
        else {
            for (int i = 0; i < len; i++) {
                if (mask[i]) {
                    const ushort* elem = src + i * cn;
                    int k = 0;
                    const int vstep = VTraits<v_uint16>::vlanes();
                    v_uint64 acc = vx_setzero_u64();
                    for (; k <= cn - vstep; k += vstep) {
                        v_uint32 lo32, hi32;
                        v_expand(vx_load(elem + k), lo32, hi32);
                        v_uint64 lo64a, lo64b, hi64a, hi64b;
                        v_expand(v_mul(lo32, lo32), lo64a, lo64b);
                        v_expand(v_mul(hi32, hi32), hi64a, hi64b);
                        acc = v_add(acc, v_add(v_add(lo64a, lo64b), v_add(hi64a, hi64b)));
                    }

                    result += (double)v_reduce_sum(acc);

                    for (; k < cn; k++) {
                        double v = elem[k];
                        result += v * v;
                    }
                }
            }
        }
        return result;
    }
};

template<typename T, typename ST> int
normInf_(const T* src, const uchar* mask, ST* _result, int len, int cn) {
    ST result = *_result;
    if( !mask ) {
        NormInf_SIMD<T, ST> op;
        result = std::max(result, op(src, len*cn));
    } else {
        MaskedNormInf_SIMD<T, ST> op;
        result = std::max(result, op(src, mask, len, cn));
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
        MaskedNormL1_SIMD<T, ST> op;
        result += op(src, mask, len, cn);
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
        MaskedNormL2_SIMD<T, ST> op;
        result += op(src, mask, len, cn);
    }
    *_result = result;
    return 0;
}

#if (CV_SIMD || CV_SIMD_SCALABLE)
inline v_uint8  v_diffmask(const v_uint8&  d, const uchar* m) { return v_and(d, v_gt(vx_load(m),          vx_setzero_u8())); }
inline v_uint16 v_diffmask(const v_uint16& d, const uchar* m) { return v_and(d, v_gt(vx_load_expand(m),   vx_setzero_u16())); }
inline v_uint32 v_diffmask(const v_uint32& d, const uchar* m) { return v_and(d, v_gt(vx_load_expand_q(m), vx_setzero_u32())); }
inline v_float32 v_diffmask(const v_float32& d, const uchar* m) {
    v_uint32 cm = v_gt(vx_load_expand_q(m), vx_setzero_u32());
    return v_reinterpret_as_f32(v_and(v_reinterpret_as_u32(d), cm));
}
#endif

template <typename T, typename ST>
struct MaskedNormDiffInf_SIMD {
    inline ST operator()(const T* s1, const T* s2, const uchar* mask, int len, int cn) const {
        ST result = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
        const int vstep = VTraits<decltype(vx_load(s1))>::vlanes();
        if (cn == 1 && len >= vstep) {
            auto acc = v_diffmask(v_absdiff(vx_load(s1), vx_load(s2)), mask);
            int i = vstep;
            for (; i <= len - vstep; i += vstep)
                acc = v_max(acc, v_diffmask(v_absdiff(vx_load(s1 + i), vx_load(s2 + i)), mask + i));
            result = (ST)v_reduce_max(acc);
            for (; i < len; i++) if (mask[i]) result = std::max(result, (ST)std::abs((double)s1[i] - (double)s2[i]));
            return result;
        }
#endif
        for (int i = 0; i < len; i++) if (mask[i]) {
            const T* e1 = s1 + i*cn; const T* e2 = s2 + i*cn;
            for (int k = 0; k < cn; k++) result = std::max(result, (ST)std::abs((double)e1[k] - (double)e2[k]));
        }
        return result;
    }
};
template <typename T, typename ST>
struct MaskedNormDiffL1_SIMD {
    inline ST operator()(const T* s1, const T* s2, const uchar* mask, int len, int cn) const {
        ST s = 0;
        for (int i = 0; i < len; i++) if (mask[i]) {
            const T* e1 = s1 + i*cn; const T* e2 = s2 + i*cn;
            for (int k = 0; k < cn; k++) s += std::abs(e1[k] - e2[k]);
        }
        return s;
    }
};
template <typename T, typename ST>
struct MaskedNormDiffL2_SIMD {
    inline ST operator()(const T* s1, const T* s2, const uchar* mask, int len, int cn) const {
        ST s = 0;
        for (int i = 0; i < len; i++) if (mask[i]) {
            const T* e1 = s1 + i*cn; const T* e2 = s2 + i*cn;
            for (int k = 0; k < cn; k++) { ST v = (ST)e1[k] - (ST)e2[k]; s += v*v; }
        }
        return s;
    }
};

#if (CV_SIMD || CV_SIMD_SCALABLE)
// Shared scalar cn>1 fallback for masked norm-diff (RT = accumulator type).
// Used by every SIMD specialization's multi-channel branch.
template<typename T, typename RT>
static inline RT maskedNormDiffL1Tail(const T* s1, const T* s2, const uchar* mask, int len, int cn, RT result) {
    for (int i = 0; i < len; i++) if (mask[i]) {
        const T* e1 = s1 + i*cn; const T* e2 = s2 + i*cn;
        for (int k = 0; k < cn; k++) result += (RT)std::abs((RT)e1[k] - (RT)e2[k]);
    }
    return result;
}
template<typename T, typename RT>
static inline RT maskedNormDiffL2Tail(const T* s1, const T* s2, const uchar* mask, int len, int cn, RT result) {
    for (int i = 0; i < len; i++) if (mask[i]) {
        const T* e1 = s1 + i*cn; const T* e2 = s2 + i*cn;
        for (int k = 0; k < cn; k++) { RT v = (RT)e1[k] - (RT)e2[k]; result += v*v; }
    }
    return result;
}

// Shared 8-bit cn==1 masked kernels (uchar/schar): v_absdiff() yields v_uint8
// for both, so a single typename-T body serves both depths.
template<typename T>
static inline int maskedNormDiffL1_8(const T* s1, const T* s2, const uchar* mask, int len) {
    int i = 0; const int vstep = VTraits<v_uint8>::vlanes();
    const v_uint8 one = vx_setall_u8(1);
    v_uint32 acc = vx_setzero_u32();
    for (; i <= len - vstep; i += vstep) {
        v_uint8 m  = v_gt(vx_load(mask + i), vx_setzero_u8());
        v_uint8 ad = v_and(v_absdiff(vx_load(s1 + i), vx_load(s2 + i)), m);
        acc = v_dotprod_expand_fast(ad, one, acc); // sum of masked |diff|
    }
    int result = (int)v_reduce_sum(acc);
    for (; i < len; i++) if (mask[i]) result += std::abs((int)s1[i] - (int)s2[i]);
    return result;
}
template<typename T>
static inline int maskedNormDiffL2_8(const T* s1, const T* s2, const uchar* mask, int len) {
    int i = 0; const int vstep = VTraits<v_uint8>::vlanes();
    v_uint32 acc = vx_setzero_u32();
    for (; i <= len - vstep; i += vstep) {
        v_uint8 m  = v_gt(vx_load(mask + i), vx_setzero_u8());
        v_uint8 ad = v_and(v_absdiff(vx_load(s1 + i), vx_load(s2 + i)), m);
        acc = v_dotprod_expand_fast(ad, ad, acc); // sum of masked diff^2
    }
    int result = (int)v_reduce_sum(acc);
    for (; i < len; i++) if (mask[i]) { int v = (int)s1[i] - (int)s2[i]; result += v*v; }
    return result;
}

template<>
struct MaskedNormDiffInf_SIMD<uchar, int> {
    int operator()(const uchar* s1, const uchar* s2, const uchar* mask, int len, int cn) const {
        int result = 0;
        const int vstep = VTraits<v_uint8>::vlanes();
        if (cn == 1) {
            int i = 0;
            v_uint8 acc = vx_setzero_u8();
            for (; i <= len - vstep; i += vstep) {
                v_uint8 ad = v_absdiff(vx_load(s1 + i), vx_load(s2 + i));
                v_uint8 m  = v_gt(vx_load(mask + i), vx_setzero_u8());
                acc = v_max(acc, v_and(ad, m));
            }
            result = (int)v_reduce_max(acc);
            for (; i < len; i++) if (mask[i]) result = std::max(result, std::abs((int)s1[i] - (int)s2[i]));
        }
        else if (cn == 4 && len >= vstep) {
            v_uint8 acc = vx_setzero_u8();
            int i = 0;
            for (;;) {
                if (i > len - vstep) i = len - vstep; // back-step (max is idempotent)
                v_uint8 a0,a1,a2,a3,b0,b1,b2,b3;
                v_load_deinterleave(s1 + i*4, a0,a1,a2,a3);
                v_load_deinterleave(s2 + i*4, b0,b1,b2,b3);
                v_uint8 ad = v_max(v_max(v_absdiff(a0,b0), v_absdiff(a1,b1)),
                                   v_max(v_absdiff(a2,b2), v_absdiff(a3,b3)));
                v_uint8 m = v_gt(vx_load(mask + i), vx_setzero_u8());
                acc = v_max(acc, v_and(ad, m));
                if (i >= len - vstep) break;
                i += vstep;
            }
            result = (int)v_reduce_max(acc);
        }
        else {
            for (int i = 0; i < len; i++) if (mask[i]) {
                const uchar* e1 = s1 + i*cn; const uchar* e2 = s2 + i*cn;
                for (int k = 0; k < cn; k++) result = std::max(result, std::abs((int)e1[k] - (int)e2[k]));
            }
        }
        return result;
    }
};

template<>
struct MaskedNormDiffInf_SIMD<int, int> {
    int operator()(const int* s1, const int* s2, const uchar* mask, int len, int cn) const {
        int result = 0;
        const int vstep = VTraits<v_int32>::vlanes();
        if (cn == 1 && len >= vstep) {
            // Use wrapping int subtraction (v_abs(v_sub)) to match the non-masked
            // NormDiffInf_SIMD<int,int> kernel. v_absdiff would compute the true
            // unsigned |a-b| which can exceed INT_MAX and overflow on the cast to int.
            v_uint32 acc = vx_setzero_u32();
            int i = 0;
            for (; i <= len - vstep; i += vstep) {
                v_uint32 ad = v_abs(v_sub(vx_load(s1 + i), vx_load(s2 + i)));
                v_uint32 m  = v_gt(vx_load_expand_q(mask + i), vx_setzero_u32());
                acc = v_max(acc, v_and(ad, m));
            }
            result = (int)v_reduce_max(acc);
            for (; i < len; i++) if (mask[i]) result = std::max(result, (int)std::abs(s1[i] - s2[i]));
            return result;
        }
        for (int i = 0; i < len; i++) if (mask[i]) {
            const int* e1 = s1 + i*cn; const int* e2 = s2 + i*cn;
            for (int k = 0; k < cn; k++) result = std::max(result, (int)std::abs(e1[k] - e2[k]));
        }
        return result;
    }
};

template<>
struct MaskedNormDiffL1_SIMD<uchar, int> {
    int operator()(const uchar* s1, const uchar* s2, const uchar* mask, int len, int cn) const {
        if (cn == 1) return maskedNormDiffL1_8(s1, s2, mask, len);
        if (cn == 4) {
            int result = 0;
            int i = 0;
            const int vstep = VTraits<v_uint8>::vlanes();
            const v_uint8 one = vx_setall_u8(1);
            v_uint32 acc = vx_setzero_u32();
            for (; i <= len - vstep; i += vstep) {
                v_uint8 a0,a1,a2,a3,b0,b1,b2,b3;
                v_load_deinterleave(s1 + i*4, a0,a1,a2,a3);
                v_load_deinterleave(s2 + i*4, b0,b1,b2,b3);
                v_uint8 m = v_gt(vx_load(mask + i), vx_setzero_u8());
                acc = v_dotprod_expand_fast(v_and(v_absdiff(a0,b0), m), one, acc);
                acc = v_dotprod_expand_fast(v_and(v_absdiff(a1,b1), m), one, acc);
                acc = v_dotprod_expand_fast(v_and(v_absdiff(a2,b2), m), one, acc);
                acc = v_dotprod_expand_fast(v_and(v_absdiff(a3,b3), m), one, acc);
            }
            result = (int)v_reduce_sum(acc);
            for (; i < len; i++) if (mask[i]) {
                const uchar* e1 = s1 + i*4; const uchar* e2 = s2 + i*4;
                for (int k = 0; k < 4; k++) result += std::abs((int)e1[k] - (int)e2[k]);
            }
            return result;
        }
        return maskedNormDiffL1Tail<uchar, int>(s1, s2, mask, len, cn, 0);
    }
};

template<>
struct MaskedNormDiffL2_SIMD<uchar, int> {
    int operator()(const uchar* s1, const uchar* s2, const uchar* mask, int len, int cn) const {
        if (cn == 1) return maskedNormDiffL2_8(s1, s2, mask, len);
        if (cn == 4) {
            int result = 0;
            int i = 0;
            const int vstep = VTraits<v_uint8>::vlanes();
            v_uint32 acc = vx_setzero_u32();
            for (; i <= len - vstep; i += vstep) {
                v_uint8 a0,a1,a2,a3,b0,b1,b2,b3;
                v_load_deinterleave(s1 + i*4, a0,a1,a2,a3);
                v_load_deinterleave(s2 + i*4, b0,b1,b2,b3);
                v_uint8 m = v_gt(vx_load(mask + i), vx_setzero_u8());
                v_uint8 d0 = v_and(v_absdiff(a0,b0), m), d1 = v_and(v_absdiff(a1,b1), m);
                v_uint8 d2 = v_and(v_absdiff(a2,b2), m), d3 = v_and(v_absdiff(a3,b3), m);
                acc = v_dotprod_expand_fast(d0, d0, acc);
                acc = v_dotprod_expand_fast(d1, d1, acc);
                acc = v_dotprod_expand_fast(d2, d2, acc);
                acc = v_dotprod_expand_fast(d3, d3, acc);
            }
            result = (int)v_reduce_sum(acc);
            for (; i < len; i++) if (mask[i]) {
                const uchar* e1 = s1 + i*4; const uchar* e2 = s2 + i*4;
                for (int k = 0; k < 4; k++) { int v = (int)e1[k] - (int)e2[k]; result += v*v; }
            }
            return result;
        }
        return maskedNormDiffL2Tail<uchar, int>(s1, s2, mask, len, cn, 0);
    }
};

template<>
struct MaskedNormDiffInf_SIMD<double, double> {
    inline double operator()(const double* s1, const double* s2, const uchar* mask, int len, int cn) const {
        double result = 0.0;
        for (int i = 0; i < len; i++) if (mask[i]) {
            const double* e1 = s1 + i*cn; const double* e2 = s2 + i*cn;
            for (int k = 0; k < cn; k++) result = std::max(result, std::abs(e1[k] - e2[k]));
        }
        return result;
    }
};

#if CV_SIMD_64F || CV_SIMD_SCALABLE_64F
template<>
struct MaskedNormDiffL1_SIMD<int, double> {
    double operator()(const int* s1, const int* s2, const uchar* mask, int len, int cn) const {
        double result = 0.0;
        if (cn == 1) {
            int i = 0;
            const int vstep = VTraits<v_int32>::vlanes();
            v_float64 acc = vx_setzero_f64();
            for (; i <= len - vstep; i += vstep) {
                v_uint32 m  = v_gt(vx_load_expand_q(mask + i), vx_setzero_u32());
                v_uint32 ad = v_and(v_abs(v_sub(vx_load(s1 + i), vx_load(s2 + i))), m);
                v_int32 adi = v_reinterpret_as_s32(ad);
                acc = v_add(acc, v_cvt_f64(adi));
                acc = v_add(acc, v_cvt_f64_high(adi));
            }
            result = v_reduce_sum(acc);
            // Use wrapping int subtraction to match the SIMD body (v_abs(v_sub))
            // and the non-masked NormDiffL1_SIMD<int,double> kernel.
            for (; i < len; i++) if (mask[i]) result += std::abs(s1[i] - s2[i]);
        }
        else {
            result = maskedNormDiffL1Tail<int, double>(s1, s2, mask, len, cn, 0.0);
        }
        return result;
    }
};

// Masked L1/L2 SIMD for the remaining depths (cn==1). Without these the masked
// norm-diff for 8s/16u/16s/32s/32f fell back to the scalar base template, which
// the compiler autovectorizes poorly once AVX-512 dispatch is enabled.
template<> struct MaskedNormDiffL1_SIMD<schar, int> {
    int operator()(const schar* s1, const schar* s2, const uchar* mask, int len, int cn) const {
        if (cn == 1) return maskedNormDiffL1_8(s1, s2, mask, len);
        return maskedNormDiffL1Tail<schar, int>(s1, s2, mask, len, cn, 0);
    }
};

// Shared 16-bit masked L1 kernel (ushort/short): v_absdiff() yields v_uint16
// for both, so a single typename-T body serves both depths.
template<typename T>
static inline int maskedNormDiffL1_16(const T* s1, const T* s2, const uchar* mask, int len, int cn) {
    int result = 0;
    if (cn == 1) {
        int i = 0; const int vstep = VTraits<v_uint16>::vlanes();
        v_uint32 acc = vx_setzero_u32();
        for (; i <= len - vstep; i += vstep) {
            v_uint16 m  = v_gt(vx_load_expand(mask + i), vx_setzero_u16());
            v_uint16 ad = v_and(v_absdiff(vx_load(s1 + i), vx_load(s2 + i)), m);
            v_uint32 lo, hi; v_expand(ad, lo, hi);
            acc = v_add(acc, v_add(lo, hi));
        }
        result = (int)v_reduce_sum(acc);
        for (; i < len; i++) if (mask[i]) result += std::abs((int)s1[i] - (int)s2[i]);
    } else {
        result = maskedNormDiffL1Tail<T, int>(s1, s2, mask, len, cn, 0);
    }
    return result;
}
template<> struct MaskedNormDiffL1_SIMD<ushort, int> {
    int operator()(const ushort* s1, const ushort* s2, const uchar* mask, int len, int cn) const
    { return maskedNormDiffL1_16(s1, s2, mask, len, cn); } };
template<> struct MaskedNormDiffL1_SIMD<short, int> {
    int operator()(const short* s1, const short* s2, const uchar* mask, int len, int cn) const
    { return maskedNormDiffL1_16(s1, s2, mask, len, cn); } };

template<>
struct MaskedNormDiffL1_SIMD<float, double> {
    double operator()(const float* s1, const float* s2, const uchar* mask, int len, int cn) const {
        double result = 0.0;
        if (cn == 1) {
            int i = 0; const int vstep = VTraits<v_float32>::vlanes();
            v_float64 acc0 = vx_setzero_f64(), acc1 = vx_setzero_f64();
            for (; i <= len - vstep; i += vstep) {
                v_float32 m  = v_reinterpret_as_f32(v_gt(vx_load_expand_q(mask + i), vx_setzero_u32()));
                v_float32 ad = v_and(v_absdiff(vx_load(s1 + i), vx_load(s2 + i)), m);
                acc0 = v_add(acc0, v_cvt_f64(ad));
                acc1 = v_add(acc1, v_cvt_f64_high(ad));
            }
            result = v_reduce_sum(v_add(acc0, acc1));
            for (; i < len; i++) if (mask[i]) result += std::abs((double)s1[i] - (double)s2[i]);
        } else {
            result = maskedNormDiffL1Tail<float, double>(s1, s2, mask, len, cn, 0.0);
        }
        return result;
    }
};

template<> struct MaskedNormDiffL2_SIMD<schar, int> {
    int operator()(const schar* s1, const schar* s2, const uchar* mask, int len, int cn) const {
        if (cn == 1) return maskedNormDiffL2_8(s1, s2, mask, len);
        return maskedNormDiffL2Tail<schar, int>(s1, s2, mask, len, cn, 0);
    }
};

// Shared 16-bit masked L2 kernel (ushort/short): v_absdiff() yields v_uint16
// for both, so a single typename-T body serves both depths.
template<typename T>
static inline double maskedNormDiffL2_16(const T* s1, const T* s2, const uchar* mask, int len, int cn) {
    double result = 0.0;
    if (cn == 1) {
        int i = 0; const int vstep = VTraits<v_uint16>::vlanes();
        v_float64 acc = vx_setzero_f64();
        for (; i <= len - vstep; i += vstep) {
            v_uint16 m  = v_gt(vx_load_expand(mask + i), vx_setzero_u16());
            v_uint16 ad = v_and(v_absdiff(vx_load(s1 + i), vx_load(s2 + i)), m);
            v_uint64 u = v_dotprod_expand_fast(ad, ad);
            acc = v_add(acc, v_cvt_f64(v_reinterpret_as_s64(u)));
        }
        result = v_reduce_sum(acc);
        for (; i < len; i++) if (mask[i]) { double v = (double)s1[i] - (double)s2[i]; result += v*v; }
    } else {
        result = maskedNormDiffL2Tail<T, double>(s1, s2, mask, len, cn, 0.0);
    }
    return result;
}
template<> struct MaskedNormDiffL2_SIMD<ushort, double> {
    double operator()(const ushort* s1, const ushort* s2, const uchar* mask, int len, int cn) const
    { return maskedNormDiffL2_16(s1, s2, mask, len, cn); } };
template<> struct MaskedNormDiffL2_SIMD<short, double> {
    double operator()(const short* s1, const short* s2, const uchar* mask, int len, int cn) const
    { return maskedNormDiffL2_16(s1, s2, mask, len, cn); } };

template<>
struct MaskedNormDiffL2_SIMD<int, double> {
    double operator()(const int* s1, const int* s2, const uchar* mask, int len, int cn) const {
        double result = 0.0;
        if (cn == 1) {
            int i = 0; const int vstep = VTraits<v_int32>::vlanes();
            v_float64 r0 = vx_setzero_f64(), r1 = vx_setzero_f64();
            for (; i <= len - vstep; i += vstep) {
                v_uint32 m  = v_gt(vx_load_expand_q(mask + i), vx_setzero_u32());
                v_uint32 ad = v_and(v_absdiff(vx_load(s1 + i), vx_load(s2 + i)), m);
                v_uint64 e0, e1; v_expand(ad, e0, e1);
                v_float64 f0 = v_cvt_f64(v_reinterpret_as_s64(e0)), f1 = v_cvt_f64(v_reinterpret_as_s64(e1));
                r0 = v_fma(f0, f0, r0); r1 = v_fma(f1, f1, r1);
            }
            result = v_reduce_sum(v_add(r0, r1));
            for (; i < len; i++) if (mask[i]) { double v = (double)s1[i] - (double)s2[i]; result += v*v; }
        } else {
            result = maskedNormDiffL2Tail<int, double>(s1, s2, mask, len, cn, 0.0);
        }
        return result;
    }
};

template<>
struct MaskedNormDiffL2_SIMD<float, double> {
    double operator()(const float* s1, const float* s2, const uchar* mask, int len, int cn) const {
        double result = 0.0;
        if (cn == 1) {
            int i = 0; const int vstep = VTraits<v_float32>::vlanes();
            v_float64 r0 = vx_setzero_f64(), r1 = vx_setzero_f64();
            for (; i <= len - vstep; i += vstep) {
                v_float32 m  = v_reinterpret_as_f32(v_gt(vx_load_expand_q(mask + i), vx_setzero_u32()));
                v_float32 ad = v_and(v_absdiff(vx_load(s1 + i), vx_load(s2 + i)), m);
                v_float64 f0 = v_cvt_f64(ad), f1 = v_cvt_f64_high(ad);
                r0 = v_fma(f0, f0, r0); r1 = v_fma(f1, f1, r1);
            }
            result = v_reduce_sum(v_add(r0, r1));
            for (; i < len; i++) if (mask[i]) { double v = (double)s1[i] - (double)s2[i]; result += v*v; }
        } else {
            result = maskedNormDiffL2Tail<float, double>(s1, s2, mask, len, cn, 0.0);
        }
        return result;
    }
};
#endif
#endif

template<typename T, typename ST> int
normDiffInf_(const T* src1, const T* src2, const uchar* mask, ST* _result, int len, int cn) {
    ST result = *_result;
    if( !mask ) {
        NormDiffInf_SIMD<T, ST> op;
        result = std::max(result, op(src1, src2, len*cn));
    } else {
        MaskedNormDiffInf_SIMD<T, ST> op;
        result = std::max(result, op(src1, src2, mask, len, cn));
    }
    *_result = result;
    return 0;
}

template<typename T, typename ST> int
normDiffL1_(const T* src1, const T* src2, const uchar* mask, ST* _result, int len, int cn) {
    ST result = *_result;
    if( !mask ) {
        NormDiffL1_SIMD<T, ST> op;
        result += op(src1, src2, len*cn);
    }
    else {
        MaskedNormDiffL1_SIMD<T, ST> op;
        result += op(src1, src2, mask, len, cn);
    }
    *_result = result;
    return 0;
}

template<typename T, typename ST> int
normDiffL2_(const T* src1, const T* src2, const uchar* mask, ST* _result, int len, int cn) {
    ST result = *_result;
    if( !mask ) {
        NormDiffL2_SIMD<T, ST> op;
        result += op(src1, src2, len*cn);
    } else {
        MaskedNormDiffL2_SIMD<T, ST> op;
        result += op(src1, src2, mask, len, cn);
    }
    *_result = result;
    return 0;
}

#define CV_DEF_NORM_FUNC(L, suffix, type, ntype) \
    static int norm##L##_##suffix(const type* src, const uchar* mask, ntype* r, int len, int cn) \
{ CV_INSTRUMENT_REGION(); return norm##L##_(src, mask, r, len, cn); } \
    static int normDiff##L##_##suffix(const type* src1, const type* src2, \
    const uchar* mask, ntype* r, int len, int cn) \
{ return normDiff##L##_(src1, src2, mask, r, (int)len, cn); }

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

    static NormFunc normTab[3][CV_DEPTH_MAX] =
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

    if (normType >= 3 || normType < 0) return nullptr;

    return normTab[normType][depth];
}

NormDiffFunc getNormDiffFunc(int normType, int depth)
{
    static NormDiffFunc normDiffTab[3][CV_DEPTH_MAX] =
    {
        {
            (NormDiffFunc)GET_OPTIMIZED(normDiffInf_8u), (NormDiffFunc)normDiffInf_8s,
            (NormDiffFunc)normDiffInf_16u, (NormDiffFunc)normDiffInf_16s,
            (NormDiffFunc)normDiffInf_32s, (NormDiffFunc)GET_OPTIMIZED(normDiffInf_32f),
            (NormDiffFunc)normDiffInf_64f, 0
        },
        {
            (NormDiffFunc)GET_OPTIMIZED(normDiffL1_8u), (NormDiffFunc)normDiffL1_8s,
            (NormDiffFunc)normDiffL1_16u, (NormDiffFunc)normDiffL1_16s,
            (NormDiffFunc)normDiffL1_32s, (NormDiffFunc)GET_OPTIMIZED(normDiffL1_32f),
            (NormDiffFunc)normDiffL1_64f, 0
        },
        {
            (NormDiffFunc)GET_OPTIMIZED(normDiffL2_8u), (NormDiffFunc)normDiffL2_8s,
            (NormDiffFunc)normDiffL2_16u, (NormDiffFunc)normDiffL2_16s,
            (NormDiffFunc)normDiffL2_32s, (NormDiffFunc)GET_OPTIMIZED(normDiffL2_32f),
            (NormDiffFunc)normDiffL2_64f, 0
        }
    };
    if (normType >= 3 || normType < 0) return nullptr;

    return normDiffTab[normType][depth];
}

#endif // CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

CV_CPU_OPTIMIZATION_NAMESPACE_END

} // cv::
