// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "opencv2/core/hal/intrin.hpp"

namespace cv {
namespace dnn {
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

void addInt8(const void* a, const void* b, void* out, int len, bool isU8,
             float c0, float c1, float offset, int lo, int hi, const uchar* lut);
void mulInt8(const void* a, const void* b, void* out, int len, bool isU8,
             float zp0, float zp1, float scale, float offset, int lo, int hi, const uchar* lut);
void lutInt8(const void* src, void* dst, int len, bool isU8, const uchar* lut);

#ifndef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

void lutInt8(const void* src_, void* dst_, int len, bool isU8, const uchar* lut)
{
    const uchar* src = (const uchar*)src_;
    uchar* dst = (uchar*)dst_;
    // int8 v sits at lut[v + 128], and v + 128 is v's byte with the top bit flipped.
    int i = 0;
#if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
    const uchar flip = isU8 ? 0 : 0x80;
    // The table is four 64-byte quarters. vqtbx4q leaves a lane alone when its index is out
    // of range, so each pass fills only the lanes that fall in its quarter.
    uint8x16x4_t q[4];
    for (int k = 0; k < 4; k++)
        for (int j = 0; j < 4; j++)
            q[k].val[j] = vld1q_u8(lut + k * 64 + j * 16);
    const uint8x16_t vflip = vdupq_n_u8(flip), v64 = vdupq_n_u8(64);
    for (; i <= len - 16; i += 16) {
        uint8x16_t idx = veorq_u8(vld1q_u8(src + i), vflip);
        uint8x16_t r = vqtbl4q_u8(q[0], idx);
        idx = vsubq_u8(idx, v64); r = vqtbx4q_u8(r, q[1], idx);
        idx = vsubq_u8(idx, v64); r = vqtbx4q_u8(r, q[2], idx);
        idx = vsubq_u8(idx, v64); r = vqtbx4q_u8(r, q[3], idx);
        vst1q_u8(dst + i, r);
    }
#endif
    // pshufb-based lookups measured no faster than this loop on x86.
    if (isU8) {
        for (; i < len; i++)
            dst[i] = lut[src[i]];
    } else {
        for (; i < len; i++)
            dst[i] = lut[src[i] ^ 0x80];
    }
}

#if (CV_SIMD || CV_SIMD_SCALABLE)
static inline v_float32 loadQuarterF32(const uchar* p)
{ return v_cvt_f32(v_reinterpret_as_s32(vx_load_expand_q(p))); }
static inline v_float32 loadQuarterF32(const schar* p)
{ return v_cvt_f32(vx_load_expand_q(p)); }

static inline void storePacked(uchar* p, const v_int32& r0, const v_int32& r1,
                               const v_int32& r2, const v_int32& r3)
{ v_store(p, v_pack_u(v_pack(r0, r1), v_pack(r2, r3))); }
static inline void storePacked(schar* p, const v_int32& r0, const v_int32& r1,
                               const v_int32& r2, const v_int32& r3)
{ v_store(p, v_pack(v_pack(r0, r1), v_pack(r2, r3))); }
#endif

enum { ELTWISE_INT8_ADD, ELTWISE_INT8_MUL };

// For ADD, (p0, p1, p2) = (c0, c1, offset): p0*a + p1*b + p2.
// For MUL, (p0, p1, p2, p3) = (zp0, zp1, scale, offset): p2*(a - p0)*(b - p1) + p3.
template<int OP, typename T>
static void binaryInt8(const T* a, const T* b, T* out, int len,
                       float p0, float p1, float p2, float p3, int lo, int hi)
{
    int i = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    const int nlanes = VTraits<v_int32>::vlanes();
    const int step = nlanes * 4;
    const v_float32 v0 = vx_setall_f32(p0), v1 = vx_setall_f32(p1);
    const v_float32 v2 = vx_setall_f32(p2), v3 = vx_setall_f32(p3);
    const v_int32 vlo = vx_setall_s32(lo), vhi = vx_setall_s32(hi);
#define CV_ELTWISE_INT8_QUARTER(k, r) \
    { \
        v_float32 fa = loadQuarterF32(a + i + (k) * nlanes); \
        v_float32 fb = loadQuarterF32(b + i + (k) * nlanes); \
        v_float32 fv = OP == ELTWISE_INT8_ADD ? v_fma(fa, v0, v_fma(fb, v1, v2)) \
                     : v_fma(v_mul(v_sub(fa, v0), v_sub(fb, v1)), v2, v3); \
        r = v_min(v_max(v_round(fv), vlo), vhi); \
    }
    for (; i <= len - step; i += step) {
        v_int32 r0, r1, r2, r3;
        CV_ELTWISE_INT8_QUARTER(0, r0)
        CV_ELTWISE_INT8_QUARTER(1, r1)
        CV_ELTWISE_INT8_QUARTER(2, r2)
        CV_ELTWISE_INT8_QUARTER(3, r3)
        storePacked(out + i, r0, r1, r2, r3);
    }
#undef CV_ELTWISE_INT8_QUARTER
#endif
    for (; i < len; i++) {
        const float fa = (float)a[i], fb = (float)b[i];
        const float fv = OP == ELTWISE_INT8_ADD ? fa * p0 + (fb * p1 + p2)
                                                : (fa - p0) * (fb - p1) * p2 + p3;
        out[i] = (T)std::max(lo, std::min(hi, cvRound(fv)));
    }
}

// Applies the lut per block, while the block is still in L1.
template<int OP>
static void binaryInt8Blocked(const void* a, const void* b, void* out, int len, bool isU8,
                              float p0, float p1, float p2, float p3, int lo, int hi,
                              const uchar* lut)
{
    const int BLOCK = lut ? 1024 : len;
    for (int i = 0; i < len; i += BLOCK) {
        const int n = std::min(BLOCK, len - i);
        if (isU8)
            binaryInt8<OP>((const uchar*)a + i, (const uchar*)b + i, (uchar*)out + i, n,
                           p0, p1, p2, p3, lo, hi);
        else
            binaryInt8<OP>((const schar*)a + i, (const schar*)b + i, (schar*)out + i, n,
                           p0, p1, p2, p3, lo, hi);
        if (lut)
            lutInt8((uchar*)out + i, (uchar*)out + i, n, isU8, lut);
    }
}

void addInt8(const void* a, const void* b, void* out, int len, bool isU8,
             float c0, float c1, float offset, int lo, int hi, const uchar* lut)
{
    binaryInt8Blocked<ELTWISE_INT8_ADD>(a, b, out, len, isU8, c0, c1, offset, 0.f, lo, hi, lut);
}

void mulInt8(const void* a, const void* b, void* out, int len, bool isU8,
             float zp0, float zp1, float scale, float offset, int lo, int hi, const uchar* lut)
{
    binaryInt8Blocked<ELTWISE_INT8_MUL>(a, b, out, len, isU8, zp0, zp1, scale, offset, lo, hi, lut);
}

#endif // CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

CV_CPU_OPTIMIZATION_NAMESPACE_END
}} // namespace cv::dnn
