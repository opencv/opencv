// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2026, Advanced Micro Devices, all rights reserved.

#include "opencv2/core/hal/intrin.hpp"

namespace cv {

CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

int remapBilinearC1_simd(int depth, const uchar* S0, size_t sstep, uchar* D,
                         const short* XY, const ushort* FXY, const float* wtab,
                         int dx, int X1, int off_y);

int remapBicubicC1_simd(int depth, const uchar* S0, size_t sstep, uchar* D,
                        const short* XY, const ushort* FXY,
                        int dx, int dwidth, unsigned width1, unsigned height1, int off_y);

int remapLanczos4C1_simd(int depth, const uchar* S0, size_t sstep, uchar* D,
                         const short* XY, const ushort* FXY, const float* wtab,
                         int dx, int dwidth, unsigned width1, unsigned height1, int off_y);


int remapBicubicC1wp_simd(int depth, const uchar* S0, size_t sstep, uchar* D,
                          const short* XY, const ushort* FXY, const float* wtab,
                          int dx, int dwidth, unsigned width1, unsigned height1, int off_y);

#ifndef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

#if (CV_SIMD || CV_SIMD_SCALABLE)

static inline v_float32 remapGatherF32(const float* base, const int* ofs)
{
    float CV_DECL_ALIGNED(CV_SIMD_WIDTH) buf[VTraits<v_float32>::max_nlanes];
    const int n = VTraits<v_float32>::vlanes();
    for (int k = 0; k < n; k++)
        buf[k] = base[ofs[k]];
    return vx_load(buf);
}

static inline v_float32 remapGatherF32(const ushort* base, const int* ofs)
{
    float CV_DECL_ALIGNED(CV_SIMD_WIDTH) buf[VTraits<v_float32>::max_nlanes];
    const int n = VTraits<v_float32>::vlanes();
    for (int k = 0; k < n; k++)
        buf[k] = (float)base[ofs[k]];
    return vx_load(buf);
}

static inline v_float32 remapGatherF32(const short* base, const int* ofs)
{
    float CV_DECL_ALIGNED(CV_SIMD_WIDTH) buf[VTraits<v_float32>::max_nlanes];
    const int n = VTraits<v_float32>::vlanes();
    for (int k = 0; k < n; k++)
        buf[k] = (float)base[ofs[k]];
    return vx_load(buf);
}

static CV_ALWAYS_INLINE void remapCorners(const short* S0, size_t sstep, const int* ofs,
                                v_float32& s0, v_float32& s1,
                                v_float32& s2, v_float32& s3)
{
    int CV_DECL_ALIGNED(CV_SIMD_WIDTH) topbuf[VTraits<v_float32>::max_nlanes];
    int CV_DECL_ALIGNED(CV_SIMD_WIDTH) botbuf[VTraits<v_float32>::max_nlanes];
    const int n = VTraits<v_float32>::vlanes();
    for (int k = 0; k < n; k++)
    {
        const short* p = S0 + ofs[k];
        int t, b;
        memcpy(&t, p, sizeof(t));
        memcpy(&b, p + sstep, sizeof(b));
        topbuf[k] = t; botbuf[k] = b;
    }
    v_int32 top = vx_load(topbuf), bot = vx_load(botbuf);
    s0 = v_cvt_f32(v_shr<16>(v_shl<16>(top)));   // low 16 bits (sign-extended)
    s1 = v_cvt_f32(v_shr<16>(top));              // high 16 bits (sign-extended)
    s2 = v_cvt_f32(v_shr<16>(v_shl<16>(bot)));
    s3 = v_cvt_f32(v_shr<16>(bot));
}

static CV_ALWAYS_INLINE void remapCorners(const ushort* S0, size_t sstep, const int* ofs,
                                v_float32& s0, v_float32& s1,
                                v_float32& s2, v_float32& s3)
{
    int CV_DECL_ALIGNED(CV_SIMD_WIDTH) topbuf[VTraits<v_float32>::max_nlanes];
    int CV_DECL_ALIGNED(CV_SIMD_WIDTH) botbuf[VTraits<v_float32>::max_nlanes];
    const int n = VTraits<v_float32>::vlanes();
    for (int k = 0; k < n; k++)
    {
        const ushort* p = S0 + ofs[k];
        int t, b;
        memcpy(&t, p, sizeof(t));
        memcpy(&b, p + sstep, sizeof(b));
        topbuf[k] = t; botbuf[k] = b;
    }
    const v_uint32 lo16 = vx_setall_u32(0xffff);
    v_uint32 top = v_reinterpret_as_u32(vx_load(topbuf));
    v_uint32 bot = v_reinterpret_as_u32(vx_load(botbuf));
    s0 = v_cvt_f32(v_reinterpret_as_s32(v_and(top, lo16)));  // low 16 (zero-ext)
    s1 = v_cvt_f32(v_reinterpret_as_s32(v_shr<16>(top)));    // high 16 (zero-ext)
    s2 = v_cvt_f32(v_reinterpret_as_s32(v_and(bot, lo16)));
    s3 = v_cvt_f32(v_reinterpret_as_s32(v_shr<16>(bot)));
}

static inline void remapStoreC1(float* D, const v_float32& res)
{
    v_store(D, res);
}

static inline void remapStoreC1(ushort* D, const v_float32& res)
{
    v_pack_u_store(D, v_round(res));
}

static inline void remapStoreC1(short* D, const v_float32& res)
{
    v_pack_store(D, v_round(res));
}

template<typename T>
static int remapBilinearC1_run(const T* S0, size_t sstep, T* D,
                               const short* XY, const ushort* FXY,
                               const float* wtab, int dx, int X1, int off_y)
{
    CV_UNUSED(wtab);
    const int vlanes = VTraits<v_float32>::vlanes();
    const int dx0 = dx;
    const v_float32 vone   = vx_setall_f32(1.f);
    const v_float32 vscale = vx_setall_f32(1.f / INTER_TAB_SIZE);
    const v_int32   vmask  = vx_setall_s32(INTER_TAB_SIZE - 1);
    int CV_DECL_ALIGNED(CV_SIMD_WIDTH) ofs[VTraits<v_float32>::max_nlanes];
    for( ; dx <= X1 - vlanes; dx += vlanes )
    {
        for( int k = 0; k < vlanes; k++ )
        {
            const int sx = XY[(dx + k) * 2];
            const int sy = XY[(dx + k) * 2 + 1] + off_y;
            ofs[k] = sy * (int)sstep + sx;
        }
        v_float32 s0, s1, s2, s3;
        remapCorners(S0, sstep, ofs, s0, s1, s2, s3);

        v_int32 fxy = v_reinterpret_as_s32(vx_load_expand(FXY + dx));
        v_float32 fx = v_mul(v_cvt_f32(v_and(fxy, vmask)),    vscale);
        v_float32 fy = v_mul(v_cvt_f32(v_shr<INTER_BITS>(fxy)), vscale);
        v_float32 cx0 = v_sub(vone, fx);
        v_float32 cy0 = v_sub(vone, fy);
        v_float32 w0 = v_mul(cx0, cy0);
        v_float32 w1 = v_mul(fx,  cy0);
        v_float32 w2 = v_mul(cx0, fy);
        v_float32 w3 = v_mul(fx,  fy);

        v_float32 res = v_fma(s0, w0, v_fma(s1, w1, v_fma(s2, w2, v_mul(s3, w3))));
        remapStoreC1(D + (dx - dx0), res);
    }
    vx_cleanup();
    return dx - dx0;
}

static int remapBilinearF32_run(const float* S0, size_t sstep, float* D,
                                const short* XY, const ushort* FXY,
                                const float* wtab, int dx, int X1, int off_y)
{
    CV_UNUSED(wtab);
    const int vlanes = VTraits<v_float32>::vlanes();
    const int dx0 = dx;
    const float* S1 = S0 + sstep;
    const v_float32 vone   = vx_setall_f32(1.f);
    const v_float32 vscale = vx_setall_f32(1.f / INTER_TAB_SIZE);
    const v_int32   vmask  = vx_setall_s32(INTER_TAB_SIZE - 1);
    int CV_DECL_ALIGNED(CV_SIMD_WIDTH) ofs[VTraits<v_float32>::max_nlanes];
    for( ; dx <= X1 - vlanes; dx += vlanes )
    {
        for( int k = 0; k < vlanes; k++ )
        {
            const int sx = XY[(dx + k) * 2];
            const int sy = XY[(dx + k) * 2 + 1] + off_y;
            ofs[k] = sy * (int)sstep + sx;
        }
        v_float32 s0 = remapGatherF32(S0,     ofs);
        v_float32 s1 = remapGatherF32(S0 + 1, ofs);
        v_float32 s2 = remapGatherF32(S1,     ofs);
        v_float32 s3 = remapGatherF32(S1 + 1, ofs);

        v_int32 fxy = v_reinterpret_as_s32(vx_load_expand(FXY + dx));
        v_float32 fx = v_mul(v_cvt_f32(v_and(fxy, vmask)),    vscale);
        v_float32 fy = v_mul(v_cvt_f32(v_shr<INTER_BITS>(fxy)), vscale);
        v_float32 cx0 = v_sub(vone, fx);
        v_float32 cy0 = v_sub(vone, fy);
        v_float32 w0 = v_mul(cx0, cy0);
        v_float32 w1 = v_mul(fx,  cy0);
        v_float32 w2 = v_mul(cx0, fy);
        v_float32 w3 = v_mul(fx,  fy);

        v_float32 res = v_fma(s0, w0, v_fma(s1, w1, v_fma(s2, w2, v_mul(s3, w3))));
        v_store(D + (dx - dx0), res);
    }
    vx_cleanup();
    return dx - dx0;
}

// Evaluate the four cubic interpolation coefficients for a vector of fractional
// positions, matching interpolateCubic() (A = -0.75). The strict remap test
// tolerates an absolute error of 1.0 for bicubic, so FMA contraction is fine.
static inline void interpolateCubicV(const v_float32& x,
                                     v_float32& c0, v_float32& c1,
                                     v_float32& c2, v_float32& c3)
{
    const v_float32 A    = vx_setall_f32(-0.75f);
    const v_float32 A5   = vx_setall_f32(-3.75f);  // 5*A
    const v_float32 A8   = vx_setall_f32(-6.0f);   // 8*A
    const v_float32 A4   = vx_setall_f32(-3.0f);   // 4*A
    const v_float32 Ap2  = vx_setall_f32(1.25f);   // A+2
    const v_float32 Ap3  = vx_setall_f32(2.25f);   // A+3
    const v_float32 one  = vx_setall_f32(1.f);

    v_float32 xp1 = v_add(x, one);
    c0 = v_sub(v_mul(v_add(v_mul(v_sub(v_mul(A, xp1), A5), xp1), A8), xp1), A4);
    c1 = v_add(v_mul(v_mul(v_sub(v_mul(Ap2, x), Ap3), x), x), one);
    v_float32 u = v_sub(one, x);
    c2 = v_add(v_mul(v_mul(v_sub(v_mul(Ap2, u), Ap3), u), u), one);
    c3 = v_sub(v_sub(v_sub(one, c0), c1), c2);
}

// Select one of four vectors by runtime index. Used instead of an array of
// vector types, which is not valid for sizeless RVV vector types.
static inline v_float32 selectV4(int i, const v_float32& a, const v_float32& b,
                                 const v_float32& c, const v_float32& d)
{
    return i == 0 ? a : (i == 1 ? b : (i == 2 ? c : d));
}

template<typename T>
static int remapBicubicC1_run(const T* S0, size_t sstep, T* D, const short* XY,
                              const ushort* FXY, int dx, int dwidth,
                              unsigned width1, unsigned height1, int off_y)
{
    const int vlanes = VTraits<v_float32>::vlanes();
    const int dx0 = dx;
    const v_float32 vscale = vx_setall_f32(1.f / INTER_TAB_SIZE);
    const v_int32   vmask  = vx_setall_s32(INTER_TAB_SIZE - 1);
    int CV_DECL_ALIGNED(CV_SIMD_WIDTH) ofs[VTraits<v_float32>::max_nlanes];
    float CV_DECL_ALIGNED(CV_SIMD_WIDTH) buf[VTraits<v_float32>::max_nlanes];
    for( ; dx <= dwidth - vlanes; dx += vlanes )
    {
        bool allIn = true;
        for( int k = 0; k < vlanes; k++ )
        {
            const unsigned sx = (unsigned)(XY[(dx + k) * 2] - 1);
            const unsigned sy = (unsigned)(XY[(dx + k) * 2 + 1] - 1 + off_y);
            if( sx >= width1 || sy >= height1 ) { allIn = false; break; }
            ofs[k] = (int)((XY[(dx + k) * 2 + 1] - 1 + off_y) * (int)sstep
                          + (XY[(dx + k) * 2] - 1));
        }
        if( !allIn )
            break;

        v_int32 fxy = v_reinterpret_as_s32(vx_load_expand(FXY + dx));
        v_float32 fx = v_mul(v_cvt_f32(v_and(fxy, vmask)),    vscale);
        v_float32 fy = v_mul(v_cvt_f32(v_shr<INTER_BITS>(fxy)), vscale);
        v_float32 vx0, vx1, vx2, vx3, vy0, vy1, vy2, vy3;
        interpolateCubicV(fx, vx0, vx1, vx2, vx3);
        interpolateCubicV(fy, vy0, vy1, vy2, vy3);

        v_float32 acc = vx_setzero_f32();
        for( int r = 0; r < 4; r++ )
        {
            const int roff = r * (int)sstep;
            const v_float32 vyr = selectV4(r, vy0, vy1, vy2, vy3);
            for( int c = 0; c < 4; c++ )
            {
                for( int k = 0; k < vlanes; k++ )
                    buf[k] = (float)S0[ofs[k] + roff + c];
                acc = v_fma(vx_load(buf), v_mul(vyr, selectV4(c, vx0, vx1, vx2, vx3)), acc);
            }
        }
        remapStoreC1(D + (dx - dx0), acc);
    }
    vx_cleanup();
    return dx - dx0;
}

#if CV_SIMD128
static inline void remapLoad8(const float* S, v_float32x4& lo, v_float32x4& hi)
{
    lo = v_load(S); hi = v_load(S + 4);
}
static inline void remapLoad8(const ushort* S, v_float32x4& lo, v_float32x4& hi)
{
    v_uint16x8 v = v_load(S);
    v_uint32x4 a, b; v_expand(v, a, b);
    lo = v_cvt_f32(v_reinterpret_as_s32(a));
    hi = v_cvt_f32(v_reinterpret_as_s32(b));
}
static inline void remapLoad8(const short* S, v_float32x4& lo, v_float32x4& hi)
{
    v_int16x8 v = v_load(S);
    v_int32x4 a, b; v_expand(v, a, b);
    lo = v_cvt_f32(a); hi = v_cvt_f32(b);
}

static inline void remapStoreScalar(float* D, float v) { *D = v; }
static inline void remapStoreScalar(ushort* D, float v) { *D = saturate_cast<ushort>(v); }
static inline void remapStoreScalar(short* D, float v) { *D = saturate_cast<short>(v); }

static inline v_float32x4 remapLoad4(const float* S)  { return v_load(S); }
static inline v_float32x4 remapLoad4(const ushort* S) { return v_cvt_f32(v_reinterpret_as_s32(v_load_expand(S))); }
static inline v_float32x4 remapLoad4(const short* S)  { return v_cvt_f32(v_load_expand(S)); }

template<typename T>
static int remapBicubicC1wp_run(const T* S0, size_t sstep, T* D, const short* XY,
                                const ushort* FXY, const float* wtab, int dx,
                                int dwidth, unsigned width1, unsigned height1, int off_y)
{
    const int dx0 = dx;
    for( ; dx < dwidth; dx++ )
    {
        const unsigned sx = (unsigned)(XY[dx * 2] - 1);
        const unsigned sy = (unsigned)(XY[dx * 2 + 1] - 1 + off_y);
        if( sx >= width1 || sy >= height1 )
            break;
        const float* w = wtab + FXY[dx] * 16;
        const T* S = S0 + (size_t)sy * sstep + sx;
        v_float32x4 acc = v_setzero_f32();
        for( int r = 0; r < 4; r++, S += sstep, w += 4 )
            acc = v_fma(remapLoad4(S), v_load(w), acc);
        remapStoreScalar(D + (dx - dx0), v_reduce_sum(acc));
    }
    vx_cleanup();
    return dx - dx0;
}

template<typename T>
static int remapLanczos4C1_run(const T* S0, size_t sstep, T* D, const short* XY,
                               const ushort* FXY, const float* wtab, int dx,
                               int dwidth, unsigned width1, unsigned height1, int off_y)
{
    const int dx0 = dx;
    for( ; dx < dwidth; dx++ )
    {
        const unsigned sx = (unsigned)(XY[dx * 2] - 3);
        const unsigned sy = (unsigned)(XY[dx * 2 + 1] - 3 + off_y);
        if( sx >= width1 || sy >= height1 )
            break;
        const float* w = wtab + FXY[dx] * 64;
        const T* S = S0 + (size_t)sy * sstep + sx;
        v_float32x4 acc = v_setzero_f32();
        for( int r = 0; r < 8; r++, S += sstep, w += 8 )
        {
            v_float32x4 s_lo, s_hi;
            remapLoad8(S, s_lo, s_hi);
            acc = v_fma(s_lo, v_load(w),     acc);
            acc = v_fma(s_hi, v_load(w + 4), acc);
        }
        remapStoreScalar(D + (dx - dx0), v_reduce_sum(acc));
    }
    vx_cleanup();
    return dx - dx0;
}
#endif // CV_SIMD128

#endif // CV_SIMD

int remapBilinearC1_simd(int depth, const uchar* S0, size_t sstep, uchar* D,
                         const short* XY, const ushort* FXY, const float* wtab,
                         int dx, int X1, int off_y)
{
#if (CV_SIMD || CV_SIMD_SCALABLE)
    switch (depth)
    {
    case CV_32F:
        return remapBilinearF32_run((const float*)S0, sstep, (float*)D,
                                    XY, FXY, wtab, dx, X1, off_y);
    case CV_16U:
        return remapBilinearC1_run<ushort>((const ushort*)S0, sstep, (ushort*)D,
                                           XY, FXY, wtab, dx, X1, off_y);
    case CV_16S:
        return remapBilinearC1_run<short>((const short*)S0, sstep, (short*)D,
                                          XY, FXY, wtab, dx, X1, off_y);
    default:
        return 0;
    }
#else
    CV_UNUSED(depth); CV_UNUSED(S0); CV_UNUSED(sstep); CV_UNUSED(D);
    CV_UNUSED(XY); CV_UNUSED(FXY); CV_UNUSED(wtab);
    CV_UNUSED(dx); CV_UNUSED(X1); CV_UNUSED(off_y);
    return 0;
#endif
}

int remapBicubicC1_simd(int depth, const uchar* S0, size_t sstep, uchar* D,
                        const short* XY, const ushort* FXY,
                        int dx, int dwidth, unsigned width1, unsigned height1, int off_y)
{
#if (CV_SIMD || CV_SIMD_SCALABLE)
    switch (depth)
    {
    case CV_32F:
        return remapBicubicC1_run<float>((const float*)S0, sstep, (float*)D,
                                         XY, FXY, dx, dwidth, width1, height1, off_y);
    case CV_16U:
        return remapBicubicC1_run<ushort>((const ushort*)S0, sstep, (ushort*)D,
                                          XY, FXY, dx, dwidth, width1, height1, off_y);
    case CV_16S:
        return remapBicubicC1_run<short>((const short*)S0, sstep, (short*)D,
                                         XY, FXY, dx, dwidth, width1, height1, off_y);
    default:
        return 0;
    }
#else
    CV_UNUSED(depth); CV_UNUSED(S0); CV_UNUSED(sstep); CV_UNUSED(D);
    CV_UNUSED(XY); CV_UNUSED(FXY); CV_UNUSED(dx); CV_UNUSED(dwidth);
    CV_UNUSED(width1); CV_UNUSED(height1); CV_UNUSED(off_y);
    return 0;
#endif
}

int remapLanczos4C1_simd(int depth, const uchar* S0, size_t sstep, uchar* D,
                         const short* XY, const ushort* FXY, const float* wtab,
                         int dx, int dwidth, unsigned width1, unsigned height1, int off_y)
{
#if CV_SIMD128
    switch (depth)
    {
    // CV_32F is intentionally omitted: the vectorized 8x8 accumulation reorders
    // the 64-tap sum, so on the tight 1e-3 float tolerance it diverges from the
    // scalar path (used for relative maps). 32F lanczos4 stays on the scalar loop.
    case CV_16U:
        return remapLanczos4C1_run<ushort>((const ushort*)S0, sstep, (ushort*)D,
                                           XY, FXY, wtab, dx, dwidth, width1, height1, off_y);
    case CV_16S:
        return remapLanczos4C1_run<short>((const short*)S0, sstep, (short*)D,
                                          XY, FXY, wtab, dx, dwidth, width1, height1, off_y);
    default:
        return 0;
    }
#else
    CV_UNUSED(depth); CV_UNUSED(S0); CV_UNUSED(sstep); CV_UNUSED(D);
    CV_UNUSED(XY); CV_UNUSED(FXY); CV_UNUSED(wtab); CV_UNUSED(dx); CV_UNUSED(dwidth);
    CV_UNUSED(width1); CV_UNUSED(height1); CV_UNUSED(off_y);
    return 0;
#endif
}

int remapBicubicC1wp_simd(int depth, const uchar* S0, size_t sstep, uchar* D,
                          const short* XY, const ushort* FXY, const float* wtab,
                          int dx, int dwidth, unsigned width1, unsigned height1, int off_y)
{
#if CV_SIMD128
    switch (depth)
    {
    case CV_32F:
        return remapBicubicC1wp_run<float>((const float*)S0, sstep, (float*)D,
                                           XY, FXY, wtab, dx, dwidth, width1, height1, off_y);
    case CV_16U:
        return remapBicubicC1wp_run<ushort>((const ushort*)S0, sstep, (ushort*)D,
                                            XY, FXY, wtab, dx, dwidth, width1, height1, off_y);
    case CV_16S:
        return remapBicubicC1wp_run<short>((const short*)S0, sstep, (short*)D,
                                           XY, FXY, wtab, dx, dwidth, width1, height1, off_y);
    default:
        return 0;
    }
#else
    CV_UNUSED(depth); CV_UNUSED(S0); CV_UNUSED(sstep); CV_UNUSED(D);
    CV_UNUSED(XY); CV_UNUSED(FXY); CV_UNUSED(wtab); CV_UNUSED(dx); CV_UNUSED(dwidth);
    CV_UNUSED(width1); CV_UNUSED(height1); CV_UNUSED(off_y);
    return 0;
#endif
}

#endif // CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

CV_CPU_OPTIMIZATION_NAMESPACE_END

} // namespace cv
