// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2026, Advanced Micro Devices, all rights reserved.

#include "opencv2/core/hal/intrin.hpp"

#include <climits>

namespace cv {

using RemapBilinearC1Fn = int (*)(int, const uchar*, size_t, uchar*, const short*, const ushort*, const float*, int, int, int);
using RemapWideC1Fn = int (*)(int, const uchar*, size_t, uchar*, const short*, const ushort*, const float*, int, int, unsigned, unsigned, int);

RemapBilinearC1Fn remapBilinearC1_fn();
RemapWideC1Fn remapLanczos4C1_fn();

CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

RemapBilinearC1Fn getRemapBilinearC1Fn_();
RemapWideC1Fn getRemapLanczos4C1Fn_();

int remapBilinearC1_simd(int depth, const uchar* S0, size_t sstep, uchar* D,
                         const short* XY, const ushort* FXY, const float* wtab,
                         int dx, int X1, int off_y);

int remapLanczos4C1_simd(int depth, const uchar* S0, size_t sstep, uchar* D,
                         const short* XY, const ushort* FXY, const float* wtab,
                         int dx, int dwidth, unsigned width1, unsigned height1, int off_y);

#ifndef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

#if (CV_SIMD || CV_SIMD_SCALABLE)

static CV_ALWAYS_INLINE bool remapOffset32(int sy, int sx, size_t sstep, int& offset)
{
    const size_t uoff = (size_t)sy * sstep + (size_t)sx;
    if (uoff > (size_t)INT_MAX)
        return false;
    offset = (int)uoff;
    return true;
}

static CV_ALWAYS_INLINE void remapCorners(const short* S0, size_t sstep, const int* ofs,
                                v_float32& s0, v_float32& s1,
                                v_float32& s2, v_float32& s3)
{
    unsigned int CV_DECL_ALIGNED(CV_SIMD_WIDTH) topbuf[VTraits<v_float32>::max_nlanes];
    unsigned int CV_DECL_ALIGNED(CV_SIMD_WIDTH) botbuf[VTraits<v_float32>::max_nlanes];
    const int n = VTraits<v_float32>::vlanes();
    for (int k = 0; k < n; k++)
    {
        const short* p = S0 + ofs[k];
        topbuf[k] = (unsigned int)(ushort)p[0] | ((unsigned int)(ushort)p[1] << 16);
        botbuf[k] = (unsigned int)(ushort)p[sstep] | ((unsigned int)(ushort)p[sstep + 1] << 16);
    }
    v_int32 top = v_reinterpret_as_s32(vx_load(topbuf));
    v_int32 bot = v_reinterpret_as_s32(vx_load(botbuf));
    s0 = v_cvt_f32(v_shr<16>(v_shl<16>(top)));   // low 16 bits (sign-extended)
    s1 = v_cvt_f32(v_shr<16>(top));              // high 16 bits (sign-extended)
    s2 = v_cvt_f32(v_shr<16>(v_shl<16>(bot)));
    s3 = v_cvt_f32(v_shr<16>(bot));
}

static CV_ALWAYS_INLINE void remapCorners(const ushort* S0, size_t sstep, const int* ofs,
                                v_float32& s0, v_float32& s1,
                                v_float32& s2, v_float32& s3)
{
    unsigned int CV_DECL_ALIGNED(CV_SIMD_WIDTH) topbuf[VTraits<v_float32>::max_nlanes];
    unsigned int CV_DECL_ALIGNED(CV_SIMD_WIDTH) botbuf[VTraits<v_float32>::max_nlanes];
    const int n = VTraits<v_float32>::vlanes();
    for (int k = 0; k < n; k++)
    {
        const ushort* p = S0 + ofs[k];
        topbuf[k] = (unsigned int)p[0] | ((unsigned int)p[1] << 16);
        botbuf[k] = (unsigned int)p[sstep] | ((unsigned int)p[sstep + 1] << 16);
    }
    const v_uint32 lo16 = vx_setall_u32(0xffff);
    v_uint32 top = vx_load(topbuf);
    v_uint32 bot = vx_load(botbuf);
    s0 = v_cvt_f32(v_reinterpret_as_s32(v_and(top, lo16)));  // low 16 (zero-ext)
    s1 = v_cvt_f32(v_reinterpret_as_s32(v_shr<16>(top)));    // high 16 (zero-ext)
    s2 = v_cvt_f32(v_reinterpret_as_s32(v_and(bot, lo16)));
    s3 = v_cvt_f32(v_reinterpret_as_s32(v_shr<16>(bot)));
}

static CV_ALWAYS_INLINE void remapCorners(const float* S0, size_t sstep, const int* ofs,
                                          v_float32& s0, v_float32& s1,
                                          v_float32& s2, v_float32& s3)
{
    const float* S1 = S0 + sstep;
    const v_int32 vofs = vx_load(ofs);
    v_lut_deinterleave(S0, vofs, s0, s1);
    v_lut_deinterleave(S1, vofs, s2, s3);
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
    for( ; dx < X1 && FXY[dx] == 0; dx++ )
    {
        const int sx = XY[dx * 2];
        const int sy = XY[dx * 2 + 1] + off_y;
        D[dx - dx0] = S0[(size_t)sy * sstep + sx];
    }

    for( ; dx <= X1 - vlanes; dx += vlanes )
    {
        bool zeroFrac = true;
        for( int k = 0; k < vlanes; k++ )
            zeroFrac = zeroFrac && FXY[dx + k] == 0;
        if( zeroFrac )
        {
            for( int k = 0; k < vlanes; k++ )
            {
                const int sx = XY[(dx + k) * 2];
                const int sy = XY[(dx + k) * 2 + 1] + off_y;
                D[dx - dx0 + k] = S0[(size_t)sy * sstep + sx];
            }
            continue;
        }
        bool offsetsFit = true;
        for( int k = 0; k < vlanes; k++ )
        {
            const int sx = XY[(dx + k) * 2];
            const int sy = XY[(dx + k) * 2 + 1] + off_y;
            offsetsFit = offsetsFit && remapOffset32(sy, sx, sstep, ofs[k]);
        }
        if( !offsetsFit )
            break;
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

static inline void remapStoreScalar(ushort* D, float v) { *D = saturate_cast<ushort>(v); }
static inline void remapStoreScalar(short* D, float v) { *D = saturate_cast<short>(v); }

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
        if( FXY[dx] == 0 )
        {
            D[dx - dx0] = S0[(size_t)(sy + 3) * sstep + (sx + 3)];
            continue;
        }
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
        return remapBilinearC1_run<float>((const float*)S0, sstep, (float*)D,
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

RemapBilinearC1Fn getRemapBilinearC1Fn_()
{
    return remapBilinearC1_simd;
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

RemapWideC1Fn getRemapLanczos4C1Fn_()
{
    return remapLanczos4C1_simd;
}

#endif // CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

CV_CPU_OPTIMIZATION_NAMESPACE_END

} // namespace cv
