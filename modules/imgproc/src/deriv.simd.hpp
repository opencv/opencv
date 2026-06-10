// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
// Copyright (C) 2026, Advanced Micro Devices, Inc., all rights reserved.

#include "precomp.hpp"
#include "opencv2/core/hal/intrin.hpp"

namespace cv {
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

void Sobel3x3(const uchar* src, size_t src_step, int srcRows, int srcCols,
              int rowStart, int rowEnd,
              short* dx, short* dy, size_t dst_step, int borderType);

void Sobel5x5(const uchar* src, size_t src_step, int srcRows, int srcCols,
              int rowStart, int rowEnd,
              short* dx, short* dy, size_t dst_step, int borderType);

void Sobel3x3f(const uchar* src, size_t src_step, int srcRows, int srcCols,
               int rowStart, int rowEnd,
               float* dx, float* dy, size_t dst_step, float scale, int borderType);

void Sobel5x5f(const uchar* src, size_t src_step, int srcRows, int srcCols,
               int rowStart, int rowEnd,
               float* dx, float* dy, size_t dst_step, float scale, int borderType);

#ifndef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

// Output store for the fused kernels.
//   - short: store the int16 result (scale is always 1 here).
//   - float: widen to int32, convert to float, multiply by scale, store.
template<typename DstT> struct SobelStore;

template<> struct SobelStore<short>
{
    explicit SobelStore(double) {}
#if (CV_SIMD || CV_SIMD_SCALABLE)
    inline void store(short* p, int j, const v_int16& v) const { v_store(p + j, v); }
#endif
    inline void scalar(short* p, int j, int v) const { p[j] = (short)v; }
};

template<> struct SobelStore<float>
{
    float s;
    explicit SobelStore(double scale) : s((float)scale) {}
#if (CV_SIMD || CV_SIMD_SCALABLE)
    inline void store(float* p, int j, const v_int16& v) const
    {
        v_int32 lo, hi;
        v_expand(v, lo, hi);
        v_float32 vs = vx_setall_f32(s);
        v_store(p + j, v_mul(v_cvt_f32(lo), vs));
        v_store(p + j + VTraits<v_float32>::vlanes(), v_mul(v_cvt_f32(hi), vs));
    }
#endif
    inline void scalar(float* p, int j, int v) const { p[j] = s * (float)v; }
};

// CV_32F output must match cv::Sobel(..., CV_32F, ..., scale, ...), which folds scale
// into the smoothing kernel (ky for dx, kx for dy) before sepFilter2D — not as a final
// post-multiply. dx column: odx = (2*scale)*hd0 + scale*(hdm+hdp); dy row: hs already
// carries scale, then ody = hsp - hsm.
static void Sobel3x3_f32(const uchar* src, size_t src_step, int srcRows, int srcCols,
                         int rowStart, int rowEnd,
                         float* dx, float* dy, size_t dst_step, float scale, int borderType)
{
    const int width = srcCols;
    if (width <= 0 || rowEnd <= rowStart)
        return;

    const float sk0 = 2.f * scale; // scaled [1,2,1] row kernel center tap
    const float sk1 = scale;       // scaled [1,2,1] row kernel side taps
    const float dk0 = sk0;         // scaled [1,2,1] column kernel center tap (dx)
    const float dk1 = sk1;         // scaled [1,2,1] column kernel side taps (dx)

    cv::AutoBuffer<float> _buf((size_t)width * 6);
    float* hd[3];
    float* hs[3];
    for (int k = 0; k < 3; ++k)
    {
        hd[k] = _buf.data() + (size_t)k * 2 * width;
        hs[k] = hd[k] + width;
    }

    auto cidx = [&](int x) { return borderInterpolate(x, width, borderType); };

    auto horiz = [&](const uchar* s, float* hdo, float* hso)
    {
        {
            float l = (float)s[cidx(-1)], c = (float)s[0], r = (float)s[cidx(1)];
            hdo[0] = r - l;
            hso[0] = sk0 * c + sk1 * (l + r);
        }

        int j = 1;
#if (CV_SIMD || CV_SIMD_SCALABLE)
        v_float32 v_sk0 = vx_setall_f32(sk0);
        v_float32 v_sk1 = vx_setall_f32(sk1);
        for (; j + VTraits<v_float32>::vlanes() <= width - 1; j += VTraits<v_float32>::vlanes())
        {
            v_float32 l = v_cvt_f32(v_reinterpret_as_s32(vx_load_expand_q(s + j - 1)));
            v_float32 c = v_cvt_f32(v_reinterpret_as_s32(vx_load_expand_q(s + j)));
            v_float32 r = v_cvt_f32(v_reinterpret_as_s32(vx_load_expand_q(s + j + 1)));
            v_store(hdo + j, v_sub(r, l));
            v_store(hso + j, v_muladd(v_add(l, r), v_sk1, v_mul(c, v_sk0)));
        }
#endif
        for (; j < width - 1; ++j)
        {
            float l = (float)s[j - 1], c = (float)s[j], r = (float)s[j + 1];
            hdo[j] = r - l;
            hso[j] = sk0 * c + sk1 * (l + r);
        }

        if (width > 1)
        {
            int x = width - 1;
            float l = (float)s[x - 1], c = (float)s[x], r = (float)s[cidx(x + 1)];
            hdo[x] = r - l;
            hso[x] = sk0 * c + sk1 * (l + r);
        }
    };

    auto combine = [&](const float* hdm, const float* hd0, const float* hdp,
                       const float* hsm, const float* hsp,
                       float* odx, float* ody)
    {
        int j = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
        v_float32 v_dk0 = vx_setall_f32(dk0);
        v_float32 v_dk1 = vx_setall_f32(dk1);
        for (; j <= width - VTraits<v_float32>::vlanes(); j += VTraits<v_float32>::vlanes())
        {
            v_float32 hd0v = vx_load(hd0 + j);
            v_store(odx + j, v_muladd(v_add(vx_load(hdm + j), vx_load(hdp + j)), v_dk1, v_mul(hd0v, v_dk0)));
            v_store(ody + j, v_sub(vx_load(hsp + j), vx_load(hsm + j)));
        }
#endif
        for (; j < width; ++j)
        {
            odx[j] = dk0 * hd0[j] + dk1 * (hdm[j] + hdp[j]);
            ody[j] = hsp[j] - hsm[j];
        }
    };

    for (int r = rowStart - 1; r <= rowEnd; ++r)
    {
        int rc = borderInterpolate(r, srcRows, borderType);
        int slot = ((r % 3) + 3) % 3;
        horiz(src + (size_t)rc * src_step, hd[slot], hs[slot]);

        int a = r - 1;
        if (a >= rowStart && a < rowEnd)
        {
            int sm = (((a - 1) % 3) + 3) % 3;
            int s0 = ((a % 3) + 3) % 3;
            int sp = (((a + 1) % 3) + 3) % 3;
            combine(hd[sm], hd[s0], hd[sp], hs[sm], hs[sp],
                    dx + (size_t)(a - rowStart) * dst_step,
                    dy + (size_t)(a - rowStart) * dst_step);
        }
    }
}

static void Sobel5x5_f32(const uchar* src, size_t src_step, int srcRows, int srcCols,
                         int rowStart, int rowEnd,
                         float* dx, float* dy, size_t dst_step, float scale, int borderType)
{
    const int width = srcCols;
    if (width <= 0 || rowEnd <= rowStart)
        return;

    const float sk0 = 6.f * scale;
    const float sk1 = 4.f * scale;
    const float sk2 = scale;
    const float dk0 = sk0;
    const float dk1 = sk1;
    const float dk2 = sk2;

    cv::AutoBuffer<float> _buf((size_t)width * 10);
    float* hd[5];
    float* hs[5];
    for (int k = 0; k < 5; ++k)
    {
        hd[k] = _buf.data() + (size_t)k * 2 * width;
        hs[k] = hd[k] + width;
    }

    auto cidx = [&](int x) { return borderInterpolate(x, width, borderType); };

    auto horiz = [&](const uchar* s, float* hdo, float* hso)
    {
        auto scalarCol = [&](int x)
        {
            float m2 = (float)s[cidx(x - 2)], m1 = (float)s[cidx(x - 1)], c = (float)s[x];
            float p1 = (float)s[cidx(x + 1)], p2 = (float)s[cidx(x + 2)];
            hdo[x] = p2 + 2.f * p1 - 2.f * m1 - m2;
            hso[x] = sk2 * (m2 + p2) + sk1 * (m1 + p1) + sk0 * c;
        };

        int j = 0;
        for (; j < 2 && j < width; ++j)
            scalarCol(j);
#if (CV_SIMD || CV_SIMD_SCALABLE)
        v_float32 v_sk0 = vx_setall_f32(sk0);
        v_float32 v_sk1 = vx_setall_f32(sk1);
        v_float32 v_sk2 = vx_setall_f32(sk2);
        for (; j + VTraits<v_float32>::vlanes() <= width - 2; j += VTraits<v_float32>::vlanes())
        {
            v_float32 m2 = v_cvt_f32(v_reinterpret_as_s32(vx_load_expand_q(s + j - 2)));
            v_float32 m1 = v_cvt_f32(v_reinterpret_as_s32(vx_load_expand_q(s + j - 1)));
            v_float32 c  = v_cvt_f32(v_reinterpret_as_s32(vx_load_expand_q(s + j)));
            v_float32 p1 = v_cvt_f32(v_reinterpret_as_s32(vx_load_expand_q(s + j + 1)));
            v_float32 p2 = v_cvt_f32(v_reinterpret_as_s32(vx_load_expand_q(s + j + 2)));
            v_store(hdo + j, v_sub(v_sub(v_add(p2, v_add(p1, p1)), v_add(m1, m1)), m2));
            v_store(hso + j, v_muladd(v_add(m2, p2), v_sk2,
                                      v_muladd(v_add(m1, p1), v_sk1, v_mul(c, v_sk0))));
        }
#endif
        for (; j < width; ++j)
            scalarCol(j);
    };

    auto combine = [&](const float* hdm2, const float* hdm1, const float* hd0,
                       const float* hdp1, const float* hdp2,
                       const float* hsm2, const float* hsm1,
                       const float* hsp1, const float* hsp2,
                       float* odx, float* ody)
    {
        int j = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
        v_float32 v_dk0 = vx_setall_f32(dk0);
        v_float32 v_dk1 = vx_setall_f32(dk1);
        v_float32 v_dk2 = vx_setall_f32(dk2);
        v_float32 v2 = vx_setall_f32(2.f);
        for (; j <= width - VTraits<v_float32>::vlanes(); j += VTraits<v_float32>::vlanes())
        {
            v_float32 hd0v = vx_load(hd0 + j);
            v_store(odx + j, v_muladd(v_add(vx_load(hdm2 + j), vx_load(hdp2 + j)), v_dk2,
                                      v_muladd(v_add(vx_load(hdm1 + j), vx_load(hdp1 + j)), v_dk1,
                                               v_mul(hd0v, v_dk0))));
            v_store(ody + j, v_add(v_sub(vx_load(hsp2 + j), vx_load(hsm2 + j)),
                                   v_mul(v2, v_sub(vx_load(hsp1 + j), vx_load(hsm1 + j)))));
        }
#endif
        for (; j < width; ++j)
        {
            odx[j] = dk0 * hd0[j] + dk1 * (hdm1[j] + hdp1[j]) + dk2 * (hdm2[j] + hdp2[j]);
            ody[j] = (hsp2[j] - hsm2[j]) + 2.f * (hsp1[j] - hsm1[j]);
        }
    };

    for (int r = rowStart - 2; r <= rowEnd + 1; ++r)
    {
        int rc = borderInterpolate(r, srcRows, borderType);
        int slot = ((r % 5) + 5) % 5;
        horiz(src + (size_t)rc * src_step, hd[slot], hs[slot]);

        int a = r - 2;
        if (a >= rowStart && a < rowEnd)
        {
            int s_m2 = (((a - 2) % 5) + 5) % 5;
            int s_m1 = (((a - 1) % 5) + 5) % 5;
            int s_0  = ((a % 5) + 5) % 5;
            int s_p1 = (((a + 1) % 5) + 5) % 5;
            int s_p2 = (((a + 2) % 5) + 5) % 5;
            combine(hd[s_m2], hd[s_m1], hd[s_0], hd[s_p1], hd[s_p2],
                    hs[s_m2], hs[s_m1], hs[s_p1], hs[s_p2],
                    dx + (size_t)(a - rowStart) * dst_step,
                    dy + (size_t)(a - rowStart) * dst_step);
        }
    }
}

// Fused 3x3 Sobel. A separable Sobel needs, per row, a horizontal difference
// (hd = src[x+1]-src[x-1]) for dx and a horizontal smoothing
// (hs = src[x-1]+2*src[x]+src[x+1]) for dy. We compute both with one read of the
// source row, cache them in a 3-row ring buffer, and combine vertically:
//   dx = hd[y-1] + 2*hd[y] + hd[y+1]
//   dy = hs[y+1] - hs[y-1]
template<typename DstT>
static void Sobel3x3_(const uchar* src, size_t src_step, int srcRows, int srcCols,
                      int rowStart, int rowEnd,
                      DstT* dx, DstT* dy, size_t dst_step, double scale, int borderType)
{
    const int width = srcCols;
    const int outRows = rowEnd - rowStart;
    if (width <= 0 || outRows <= 0)
        return;

    const SobelStore<DstT> t_store(scale);

    // 3-row ring buffer of horizontal results; slot for absolute row r is ((r%3)+3)%3.
    cv::AutoBuffer<short> _buf((size_t)width * 6);
    short* hd[3];
    short* hs[3];
    for (int k = 0; k < 3; ++k)
    {
        hd[k] = _buf.data() + (size_t)k * 2 * width;
        hs[k] = hd[k] + width;
    }

    auto cidx = [&](int x) { return borderInterpolate(x, width, borderType); };

    auto horiz = [&](const uchar* s, short* hdo, short* hso)
    {
        // left border column: neighbour x-1 is out of range.
        {
            int l = s[cidx(-1)], c = s[0], r = s[cidx(1)];
            hdo[0] = (short)(r - l);
            hso[0] = (short)(l + 2 * c + r);
        }

        int j = 1;
#if (CV_SIMD || CV_SIMD_SCALABLE)
        for (; j + VTraits<v_int16>::vlanes() <= width - 1; j += VTraits<v_int16>::vlanes())
        {
            v_int16 c = v_reinterpret_as_s16(vx_load_expand(s + j));
            v_int16 l = v_reinterpret_as_s16(vx_load_expand(s + j - 1));
            v_int16 r = v_reinterpret_as_s16(vx_load_expand(s + j + 1));
            v_store(hdo + j, v_sub(r, l));
            v_store(hso + j, v_add(v_add(l, r), v_add(c, c)));
        }
#endif
        for (; j < width - 1; ++j)
        {
            int l = s[j - 1], c = s[j], r = s[j + 1];
            hdo[j] = (short)(r - l);
            hso[j] = (short)(l + 2 * c + r);
        }

        // right border column: neighbour x+1 is out of range.
        if (width > 1)
        {
            int x = width - 1;
            int l = s[x - 1], c = s[x], r = s[cidx(x + 1)];
            hdo[x] = (short)(r - l);
            hso[x] = (short)(l + 2 * c + r);
        }
    };

    auto combine = [&](const short* hdm, const short* hd0, const short* hdp,
                       const short* hsm, const short* hsp,
                       DstT* odx, DstT* ody)
    {
        int j = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
        for (; j <= width - VTraits<v_int16>::vlanes(); j += VTraits<v_int16>::vlanes())
        {
            v_int16 a = vx_load(hdm + j);
            v_int16 b = vx_load(hd0 + j);
            v_int16 c = vx_load(hdp + j);
            t_store.store(odx, j, v_add(v_add(a, c), v_add(b, b)));

            v_int16 p = vx_load(hsp + j);
            v_int16 m = vx_load(hsm + j);
            t_store.store(ody, j, v_sub(p, m));
        }
#endif
        for (; j < width; ++j)
        {
            t_store.scalar(odx, j, hdm[j] + 2 * hd0[j] + hdp[j]);
            t_store.scalar(ody, j, hsp[j] - hsm[j]);
        }
    };

    // Slide a 3-row window over absolute source rows [rowStart-1, rowEnd].
    // After computing the horizontal pass for row r, output row a = r-1 can be
    // emitted from rows {a-1, a, a+1} == {r-2, r-1, r}.
    for (int r = rowStart - 1; r <= rowEnd; ++r)
    {
        int rc = borderInterpolate(r, srcRows, borderType);
        int slot = ((r % 3) + 3) % 3;
        horiz(src + (size_t)rc * src_step, hd[slot], hs[slot]);

        int a = r - 1;
        if (a >= rowStart && a < rowEnd)
        {
            int sm = (((a - 1) % 3) + 3) % 3;
            int s0 = ((a % 3) + 3) % 3;
            int sp = (((a + 1) % 3) + 3) % 3;
            combine(hd[sm], hd[s0], hd[sp], hs[sm], hs[sp],
                    dx + (size_t)(a - rowStart) * dst_step,
                    dy + (size_t)(a - rowStart) * dst_step);
        }
    }
}

// Fused 5x5 Sobel. Separable 5x5 Sobel kernels (cv::getSobelKernels, ksize=5):
//   smoothing  s5 = [1, 4, 6, 4, 1]
//   derivative d5 = [-1, -2, 0, 2, 1]
// Per row we compute, with one read of the source row:
//   hd = s[x+2] + 2*s[x+1] - 2*s[x-1] - s[x-2]   (horizontal derivative, for dx)
//   hs = s[x-2] + 4*s[x-1] + 6*s[x] + 4*s[x+1] + s[x+2]  (horizontal smoothing, for dy)
// cache hd/hs in a 5-row ring buffer and combine vertically:
//   dx = hd[y-2] + 4*hd[y-1] + 6*hd[y] + 4*hd[y+1] + hd[y+2]
//   dy = -hs[y-2] - 2*hs[y-1] + 2*hs[y+1] + hs[y+2]
template<typename DstT>
static void Sobel5x5_(const uchar* src, size_t src_step, int srcRows, int srcCols,
                      int rowStart, int rowEnd,
                      DstT* dx, DstT* dy, size_t dst_step, double scale, int borderType)
{
    const int width = srcCols;
    const int outRows = rowEnd - rowStart;
    if (width <= 0 || outRows <= 0)
        return;

    const SobelStore<DstT> t_store(scale);

    // 5-row ring buffer of horizontal results; slot for absolute row r is ((r%5)+5)%5.
    cv::AutoBuffer<short> _buf((size_t)width * 10);
    short* hd[5];
    short* hs[5];
    for (int k = 0; k < 5; ++k)
    {
        hd[k] = _buf.data() + (size_t)k * 2 * width;
        hs[k] = hd[k] + width;
    }

    auto cidx = [&](int x) { return borderInterpolate(x, width, borderType); };

    auto horiz = [&](const uchar* s, short* hdo, short* hso)
    {
        auto scalarCol = [&](int x)
        {
            int m2 = s[cidx(x - 2)], m1 = s[cidx(x - 1)], c = s[x];
            int p1 = s[cidx(x + 1)], p2 = s[cidx(x + 2)];
            hdo[x] = (short)(p2 + 2 * p1 - 2 * m1 - m2);
            hso[x] = (short)(m2 + 4 * m1 + 6 * c + 4 * p1 + p2);
        };

        int j = 0;
        for (; j < 2 && j < width; ++j)        // left border columns
            scalarCol(j);
#if (CV_SIMD || CV_SIMD_SCALABLE)
        for (; j + VTraits<v_int16>::vlanes() <= width - 2; j += VTraits<v_int16>::vlanes())
        {
            v_int16 m2 = v_reinterpret_as_s16(vx_load_expand(s + j - 2));
            v_int16 m1 = v_reinterpret_as_s16(vx_load_expand(s + j - 1));
            v_int16 c  = v_reinterpret_as_s16(vx_load_expand(s + j));
            v_int16 p1 = v_reinterpret_as_s16(vx_load_expand(s + j + 1));
            v_int16 p2 = v_reinterpret_as_s16(vx_load_expand(s + j + 2));
            // hd = p2 + 2*p1 - 2*m1 - m2
            v_store(hdo + j, v_sub(v_sub(v_add(p2, v_add(p1, p1)), v_add(m1, m1)), m2));
            // hs = (m2 + p2) + 4*(m1 + p1) + 6*c
            v_int16 c6 = v_add(v_shl<2>(c), v_add(c, c));
            v_store(hso + j, v_add(v_add(v_add(m2, p2), v_shl<2>(v_add(m1, p1))), c6));
        }
#endif
        for (; j < width; ++j)                  // interior tail + right border
            scalarCol(j);
    };

    auto combine = [&](const short* hdm2, const short* hdm1, const short* hd0,
                       const short* hdp1, const short* hdp2,
                       const short* hsm2, const short* hsm1,
                       const short* hsp1, const short* hsp2,
                       DstT* odx, DstT* ody)
    {
        int j = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
        for (; j <= width - VTraits<v_int16>::vlanes(); j += VTraits<v_int16>::vlanes())
        {
            v_int16 a = vx_load(hdm2 + j), b = vx_load(hdm1 + j), c = vx_load(hd0 + j);
            v_int16 d = vx_load(hdp1 + j), e = vx_load(hdp2 + j);
            // dx = (a + e) + 4*(b + d) + 6*c
            v_int16 c6 = v_add(v_shl<2>(c), v_add(c, c));
            t_store.store(odx, j, v_add(v_add(v_add(a, e), v_shl<2>(v_add(b, d))), c6));

            v_int16 g = vx_load(hsm2 + j), h = vx_load(hsm1 + j);
            v_int16 p = vx_load(hsp1 + j), q = vx_load(hsp2 + j);
            // dy = (q - g) + 2*(p - h)
            t_store.store(ody, j, v_add(v_sub(q, g), v_shl<1>(v_sub(p, h))));
        }
#endif
        for (; j < width; ++j)
        {
            t_store.scalar(odx, j, hdm2[j] + 4 * hdm1[j] + 6 * hd0[j] + 4 * hdp1[j] + hdp2[j]);
            t_store.scalar(ody, j, -hsm2[j] - 2 * hsm1[j] + 2 * hsp1[j] + hsp2[j]);
        }
    };

    // Slide a 5-row window over absolute source rows [rowStart-2, rowEnd+1].
    // After the horizontal pass for row r, output row a = r-2 is emitted from
    // rows {a-2, a-1, a, a+1, a+2} == {r-4, r-3, r-2, r-1, r}.
    for (int r = rowStart - 2; r <= rowEnd + 1; ++r)
    {
        int rc = borderInterpolate(r, srcRows, borderType);
        int slot = ((r % 5) + 5) % 5;
        horiz(src + (size_t)rc * src_step, hd[slot], hs[slot]);

        int a = r - 2;
        if (a >= rowStart && a < rowEnd)
        {
            int s_m2 = (((a - 2) % 5) + 5) % 5;
            int s_m1 = (((a - 1) % 5) + 5) % 5;
            int s_0  = ((a % 5) + 5) % 5;
            int s_p1 = (((a + 1) % 5) + 5) % 5;
            int s_p2 = (((a + 2) % 5) + 5) % 5;
            combine(hd[s_m2], hd[s_m1], hd[s_0], hd[s_p1], hd[s_p2],
                    hs[s_m2], hs[s_m1], hs[s_p1], hs[s_p2],
                    dx + (size_t)(a - rowStart) * dst_step,
                    dy + (size_t)(a - rowStart) * dst_step);
        }
    }
}

void Sobel3x3(const uchar* src, size_t src_step, int srcRows, int srcCols,
              int rowStart, int rowEnd,
              short* dx, short* dy, size_t dst_step, int borderType)
{
    Sobel3x3_<short>(src, src_step, srcRows, srcCols, rowStart, rowEnd,
                     dx, dy, dst_step, 1.0, borderType);
}

void Sobel5x5(const uchar* src, size_t src_step, int srcRows, int srcCols,
              int rowStart, int rowEnd,
              short* dx, short* dy, size_t dst_step, int borderType)
{
    Sobel5x5_<short>(src, src_step, srcRows, srcCols, rowStart, rowEnd,
                     dx, dy, dst_step, 1.0, borderType);
}

void Sobel3x3f(const uchar* src, size_t src_step, int srcRows, int srcCols,
               int rowStart, int rowEnd,
               float* dx, float* dy, size_t dst_step, float scale, int borderType)
{
    Sobel3x3_f32(src, src_step, srcRows, srcCols, rowStart, rowEnd,
                 dx, dy, dst_step, scale, borderType);
}

void Sobel5x5f(const uchar* src, size_t src_step, int srcRows, int srcCols,
               int rowStart, int rowEnd,
               float* dx, float* dy, size_t dst_step, float scale, int borderType)
{
    Sobel5x5_f32(src, src_step, srcRows, srcCols, rowStart, rowEnd,
                 dx, dy, dst_step, scale, borderType);
}

#endif // CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

CV_CPU_OPTIMIZATION_NAMESPACE_END
} // namespace cv
