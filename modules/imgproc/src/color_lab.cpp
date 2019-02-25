// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

/********************************* COPYRIGHT NOTICE *******************************\
  The function for RGB to Lab conversion is based on the MATLAB script
  RGB2Lab.m translated by Mark Ruzon from C code by Yossi Rubner, 23 September 1997.
  See the page [http://vision.stanford.edu/~ruzon/software/rgblab.html]
\**********************************************************************************/

#include "precomp.hpp"
#include "opencl_kernels_imgproc.hpp"
#include "opencv2/core/hal/intrin.hpp"
#include "opencv2/core/softfloat.hpp"

#include "color.hpp"

using cv::softfloat;

static const float * splineBuild(const softfloat* f, size_t n)
{
    float* tab = cv::allocSingleton<float>(n * 4);
    const softfloat f2(2), f3(3), f4(4);
    softfloat cn(0);
    softfloat* sftab = reinterpret_cast<softfloat*>(tab);
    tab[0] = tab[1] = 0.0f;

    for(size_t i = 1; i < n; i++)
    {
        softfloat t = (f[i+1] - f[i]*f2 + f[i-1])*f3;
        softfloat l = softfloat::one()/(f4 - sftab[(i-1)*4]);
        sftab[i*4] = l; sftab[i*4+1] = (t - sftab[(i-1)*4+1])*l;
    }

    for(size_t j = 0; j < n; ++j)
    {
        size_t i = n - j - 1;
        softfloat c = sftab[i*4+1] - sftab[i*4]*cn;
        softfloat b = f[i+1] - f[i] - (cn + c*f2)/f3;
        softfloat d = (cn - c)/f3;
        sftab[i*4] = f[i]; sftab[i*4+1] = b;
        sftab[i*4+2] = c; sftab[i*4+3] = d;
        cn = c;
    }
    return tab;
}


// interpolates value of a function at x, 0 <= x <= n using a cubic spline.
template<typename _Tp> static inline _Tp splineInterpolate(_Tp x, const _Tp* tab, int n)
{
    // don't touch this function without urgent need - some versions of gcc fail to inline it correctly
    int ix = std::min(std::max(int(x), 0), n-1);
    x -= ix;
    tab += ix*4;
    return ((tab[3]*x + tab[2])*x + tab[1])*x + tab[0];
}

#if CV_SIMD
template<typename _Tp> static inline cv::v_float32 splineInterpolate(cv::v_float32 x, const _Tp* tab, int n)
{
    using namespace cv;
    v_int32 ix = v_min(v_max(v_trunc(x), vx_setzero_s32()), vx_setall_s32(n-1));
    x -= v_cvt_f32(ix);
    ix = ix << 2;

    v_float32 t[4];
    for(int i = 0; i < 4; i++)
        t[i] = v_lut(tab + i, ix);

    return v_fma(v_fma(v_fma(t[3], x, t[2]), x, t[1]), x, t[0]);
}
#endif

#if CV_NEON
template<typename _Tp> static inline void splineInterpolate(float32x4_t& v_x, const _Tp* tab, int n)
{
    int32x4_t v_ix = vcvtq_s32_f32(vminq_f32(vmaxq_f32(v_x, vdupq_n_f32(0)), vdupq_n_f32(n - 1)));
    v_x = vsubq_f32(v_x, vcvtq_f32_s32(v_ix));
    v_ix = vshlq_n_s32(v_ix, 2);

    int CV_DECL_ALIGNED(16) ix[4];
    vst1q_s32(ix, v_ix);

    float32x4_t v_tab0 = vld1q_f32(tab + ix[0]);
    float32x4_t v_tab1 = vld1q_f32(tab + ix[1]);
    float32x4_t v_tab2 = vld1q_f32(tab + ix[2]);
    float32x4_t v_tab3 = vld1q_f32(tab + ix[3]);

    float32x4x2_t v01 = vtrnq_f32(v_tab0, v_tab1);
    float32x4x2_t v23 = vtrnq_f32(v_tab2, v_tab3);

    v_tab0 = vcombine_f32(vget_low_f32(v01.val[0]), vget_low_f32(v23.val[0]));
    v_tab1 = vcombine_f32(vget_low_f32(v01.val[1]), vget_low_f32(v23.val[1]));
    v_tab2 = vcombine_f32(vget_high_f32(v01.val[0]), vget_high_f32(v23.val[0]));
    v_tab3 = vcombine_f32(vget_high_f32(v01.val[1]), vget_high_f32(v23.val[1]));

    v_x = vmlaq_f32(v_tab0, vmlaq_f32(v_tab1, vmlaq_f32(v_tab2, v_tab3, v_x), v_x), v_x);
}
#elif CV_SSE2
template<typename _Tp> static inline void splineInterpolate(__m128& v_x, const _Tp* tab, int n)
{
    __m128i v_ix = _mm_cvttps_epi32(_mm_min_ps(_mm_max_ps(v_x, _mm_setzero_ps()), _mm_set1_ps(float(n - 1))));
    v_x = _mm_sub_ps(v_x, _mm_cvtepi32_ps(v_ix));
    v_ix = _mm_slli_epi32(v_ix, 2);

    int CV_DECL_ALIGNED(16) ix[4];
    _mm_store_si128((__m128i *)ix, v_ix);

    __m128 v_tab0 = _mm_loadu_ps(tab + ix[0]);
    __m128 v_tab1 = _mm_loadu_ps(tab + ix[1]);
    __m128 v_tab2 = _mm_loadu_ps(tab + ix[2]);
    __m128 v_tab3 = _mm_loadu_ps(tab + ix[3]);

    __m128 v_tmp0 = _mm_unpacklo_ps(v_tab0, v_tab1);
    __m128 v_tmp1 = _mm_unpacklo_ps(v_tab2, v_tab3);
    __m128 v_tmp2 = _mm_unpackhi_ps(v_tab0, v_tab1);
    __m128 v_tmp3 = _mm_unpackhi_ps(v_tab2, v_tab3);

    v_tab0 = _mm_shuffle_ps(v_tmp0, v_tmp1, 0x44);
    v_tab2 = _mm_shuffle_ps(v_tmp2, v_tmp3, 0x44);
    v_tab1 = _mm_shuffle_ps(v_tmp0, v_tmp1, 0xee);
    v_tab3 = _mm_shuffle_ps(v_tmp2, v_tmp3, 0xee);

    __m128 v_l = _mm_mul_ps(v_x, v_tab3);
    v_l = _mm_add_ps(v_l, v_tab2);
    v_l = _mm_mul_ps(v_l, v_x);
    v_l = _mm_add_ps(v_l, v_tab1);
    v_l = _mm_mul_ps(v_l, v_x);
    v_x = _mm_add_ps(v_l, v_tab0);
}
#endif

namespace cv
{

////////////////////////////////////// RGB <-> XYZ ///////////////////////////////////////

// 0.412453, 0.357580, 0.180423,
// 0.212671, 0.715160, 0.072169,
// 0.019334, 0.119193, 0.950227
static const softdouble sRGB2XYZ_D65[] =
{
    softdouble::fromRaw(0x3fda65a14488c60d),
    softdouble::fromRaw(0x3fd6e297396d0918),
    softdouble::fromRaw(0x3fc71819d2391d58),
    softdouble::fromRaw(0x3fcb38cda6e75ff6),
    softdouble::fromRaw(0x3fe6e297396d0918),
    softdouble::fromRaw(0x3fb279aae6c8f755),
    softdouble::fromRaw(0x3f93cc4ac6cdaf4b),
    softdouble::fromRaw(0x3fbe836eb4e98138),
    softdouble::fromRaw(0x3fee68427418d691)
};

//  3.240479, -1.53715, -0.498535,
// -0.969256, 1.875991, 0.041556,
//  0.055648, -0.204043, 1.057311
static const softdouble XYZ2sRGB_D65[] =
{
    softdouble::fromRaw(0x4009ec804102ff8f),
    softdouble::fromRaw(0xbff8982a9930be0e),
    softdouble::fromRaw(0xbfdfe7ff583a53b9),
    softdouble::fromRaw(0xbfef042528ae74f3),
    softdouble::fromRaw(0x3ffe040f23897204),
    softdouble::fromRaw(0x3fa546d3f9e7b80b),
    softdouble::fromRaw(0x3fac7de5082cf52c),
    softdouble::fromRaw(0xbfca1e14bdfd2631),
    softdouble::fromRaw(0x3ff0eabef06b3786)
};

static const int sRGB2XYZ_D65_i[] =
{
    1689,    1465,    739,
    871,     2929,    296,
    79,      488,     3892
};

static const int XYZ2sRGB_D65_i[] =
{
    13273,  -6296,  -2042,
    -3970,   7684,    170,
      228,   -836,   4331
};

template<typename _Tp> struct RGB2XYZ_f
{
    typedef _Tp channel_type;

    RGB2XYZ_f(int _srccn, int blueIdx, const float* _coeffs) : srccn(_srccn)
    {
        for(int i = 0; i < 9; i++)
            coeffs[i] = _coeffs ? _coeffs[i] : (float)sRGB2XYZ_D65[i];
        if(blueIdx == 0)
        {
            std::swap(coeffs[0], coeffs[2]);
            std::swap(coeffs[3], coeffs[5]);
            std::swap(coeffs[6], coeffs[8]);
        }
    }
    void operator()(const _Tp* src, _Tp* dst, int n) const
    {
        int scn = srccn;
        float C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2],
              C3 = coeffs[3], C4 = coeffs[4], C5 = coeffs[5],
              C6 = coeffs[6], C7 = coeffs[7], C8 = coeffs[8];

        n *= 3;
        for(int i = 0; i < n; i += 3, src += scn)
        {
            _Tp X = saturate_cast<_Tp>(src[0]*C0 + src[1]*C1 + src[2]*C2);
            _Tp Y = saturate_cast<_Tp>(src[0]*C3 + src[1]*C4 + src[2]*C5);
            _Tp Z = saturate_cast<_Tp>(src[0]*C6 + src[1]*C7 + src[2]*C8);
            dst[i] = X; dst[i+1] = Y; dst[i+2] = Z;
        }
    }
    int srccn;
    float coeffs[9];
};


template <>
struct RGB2XYZ_f<float>
{
    typedef float channel_type;

    RGB2XYZ_f(int _srccn, int blueIdx, const float* _coeffs) : srccn(_srccn)
    {
        for(int i = 0; i < 9; i++)
            coeffs[i] = _coeffs ? _coeffs[i] : (float)sRGB2XYZ_D65[i];
        if(blueIdx == 0)
        {
            std::swap(coeffs[0], coeffs[2]);
            std::swap(coeffs[3], coeffs[5]);
            std::swap(coeffs[6], coeffs[8]);
        }
    }

    void operator()(const float* src, float* dst, int n) const
    {
        int scn = srccn;
        float C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2],
              C3 = coeffs[3], C4 = coeffs[4], C5 = coeffs[5],
              C6 = coeffs[6], C7 = coeffs[7], C8 = coeffs[8];
        int i = 0;
#if CV_SIMD
        const int vsize = v_float32::nlanes;
        v_float32 vc0 = vx_setall_f32(C0), vc1 = vx_setall_f32(C1), vc2 = vx_setall_f32(C2);
        v_float32 vc3 = vx_setall_f32(C3), vc4 = vx_setall_f32(C4), vc5 = vx_setall_f32(C5);
        v_float32 vc6 = vx_setall_f32(C6), vc7 = vx_setall_f32(C7), vc8 = vx_setall_f32(C8);
        for( ; i <= n-vsize;
             i += vsize, src += scn*vsize, dst += vsize*3)
        {
            v_float32 b, g, r, a;
            if(scn == 4)
            {
                v_load_deinterleave(src, b, g, r, a);
            }
            else // scn == 3
            {
                v_load_deinterleave(src, b, g, r);
            }

            v_float32 x, y, z;
            x = v_fma(b, vc0, v_fma(g, vc1, r*vc2));
            y = v_fma(b, vc3, v_fma(g, vc4, r*vc5));
            z = v_fma(b, vc6, v_fma(g, vc7, r*vc8));

            v_store_interleave(dst, x, y, z);
        }
        vx_cleanup();
#endif
        for( ; i < n; i++, src += scn, dst += 3)
        {
            float b = src[0], g = src[1], r = src[2];

            float X = saturate_cast<float>(b*C0 + g*C1 + r*C2);
            float Y = saturate_cast<float>(b*C3 + g*C4 + r*C5);
            float Z = saturate_cast<float>(b*C6 + g*C7 + r*C8);

            dst[0] = X; dst[1] = Y; dst[2] = Z;
        }
    }

    int srccn;
    float coeffs[9];
};


template<typename _Tp> struct RGB2XYZ_i
{
    typedef _Tp channel_type;

    RGB2XYZ_i(int _srccn, int blueIdx, const float* _coeffs) : srccn(_srccn)
    {
        for( int i = 0; i < 9; i++ )
            coeffs[i] = _coeffs ? cvRound(_coeffs[i]*(1 << xyz_shift)) : sRGB2XYZ_D65_i[i];
        if(blueIdx == 0)
        {
            std::swap(coeffs[0], coeffs[2]);
            std::swap(coeffs[3], coeffs[5]);
            std::swap(coeffs[6], coeffs[8]);
        }
    }
    void operator()(const _Tp* src, _Tp* dst, int n) const
    {
        int scn = srccn;
        int C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2],
            C3 = coeffs[3], C4 = coeffs[4], C5 = coeffs[5],
            C6 = coeffs[6], C7 = coeffs[7], C8 = coeffs[8];
        n *= 3;

        for(int i = 0; i < n; i += 3, src += scn)
        {
            int X = CV_DESCALE(src[0]*C0 + src[1]*C1 + src[2]*C2, xyz_shift);
            int Y = CV_DESCALE(src[0]*C3 + src[1]*C4 + src[2]*C5, xyz_shift);
            int Z = CV_DESCALE(src[0]*C6 + src[1]*C7 + src[2]*C8, xyz_shift);
            dst[i] = saturate_cast<_Tp>(X); dst[i+1] = saturate_cast<_Tp>(Y);
            dst[i+2] = saturate_cast<_Tp>(Z);
        }
    }
    int srccn;
    int coeffs[9];
};


template <>
struct RGB2XYZ_i<uchar>
{
    typedef uchar channel_type;
    static const int shift = xyz_shift;

    RGB2XYZ_i(int _srccn, int blueIdx, const float* _coeffs) : srccn(_srccn)
    {
        for( int i = 0; i < 9; i++ )
            coeffs[i] = _coeffs ? cvRound(_coeffs[i]*(1 << shift)) : sRGB2XYZ_D65_i[i];
        if(blueIdx == 0)
        {
            std::swap(coeffs[0], coeffs[2]);
            std::swap(coeffs[3], coeffs[5]);
            std::swap(coeffs[6], coeffs[8]);
        }
    }
    void operator()(const uchar * src, uchar * dst, int n) const
    {
        int scn = srccn, i = 0;
        int C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2],
            C3 = coeffs[3], C4 = coeffs[4], C5 = coeffs[5],
            C6 = coeffs[6], C7 = coeffs[7], C8 = coeffs[8];

#if CV_SIMD
        const int vsize = v_uint8::nlanes;
        int descaleShift = 1 << (shift-1);
        v_int16 vdescale = vx_setall_s16(descaleShift);
        v_int16 cxbg, cxr1, cybg, cyr1, czbg, czr1;
        v_int16 dummy;
        v_zip(vx_setall_s16((short)C0), vx_setall_s16((short)C1), cxbg, dummy);
        v_zip(vx_setall_s16((short)C2), vx_setall_s16(        1), cxr1, dummy);
        v_zip(vx_setall_s16((short)C3), vx_setall_s16((short)C4), cybg, dummy);
        v_zip(vx_setall_s16((short)C5), vx_setall_s16(        1), cyr1, dummy);
        v_zip(vx_setall_s16((short)C6), vx_setall_s16((short)C7), czbg, dummy);
        v_zip(vx_setall_s16((short)C8), vx_setall_s16(        1), czr1, dummy);

        for( ; i <= n-vsize;
             i += vsize, src += scn*vsize, dst += 3*vsize)
        {
            v_uint8 b, g, r, a;
            if(scn == 4)
            {
                v_load_deinterleave(src, b, g, r, a);
            }
            else // scn == 3
            {
                v_load_deinterleave(src, b, g, r);
            }

            v_uint16 b0, b1, g0, g1, r0, r1;
            v_expand(b, b0, b1);
            v_expand(g, g0, g1);
            v_expand(r, r0, r1);

            v_int16 sb0, sb1, sg0, sg1, sr0, sr1;
            sr0 = v_reinterpret_as_s16(r0); sr1 = v_reinterpret_as_s16(r1);
            sg0 = v_reinterpret_as_s16(g0); sg1 = v_reinterpret_as_s16(g1);
            sb0 = v_reinterpret_as_s16(b0); sb1 = v_reinterpret_as_s16(b1);

            v_int16 bg[4], rd[4];
            v_zip(sb0, sg0, bg[0], bg[1]);
            v_zip(sb1, sg1, bg[2], bg[3]);
            v_zip(sr0, vdescale, rd[0], rd[1]);
            v_zip(sr1, vdescale, rd[2], rd[3]);

            v_uint32 vx[4], vy[4], vz[4];
            for(int j = 0; j < 4; j++)
            {
                vx[j] = v_reinterpret_as_u32(v_dotprod(bg[j], cxbg) + v_dotprod(rd[j], cxr1)) >> shift;
                vy[j] = v_reinterpret_as_u32(v_dotprod(bg[j], cybg) + v_dotprod(rd[j], cyr1)) >> shift;
                vz[j] = v_reinterpret_as_u32(v_dotprod(bg[j], czbg) + v_dotprod(rd[j], czr1)) >> shift;
            }

            v_uint16 x0, x1, y0, y1, z0, z1;
            x0 = v_pack(vx[0], vx[1]);
            x1 = v_pack(vx[2], vx[3]);
            y0 = v_pack(vy[0], vy[1]);
            y1 = v_pack(vy[2], vy[3]);
            z0 = v_pack(vz[0], vz[1]);
            z1 = v_pack(vz[2], vz[3]);

            v_uint8 x, y, z;
            x = v_pack(x0, x1);
            y = v_pack(y0, y1);
            z = v_pack(z0, z1);

            v_store_interleave(dst, x, y, z);
        }
        vx_cleanup();
#endif

        for ( ; i < n; i++, src += scn, dst += 3)
        {
            uchar b = src[0], g = src[1], r = src[2];

            int X = CV_DESCALE(b*C0 + g*C1 + r*C2, shift);
            int Y = CV_DESCALE(b*C3 + g*C4 + r*C5, shift);
            int Z = CV_DESCALE(b*C6 + g*C7 + r*C8, shift);
            dst[0] = saturate_cast<uchar>(X);
            dst[1] = saturate_cast<uchar>(Y);
            dst[2] = saturate_cast<uchar>(Z);
        }
    }

    int srccn, coeffs[9];
};


template <>
struct RGB2XYZ_i<ushort>
{
    typedef ushort channel_type;
    static const int shift = xyz_shift;
    static const int fix_shift = (int)(sizeof(short)*8 - shift);

    RGB2XYZ_i(int _srccn, int blueIdx, const float* _coeffs) : srccn(_srccn)
    {
        for( int i = 0; i < 9; i++ )
            coeffs[i] = _coeffs ? cvRound(_coeffs[i]*(1 << shift)) : sRGB2XYZ_D65_i[i];
        if(blueIdx == 0)
        {
            std::swap(coeffs[0], coeffs[2]);
            std::swap(coeffs[3], coeffs[5]);
            std::swap(coeffs[6], coeffs[8]);
        }
    }

    void operator()(const ushort * src, ushort * dst, int n) const
    {
        int scn = srccn, i = 0;
        int C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2],
            C3 = coeffs[3], C4 = coeffs[4], C5 = coeffs[5],
            C6 = coeffs[6], C7 = coeffs[7], C8 = coeffs[8];
#if CV_SIMD
        const int vsize = v_uint16::nlanes;
        const int descaleShift = 1 << (shift-1);
        v_int16 vdescale = vx_setall_s16(descaleShift);
        v_int16 vc0 = vx_setall_s16((short)C0), vc1 = vx_setall_s16((short)C1), vc2 = vx_setall_s16((short)C2);
        v_int16 vc3 = vx_setall_s16((short)C3), vc4 = vx_setall_s16((short)C4), vc5 = vx_setall_s16((short)C5);
        v_int16 vc6 = vx_setall_s16((short)C6), vc7 = vx_setall_s16((short)C7), vc8 = vx_setall_s16((short)C8);
        v_int16 zero = vx_setzero_s16(), one = vx_setall_s16(1);
        v_int16 cxbg, cxr1, cybg, cyr1, czbg, czr1;
        v_int16 dummy;
        v_zip(vc0, vc1, cxbg, dummy);
        v_zip(vc2, one, cxr1, dummy);
        v_zip(vc3, vc4, cybg, dummy);
        v_zip(vc5, one, cyr1, dummy);
        v_zip(vc6, vc7, czbg, dummy);
        v_zip(vc8, one, czr1, dummy);

        for (; i <= n-vsize;
             i += vsize, src += scn*vsize, dst += 3*vsize)
        {
            v_uint16 b, g, r, a;
            if(scn == 4)
            {
                v_load_deinterleave(src, b, g, r, a);
            }
            else // scn == 3
            {
                v_load_deinterleave(src, b, g, r);
            }

            v_int16 sb, sg, sr;
            sr = v_reinterpret_as_s16(r);
            sg = v_reinterpret_as_s16(g);
            sb = v_reinterpret_as_s16(b);

            // fixing 16bit signed multiplication
            v_int16 xmr, xmg, xmb;
            v_int16 ymr, ymg, ymb;
            v_int16 zmr, zmg, zmb;

            v_int16 mr = sr < zero, mg = sg < zero, mb = sb < zero;

            xmb = mb & vc0;
            xmg = mg & vc1;
            xmr = mr & vc2;
            ymb = mb & vc3;
            ymg = mg & vc4;
            ymr = mr & vc5;
            zmb = mb & vc6;
            zmg = mg & vc7;
            zmr = mr & vc8;

            v_int32 xfix0, xfix1, yfix0, yfix1, zfix0, zfix1;
            v_expand(xmr + xmg + xmb, xfix0, xfix1);
            v_expand(ymr + ymg + ymb, yfix0, yfix1);
            v_expand(zmr + zmg + zmb, zfix0, zfix1);

            xfix0 = xfix0 << 16;
            xfix1 = xfix1 << 16;
            yfix0 = yfix0 << 16;
            yfix1 = yfix1 << 16;
            zfix0 = zfix0 << 16;
            zfix1 = zfix1 << 16;

            v_int16 bg0, bg1, rd0, rd1;
            v_zip(sb, sg, bg0, bg1);
            v_zip(sr, vdescale, rd0, rd1);

            v_uint32 x0, x1, y0, y1, z0, z1;

            x0 = v_reinterpret_as_u32(v_dotprod(bg0, cxbg) + v_dotprod(rd0, cxr1) + xfix0) >> shift;
            x1 = v_reinterpret_as_u32(v_dotprod(bg1, cxbg) + v_dotprod(rd1, cxr1) + xfix1) >> shift;
            y0 = v_reinterpret_as_u32(v_dotprod(bg0, cybg) + v_dotprod(rd0, cyr1) + yfix0) >> shift;
            y1 = v_reinterpret_as_u32(v_dotprod(bg1, cybg) + v_dotprod(rd1, cyr1) + yfix1) >> shift;
            z0 = v_reinterpret_as_u32(v_dotprod(bg0, czbg) + v_dotprod(rd0, czr1) + zfix0) >> shift;
            z1 = v_reinterpret_as_u32(v_dotprod(bg1, czbg) + v_dotprod(rd1, czr1) + zfix1) >> shift;

            v_uint16 x, y, z;
            x = v_pack(x0, x1);
            y = v_pack(y0, y1);
            z = v_pack(z0, z1);

            v_store_interleave(dst, x, y, z);
        }

        vx_cleanup();
#endif
        for ( ; i < n; i++, src += scn, dst += 3)
        {
            ushort b = src[0], g = src[1], r = src[2];
            int X = CV_DESCALE(b*C0 + g*C1 + r*C2, shift);
            int Y = CV_DESCALE(b*C3 + g*C4 + r*C5, shift);
            int Z = CV_DESCALE(b*C6 + g*C7 + r*C8, shift);
            dst[0] = saturate_cast<ushort>(X);
            dst[1] = saturate_cast<ushort>(Y);
            dst[2] = saturate_cast<ushort>(Z);
        }
    }

    int srccn, coeffs[9];
};


template<typename _Tp> struct XYZ2RGB_f
{
    typedef _Tp channel_type;

    XYZ2RGB_f(int _dstcn, int _blueIdx, const float* _coeffs)
    : dstcn(_dstcn), blueIdx(_blueIdx)
    {
        for(int i = 0; i < 9; i++)
            coeffs[i] = _coeffs ? _coeffs[i] : (float)XYZ2sRGB_D65[i];
        if(blueIdx == 0)
        {
            std::swap(coeffs[0], coeffs[6]);
            std::swap(coeffs[1], coeffs[7]);
            std::swap(coeffs[2], coeffs[8]);
        }
    }

    void operator()(const _Tp* src, _Tp* dst, int n) const
    {
        int dcn = dstcn;
        _Tp alpha = ColorChannel<_Tp>::max();
        float C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2],
              C3 = coeffs[3], C4 = coeffs[4], C5 = coeffs[5],
              C6 = coeffs[6], C7 = coeffs[7], C8 = coeffs[8];
        n *= 3;
        for(int i = 0; i < n; i += 3, dst += dcn)
        {
            _Tp B = saturate_cast<_Tp>(src[i]*C0 + src[i+1]*C1 + src[i+2]*C2);
            _Tp G = saturate_cast<_Tp>(src[i]*C3 + src[i+1]*C4 + src[i+2]*C5);
            _Tp R = saturate_cast<_Tp>(src[i]*C6 + src[i+1]*C7 + src[i+2]*C8);
            dst[0] = B; dst[1] = G; dst[2] = R;
            if( dcn == 4 )
                dst[3] = alpha;
        }
    }
    int dstcn, blueIdx;
    float coeffs[9];
};


template <>
struct XYZ2RGB_f<float>
{
    typedef float channel_type;

    XYZ2RGB_f(int _dstcn, int _blueIdx, const float* _coeffs)
    : dstcn(_dstcn), blueIdx(_blueIdx)
    {
        for(int i = 0; i < 9; i++)
            coeffs[i] = _coeffs ? _coeffs[i] : (float)XYZ2sRGB_D65[i];
        if(blueIdx == 0)
        {
            std::swap(coeffs[0], coeffs[6]);
            std::swap(coeffs[1], coeffs[7]);
            std::swap(coeffs[2], coeffs[8]);
        }
    }

    void operator()(const float* src, float* dst, int n) const
    {
        int dcn = dstcn;
        float alpha = ColorChannel<float>::max();
        float C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2],
              C3 = coeffs[3], C4 = coeffs[4], C5 = coeffs[5],
              C6 = coeffs[6], C7 = coeffs[7], C8 = coeffs[8];
        int i = 0;
#if CV_SIMD
        const int vsize = v_float32::nlanes;
        v_float32 valpha = vx_setall_f32(alpha);
        v_float32 vc0 = vx_setall_f32(C0), vc1 = vx_setall_f32(C1), vc2 = vx_setall_f32(C2);
        v_float32 vc3 = vx_setall_f32(C3), vc4 = vx_setall_f32(C4), vc5 = vx_setall_f32(C5);
        v_float32 vc6 = vx_setall_f32(C6), vc7 = vx_setall_f32(C7), vc8 = vx_setall_f32(C8);
        for( ; i <= n-vsize;
             i += vsize, src += 3*vsize, dst += dcn*vsize)
        {
            v_float32 x, y, z;
            v_load_deinterleave(src, x, y, z);

            v_float32 b, g, r;
            b = v_fma(x, vc0, v_fma(y, vc1, z*vc2));
            g = v_fma(x, vc3, v_fma(y, vc4, z*vc5));
            r = v_fma(x, vc6, v_fma(y, vc7, z*vc8));

            if(dcn == 4)
            {
                v_store_interleave(dst, b, g, r, valpha);
            }
            else // dcn == 3
            {
                v_store_interleave(dst, b, g, r);
            }
        }
        vx_cleanup();
#endif
        for( ; i < n; i++, src += 3, dst += dcn)
        {
            float x = src[0], y = src[1], z = src[2];
            float B = saturate_cast<float>(x*C0 + y*C1 + z*C2);
            float G = saturate_cast<float>(x*C3 + y*C4 + z*C5);
            float R = saturate_cast<float>(x*C6 + y*C7 + z*C8);
            dst[0] = B; dst[1] = G; dst[2] = R;
            if( dcn == 4 )
                dst[3] = alpha;
        }
    }

    int dstcn, blueIdx;
    float coeffs[9];
};


template<typename _Tp> struct XYZ2RGB_i
{
    typedef _Tp channel_type;

    XYZ2RGB_i(int _dstcn, int _blueIdx, const int* _coeffs)
    : dstcn(_dstcn), blueIdx(_blueIdx)
    {
        for(int i = 0; i < 9; i++)
            coeffs[i] = _coeffs ? cvRound(_coeffs[i]*(1 << xyz_shift)) : XYZ2sRGB_D65_i[i];

        if(blueIdx == 0)
        {
            std::swap(coeffs[0], coeffs[6]);
            std::swap(coeffs[1], coeffs[7]);
            std::swap(coeffs[2], coeffs[8]);
        }
    }
    void operator()(const _Tp* src, _Tp* dst, int n) const
    {
        int dcn = dstcn;
        _Tp alpha = ColorChannel<_Tp>::max();
        int C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2],
            C3 = coeffs[3], C4 = coeffs[4], C5 = coeffs[5],
            C6 = coeffs[6], C7 = coeffs[7], C8 = coeffs[8];
        n *= 3;
        for(int i = 0; i < n; i += 3, dst += dcn)
        {
            int B = CV_DESCALE(src[i]*C0 + src[i+1]*C1 + src[i+2]*C2, xyz_shift);
            int G = CV_DESCALE(src[i]*C3 + src[i+1]*C4 + src[i+2]*C5, xyz_shift);
            int R = CV_DESCALE(src[i]*C6 + src[i+1]*C7 + src[i+2]*C8, xyz_shift);
            dst[0] = saturate_cast<_Tp>(B); dst[1] = saturate_cast<_Tp>(G);
            dst[2] = saturate_cast<_Tp>(R);
            if( dcn == 4 )
                dst[3] = alpha;
        }
    }
    int dstcn, blueIdx;
    int coeffs[9];
};


template <>
struct XYZ2RGB_i<uchar>
{
    typedef uchar channel_type;
    static const int shift = xyz_shift;

    XYZ2RGB_i(int _dstcn, int _blueIdx, const int* _coeffs)
    : dstcn(_dstcn), blueIdx(_blueIdx)
    {
        for(int i = 0; i < 9; i++)
            coeffs[i] = _coeffs ? cvRound(_coeffs[i]*(1 << shift)) : XYZ2sRGB_D65_i[i];

        if(blueIdx == 0)
        {
            std::swap(coeffs[0], coeffs[6]);
            std::swap(coeffs[1], coeffs[7]);
            std::swap(coeffs[2], coeffs[8]);
        }
    }

    void operator()(const uchar* src, uchar* dst, int n) const
    {
        int dcn = dstcn, i = 0;
        uchar alpha = ColorChannel<uchar>::max();
        int C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2],
            C3 = coeffs[3], C4 = coeffs[4], C5 = coeffs[5],
            C6 = coeffs[6], C7 = coeffs[7], C8 = coeffs[8];
#if CV_SIMD
        const int vsize = v_uint8::nlanes;
        const int descaleShift = 1 << (shift - 1);
        v_uint8 valpha = vx_setall_u8(alpha);
        v_int16 vdescale = vx_setall_s16(descaleShift);
        v_int16 cbxy, cbz1, cgxy, cgz1, crxy, crz1;
        v_int16 dummy;
        v_zip(vx_setall_s16((short)C0), vx_setall_s16((short)C1), cbxy, dummy);
        v_zip(vx_setall_s16((short)C2), vx_setall_s16(        1), cbz1, dummy);
        v_zip(vx_setall_s16((short)C3), vx_setall_s16((short)C4), cgxy, dummy);
        v_zip(vx_setall_s16((short)C5), vx_setall_s16(        1), cgz1, dummy);
        v_zip(vx_setall_s16((short)C6), vx_setall_s16((short)C7), crxy, dummy);
        v_zip(vx_setall_s16((short)C8), vx_setall_s16(        1), crz1, dummy);

        for ( ; i <= n-vsize;
              i += vsize, src += 3*vsize, dst += dcn*vsize)
        {
            v_uint8 x, y, z;
            v_load_deinterleave(src, x, y, z);

            v_uint16 ux0, ux1, uy0, uy1, uz0, uz1;
            v_expand(x, ux0, ux1);
            v_expand(y, uy0, uy1);
            v_expand(z, uz0, uz1);
            v_int16 x0, x1, y0, y1, z0, z1;
            x0 = v_reinterpret_as_s16(ux0);
            x1 = v_reinterpret_as_s16(ux1);
            y0 = v_reinterpret_as_s16(uy0);
            y1 = v_reinterpret_as_s16(uy1);
            z0 = v_reinterpret_as_s16(uz0);
            z1 = v_reinterpret_as_s16(uz1);

            v_int32 b[4], g[4], r[4];

            v_int16 xy[4], zd[4];
            v_zip(x0, y0, xy[0], xy[1]);
            v_zip(x1, y1, xy[2], xy[3]);
            v_zip(z0, vdescale, zd[0], zd[1]);
            v_zip(z1, vdescale, zd[2], zd[3]);

            for(int j = 0; j < 4; j++)
            {
                b[j] = (v_dotprod(xy[j], cbxy) + v_dotprod(zd[j], cbz1)) >> shift;
                g[j] = (v_dotprod(xy[j], cgxy) + v_dotprod(zd[j], cgz1)) >> shift;
                r[j] = (v_dotprod(xy[j], crxy) + v_dotprod(zd[j], crz1)) >> shift;
            }

            v_uint16 b0, b1, g0, g1, r0, r1;
            b0 = v_pack_u(b[0], b[1]); b1 = v_pack_u(b[2], b[3]);
            g0 = v_pack_u(g[0], g[1]); g1 = v_pack_u(g[2], g[3]);
            r0 = v_pack_u(r[0], r[1]); r1 = v_pack_u(r[2], r[3]);

            v_uint8 bb, gg, rr;
            bb = v_pack(b0, b1);
            gg = v_pack(g0, g1);
            rr = v_pack(r0, r1);

            if(dcn == 4)
            {
                v_store_interleave(dst, bb, gg, rr, valpha);
            }
            else // dcn == 3
            {
                v_store_interleave(dst, bb, gg, rr);
            }
        }
        vx_cleanup();
#endif
        for ( ; i < n; i++, src += 3, dst += dcn)
        {
            uchar x = src[0], y = src[1], z = src[2];
            int B = CV_DESCALE(x*C0 + y*C1 + z*C2, shift);
            int G = CV_DESCALE(x*C3 + y*C4 + z*C5, shift);
            int R = CV_DESCALE(x*C6 + y*C7 + z*C8, shift);
            dst[0] = saturate_cast<uchar>(B); dst[1] = saturate_cast<uchar>(G);
            dst[2] = saturate_cast<uchar>(R);
            if( dcn == 4 )
                dst[3] = alpha;
        }
    }
    int dstcn, blueIdx;
    int coeffs[9];
};


template <>
struct XYZ2RGB_i<ushort>
{
    typedef ushort channel_type;
    static const int shift = xyz_shift;

    XYZ2RGB_i(int _dstcn, int _blueIdx, const int* _coeffs)
    : dstcn(_dstcn), blueIdx(_blueIdx)
    {
        for(int i = 0; i < 9; i++)
            coeffs[i] = _coeffs ? cvRound(_coeffs[i]*(1 << shift)) : XYZ2sRGB_D65_i[i];

        if(blueIdx == 0)
        {
            std::swap(coeffs[0], coeffs[6]);
            std::swap(coeffs[1], coeffs[7]);
            std::swap(coeffs[2], coeffs[8]);
        }
    }

    void operator()(const ushort* src, ushort* dst, int n) const
    {
        int dcn = dstcn, i = 0;
        ushort alpha = ColorChannel<ushort>::max();
        int C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2],
            C3 = coeffs[3], C4 = coeffs[4], C5 = coeffs[5],
            C6 = coeffs[6], C7 = coeffs[7], C8 = coeffs[8];
#if CV_SIMD
        const int vsize = v_uint16::nlanes;
        const int descaleShift = 1 << (shift-1);
        v_uint16 valpha = vx_setall_u16(alpha);
        v_int16 vdescale = vx_setall_s16(descaleShift);
        v_int16 vc0 = vx_setall_s16((short)C0), vc1 = vx_setall_s16((short)C1), vc2 = vx_setall_s16((short)C2);
        v_int16 vc3 = vx_setall_s16((short)C3), vc4 = vx_setall_s16((short)C4), vc5 = vx_setall_s16((short)C5);
        v_int16 vc6 = vx_setall_s16((short)C6), vc7 = vx_setall_s16((short)C7), vc8 = vx_setall_s16((short)C8);
        v_int16 zero = vx_setzero_s16(), one = vx_setall_s16(1);
        v_int16 cbxy, cbz1, cgxy, cgz1, crxy, crz1;
        v_int16 dummy;
        v_zip(vc0, vc1, cbxy, dummy);
        v_zip(vc2, one, cbz1, dummy);
        v_zip(vc3, vc4, cgxy, dummy);
        v_zip(vc5, one, cgz1, dummy);
        v_zip(vc6, vc7, crxy, dummy);
        v_zip(vc8, one, crz1, dummy);

        for( ; i <= n-vsize;
             i += vsize, src += 3*vsize, dst += dcn*vsize)
        {
            v_uint16 x, y, z;
            v_load_deinterleave(src, x, y, z);

            v_int16 sx, sy, sz;
            sx = v_reinterpret_as_s16(x);
            sy = v_reinterpret_as_s16(y);
            sz = v_reinterpret_as_s16(z);

            // fixing 16bit signed multiplication
            v_int16 mx = sx < zero, my = sy < zero, mz = sz < zero;

            v_int16 bmx, bmy, bmz;
            v_int16 gmx, gmy, gmz;
            v_int16 rmx, rmy, rmz;

            bmx = mx & vc0;
            bmy = my & vc1;
            bmz = mz & vc2;
            gmx = mx & vc3;
            gmy = my & vc4;
            gmz = mz & vc5;
            rmx = mx & vc6;
            rmy = my & vc7;
            rmz = mz & vc8;

            v_int32 bfix0, bfix1, gfix0, gfix1, rfix0, rfix1;
            v_expand(bmx + bmy + bmz, bfix0, bfix1);
            v_expand(gmx + gmy + gmz, gfix0, gfix1);
            v_expand(rmx + rmy + rmz, rfix0, rfix1);

            bfix0 = bfix0 << 16; bfix1 = bfix1 << 16;
            gfix0 = gfix0 << 16; gfix1 = gfix1 << 16;
            rfix0 = rfix0 << 16; rfix1 = rfix1 << 16;

            v_int16 xy0, xy1, zd0, zd1;
            v_zip(sx, sy, xy0, xy1);
            v_zip(sz, vdescale, zd0, zd1);

            v_int32 b0, b1, g0, g1, r0, r1;

            b0 = (v_dotprod(xy0, cbxy) + v_dotprod(zd0, cbz1) + bfix0) >> shift;
            b1 = (v_dotprod(xy1, cbxy) + v_dotprod(zd1, cbz1) + bfix1) >> shift;
            g0 = (v_dotprod(xy0, cgxy) + v_dotprod(zd0, cgz1) + gfix0) >> shift;
            g1 = (v_dotprod(xy1, cgxy) + v_dotprod(zd1, cgz1) + gfix1) >> shift;
            r0 = (v_dotprod(xy0, crxy) + v_dotprod(zd0, crz1) + rfix0) >> shift;
            r1 = (v_dotprod(xy1, crxy) + v_dotprod(zd1, crz1) + rfix1) >> shift;

            v_uint16 b, g, r;
            b = v_pack_u(b0, b1); g = v_pack_u(g0, g1); r = v_pack_u(r0, r1);

            if(dcn == 4)
            {
                v_store_interleave(dst, b, g, r, valpha);
            }
            else // dcn == 3
            {
                v_store_interleave(dst, b, g, r);
            }
        }
        vx_cleanup();
#endif
        for ( ; i < n; i++, src += 3, dst += dcn)
        {
            ushort x = src[0], y = src[1], z = src[2];
            int B = CV_DESCALE(x*C0 + y*C1 + z*C2, shift);
            int G = CV_DESCALE(x*C3 + y*C4 + z*C5, shift);
            int R = CV_DESCALE(x*C6 + y*C7 + z*C8, shift);
            dst[0] = saturate_cast<ushort>(B); dst[1] = saturate_cast<ushort>(G);
            dst[2] = saturate_cast<ushort>(R);
            if( dcn == 4 )
                dst[3] = alpha;
        }
    }
    int dstcn, blueIdx;
    int coeffs[9];
};


///////////////////////////////////// RGB <-> L*a*b* /////////////////////////////////////

//0.950456, 1., 1.088754
static const softdouble D65[] = {softdouble::fromRaw(0x3fee6a22b3892ee8),
                                 softdouble::one(),
                                 softdouble::fromRaw(0x3ff16b8950763a19)};

enum { LAB_CBRT_TAB_SIZE = 1024, GAMMA_TAB_SIZE = 1024 };
static const float *LabCbrtTab = 0;
static const float LabCbrtTabScale = softfloat(LAB_CBRT_TAB_SIZE*2)/softfloat(3);

static const float *sRGBGammaTab = 0;
static const float *sRGBInvGammaTab = 0;
static const float GammaTabScale((int)GAMMA_TAB_SIZE);

static ushort sRGBGammaTab_b[256], linearGammaTab_b[256];
enum { inv_gamma_shift = 12, INV_GAMMA_TAB_SIZE = (1 << inv_gamma_shift) };
static ushort sRGBInvGammaTab_b[INV_GAMMA_TAB_SIZE], linearInvGammaTab_b[INV_GAMMA_TAB_SIZE];
#undef lab_shift
#define lab_shift xyz_shift
#define gamma_shift 3
#define lab_shift2 (lab_shift + gamma_shift)
#define LAB_CBRT_TAB_SIZE_B (256*3/2*(1<<gamma_shift))
static ushort LabCbrtTab_b[LAB_CBRT_TAB_SIZE_B];

static const bool enableBitExactness = true;
static const bool enableRGB2LabInterpolation = true;
static const bool enablePackedLab = true;
enum
{
    lab_lut_shift = 5,
    LAB_LUT_DIM = (1 << lab_lut_shift)+1,
    lab_base_shift = 14,
    LAB_BASE = (1 << lab_base_shift),
    LUT_BASE = (1 << 14),
    trilinear_shift = 8 - lab_lut_shift + 1,
    TRILINEAR_BASE = (1 << trilinear_shift)
};
static int16_t trilinearLUT[TRILINEAR_BASE*TRILINEAR_BASE*TRILINEAR_BASE*8];
static ushort LabToYF_b[256*2];
static const int minABvalue = -8145;
static const int *abToXZ_b;
// Luv constants
static const bool enableRGB2LuvInterpolation = true;
static const bool enablePackedRGB2Luv = true;
static const bool enablePackedLuv2RGB = true;
static const softfloat uLow(-134), uHigh(220), uRange(uHigh-uLow);
static const softfloat vLow(-140), vHigh(122), vRange(vHigh-vLow);

static struct LABLUVLUT_s16_t {
    const int16_t *RGB2LabLUT_s16;
    const int16_t *RGB2LuvLUT_s16;
} LABLUVLUTs16 = {0, 0};

static struct LUVLUT_T {
    const int *LuToUp_b;
    const int *LvToVp_b;
    const long long int *LvToVpl_b;
} LUVLUT = {0, 0, 0};

#define clip(value) \
    value < 0.0f ? 0.0f : value > 1.0f ? 1.0f : value;

//all constants should be presented through integers to keep bit-exactness
static const softdouble gammaThreshold    = softdouble(809)/softdouble(20000);    //  0.04045
static const softdouble gammaInvThreshold = softdouble(7827)/softdouble(2500000); //  0.0031308
static const softdouble gammaLowScale     = softdouble(323)/softdouble(25);       // 12.92
static const softdouble gammaPower        = softdouble(12)/softdouble(5);         //  2.4
static const softdouble gammaXshift       = softdouble(11)/softdouble(200);       //  0.055

static const softfloat lthresh = softfloat(216) / softfloat(24389); // 0.008856f = (6/29)^3
static const softfloat lscale  = softfloat(841) / softfloat(108); // 7.787f = (29/3)^3/(29*4)
static const softfloat lbias = softfloat(16) / softfloat(116);
static const softfloat f255(255);


static inline softfloat applyGamma(softfloat x)
{
    //return x <= 0.04045f ? x*(1.f/12.92f) : (float)std::pow((double)(x + 0.055)*(1./1.055), 2.4);
    softdouble xd = x;
    return (xd <= gammaThreshold ?
                xd/gammaLowScale :
                pow((xd + gammaXshift)/(softdouble::one()+gammaXshift), gammaPower));
}


static inline softfloat applyInvGamma(softfloat x)
{
    //return x <= 0.0031308 ? x*12.92f : (float)(1.055*std::pow((double)x, 1./2.4) - 0.055);
    softdouble xd = x;
    return (xd <= gammaInvThreshold ?
                xd*gammaLowScale :
                pow(xd, softdouble::one()/gammaPower)*(softdouble::one()+gammaXshift) - gammaXshift);
}


static LUVLUT_T initLUTforLUV(const softfloat &un, const softfloat &vn)
{
    //when XYZ are limited to [0, 2]
    /*
        up: [-402, 1431.57]
        min abs diff up: 0.010407
        vp: [-0.25, 0.25]
        min abs(vp): 0.00034207
    */

    const softfloat oneof4 = softfloat::one()/softfloat(4);
    int *LuToUp_b = cv::allocSingleton<int>(256*256);
    int *LvToVp_b = cv::allocSingleton<int>(256*256);
    long long int *LvToVpl_b = cv::allocSingleton<long long int>(256*256);
    for(int LL = 0; LL < 256; LL++)
    {
        softfloat L = softfloat(LL*100)/f255;
        for(int uu = 0; uu < 256; uu++)
        {
            softfloat u = softfloat(uu)*uRange/f255 + uLow;
            softfloat up = softfloat(9)*(u + L*un);
            LuToUp_b[LL*256+uu] = cvRound(up*softfloat(LUT_BASE/1024));//1024 is OK, 2048 gave maxerr 3
        }
        for(int vv = 0; vv < 256; vv++)
        {
            softfloat v = softfloat(vv)*vRange/f255 + vLow;
            softfloat vp = oneof4/(v + L*vn);
            if(vp >  oneof4) vp =  oneof4;
            if(vp < -oneof4) vp = -oneof4;
            int ivp = cvRound(vp*softfloat(LUT_BASE*1024));
            LvToVp_b[LL*256+vv] = ivp;
            int vpl = ivp*LL;
            LvToVpl_b[LL*256+vv] = (12*13*100*(LUT_BASE/1024))*(long long)vpl;
        }
    }
    LUVLUT_T res;
    res.LuToUp_b = LuToUp_b;
    res.LvToVp_b = LvToVp_b;
    res.LvToVpl_b = LvToVpl_b;
    return res;
}


static int * initLUTforABXZ()
{
    int * res = cv::allocSingleton<int>(LAB_BASE*9/4);
    for(int i = minABvalue; i < LAB_BASE*9/4+minABvalue; i++)
    {
        int v;
        //6.f/29.f*BASE = 3389.730
        if(i <= 3390)
        {
            //fxz[k] = (fxz[k] - 16.0f / 116.0f) / 7.787f;
            // 7.787f = (29/3)^3/(29*4)
            v = i*108/841 - LUT_BASE*16/116*108/841;
        }
        else
        {
            //fxz[k] = fxz[k] * fxz[k] * fxz[k];
            v = i*i/LUT_BASE*i/LUT_BASE;
        }
        res[i-minABvalue] = v; // -1335 <= v <= 88231
    }
    return res;
}


inline void fill_one(int16_t *LAB, const int16_t *LAB_prev, int16_t *LUV, const int16_t *LUV_prev, int p, int q, int r, int _p, int _q, int _r)
{
    int idxold = 0;
    idxold += min(p+(_p), (int)(LAB_LUT_DIM-1))*3;
    idxold += min(q+(_q), (int)(LAB_LUT_DIM-1))*LAB_LUT_DIM*3;
    idxold += min(r+(_r), (int)(LAB_LUT_DIM-1))*LAB_LUT_DIM*LAB_LUT_DIM*3;
    int idxnew = p*3*8 + q*LAB_LUT_DIM*3*8 + r*LAB_LUT_DIM*LAB_LUT_DIM*3*8+4*(_p)+2*(_q)+(_r);
    LAB[idxnew]    = LAB_prev[idxold];
    LAB[idxnew+8]  = LAB_prev[idxold+1];
    LAB[idxnew+16] = LAB_prev[idxold+2];
    LUV[idxnew]    = LUV_prev[idxold];
    LUV[idxnew+8]  = LUV_prev[idxold+1];
    LUV[idxnew+16] = LUV_prev[idxold+2];
}


static LABLUVLUT_s16_t initLUTforLABLUVs16(const softfloat & un, const softfloat & vn)
{
    int i;
    softfloat scaledCoeffs[9], coeffs[9];

    //RGB2Lab coeffs
    softdouble scaleWhite[] = { softdouble::one()/D65[0],
                                softdouble::one(),
                                softdouble::one()/D65[2] };

    for(i = 0; i < 3; i++ )
    {
        coeffs[i*3+2] = sRGB2XYZ_D65[i*3+0];
        coeffs[i*3+1] = sRGB2XYZ_D65[i*3+1];
        coeffs[i*3+0] = sRGB2XYZ_D65[i*3+2];
        scaledCoeffs[i*3+0] = sRGB2XYZ_D65[i*3+2] * scaleWhite[i];
        scaledCoeffs[i*3+1] = sRGB2XYZ_D65[i*3+1] * scaleWhite[i];
        scaledCoeffs[i*3+2] = sRGB2XYZ_D65[i*3+0] * scaleWhite[i];
    }

    softfloat S0 = scaledCoeffs[0], S1 = scaledCoeffs[1], S2 = scaledCoeffs[2],
              S3 = scaledCoeffs[3], S4 = scaledCoeffs[4], S5 = scaledCoeffs[5],
              S6 = scaledCoeffs[6], S7 = scaledCoeffs[7], S8 = scaledCoeffs[8];
    softfloat C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2],
              C3 = coeffs[3], C4 = coeffs[4], C5 = coeffs[5],
              C6 = coeffs[6], C7 = coeffs[7], C8 = coeffs[8];

    //u, v: [-134.0, 220.0], [-140.0, 122.0]
    static const softfloat lld(LAB_LUT_DIM - 1), f116(116), f16(16), f500(500), f200(200);
    static const softfloat f100(100), f128(128), f256(256), lbase((int)LAB_BASE);
    //903.3f = (29/3)^3
    static const softfloat f9033 = softfloat(29*29*29)/softfloat(27);
    static const softfloat f9of4 = softfloat(9)/softfloat(4);
    static const softfloat f15(15), f3(3);

    AutoBuffer<int16_t> RGB2Labprev(LAB_LUT_DIM*LAB_LUT_DIM*LAB_LUT_DIM*3);
    AutoBuffer<int16_t> RGB2Luvprev(LAB_LUT_DIM*LAB_LUT_DIM*LAB_LUT_DIM*3);
    for(int p = 0; p < LAB_LUT_DIM; p++)
    {
        for(int q = 0; q < LAB_LUT_DIM; q++)
        {
            for(int r = 0; r < LAB_LUT_DIM; r++)
            {
                int idx = p*3 + q*LAB_LUT_DIM*3 + r*LAB_LUT_DIM*LAB_LUT_DIM*3;
                softfloat R = softfloat(p)/lld;
                softfloat G = softfloat(q)/lld;
                softfloat B = softfloat(r)/lld;

                R = applyGamma(R);
                G = applyGamma(G);
                B = applyGamma(B);

                //RGB 2 Lab LUT building
                {
                    softfloat X = R*S0 + G*S1 + B*S2;
                    softfloat Y = R*S3 + G*S4 + B*S5;
                    softfloat Z = R*S6 + G*S7 + B*S8;

                    softfloat FX = X > lthresh ? cbrt(X) : mulAdd(X, lscale, lbias);
                    softfloat FY = Y > lthresh ? cbrt(Y) : mulAdd(Y, lscale, lbias);
                    softfloat FZ = Z > lthresh ? cbrt(Z) : mulAdd(Z, lscale, lbias);

                    softfloat L = Y > lthresh ? (f116*FY - f16) : (f9033*Y);
                    softfloat a = f500 * (FX - FY);
                    softfloat b = f200 * (FY - FZ);

                    RGB2Labprev[idx]   = (int16_t)(cvRound(lbase*L/f100));
                    RGB2Labprev[idx+1] = (int16_t)(cvRound(lbase*(a + f128)/f256));
                    RGB2Labprev[idx+2] = (int16_t)(cvRound(lbase*(b + f128)/f256));
                }

                //RGB 2 Luv LUT building
                {
                    softfloat X = R*C0 + G*C1 + B*C2;
                    softfloat Y = R*C3 + G*C4 + B*C5;
                    softfloat Z = R*C6 + G*C7 + B*C8;

                    softfloat L = Y < lthresh ? mulAdd(Y, lscale, lbias) : cbrt(Y);
                    L = L*f116 - f16;

                    softfloat d = softfloat(4*13)/max(X + f15 * Y + f3 * Z, softfloat(FLT_EPSILON));
                    softfloat u = L*(X*d - un);
                    softfloat v = L*(f9of4*Y*d - vn);

                    RGB2Luvprev[idx  ] = (int16_t)cvRound(lbase*L/f100);
                    RGB2Luvprev[idx+1] = (int16_t)cvRound(lbase*(u-uLow)/uRange);
                    RGB2Luvprev[idx+2] = (int16_t)cvRound(lbase*(v-vLow)/vRange);
                }
            }
        }
    }

    int16_t *RGB2LabLUT_s16 = cv::allocSingleton<int16_t>(LAB_LUT_DIM*LAB_LUT_DIM*LAB_LUT_DIM*3*8);
    int16_t *RGB2LuvLUT_s16 = cv::allocSingleton<int16_t>(LAB_LUT_DIM*LAB_LUT_DIM*LAB_LUT_DIM*3*8);
    for(int p = 0; p < LAB_LUT_DIM; p++)
        for(int q = 0; q < LAB_LUT_DIM; q++)
            for(int r = 0; r < LAB_LUT_DIM; r++)
                for (int p_ = 0; p_ < 2; ++p_)
                    for (int q_ = 0; q_ < 2; ++q_)
                        for (int r_ = 0; r_ < 2; ++r_)
                            fill_one(RGB2LabLUT_s16, RGB2Labprev.data(), RGB2LuvLUT_s16, RGB2Luvprev.data(), p, q, r, p_, q_, r_);
    LABLUVLUT_s16_t res;
    res.RGB2LabLUT_s16 = RGB2LabLUT_s16;
    res.RGB2LuvLUT_s16 = RGB2LuvLUT_s16;
    return res;
}


static void initLabTabs()
{
    static bool initialized = false;
    if(!initialized)
    {
        softfloat f[LAB_CBRT_TAB_SIZE+1], g[GAMMA_TAB_SIZE+1], ig[GAMMA_TAB_SIZE+1];
        softfloat scale = softfloat::one()/softfloat(LabCbrtTabScale);
        int i;
        for(i = 0; i <= LAB_CBRT_TAB_SIZE; i++)
        {
            softfloat x = scale*softfloat(i);
            f[i] = x < lthresh ? mulAdd(x, lscale, lbias) : cbrt(x);
        }
        LabCbrtTab = splineBuild(f, LAB_CBRT_TAB_SIZE);

        scale = softfloat::one()/softfloat(GammaTabScale);
        for(i = 0; i <= GAMMA_TAB_SIZE; i++)
        {
            softfloat x = scale*softfloat(i);
            g[i] = applyGamma(x);
            ig[i] = applyInvGamma(x);
        }

        sRGBGammaTab = splineBuild(g, GAMMA_TAB_SIZE);
        sRGBInvGammaTab = splineBuild(ig, GAMMA_TAB_SIZE);

        static const softfloat intScale(255*(1 << gamma_shift));
        for(i = 0; i < 256; i++)
        {
            softfloat x = softfloat(i)/f255;
            sRGBGammaTab_b[i] = (ushort)(cvRound(intScale*applyGamma(x)));
            linearGammaTab_b[i] = (ushort)(i*(1 << gamma_shift));
        }
        static const softfloat invScale = softfloat::one()/softfloat((int)INV_GAMMA_TAB_SIZE);
        for(i = 0; i < INV_GAMMA_TAB_SIZE; i++)
        {
            softfloat x = invScale*softfloat(i);
            sRGBInvGammaTab_b[i] = (ushort)(cvRound(f255*applyInvGamma(x)));
            linearInvGammaTab_b[i] = (ushort)(cvTrunc(f255*x));
        }

        static const softfloat cbTabScale(softfloat::one()/(f255*(1 << gamma_shift)));
        static const softfloat lshift2(1 << lab_shift2);
        for(i = 0; i < LAB_CBRT_TAB_SIZE_B; i++)
        {
            softfloat x = cbTabScale*softfloat(i);
            LabCbrtTab_b[i] = (ushort)(cvRound(lshift2 * (x < lthresh ? mulAdd(x, lscale, lbias) : cbrt(x))));
        }

        //Lookup table for L to y and ify calculations
        for(i = 0; i < 256; i++)
        {
            int y, ify;
            //8 * 255.0 / 100.0 == 20.4
            if( i <= 20)
            {
                //yy = li / 903.3f;
                //y = L*100/903.3f; 903.3f = (29/3)^3, 255 = 17*3*5
                y = cvRound(softfloat(i*LUT_BASE*20*9)/softfloat(17*29*29*29));
                //fy = 7.787f * yy + 16.0f / 116.0f; 7.787f = (29/3)^3/(29*4)
                ify = cvRound(softfloat((int)LUT_BASE)*(softfloat(16)/softfloat(116) + softfloat(i*5)/softfloat(3*17*29)));
            }
            else
            {
                //fy = (li + 16.0f) / 116.0f;
                softfloat fy = (softfloat(i*100*LUT_BASE)/softfloat(255*116) +
                                softfloat(16*LUT_BASE)/softfloat(116));
                ify = cvRound(fy);
                //yy = fy * fy * fy;
                y = cvRound(fy*fy*fy/softfloat(LUT_BASE*LUT_BASE));
            }

            LabToYF_b[i*2  ] = (ushort)y;   // 2260 <= y <= BASE
            LabToYF_b[i*2+1] = (ushort)ify; // 0 <= ify <= BASE
        }

        //Lookup table for a,b to x,z conversion
        abToXZ_b = initLUTforABXZ();

        softfloat dd = D65[0] + D65[1]*softdouble(15) + D65[2]*softdouble(3);
        dd = softfloat::one()/max(dd, softfloat::eps());
        softfloat un = dd*softfloat(13*4)*D65[0];
        softfloat vn = dd*softfloat(13*9)*D65[1];

        //Luv LUT
        LUVLUT = initLUTforLUV(un, vn);

        //try to suppress warning
        static const bool calcLUT = enableRGB2LabInterpolation || enableRGB2LuvInterpolation;
        if(calcLUT)
        {

            LABLUVLUTs16 = initLUTforLABLUVs16(un, vn);

            for(int16_t p = 0; p < TRILINEAR_BASE; p++)
            {
                int16_t pp = TRILINEAR_BASE - p;
                for(int16_t q = 0; q < TRILINEAR_BASE; q++)
                {
                    int16_t qq = TRILINEAR_BASE - q;
                    for(int16_t r = 0; r < TRILINEAR_BASE; r++)
                    {
                        int16_t rr = TRILINEAR_BASE - r;
                        int16_t* w = &trilinearLUT[8*p + 8*TRILINEAR_BASE*q + 8*TRILINEAR_BASE*TRILINEAR_BASE*r];
                        w[0]  = pp * qq * rr; w[1]  = pp * qq * r ; w[2]  = pp * q  * rr; w[3]  = pp * q  * r ;
                        w[4]  = p  * qq * rr; w[5]  = p  * qq * r ; w[6]  = p  * q  * rr; w[7]  = p  * q  * r ;
                    }
                }
            }
        }

        initialized = true;
    }
}


// cx, cy, cz are in [0; LAB_BASE]
static inline void trilinearInterpolate(int cx, int cy, int cz, const int16_t* LUT,
                                        int& a, int& b, int& c)
{
    //LUT idx of origin pt of cube
    int tx = cx >> (lab_base_shift - lab_lut_shift);
    int ty = cy >> (lab_base_shift - lab_lut_shift);
    int tz = cz >> (lab_base_shift - lab_lut_shift);

    const int16_t* baseLUT = &LUT[3*8*tx + (3*8*LAB_LUT_DIM)*ty + (3*8*LAB_LUT_DIM*LAB_LUT_DIM)*tz];
    int aa[8], bb[8], cc[8];
    for(int i = 0; i < 8; i++)
    {
        aa[i] = baseLUT[i]; bb[i] = baseLUT[i+8]; cc[i] = baseLUT[i+16];
    }

    //x, y, z are [0; TRILINEAR_BASE)
    static const int bitMask = (1 << trilinear_shift) - 1;
    int x = (cx >> (lab_base_shift - 8 - 1)) & bitMask;
    int y = (cy >> (lab_base_shift - 8 - 1)) & bitMask;
    int z = (cz >> (lab_base_shift - 8 - 1)) & bitMask;

    int w[8];
    for(int i = 0; i < 8; i++)
    {
        w[i] = trilinearLUT[8*x + 8*TRILINEAR_BASE*y + 8*TRILINEAR_BASE*TRILINEAR_BASE*z + i];
    }

    a = aa[0]*w[0]+aa[1]*w[1]+aa[2]*w[2]+aa[3]*w[3]+aa[4]*w[4]+aa[5]*w[5]+aa[6]*w[6]+aa[7]*w[7];
    b = bb[0]*w[0]+bb[1]*w[1]+bb[2]*w[2]+bb[3]*w[3]+bb[4]*w[4]+bb[5]*w[5]+bb[6]*w[6]+bb[7]*w[7];
    c = cc[0]*w[0]+cc[1]*w[1]+cc[2]*w[2]+cc[3]*w[3]+cc[4]*w[4]+cc[5]*w[5]+cc[6]*w[6]+cc[7]*w[7];

    a = CV_DESCALE(a, trilinear_shift*3);
    b = CV_DESCALE(b, trilinear_shift*3);
    c = CV_DESCALE(c, trilinear_shift*3);
}

#if CV_SIMD

// inValues are in [0; LAB_BASE]
static inline void trilinearPackedInterpolate(const v_uint16& inX, const v_uint16& inY, const v_uint16& inZ,
                                              const int16_t* LUT,
                                              v_uint16& outA, v_uint16& outB, v_uint16& outC)
{
    // LUT idx of origin pt of cube
    v_uint16 tx = inX >> (lab_base_shift - lab_lut_shift);
    v_uint16 ty = inY >> (lab_base_shift - lab_lut_shift);
    v_uint16 tz = inZ >> (lab_base_shift - lab_lut_shift);

    v_uint32 btmp00, btmp01, btmp10, btmp11, btmp20, btmp21;
    v_int32 baseIdx0, baseIdx1;
    // baseIdx = tx*(3*8)+ty*(3*8*LAB_LUT_DIM)+tz*(3*8*LAB_LUT_DIM*LAB_LUT_DIM)
    // 4 instead of 8, because addresses are casted to int32_t instead of int16_t
    v_mul_expand(tx, vx_setall_u16(3*4), btmp00, btmp01);
    v_mul_expand(ty, vx_setall_u16(3*4*LAB_LUT_DIM), btmp10, btmp11);
    v_mul_expand(tz, vx_setall_u16(3*4*LAB_LUT_DIM*LAB_LUT_DIM), btmp20, btmp21);
    baseIdx0 = v_reinterpret_as_s32(btmp00 + btmp10 + btmp20);
    baseIdx1 = v_reinterpret_as_s32(btmp01 + btmp11 + btmp21);

    // fracX, fracY, fracZ are [0; TRILINEAR_BASE)
    const uint16_t bitMask = (1 << trilinear_shift) - 1;
    v_uint16 bitMaskReg = vx_setall_u16(bitMask);
    v_uint16 fracX = (inX >> (lab_base_shift - 8 - 1)) & bitMaskReg;
    v_uint16 fracY = (inY >> (lab_base_shift - 8 - 1)) & bitMaskReg;
    v_uint16 fracZ = (inZ >> (lab_base_shift - 8 - 1)) & bitMaskReg;

    // trilinearIdx = 8*x + 8*TRILINEAR_BASE*y + 8*TRILINEAR_BASE*TRILINEAR_BASE*z
    v_int32 trilinearIdx0, trilinearIdx1;
    v_int32 fracX0, fracX1, fracY0, fracY1, fracZ0, fracZ1;
    v_expand(v_reinterpret_as_s16(fracX), fracX0, fracX1);
    v_expand(v_reinterpret_as_s16(fracY), fracY0, fracY1);
    v_expand(v_reinterpret_as_s16(fracZ), fracZ0, fracZ1);
    // shift minus 1, because addresses are casted to int32_t instead of int16_t
    trilinearIdx0 = (fracX0 << (3-1)) + (fracY0 << (3-1+trilinear_shift)) + (fracZ0 << (3-1+trilinear_shift*2));
    trilinearIdx1 = (fracX1 << (3-1)) + (fracY1 << (3-1+trilinear_shift)) + (fracZ1 << (3-1+trilinear_shift*2));

    v_int32 sa0, sa1, sb0, sb1, sc0, sc1;
    sa0 = vx_setzero_s32(), sa1 = vx_setzero_s32();
    sb0 = vx_setzero_s32(), sb1 = vx_setzero_s32();
    sc0 = vx_setzero_s32(), sc1 = vx_setzero_s32();

    for(int i = 0; i < 4; i++)
    {
        v_int16 a0q, a1q, b0q, b1q, c0q, c1q, w0q, w1q;

        //load values to interpolate for pix0, pix1,..
        a0q = v_reinterpret_as_s16(v_lut(((const int32_t*)LUT)+i, baseIdx0));
        a1q = v_reinterpret_as_s16(v_lut(((const int32_t*)LUT)+i, baseIdx1));

        b0q = v_reinterpret_as_s16(v_lut(((const int32_t*)(LUT+8))+i, baseIdx0));
        b1q = v_reinterpret_as_s16(v_lut(((const int32_t*)(LUT+8))+i, baseIdx1));

        c0q = v_reinterpret_as_s16(v_lut(((const int32_t*)(LUT+16))+i, baseIdx0));
        c1q = v_reinterpret_as_s16(v_lut(((const int32_t*)(LUT+16))+i, baseIdx1));

        //interpolation weights for pix0, pix1,..
        w0q = v_reinterpret_as_s16(v_lut(((const int32_t*)trilinearLUT)+i, trilinearIdx0));
        w1q = v_reinterpret_as_s16(v_lut(((const int32_t*)trilinearLUT)+i, trilinearIdx1));

        sa0 += v_dotprod(a0q, w0q); sa1 += v_dotprod(a1q, w1q);
        sb0 += v_dotprod(b0q, w0q); sb1 += v_dotprod(b1q, w1q);
        sc0 += v_dotprod(c0q, w0q); sc1 += v_dotprod(c1q, w1q);
    }

    v_uint32 a0, a1, b0, b1, c0, c1;
    v_uint32 descaleShift = vx_setall_u32(1 << (trilinear_shift*3 - 1));
    a0 = (v_reinterpret_as_u32(sa0) + descaleShift) >> (trilinear_shift*3);
    a1 = (v_reinterpret_as_u32(sa1) + descaleShift) >> (trilinear_shift*3);
    b0 = (v_reinterpret_as_u32(sb0) + descaleShift) >> (trilinear_shift*3);
    b1 = (v_reinterpret_as_u32(sb1) + descaleShift) >> (trilinear_shift*3);
    c0 = (v_reinterpret_as_u32(sc0) + descaleShift) >> (trilinear_shift*3);
    c1 = (v_reinterpret_as_u32(sc1) + descaleShift) >> (trilinear_shift*3);

    outA = v_pack(a0, a1);
    outB = v_pack(b0, b1);
    outC = v_pack(c0, c1);
}

#endif // CV_SIMD


struct RGB2Lab_b
{
    typedef uchar channel_type;

    RGB2Lab_b(int _srccn, int blueIdx, const float* _coeffs,
              const float* _whitept, bool _srgb)
    : srccn(_srccn), srgb(_srgb)
    {
        static volatile int _3 = 3;
        initLabTabs();

        softdouble whitePt[3];
        for(int i = 0; i < 3; i++)
            if(_whitept)
                whitePt[i] = softdouble(_whitept[i]);
            else
                whitePt[i] = D65[i];

        static const softdouble lshift(1 << lab_shift);
        for( int i = 0; i < _3; i++ )
        {
            softdouble c[3];
            for(int j = 0; j < 3; j++)
                if(_coeffs)
                    c[j] = softdouble(_coeffs[i*3+j]);
                else
                    c[j] = sRGB2XYZ_D65[i*3+j];
            coeffs[i*3+(blueIdx^2)] = cvRound(lshift*c[0]/whitePt[i]);
            coeffs[i*3+1]           = cvRound(lshift*c[1]/whitePt[i]);
            coeffs[i*3+blueIdx]     = cvRound(lshift*c[2]/whitePt[i]);

            CV_Assert(coeffs[i*3] >= 0 && coeffs[i*3+1] >= 0 && coeffs[i*3+2] >= 0 &&
                      coeffs[i*3] + coeffs[i*3+1] + coeffs[i*3+2] < 2*(1 << lab_shift));
        }
    }

    void operator()(const uchar* src, uchar* dst, int n) const
    {
        const int Lscale = (116*255+50)/100;
        const int Lshift = -((16*255*(1 << lab_shift2) + 50)/100);
        const ushort* tab = srgb ? sRGBGammaTab_b : linearGammaTab_b;
        int i, scn = srccn;
        int C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2],
            C3 = coeffs[3], C4 = coeffs[4], C5 = coeffs[5],
            C6 = coeffs[6], C7 = coeffs[7], C8 = coeffs[8];
        n *= 3;

        i = 0;
        for(; i < n; i += 3, src += scn )
        {
            int R = tab[src[0]], G = tab[src[1]], B = tab[src[2]];
            int fX = LabCbrtTab_b[CV_DESCALE(R*C0 + G*C1 + B*C2, lab_shift)];
            int fY = LabCbrtTab_b[CV_DESCALE(R*C3 + G*C4 + B*C5, lab_shift)];
            int fZ = LabCbrtTab_b[CV_DESCALE(R*C6 + G*C7 + B*C8, lab_shift)];

            int L = CV_DESCALE( Lscale*fY + Lshift, lab_shift2 );
            int a = CV_DESCALE( 500*(fX - fY) + 128*(1 << lab_shift2), lab_shift2 );
            int b = CV_DESCALE( 200*(fY - fZ) + 128*(1 << lab_shift2), lab_shift2 );

            dst[i] = saturate_cast<uchar>(L);
            dst[i+1] = saturate_cast<uchar>(a);
            dst[i+2] = saturate_cast<uchar>(b);
        }
    }

    int srccn;
    int coeffs[9];
    bool srgb;
};


struct RGB2Lab_f
{
    typedef float channel_type;

    RGB2Lab_f(int _srccn, int _blueIdx, const float* _coeffs,
              const float* _whitept, bool _srgb)
    : srccn(_srccn), srgb(_srgb), blueIdx(_blueIdx)
    {
        volatile int _3 = 3;
        initLabTabs();

        useInterpolation = (!_coeffs && !_whitept && srgb && enableRGB2LabInterpolation);

        softdouble whitePt[3];
        for(int i = 0; i < 3; i++)
            if(_whitept)
                whitePt[i] = softdouble((double)_whitept[i]);
            else
                whitePt[i] = D65[i];

        softdouble scale[] = { softdouble::one() / whitePt[0],
                               softdouble::one(),
                               softdouble::one() / whitePt[2] };

        for( int i = 0; i < _3; i++ )
        {
            softfloat c[3];
            for(int k = 0; k < 3; k++)
                if(_coeffs)
                    c[k] = scale[i] * softdouble((double)_coeffs[i*3 + k]);
                else
                    c[k] = scale[i] * sRGB2XYZ_D65[i*3 + k];
            coeffs[i*3 + (blueIdx ^ 2)] = c[0];
            coeffs[i*3 + 1]             = c[1];
            coeffs[i*3 + blueIdx]       = c[2];

            CV_Assert( c[0] >= 0 && c[1] >= 0 && c[2] >= 0 &&
                       c[0] + c[1] + c[2] < softfloat((int)LAB_CBRT_TAB_SIZE) );
        }
    }

    void operator()(const float* src, float* dst, int n) const
    {
        int i, scn = srccn, bIdx = blueIdx;
        float gscale = GammaTabScale;
        const float* gammaTab = srgb ? sRGBGammaTab : 0;
        float C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2],
              C3 = coeffs[3], C4 = coeffs[4], C5 = coeffs[5],
              C6 = coeffs[6], C7 = coeffs[7], C8 = coeffs[8];
        n *= 3;

        i = 0;
        if(useInterpolation)
        {

#if CV_SIMD
            if(enablePackedLab)
            {
                const int vsize = v_float32::nlanes;
                static const int nPixels = vsize*2;
                for(; i < n - 3*nPixels; i += 3*nPixels, src += scn*nPixels)
                {
                    v_float32 rvec0, gvec0, bvec0, rvec1, gvec1, bvec1;
                    if(scn == 3)
                    {
                        v_load_deinterleave(src + 0*vsize, rvec0, gvec0, bvec0);
                        v_load_deinterleave(src + 3*vsize, rvec1, gvec1, bvec1);
                    }
                    else // scn == 4
                    {
                        v_float32 dummy0, dummy1;
                        v_load_deinterleave(src + 0*vsize, rvec0, gvec0, bvec0, dummy0);
                        v_load_deinterleave(src + 4*vsize, rvec1, gvec1, bvec1, dummy1);
                    }

                    if(bIdx)
                    {
                        swap(rvec0, bvec0);
                        swap(rvec1, bvec1);
                    }

                    v_float32 zerof = vx_setzero_f32(), onef = vx_setall_f32(1.0f);
                    /* clip() */
                    #define clipv(r) (r) = v_min(v_max((r), zerof), onef)
                    clipv(rvec0); clipv(rvec1);
                    clipv(gvec0); clipv(gvec1);
                    clipv(bvec0); clipv(bvec1);
                    #undef clipv
                    /* int iR = R*LAB_BASE, iG = G*LAB_BASE, iB = B*LAB_BASE, iL, ia, ib; */
                    v_float32 basef = vx_setall_f32(LAB_BASE);
                    rvec0 *= basef, gvec0 *= basef, bvec0 *= basef;
                    rvec1 *= basef, gvec1 *= basef, bvec1 *= basef;

                    v_int32 irvec0, igvec0, ibvec0, irvec1, igvec1, ibvec1;
                    irvec0 = v_round(rvec0); irvec1 = v_round(rvec1);
                    igvec0 = v_round(gvec0); igvec1 = v_round(gvec1);
                    ibvec0 = v_round(bvec0); ibvec1 = v_round(bvec1);

                    v_uint16 uirvec = v_pack_u(irvec0, irvec1);
                    v_uint16 uigvec = v_pack_u(igvec0, igvec1);
                    v_uint16 uibvec = v_pack_u(ibvec0, ibvec1);

                    v_uint16 ui_lvec, ui_avec, ui_bvec;
                    trilinearPackedInterpolate(uirvec, uigvec, uibvec, LABLUVLUTs16.RGB2LabLUT_s16, ui_lvec, ui_avec, ui_bvec);
                    v_int16 i_lvec = v_reinterpret_as_s16(ui_lvec);
                    v_int16 i_avec = v_reinterpret_as_s16(ui_avec);
                    v_int16 i_bvec = v_reinterpret_as_s16(ui_bvec);

                    /* float L = iL*1.0f/LAB_BASE, a = ia*1.0f/LAB_BASE, b = ib*1.0f/LAB_BASE; */
                    v_int32 i_lvec0, i_avec0, i_bvec0, i_lvec1, i_avec1, i_bvec1;
                    v_expand(i_lvec, i_lvec0, i_lvec1);
                    v_expand(i_avec, i_avec0, i_avec1);
                    v_expand(i_bvec, i_bvec0, i_bvec1);

                    v_float32 l_vec0, a_vec0, b_vec0, l_vec1, a_vec1, b_vec1;
                    l_vec0 = v_cvt_f32(i_lvec0); l_vec1 = v_cvt_f32(i_lvec1);
                    a_vec0 = v_cvt_f32(i_avec0); a_vec1 = v_cvt_f32(i_avec1);
                    b_vec0 = v_cvt_f32(i_bvec0); b_vec1 = v_cvt_f32(i_bvec1);

                    /* dst[i] = L*100.0f */
                    v_float32 v100dBase = vx_setall_f32(100.0f/LAB_BASE);
                    l_vec0 = l_vec0*v100dBase;
                    l_vec1 = l_vec1*v100dBase;
                    /*
                    dst[i + 1] = a*256.0f - 128.0f;
                    dst[i + 2] = b*256.0f - 128.0f;
                    */
                    v_float32 v256dBase = vx_setall_f32(256.0f/LAB_BASE), vm128 = vx_setall_f32(-128.f);
                    a_vec0 = v_fma(a_vec0, v256dBase, vm128);
                    a_vec1 = v_fma(a_vec1, v256dBase, vm128);
                    b_vec0 = v_fma(b_vec0, v256dBase, vm128);
                    b_vec1 = v_fma(b_vec1, v256dBase, vm128);

                    v_store_interleave(dst + i + 0*vsize, l_vec0, a_vec0, b_vec0);
                    v_store_interleave(dst + i + 3*vsize, l_vec1, a_vec1, b_vec1);
                }
                vx_cleanup();
            }
#endif // CV_SIMD

            for(; i < n; i += 3, src += scn)
            {
                float R = clip(src[bIdx]);
                float G = clip(src[1]);
                float B = clip(src[bIdx^2]);

                int iR = cvRound(R*LAB_BASE), iG = cvRound(G*LAB_BASE), iB = cvRound(B*LAB_BASE);
                int iL, ia, ib;
                trilinearInterpolate(iR, iG, iB, LABLUVLUTs16.RGB2LabLUT_s16, iL, ia, ib);
                float L = iL*1.0f/LAB_BASE, a = ia*1.0f/LAB_BASE, b = ib*1.0f/LAB_BASE;

                dst[i] = L*100.0f;
                dst[i + 1] = a*256.0f - 128.0f;
                dst[i + 2] = b*256.0f - 128.0f;
            }
        }
        else
        {
            static const float _a = (softfloat(16) / softfloat(116));
            for (; i < n; i += 3, src += scn )
            {
                float R = clip(src[0]);
                float G = clip(src[1]);
                float B = clip(src[2]);

                if (gammaTab)
                {
                    R = splineInterpolate(R * gscale, gammaTab, GAMMA_TAB_SIZE);
                    G = splineInterpolate(G * gscale, gammaTab, GAMMA_TAB_SIZE);
                    B = splineInterpolate(B * gscale, gammaTab, GAMMA_TAB_SIZE);
                }
                float X = R*C0 + G*C1 + B*C2;
                float Y = R*C3 + G*C4 + B*C5;
                float Z = R*C6 + G*C7 + B*C8;
                // 7.787f = (29/3)^3/(29*4), 0.008856f = (6/29)^3, 903.3 = (29/3)^3
                float FX = X > 0.008856f ? cubeRoot(X) : (7.787f * X + _a);
                float FY = Y > 0.008856f ? cubeRoot(Y) : (7.787f * Y + _a);
                float FZ = Z > 0.008856f ? cubeRoot(Z) : (7.787f * Z + _a);

                float L = Y > 0.008856f ? (116.f * FY - 16.f) : (903.3f * Y);
                float a = 500.f * (FX - FY);
                float b = 200.f * (FY - FZ);

                dst[i] = L;
                dst[i + 1] = a;
                dst[i + 2] = b;
            }
        }
    }

    int srccn;
    float coeffs[9];
    bool srgb;
    bool useInterpolation;
    int blueIdx;
};


// Performs conversion in floats
struct Lab2RGBfloat
{
    typedef float channel_type;

    Lab2RGBfloat( int _dstcn, int _blueIdx, const float* _coeffs,
              const float* _whitept, bool _srgb )
    : dstcn(_dstcn), srgb(_srgb), blueIdx(_blueIdx)
    {
        initLabTabs();

        softdouble whitePt[3];
        for(int i = 0; i < 3; i++)
            if(_whitept)
                whitePt[i] = softdouble((double)_whitept[i]);
            else
                whitePt[i] = D65[i];

        for( int i = 0; i < 3; i++ )
        {
            softdouble c[3];
            for(int j = 0; j < 3; j++)
                if(_coeffs)
                    c[j] = softdouble(_coeffs[i+j*3]);
                else
                    c[j] = XYZ2sRGB_D65[i+j*3];

            coeffs[i+(blueIdx^2)*3] = (float)(c[0]*whitePt[i]);
            coeffs[i+3]             = (float)(c[1]*whitePt[i]);
            coeffs[i+blueIdx*3]     = (float)(c[2]*whitePt[i]);
        }

        lThresh = softfloat(8); // 0.008856f * 903.3f  = (6/29)^3*(29/3)^3 = 8
        fThresh = softfloat(6)/softfloat(29); // 7.787f * 0.008856f + 16.0f / 116.0f = 6/29

        #if CV_SSE2
        haveSIMD = checkHardwareSupport(CV_CPU_SSE2);
        #endif
    }

    #if CV_SSE2
    void process(__m128& v_li0, __m128& v_li1, __m128& v_ai0,
                 __m128& v_ai1, __m128& v_bi0, __m128& v_bi1) const
    {
        // 903.3 = (29/3)^3, 7.787 = (29/3)^3/(29*4)
        __m128 v_y00 = _mm_mul_ps(v_li0, _mm_set1_ps(1.0f/903.3f));
        __m128 v_y01 = _mm_mul_ps(v_li1, _mm_set1_ps(1.0f/903.3f));
        __m128 v_fy00 = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(7.787f), v_y00), _mm_set1_ps(16.0f/116.0f));
        __m128 v_fy01 = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(7.787f), v_y01), _mm_set1_ps(16.0f/116.0f));

        __m128 v_fy10 = _mm_mul_ps(_mm_add_ps(v_li0, _mm_set1_ps(16.0f)), _mm_set1_ps(1.0f/116.0f));
        __m128 v_fy11 = _mm_mul_ps(_mm_add_ps(v_li1, _mm_set1_ps(16.0f)), _mm_set1_ps(1.0f/116.0f));
        __m128 v_y10 = _mm_mul_ps(_mm_mul_ps(v_fy10, v_fy10), v_fy10);
        __m128 v_y11 = _mm_mul_ps(_mm_mul_ps(v_fy11, v_fy11), v_fy11);

        __m128 v_cmpli0 = _mm_cmple_ps(v_li0, _mm_set1_ps(lThresh));
        __m128 v_cmpli1 = _mm_cmple_ps(v_li1, _mm_set1_ps(lThresh));
        v_y00 = _mm_and_ps(v_cmpli0, v_y00);
        v_y01 = _mm_and_ps(v_cmpli1, v_y01);
        v_fy00 = _mm_and_ps(v_cmpli0, v_fy00);
        v_fy01 = _mm_and_ps(v_cmpli1, v_fy01);
        v_y10 = _mm_andnot_ps(v_cmpli0, v_y10);
        v_y11 = _mm_andnot_ps(v_cmpli1, v_y11);
        v_fy10 = _mm_andnot_ps(v_cmpli0, v_fy10);
        v_fy11 = _mm_andnot_ps(v_cmpli1, v_fy11);
        __m128 v_y0 = _mm_or_ps(v_y00, v_y10);
        __m128 v_y1 = _mm_or_ps(v_y01, v_y11);
        __m128 v_fy0 = _mm_or_ps(v_fy00, v_fy10);
        __m128 v_fy1 = _mm_or_ps(v_fy01, v_fy11);

        __m128 v_fxz00 = _mm_add_ps(v_fy0, _mm_mul_ps(v_ai0, _mm_set1_ps(0.002f)));
        __m128 v_fxz01 = _mm_add_ps(v_fy1, _mm_mul_ps(v_ai1, _mm_set1_ps(0.002f)));
        __m128 v_fxz10 = _mm_sub_ps(v_fy0, _mm_mul_ps(v_bi0, _mm_set1_ps(0.005f)));
        __m128 v_fxz11 = _mm_sub_ps(v_fy1, _mm_mul_ps(v_bi1, _mm_set1_ps(0.005f)));

        __m128 v_fxz000 = _mm_mul_ps(_mm_sub_ps(v_fxz00, _mm_set1_ps(16.0f/116.0f)), _mm_set1_ps(1.0f/7.787f));
        __m128 v_fxz001 = _mm_mul_ps(_mm_sub_ps(v_fxz01, _mm_set1_ps(16.0f/116.0f)), _mm_set1_ps(1.0f/7.787f));
        __m128 v_fxz010 = _mm_mul_ps(_mm_sub_ps(v_fxz10, _mm_set1_ps(16.0f/116.0f)), _mm_set1_ps(1.0f/7.787f));
        __m128 v_fxz011 = _mm_mul_ps(_mm_sub_ps(v_fxz11, _mm_set1_ps(16.0f/116.0f)), _mm_set1_ps(1.0f/7.787f));

        __m128 v_fxz100 = _mm_mul_ps(_mm_mul_ps(v_fxz00, v_fxz00), v_fxz00);
        __m128 v_fxz101 = _mm_mul_ps(_mm_mul_ps(v_fxz01, v_fxz01), v_fxz01);
        __m128 v_fxz110 = _mm_mul_ps(_mm_mul_ps(v_fxz10, v_fxz10), v_fxz10);
        __m128 v_fxz111 = _mm_mul_ps(_mm_mul_ps(v_fxz11, v_fxz11), v_fxz11);

        __m128 v_cmpfxz00 = _mm_cmple_ps(v_fxz00, _mm_set1_ps(fThresh));
        __m128 v_cmpfxz01 = _mm_cmple_ps(v_fxz01, _mm_set1_ps(fThresh));
        __m128 v_cmpfxz10 = _mm_cmple_ps(v_fxz10, _mm_set1_ps(fThresh));
        __m128 v_cmpfxz11 = _mm_cmple_ps(v_fxz11, _mm_set1_ps(fThresh));
        v_fxz000 = _mm_and_ps(v_cmpfxz00, v_fxz000);
        v_fxz001 = _mm_and_ps(v_cmpfxz01, v_fxz001);
        v_fxz010 = _mm_and_ps(v_cmpfxz10, v_fxz010);
        v_fxz011 = _mm_and_ps(v_cmpfxz11, v_fxz011);
        v_fxz100 = _mm_andnot_ps(v_cmpfxz00, v_fxz100);
        v_fxz101 = _mm_andnot_ps(v_cmpfxz01, v_fxz101);
        v_fxz110 = _mm_andnot_ps(v_cmpfxz10, v_fxz110);
        v_fxz111 = _mm_andnot_ps(v_cmpfxz11, v_fxz111);
        __m128 v_x0 = _mm_or_ps(v_fxz000, v_fxz100);
        __m128 v_x1 = _mm_or_ps(v_fxz001, v_fxz101);
        __m128 v_z0 = _mm_or_ps(v_fxz010, v_fxz110);
        __m128 v_z1 = _mm_or_ps(v_fxz011, v_fxz111);

        __m128 v_ro0 = _mm_mul_ps(_mm_set1_ps(coeffs[0]), v_x0);
        __m128 v_ro1 = _mm_mul_ps(_mm_set1_ps(coeffs[0]), v_x1);
        __m128 v_go0 = _mm_mul_ps(_mm_set1_ps(coeffs[3]), v_x0);
        __m128 v_go1 = _mm_mul_ps(_mm_set1_ps(coeffs[3]), v_x1);
        __m128 v_bo0 = _mm_mul_ps(_mm_set1_ps(coeffs[6]), v_x0);
        __m128 v_bo1 = _mm_mul_ps(_mm_set1_ps(coeffs[6]), v_x1);
        v_ro0 = _mm_add_ps(v_ro0, _mm_mul_ps(_mm_set1_ps(coeffs[1]), v_y0));
        v_ro1 = _mm_add_ps(v_ro1, _mm_mul_ps(_mm_set1_ps(coeffs[1]), v_y1));
        v_go0 = _mm_add_ps(v_go0, _mm_mul_ps(_mm_set1_ps(coeffs[4]), v_y0));
        v_go1 = _mm_add_ps(v_go1, _mm_mul_ps(_mm_set1_ps(coeffs[4]), v_y1));
        v_bo0 = _mm_add_ps(v_bo0, _mm_mul_ps(_mm_set1_ps(coeffs[7]), v_y0));
        v_bo1 = _mm_add_ps(v_bo1, _mm_mul_ps(_mm_set1_ps(coeffs[7]), v_y1));
        v_ro0 = _mm_add_ps(v_ro0, _mm_mul_ps(_mm_set1_ps(coeffs[2]), v_z0));
        v_ro1 = _mm_add_ps(v_ro1, _mm_mul_ps(_mm_set1_ps(coeffs[2]), v_z1));
        v_go0 = _mm_add_ps(v_go0, _mm_mul_ps(_mm_set1_ps(coeffs[5]), v_z0));
        v_go1 = _mm_add_ps(v_go1, _mm_mul_ps(_mm_set1_ps(coeffs[5]), v_z1));
        v_bo0 = _mm_add_ps(v_bo0, _mm_mul_ps(_mm_set1_ps(coeffs[8]), v_z0));
        v_bo1 = _mm_add_ps(v_bo1, _mm_mul_ps(_mm_set1_ps(coeffs[8]), v_z1));

        v_li0 = _mm_min_ps(_mm_max_ps(v_ro0, _mm_setzero_ps()), _mm_set1_ps(1.0f));
        v_li1 = _mm_min_ps(_mm_max_ps(v_ro1, _mm_setzero_ps()), _mm_set1_ps(1.0f));
        v_ai0 = _mm_min_ps(_mm_max_ps(v_go0, _mm_setzero_ps()), _mm_set1_ps(1.0f));
        v_ai1 = _mm_min_ps(_mm_max_ps(v_go1, _mm_setzero_ps()), _mm_set1_ps(1.0f));
        v_bi0 = _mm_min_ps(_mm_max_ps(v_bo0, _mm_setzero_ps()), _mm_set1_ps(1.0f));
        v_bi1 = _mm_min_ps(_mm_max_ps(v_bo1, _mm_setzero_ps()), _mm_set1_ps(1.0f));
    }
    #endif

    void operator()(const float* src, float* dst, int n) const
    {
        int i = 0, dcn = dstcn;
        const float* gammaTab = srgb ? sRGBInvGammaTab : 0;
        float gscale = GammaTabScale;
        float C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2],
        C3 = coeffs[3], C4 = coeffs[4], C5 = coeffs[5],
        C6 = coeffs[6], C7 = coeffs[7], C8 = coeffs[8];
        float alpha = ColorChannel<float>::max();
        n *= 3;

        #if CV_SSE2
        if (haveSIMD)
        {
            for (; i <= n - 24; i += 24, dst += dcn * 8)
            {
                __m128 v_li0 = _mm_loadu_ps(src + i +  0);
                __m128 v_li1 = _mm_loadu_ps(src + i +  4);
                __m128 v_ai0 = _mm_loadu_ps(src + i +  8);
                __m128 v_ai1 = _mm_loadu_ps(src + i + 12);
                __m128 v_bi0 = _mm_loadu_ps(src + i + 16);
                __m128 v_bi1 = _mm_loadu_ps(src + i + 20);

                _mm_deinterleave_ps(v_li0, v_li1, v_ai0, v_ai1, v_bi0, v_bi1);

                process(v_li0, v_li1, v_ai0, v_ai1, v_bi0, v_bi1);

                if (gammaTab)
                {
                    __m128 v_gscale = _mm_set1_ps(gscale);
                    v_li0 = _mm_mul_ps(v_li0, v_gscale);
                    v_li1 = _mm_mul_ps(v_li1, v_gscale);
                    v_ai0 = _mm_mul_ps(v_ai0, v_gscale);
                    v_ai1 = _mm_mul_ps(v_ai1, v_gscale);
                    v_bi0 = _mm_mul_ps(v_bi0, v_gscale);
                    v_bi1 = _mm_mul_ps(v_bi1, v_gscale);

                    splineInterpolate(v_li0, gammaTab, GAMMA_TAB_SIZE);
                    splineInterpolate(v_li1, gammaTab, GAMMA_TAB_SIZE);
                    splineInterpolate(v_ai0, gammaTab, GAMMA_TAB_SIZE);
                    splineInterpolate(v_ai1, gammaTab, GAMMA_TAB_SIZE);
                    splineInterpolate(v_bi0, gammaTab, GAMMA_TAB_SIZE);
                    splineInterpolate(v_bi1, gammaTab, GAMMA_TAB_SIZE);
                }

                if( dcn == 4 )
                {
                    __m128 v_a0 = _mm_set1_ps(alpha);
                    __m128 v_a1 = _mm_set1_ps(alpha);
                    _mm_interleave_ps(v_li0, v_li1, v_ai0, v_ai1, v_bi0, v_bi1, v_a0, v_a1);

                    _mm_storeu_ps(dst +  0, v_li0);
                    _mm_storeu_ps(dst +  4, v_li1);
                    _mm_storeu_ps(dst +  8, v_ai0);
                    _mm_storeu_ps(dst + 12, v_ai1);
                    _mm_storeu_ps(dst + 16, v_bi0);
                    _mm_storeu_ps(dst + 20, v_bi1);
                    _mm_storeu_ps(dst + 24, v_a0);
                    _mm_storeu_ps(dst + 28, v_a1);
                }
                else
                {
                    _mm_interleave_ps(v_li0, v_li1, v_ai0, v_ai1, v_bi0, v_bi1);

                    _mm_storeu_ps(dst +  0, v_li0);
                    _mm_storeu_ps(dst +  4, v_li1);
                    _mm_storeu_ps(dst +  8, v_ai0);
                    _mm_storeu_ps(dst + 12, v_ai1);
                    _mm_storeu_ps(dst + 16, v_bi0);
                    _mm_storeu_ps(dst + 20, v_bi1);
                }
            }
        }
        #endif
        for (; i < n; i += 3, dst += dcn)
        {
            float li = src[i];
            float ai = src[i + 1];
            float bi = src[i + 2];

            // 903.3 = (29/3)^3, 7.787 = (29/3)^3/(29*4)
            float y, fy;
            if (li <= lThresh)
            {
                y = li / 903.3f;
                fy = 7.787f * y + 16.0f / 116.0f;
            }
            else
            {
                fy = (li + 16.0f) / 116.0f;
                y = fy * fy * fy;
            }

            float fxz[] = { ai / 500.0f + fy, fy - bi / 200.0f };

            for (int j = 0; j < 2; j++)
                if (fxz[j] <= fThresh)
                    fxz[j] = (fxz[j] - 16.0f / 116.0f) / 7.787f;
                else
                    fxz[j] = fxz[j] * fxz[j] * fxz[j];

            float x = fxz[0], z = fxz[1];
            float ro = C0 * x + C1 * y + C2 * z;
            float go = C3 * x + C4 * y + C5 * z;
            float bo = C6 * x + C7 * y + C8 * z;
            ro = clip(ro);
            go = clip(go);
            bo = clip(bo);

            if (gammaTab)
            {
                ro = splineInterpolate(ro * gscale, gammaTab, GAMMA_TAB_SIZE);
                go = splineInterpolate(go * gscale, gammaTab, GAMMA_TAB_SIZE);
                bo = splineInterpolate(bo * gscale, gammaTab, GAMMA_TAB_SIZE);
            }

            dst[0] = ro, dst[1] = go, dst[2] = bo;
            if( dcn == 4 )
                dst[3] = alpha;
        }
    }

    int dstcn;
    float coeffs[9];
    bool srgb;
    float lThresh;
    float fThresh;
    #if CV_SSE2
    bool haveSIMD;
    #endif
    int blueIdx;
};


// Performs conversion in integers
struct Lab2RGBinteger
{
    typedef uchar channel_type;

    static const int base_shift = 14;
    static const int BASE = (1 << base_shift);
    // lThresh == (6/29)^3 * (29/3)^3 * BASE/100
    static const int lThresh = 1311;
    // fThresh == ((29/3)^3/(29*4) * (6/29)^3 + 16/116)*BASE
    static const int fThresh = 3390;
    // base16_116 == BASE*16/116
    static const int base16_116 = 2260;
    static const int shift = lab_shift+(base_shift-inv_gamma_shift);

    Lab2RGBinteger( int _dstcn, int blueIdx, const float* _coeffs,
                    const float* _whitept, bool srgb )
    : dstcn(_dstcn)
    {
        softdouble whitePt[3];
        for(int i = 0; i < 3; i++)
            if(_whitept)
                whitePt[i] = softdouble(_whitept[i]);
            else
                whitePt[i] = D65[i];

        static const softdouble lshift(1 << lab_shift);
        for(int i = 0; i < 3; i++)
        {
            softdouble c[3];
            for(int j = 0; j < 3; j++)
                if(_coeffs)
                    c[j] = softdouble(_coeffs[i+j*3]);
                else
                    c[j] = XYZ2sRGB_D65[i+j*3];

            coeffs[i+(blueIdx)*3]   = cvRound(lshift*c[0]*whitePt[i]);
            coeffs[i+3]             = cvRound(lshift*c[1]*whitePt[i]);
            coeffs[i+(blueIdx^2)*3] = cvRound(lshift*c[2]*whitePt[i]);
        }

        tab = srgb ? sRGBInvGammaTab_b : linearInvGammaTab_b;
    }

    // L, a, b should be in their natural range
    inline void process(const uchar LL, const uchar aa, const uchar bb, int& ro, int& go, int& bo) const
    {
        int x, y, z;
        int ify;

        y   = LabToYF_b[LL*2  ];
        ify = LabToYF_b[LL*2+1];

        //float fxz[] = { ai / 500.0f + fy, fy - bi / 200.0f };
        int adiv, bdiv;
        //adiv = aa*BASE/500 - 128*BASE/500, bdiv = bb*BASE/200 - 128*BASE/200;
        //approximations with reasonable precision
        adiv = ((5*aa*53687 + (1 << 7)) >> 13) - 128*BASE/500;
        bdiv = ((  bb*41943 + (1 << 4)) >>  9) - 128*BASE/200+1;

        int ifxz[] = {ify + adiv, ify - bdiv};

        for(int k = 0; k < 2; k++)
        {
            int& v = ifxz[k];
            v = abToXZ_b[v-minABvalue];
        }
        x = ifxz[0]; /* y = y */; z = ifxz[1];

        int C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2];
        int C3 = coeffs[3], C4 = coeffs[4], C5 = coeffs[5];
        int C6 = coeffs[6], C7 = coeffs[7], C8 = coeffs[8];

        ro = CV_DESCALE(C0 * x + C1 * y + C2 * z, shift);
        go = CV_DESCALE(C3 * x + C4 * y + C5 * z, shift);
        bo = CV_DESCALE(C6 * x + C7 * y + C8 * z, shift);

        ro = max(0, min((int)INV_GAMMA_TAB_SIZE-1, ro));
        go = max(0, min((int)INV_GAMMA_TAB_SIZE-1, go));
        bo = max(0, min((int)INV_GAMMA_TAB_SIZE-1, bo));

        ro = tab[ro];
        go = tab[go];
        bo = tab[bo];
    }

    // L, a, b should be in their natural range
    inline void processLabToXYZ(const v_uint8x16& lv, const v_uint8x16& av, const v_uint8x16& bv,
                                v_int32x4& xiv00, v_int32x4& yiv00, v_int32x4& ziv00,
                                v_int32x4& xiv01, v_int32x4& yiv01, v_int32x4& ziv01,
                                v_int32x4& xiv10, v_int32x4& yiv10, v_int32x4& ziv10,
                                v_int32x4& xiv11, v_int32x4& yiv11, v_int32x4& ziv11) const
    {
        v_uint16x8 lv0, lv1;
        v_expand(lv, lv0, lv1);
        // Load Y and IFY values from lookup-table
        // y = LabToYF_b[L_value*2], ify = LabToYF_b[L_value*2 + 1]
        // LabToYF_b[i*2  ] = y;   // 2260 <= y <= BASE
        // LabToYF_b[i*2+1] = ify; // 0 <= ify <= BASE
        uint16_t CV_DECL_ALIGNED(16) v_lv0[8], v_lv1[8];
        v_store_aligned(v_lv0, (lv0 << 1)); v_store_aligned(v_lv1, (lv1 << 1));
        v_int16x8 ify0, ify1;

        yiv00 = v_int32x4(LabToYF_b[v_lv0[0]  ], LabToYF_b[v_lv0[1]  ], LabToYF_b[v_lv0[2]  ], LabToYF_b[v_lv0[3]  ]);
        yiv01 = v_int32x4(LabToYF_b[v_lv0[4]  ], LabToYF_b[v_lv0[5]  ], LabToYF_b[v_lv0[6]  ], LabToYF_b[v_lv0[7]  ]);
        yiv10 = v_int32x4(LabToYF_b[v_lv1[0]  ], LabToYF_b[v_lv1[1]  ], LabToYF_b[v_lv1[2]  ], LabToYF_b[v_lv1[3]  ]);
        yiv11 = v_int32x4(LabToYF_b[v_lv1[4]  ], LabToYF_b[v_lv1[5]  ], LabToYF_b[v_lv1[6]  ], LabToYF_b[v_lv1[7]  ]);

        ify0 = v_int16x8(LabToYF_b[v_lv0[0]+1], LabToYF_b[v_lv0[1]+1], LabToYF_b[v_lv0[2]+1], LabToYF_b[v_lv0[3]+1],
                         LabToYF_b[v_lv0[4]+1], LabToYF_b[v_lv0[5]+1], LabToYF_b[v_lv0[6]+1], LabToYF_b[v_lv0[7]+1]);
        ify1 = v_int16x8(LabToYF_b[v_lv1[0]+1], LabToYF_b[v_lv1[1]+1], LabToYF_b[v_lv1[2]+1], LabToYF_b[v_lv1[3]+1],
                         LabToYF_b[v_lv1[4]+1], LabToYF_b[v_lv1[5]+1], LabToYF_b[v_lv1[6]+1], LabToYF_b[v_lv1[7]+1]);

        v_int16x8 adiv0, adiv1, bdiv0, bdiv1;
        v_uint16x8 av0, av1, bv0, bv1;
        v_expand(av, av0, av1); v_expand(bv, bv0, bv1);
        //adiv = aa*BASE/500 - 128*BASE/500, bdiv = bb*BASE/200 - 128*BASE/200;
        //approximations with reasonable precision
        v_uint16x8 mulA = v_setall_u16(53687);
        v_uint32x4 ma00, ma01, ma10, ma11;
        v_uint32x4 addA = v_setall_u32(1 << 7);
        v_mul_expand((av0 + (av0 << 2)), mulA, ma00, ma01);
        v_mul_expand((av1 + (av1 << 2)), mulA, ma10, ma11);
        adiv0 = v_reinterpret_as_s16(v_pack(((ma00 + addA) >> 13), ((ma01 + addA) >> 13)));
        adiv1 = v_reinterpret_as_s16(v_pack(((ma10 + addA) >> 13), ((ma11 + addA) >> 13)));

        v_uint16x8 mulB = v_setall_u16(41943);
        v_uint32x4 mb00, mb01, mb10, mb11;
        v_uint32x4 addB = v_setall_u32(1 << 4);
        v_mul_expand(bv0, mulB, mb00, mb01);
        v_mul_expand(bv1, mulB, mb10, mb11);
        bdiv0 = v_reinterpret_as_s16(v_pack((mb00 + addB) >> 9, (mb01 + addB) >> 9));
        bdiv1 = v_reinterpret_as_s16(v_pack((mb10 + addB) >> 9, (mb11 + addB) >> 9));

        // 0 <= adiv <= 8356, 0 <= bdiv <= 20890
        /* x = ifxz[0]; y = y; z = ifxz[1]; */
        v_uint16x8 xiv0, xiv1, ziv0, ziv1;
        v_int16x8 vSubA = v_setall_s16(-128*BASE/500 - minABvalue), vSubB = v_setall_s16(128*BASE/200-1 - minABvalue);

        // int ifxz[] = {ify + adiv, ify - bdiv};
        // ifxz[k] = abToXZ_b[ifxz[k]-minABvalue];
        xiv0 = v_reinterpret_as_u16(v_add_wrap(v_add_wrap(ify0, adiv0), vSubA));
        xiv1 = v_reinterpret_as_u16(v_add_wrap(v_add_wrap(ify1, adiv1), vSubA));
        ziv0 = v_reinterpret_as_u16(v_add_wrap(v_sub_wrap(ify0, bdiv0), vSubB));
        ziv1 = v_reinterpret_as_u16(v_add_wrap(v_sub_wrap(ify1, bdiv1), vSubB));

        uint16_t CV_DECL_ALIGNED(16) v_x0[8], v_x1[8], v_z0[8], v_z1[8];
        v_store_aligned(v_x0, xiv0 ); v_store_aligned(v_x1, xiv1 );
        v_store_aligned(v_z0, ziv0 ); v_store_aligned(v_z1, ziv1 );

        xiv00 = v_int32x4(abToXZ_b[v_x0[0]], abToXZ_b[v_x0[1]], abToXZ_b[v_x0[2]], abToXZ_b[v_x0[3]]);
        xiv01 = v_int32x4(abToXZ_b[v_x0[4]], abToXZ_b[v_x0[5]], abToXZ_b[v_x0[6]], abToXZ_b[v_x0[7]]);
        xiv10 = v_int32x4(abToXZ_b[v_x1[0]], abToXZ_b[v_x1[1]], abToXZ_b[v_x1[2]], abToXZ_b[v_x1[3]]);
        xiv11 = v_int32x4(abToXZ_b[v_x1[4]], abToXZ_b[v_x1[5]], abToXZ_b[v_x1[6]], abToXZ_b[v_x1[7]]);
        ziv00 = v_int32x4(abToXZ_b[v_z0[0]], abToXZ_b[v_z0[1]], abToXZ_b[v_z0[2]], abToXZ_b[v_z0[3]]);
        ziv01 = v_int32x4(abToXZ_b[v_z0[4]], abToXZ_b[v_z0[5]], abToXZ_b[v_z0[6]], abToXZ_b[v_z0[7]]);
        ziv10 = v_int32x4(abToXZ_b[v_z1[0]], abToXZ_b[v_z1[1]], abToXZ_b[v_z1[2]], abToXZ_b[v_z1[3]]);
        ziv11 = v_int32x4(abToXZ_b[v_z1[4]], abToXZ_b[v_z1[5]], abToXZ_b[v_z1[6]], abToXZ_b[v_z1[7]]);
        // abToXZ_b[i-minABvalue] = v; // -1335 <= v <= 88231
    }

    void operator()(const float* src, float* dst, int n) const
    {
        int dcn = dstcn;
        float alpha = ColorChannel<float>::max();

        int i = 0;

#if CV_SIMD128
        if(enablePackedLab)
        {
            v_float32x4 vldiv  = v_setall_f32(256.f/100.0f);
            v_float32x4 vf255  = v_setall_f32(255.f);
            static const int nPixels = 16;
            for(; i <= n*3-3*nPixels; i += 3*nPixels, dst += dcn*nPixels)
            {
                /*
                int L = saturate_cast<int>(src[i]*BASE/100.0f);
                int a = saturate_cast<int>(src[i + 1]*BASE/256);
                int b = saturate_cast<int>(src[i + 2]*BASE/256);
                */
                v_float32x4 vl[4], va[4], vb[4];
                for(int k = 0; k < 4; k++)
                {
                    v_load_deinterleave(src + i + k*3*4, vl[k], va[k], vb[k]);
                    vl[k] *= vldiv;
                }

                v_int32x4 ivl[4], iva[4], ivb[4];
                for(int k = 0; k < 4; k++)
                {
                    ivl[k] = v_round(vl[k]), iva[k] = v_round(va[k]), ivb[k] = v_round(vb[k]);
                }
                v_int16x8 ivl16[2], iva16[2], ivb16[2];
                ivl16[0] = v_pack(ivl[0], ivl[1]); iva16[0] = v_pack(iva[0], iva[1]); ivb16[0] = v_pack(ivb[0], ivb[1]);
                ivl16[1] = v_pack(ivl[2], ivl[3]); iva16[1] = v_pack(iva[2], iva[3]); ivb16[1] = v_pack(ivb[2], ivb[3]);
                v_uint8x16 ivl8, iva8, ivb8;
                ivl8 = v_reinterpret_as_u8(v_pack(ivl16[0], ivl16[1]));
                iva8 = v_reinterpret_as_u8(v_pack(iva16[0], iva16[1]));
                ivb8 = v_reinterpret_as_u8(v_pack(ivb16[0], ivb16[1]));

                v_int32x4 ixv[4], iyv[4], izv[4];

                processLabToXYZ(ivl8, iva8, ivb8, ixv[0], iyv[0], izv[0],
                                                  ixv[1], iyv[1], izv[1],
                                                  ixv[2], iyv[2], izv[2],
                                                  ixv[3], iyv[3], izv[3]);
                /*
                ro = CV_DESCALE(C0 * x + C1 * y + C2 * z, shift);
                go = CV_DESCALE(C3 * x + C4 * y + C5 * z, shift);
                bo = CV_DESCALE(C6 * x + C7 * y + C8 * z, shift);
                */
                v_int32x4 C0 = v_setall_s32(coeffs[0]), C1 = v_setall_s32(coeffs[1]), C2 = v_setall_s32(coeffs[2]);
                v_int32x4 C3 = v_setall_s32(coeffs[3]), C4 = v_setall_s32(coeffs[4]), C5 = v_setall_s32(coeffs[5]);
                v_int32x4 C6 = v_setall_s32(coeffs[6]), C7 = v_setall_s32(coeffs[7]), C8 = v_setall_s32(coeffs[8]);
                v_int32x4 descaleShift = v_setall_s32(1 << (shift-1)), tabsz = v_setall_s32((int)INV_GAMMA_TAB_SIZE-1);
                for(int k = 0; k < 4; k++)
                {
                    v_int32x4 i_r, i_g, i_b;
                    v_uint32x4 r_vecs, g_vecs, b_vecs;
                    i_r = (ixv[k]*C0 + iyv[k]*C1 + izv[k]*C2 + descaleShift) >> shift;
                    i_g = (ixv[k]*C3 + iyv[k]*C4 + izv[k]*C5 + descaleShift) >> shift;
                    i_b = (ixv[k]*C6 + iyv[k]*C7 + izv[k]*C8 + descaleShift) >> shift;

                    //limit indices in table and then substitute
                    //ro = tab[ro]; go = tab[go]; bo = tab[bo];
                    int32_t CV_DECL_ALIGNED(16) rshifts[4], gshifts[4], bshifts[4];
                    v_int32x4 rs = v_max(v_setzero_s32(), v_min(tabsz, i_r));
                    v_int32x4 gs = v_max(v_setzero_s32(), v_min(tabsz, i_g));
                    v_int32x4 bs = v_max(v_setzero_s32(), v_min(tabsz, i_b));

                    v_store_aligned(rshifts, rs);
                    v_store_aligned(gshifts, gs);
                    v_store_aligned(bshifts, bs);

                    r_vecs = v_uint32x4(tab[rshifts[0]], tab[rshifts[1]], tab[rshifts[2]], tab[rshifts[3]]);
                    g_vecs = v_uint32x4(tab[gshifts[0]], tab[gshifts[1]], tab[gshifts[2]], tab[gshifts[3]]);
                    b_vecs = v_uint32x4(tab[bshifts[0]], tab[bshifts[1]], tab[bshifts[2]], tab[bshifts[3]]);

                    v_float32x4 v_r, v_g, v_b;
                    v_r = v_cvt_f32(v_reinterpret_as_s32(r_vecs))/vf255;
                    v_g = v_cvt_f32(v_reinterpret_as_s32(g_vecs))/vf255;
                    v_b = v_cvt_f32(v_reinterpret_as_s32(b_vecs))/vf255;

                    if(dcn == 4)
                    {
                        v_store_interleave(dst + k*dcn*4, v_b, v_g, v_r, v_setall_f32(alpha));
                    }
                    else // dcn == 3
                    {
                        v_store_interleave(dst + k*dcn*4, v_b, v_g, v_r);
                    }
                }
            }
            vx_cleanup();
        }
#endif

        for(; i < n*3; i += 3, dst += dcn)
        {
            int ro, go, bo;
            process((uchar)(src[i + 0]*255.f/100.f), (uchar)src[i + 1], (uchar)src[i + 2], ro, go, bo);

            dst[0] = bo/255.f;
            dst[1] = go/255.f;
            dst[2] = ro/255.f;
            if(dcn == 4)
                dst[3] = alpha;
        }
    }

    void operator()(const uchar* src, uchar* dst, int n) const
    {
        int i, dcn = dstcn;
        uchar alpha = ColorChannel<uchar>::max();
        i = 0;

#if CV_SIMD128
        if(enablePackedLab)
        {
            static const int nPixels = 8*2;
            for(; i <= n*3-3*nPixels; i += 3*nPixels, dst += dcn*nPixels)
            {
                /*
                    int L = src[i + 0];
                    int a = src[i + 1];
                    int b = src[i + 2];
                */
                v_uint8x16 u8l, u8a, u8b;
                v_load_deinterleave(src + i, u8l, u8a, u8b);

                v_int32x4 xiv[4], yiv[4], ziv[4];
                processLabToXYZ(u8l, u8a, u8b, xiv[0], yiv[0], ziv[0],
                                               xiv[1], yiv[1], ziv[1],
                                               xiv[2], yiv[2], ziv[2],
                                               xiv[3], yiv[3], ziv[3]);
                /*
                        ro = CV_DESCALE(C0 * x + C1 * y + C2 * z, shift);
                        go = CV_DESCALE(C3 * x + C4 * y + C5 * z, shift);
                        bo = CV_DESCALE(C6 * x + C7 * y + C8 * z, shift);
                */
                v_int32x4 C0 = v_setall_s32(coeffs[0]), C1 = v_setall_s32(coeffs[1]), C2 = v_setall_s32(coeffs[2]);
                v_int32x4 C3 = v_setall_s32(coeffs[3]), C4 = v_setall_s32(coeffs[4]), C5 = v_setall_s32(coeffs[5]);
                v_int32x4 C6 = v_setall_s32(coeffs[6]), C7 = v_setall_s32(coeffs[7]), C8 = v_setall_s32(coeffs[8]);
                v_int32x4 descaleShift = v_setall_s32(1 << (shift-1));
                v_int32x4 tabsz = v_setall_s32((int)INV_GAMMA_TAB_SIZE-1);
                v_uint32x4 r_vecs[4], g_vecs[4], b_vecs[4];
                for(int k = 0; k < 4; k++)
                {
                    v_int32x4 i_r, i_g, i_b;
                    i_r = (xiv[k]*C0 + yiv[k]*C1 + ziv[k]*C2 + descaleShift) >> shift;
                    i_g = (xiv[k]*C3 + yiv[k]*C4 + ziv[k]*C5 + descaleShift) >> shift;
                    i_b = (xiv[k]*C6 + yiv[k]*C7 + ziv[k]*C8 + descaleShift) >> shift;

                    //limit indices in table and then substitute
                    //ro = tab[ro]; go = tab[go]; bo = tab[bo];
                    int32_t CV_DECL_ALIGNED(16) rshifts[4], gshifts[4], bshifts[4];
                    v_int32x4 rs = v_max(v_setzero_s32(), v_min(tabsz, i_r));
                    v_int32x4 gs = v_max(v_setzero_s32(), v_min(tabsz, i_g));
                    v_int32x4 bs = v_max(v_setzero_s32(), v_min(tabsz, i_b));

                    v_store_aligned(rshifts, rs);
                    v_store_aligned(gshifts, gs);
                    v_store_aligned(bshifts, bs);

                    r_vecs[k] = v_uint32x4(tab[rshifts[0]], tab[rshifts[1]], tab[rshifts[2]], tab[rshifts[3]]);
                    g_vecs[k] = v_uint32x4(tab[gshifts[0]], tab[gshifts[1]], tab[gshifts[2]], tab[gshifts[3]]);
                    b_vecs[k] = v_uint32x4(tab[bshifts[0]], tab[bshifts[1]], tab[bshifts[2]], tab[bshifts[3]]);
                }

                v_uint16x8 u_rvec0 = v_pack(r_vecs[0], r_vecs[1]), u_rvec1 = v_pack(r_vecs[2], r_vecs[3]);
                v_uint16x8 u_gvec0 = v_pack(g_vecs[0], g_vecs[1]), u_gvec1 = v_pack(g_vecs[2], g_vecs[3]);
                v_uint16x8 u_bvec0 = v_pack(b_vecs[0], b_vecs[1]), u_bvec1 = v_pack(b_vecs[2], b_vecs[3]);

                v_uint8x16 u8_b, u8_g, u8_r;
                u8_b = v_pack(u_bvec0, u_bvec1);
                u8_g = v_pack(u_gvec0, u_gvec1);
                u8_r = v_pack(u_rvec0, u_rvec1);

                if(dcn == 4)
                {
                    v_store_interleave(dst, u8_b, u8_g, u8_r, v_setall_u8(alpha));
                }
                else
                {
                    v_store_interleave(dst, u8_b, u8_g, u8_r);
                }
            }
        }
#endif

        for (; i < n*3; i += 3, dst += dcn)
        {
            int ro, go, bo;
            process(src[i + 0], src[i + 1], src[i + 2], ro, go, bo);

            dst[0] = saturate_cast<uchar>(bo);
            dst[1] = saturate_cast<uchar>(go);
            dst[2] = saturate_cast<uchar>(ro);
            if( dcn == 4 )
                dst[3] = alpha;
        }
    }

    int dstcn;
    int coeffs[9];
    ushort* tab;
};


struct Lab2RGB_f
{
    typedef float channel_type;

    Lab2RGB_f( int _dstcn, int _blueIdx, const float* _coeffs,
              const float* _whitept, bool _srgb )
    : fcvt(_dstcn, _blueIdx, _coeffs, _whitept, _srgb), dstcn(_dstcn)
    { }

    void operator()(const float* src, float* dst, int n) const
    {
        fcvt(src, dst, n);
    }

    Lab2RGBfloat fcvt;
    int dstcn;
};


struct Lab2RGB_b
{
    typedef uchar channel_type;

    Lab2RGB_b( int _dstcn, int _blueIdx, const float* _coeffs,
               const float* _whitept, bool _srgb )
    : fcvt(3, _blueIdx, _coeffs, _whitept, _srgb ), icvt(_dstcn, _blueIdx, _coeffs, _whitept, _srgb), dstcn(_dstcn)
    {
        #if CV_NEON
        v_scale_inv = vdupq_n_f32(100.f/255.f);
        v_scale = vdupq_n_f32(255.f);
        v_alpha = vdup_n_u8(ColorChannel<uchar>::max());
        v_128 = vdupq_n_f32(128.0f);
        #elif CV_SSE2
        v_scale = _mm_set1_ps(255.f);
        v_alpha = _mm_set1_ps(ColorChannel<uchar>::max());
        v_zero = _mm_setzero_si128();
        haveSIMD = checkHardwareSupport(CV_CPU_SSE2);
        #endif
    }

    #if CV_SSE2
    // 16s x 8
    void process(__m128i v_r, __m128i v_g, __m128i v_b,
                 const __m128& v_coeffs_, const __m128& v_res_,
                 float * buf) const
    {
        __m128 v_r0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(v_r, v_zero));
        __m128 v_g0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(v_g, v_zero));
        __m128 v_b0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(v_b, v_zero));

        __m128 v_r1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(v_r, v_zero));
        __m128 v_g1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(v_g, v_zero));
        __m128 v_b1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(v_b, v_zero));

        __m128 v_coeffs = v_coeffs_;
        __m128 v_res = v_res_;

        v_r0 = _mm_sub_ps(_mm_mul_ps(v_r0, v_coeffs), v_res);
        v_g1 = _mm_sub_ps(_mm_mul_ps(v_g1, v_coeffs), v_res);

        v_coeffs = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v_coeffs), 0x49));
        v_res = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v_res), 0x49));

        v_r1 = _mm_sub_ps(_mm_mul_ps(v_r1, v_coeffs), v_res);
        v_b0 = _mm_sub_ps(_mm_mul_ps(v_b0, v_coeffs), v_res);

        v_coeffs = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v_coeffs), 0x49));
        v_res = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v_res), 0x49));

        v_g0 = _mm_sub_ps(_mm_mul_ps(v_g0, v_coeffs), v_res);
        v_b1 = _mm_sub_ps(_mm_mul_ps(v_b1, v_coeffs), v_res);

        _mm_store_ps(buf, v_r0);
        _mm_store_ps(buf + 4, v_r1);
        _mm_store_ps(buf + 8, v_g0);
        _mm_store_ps(buf + 12, v_g1);
        _mm_store_ps(buf + 16, v_b0);
        _mm_store_ps(buf + 20, v_b1);
    }
    #endif

    void operator()(const uchar* src, uchar* dst, int n) const
    {
        if(enableBitExactness)
        {
            icvt(src, dst, n);
            return;
        }

        int i, j, dcn = dstcn;
        uchar alpha = ColorChannel<uchar>::max();
        float CV_DECL_ALIGNED(16) buf[3*BLOCK_SIZE];
        #if CV_SSE2
        __m128 v_coeffs = _mm_set_ps(100.f/255.f, 1.f, 1.f, 100.f/255.f);
        __m128 v_res = _mm_set_ps(0.f, 128.f, 128.f, 0.f);
        #endif

        i = 0;
        for(; i < n; i += BLOCK_SIZE, src += BLOCK_SIZE*3 )
        {
            int dn = std::min(n - i, (int)BLOCK_SIZE);
            j = 0;

            #if CV_NEON
            for ( ; j <= (dn - 8) * 3; j += 24)
            {
                uint8x8x3_t v_src = vld3_u8(src + j);
                uint16x8_t v_t0 = vmovl_u8(v_src.val[0]),
                           v_t1 = vmovl_u8(v_src.val[1]),
                           v_t2 = vmovl_u8(v_src.val[2]);

                float32x4x3_t v_dst;
                v_dst.val[0] = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_t0))), v_scale_inv);
                v_dst.val[1] = vsubq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_t1))), v_128);
                v_dst.val[2] = vsubq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_t2))), v_128);
                vst3q_f32(buf + j, v_dst);

                v_dst.val[0] = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_t0))), v_scale_inv);
                v_dst.val[1] = vsubq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_t1))), v_128);
                v_dst.val[2] = vsubq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_t2))), v_128);
                vst3q_f32(buf + j + 12, v_dst);
            }
            #elif CV_SSE2
            if (haveSIMD)
            {
                for ( ; j <= (dn - 8) * 3; j += 24)
                {
                    __m128i v_src0 = _mm_loadu_si128((__m128i const *)(src + j));
                    __m128i v_src1 = _mm_loadl_epi64((__m128i const *)(src + j + 16));

                    process(_mm_unpacklo_epi8(v_src0, v_zero),
                            _mm_unpackhi_epi8(v_src0, v_zero),
                            _mm_unpacklo_epi8(v_src1, v_zero),
                            v_coeffs, v_res,
                            buf + j);
                }
            }
            #endif

            for( ; j < dn*3; j += 3 )
            {
                buf[j] = src[j]*(100.f/255.f);
                buf[j+1] = (float)(src[j+1] - 128);
                buf[j+2] = (float)(src[j+2] - 128);
            }
            fcvt(buf, buf, dn);
            j = 0;

            #if CV_NEON
            for ( ; j <= (dn - 8) * 3; j += 24, dst += dcn * 8)
            {
                float32x4x3_t v_src0 = vld3q_f32(buf + j), v_src1 = vld3q_f32(buf + j + 12);
                uint8x8_t v_dst0 = vqmovn_u16(vcombine_u16(vqmovn_u32(cv_vrndq_u32_f32(vmulq_f32(v_src0.val[0], v_scale))),
                                                           vqmovn_u32(cv_vrndq_u32_f32(vmulq_f32(v_src1.val[0], v_scale)))));
                uint8x8_t v_dst1 = vqmovn_u16(vcombine_u16(vqmovn_u32(cv_vrndq_u32_f32(vmulq_f32(v_src0.val[1], v_scale))),
                                                           vqmovn_u32(cv_vrndq_u32_f32(vmulq_f32(v_src1.val[1], v_scale)))));
                uint8x8_t v_dst2 = vqmovn_u16(vcombine_u16(vqmovn_u32(cv_vrndq_u32_f32(vmulq_f32(v_src0.val[2], v_scale))),
                                                           vqmovn_u32(cv_vrndq_u32_f32(vmulq_f32(v_src1.val[2], v_scale)))));

                if (dcn == 4)
                {
                    uint8x8x4_t v_dst;
                    v_dst.val[0] = v_dst0;
                    v_dst.val[1] = v_dst1;
                    v_dst.val[2] = v_dst2;
                    v_dst.val[3] = v_alpha;
                    vst4_u8(dst, v_dst);
                }
                else
                {
                    uint8x8x3_t v_dst;
                    v_dst.val[0] = v_dst0;
                    v_dst.val[1] = v_dst1;
                    v_dst.val[2] = v_dst2;
                    vst3_u8(dst, v_dst);
                }
            }
            #elif CV_SSE2
            if (dcn == 3 && haveSIMD)
            {
                for ( ; j <= (dn * 3 - 16); j += 16, dst += 16)
                {
                    __m128 v_src0 = _mm_mul_ps(_mm_load_ps(buf + j), v_scale);
                    __m128 v_src1 = _mm_mul_ps(_mm_load_ps(buf + j + 4), v_scale);
                    __m128 v_src2 = _mm_mul_ps(_mm_load_ps(buf + j + 8), v_scale);
                    __m128 v_src3 = _mm_mul_ps(_mm_load_ps(buf + j + 12), v_scale);

                    __m128i v_dst0 = _mm_packs_epi32(_mm_cvtps_epi32(v_src0),
                                                     _mm_cvtps_epi32(v_src1));
                    __m128i v_dst1 = _mm_packs_epi32(_mm_cvtps_epi32(v_src2),
                                                     _mm_cvtps_epi32(v_src3));

                    _mm_storeu_si128((__m128i *)dst, _mm_packus_epi16(v_dst0, v_dst1));
                }

                int jr = j % 3;
                if (jr)
                    dst -= jr, j -= jr;
            }
            else if (dcn == 4 && haveSIMD)
            {
                for ( ; j <= (dn * 3 - 12); j += 12, dst += 16)
                {
                    __m128 v_buf0 = _mm_mul_ps(_mm_load_ps(buf + j), v_scale);
                    __m128 v_buf1 = _mm_mul_ps(_mm_load_ps(buf + j + 4), v_scale);
                    __m128 v_buf2 = _mm_mul_ps(_mm_load_ps(buf + j + 8), v_scale);

                    __m128 v_ba0 = _mm_unpackhi_ps(v_buf0, v_alpha);
                    __m128 v_ba1 = _mm_unpacklo_ps(v_buf2, v_alpha);

                    __m128i v_src0 = _mm_cvtps_epi32(_mm_shuffle_ps(v_buf0, v_ba0, 0x44));
                    __m128i v_src1 = _mm_shuffle_epi32(_mm_cvtps_epi32(_mm_shuffle_ps(v_ba0, v_buf1, 0x4e)), 0x78);
                    __m128i v_src2 = _mm_cvtps_epi32(_mm_shuffle_ps(v_buf1, v_ba1, 0x4e));
                    __m128i v_src3 = _mm_shuffle_epi32(_mm_cvtps_epi32(_mm_shuffle_ps(v_ba1, v_buf2, 0xee)), 0x78);

                    __m128i v_dst0 = _mm_packs_epi32(v_src0, v_src1);
                    __m128i v_dst1 = _mm_packs_epi32(v_src2, v_src3);

                    _mm_storeu_si128((__m128i *)dst, _mm_packus_epi16(v_dst0, v_dst1));
                }

                int jr = j % 3;
                if (jr)
                    dst -= jr, j -= jr;
            }
            #endif

            for( ; j < dn*3; j += 3, dst += dcn )
            {
                dst[0] = saturate_cast<uchar>(buf[j]*255.f);
                dst[1] = saturate_cast<uchar>(buf[j+1]*255.f);
                dst[2] = saturate_cast<uchar>(buf[j+2]*255.f);
                if( dcn == 4 )
                    dst[3] = alpha;
            }
        }
    }

    Lab2RGBfloat   fcvt;
    Lab2RGBinteger icvt;
    #if CV_NEON
    float32x4_t v_scale, v_scale_inv, v_128;
    uint8x8_t v_alpha;
    #elif CV_SSE2
    __m128 v_scale;
    __m128 v_alpha;
    __m128i v_zero;
    bool haveSIMD;
    #endif
    int dstcn;
};

///////////////////////////////////// RGB <-> L*u*v* /////////////////////////////////////

struct RGB2Luvfloat
{
    typedef float channel_type;

    RGB2Luvfloat( int _srccn, int blueIdx, const float* _coeffs,
               const float* whitept, bool _srgb )
    : srccn(_srccn), srgb(_srgb)
    {
        volatile int i;
        initLabTabs();

        softdouble whitePt[3];
        for( i = 0; i < 3; i++ )
            if(whitept)
                whitePt[i] = softdouble(whitept[i]);
            else
                whitePt[i] = D65[i];

        for( i = 0; i < 3; i++ )
        {
            for(int j = 0; j < 3; j++)
                if(_coeffs)
                    coeffs[i*3+j] = _coeffs[i*3+j];
                else
                    coeffs[i*3+j] = (float)(sRGB2XYZ_D65[i*3+j]);

            if( blueIdx == 0 )
                std::swap(coeffs[i*3], coeffs[i*3+2]);
            CV_Assert( coeffs[i*3] >= 0 && coeffs[i*3+1] >= 0 && coeffs[i*3+2] >= 0 &&
                      softfloat(coeffs[i*3]) +
                      softfloat(coeffs[i*3+1]) +
                      softfloat(coeffs[i*3+2]) < softfloat(1.5f) );
        }

        softfloat d = whitePt[0] +
                      whitePt[1]*softdouble(15) +
                      whitePt[2]*softdouble(3);
        d = softfloat::one()/max(d, softfloat::eps());
        un = d*softfloat(13*4)*whitePt[0];
        vn = d*softfloat(13*9)*whitePt[1];

        CV_Assert(whitePt[1] == softdouble::one());
    }

    void operator()(const float* src, float* dst, int n) const
    {
        int i = 0, scn = srccn;
        float gscale = GammaTabScale;
        const float* gammaTab = srgb ? sRGBGammaTab : 0;
        float C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2],
              C3 = coeffs[3], C4 = coeffs[4], C5 = coeffs[5],
              C6 = coeffs[6], C7 = coeffs[7], C8 = coeffs[8];

#if CV_SIMD
        const int vsize = v_float32::nlanes;
        for( ; i <= n-vsize;
             i+= vsize, src += scn*vsize, dst += 3*vsize)
        {
            v_float32 R, G, B, A;
            if(scn == 4)
            {
                v_load_deinterleave(src, R, G, B, A);
            }
            else // scn == 3
            {
                v_load_deinterleave(src, R, G, B);
            }

            v_float32 zero = vx_setzero_f32(), one = vx_setall_f32(1.f);
            R = v_min(v_max(R, zero), one);
            G = v_min(v_max(G, zero), one);
            B = v_min(v_max(B, zero), one);

            if(gammaTab)
            {
                v_float32 vgscale = vx_setall_f32(gscale);
                R = splineInterpolate(R*vgscale, gammaTab, GAMMA_TAB_SIZE);
                G = splineInterpolate(G*vgscale, gammaTab, GAMMA_TAB_SIZE);
                B = splineInterpolate(B*vgscale, gammaTab, GAMMA_TAB_SIZE);
            }

            v_float32 X, Y, Z;
            v_float32 vc0 = vx_setall_f32(C0), vc1 = vx_setall_f32(C1), vc2 = vx_setall_f32(C2);
            v_float32 vc3 = vx_setall_f32(C3), vc4 = vx_setall_f32(C4), vc5 = vx_setall_f32(C5);
            v_float32 vc6 = vx_setall_f32(C6), vc7 = vx_setall_f32(C7), vc8 = vx_setall_f32(C8);
            X = v_fma(R, vc0, v_fma(G, vc1, B*vc2));
            Y = v_fma(R, vc3, v_fma(G, vc4, B*vc5));
            Z = v_fma(R, vc6, v_fma(G, vc7, B*vc8));

            v_float32 L, u, v;

            L = splineInterpolate(Y*vx_setall_f32(LabCbrtTabScale), LabCbrtTab, LAB_CBRT_TAB_SIZE);
            // L = 116.f*L - 16.f;
            L = v_fma(L, vx_setall_f32(116.f), vx_setall_f32(-16.f));

            v_float32 d;
            // d = (4*13) / max(X + 15 * Y + 3 * Z, FLT_EPSILON)
            d = v_fma(Y, vx_setall_f32(15.f), v_fma(Z, vx_setall_f32(3.f), X));
            d = vx_setall_f32(4.f*13.f) / v_max(d, vx_setall_f32(FLT_EPSILON));
            // u = L*(X*d - un)
            u = L*v_fma(X, d, vx_setall_f32(-un));
            // v = L*((9*0.25f)*Y*d - vn);
            v = L*v_fma(vx_setall_f32(9.f*0.25f)*Y, d, vx_setall_f32(-vn));

            v_store_interleave(dst, L, u, v);
        }
        vx_cleanup();
#endif

        for( ; i < n; i++, src += scn, dst += 3 )
        {
            float R = src[0], G = src[1], B = src[2];
            R = std::min(std::max(R, 0.f), 1.f);
            G = std::min(std::max(G, 0.f), 1.f);
            B = std::min(std::max(B, 0.f), 1.f);
            if( gammaTab )
            {
                R = splineInterpolate(R*gscale, gammaTab, GAMMA_TAB_SIZE);
                G = splineInterpolate(G*gscale, gammaTab, GAMMA_TAB_SIZE);
                B = splineInterpolate(B*gscale, gammaTab, GAMMA_TAB_SIZE);
            }

            float X = R*C0 + G*C1 + B*C2;
            float Y = R*C3 + G*C4 + B*C5;
            float Z = R*C6 + G*C7 + B*C8;

            float L = splineInterpolate(Y*LabCbrtTabScale, LabCbrtTab, LAB_CBRT_TAB_SIZE);
            L = 116.f*L - 16.f;

            float d = (4*13) / std::max(X + 15 * Y + 3 * Z, FLT_EPSILON);
            float u = L*(X*d - un);
            float v = L*((9*0.25f)*Y*d - vn);

            dst[0] = L; dst[1] = u; dst[2] = v;
        }
    }

    int srccn;
    float coeffs[9], un, vn;
    bool srgb;
};

struct RGB2Luv_f
{
    typedef float channel_type;

    RGB2Luv_f( int _srccn, int blueIdx, const float* _coeffs,
               const float* whitept, bool _srgb )
    : fcvt(_srccn, blueIdx, _coeffs, whitept, _srgb), srccn(_srccn)
    { }

    void operator()(const float* src, float* dst, int n) const
    {
        fcvt(src, dst, n);
    }

    RGB2Luvfloat fcvt;
    int srccn;
};

struct Luv2RGBfloat
{
    typedef float channel_type;

    Luv2RGBfloat( int _dstcn, int blueIdx, const float* _coeffs,
                  const float* whitept, bool _srgb )
    : dstcn(_dstcn),  srgb(_srgb)
    {
        initLabTabs();

        softdouble whitePt[3];
        for(int i = 0; i < 3; i++)
            if(whitept)
                whitePt[i] = softdouble(whitept[i]);
            else
                whitePt[i] = D65[i];

        for( int i = 0; i < 3; i++ )
        {
            softfloat c[3];
            for(int j = 0; j < 3; j++)
                if(_coeffs)
                    c[j] = softfloat(_coeffs[i+j*3]);
                else
                    c[j] = XYZ2sRGB_D65[i+j*3];

            coeffs[i+(blueIdx^2)*3] = c[0];
            coeffs[i+3]             = c[1];
            coeffs[i+blueIdx*3]     = c[2];
        }

        softfloat d = whitePt[0] +
                      whitePt[1]*softdouble(15) +
                      whitePt[2]*softdouble(3);
        d = softfloat::one()/max(d, softfloat::eps());
        un = softfloat(4*13)*d*whitePt[0];
        vn = softfloat(9*13)*d*whitePt[1];

        CV_Assert(whitePt[1] == softdouble::one());
    }

    void operator()(const float* src, float* dst, int n) const
    {
        int i = 0, dcn = dstcn;
        const float* gammaTab = srgb ? sRGBInvGammaTab : 0;
        float gscale = GammaTabScale;
        float C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2],
              C3 = coeffs[3], C4 = coeffs[4], C5 = coeffs[5],
              C6 = coeffs[6], C7 = coeffs[7], C8 = coeffs[8];
        float alpha = ColorChannel<float>::max();
        float _un = un, _vn = vn;

#if CV_SIMD
        const int vsize = v_float32::nlanes;
        for( ; i <= n - vsize;
             i += vsize, src += vsize*3, dst += dcn*vsize)
        {
            v_float32 L, u, v;
            v_load_deinterleave(src, L, u, v);

            v_float32 Ylo, Yhi;
            v_float32 v16 = vx_setall_f32(16.f);
            v_float32 v116inv = vx_setall_f32(1.f/116.f);
            v_float32 v903inv = vx_setall_f32(1.0f/903.296296f); //(3./29.)^3
            // ((L + 16)/116)^3
            Ylo = (L + v16) * v116inv;
            Ylo = Ylo*Ylo*Ylo;
            // L*(3./29.)^3
            Yhi = L * v903inv;

            v_float32 X, Y, Z;
            // Y = (L <= 8) ? Y0 : Y1;
            Y = v_select(L >= vx_setall_f32(8.f), Ylo, Yhi);

            v_float32 up, vp;
            v_float32 v4inv = vx_setall_f32(0.25f), v3 = vx_setall_f32(3.f);
            // up = 3*(u + L*_un);
            up = v3*(v_fma(L, vx_setall_f32(_un), u));
            // vp = 0.25/(v + L*_vn);
            vp = v4inv/(v_fma(L, vx_setall_f32(_vn), v));

            // vp = max(-0.25, min(0.25, vp));
            vp = v_max(vx_setall_f32(-0.25f), v_min(v4inv, vp));

            //X = 3*up*vp; // (*Y) is done later
            X = v3*up*vp;
            //Z = ((12*13*L - up)*vp - 5); // (*Y) is done later
            // xor flips the sign, works like unary minus
            Z = v_fma(v_fma(L, vx_setall_f32(12.f*13.f), (vx_setall_f32(-0.f) ^ up)), vp, vx_setall_f32(-5.f));

            v_float32 R, G, B;
            v_float32 vc0 = vx_setall_f32(C0), vc1 = vx_setall_f32(C1), vc2 = vx_setall_f32(C2);
            v_float32 vc3 = vx_setall_f32(C3), vc4 = vx_setall_f32(C4), vc5 = vx_setall_f32(C5);
            v_float32 vc6 = vx_setall_f32(C6), vc7 = vx_setall_f32(C7), vc8 = vx_setall_f32(C8);
            // R = (X*C0 + C1 + Z*C2)*Y; // here (*Y) is done
            R = v_fma(Z, vc2, v_fma(X, vc0, vc1))*Y;
            G = v_fma(Z, vc5, v_fma(X, vc3, vc4))*Y;
            B = v_fma(Z, vc8, v_fma(X, vc6, vc7))*Y;

            v_float32 vzero = vx_setzero_f32(), v1 = vx_setall_f32(1.f);
            R = v_min(v_max(R, vzero), v1);
            G = v_min(v_max(G, vzero), v1);
            B = v_min(v_max(B, vzero), v1);

            if(gammaTab)
            {
                v_float32 vgscale = vx_setall_f32(gscale);
                R = splineInterpolate(R*vgscale, gammaTab, GAMMA_TAB_SIZE);
                G = splineInterpolate(G*vgscale, gammaTab, GAMMA_TAB_SIZE);
                B = splineInterpolate(B*vgscale, gammaTab, GAMMA_TAB_SIZE);
            }

            if(dcn == 4)
            {
                v_store_interleave(dst, R, G, B, vx_setall_f32(alpha));
            }
            else // dcn == 3
            {
                v_store_interleave(dst, R, G, B);
            }
        }

        vx_cleanup();
#endif

        for( ; i < n; i++, src += 3,  dst += dcn )
        {
            float L = src[0], u = src[1], v = src[2], X, Y, Z;
            if(L >= 8)
            {
                Y = (L + 16.f) * (1.f/116.f);
                Y = Y*Y*Y;
            }
            else
            {
                Y = L * (1.0f/903.3f); // L*(3./29.)^3
            }
            float up = 3.f*(u + L*_un);
            float vp = 0.25f/(v + L*_vn);
            if(vp >  0.25f) vp =  0.25f;
            if(vp < -0.25f) vp = -0.25f;
            X = Y*3.f*up*vp;
            Z = Y*(((12.f*13.f)*L - up)*vp - 5.f);

            float R = X*C0 + Y*C1 + Z*C2;
            float G = X*C3 + Y*C4 + Z*C5;
            float B = X*C6 + Y*C7 + Z*C8;

            R = std::min(std::max(R, 0.f), 1.f);
            G = std::min(std::max(G, 0.f), 1.f);
            B = std::min(std::max(B, 0.f), 1.f);

            if( gammaTab )
            {
                R = splineInterpolate(R*gscale, gammaTab, GAMMA_TAB_SIZE);
                G = splineInterpolate(G*gscale, gammaTab, GAMMA_TAB_SIZE);
                B = splineInterpolate(B*gscale, gammaTab, GAMMA_TAB_SIZE);
            }

            dst[0] = R; dst[1] = G; dst[2] = B;
            if( dcn == 4 )
                dst[3] = alpha;
        }
    }

    int dstcn;
    float coeffs[9], un, vn;
    bool srgb;
};


struct Luv2RGB_f
{
    typedef float channel_type;

    Luv2RGB_f( int _dstcn, int blueIdx, const float* _coeffs,
              const float* whitept, bool _srgb )
    : fcvt(_dstcn, blueIdx, _coeffs, whitept, _srgb), dstcn(_dstcn)
    { }

    void operator()(const float* src, float* dst, int n) const
    {
        fcvt(src, dst, n);
    }

    Luv2RGBfloat fcvt;
    int dstcn;
};

struct RGB2Luvinterpolate
{
    typedef uchar channel_type;

    RGB2Luvinterpolate( int _srccn, int _blueIdx, const float* /* _coeffs */,
                        const float* /* _whitept */, bool /*_srgb*/ )
    : srccn(_srccn), blueIdx(_blueIdx)
    {
        initLabTabs();
    }

    void operator()(const uchar* src, uchar* dst, int n) const
    {
        int i, scn = srccn, bIdx = blueIdx;

        i = 0; n *= 3;

#if CV_SIMD
        if(enablePackedRGB2Luv)
        {
            const int vsize = v_uint16::nlanes;
            static const int nPixels = vsize*2;
            for(; i < n - 3*nPixels; i += 3*nPixels, src += scn*nPixels)
            {
                /*
                    int R = src[bIdx], G = src[1], B = src[bIdx^2];
                */
                v_uint8 r, g, b, dummy;
                if(scn == 3)
                {
                    v_load_deinterleave(src, r, g, b);
                }
                else // scn == 4
                {
                    v_load_deinterleave(src, r, g, b, dummy);
                }

                if(bIdx)
                {
                    swap(r, b);
                }

                /*
                    static const int baseDiv = LAB_BASE/256;
                    R = R*baseDiv, G = G*baseDiv, B = B*baseDiv;
                */
                v_uint16 r0, r1, g0, g1, b0, b1;
                v_expand(r, r0, r1);
                v_expand(g, g0, g1);
                v_expand(b, b0, b1);
                r0 = r0 << (lab_base_shift - 8); r1 = r1 << (lab_base_shift - 8);
                g0 = g0 << (lab_base_shift - 8); g1 = g1 << (lab_base_shift - 8);
                b0 = b0 << (lab_base_shift - 8); b1 = b1 << (lab_base_shift - 8);

                /*
                    int L, u, v;
                    trilinearInterpolate(R, G, B, RGB2LuvLUT_s16, L, u, v);
                 */
                v_uint16 l0, u0, v0, l1, u1, v1;
                trilinearPackedInterpolate(r0, g0, b0, LABLUVLUTs16.RGB2LuvLUT_s16, l0, u0, v0);
                trilinearPackedInterpolate(r1, g1, b1, LABLUVLUTs16.RGB2LuvLUT_s16, l1, u1, v1);

                /*
                    dst[i]   = saturate_cast<uchar>(L/baseDiv);
                    dst[i+1] = saturate_cast<uchar>(u/baseDiv);
                    dst[i+2] = saturate_cast<uchar>(v/baseDiv);
                 */
                l0 = l0 >> (lab_base_shift - 8); l1 = l1 >> (lab_base_shift - 8);
                u0 = u0 >> (lab_base_shift - 8); u1 = u1 >> (lab_base_shift - 8);
                v0 = v0 >> (lab_base_shift - 8); v1 = v1 >> (lab_base_shift - 8);
                v_uint8 l = v_pack(l0, l1);
                v_uint8 u = v_pack(u0, u1);
                v_uint8 v = v_pack(v0, v1);
                v_store_interleave(dst + i, l, u, v);
            }
            vx_cleanup();
        }
#endif // CV_SIMD

        for(; i < n; i += 3, src += scn)
        {
            int R = src[bIdx], G = src[1], B = src[bIdx^2];

            // (LAB_BASE/255) gives more accuracy but not very much
            static const int baseDiv = LAB_BASE/256;
            R = R*baseDiv, G = G*baseDiv, B = B*baseDiv;

            int L, u, v;
            trilinearInterpolate(R, G, B, LABLUVLUTs16.RGB2LuvLUT_s16, L, u, v);

            dst[i] = saturate_cast<uchar>(L/baseDiv);
            dst[i+1] = saturate_cast<uchar>(u/baseDiv);
            dst[i+2] = saturate_cast<uchar>(v/baseDiv);
        }
    }

    int srccn;
    int blueIdx;
};


struct RGB2Luv_b
{
    typedef uchar channel_type;
    static const int bufChannels = 3;

    RGB2Luv_b( int _srccn, int blueIdx, const float* _coeffs,
               const float* _whitept, bool _srgb )
    : srccn(_srccn),
      fcvt(bufChannels, blueIdx, _coeffs, _whitept, _srgb),
      icvt(_srccn, blueIdx, _coeffs, _whitept, _srgb)
    {
        // using interpolation for LRGB gives error up to 8 of 255, don't use it
        useInterpolation = (!_coeffs && !_whitept && _srgb
                            && enableBitExactness
                            && enableRGB2LuvInterpolation);
    }

    void operator()(const uchar* src, uchar* dst, int n) const
    {
        if(useInterpolation)
        {
            icvt(src, dst, n);
            return;
        }

        int i, j, scn = srccn;
#if CV_SIMD
        float CV_DECL_ALIGNED(v_uint8::nlanes) buf[bufChannels*BLOCK_SIZE];
#else
        float CV_DECL_ALIGNED(16) buf[bufChannels*BLOCK_SIZE];
#endif

        static const softfloat fL = f255/softfloat(100);
        static const softfloat fu = f255/uRange;
        static const softfloat fv = f255/vRange;
        static const softfloat su = -uLow*f255/uRange;
        static const softfloat sv = -vLow*f255/vRange;
#if CV_SIMD
        const int fsize = v_float32::nlanes;
        v_float32 ml = vx_setall_f32((float)fL), al = vx_setzero_f32();
        v_float32 mu = vx_setall_f32((float)fu), au = vx_setall_f32((float)su);
        v_float32 mv = vx_setall_f32((float)fv), av = vx_setall_f32((float)sv);
        //TODO: fix that when v_interleave is available
        float CV_DECL_ALIGNED(v_uint8::nlanes) interTmpM[fsize*3], interTmpA[fsize*3];
        v_store_interleave(interTmpM, ml, mu, mv);
        v_store_interleave(interTmpA, al, au, av);
        v_float32 mluv[3], aluv[3];
        for(int k = 0; k < 3; k++)
        {
            mluv[k] = vx_load_aligned(interTmpM + k*fsize);
            aluv[k] = vx_load_aligned(interTmpA + k*fsize);
        }
#endif

        for( i = 0; i < n; i += BLOCK_SIZE, dst += BLOCK_SIZE*bufChannels )
        {
            int dn = std::min(n - i, (int)BLOCK_SIZE);
            j = 0;

            static const softfloat f255inv = softfloat::one()/f255;
#if CV_SIMD
            v_float32 v255inv = vx_setall_f32((float)f255inv);
            if(scn == 4)
            {
                static const int nBlock = fsize*4;
                for( ; j <= dn*bufChannels - nBlock*3;
                     j += nBlock*3, src += nBlock*4)
                {
                    v_uint8 rgb[3], dummy;
                    v_load_deinterleave(src, rgb[0], rgb[1], rgb[2], dummy);

                    v_uint16 d[3*2];
                    for(int k = 0; k < 3; k++)
                    {
                        v_expand(rgb[k], d[k*2+0], d[k*2+1]);
                    }
                    v_int32 q[3*4];
                    for(int k = 0; k < 3*2; k++)
                    {
                        v_expand(v_reinterpret_as_s16(d[k]), q[k*2+0], q[k*2+1]);
                    }

                    v_float32 f[3*4];
                    for(int k = 0; k < 3*4; k++)
                    {
                        f[k] = v_cvt_f32(q[k])*v255inv;
                    }

                    for(int k = 0; k < 4; k++)
                    {
                        v_store_interleave(buf + j + k*3*fsize, f[0*4+k], f[1*4+k], f[2*4+k]);
                    }
                }
            }
            else // scn == 3
            {
                static const int nBlock = fsize*2;
                for( ; j <= dn*bufChannels - nBlock;
                     j += nBlock, src += nBlock)
                {
                    v_uint16 d = vx_load_expand(src);
                    v_int32 q0, q1;
                    v_expand(v_reinterpret_as_s16(d), q0, q1);

                    v_store_aligned(buf + j + 0*fsize, v_cvt_f32(q0)*v255inv);
                    v_store_aligned(buf + j + 1*fsize, v_cvt_f32(q1)*v255inv);
                }
            }
            vx_cleanup();
#endif
            for( ; j < dn*bufChannels; j += bufChannels, src += scn )
            {
                buf[j  ] = (float)(src[0]*((float)f255inv));
                buf[j+1] = (float)(src[1]*((float)f255inv));
                buf[j+2] = (float)(src[2]*((float)f255inv));
            }
            fcvt(buf, buf, dn);

            j = 0;

#if CV_SIMD
            for( ; j <= dn*3 - fsize*3*4; j += fsize*3*4)
            {
                v_float32 f[3*4];
                for(int k = 0; k < 3*4; k++)
                    f[k] = vx_load_aligned(buf + j + k*fsize);

                for(int k = 0; k < 4; k++)
                {
                    f[k*3+0] = v_fma(f[k*3+0], mluv[0], aluv[0]);
                    f[k*3+1] = v_fma(f[k*3+1], mluv[1], aluv[1]);
                    f[k*3+2] = v_fma(f[k*3+2], mluv[2], aluv[2]);
                }

                v_int32 q[3*4];
                for(int k = 0; k < 3*4; k++)
                {
                    q[k] = v_round(f[k]);
                }

                for(int k = 0; k < 3; k++)
                {
                    v_store(dst + j + k*fsize*4, v_pack_u(v_pack(q[k*4+0], q[k*4+1]),
                                                          v_pack(q[k*4+2], q[k*4+3])));
                }
            }
            vx_cleanup();
#endif
            for( ; j < dn*3; j += 3 )
            {
                dst[j] = saturate_cast<uchar>(buf[j]*(float)fL);
                dst[j+1] = saturate_cast<uchar>(buf[j+1]*(float)fu + (float)su);
                dst[j+2] = saturate_cast<uchar>(buf[j+2]*(float)fv + (float)sv);
            }
        }
    }

    int srccn;
    RGB2Luvfloat fcvt;
    RGB2Luvinterpolate icvt;

    bool useInterpolation;
};


struct Luv2RGBinteger
{
    typedef uchar channel_type;

    static const int base_shift = 14;
    static const int BASE = (1 << base_shift);
    static const int shift = lab_shift+(base_shift-inv_gamma_shift);

    // whitept is fixed for int calculations
    Luv2RGBinteger( int _dstcn, int blueIdx, const float* _coeffs,
                    const float* /*_whitept*/, bool _srgb )
    : dstcn(_dstcn)
    {
        initLabTabs();

        static const softdouble lshift(1 << lab_shift);
        for(int i = 0; i < 3; i++)
        {
            softdouble c[3];
            for(int j = 0; j < 3; j++)
                if(_coeffs)
                    c[j] = softdouble(_coeffs[i + j*3]);
                else
                    c[j] = XYZ2sRGB_D65[i + j*3];

            coeffs[i+blueIdx*3]     = cvRound(lshift*c[0]);
            coeffs[i+3]             = cvRound(lshift*c[1]);
            coeffs[i+(blueIdx^2)*3] = cvRound(lshift*c[2]);
        }

        tab = _srgb ? sRGBInvGammaTab_b : linearInvGammaTab_b;
    }

    // L, u, v should be in their natural range
    inline void process(const uchar LL, const uchar uu, const uchar vv, int& ro, int& go, int& bo) const
    {
        ushort y = LabToYF_b[LL*2];

        // y : [0, BASE]
        // up: [-402, 1431.57]*(BASE/1024)
        // vp: +/- 0.25*BASE*1024
        int up = LUVLUT.LuToUp_b[LL*256+uu];
        int vp = LUVLUT.LvToVp_b[LL*256+vv];
        //X = y*3.f* up/((float)BASE/1024) *vp/((float)BASE*1024);
        //Z = y*(((12.f*13.f)*((float)LL)*100.f/255.f - up/((float)BASE))*vp/((float)BASE*1024) - 5.f);

        long long int xv = ((int)up)*(long long)vp;
        int x = (int)(xv/BASE);
        x = y*x/BASE;

        long long int vpl = LUVLUT.LvToVpl_b[LL*256+vv];
        long long int zp = vpl - xv*(255/3);
        zp /= BASE;
        long long int zq = zp - (long long)(5*255*BASE);
        int zm = (int)(y*zq/BASE);
        int z = zm/256 + zm/65536;

        //limit X, Y, Z to [0, 2] to fit white point
        x = max(0, min(2*BASE, x)); z = max(0, min(2*BASE, z));

        int C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2];
        int C3 = coeffs[3], C4 = coeffs[4], C5 = coeffs[5];
        int C6 = coeffs[6], C7 = coeffs[7], C8 = coeffs[8];

        ro = CV_DESCALE(C0 * x + C1 * y + C2 * z, shift);
        go = CV_DESCALE(C3 * x + C4 * y + C5 * z, shift);
        bo = CV_DESCALE(C6 * x + C7 * y + C8 * z, shift);

        ro = max(0, min((int)INV_GAMMA_TAB_SIZE-1, ro));
        go = max(0, min((int)INV_GAMMA_TAB_SIZE-1, go));
        bo = max(0, min((int)INV_GAMMA_TAB_SIZE-1, bo));

        ro = tab[ro];
        go = tab[go];
        bo = tab[bo];
    }

    inline void processLuvToXYZ(const v_uint8x16& lv, const v_uint8x16& uv, const v_uint8x16& vv,
                                int32_t* xyz) const
    {
        uint8_t CV_DECL_ALIGNED(v_uint8::nlanes) lvstore[16], uvstore[16], vvstore[16];
        v_store_aligned(lvstore, lv); v_store_aligned(uvstore, uv); v_store_aligned(vvstore, vv);

        for(int i = 0; i < 16; i++)
        {
            int LL = lvstore[i];
            int u = uvstore[i];
            int v = vvstore[i];
            int y = LabToYF_b[LL*2];

            int up = LUVLUT.LuToUp_b[LL*256+u];
            int vp = LUVLUT.LvToVp_b[LL*256+v];

            long long int xv = up*(long long int)vp;
            long long int vpl = LUVLUT.LvToVpl_b[LL*256+v];
            long long int zp = vpl - xv*(255/3);
            zp = zp >> base_shift;
            long long int zq = zp - (5*255*BASE);
            int zm = (int)((y*zq) >> base_shift);

            int x = (int)(xv >> base_shift);
            x = (y*x) >> base_shift;

            int z = zm/256 + zm/65536;
            x = max(0, min(2*BASE, x)); z = max(0, min(2*BASE, z));

            xyz[i] = x; xyz[i + 16] = y; xyz[i + 32] = z;
        }
    }

    void operator()(const uchar* src, uchar* dst, int n) const
    {
        int i, dcn = dstcn;
        uchar alpha = ColorChannel<uchar>::max();

        i = 0;
#if CV_SIMD128
        if(enablePackedLuv2RGB)
        {
            static const int nPixels = 16;
            for (; i < n*3-3*nPixels; i += 3*nPixels, dst += dcn*nPixels)
            {
                v_uint8x16 u8l, u8u, u8v;
                v_load_deinterleave(src + i, u8l, u8u, u8v);

                int32_t CV_DECL_ALIGNED(v_uint8x16::nlanes) xyz[48];
                processLuvToXYZ(u8l, u8u, u8v, xyz);

                v_int32x4 xiv[4], yiv[4], ziv[4];
                for(int k = 0; k < 4; k++)
                {
                    xiv[k] = v_load_aligned(xyz + 4*k);
                    yiv[k] = v_load_aligned(xyz + 4*k + 16);
                    ziv[k] = v_load_aligned(xyz + 4*k + 32);
                }

                /*
                        ro = CV_DESCALE(C0 * x + C1 * y + C2 * z, shift);
                        go = CV_DESCALE(C3 * x + C4 * y + C5 * z, shift);
                        bo = CV_DESCALE(C6 * x + C7 * y + C8 * z, shift);
                */
                v_int32x4 C0 = v_setall_s32(coeffs[0]), C1 = v_setall_s32(coeffs[1]), C2 = v_setall_s32(coeffs[2]);
                v_int32x4 C3 = v_setall_s32(coeffs[3]), C4 = v_setall_s32(coeffs[4]), C5 = v_setall_s32(coeffs[5]);
                v_int32x4 C6 = v_setall_s32(coeffs[6]), C7 = v_setall_s32(coeffs[7]), C8 = v_setall_s32(coeffs[8]);
                v_int32x4 descaleShift = v_setall_s32(1 << (shift-1));
                v_int32x4 tabsz = v_setall_s32((int)INV_GAMMA_TAB_SIZE-1);
                v_uint32x4 r_vecs[4], g_vecs[4], b_vecs[4];
                for(int k = 0; k < 4; k++)
                {
                    v_int32x4 i_r, i_g, i_b;
                    i_r = (xiv[k]*C0 + yiv[k]*C1 + ziv[k]*C2 + descaleShift) >> shift;
                    i_g = (xiv[k]*C3 + yiv[k]*C4 + ziv[k]*C5 + descaleShift) >> shift;
                    i_b = (xiv[k]*C6 + yiv[k]*C7 + ziv[k]*C8 + descaleShift) >> shift;

                    //limit indices in table and then substitute
                    //ro = tab[ro]; go = tab[go]; bo = tab[bo];
                    int32_t CV_DECL_ALIGNED(v_uint8x16::nlanes) rshifts[4], gshifts[4], bshifts[4];
                    v_int32x4 rs = v_max(v_setzero_s32(), v_min(tabsz, i_r));
                    v_int32x4 gs = v_max(v_setzero_s32(), v_min(tabsz, i_g));
                    v_int32x4 bs = v_max(v_setzero_s32(), v_min(tabsz, i_b));

                    v_store_aligned(rshifts, rs);
                    v_store_aligned(gshifts, gs);
                    v_store_aligned(bshifts, bs);

                    r_vecs[k] = v_uint32x4(tab[rshifts[0]], tab[rshifts[1]], tab[rshifts[2]], tab[rshifts[3]]);
                    g_vecs[k] = v_uint32x4(tab[gshifts[0]], tab[gshifts[1]], tab[gshifts[2]], tab[gshifts[3]]);
                    b_vecs[k] = v_uint32x4(tab[bshifts[0]], tab[bshifts[1]], tab[bshifts[2]], tab[bshifts[3]]);
                }

                v_uint16x8 u_rvec0 = v_pack(r_vecs[0], r_vecs[1]), u_rvec1 = v_pack(r_vecs[2], r_vecs[3]);
                v_uint16x8 u_gvec0 = v_pack(g_vecs[0], g_vecs[1]), u_gvec1 = v_pack(g_vecs[2], g_vecs[3]);
                v_uint16x8 u_bvec0 = v_pack(b_vecs[0], b_vecs[1]), u_bvec1 = v_pack(b_vecs[2], b_vecs[3]);

                v_uint8x16 u8_b, u8_g, u8_r;
                u8_b = v_pack(u_bvec0, u_bvec1);
                u8_g = v_pack(u_gvec0, u_gvec1);
                u8_r = v_pack(u_rvec0, u_rvec1);

                if(dcn == 4)
                {
                    v_store_interleave(dst, u8_b, u8_g, u8_r, v_setall_u8(alpha));
                }
                else
                {
                    v_store_interleave(dst, u8_b, u8_g, u8_r);
                }
            }
            vx_cleanup();
        }
#endif

        for (; i < n*3; i += 3, dst += dcn)
        {
            int ro, go, bo;
            process(src[i + 0], src[i + 1], src[i + 2], ro, go, bo);

            dst[0] = saturate_cast<uchar>(bo);
            dst[1] = saturate_cast<uchar>(go);
            dst[2] = saturate_cast<uchar>(ro);
            if( dcn == 4 )
                dst[3] = alpha;
        }

    }

    int dstcn;
    int coeffs[9];
    ushort* tab;
};


struct Luv2RGB_b
{
    typedef uchar channel_type;

    Luv2RGB_b( int _dstcn, int blueIdx, const float* _coeffs,
               const float* _whitept, bool _srgb )
    : dstcn(_dstcn),
      fcvt(3, blueIdx, _coeffs, _whitept, _srgb),
      icvt(_dstcn, blueIdx, _coeffs, _whitept, _srgb)
    {
        // whitept is fixed for int calculations
        useBitExactness = (!_whitept && enableBitExactness);
    }

    void operator()(const uchar* src, uchar* dst, int n) const
    {
        if(useBitExactness)
        {
            icvt(src, dst, n);
            return;
        }

        int i, j, dcn = dstcn;
        uchar alpha = ColorChannel<uchar>::max();
#if CV_SIMD
        float CV_DECL_ALIGNED(v_uint8::nlanes) buf[3*BLOCK_SIZE];
#else
        float CV_DECL_ALIGNED(16) buf[3*BLOCK_SIZE];
#endif

        static const softfloat fl = softfloat(100)/f255;
        static const softfloat fu = uRange/f255;
        static const softfloat fv = vRange/f255;

#if CV_SIMD
        const int fsize = v_float32::nlanes;
        v_float32 vl = vx_setall_f32((float)fl);
        v_float32 vu = vx_setall_f32((float)fu);
        v_float32 vv = vx_setall_f32((float)fv);
        v_float32 vuLow = vx_setall_f32((float)uLow), vvLow = vx_setall_f32((float)vLow);
        //TODO: fix that when v_interleave is available
        float CV_DECL_ALIGNED(v_uint8::nlanes) interTmpM[fsize*3], interTmpA[fsize*3];
        v_store_interleave(interTmpM, vl, vu, vv);
        v_store_interleave(interTmpA, vx_setzero_f32(), vuLow, vvLow);
        v_float32 mluv[3], aluv[3];
        for(int k = 0; k < 3; k++)
        {
            mluv[k] = vx_load_aligned(interTmpM + k*fsize);
            aluv[k] = vx_load_aligned(interTmpA + k*fsize);
        }
#endif

        for( i = 0; i < n; i += BLOCK_SIZE, src += BLOCK_SIZE*3 )
        {
            int dn = std::min(n - i, (int)BLOCK_SIZE);
            j = 0;

#if CV_SIMD
            const int vsize = v_uint8::nlanes;
            for( ; j <= (dn - vsize)*3; j += 3*vsize )
            {
                v_uint8 s0, s1, s2;
                s0 = vx_load(src + j + 0*vsize);
                s1 = vx_load(src + j + 1*vsize);
                s2 = vx_load(src + j + 2*vsize);

                v_uint16 ss[6];
                v_expand(s0, ss[0], ss[1]);
                v_expand(s1, ss[2], ss[3]);
                v_expand(s2, ss[4], ss[5]);
                v_int32 vs[12];
                for(int k = 0; k < 6; k++)
                {
                    v_expand(v_reinterpret_as_s16(ss[k]), vs[k*2+0], vs[k*2+1]);
                }

                for(int bufp = 0; bufp < 12; bufp++)
                {
                    v_store_aligned(buf + j + bufp, v_muladd(v_cvt_f32(vs[bufp]), mluv[bufp%3], aluv[bufp%3]));
                }
            }
            vx_cleanup();
#endif
            for( ; j < dn*3; j += 3 )
            {
                buf[j] = src[j]*((float)fl);
                buf[j+1] = (float)(src[j+1]*(float)fu + (float)uLow);
                buf[j+2] = (float)(src[j+2]*(float)fv + (float)vLow);
            }

            fcvt(buf, buf, dn);

            j = 0;

#if CV_SIMD
            static const int nBlock = 4*fsize;
            v_float32 v255 = vx_setall_f32(255.f);
            if(dcn == 4)
            {
                v_uint8 valpha = vx_setall_u8(alpha);
                for( ; j <= (dn-nBlock)*3;
                     j += nBlock*3, dst += nBlock)
                {
                    v_float32 vf[4*3];
                    for(int k = 0; k < 4; k++)
                    {
                        v_load_deinterleave(buf + j, vf[k*3+0], vf[k*3+1], vf[k*3+2]);
                    }

                    v_int32 vi[4*3];
                    for(int k = 0; k < 4*3; k++)
                    {
                        vi[k] = v_round(vf[k]*v255);
                    }

                    v_uint8 rgb[3];
                    for(int k = 0; k < 3; k++)
                    {
                        rgb[k] = v_pack_u(v_pack(vi[0*3+k], vi[1*3+k]),
                                          v_pack(vi[2*3+k], vi[3*3+k]));
                    }

                    v_store_interleave(dst, rgb[0], rgb[1], rgb[2], valpha);
                }
            }
            else // dcn == 3
            {
                for(; j < dn*3 - nBlock; j += nBlock, dst += nBlock)
                {
                    v_float32 vf[4];
                    v_int32 vi[4];
                    for(int k = 0; k < 4; k++)
                    {
                        vf[k] = vx_load_aligned(buf + j + k*fsize);
                        vi[k] = v_round(vf[k]*v255);
                    }
                    v_store(dst, v_pack_u(v_pack(vi[0], vi[1]), v_pack(vi[2], vi[3])));
                }
            }
            vx_cleanup();
#endif

            for( ; j < dn*3; j += 3, dst += dcn )
            {
                dst[0] = saturate_cast<uchar>(buf[j]*255.f);
                dst[1] = saturate_cast<uchar>(buf[j+1]*255.f);
                dst[2] = saturate_cast<uchar>(buf[j+2]*255.f);
                if( dcn == 4 )
                    dst[3] = alpha;
            }
        }
    }

    int dstcn;
    Luv2RGBfloat   fcvt;
    Luv2RGBinteger icvt;

    bool useBitExactness;
};

//
// IPP functions
//

#if NEED_IPP

#if !IPP_DISABLE_RGB_XYZ
static ippiGeneralFunc ippiRGB2XYZTab[] =
{
    (ippiGeneralFunc)ippiRGBToXYZ_8u_C3R, 0, (ippiGeneralFunc)ippiRGBToXYZ_16u_C3R, 0,
    0, (ippiGeneralFunc)ippiRGBToXYZ_32f_C3R, 0, 0
};
#endif

#if !IPP_DISABLE_XYZ_RGB
static ippiGeneralFunc ippiXYZ2RGBTab[] =
{
    (ippiGeneralFunc)ippiXYZToRGB_8u_C3R, 0, (ippiGeneralFunc)ippiXYZToRGB_16u_C3R, 0,
    0, (ippiGeneralFunc)ippiXYZToRGB_32f_C3R, 0, 0
};
#endif

#if !IPP_DISABLE_RGB_LAB
static ippiGeneralFunc ippiRGBToLUVTab[] =
{
    (ippiGeneralFunc)ippiRGBToLUV_8u_C3R, 0, (ippiGeneralFunc)ippiRGBToLUV_16u_C3R, 0,
    0, (ippiGeneralFunc)ippiRGBToLUV_32f_C3R, 0, 0
};
#endif

#if !IPP_DISABLE_LAB_RGB
static ippiGeneralFunc ippiLUVToRGBTab[] =
{
    (ippiGeneralFunc)ippiLUVToRGB_8u_C3R, 0, (ippiGeneralFunc)ippiLUVToRGB_16u_C3R, 0,
    0, (ippiGeneralFunc)ippiLUVToRGB_32f_C3R, 0, 0
};
#endif

#endif


//
// HAL functions
//

namespace hal
{

void cvtBGRtoXYZ(const uchar * src_data, size_t src_step,
                 uchar * dst_data, size_t dst_step,
                 int width, int height,
                 int depth, int scn, bool swapBlue)
{
    CV_INSTRUMENT_REGION();

    CALL_HAL(cvtBGRtoXYZ, cv_hal_cvtBGRtoXYZ, src_data, src_step, dst_data, dst_step, width, height, depth, scn, swapBlue);

#if defined(HAVE_IPP) && IPP_VERSION_X100 >= 700
#if !IPP_DISABLE_RGB_XYZ
    CV_IPP_CHECK()
    {
        if(scn == 3 && depth != CV_32F && !swapBlue)
        {
            if( CvtColorIPPLoopCopy(src_data, src_step, CV_MAKETYPE(depth, scn), dst_data, dst_step, width, height,
                                    IPPReorderGeneralFunctor(ippiSwapChannelsC3RTab[depth], ippiRGB2XYZTab[depth], 2, 1, 0, depth)) )
                return;
        }
        else if(scn == 4 && depth != CV_32F && !swapBlue)
        {
            if( CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                                IPPReorderGeneralFunctor(ippiSwapChannelsC4C3RTab[depth], ippiRGB2XYZTab[depth], 2, 1, 0, depth)) )
                return;
        }
        else if(scn == 3 && depth != CV_32F && swapBlue)
        {
            if( CvtColorIPPLoopCopy(src_data, src_step, CV_MAKETYPE(depth, scn), dst_data, dst_step, width, height,
                                    IPPGeneralFunctor(ippiRGB2XYZTab[depth])) )
                return;
        }
        else if(scn == 4 && depth != CV_32F && swapBlue)
        {
            if( CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                                IPPReorderGeneralFunctor(ippiSwapChannelsC4C3RTab[depth], ippiRGB2XYZTab[depth], 0, 1, 2, depth)) )
                return;
        }
    }
#endif
#endif

    int blueIdx = swapBlue ? 2 : 0;
    if( depth == CV_8U )
        CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, RGB2XYZ_i<uchar>(scn, blueIdx, 0));
    else if( depth == CV_16U )
        CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, RGB2XYZ_i<ushort>(scn, blueIdx, 0));
    else
        CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, RGB2XYZ_f<float>(scn, blueIdx, 0));
}


void cvtXYZtoBGR(const uchar * src_data, size_t src_step,
                 uchar * dst_data, size_t dst_step,
                 int width, int height,
                 int depth, int dcn, bool swapBlue)
{
    CV_INSTRUMENT_REGION();

    CALL_HAL(cvtXYZtoBGR, cv_hal_cvtXYZtoBGR, src_data, src_step, dst_data, dst_step, width, height, depth, dcn, swapBlue);

#if defined(HAVE_IPP) && IPP_VERSION_X100 >= 700
#if !IPP_DISABLE_XYZ_RGB
    CV_IPP_CHECK()
    {
        if(dcn == 3 && depth != CV_32F && !swapBlue)
        {
            if( CvtColorIPPLoopCopy(src_data, src_step, CV_MAKETYPE(depth, 3), dst_data, dst_step, width, height,
                                    IPPGeneralReorderFunctor(ippiXYZ2RGBTab[depth], ippiSwapChannelsC3RTab[depth], 2, 1, 0, depth)) )
                return;
        }
        else if(dcn == 4 && depth != CV_32F && !swapBlue)
        {
            if( CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                                IPPGeneralReorderFunctor(ippiXYZ2RGBTab[depth], ippiSwapChannelsC3C4RTab[depth], 2, 1, 0, depth)) )
                return;
        }
        if(dcn == 3 && depth != CV_32F && swapBlue)
        {
            if( CvtColorIPPLoopCopy(src_data, src_step, CV_MAKETYPE(depth, 3), dst_data, dst_step, width, height,
                                    IPPGeneralFunctor(ippiXYZ2RGBTab[depth])) )
                return;
        }
        else if(dcn == 4 && depth != CV_32F && swapBlue)
        {
            if( CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                                IPPGeneralReorderFunctor(ippiXYZ2RGBTab[depth], ippiSwapChannelsC3C4RTab[depth], 0, 1, 2, depth)) )
                return;
        }
    }
#endif
#endif

    int blueIdx = swapBlue ? 2 : 0;
    if( depth == CV_8U )
        CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, XYZ2RGB_i<uchar>(dcn, blueIdx, 0));
    else if( depth == CV_16U )
        CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, XYZ2RGB_i<ushort>(dcn, blueIdx, 0));
    else
        CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, XYZ2RGB_f<float>(dcn, blueIdx, 0));
}


// 8u, 32f
void cvtBGRtoLab(const uchar * src_data, size_t src_step,
                 uchar * dst_data, size_t dst_step,
                 int width, int height,
                 int depth, int scn, bool swapBlue, bool isLab, bool srgb)
{
    CV_INSTRUMENT_REGION();

    CALL_HAL(cvtBGRtoLab, cv_hal_cvtBGRtoLab, src_data, src_step, dst_data, dst_step, width, height, depth, scn, swapBlue, isLab, srgb);

#if defined(HAVE_IPP) && !IPP_DISABLE_RGB_LAB
    CV_IPP_CHECK()
    {
        if (!srgb)
        {
            if (isLab)
            {
                if (scn == 3 && depth == CV_8U && !swapBlue)
                {
                    if (CvtColorIPPLoop(src_data, src_step, dst_data,dst_step, width, height,
                                        IPPGeneralFunctor((ippiGeneralFunc)ippiBGRToLab_8u_C3R)))
                        return;
                }
                else if (scn == 4 && depth == CV_8U && !swapBlue)
                {
                    if (CvtColorIPPLoop(src_data, src_step, dst_data,dst_step, width, height,
                                        IPPReorderGeneralFunctor(ippiSwapChannelsC4C3RTab[depth],
                                                                 (ippiGeneralFunc)ippiBGRToLab_8u_C3R, 0, 1, 2, depth)))
                        return;
                }
                else if (scn == 3 && depth == CV_8U && swapBlue) // slower than OpenCV
                {
                    if (CvtColorIPPLoop(src_data, src_step, dst_data,dst_step, width, height,
                                        IPPReorderGeneralFunctor(ippiSwapChannelsC3RTab[depth],
                                                                 (ippiGeneralFunc)ippiBGRToLab_8u_C3R, 2, 1, 0, depth)))
                        return;
                }
                else if (scn == 4 && depth == CV_8U && swapBlue) // slower than OpenCV
                {
                    if (CvtColorIPPLoop(src_data, src_step, dst_data,dst_step, width, height,
                                        IPPReorderGeneralFunctor(ippiSwapChannelsC4C3RTab[depth],
                                                                 (ippiGeneralFunc)ippiBGRToLab_8u_C3R, 2, 1, 0, depth)))
                        return;
                }
            }
            else
            {
                if (scn == 3 && swapBlue)
                {
                    if (CvtColorIPPLoop(src_data, src_step, dst_data,dst_step, width, height,
                                        IPPGeneralFunctor(ippiRGBToLUVTab[depth])))
                        return;
                }
                else if (scn == 4 && swapBlue)
                {
                    if (CvtColorIPPLoop(src_data, src_step, dst_data,dst_step, width, height,
                                        IPPReorderGeneralFunctor(ippiSwapChannelsC4C3RTab[depth],
                                                                 ippiRGBToLUVTab[depth], 0, 1, 2, depth)))
                        return;
                }
                else if (scn == 3 && !swapBlue)
                {
                    if (CvtColorIPPLoop(src_data, src_step, dst_data,dst_step, width, height,
                                        IPPReorderGeneralFunctor(ippiSwapChannelsC3RTab[depth],
                                                                 ippiRGBToLUVTab[depth], 2, 1, 0, depth)))
                        return;
                }
                else if (scn == 4 && !swapBlue)
                {
                    if (CvtColorIPPLoop(src_data, src_step, dst_data,dst_step, width, height,
                                        IPPReorderGeneralFunctor(ippiSwapChannelsC4C3RTab[depth],
                                                                 ippiRGBToLUVTab[depth], 2, 1, 0, depth)))
                        return;
                }
            }
        }
    }
#endif

    int blueIdx = swapBlue ? 2 : 0;
    if(isLab)
    {
        if( depth == CV_8U )
            CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, RGB2Lab_b(scn, blueIdx, 0, 0, srgb));
        else
            CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, RGB2Lab_f(scn, blueIdx, 0, 0, srgb));
    }
    else
    {
        if( depth == CV_8U )
            CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, RGB2Luv_b(scn, blueIdx, 0, 0, srgb));
        else
            CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, RGB2Luv_f(scn, blueIdx, 0, 0, srgb));
    }
}


// 8u, 32f
void cvtLabtoBGR(const uchar * src_data, size_t src_step,
                 uchar * dst_data, size_t dst_step,
                 int width, int height,
                 int depth, int dcn, bool swapBlue, bool isLab, bool srgb)
{
    CV_INSTRUMENT_REGION();

    CALL_HAL(cvtLabtoBGR, cv_hal_cvtLabtoBGR, src_data, src_step, dst_data, dst_step, width, height, depth, dcn, swapBlue, isLab, srgb);

#if defined(HAVE_IPP) && !IPP_DISABLE_LAB_RGB
    CV_IPP_CHECK()
    {
        if (!srgb)
        {
            if (isLab)
            {
                if( dcn == 3 && depth == CV_8U && !swapBlue)
                {
                    if( CvtColorIPPLoop(src_data, src_step, dst_data,dst_step, width, height,
                                        IPPGeneralFunctor((ippiGeneralFunc)ippiLabToBGR_8u_C3R)) )
                        return;
                }
                else if( dcn == 4 && depth == CV_8U && !swapBlue)
                {
                    if( CvtColorIPPLoop(src_data, src_step, dst_data,dst_step, width, height,
                                        IPPGeneralReorderFunctor((ippiGeneralFunc)ippiLabToBGR_8u_C3R,
                                                                 ippiSwapChannelsC3C4RTab[depth], 0, 1, 2, depth)) )
                        return;
                }
                if( dcn == 3 && depth == CV_8U && swapBlue)
                {
                    if( CvtColorIPPLoop(src_data, src_step, dst_data,dst_step, width, height,
                                        IPPGeneralReorderFunctor((ippiGeneralFunc)ippiLabToBGR_8u_C3R,
                                                                 ippiSwapChannelsC3RTab[depth], 2, 1, 0, depth)) )
                        return;
                }
                else if( dcn == 4 && depth == CV_8U && swapBlue)
                {
                    if( CvtColorIPPLoop(src_data, src_step, dst_data,dst_step, width, height,
                                        IPPGeneralReorderFunctor((ippiGeneralFunc)ippiLabToBGR_8u_C3R,
                                                                 ippiSwapChannelsC3C4RTab[depth], 2, 1, 0, depth)) )
                        return;
                }
            }
            else
            {
                if( dcn == 3 && swapBlue)
                {
                    if( CvtColorIPPLoop(src_data, src_step, dst_data,dst_step, width, height,
                                        IPPGeneralFunctor(ippiLUVToRGBTab[depth])) )
                        return;
                }
                else if( dcn == 4 && swapBlue)
                {
                    if( CvtColorIPPLoop(src_data, src_step, dst_data,dst_step, width, height,
                                        IPPGeneralReorderFunctor(ippiLUVToRGBTab[depth],
                                                                 ippiSwapChannelsC3C4RTab[depth], 0, 1, 2, depth)) )
                        return;
                }
                if( dcn == 3 && !swapBlue)
                {
                    if( CvtColorIPPLoop(src_data, src_step, dst_data,dst_step, width, height,
                                        IPPGeneralReorderFunctor(ippiLUVToRGBTab[depth],
                                                                 ippiSwapChannelsC3RTab[depth], 2, 1, 0, depth)) )
                        return;
                }
                else if( dcn == 4 && !swapBlue)
                {
                    if( CvtColorIPPLoop(src_data, src_step, dst_data,dst_step, width, height,
                                        IPPGeneralReorderFunctor(ippiLUVToRGBTab[depth],
                                                                 ippiSwapChannelsC3C4RTab[depth], 2, 1, 0, depth)) )
                        return;
                }
            }
        }
    }
#endif

    int blueIdx = swapBlue ? 2 : 0;
    if(isLab)
    {
        if( depth == CV_8U )
            CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, Lab2RGB_b(dcn, blueIdx, 0, 0, srgb));
        else
            CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, Lab2RGB_f(dcn, blueIdx, 0, 0, srgb));
    }
    else
    {
        if( depth == CV_8U )
            CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, Luv2RGB_b(dcn, blueIdx, 0, 0, srgb));
        else
            CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, Luv2RGB_f(dcn, blueIdx, 0, 0, srgb));
    }
}

} // namespace hal

//
// OCL calls
//

#ifdef HAVE_OPENCL

bool oclCvtColorBGR2Luv( InputArray _src, OutputArray _dst, int bidx, bool srgb)
{
    OclHelper< Set<3, 4>, Set<3>, Set<CV_8U, CV_32F> > h(_src, _dst, 3);

    if(!h.createKernel("BGR2Luv", ocl::imgproc::color_lab_oclsrc,
                       format("-D dcn=3 -D bidx=%d%s", bidx, srgb ? " -D SRGB" : "")))
    {
        return false;
    }

    // Prepare additional arguments

    initLabTabs();

    static UMat usRGBGammaTab, ucoeffs, uLabCbrtTab;

    if (srgb && usRGBGammaTab.empty())
        Mat(1, GAMMA_TAB_SIZE * 4, CV_32FC1, const_cast<float*>(sRGBGammaTab)).copyTo(usRGBGammaTab);
    if (uLabCbrtTab.empty())
        Mat(1, LAB_CBRT_TAB_SIZE * 4, CV_32FC1, const_cast<float*>(LabCbrtTab)).copyTo(uLabCbrtTab);

    float coeffs[9];
    softdouble whitePt[3];
    for(int i = 0; i < 3; i++)
        whitePt[i] = D65[i];

    for (int i = 0; i < 3; i++)
    {
        int j = i * 3;

        softfloat c0 = sRGB2XYZ_D65[j    ];
        softfloat c1 = sRGB2XYZ_D65[j + 1];
        softfloat c2 = sRGB2XYZ_D65[j + 2];

        coeffs[j + (bidx ^ 2)] = c0;
        coeffs[j + 1]          = c1;
        coeffs[j + bidx]       = c2;

        CV_Assert( c0 >= 0 && c1 >= 0 && c2 >= 0 &&
                   c0 + c1 + c2 < softfloat(3)/softfloat(2));
    }

    softfloat d = whitePt[0] +
                  whitePt[1]*softdouble(15) +
                  whitePt[2]*softdouble(3);
    d = softfloat::one()/max(d, softfloat(FLT_EPSILON));
    float un = d*softfloat(13*4)*whitePt[0];
    float vn = d*softfloat(13*9)*whitePt[1];

    Mat(1, 9, CV_32FC1, coeffs).copyTo(ucoeffs);

    ocl::KernelArg ucoeffsarg = ocl::KernelArg::PtrReadOnly(ucoeffs);

    ocl::KernelArg LabCbrtTabarg = ocl::KernelArg::PtrReadOnly(uLabCbrtTab);

    // Setup additional arguments and run

    if(srgb)
        h.setArg(ocl::KernelArg::PtrReadOnly(usRGBGammaTab));

    h.setArg(LabCbrtTabarg);
    h.setArg(ucoeffsarg);
    h.setArg(un); h.setArg(vn);

    return h.run();
}


bool oclCvtColorBGR2Lab( InputArray _src, OutputArray _dst, int bidx, bool srgb )
{
    OclHelper< Set<3, 4>, Set<3>, Set<CV_8U, CV_32F> > h(_src, _dst, 3);

    if(!h.createKernel("BGR2Lab", ocl::imgproc::color_lab_oclsrc,
                       format("-D dcn=3 -D bidx=%d%s", bidx, srgb ? " -D SRGB" : "")))
    {
        return false;
    }

    // Prepare and set additional arguments

    initLabTabs();

    if (_src.depth() == CV_8U)
    {
        static UMat usRGBGammaTab, ulinearGammaTab, uLabCbrtTab, ucoeffs;

        if (srgb && usRGBGammaTab.empty())
            Mat(1, 256, CV_16UC1, sRGBGammaTab_b).copyTo(usRGBGammaTab);
        else if (ulinearGammaTab.empty())
            Mat(1, 256, CV_16UC1, linearGammaTab_b).copyTo(ulinearGammaTab);
        if (uLabCbrtTab.empty())
            Mat(1, LAB_CBRT_TAB_SIZE_B, CV_16UC1, LabCbrtTab_b).copyTo(uLabCbrtTab);

        int coeffs[9];
        static const softfloat lshift(1 << lab_shift);
        for( int i = 0; i < 3; i++ )
        {
            coeffs[i*3+(bidx^2)] = cvRound(lshift*sRGB2XYZ_D65[i*3  ]/D65[i]);
            coeffs[i*3+1]        = cvRound(lshift*sRGB2XYZ_D65[i*3+1]/D65[i]);
            coeffs[i*3+bidx]     = cvRound(lshift*sRGB2XYZ_D65[i*3+2]/D65[i]);

            CV_Assert(coeffs[i*3] >= 0 && coeffs[i*3+1] >= 0 && coeffs[i*3+2] >= 0 &&
                      coeffs[i*3] + coeffs[i*3+1] + coeffs[i*3+2] < 2*(1 << lab_shift));
        }
        Mat(1, 9, CV_32SC1, coeffs).copyTo(ucoeffs);

        const int Lscale = (116*255+50)/100;
        const int Lshift = -((16*255*(1 << lab_shift2) + 50)/100);

        h.setArg(ocl::KernelArg::PtrReadOnly(srgb ? usRGBGammaTab : ulinearGammaTab));
        h.setArg(ocl::KernelArg::PtrReadOnly(uLabCbrtTab));
        h.setArg(ocl::KernelArg::PtrReadOnly(ucoeffs));
        h.setArg(Lscale); h.setArg(Lshift);
    }
    else
    {
        static UMat usRGBGammaTab, ucoeffs;

        if (srgb && usRGBGammaTab.empty())
            Mat(1, GAMMA_TAB_SIZE * 4, CV_32FC1, const_cast<float*>(sRGBGammaTab)).copyTo(usRGBGammaTab);

        softdouble whitePt[3];
        for(int i = 0; i < 3; i++)
            whitePt[i] = D65[i];

        softdouble scale[] = { softdouble::one() / whitePt[0],
                               softdouble::one(),
                               softdouble::one() / whitePt[2] };

        float coeffs[9];
        for (int i = 0; i < 3; i++)
        {
            int j = i * 3;

            softfloat c0 = scale[i] * sRGB2XYZ_D65[j    ];
            softfloat c1 = scale[i] * sRGB2XYZ_D65[j + 1];
            softfloat c2 = scale[i] * sRGB2XYZ_D65[j + 2];

            coeffs[j + (bidx ^ 2)] = c0;
            coeffs[j + 1]          = c1;
            coeffs[j + bidx]       = c2;

            CV_Assert( c0 >= 0 && c1 >= 0 && c2 >= 0 &&
                       c0 + c1 + c2 < softfloat((int)LAB_CBRT_TAB_SIZE));
        }

        Mat(1, 9, CV_32FC1, coeffs).copyTo(ucoeffs);

        static const float _a = softfloat(16)/softfloat(116);
        static const float _1_3f = softfloat::one()/softfloat(3);
        ocl::KernelArg ucoeffsarg = ocl::KernelArg::PtrReadOnly(ucoeffs);

        if (srgb)
            h.setArg(ocl::KernelArg::PtrReadOnly(usRGBGammaTab));

        h.setArg(ucoeffsarg);
        h.setArg(_1_3f); h.setArg(_a);
    }

    return h.run();
}


bool oclCvtColorLab2BGR(InputArray _src, OutputArray _dst, int dcn, int bidx, bool srgb)
{
    OclHelper< Set<3>, Set<3, 4>, Set<CV_8U, CV_32F> > h(_src, _dst, dcn);

    if(!h.createKernel("Lab2BGR", ocl::imgproc::color_lab_oclsrc,
                       format("-D dcn=%d -D bidx=%d%s", dcn, bidx, srgb ? " -D SRGB" : "")))
    {
        return false;
    }

    // Prepare additional arguments

    initLabTabs();

    static UMat ucoeffs, usRGBInvGammaTab;

    if (srgb && usRGBInvGammaTab.empty())
        Mat(1, GAMMA_TAB_SIZE*4, CV_32FC1, const_cast<float*>(sRGBInvGammaTab)).copyTo(usRGBInvGammaTab);

    float coeffs[9];
    softdouble whitePt[3];
    for(int i = 0; i < 3; i++)
        whitePt[i] = D65[i];

    for( int i = 0; i < 3; i++ )
    {
        coeffs[i+(bidx^2)*3] = (float)(XYZ2sRGB_D65[i  ]*whitePt[i]);
        coeffs[i+3]          = (float)(XYZ2sRGB_D65[i+3]*whitePt[i]);
        coeffs[i+bidx*3]     = (float)(XYZ2sRGB_D65[i+6]*whitePt[i]);
    }

    Mat(1, 9, CV_32FC1, coeffs).copyTo(ucoeffs);

    float lThresh = softfloat(8); // 0.008856f * 903.3f  = (6/29)^3*(29/3)^3 = 8
    float fThresh = softfloat(6)/softfloat(29); // 7.787f * 0.008856f + 16.0f / 116.0f = 6/29

    ocl::KernelArg coeffsarg = ocl::KernelArg::PtrReadOnly(ucoeffs);

    // Set additional arguments and run

    if(srgb)
        h.setArg(ocl::KernelArg::PtrReadOnly(usRGBInvGammaTab));

    h.setArg(coeffsarg);
    h.setArg(lThresh);
    h.setArg(fThresh);

    return h.run();
}


bool oclCvtColorLuv2BGR(InputArray _src, OutputArray _dst, int dcn, int bidx, bool srgb)
{
    OclHelper< Set<3>, Set<3, 4>, Set<CV_8U, CV_32F> > h(_src, _dst, dcn);

    if(!h.createKernel("Luv2BGR", ocl::imgproc::color_lab_oclsrc,
                       format("-D dcn=%d -D bidx=%d%s", dcn, bidx, srgb ? " -D SRGB" : "")))
    {
        return false;
    }

    // Prepare additional arguments

    initLabTabs();

    static UMat ucoeffs, usRGBInvGammaTab;

    if (srgb && usRGBInvGammaTab.empty())
        Mat(1, GAMMA_TAB_SIZE*4, CV_32FC1, const_cast<float*>(sRGBInvGammaTab)).copyTo(usRGBInvGammaTab);

    float coeffs[9];
    softdouble whitePt[3];
    for(int i = 0; i < 3; i++)
        whitePt[i] = D65[i];

    for( int i = 0; i < 3; i++ )
    {
        coeffs[i+(bidx^2)*3] = (float)(XYZ2sRGB_D65[i  ]);
        coeffs[i+3]          = (float)(XYZ2sRGB_D65[i+3]);
        coeffs[i+bidx*3]     = (float)(XYZ2sRGB_D65[i+6]);
    }

    softfloat d = whitePt[0] +
            whitePt[1]*softdouble(15) +
            whitePt[2]*softdouble(3);
    d = softfloat::one()/max(d, softfloat(FLT_EPSILON));
    float un = softfloat(4*13)*d*whitePt[0];
    float vn = softfloat(9*13)*d*whitePt[1];

    Mat(1, 9, CV_32FC1, coeffs).copyTo(ucoeffs);

    ocl::KernelArg coeffsarg = ocl::KernelArg::PtrReadOnly(ucoeffs);

    // Set additional arguments and run

    if(srgb)
        h.setArg(ocl::KernelArg::PtrReadOnly(usRGBInvGammaTab));

    h.setArg(coeffsarg);
    h.setArg(un); h.setArg(vn);

    return h.run();
}


bool oclCvtColorBGR2XYZ( InputArray _src, OutputArray _dst, int bidx )
{
    OclHelper< Set<3, 4>, Set<3>, Set<CV_8U, CV_16U, CV_32F> > h(_src, _dst, 3);

    if(!h.createKernel("RGB2XYZ", ocl::imgproc::color_lab_oclsrc,
                       format("-D dcn=3 -D bidx=%d", bidx)))
    {
        return false;
    }

    // Prepare additional arguments

    UMat c;
    if (_src.depth() == CV_32F)
    {
        float coeffs[9];
        for(int i = 0; i < 9; i++)
            coeffs[i] = (float)sRGB2XYZ_D65[i];
        if (bidx == 0)
        {
            std::swap(coeffs[0], coeffs[2]);
            std::swap(coeffs[3], coeffs[5]);
            std::swap(coeffs[6], coeffs[8]);
        }
        Mat(1, 9, CV_32FC1, &coeffs[0]).copyTo(c);
    }
    else
    {
        int coeffs[9];
        for(int i = 0; i < 9; i++)
            coeffs[i] = sRGB2XYZ_D65_i[i];
        if (bidx == 0)
        {
            std::swap(coeffs[0], coeffs[2]);
            std::swap(coeffs[3], coeffs[5]);
            std::swap(coeffs[6], coeffs[8]);
        }
        Mat(1, 9, CV_32SC1, &coeffs[0]).copyTo(c);
    }

    // Set additional arguments and run

    h.setArg(ocl::KernelArg::PtrReadOnly(c));

    return h.run();
}


bool oclCvtColorXYZ2BGR( InputArray _src, OutputArray _dst, int dcn, int bidx )
{
    OclHelper< Set<3>, Set<3, 4>, Set<CV_8U, CV_16U, CV_32F> > h(_src, _dst, dcn);

    if(!h.createKernel("XYZ2RGB", ocl::imgproc::color_lab_oclsrc,
                       format("-D dcn=%d -D bidx=%d", dcn, bidx)))
    {
        return false;
    }

    // Prepare additional arguments

    UMat c;
    if (_src.depth() == CV_32F)
    {
        float coeffs[9];
        for(int i = 0; i < 9; i++)
            coeffs[i] = (float)XYZ2sRGB_D65[i];
        if (bidx == 0)
        {
            std::swap(coeffs[0], coeffs[6]);
            std::swap(coeffs[1], coeffs[7]);
            std::swap(coeffs[2], coeffs[8]);
        }
        Mat(1, 9, CV_32FC1, &coeffs[0]).copyTo(c);
    }
    else
    {
        int coeffs[9];
        for(int i = 0; i < 9; i++)
            coeffs[i] = XYZ2sRGB_D65_i[i];
        if (bidx == 0)
        {
            std::swap(coeffs[0], coeffs[6]);
            std::swap(coeffs[1], coeffs[7]);
            std::swap(coeffs[2], coeffs[8]);
        }
        Mat(1, 9, CV_32SC1, &coeffs[0]).copyTo(c);
    }

    // Set additional arguments and run

    h.setArg(ocl::KernelArg::PtrReadOnly(c));

    return h.run();
}

#endif

//
// HAL calls
//

void cvtColorBGR2Lab( InputArray _src, OutputArray _dst, bool swapb, bool srgb)
{
    CvtHelper<Set<3, 4>, Set<3>, Set<CV_8U, CV_32F> > h(_src, _dst, 3);

    hal::cvtBGRtoLab(h.src.data, h.src.step, h.dst.data, h.dst.step, h.src.cols, h.src.rows,
                     h.depth, h.scn, swapb, true, srgb);
}


void cvtColorBGR2Luv( InputArray _src, OutputArray _dst, bool swapb, bool srgb)
{
    CvtHelper< Set<3, 4>, Set<3>, Set<CV_8U, CV_32F> > h(_src, _dst, 3);

    hal::cvtBGRtoLab(h.src.data, h.src.step, h.dst.data, h.dst.step, h.src.cols, h.src.rows,
                     h.depth, h.scn, swapb, false, srgb);
}


void cvtColorLab2BGR( InputArray _src, OutputArray _dst, int dcn, bool swapb, bool srgb )
{
    if( dcn <= 0 ) dcn = 3;
    CvtHelper< Set<3>, Set<3, 4>, Set<CV_8U, CV_32F> > h(_src, _dst, dcn);

    hal::cvtLabtoBGR(h.src.data, h.src.step, h.dst.data, h.dst.step, h.src.cols, h.src.rows,
                     h.depth, dcn, swapb, true, srgb);
}


void cvtColorLuv2BGR( InputArray _src, OutputArray _dst, int dcn, bool swapb, bool srgb )
{
    if( dcn <= 0 ) dcn = 3;
    CvtHelper< Set<3>, Set<3, 4>, Set<CV_8U, CV_32F> > h(_src, _dst, dcn);

    hal::cvtLabtoBGR(h.src.data, h.src.step, h.dst.data, h.dst.step, h.src.cols, h.src.rows,
                     h.depth, dcn, swapb, false, srgb);
}


void cvtColorBGR2XYZ( InputArray _src, OutputArray _dst, bool swapb )
{
    CvtHelper< Set<3, 4>, Set<3>, Set<CV_8U, CV_16U, CV_32F> > h(_src, _dst, 3);

    hal::cvtBGRtoXYZ(h.src.data, h.src.step, h.dst.data, h.dst.step, h.src.cols, h.src.rows, h.depth, h.scn, swapb);
}


void cvtColorXYZ2BGR( InputArray _src, OutputArray _dst, int dcn, bool swapb )
{
    if( dcn <= 0 ) dcn = 3;
    CvtHelper< Set<3>, Set<3, 4>, Set<CV_8U, CV_16U, CV_32F> > h(_src, _dst, dcn);

    hal::cvtXYZtoBGR(h.src.data, h.src.step, h.dst.data, h.dst.step, h.src.cols, h.src.rows, h.depth, dcn, swapb);
}

} // namespace cv
