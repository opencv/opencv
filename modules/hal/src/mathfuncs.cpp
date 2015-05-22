/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"

#undef HAVE_IPP

namespace cv { namespace hal {

///////////////////////////////////// ATAN2 ////////////////////////////////////
static const float atan2_p1 = 0.9997878412794807f*(float)(180/CV_PI);
static const float atan2_p3 = -0.3258083974640975f*(float)(180/CV_PI);
static const float atan2_p5 = 0.1555786518463281f*(float)(180/CV_PI);
static const float atan2_p7 = -0.04432655554792128f*(float)(180/CV_PI);

#if CV_NEON
static inline float32x4_t cv_vrecpq_f32(float32x4_t val)
{
    float32x4_t reciprocal = vrecpeq_f32(val);
    reciprocal = vmulq_f32(vrecpsq_f32(val, reciprocal), reciprocal);
    reciprocal = vmulq_f32(vrecpsq_f32(val, reciprocal), reciprocal);
    return reciprocal;
}
#endif

void fastAtan2(const float *Y, const float *X, float *angle, int len, bool angleInDegrees )
{
    int i = 0;
    float scale = angleInDegrees ? 1 : (float)(CV_PI/180);

#ifdef HAVE_TEGRA_OPTIMIZATION
    if (tegra::useTegra() && tegra::FastAtan2_32f(Y, X, angle, len, scale))
        return;
#endif

#if CV_SSE2
    Cv32suf iabsmask; iabsmask.i = 0x7fffffff;
    __m128 eps = _mm_set1_ps((float)DBL_EPSILON), absmask = _mm_set1_ps(iabsmask.f);
    __m128 _90 = _mm_set1_ps(90.f), _180 = _mm_set1_ps(180.f), _360 = _mm_set1_ps(360.f);
    __m128 z = _mm_setzero_ps(), scale4 = _mm_set1_ps(scale);
    __m128 p1 = _mm_set1_ps(atan2_p1), p3 = _mm_set1_ps(atan2_p3);
    __m128 p5 = _mm_set1_ps(atan2_p5), p7 = _mm_set1_ps(atan2_p7);

    for( ; i <= len - 4; i += 4 )
    {
        __m128 x = _mm_loadu_ps(X + i), y = _mm_loadu_ps(Y + i);
        __m128 ax = _mm_and_ps(x, absmask), ay = _mm_and_ps(y, absmask);
        __m128 mask = _mm_cmplt_ps(ax, ay);
        __m128 tmin = _mm_min_ps(ax, ay), tmax = _mm_max_ps(ax, ay);
        __m128 c = _mm_div_ps(tmin, _mm_add_ps(tmax, eps));
        __m128 c2 = _mm_mul_ps(c, c);
        __m128 a = _mm_mul_ps(c2, p7);
        a = _mm_mul_ps(_mm_add_ps(a, p5), c2);
        a = _mm_mul_ps(_mm_add_ps(a, p3), c2);
        a = _mm_mul_ps(_mm_add_ps(a, p1), c);

        __m128 b = _mm_sub_ps(_90, a);
        a = _mm_xor_ps(a, _mm_and_ps(_mm_xor_ps(a, b), mask));

        b = _mm_sub_ps(_180, a);
        mask = _mm_cmplt_ps(x, z);
        a = _mm_xor_ps(a, _mm_and_ps(_mm_xor_ps(a, b), mask));

        b = _mm_sub_ps(_360, a);
        mask = _mm_cmplt_ps(y, z);
        a = _mm_xor_ps(a, _mm_and_ps(_mm_xor_ps(a, b), mask));

        a = _mm_mul_ps(a, scale4);
        _mm_storeu_ps(angle + i, a);
    }
#elif CV_NEON
    float32x4_t eps = vdupq_n_f32((float)DBL_EPSILON);
    float32x4_t _90 = vdupq_n_f32(90.f), _180 = vdupq_n_f32(180.f), _360 = vdupq_n_f32(360.f);
    float32x4_t z = vdupq_n_f32(0.0f), scale4 = vdupq_n_f32(scale);
    float32x4_t p1 = vdupq_n_f32(atan2_p1), p3 = vdupq_n_f32(atan2_p3);
    float32x4_t p5 = vdupq_n_f32(atan2_p5), p7 = vdupq_n_f32(atan2_p7);

    for( ; i <= len - 4; i += 4 )
    {
        float32x4_t x = vld1q_f32(X + i), y = vld1q_f32(Y + i);
        float32x4_t ax = vabsq_f32(x), ay = vabsq_f32(y);
        float32x4_t tmin = vminq_f32(ax, ay), tmax = vmaxq_f32(ax, ay);
        float32x4_t c = vmulq_f32(tmin, cv_vrecpq_f32(vaddq_f32(tmax, eps)));
        float32x4_t c2 = vmulq_f32(c, c);
        float32x4_t a = vmulq_f32(c2, p7);
        a = vmulq_f32(vaddq_f32(a, p5), c2);
        a = vmulq_f32(vaddq_f32(a, p3), c2);
        a = vmulq_f32(vaddq_f32(a, p1), c);

        a = vbslq_f32(vcgeq_f32(ax, ay), a, vsubq_f32(_90, a));
        a = vbslq_f32(vcltq_f32(x, z), vsubq_f32(_180, a), a);
        a = vbslq_f32(vcltq_f32(y, z), vsubq_f32(_360, a), a);

        vst1q_f32(angle + i, vmulq_f32(a, scale4));
    }
#endif

    for( ; i < len; i++ )
    {
        float x = X[i], y = Y[i];
        float ax = std::abs(x), ay = std::abs(y);
        float a, c, c2;
        if( ax >= ay )
        {
            c = ay/(ax + (float)DBL_EPSILON);
            c2 = c*c;
            a = (((atan2_p7*c2 + atan2_p5)*c2 + atan2_p3)*c2 + atan2_p1)*c;
        }
        else
        {
            c = ax/(ay + (float)DBL_EPSILON);
            c2 = c*c;
            a = 90.f - (((atan2_p7*c2 + atan2_p5)*c2 + atan2_p3)*c2 + atan2_p1)*c;
        }
        if( x < 0 )
            a = 180.f - a;
        if( y < 0 )
            a = 360.f - a;
        angle[i] = (float)(a*scale);
    }
}


void magnitude(const float* x, const float* y, float* mag, int len)
{
#if defined HAVE_IPP
    CV_IPP_CHECK()
    {
        IppStatus status = ippsMagnitude_32f(x, y, mag, len);
        if (status >= 0)
        {
            CV_IMPL_ADD(CV_IMPL_IPP);
            return;
        }
        setIppErrorStatus();
    }
#endif

    int i = 0;

#if CV_SIMD128
    for( ; i <= len - 8; i += 8 )
    {
        v_float32x4 x0 = v_load(x + i), x1 = v_load(x + i + 4);
        v_float32x4 y0 = v_load(y + i), y1 = v_load(y + i + 4);
        x0 = v_sqrt(v_muladd(x0, x0, y0*y0));
        x1 = v_sqrt(v_muladd(x1, x1, y1*y1));
        v_store(mag + i, x0);
        v_store(mag + i + 4, x1);
    }
#endif

    for( ; i < len; i++ )
    {
        float x0 = x[i], y0 = y[i];
        mag[i] = std::sqrt(x0*x0 + y0*y0);
    }
}

void magnitude(const double* x, const double* y, double* mag, int len)
{
#if defined(HAVE_IPP)
    CV_IPP_CHECK()
    {
        IppStatus status = ippsMagnitude_64f(x, y, mag, len);
        if (status >= 0)
        {
            CV_IMPL_ADD(CV_IMPL_IPP);
            return;
        }
        setIppErrorStatus();
    }
#endif

    int i = 0;

#if CV_SIMD128_64F
    for( ; i <= len - 4; i += 4 )
    {
        v_float64x2 x0 = v_load(x + i), x1 = v_load(x + i + 2);
        v_float64x2 y0 = v_load(y + i), y1 = v_load(y + i + 2);
        x0 = v_sqrt(v_muladd(x0, x0, y0*y0));
        x1 = v_sqrt(v_muladd(x1, x1, y1*y1));
        v_store(mag + i, x0);
        v_store(mag + i + 2, x1);
    }
#endif

    for( ; i < len; i++ )
    {
        double x0 = x[i], y0 = y[i];
        mag[i] = std::sqrt(x0*x0 + y0*y0);
    }
}


void invSqrt(const float* src, float* dst, int len)
{
#if defined(HAVE_IPP)
    CV_IPP_CHECK()
    {
        if (ippsInvSqrt_32f_A21(src, dst, len) >= 0)
        {
            CV_IMPL_ADD(CV_IMPL_IPP);
            return;
        }
        setIppErrorStatus();
    }
#endif

    int i = 0;

#if CV_SIMD128
    for( ; i <= len - 8; i += 8 )
    {
        v_float32x4 t0 = v_load(src + i), t1 = v_load(src + i + 4);
        t0 = v_invsqrt(t0);
        t1 = v_invsqrt(t1);
        v_store(dst + i, t0); v_store(dst + i + 4, t1);
    }
#endif

    for( ; i < len; i++ )
        dst[i] = 1/std::sqrt(src[i]);
}


void invSqrt(const double* src, double* dst, int len)
{
    int i = 0;

#if CV_SSE2
    __m128d v_1 = _mm_set1_pd(1.0);
    for ( ; i <= len - 2; i += 2)
        _mm_storeu_pd(dst + i, _mm_div_pd(v_1, _mm_sqrt_pd(_mm_loadu_pd(src + i))));
#endif

    for( ; i < len; i++ )
        dst[i] = 1/std::sqrt(src[i]);
}


void sqrt(const float* src, float* dst, int len)
{
#if defined(HAVE_IPP)
    CV_IPP_CHECK()
    {
        if (ippsSqrt_32f_A21(src, dst, len) >= 0)
        {
            CV_IMPL_ADD(CV_IMPL_IPP);
            return;
        }
        setIppErrorStatus();
    }
#endif

    int i = 0;

#if CV_SIMD128
    for( ; i <= len - 8; i += 8 )
    {
        v_float32x4 t0 = v_load(src + i), t1 = v_load(src + i + 4);
        t0 = v_sqrt(t0);
        t1 = v_sqrt(t1);
        v_store(dst + i, t0); v_store(dst + i + 4, t1);
    }
#endif

    for( ; i < len; i++ )
        dst[i] = std::sqrt(src[i]);
}


void sqrt(const double* src, double* dst, int len)
{
#if defined(HAVE_IPP)
    CV_IPP_CHECK()
    {
        if (ippsSqrt_64f_A50(src, dst, len) >= 0)
        {
            CV_IMPL_ADD(CV_IMPL_IPP);
            return;
        }
        setIppErrorStatus();
    }
#endif

    int i = 0;

#if CV_SIMD128_64F
    for( ; i <= len - 4; i += 4 )
    {
        v_float64x2 t0 = v_load(src + i), t1 = v_load(src + i + 2);
        t0 = v_sqrt(t0);
        t1 = v_sqrt(t1);
        v_store(dst + i, t0); v_store(dst + i + 2, t1);
    }
#endif

    for( ; i < len; i++ )
        dst[i] = std::sqrt(src[i]);
}

////////////////////////////////////// EXP /////////////////////////////////////

typedef union
{
    struct {
#if ( defined( WORDS_BIGENDIAN ) && !defined( OPENCV_UNIVERSAL_BUILD ) ) || defined( __BIG_ENDIAN__ )
        int hi;
        int lo;
#else
        int lo;
        int hi;
#endif
    } i;
    double d;
}
DBLINT;

#define EXPTAB_SCALE 6
#define EXPTAB_MASK  ((1 << EXPTAB_SCALE) - 1)

#define EXPPOLY_32F_A0 .9670371139572337719125840413672004409288e-2

static const double expTab[] = {
    1.0 * EXPPOLY_32F_A0,
    1.0108892860517004600204097905619 * EXPPOLY_32F_A0,
    1.0218971486541166782344801347833 * EXPPOLY_32F_A0,
    1.0330248790212284225001082839705 * EXPPOLY_32F_A0,
    1.0442737824274138403219664787399 * EXPPOLY_32F_A0,
    1.0556451783605571588083413251529 * EXPPOLY_32F_A0,
    1.0671404006768236181695211209928 * EXPPOLY_32F_A0,
    1.0787607977571197937406800374385 * EXPPOLY_32F_A0,
    1.0905077326652576592070106557607 * EXPPOLY_32F_A0,
    1.1023825833078409435564142094256 * EXPPOLY_32F_A0,
    1.1143867425958925363088129569196 * EXPPOLY_32F_A0,
    1.126521618608241899794798643787 * EXPPOLY_32F_A0,
    1.1387886347566916537038302838415 * EXPPOLY_32F_A0,
    1.151189229952982705817759635202 * EXPPOLY_32F_A0,
    1.1637248587775775138135735990922 * EXPPOLY_32F_A0,
    1.1763969916502812762846457284838 * EXPPOLY_32F_A0,
    1.1892071150027210667174999705605 * EXPPOLY_32F_A0,
    1.2021567314527031420963969574978 * EXPPOLY_32F_A0,
    1.2152473599804688781165202513388 * EXPPOLY_32F_A0,
    1.2284805361068700056940089577928 * EXPPOLY_32F_A0,
    1.2418578120734840485936774687266 * EXPPOLY_32F_A0,
    1.2553807570246910895793906574423 * EXPPOLY_32F_A0,
    1.2690509571917332225544190810323 * EXPPOLY_32F_A0,
    1.2828700160787782807266697810215 * EXPPOLY_32F_A0,
    1.2968395546510096659337541177925 * EXPPOLY_32F_A0,
    1.3109612115247643419229917863308 * EXPPOLY_32F_A0,
    1.3252366431597412946295370954987 * EXPPOLY_32F_A0,
    1.3396675240533030053600306697244 * EXPPOLY_32F_A0,
    1.3542555469368927282980147401407 * EXPPOLY_32F_A0,
    1.3690024229745906119296011329822 * EXPPOLY_32F_A0,
    1.3839098819638319548726595272652 * EXPPOLY_32F_A0,
    1.3989796725383111402095281367152 * EXPPOLY_32F_A0,
    1.4142135623730950488016887242097 * EXPPOLY_32F_A0,
    1.4296133383919700112350657782751 * EXPPOLY_32F_A0,
    1.4451808069770466200370062414717 * EXPPOLY_32F_A0,
    1.4609177941806469886513028903106 * EXPPOLY_32F_A0,
    1.476826145939499311386907480374 * EXPPOLY_32F_A0,
    1.4929077282912648492006435314867 * EXPPOLY_32F_A0,
    1.5091644275934227397660195510332 * EXPPOLY_32F_A0,
    1.5255981507445383068512536895169 * EXPPOLY_32F_A0,
    1.5422108254079408236122918620907 * EXPPOLY_32F_A0,
    1.5590044002378369670337280894749 * EXPPOLY_32F_A0,
    1.5759808451078864864552701601819 * EXPPOLY_32F_A0,
    1.5931421513422668979372486431191 * EXPPOLY_32F_A0,
    1.6104903319492543081795206673574 * EXPPOLY_32F_A0,
    1.628027421857347766848218522014 * EXPPOLY_32F_A0,
    1.6457554781539648445187567247258 * EXPPOLY_32F_A0,
    1.6636765803267364350463364569764 * EXPPOLY_32F_A0,
    1.6817928305074290860622509524664 * EXPPOLY_32F_A0,
    1.7001063537185234695013625734975 * EXPPOLY_32F_A0,
    1.7186192981224779156293443764563 * EXPPOLY_32F_A0,
    1.7373338352737062489942020818722 * EXPPOLY_32F_A0,
    1.7562521603732994831121606193753 * EXPPOLY_32F_A0,
    1.7753764925265212525505592001993 * EXPPOLY_32F_A0,
    1.7947090750031071864277032421278 * EXPPOLY_32F_A0,
    1.8142521755003987562498346003623 * EXPPOLY_32F_A0,
    1.8340080864093424634870831895883 * EXPPOLY_32F_A0,
    1.8539791250833855683924530703377 * EXPPOLY_32F_A0,
    1.8741676341102999013299989499544 * EXPPOLY_32F_A0,
    1.8945759815869656413402186534269 * EXPPOLY_32F_A0,
    1.9152065613971472938726112702958 * EXPPOLY_32F_A0,
    1.9360617934922944505980559045667 * EXPPOLY_32F_A0,
    1.9571441241754002690183222516269 * EXPPOLY_32F_A0,
    1.9784560263879509682582499181312 * EXPPOLY_32F_A0,
};


// the code below uses _mm_cast* intrinsics, which are not avialable on VS2005
#if (defined _MSC_VER && _MSC_VER < 1500) || \
(!defined __APPLE__ && defined __GNUC__ && __GNUC__*100 + __GNUC_MINOR__ < 402)
#undef CV_SSE2
#define CV_SSE2 0
#endif

static const double exp_prescale = 1.4426950408889634073599246810019 * (1 << EXPTAB_SCALE);
static const double exp_postscale = 1./(1 << EXPTAB_SCALE);
static const double exp_max_val = 3000.*(1 << EXPTAB_SCALE); // log10(DBL_MAX) < 3000

void exp( const float *_x, float *y, int n )
{
    static const float
    A4 = (float)(1.000000000000002438532970795181890933776 / EXPPOLY_32F_A0),
    A3 = (float)(.6931471805521448196800669615864773144641 / EXPPOLY_32F_A0),
    A2 = (float)(.2402265109513301490103372422686535526573 / EXPPOLY_32F_A0),
    A1 = (float)(.5550339366753125211915322047004666939128e-1 / EXPPOLY_32F_A0);

#undef EXPPOLY
#define EXPPOLY(x)  \
(((((x) + A1)*(x) + A2)*(x) + A3)*(x) + A4)

    int i = 0;
    const Cv32suf* x = (const Cv32suf*)_x;
    Cv32suf buf[4];

#if CV_SSE2
    if( n >= 8 )
    {
        static const __m128d prescale2 = _mm_set1_pd(exp_prescale);
        static const __m128 postscale4 = _mm_set1_ps((float)exp_postscale);
        static const __m128 maxval4 = _mm_set1_ps((float)(exp_max_val/exp_prescale));
        static const __m128 minval4 = _mm_set1_ps((float)(-exp_max_val/exp_prescale));

        static const __m128 mA1 = _mm_set1_ps(A1);
        static const __m128 mA2 = _mm_set1_ps(A2);
        static const __m128 mA3 = _mm_set1_ps(A3);
        static const __m128 mA4 = _mm_set1_ps(A4);
        bool y_aligned = (size_t)(void*)y % 16 == 0;

        ushort CV_DECL_ALIGNED(16) tab_idx[8];

        for( ; i <= n - 8; i += 8 )
        {
            __m128 xf0, xf1;
            xf0 = _mm_loadu_ps(&x[i].f);
            xf1 = _mm_loadu_ps(&x[i+4].f);
            __m128i xi0, xi1, xi2, xi3;

            xf0 = _mm_min_ps(_mm_max_ps(xf0, minval4), maxval4);
            xf1 = _mm_min_ps(_mm_max_ps(xf1, minval4), maxval4);

            __m128d xd0 = _mm_cvtps_pd(xf0);
            __m128d xd2 = _mm_cvtps_pd(_mm_movehl_ps(xf0, xf0));
            __m128d xd1 = _mm_cvtps_pd(xf1);
            __m128d xd3 = _mm_cvtps_pd(_mm_movehl_ps(xf1, xf1));

            xd0 = _mm_mul_pd(xd0, prescale2);
            xd2 = _mm_mul_pd(xd2, prescale2);
            xd1 = _mm_mul_pd(xd1, prescale2);
            xd3 = _mm_mul_pd(xd3, prescale2);

            xi0 = _mm_cvtpd_epi32(xd0);
            xi2 = _mm_cvtpd_epi32(xd2);

            xi1 = _mm_cvtpd_epi32(xd1);
            xi3 = _mm_cvtpd_epi32(xd3);

            xd0 = _mm_sub_pd(xd0, _mm_cvtepi32_pd(xi0));
            xd2 = _mm_sub_pd(xd2, _mm_cvtepi32_pd(xi2));
            xd1 = _mm_sub_pd(xd1, _mm_cvtepi32_pd(xi1));
            xd3 = _mm_sub_pd(xd3, _mm_cvtepi32_pd(xi3));

            xf0 = _mm_movelh_ps(_mm_cvtpd_ps(xd0), _mm_cvtpd_ps(xd2));
            xf1 = _mm_movelh_ps(_mm_cvtpd_ps(xd1), _mm_cvtpd_ps(xd3));

            xf0 = _mm_mul_ps(xf0, postscale4);
            xf1 = _mm_mul_ps(xf1, postscale4);

            xi0 = _mm_unpacklo_epi64(xi0, xi2);
            xi1 = _mm_unpacklo_epi64(xi1, xi3);
            xi0 = _mm_packs_epi32(xi0, xi1);

            _mm_store_si128((__m128i*)tab_idx, _mm_and_si128(xi0, _mm_set1_epi16(EXPTAB_MASK)));

            xi0 = _mm_add_epi16(_mm_srai_epi16(xi0, EXPTAB_SCALE), _mm_set1_epi16(127));
            xi0 = _mm_max_epi16(xi0, _mm_setzero_si128());
            xi0 = _mm_min_epi16(xi0, _mm_set1_epi16(255));
            xi1 = _mm_unpackhi_epi16(xi0, _mm_setzero_si128());
            xi0 = _mm_unpacklo_epi16(xi0, _mm_setzero_si128());

            __m128d yd0 = _mm_unpacklo_pd(_mm_load_sd(expTab + tab_idx[0]), _mm_load_sd(expTab + tab_idx[1]));
            __m128d yd1 = _mm_unpacklo_pd(_mm_load_sd(expTab + tab_idx[2]), _mm_load_sd(expTab + tab_idx[3]));
            __m128d yd2 = _mm_unpacklo_pd(_mm_load_sd(expTab + tab_idx[4]), _mm_load_sd(expTab + tab_idx[5]));
            __m128d yd3 = _mm_unpacklo_pd(_mm_load_sd(expTab + tab_idx[6]), _mm_load_sd(expTab + tab_idx[7]));

            __m128 yf0 = _mm_movelh_ps(_mm_cvtpd_ps(yd0), _mm_cvtpd_ps(yd1));
            __m128 yf1 = _mm_movelh_ps(_mm_cvtpd_ps(yd2), _mm_cvtpd_ps(yd3));

            yf0 = _mm_mul_ps(yf0, _mm_castsi128_ps(_mm_slli_epi32(xi0, 23)));
            yf1 = _mm_mul_ps(yf1, _mm_castsi128_ps(_mm_slli_epi32(xi1, 23)));

            __m128 zf0 = _mm_add_ps(xf0, mA1);
            __m128 zf1 = _mm_add_ps(xf1, mA1);

            zf0 = _mm_add_ps(_mm_mul_ps(zf0, xf0), mA2);
            zf1 = _mm_add_ps(_mm_mul_ps(zf1, xf1), mA2);

            zf0 = _mm_add_ps(_mm_mul_ps(zf0, xf0), mA3);
            zf1 = _mm_add_ps(_mm_mul_ps(zf1, xf1), mA3);

            zf0 = _mm_add_ps(_mm_mul_ps(zf0, xf0), mA4);
            zf1 = _mm_add_ps(_mm_mul_ps(zf1, xf1), mA4);

            zf0 = _mm_mul_ps(zf0, yf0);
            zf1 = _mm_mul_ps(zf1, yf1);

            if( y_aligned )
            {
                _mm_store_ps(y + i, zf0);
                _mm_store_ps(y + i + 4, zf1);
            }
            else
            {
                _mm_storeu_ps(y + i, zf0);
                _mm_storeu_ps(y + i + 4, zf1);
            }
        }
    }
    else
#endif
        for( ; i <= n - 4; i += 4 )
        {
            double x0 = x[i].f * exp_prescale;
            double x1 = x[i + 1].f * exp_prescale;
            double x2 = x[i + 2].f * exp_prescale;
            double x3 = x[i + 3].f * exp_prescale;
            int val0, val1, val2, val3, t;

            if( ((x[i].i >> 23) & 255) > 127 + 10 )
                x0 = x[i].i < 0 ? -exp_max_val : exp_max_val;

            if( ((x[i+1].i >> 23) & 255) > 127 + 10 )
                x1 = x[i+1].i < 0 ? -exp_max_val : exp_max_val;

            if( ((x[i+2].i >> 23) & 255) > 127 + 10 )
                x2 = x[i+2].i < 0 ? -exp_max_val : exp_max_val;

            if( ((x[i+3].i >> 23) & 255) > 127 + 10 )
                x3 = x[i+3].i < 0 ? -exp_max_val : exp_max_val;

            val0 = cvRound(x0);
            val1 = cvRound(x1);
            val2 = cvRound(x2);
            val3 = cvRound(x3);

            x0 = (x0 - val0)*exp_postscale;
            x1 = (x1 - val1)*exp_postscale;
            x2 = (x2 - val2)*exp_postscale;
            x3 = (x3 - val3)*exp_postscale;

            t = (val0 >> EXPTAB_SCALE) + 127;
            t = !(t & ~255) ? t : t < 0 ? 0 : 255;
            buf[0].i = t << 23;

            t = (val1 >> EXPTAB_SCALE) + 127;
            t = !(t & ~255) ? t : t < 0 ? 0 : 255;
            buf[1].i = t << 23;

            t = (val2 >> EXPTAB_SCALE) + 127;
            t = !(t & ~255) ? t : t < 0 ? 0 : 255;
            buf[2].i = t << 23;

            t = (val3 >> EXPTAB_SCALE) + 127;
            t = !(t & ~255) ? t : t < 0 ? 0 : 255;
            buf[3].i = t << 23;

            x0 = buf[0].f * expTab[val0 & EXPTAB_MASK] * EXPPOLY( x0 );
            x1 = buf[1].f * expTab[val1 & EXPTAB_MASK] * EXPPOLY( x1 );

            y[i] = (float)x0;
            y[i + 1] = (float)x1;

            x2 = buf[2].f * expTab[val2 & EXPTAB_MASK] * EXPPOLY( x2 );
            x3 = buf[3].f * expTab[val3 & EXPTAB_MASK] * EXPPOLY( x3 );

            y[i + 2] = (float)x2;
            y[i + 3] = (float)x3;
        }

    for( ; i < n; i++ )
    {
        double x0 = x[i].f * exp_prescale;
        int val0, t;

        if( ((x[i].i >> 23) & 255) > 127 + 10 )
            x0 = x[i].i < 0 ? -exp_max_val : exp_max_val;

        val0 = cvRound(x0);
        t = (val0 >> EXPTAB_SCALE) + 127;
        t = !(t & ~255) ? t : t < 0 ? 0 : 255;

        buf[0].i = t << 23;
        x0 = (x0 - val0)*exp_postscale;

        y[i] = (float)(buf[0].f * expTab[val0 & EXPTAB_MASK] * EXPPOLY(x0));
    }
}

void exp( const double *_x, double *y, int n )
{
    static const double
    A5 = .99999999999999999998285227504999 / EXPPOLY_32F_A0,
    A4 = .69314718055994546743029643825322 / EXPPOLY_32F_A0,
    A3 = .24022650695886477918181338054308 / EXPPOLY_32F_A0,
    A2 = .55504108793649567998466049042729e-1 / EXPPOLY_32F_A0,
    A1 = .96180973140732918010002372686186e-2 / EXPPOLY_32F_A0,
    A0 = .13369713757180123244806654839424e-2 / EXPPOLY_32F_A0;

#undef EXPPOLY
#define EXPPOLY(x)  (((((A0*(x) + A1)*(x) + A2)*(x) + A3)*(x) + A4)*(x) + A5)

    int i = 0;
    Cv64suf buf[4];
    const Cv64suf* x = (const Cv64suf*)_x;

#if CV_SSE2
    static const __m128d prescale2 = _mm_set1_pd(exp_prescale);
    static const __m128d postscale2 = _mm_set1_pd(exp_postscale);
    static const __m128d maxval2 = _mm_set1_pd(exp_max_val);
    static const __m128d minval2 = _mm_set1_pd(-exp_max_val);

    static const __m128d mA0 = _mm_set1_pd(A0);
    static const __m128d mA1 = _mm_set1_pd(A1);
    static const __m128d mA2 = _mm_set1_pd(A2);
    static const __m128d mA3 = _mm_set1_pd(A3);
    static const __m128d mA4 = _mm_set1_pd(A4);
    static const __m128d mA5 = _mm_set1_pd(A5);

    int CV_DECL_ALIGNED(16) tab_idx[4];

    for( ; i <= n - 4; i += 4 )
    {
        __m128d xf0 = _mm_loadu_pd(&x[i].f), xf1 = _mm_loadu_pd(&x[i+2].f);
        __m128i xi0, xi1;
        xf0 = _mm_min_pd(_mm_max_pd(xf0, minval2), maxval2);
        xf1 = _mm_min_pd(_mm_max_pd(xf1, minval2), maxval2);
        xf0 = _mm_mul_pd(xf0, prescale2);
        xf1 = _mm_mul_pd(xf1, prescale2);

        xi0 = _mm_cvtpd_epi32(xf0);
        xi1 = _mm_cvtpd_epi32(xf1);
        xf0 = _mm_mul_pd(_mm_sub_pd(xf0, _mm_cvtepi32_pd(xi0)), postscale2);
        xf1 = _mm_mul_pd(_mm_sub_pd(xf1, _mm_cvtepi32_pd(xi1)), postscale2);

        xi0 = _mm_unpacklo_epi64(xi0, xi1);
        _mm_store_si128((__m128i*)tab_idx, _mm_and_si128(xi0, _mm_set1_epi32(EXPTAB_MASK)));

        xi0 = _mm_add_epi32(_mm_srai_epi32(xi0, EXPTAB_SCALE), _mm_set1_epi32(1023));
        xi0 = _mm_packs_epi32(xi0, xi0);
        xi0 = _mm_max_epi16(xi0, _mm_setzero_si128());
        xi0 = _mm_min_epi16(xi0, _mm_set1_epi16(2047));
        xi0 = _mm_unpacklo_epi16(xi0, _mm_setzero_si128());
        xi1 = _mm_unpackhi_epi32(xi0, _mm_setzero_si128());
        xi0 = _mm_unpacklo_epi32(xi0, _mm_setzero_si128());

        __m128d yf0 = _mm_unpacklo_pd(_mm_load_sd(expTab + tab_idx[0]), _mm_load_sd(expTab + tab_idx[1]));
        __m128d yf1 = _mm_unpacklo_pd(_mm_load_sd(expTab + tab_idx[2]), _mm_load_sd(expTab + tab_idx[3]));
        yf0 = _mm_mul_pd(yf0, _mm_castsi128_pd(_mm_slli_epi64(xi0, 52)));
        yf1 = _mm_mul_pd(yf1, _mm_castsi128_pd(_mm_slli_epi64(xi1, 52)));

        __m128d zf0 = _mm_add_pd(_mm_mul_pd(mA0, xf0), mA1);
        __m128d zf1 = _mm_add_pd(_mm_mul_pd(mA0, xf1), mA1);

        zf0 = _mm_add_pd(_mm_mul_pd(zf0, xf0), mA2);
        zf1 = _mm_add_pd(_mm_mul_pd(zf1, xf1), mA2);

        zf0 = _mm_add_pd(_mm_mul_pd(zf0, xf0), mA3);
        zf1 = _mm_add_pd(_mm_mul_pd(zf1, xf1), mA3);

        zf0 = _mm_add_pd(_mm_mul_pd(zf0, xf0), mA4);
        zf1 = _mm_add_pd(_mm_mul_pd(zf1, xf1), mA4);

        zf0 = _mm_add_pd(_mm_mul_pd(zf0, xf0), mA5);
        zf1 = _mm_add_pd(_mm_mul_pd(zf1, xf1), mA5);

        zf0 = _mm_mul_pd(zf0, yf0);
        zf1 = _mm_mul_pd(zf1, yf1);

        _mm_storeu_pd(y + i, zf0);
        _mm_storeu_pd(y + i + 2, zf1);
    }
#endif
    for( ; i <= n - 4; i += 4 )
    {
        double x0 = x[i].f * exp_prescale;
        double x1 = x[i + 1].f * exp_prescale;
        double x2 = x[i + 2].f * exp_prescale;
        double x3 = x[i + 3].f * exp_prescale;

        double y0, y1, y2, y3;
        int val0, val1, val2, val3, t;

        t = (int)(x[i].i >> 52);
        if( (t & 2047) > 1023 + 10 )
            x0 = t < 0 ? -exp_max_val : exp_max_val;

        t = (int)(x[i+1].i >> 52);
        if( (t & 2047) > 1023 + 10 )
            x1 = t < 0 ? -exp_max_val : exp_max_val;

        t = (int)(x[i+2].i >> 52);
        if( (t & 2047) > 1023 + 10 )
            x2 = t < 0 ? -exp_max_val : exp_max_val;

        t = (int)(x[i+3].i >> 52);
        if( (t & 2047) > 1023 + 10 )
            x3 = t < 0 ? -exp_max_val : exp_max_val;

        val0 = cvRound(x0);
        val1 = cvRound(x1);
        val2 = cvRound(x2);
        val3 = cvRound(x3);

        x0 = (x0 - val0)*exp_postscale;
        x1 = (x1 - val1)*exp_postscale;
        x2 = (x2 - val2)*exp_postscale;
        x3 = (x3 - val3)*exp_postscale;

        t = (val0 >> EXPTAB_SCALE) + 1023;
        t = !(t & ~2047) ? t : t < 0 ? 0 : 2047;
        buf[0].i = (int64)t << 52;

        t = (val1 >> EXPTAB_SCALE) + 1023;
        t = !(t & ~2047) ? t : t < 0 ? 0 : 2047;
        buf[1].i = (int64)t << 52;

        t = (val2 >> EXPTAB_SCALE) + 1023;
        t = !(t & ~2047) ? t : t < 0 ? 0 : 2047;
        buf[2].i = (int64)t << 52;

        t = (val3 >> EXPTAB_SCALE) + 1023;
        t = !(t & ~2047) ? t : t < 0 ? 0 : 2047;
        buf[3].i = (int64)t << 52;

        y0 = buf[0].f * expTab[val0 & EXPTAB_MASK] * EXPPOLY( x0 );
        y1 = buf[1].f * expTab[val1 & EXPTAB_MASK] * EXPPOLY( x1 );

        y[i] = y0;
        y[i + 1] = y1;

        y2 = buf[2].f * expTab[val2 & EXPTAB_MASK] * EXPPOLY( x2 );
        y3 = buf[3].f * expTab[val3 & EXPTAB_MASK] * EXPPOLY( x3 );

        y[i + 2] = y2;
        y[i + 3] = y3;
    }

    for( ; i < n; i++ )
    {
        double x0 = x[i].f * exp_prescale;
        int val0, t;

        t = (int)(x[i].i >> 52);
        if( (t & 2047) > 1023 + 10 )
            x0 = t < 0 ? -exp_max_val : exp_max_val;

        val0 = cvRound(x0);
        t = (val0 >> EXPTAB_SCALE) + 1023;
        t = !(t & ~2047) ? t : t < 0 ? 0 : 2047;

        buf[0].i = (int64)t << 52;
        x0 = (x0 - val0)*exp_postscale;

        y[i] = buf[0].f * expTab[val0 & EXPTAB_MASK] * EXPPOLY( x0 );
    }
}

#undef EXPTAB_SCALE
#undef EXPTAB_MASK
#undef EXPPOLY_32F_A0

/////////////////////////////////////////// LOG ///////////////////////////////////////

#define LOGTAB_SCALE    8
#define LOGTAB_MASK         ((1 << LOGTAB_SCALE) - 1)
#define LOGTAB_MASK2        ((1 << (20 - LOGTAB_SCALE)) - 1)
#define LOGTAB_MASK2_32F    ((1 << (23 - LOGTAB_SCALE)) - 1)

static const double CV_DECL_ALIGNED(16) icvLogTab[] = {
    0.0000000000000000000000000000000000000000,    1.000000000000000000000000000000000000000,
    .00389864041565732288852075271279318258166,    .9961089494163424124513618677042801556420,
    .00778214044205494809292034119607706088573,    .9922480620155038759689922480620155038760,
    .01165061721997527263705585198749759001657,    .9884169884169884169884169884169884169884,
    .01550418653596525274396267235488267033361,    .9846153846153846153846153846153846153846,
    .01934296284313093139406447562578250654042,    .9808429118773946360153256704980842911877,
    .02316705928153437593630670221500622574241,    .9770992366412213740458015267175572519084,
    .02697658769820207233514075539915211265906,    .9733840304182509505703422053231939163498,
    .03077165866675368732785500469617545604706,    .9696969696969696969696969696969696969697,
    .03455238150665972812758397481047722976656,    .9660377358490566037735849056603773584906,
    .03831886430213659461285757856785494368522,    .9624060150375939849624060150375939849624,
    .04207121392068705056921373852674150839447,    .9588014981273408239700374531835205992509,
    .04580953603129420126371940114040626212953,    .9552238805970149253731343283582089552239,
    .04953393512227662748292900118940451648088,    .9516728624535315985130111524163568773234,
    .05324451451881227759255210685296333394944,    .9481481481481481481481481481481481481481,
    .05694137640013842427411105973078520037234,    .9446494464944649446494464944649446494465,
    .06062462181643483993820353816772694699466,    .9411764705882352941176470588235294117647,
    .06429435070539725460836422143984236754475,    .9377289377289377289377289377289377289377,
    .06795066190850773679699159401934593915938,    .9343065693430656934306569343065693430657,
    .07159365318700880442825962290953611955044,    .9309090909090909090909090909090909090909,
    .07522342123758751775142172846244648098944,    .9275362318840579710144927536231884057971,
    .07884006170777602129362549021607264876369,    .9241877256317689530685920577617328519856,
    .08244366921107458556772229485432035289706,    .9208633093525179856115107913669064748201,
    .08603433734180314373940490213499288074675,    .9175627240143369175627240143369175627240,
    .08961215868968712416897659522874164395031,    .9142857142857142857142857142857142857143,
    .09317722485418328259854092721070628613231,    .9110320284697508896797153024911032028470,
    .09672962645855109897752299730200320482256,    .9078014184397163120567375886524822695035,
    .10026945316367513738597949668474029749630,    .9045936395759717314487632508833922261484,
    .10379679368164355934833764649738441221420,    .9014084507042253521126760563380281690141,
    .10731173578908805021914218968959175981580,    .8982456140350877192982456140350877192982,
    .11081436634029011301105782649756292812530,    .8951048951048951048951048951048951048951,
    .11430477128005862852422325204315711744130,    .8919860627177700348432055749128919860627,
    .11778303565638344185817487641543266363440,    .8888888888888888888888888888888888888889,
    .12124924363286967987640707633545389398930,    .8858131487889273356401384083044982698962,
    .12470347850095722663787967121606925502420,    .8827586206896551724137931034482758620690,
    .12814582269193003360996385708858724683530,    .8797250859106529209621993127147766323024,
    .13157635778871926146571524895989568904040,    .8767123287671232876712328767123287671233,
    .13499516453750481925766280255629681050780,    .8737201365187713310580204778156996587031,
    .13840232285911913123754857224412262439730,    .8707482993197278911564625850340136054422,
    .14179791186025733629172407290752744302150,    .8677966101694915254237288135593220338983,
    .14518200984449788903951628071808954700830,    .8648648648648648648648648648648648648649,
    .14855469432313711530824207329715136438610,    .8619528619528619528619528619528619528620,
    .15191604202584196858794030049466527998450,    .8590604026845637583892617449664429530201,
    .15526612891112392955683674244937719777230,    .8561872909698996655518394648829431438127,
    .15860503017663857283636730244325008243330,    .8533333333333333333333333333333333333333,
    .16193282026931324346641360989451641216880,    .8504983388704318936877076411960132890365,
    .16524957289530714521497145597095368430010,    .8476821192052980132450331125827814569536,
    .16855536102980664403538924034364754334090,    .8448844884488448844884488448844884488449,
    .17185025692665920060697715143760433420540,    .8421052631578947368421052631578947368421,
    .17513433212784912385018287750426679849630,    .8393442622950819672131147540983606557377,
    .17840765747281828179637841458315961062910,    .8366013071895424836601307189542483660131,
    .18167030310763465639212199675966985523700,    .8338762214983713355048859934853420195440,
    .18492233849401198964024217730184318497780,    .8311688311688311688311688311688311688312,
    .18816383241818296356839823602058459073300,    .8284789644012944983818770226537216828479,
    .19139485299962943898322009772527962923050,    .8258064516129032258064516129032258064516,
    .19461546769967164038916962454095482826240,    .8231511254019292604501607717041800643087,
    .19782574332991986754137769821682013571260,    .8205128205128205128205128205128205128205,
    .20102574606059073203390141770796617493040,    .8178913738019169329073482428115015974441,
    .20421554142869088876999228432396193966280,    .8152866242038216560509554140127388535032,
    .20739519434607056602715147164417430758480,    .8126984126984126984126984126984126984127,
    .21056476910734961416338251183333341032260,    .8101265822784810126582278481012658227848,
    .21372432939771812687723695489694364368910,    .8075709779179810725552050473186119873817,
    .21687393830061435506806333251006435602900,    .8050314465408805031446540880503144654088,
    .22001365830528207823135744547471404075630,    .8025078369905956112852664576802507836991,
    .22314355131420973710199007200571941211830,    .8000000000000000000000000000000000000000,
    .22626367865045338145790765338460914790630,    .7975077881619937694704049844236760124611,
    .22937410106484582006380890106811420992010,    .7950310559006211180124223602484472049689,
    .23247487874309405442296849741978803649550,    .7925696594427244582043343653250773993808,
    .23556607131276688371634975283086532726890,    .7901234567901234567901234567901234567901,
    .23864773785017498464178231643018079921600,    .7876923076923076923076923076923076923077,
    .24171993688714515924331749374687206000090,    .7852760736196319018404907975460122699387,
    .24478272641769091566565919038112042471760,    .7828746177370030581039755351681957186544,
    .24783616390458124145723672882013488560910,    .7804878048780487804878048780487804878049,
    .25088030628580937353433455427875742316250,    .7781155015197568389057750759878419452888,
    .25391520998096339667426946107298135757450,    .7757575757575757575757575757575757575758,
    .25694093089750041913887912414793390780680,    .7734138972809667673716012084592145015106,
    .25995752443692604627401010475296061486000,    .7710843373493975903614457831325301204819,
    .26296504550088134477547896494797896593800,    .7687687687687687687687687687687687687688,
    .26596354849713793599974565040611196309330,    .7664670658682634730538922155688622754491,
    .26895308734550393836570947314612567424780,    .7641791044776119402985074626865671641791,
    .27193371548364175804834985683555714786050,    .7619047619047619047619047619047619047619,
    .27490548587279922676529508862586226314300,    .7596439169139465875370919881305637982196,
    .27786845100345625159121709657483734190480,    .7573964497041420118343195266272189349112,
    .28082266290088775395616949026589281857030,    .7551622418879056047197640117994100294985,
    .28376817313064456316240580235898960381750,    .7529411764705882352941176470588235294118,
    .28670503280395426282112225635501090437180,    .7507331378299120234604105571847507331378,
    .28963329258304265634293983566749375313530,    .7485380116959064327485380116959064327485,
    .29255300268637740579436012922087684273730,    .7463556851311953352769679300291545189504,
    .29546421289383584252163927885703742504130,    .7441860465116279069767441860465116279070,
    .29836697255179722709783618483925238251680,    .7420289855072463768115942028985507246377,
    .30126133057816173455023545102449133992200,    .7398843930635838150289017341040462427746,
    .30414733546729666446850615102448500692850,    .7377521613832853025936599423631123919308,
    .30702503529491181888388950937951449304830,    .7356321839080459770114942528735632183908,
    .30989447772286465854207904158101882785550,    .7335243553008595988538681948424068767908,
    .31275571000389684739317885942000430077330,    .7314285714285714285714285714285714285714,
    .31560877898630329552176476681779604405180,    .7293447293447293447293447293447293447293,
    .31845373111853458869546784626436419785030,    .7272727272727272727272727272727272727273,
    .32129061245373424782201254856772720813750,    .7252124645892351274787535410764872521246,
    .32411946865421192853773391107097268104550,    .7231638418079096045197740112994350282486,
    .32694034499585328257253991068864706903700,    .7211267605633802816901408450704225352113,
    .32975328637246797969240219572384376078850,    .7191011235955056179775280898876404494382,
    .33255833730007655635318997155991382896900,    .7170868347338935574229691876750700280112,
    .33535554192113781191153520921943709254280,    .7150837988826815642458100558659217877095,
    .33814494400871636381467055798566434532400,    .7130919220055710306406685236768802228412,
    .34092658697059319283795275623560883104800,    .7111111111111111111111111111111111111111,
    .34370051385331840121395430287520866841080,    .7091412742382271468144044321329639889197,
    .34646676734620857063262633346312213689100,    .7071823204419889502762430939226519337017,
    .34922538978528827602332285096053965389730,    .7052341597796143250688705234159779614325,
    .35197642315717814209818925519357435405250,    .7032967032967032967032967032967032967033,
    .35471990910292899856770532096561510115850,    .7013698630136986301369863013698630136986,
    .35745588892180374385176833129662554711100,    .6994535519125683060109289617486338797814,
    .36018440357500774995358483465679455548530,    .6975476839237057220708446866485013623978,
    .36290549368936841911903457003063522279280,    .6956521739130434782608695652173913043478,
    .36561919956096466943762379742111079394830,    .6937669376693766937669376693766937669377,
    .36832556115870762614150635272380895912650,    .6918918918918918918918918918918918918919,
    .37102461812787262962487488948681857436900,    .6900269541778975741239892183288409703504,
    .37371640979358405898480555151763837784530,    .6881720430107526881720430107526881720430,
    .37640097516425302659470730759494472295050,    .6863270777479892761394101876675603217158,
    .37907835293496944251145919224654790014030,    .6844919786096256684491978609625668449198,
    .38174858149084833769393299007788300514230,    .6826666666666666666666666666666666666667,
    .38441169891033200034513583887019194662580,    .6808510638297872340425531914893617021277,
    .38706774296844825844488013899535872042180,    .6790450928381962864721485411140583554377,
    .38971675114002518602873692543653305619950,    .6772486772486772486772486772486772486772,
    .39235876060286384303665840889152605086580,    .6754617414248021108179419525065963060686,
    .39499380824086893770896722344332374632350,    .6736842105263157894736842105263157894737,
    .39762193064713846624158577469643205404280,    .6719160104986876640419947506561679790026,
    .40024316412701266276741307592601515352730,    .6701570680628272251308900523560209424084,
    .40285754470108348090917615991202183067800,    .6684073107049608355091383812010443864230,
    .40546510810816432934799991016916465014230,    .6666666666666666666666666666666666666667,
    .40806588980822172674223224930756259709600,    .6649350649350649350649350649350649350649,
    .41065992498526837639616360320360399782650,    .6632124352331606217616580310880829015544,
    .41324724855021932601317757871584035456180,    .6614987080103359173126614987080103359173,
    .41582789514371093497757669865677598863850,    .6597938144329896907216494845360824742268,
    .41840189913888381489925905043492093682300,    .6580976863753213367609254498714652956298,
    .42096929464412963239894338585145305842150,    .6564102564102564102564102564102564102564,
    .42353011550580327293502591601281892508280,    .6547314578005115089514066496163682864450,
    .42608439531090003260516141381231136620050,    .6530612244897959183673469387755102040816,
    .42863216738969872610098832410585600882780,    .6513994910941475826972010178117048346056,
    .43117346481837132143866142541810404509300,    .6497461928934010152284263959390862944162,
    .43370832042155937902094819946796633303180,    .6481012658227848101265822784810126582278,
    .43623676677491801667585491486534010618930,    .6464646464646464646464646464646464646465,
    .43875883620762790027214350629947148263450,    .6448362720403022670025188916876574307305,
    .44127456080487520440058801796112675219780,    .6432160804020100502512562814070351758794,
    .44378397241030093089975139264424797147500,    .6416040100250626566416040100250626566416,
    .44628710262841947420398014401143882423650,    .6400000000000000000000000000000000000000,
    .44878398282700665555822183705458883196130,    .6384039900249376558603491271820448877805,
    .45127464413945855836729492693848442286250,    .6368159203980099502487562189054726368159,
    .45375911746712049854579618113348260521900,    .6352357320099255583126550868486352357320,
    .45623743348158757315857769754074979573500,    .6336633663366336633663366336633663366337,
    .45870962262697662081833982483658473938700,    .6320987654320987654320987654320987654321,
    .46117571512217014895185229761409573256980,    .6305418719211822660098522167487684729064,
    .46363574096303250549055974261136725544930,    .6289926289926289926289926289926289926290,
    .46608972992459918316399125615134835243230,    .6274509803921568627450980392156862745098,
    .46853771156323925639597405279346276074650,    .6259168704156479217603911980440097799511,
    .47097971521879100631480241645476780831830,    .6243902439024390243902439024390243902439,
    .47341577001667212165614273544633761048330,    .6228710462287104622871046228710462287105,
    .47584590486996386493601107758877333253630,    .6213592233009708737864077669902912621359,
    .47827014848147025860569669930555392056700,    .6198547215496368038740920096852300242131,
    .48068852934575190261057286988943815231330,    .6183574879227053140096618357487922705314,
    .48310107575113581113157579238759353756900,    .6168674698795180722891566265060240963855,
    .48550781578170076890899053978500887751580,    .6153846153846153846153846153846153846154,
    .48790877731923892879351001283794175833480,    .6139088729016786570743405275779376498801,
    .49030398804519381705802061333088204264650,    .6124401913875598086124401913875598086124,
    .49269347544257524607047571407747454941280,    .6109785202863961813842482100238663484487,
    .49507726679785146739476431321236304938800,    .6095238095238095238095238095238095238095,
    .49745538920281889838648226032091770321130,    .6080760095011876484560570071258907363420,
    .49982786955644931126130359189119189977650,    .6066350710900473933649289099526066350711,
    .50219473456671548383667413872899487614650,    .6052009456264775413711583924349881796690,
    .50455601075239520092452494282042607665050,    .6037735849056603773584905660377358490566,
    .50691172444485432801997148999362252652650,    .6023529411764705882352941176470588235294,
    .50926190178980790257412536448100581765150,    .6009389671361502347417840375586854460094,
    .51160656874906207391973111953120678663250,    .5995316159250585480093676814988290398126,
    .51394575110223428282552049495279788970950,    .5981308411214953271028037383177570093458,
    .51627947444845445623684554448118433356300,    .5967365967365967365967365967365967365967,
    .51860776420804555186805373523384332656850,    .5953488372093023255813953488372093023256,
    .52093064562418522900344441950437612831600,    .5939675174013921113689095127610208816705,
    .52324814376454775732838697877014055848100,    .5925925925925925925925925925925925925926,
    .52556028352292727401362526507000438869000,    .5912240184757505773672055427251732101617,
    .52786708962084227803046587723656557500350,    .5898617511520737327188940092165898617512,
    .53016858660912158374145519701414741575700,    .5885057471264367816091954022988505747126,
    .53246479886947173376654518506256863474850,    .5871559633027522935779816513761467889908,
    .53475575061602764748158733709715306758900,    .5858123569794050343249427917620137299771,
    .53704146589688361856929077475797384977350,    .5844748858447488584474885844748858447489,
    .53932196859560876944783558428753167390800,    .5831435079726651480637813211845102505695,
    .54159728243274429804188230264117009937750,    .5818181818181818181818181818181818181818,
    .54386743096728351609669971367111429572100,    .5804988662131519274376417233560090702948,
    .54613243759813556721383065450936555862450,    .5791855203619909502262443438914027149321,
    .54839232556557315767520321969641372561450,    .5778781038374717832957110609480812641084,
    .55064711795266219063194057525834068655950,    .5765765765765765765765765765765765765766,
    .55289683768667763352766542084282264113450,    .5752808988764044943820224719101123595506,
    .55514150754050151093110798683483153581600,    .5739910313901345291479820627802690582960,
    .55738115013400635344709144192165695130850,    .5727069351230425055928411633109619686801,
    .55961578793542265941596269840374588966350,    .5714285714285714285714285714285714285714,
    .56184544326269181269140062795486301183700,    .5701559020044543429844097995545657015590,
    .56407013828480290218436721261241473257550,    .5688888888888888888888888888888888888889,
    .56628989502311577464155334382667206227800,    .5676274944567627494456762749445676274945,
    .56850473535266865532378233183408156037350,    .5663716814159292035398230088495575221239,
    .57071468100347144680739575051120482385150,    .5651214128035320088300220750551876379691,
    .57291975356178548306473885531886480748650,    .5638766519823788546255506607929515418502,
    .57511997447138785144460371157038025558000,    .5626373626373626373626373626373626373626,
    .57731536503482350219940144597785547375700,    .5614035087719298245614035087719298245614,
    .57950594641464214795689713355386629700650,    .5601750547045951859956236323851203501094,
    .58169173963462239562716149521293118596100,    .5589519650655021834061135371179039301310,
    .58387276558098266665552955601015128195300,    .5577342047930283224400871459694989106754,
    .58604904500357812846544902640744112432000,    .5565217391304347826086956521739130434783,
    .58822059851708596855957011939608491957200,    .5553145336225596529284164859002169197397,
    .59038744660217634674381770309992134571100,    .5541125541125541125541125541125541125541,
    .59254960960667157898740242671919986605650,    .5529157667386609071274298056155507559395,
    .59470710774669277576265358220553025603300,    .5517241379310344827586206896551724137931,
    .59685996110779382384237123915227130055450,    .5505376344086021505376344086021505376344,
    .59900818964608337768851242799428291618800,    .5493562231759656652360515021459227467811,
    .60115181318933474940990890900138765573500,    .5481798715203426124197002141327623126338,
    .60329085143808425240052883964381180703650,    .5470085470085470085470085470085470085470,
    .60542532396671688843525771517306566238400,    .5458422174840085287846481876332622601279,
    .60755525022454170969155029524699784815300,    .5446808510638297872340425531914893617021,
    .60968064953685519036241657886421307921400,    .5435244161358811040339702760084925690021,
    .61180154110599282990534675263916142284850,    .5423728813559322033898305084745762711864,
    .61391794401237043121710712512140162289150,    .5412262156448202959830866807610993657505,
    .61602987721551394351138242200249806046500,    .5400843881856540084388185654008438818565,
    .61813735955507864705538167982012964785100,    .5389473684210526315789473684210526315789,
    .62024040975185745772080281312810257077200,    .5378151260504201680672268907563025210084,
    .62233904640877868441606324267922900617100,    .5366876310272536687631027253668763102725,
    .62443328801189346144440150965237990021700,    .5355648535564853556485355648535564853556,
    .62652315293135274476554741340805776417250,    .5344467640918580375782881002087682672234,
    .62860865942237409420556559780379757285100,    .5333333333333333333333333333333333333333,
    .63068982562619868570408243613201193511500,    .5322245322245322245322245322245322245322,
    .63276666957103777644277897707070223987100,    .5311203319502074688796680497925311203320,
    .63483920917301017716738442686619237065300,    .5300207039337474120082815734989648033126,
    .63690746223706917739093569252872839570050,    .5289256198347107438016528925619834710744,
    .63897144645792069983514238629140891134750,    .5278350515463917525773195876288659793814,
    .64103117942093124081992527862894348800200,    .5267489711934156378600823045267489711934,
    .64308667860302726193566513757104985415950,    .5256673511293634496919917864476386036961,
    .64513796137358470073053240412264131009600,    .5245901639344262295081967213114754098361,
    .64718504499530948859131740391603671014300,    .5235173824130879345603271983640081799591,
    .64922794662510974195157587018911726772800,    .5224489795918367346938775510204081632653,
    .65126668331495807251485530287027359008800,    .5213849287169042769857433808553971486762,
    .65330127201274557080523663898929953575150,    .5203252032520325203252032520325203252033,
    .65533172956312757406749369692988693714150,    .5192697768762677484787018255578093306288,
    .65735807270835999727154330685152672231200,    .5182186234817813765182186234817813765182,
    .65938031808912778153342060249997302889800,    .5171717171717171717171717171717171717172,
    .66139848224536490484126716182800009846700,    .5161290322580645161290322580645161290323,
    .66341258161706617713093692145776003599150,    .5150905432595573440643863179074446680080,
    .66542263254509037562201001492212526500250,    .5140562248995983935742971887550200803213,
    .66742865127195616370414654738851822912700,    .5130260521042084168336673346693386773547,
    .66943065394262923906154583164607174694550,    .5120000000000000000000000000000000000000,
    .67142865660530226534774556057527661323550,    .5109780439121756487025948103792415169661,
    .67342267521216669923234121597488410770900,    .5099601593625498007968127490039840637450,
    .67541272562017662384192817626171745359900,    .5089463220675944333996023856858846918489,
    .67739882359180603188519853574689477682100,    .5079365079365079365079365079365079365079,
    .67938098479579733801614338517538271844400,    .5069306930693069306930693069306930693069,
    .68135922480790300781450241629499942064300,    .5059288537549407114624505928853754940711,
    .68333355911162063645036823800182901322850,    .5049309664694280078895463510848126232742,
    .68530400309891936760919861626462079584600,    .5039370078740157480314960629921259842520,
    .68727057207096020619019327568821609020250,    .5029469548133595284872298624754420432220,
    .68923328123880889251040571252815425395950,    .5019607843137254901960784313725490196078,
    .69314718055994530941723212145818, 5.0e-01,
};



#define LOGTAB_TRANSLATE(x,h) (((x) - 1.)*icvLogTab[(h)+1])
static const double ln_2 = 0.69314718055994530941723212145818;

void log( const float *_x, float *y, int n )
{
    static const float shift[] = { 0, -1.f/512 };
    static const float
    A0 = 0.3333333333333333333333333f,
    A1 = -0.5f,
    A2 = 1.f;

#undef LOGPOLY
#define LOGPOLY(x) (((A0*(x) + A1)*(x) + A2)*(x))

    int i = 0;
    Cv32suf buf[4];
    const int* x = (const int*)_x;

#if CV_SSE2
    static const __m128d ln2_2 = _mm_set1_pd(ln_2);
    static const __m128 _1_4 = _mm_set1_ps(1.f);
    static const __m128 shift4 = _mm_set1_ps(-1.f/512);

    static const __m128 mA0 = _mm_set1_ps(A0);
    static const __m128 mA1 = _mm_set1_ps(A1);
    static const __m128 mA2 = _mm_set1_ps(A2);

    int CV_DECL_ALIGNED(16) idx[4];

    for( ; i <= n - 4; i += 4 )
    {
        __m128i h0 = _mm_loadu_si128((const __m128i*)(x + i));
        __m128i yi0 = _mm_sub_epi32(_mm_and_si128(_mm_srli_epi32(h0, 23), _mm_set1_epi32(255)), _mm_set1_epi32(127));
        __m128d yd0 = _mm_mul_pd(_mm_cvtepi32_pd(yi0), ln2_2);
        __m128d yd1 = _mm_mul_pd(_mm_cvtepi32_pd(_mm_unpackhi_epi64(yi0,yi0)), ln2_2);

        __m128i xi0 = _mm_or_si128(_mm_and_si128(h0, _mm_set1_epi32(LOGTAB_MASK2_32F)), _mm_set1_epi32(127 << 23));

        h0 = _mm_and_si128(_mm_srli_epi32(h0, 23 - LOGTAB_SCALE - 1), _mm_set1_epi32(LOGTAB_MASK*2));
        _mm_store_si128((__m128i*)idx, h0);
        h0 = _mm_cmpeq_epi32(h0, _mm_set1_epi32(510));

        __m128d t0, t1, t2, t3, t4;
        t0 = _mm_load_pd(icvLogTab + idx[0]);
        t2 = _mm_load_pd(icvLogTab + idx[1]);
        t1 = _mm_unpackhi_pd(t0, t2);
        t0 = _mm_unpacklo_pd(t0, t2);
        t2 = _mm_load_pd(icvLogTab + idx[2]);
        t4 = _mm_load_pd(icvLogTab + idx[3]);
        t3 = _mm_unpackhi_pd(t2, t4);
        t2 = _mm_unpacklo_pd(t2, t4);

        yd0 = _mm_add_pd(yd0, t0);
        yd1 = _mm_add_pd(yd1, t2);

        __m128 yf0 = _mm_movelh_ps(_mm_cvtpd_ps(yd0), _mm_cvtpd_ps(yd1));

        __m128 xf0 = _mm_sub_ps(_mm_castsi128_ps(xi0), _1_4);
        xf0 = _mm_mul_ps(xf0, _mm_movelh_ps(_mm_cvtpd_ps(t1), _mm_cvtpd_ps(t3)));
        xf0 = _mm_add_ps(xf0, _mm_and_ps(_mm_castsi128_ps(h0), shift4));

        __m128 zf0 = _mm_mul_ps(xf0, mA0);
        zf0 = _mm_mul_ps(_mm_add_ps(zf0, mA1), xf0);
        zf0 = _mm_mul_ps(_mm_add_ps(zf0, mA2), xf0);
        yf0 = _mm_add_ps(yf0, zf0);

        _mm_storeu_ps(y + i, yf0);
    }
#endif
    for( ; i <= n - 4; i += 4 )
    {
        double x0, x1, x2, x3;
        double y0, y1, y2, y3;
        int h0, h1, h2, h3;

        h0 = x[i];
        h1 = x[i+1];
        buf[0].i = (h0 & LOGTAB_MASK2_32F) | (127 << 23);
        buf[1].i = (h1 & LOGTAB_MASK2_32F) | (127 << 23);

        y0 = (((h0 >> 23) & 0xff) - 127) * ln_2;
        y1 = (((h1 >> 23) & 0xff) - 127) * ln_2;

        h0 = (h0 >> (23 - LOGTAB_SCALE - 1)) & LOGTAB_MASK * 2;
        h1 = (h1 >> (23 - LOGTAB_SCALE - 1)) & LOGTAB_MASK * 2;

        y0 += icvLogTab[h0];
        y1 += icvLogTab[h1];

        h2 = x[i+2];
        h3 = x[i+3];

        x0 = LOGTAB_TRANSLATE( buf[0].f, h0 );
        x1 = LOGTAB_TRANSLATE( buf[1].f, h1 );

        buf[2].i = (h2 & LOGTAB_MASK2_32F) | (127 << 23);
        buf[3].i = (h3 & LOGTAB_MASK2_32F) | (127 << 23);

        y2 = (((h2 >> 23) & 0xff) - 127) * ln_2;
        y3 = (((h3 >> 23) & 0xff) - 127) * ln_2;

        h2 = (h2 >> (23 - LOGTAB_SCALE - 1)) & LOGTAB_MASK * 2;
        h3 = (h3 >> (23 - LOGTAB_SCALE - 1)) & LOGTAB_MASK * 2;

        y2 += icvLogTab[h2];
        y3 += icvLogTab[h3];

        x2 = LOGTAB_TRANSLATE( buf[2].f, h2 );
        x3 = LOGTAB_TRANSLATE( buf[3].f, h3 );

        x0 += shift[h0 == 510];
        x1 += shift[h1 == 510];
        y0 += LOGPOLY( x0 );
        y1 += LOGPOLY( x1 );

        y[i] = (float) y0;
        y[i + 1] = (float) y1;

        x2 += shift[h2 == 510];
        x3 += shift[h3 == 510];
        y2 += LOGPOLY( x2 );
        y3 += LOGPOLY( x3 );

        y[i + 2] = (float) y2;
        y[i + 3] = (float) y3;
    }

    for( ; i < n; i++ )
    {
        int h0 = x[i];
        double y0;
        float x0;

        y0 = (((h0 >> 23) & 0xff) - 127) * ln_2;

        buf[0].i = (h0 & LOGTAB_MASK2_32F) | (127 << 23);
        h0 = (h0 >> (23 - LOGTAB_SCALE - 1)) & LOGTAB_MASK * 2;

        y0 += icvLogTab[h0];
        x0 = (float)LOGTAB_TRANSLATE( buf[0].f, h0 );
        x0 += shift[h0 == 510];
        y0 += LOGPOLY( x0 );

        y[i] = (float)y0;
    }
}

void log( const double *x, double *y, int n )
{
    static const double shift[] = { 0, -1./512 };
    static const double
    A7 = 1.0,
    A6 = -0.5,
    A5 = 0.333333333333333314829616256247390992939472198486328125,
    A4 = -0.25,
    A3 = 0.2,
    A2 = -0.1666666666666666574148081281236954964697360992431640625,
    A1 = 0.1428571428571428769682682968777953647077083587646484375,
    A0 = -0.125;

#undef LOGPOLY
#define LOGPOLY(x,k) ((x)+=shift[k], xq = (x)*(x),\
(((A0*xq + A2)*xq + A4)*xq + A6)*xq + \
(((A1*xq + A3)*xq + A5)*xq + A7)*(x))

    int i = 0;
    DBLINT buf[4];
    DBLINT *X = (DBLINT *) x;

#if CV_SSE2
    static const __m128d ln2_2 = _mm_set1_pd(ln_2);
    static const __m128d _1_2 = _mm_set1_pd(1.);
    static const __m128d shift2 = _mm_set1_pd(-1./512);

    static const __m128i log_and_mask2 = _mm_set_epi32(LOGTAB_MASK2, 0xffffffff, LOGTAB_MASK2, 0xffffffff);
    static const __m128i log_or_mask2 = _mm_set_epi32(1023 << 20, 0, 1023 << 20, 0);

    static const __m128d mA0 = _mm_set1_pd(A0);
    static const __m128d mA1 = _mm_set1_pd(A1);
    static const __m128d mA2 = _mm_set1_pd(A2);
    static const __m128d mA3 = _mm_set1_pd(A3);
    static const __m128d mA4 = _mm_set1_pd(A4);
    static const __m128d mA5 = _mm_set1_pd(A5);
    static const __m128d mA6 = _mm_set1_pd(A6);
    static const __m128d mA7 = _mm_set1_pd(A7);

    int CV_DECL_ALIGNED(16) idx[4];

    for( ; i <= n - 4; i += 4 )
    {
        __m128i h0 = _mm_loadu_si128((const __m128i*)(x + i));
        __m128i h1 = _mm_loadu_si128((const __m128i*)(x + i + 2));

        __m128d xd0 = _mm_castsi128_pd(_mm_or_si128(_mm_and_si128(h0, log_and_mask2), log_or_mask2));
        __m128d xd1 = _mm_castsi128_pd(_mm_or_si128(_mm_and_si128(h1, log_and_mask2), log_or_mask2));

        h0 = _mm_unpackhi_epi32(_mm_unpacklo_epi32(h0, h1), _mm_unpackhi_epi32(h0, h1));

        __m128i yi0 = _mm_sub_epi32(_mm_and_si128(_mm_srli_epi32(h0, 20),
                                                  _mm_set1_epi32(2047)), _mm_set1_epi32(1023));
        __m128d yd0 = _mm_mul_pd(_mm_cvtepi32_pd(yi0), ln2_2);
        __m128d yd1 = _mm_mul_pd(_mm_cvtepi32_pd(_mm_unpackhi_epi64(yi0, yi0)), ln2_2);

        h0 = _mm_and_si128(_mm_srli_epi32(h0, 20 - LOGTAB_SCALE - 1), _mm_set1_epi32(LOGTAB_MASK * 2));
        _mm_store_si128((__m128i*)idx, h0);
        h0 = _mm_cmpeq_epi32(h0, _mm_set1_epi32(510));

        __m128d t0, t1, t2, t3, t4;
        t0 = _mm_load_pd(icvLogTab + idx[0]);
        t2 = _mm_load_pd(icvLogTab + idx[1]);
        t1 = _mm_unpackhi_pd(t0, t2);
        t0 = _mm_unpacklo_pd(t0, t2);
        t2 = _mm_load_pd(icvLogTab + idx[2]);
        t4 = _mm_load_pd(icvLogTab + idx[3]);
        t3 = _mm_unpackhi_pd(t2, t4);
        t2 = _mm_unpacklo_pd(t2, t4);

        yd0 = _mm_add_pd(yd0, t0);
        yd1 = _mm_add_pd(yd1, t2);

        xd0 = _mm_mul_pd(_mm_sub_pd(xd0, _1_2), t1);
        xd1 = _mm_mul_pd(_mm_sub_pd(xd1, _1_2), t3);

        xd0 = _mm_add_pd(xd0, _mm_and_pd(_mm_castsi128_pd(_mm_unpacklo_epi32(h0, h0)), shift2));
        xd1 = _mm_add_pd(xd1, _mm_and_pd(_mm_castsi128_pd(_mm_unpackhi_epi32(h0, h0)), shift2));

        __m128d zd0 = _mm_mul_pd(xd0, mA0);
        __m128d zd1 = _mm_mul_pd(xd1, mA0);
        zd0 = _mm_mul_pd(_mm_add_pd(zd0, mA1), xd0);
        zd1 = _mm_mul_pd(_mm_add_pd(zd1, mA1), xd1);
        zd0 = _mm_mul_pd(_mm_add_pd(zd0, mA2), xd0);
        zd1 = _mm_mul_pd(_mm_add_pd(zd1, mA2), xd1);
        zd0 = _mm_mul_pd(_mm_add_pd(zd0, mA3), xd0);
        zd1 = _mm_mul_pd(_mm_add_pd(zd1, mA3), xd1);
        zd0 = _mm_mul_pd(_mm_add_pd(zd0, mA4), xd0);
        zd1 = _mm_mul_pd(_mm_add_pd(zd1, mA4), xd1);
        zd0 = _mm_mul_pd(_mm_add_pd(zd0, mA5), xd0);
        zd1 = _mm_mul_pd(_mm_add_pd(zd1, mA5), xd1);
        zd0 = _mm_mul_pd(_mm_add_pd(zd0, mA6), xd0);
        zd1 = _mm_mul_pd(_mm_add_pd(zd1, mA6), xd1);
        zd0 = _mm_mul_pd(_mm_add_pd(zd0, mA7), xd0);
        zd1 = _mm_mul_pd(_mm_add_pd(zd1, mA7), xd1);

        yd0 = _mm_add_pd(yd0, zd0);
        yd1 = _mm_add_pd(yd1, zd1);

        _mm_storeu_pd(y + i, yd0);
        _mm_storeu_pd(y + i + 2, yd1);
    }
#endif
    for( ; i <= n - 4; i += 4 )
    {
        double xq;
        double x0, x1, x2, x3;
        double y0, y1, y2, y3;
        int h0, h1, h2, h3;

        h0 = X[i].i.lo;
        h1 = X[i + 1].i.lo;
        buf[0].i.lo = h0;
        buf[1].i.lo = h1;

        h0 = X[i].i.hi;
        h1 = X[i + 1].i.hi;
        buf[0].i.hi = (h0 & LOGTAB_MASK2) | (1023 << 20);
        buf[1].i.hi = (h1 & LOGTAB_MASK2) | (1023 << 20);

        y0 = (((h0 >> 20) & 0x7ff) - 1023) * ln_2;
        y1 = (((h1 >> 20) & 0x7ff) - 1023) * ln_2;

        h2 = X[i + 2].i.lo;
        h3 = X[i + 3].i.lo;
        buf[2].i.lo = h2;
        buf[3].i.lo = h3;

        h0 = (h0 >> (20 - LOGTAB_SCALE - 1)) & LOGTAB_MASK * 2;
        h1 = (h1 >> (20 - LOGTAB_SCALE - 1)) & LOGTAB_MASK * 2;

        y0 += icvLogTab[h0];
        y1 += icvLogTab[h1];

        h2 = X[i + 2].i.hi;
        h3 = X[i + 3].i.hi;

        x0 = LOGTAB_TRANSLATE( buf[0].d, h0 );
        x1 = LOGTAB_TRANSLATE( buf[1].d, h1 );

        buf[2].i.hi = (h2 & LOGTAB_MASK2) | (1023 << 20);
        buf[3].i.hi = (h3 & LOGTAB_MASK2) | (1023 << 20);

        y2 = (((h2 >> 20) & 0x7ff) - 1023) * ln_2;
        y3 = (((h3 >> 20) & 0x7ff) - 1023) * ln_2;

        h2 = (h2 >> (20 - LOGTAB_SCALE - 1)) & LOGTAB_MASK * 2;
        h3 = (h3 >> (20 - LOGTAB_SCALE - 1)) & LOGTAB_MASK * 2;

        y2 += icvLogTab[h2];
        y3 += icvLogTab[h3];

        x2 = LOGTAB_TRANSLATE( buf[2].d, h2 );
        x3 = LOGTAB_TRANSLATE( buf[3].d, h3 );

        y0 += LOGPOLY( x0, h0 == 510 );
        y1 += LOGPOLY( x1, h1 == 510 );

        y[i] = y0;
        y[i + 1] = y1;

        y2 += LOGPOLY( x2, h2 == 510 );
        y3 += LOGPOLY( x3, h3 == 510 );

        y[i + 2] = y2;
        y[i + 3] = y3;
    }

    for( ; i < n; i++ )
    {
        int h0 = X[i].i.hi;
        double xq;
        double x0, y0 = (((h0 >> 20) & 0x7ff) - 1023) * ln_2;

        buf[0].i.hi = (h0 & LOGTAB_MASK2) | (1023 << 20);
        buf[0].i.lo = X[i].i.lo;
        h0 = (h0 >> (20 - LOGTAB_SCALE - 1)) & LOGTAB_MASK * 2;

        y0 += icvLogTab[h0];
        x0 = LOGTAB_TRANSLATE( buf[0].d, h0 );
        y0 += LOGPOLY( x0, h0 == 510 );
        y[i] = y0;
    }
}

}}
