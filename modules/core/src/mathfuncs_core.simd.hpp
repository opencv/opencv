// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "mathfuncs.hpp"

namespace cv { namespace hal {

CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

// forward declarations
void cartToPolar32f(const float *X, const float *Y, float* mag, float *angle, int len, bool angleInDegrees);
void cartToPolar64f(const double *X, const double *Y, double* mag, double *angle, int len, bool angleInDegrees);
void fastAtan32f(const float *Y, const float *X, float *angle, int len, bool angleInDegrees);
void fastAtan64f(const double *Y, const double *X, double *angle, int len, bool angleInDegrees);
void fastAtan2(const float *Y, const float *X, float *angle, int len, bool angleInDegrees);
void magnitude32f(const float* x, const float* y, float* mag, int len);
void magnitude64f(const double* x, const double* y, double* mag, int len);
void invSqrt32f(const float* src, float* dst, int len);
void invSqrt64f(const double* src, double* dst, int len);
void sqrt32f(const float* src, float* dst, int len);
void sqrt64f(const double* src, double* dst, int len);
void exp32f(const float *src, float *dst, int n);
void exp64f(const double *src, double *dst, int n);
void log32f(const float *src, float *dst, int n);
void log64f(const double *src, double *dst, int n);
float fastAtan2(float y, float x);

#ifndef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

using namespace std;
using namespace cv;

namespace {

static const float atan2_p1 = 0.9997878412794807f*(float)(180/CV_PI);
static const float atan2_p3 = -0.3258083974640975f*(float)(180/CV_PI);
static const float atan2_p5 = 0.1555786518463281f*(float)(180/CV_PI);
static const float atan2_p7 = -0.04432655554792128f*(float)(180/CV_PI);

#ifdef __EMSCRIPTEN__
static inline float atan_f32(float y, float x)
{
    float a = atan2(y, x) * 180.0f / CV_PI;
    if (a < 0.0f)
        a += 360.0f;
    if (a >= 360.0f)
        a -= 360.0f;
    return a; // range [0; 360)
}
#else
static inline float atan_f32(float y, float x)
{
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
    return a;
}
#endif

#if CV_SIMD

struct v_atan_f32
{
    explicit v_atan_f32(const float& scale)
    {
        eps = vx_setall_f32((float)DBL_EPSILON);
        z = vx_setzero_f32();
        p7 = vx_setall_f32(atan2_p7);
        p5 = vx_setall_f32(atan2_p5);
        p3 = vx_setall_f32(atan2_p3);
        p1 = vx_setall_f32(atan2_p1);
        val90 = vx_setall_f32(90.f);
        val180 = vx_setall_f32(180.f);
        val360 = vx_setall_f32(360.f);
        s = vx_setall_f32(scale);
    }

    v_float32 compute(const v_float32& y, const v_float32& x)
    {
        v_float32 ax = v_abs(x);
        v_float32 ay = v_abs(y);
        v_float32 c = v_div(v_min(ax, ay), v_add(v_max(ax, ay), this->eps));
        v_float32 cc = v_mul(c, c);
        v_float32 a = v_mul(v_fma(v_fma(v_fma(cc, this->p7, this->p5), cc, this->p3), cc, this->p1), c);
        a = v_select(v_ge(ax, ay), a, v_sub(this->val90, a));
        a = v_select(v_lt(x, this->z), v_sub(this->val180, a), a);
        a = v_select(v_lt(y, this->z), v_sub(this->val360, a), a);
        return v_mul(a, this->s);
    }

    v_float32 eps;
    v_float32 z;
    v_float32 p7;
    v_float32 p5;
    v_float32 p3;
    v_float32 p1;
    v_float32 val90;
    v_float32 val180;
    v_float32 val360;
    v_float32 s;
};

#endif

} // anonymous::

static void cartToPolar32f_(const float *X, const float *Y, float *mag, float *angle, int len, bool angleInDegrees )
{
    float scale = angleInDegrees ? 1.f : (float)(CV_PI/180);
    int i = 0;
#if CV_SIMD
    const int VECSZ = VTraits<v_float32>::vlanes();
    v_atan_f32 v(scale);

    for( ; i < len; i += VECSZ*2 )
    {
        if( i + VECSZ*2 > len )
        {
            // if it's inplace operation, we cannot repeatedly process
            // the tail for the second time, so we have to use the
            // scalar code
            if( i == 0 || angle == X || angle == Y )
                break;
            i = len - VECSZ*2;
        }

        v_float32 x0 = vx_load(X + i);
        v_float32 y0 = vx_load(Y + i);
        v_float32 x1 = vx_load(X + i + VECSZ);
        v_float32 y1 = vx_load(Y + i + VECSZ);

        v_float32 m0 = v_sqrt(v_muladd(x0, x0, v_mul(y0, y0)));
        v_float32 m1 = v_sqrt(v_muladd(x1, x1, v_mul(y1, y1)));

        v_float32 r0 = v.compute(y0, x0);
        v_float32 r1 = v.compute(y1, x1);

        v_store(mag + i, m0);
        v_store(mag + i + VECSZ, m1);

        v_store(angle + i, r0);
        v_store(angle + i + VECSZ, r1);
    }
    vx_cleanup();
#endif

    for( ; i < len; i++ )
    {
        float x0 = X[i], y0 = Y[i];
        mag[i] = std::sqrt(x0*x0 + y0*y0);
        angle[i] = atan_f32(y0, x0)*scale;
    }
}

void cartToPolar32f(const float *X, const float *Y, float *mag, float *angle, int len, bool angleInDegrees )
{
    CV_INSTRUMENT_REGION();
    cartToPolar32f_(X, Y, mag, angle, len, angleInDegrees );
}

void cartToPolar64f(const double *X, const double *Y, double *mag, double *angle, int len, bool angleInDegrees)
{
    CV_INSTRUMENT_REGION();

    const int BLKSZ = 128;
    float ybuf[BLKSZ], xbuf[BLKSZ], mbuf[BLKSZ], abuf[BLKSZ];
    for( int i = 0; i < len; i += BLKSZ )
    {
        int j, blksz = std::min(BLKSZ, len - i);
        for( j = 0; j < blksz; j++ )
        {
            xbuf[j] = (float)X[i + j];
            ybuf[j] = (float)Y[i + j];
        }
        cartToPolar32f_(xbuf, ybuf, mbuf, abuf, blksz, angleInDegrees);
        for( j = 0; j < blksz; j++ )
            mag[i + j] = mbuf[j];
        for( j = 0; j < blksz; j++ )
            angle[i + j] = abuf[j];
    }
}

static void fastAtan32f_(const float *Y, const float *X, float *angle, int len, bool angleInDegrees )
{
    float scale = angleInDegrees ? 1.f : (float)(CV_PI/180);
    int i = 0;
#if CV_SIMD
    const int VECSZ = VTraits<v_float32>::vlanes();
    v_atan_f32 v(scale);

    for( ; i < len; i += VECSZ*2 )
    {
        if( i + VECSZ*2 > len )
        {
            // if it's inplace operation, we cannot repeatedly process
            // the tail for the second time, so we have to use the
            // scalar code
            if( i == 0 || angle == X || angle == Y )
                break;
            i = len - VECSZ*2;
        }

        v_float32 y0 = vx_load(Y + i);
        v_float32 x0 = vx_load(X + i);
        v_float32 y1 = vx_load(Y + i + VECSZ);
        v_float32 x1 = vx_load(X + i + VECSZ);

        v_float32 r0 = v.compute(y0, x0);
        v_float32 r1 = v.compute(y1, x1);

        v_store(angle + i, r0);
        v_store(angle + i + VECSZ, r1);
    }
    vx_cleanup();
#endif

    for( ; i < len; i++ )
        angle[i] = atan_f32(Y[i], X[i])*scale;
}

void fastAtan32f(const float *Y, const float *X, float *angle, int len, bool angleInDegrees )
{
    CV_INSTRUMENT_REGION();
    fastAtan32f_(Y, X, angle, len, angleInDegrees );
}

void fastAtan64f(const double *Y, const double *X, double *angle, int len, bool angleInDegrees)
{
    CV_INSTRUMENT_REGION();

    const int BLKSZ = 128;
    float ybuf[BLKSZ], xbuf[BLKSZ], abuf[BLKSZ];
    for( int i = 0; i < len; i += BLKSZ )
    {
        int j, blksz = std::min(BLKSZ, len - i);
        for( j = 0; j < blksz; j++ )
        {
            ybuf[j] = (float)Y[i + j];
            xbuf[j] = (float)X[i + j];
        }
        fastAtan32f_(ybuf, xbuf, abuf, blksz, angleInDegrees);
        for( j = 0; j < blksz; j++ )
            angle[i + j] = abuf[j];
    }
}

// deprecated
void fastAtan2(const float *Y, const float *X, float *angle, int len, bool angleInDegrees )
{
    CV_INSTRUMENT_REGION();
    fastAtan32f(Y, X, angle, len, angleInDegrees);
}

void magnitude32f(const float* x, const float* y, float* mag, int len)
{
    CV_INSTRUMENT_REGION();

    int i = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
    const int VECSZ = VTraits<v_float32>::vlanes();
    for( ; i < len; i += VECSZ*2 )
    {
        if( i + VECSZ*2 > len )
        {
            if( i == 0 || mag == x || mag == y )
                break;
            i = len - VECSZ*2;
        }
        v_float32 x0 = vx_load(x + i), x1 = vx_load(x + i + VECSZ);
        v_float32 y0 = vx_load(y + i), y1 = vx_load(y + i + VECSZ);
        x0 = v_sqrt(v_muladd(x0, x0, v_mul(y0, y0)));
        x1 = v_sqrt(v_muladd(x1, x1, v_mul(y1, y1)));
        v_store(mag + i, x0);
        v_store(mag + i + VECSZ, x1);
    }
    vx_cleanup();
#endif

    for( ; i < len; i++ )
    {
        float x0 = x[i], y0 = y[i];
        mag[i] = std::sqrt(x0*x0 + y0*y0);
    }
}

void magnitude64f(const double* x, const double* y, double* mag, int len)
{
    CV_INSTRUMENT_REGION();

    int i = 0;

#if (CV_SIMD_64F || CV_SIMD_SCALABLE_64F)
    const int VECSZ = VTraits<v_float64>::vlanes();
    for( ; i < len; i += VECSZ*2 )
    {
        if( i + VECSZ*2 > len )
        {
            if( i == 0 || mag == x || mag == y )
                break;
            i = len - VECSZ*2;
        }
        v_float64 x0 = vx_load(x + i), x1 = vx_load(x + i + VECSZ);
        v_float64 y0 = vx_load(y + i), y1 = vx_load(y + i + VECSZ);
        x0 = v_sqrt(v_muladd(x0, x0, v_mul(y0, y0)));
        x1 = v_sqrt(v_muladd(x1, x1, v_mul(y1, y1)));
        v_store(mag + i, x0);
        v_store(mag + i + VECSZ, x1);
    }
    vx_cleanup();
#endif

    for( ; i < len; i++ )
    {
        double x0 = x[i], y0 = y[i];
        mag[i] = std::sqrt(x0*x0 + y0*y0);
    }
}


void invSqrt32f(const float* src, float* dst, int len)
{
    CV_INSTRUMENT_REGION();

    int i = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
    const int VECSZ = VTraits<v_float32>::vlanes();
    for( ; i < len; i += VECSZ*2 )
    {
        if( i + VECSZ*2 > len )
        {
            if( i == 0 || src == dst )
                break;
            i = len - VECSZ*2;
        }
        v_float32 t0 = vx_load(src + i), t1 = vx_load(src + i + VECSZ);
        t0 = v_invsqrt(t0);
        t1 = v_invsqrt(t1);
        v_store(dst + i, t0); v_store(dst + i + VECSZ, t1);
    }
    vx_cleanup();
#endif

    for( ; i < len; i++ )
        dst[i] = 1/std::sqrt(src[i]);
}


void invSqrt64f(const double* src, double* dst, int len)
{
    CV_INSTRUMENT_REGION();
    int i = 0;

#if (CV_SIMD_64F || CV_SIMD_SCALABLE_64F)
    const int VECSZ = VTraits<v_float64>::vlanes();
    for ( ; i < len; i += VECSZ*2)
    {
        if( i + VECSZ*2 > len )
        {
            if( i == 0 || src == dst )
                break;
            i = len - VECSZ*2;
        }
        v_float64 t0 = vx_load(src + i), t1 = vx_load(src + i + VECSZ);
        t0 = v_invsqrt(t0);
        t1 = v_invsqrt(t1);
        v_store(dst + i, t0); v_store(dst + i + VECSZ, t1);
    }
#endif

    for( ; i < len; i++ )
        dst[i] = 1/std::sqrt(src[i]);
}


void sqrt32f(const float* src, float* dst, int len)
{
    CV_INSTRUMENT_REGION();

    int i = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
    const int VECSZ = VTraits<v_float32>::vlanes();
    for( ; i < len; i += VECSZ*2 )
    {
        if( i + VECSZ*2 > len )
        {
            if( i == 0 || src == dst )
                break;
            i = len - VECSZ*2;
        }
        v_float32 t0 = vx_load(src + i), t1 = vx_load(src + i + VECSZ);
        t0 = v_sqrt(t0);
        t1 = v_sqrt(t1);
        v_store(dst + i, t0); v_store(dst + i + VECSZ, t1);
    }
    vx_cleanup();
#endif

    for( ; i < len; i++ )
        dst[i] = std::sqrt(src[i]);
}


void sqrt64f(const double* src, double* dst, int len)
{
    CV_INSTRUMENT_REGION();

    int i = 0;

#if (CV_SIMD_64F || CV_SIMD_SCALABLE_64F)
    const int VECSZ = VTraits<v_float64>::vlanes();
    for( ; i < len; i += VECSZ*2 )
    {
        if( i + VECSZ*2 > len )
        {
            if( i == 0 || src == dst )
                break;
            i = len - VECSZ*2;
        }
        v_float64 t0 = vx_load(src + i), t1 = vx_load(src + i + VECSZ);
        t0 = v_sqrt(t0);
        t1 = v_sqrt(t1);
        v_store(dst + i, t0); v_store(dst + i + VECSZ, t1);
    }
    vx_cleanup();
#endif

    for( ; i < len; i++ )
        dst[i] = std::sqrt(src[i]);
}

// Workaround for ICE in MSVS 2015 update 3 (issue #7795)
// CV_AVX is not used here, because generated code is faster in non-AVX mode.
// (tested with disabled IPP on i5-6300U)
#if (defined _MSC_VER && _MSC_VER >= 1900) || defined(__EMSCRIPTEN__)
void exp32f(const float *src, float *dst, int n)
{
    CV_INSTRUMENT_REGION();

    for (int i = 0; i < n; i++)
    {
        dst[i] = std::exp(src[i]);
    }
}

void exp64f(const double *src, double *dst, int n)
{
    CV_INSTRUMENT_REGION();

    for (int i = 0; i < n; i++)
    {
        dst[i] = std::exp(src[i]);
    }
}

void log32f(const float *src, float *dst, int n)
{
    CV_INSTRUMENT_REGION();

    for (int i = 0; i < n; i++)
    {
        dst[i] = std::log(src[i]);
    }
}
void log64f(const double *src, double *dst, int n)
{
    CV_INSTRUMENT_REGION();

    for (int i = 0; i < n; i++)
    {
        dst[i] = std::log(src[i]);
    }
}
#else

////////////////////////////////////// EXP /////////////////////////////////////

#define EXPTAB_SCALE 6
#define EXPTAB_MASK  ((1 << EXPTAB_SCALE) - 1)

#define EXPPOLY_32F_A0 .9670371139572337719125840413672004409288e-2

// the code below uses _mm_cast* intrinsics, which are not available on VS2005
#if (defined _MSC_VER && _MSC_VER < 1500) || \
(!defined __APPLE__ && defined __GNUC__ && __GNUC__*100 + __GNUC_MINOR__ < 402)
#undef CV_SSE2
#define CV_SSE2 0
#endif

static const double exp_prescale = 1.4426950408889634073599246810019 * (1 << EXPTAB_SCALE);
static const double exp_postscale = 1./(1 << EXPTAB_SCALE);
static const double exp_max_val = 3000.*(1 << EXPTAB_SCALE); // log10(DBL_MAX) < 3000

void exp32f( const float *_x, float *y, int n )
{
    CV_INSTRUMENT_REGION();

    const float* const expTab_f = cv::details::getExpTab32f();

    const float
    A4 = (float)(1.000000000000002438532970795181890933776 / EXPPOLY_32F_A0),
    A3 = (float)(.6931471805521448196800669615864773144641 / EXPPOLY_32F_A0),
    A2 = (float)(.2402265109513301490103372422686535526573 / EXPPOLY_32F_A0),
    A1 = (float)(.5550339366753125211915322047004666939128e-1 / EXPPOLY_32F_A0);

    int i = 0;
    const Cv32suf* x = (const Cv32suf*)_x;
    float minval = (float)(-exp_max_val/exp_prescale);
    float maxval = (float)(exp_max_val/exp_prescale);
    float postscale = (float)exp_postscale;

#if (CV_SIMD || CV_SIMD_SCALABLE)
    const int VECSZ = VTraits<v_float32>::vlanes();
    const v_float32 vprescale = vx_setall_f32((float)exp_prescale);
    const v_float32 vpostscale = vx_setall_f32((float)exp_postscale);
    const v_float32 vminval = vx_setall_f32(minval);
    const v_float32 vmaxval = vx_setall_f32(maxval);

    const v_float32 vA1 = vx_setall_f32((float)A1);
    const v_float32 vA2 = vx_setall_f32((float)A2);
    const v_float32 vA3 = vx_setall_f32((float)A3);
    const v_float32 vA4 = vx_setall_f32((float)A4);

    const v_int32 vidxmask = vx_setall_s32(EXPTAB_MASK);
    bool y_aligned = (size_t)(void*)y % 32 == 0;

    for( ; i < n; i += VECSZ*2 )
    {
        if( i + VECSZ*2 > n )
        {
            if( i == 0 || _x == y )
                break;
            i = n - VECSZ*2;
            y_aligned = false;
        }

        v_float32 xf0 = vx_load(&x[i].f), xf1 = vx_load(&x[i + VECSZ].f);

        xf0 = v_min(v_max(xf0, vminval), vmaxval);
        xf1 = v_min(v_max(xf1, vminval), vmaxval);

        xf0 = v_mul(xf0, vprescale);
        xf1 = v_mul(xf1, vprescale);

        v_int32 xi0 = v_round(xf0);
        v_int32 xi1 = v_round(xf1);
        xf0 = v_mul(v_sub(xf0, v_cvt_f32(xi0)), vpostscale);
        xf1 = v_mul(v_sub(xf1, v_cvt_f32(xi1)), vpostscale);

        v_float32 yf0 = v_lut(expTab_f, v_and(xi0, vidxmask));
        v_float32 yf1 = v_lut(expTab_f, v_and(xi1, vidxmask));

        v_int32 v0 = vx_setzero_s32(), v127 = vx_setall_s32(127), v255 = vx_setall_s32(255);
        xi0 = v_min(v_max(v_add(v_shr<6>(xi0), v127), v0), v255);
        xi1 = v_min(v_max(v_add(v_shr<6>(xi1), v127), v0), v255);

        yf0 = v_mul(yf0, v_reinterpret_as_f32(v_shl<23>(xi0)));
        yf1 = v_mul(yf1, v_reinterpret_as_f32(v_shl<23>(xi1)));

        v_float32 zf0 = v_add(xf0, vA1);
        v_float32 zf1 = v_add(xf1, vA1);

        zf0 = v_fma(zf0, xf0, vA2);
        zf1 = v_fma(zf1, xf1, vA2);

        zf0 = v_fma(zf0, xf0, vA3);
        zf1 = v_fma(zf1, xf1, vA3);

        zf0 = v_fma(zf0, xf0, vA4);
        zf1 = v_fma(zf1, xf1, vA4);

        zf0 = v_mul(zf0, yf0);
        zf1 = v_mul(zf1, yf1);

        if( y_aligned )
        {
            v_store_aligned(y + i, zf0);
            v_store_aligned(y + i + VECSZ, zf1);
        }
        else
        {
            v_store(y + i, zf0);
            v_store(y + i + VECSZ, zf1);
        }
    }
    vx_cleanup();
#endif

    for( ; i < n; i++ )
    {
        float x0 = x[i].f;
        x0 = std::min(std::max(x0, minval), maxval);
        x0 *= (float)exp_prescale;
        Cv32suf buf;

        int xi = saturate_cast<int>(x0);
        x0 = (x0 - xi)*postscale;

        int t = (xi >> EXPTAB_SCALE) + 127;
        t = !(t & ~255) ? t : t < 0 ? 0 : 255;
        buf.i = t << 23;

        y[i] = buf.f * expTab_f[xi & EXPTAB_MASK] * ((((x0 + A1)*x0 + A2)*x0 + A3)*x0 + A4);
    }
}

void exp64f( const double *_x, double *y, int n )
{
    CV_INSTRUMENT_REGION();

    const double* const expTab = cv::details::getExpTab64f();

    const double
    A5 = .99999999999999999998285227504999 / EXPPOLY_32F_A0,
    A4 = .69314718055994546743029643825322 / EXPPOLY_32F_A0,
    A3 = .24022650695886477918181338054308 / EXPPOLY_32F_A0,
    A2 = .55504108793649567998466049042729e-1 / EXPPOLY_32F_A0,
    A1 = .96180973140732918010002372686186e-2 / EXPPOLY_32F_A0,
    A0 = .13369713757180123244806654839424e-2 / EXPPOLY_32F_A0;

    int i = 0;
    const Cv64suf* x = (const Cv64suf*)_x;
    double minval = (-exp_max_val/exp_prescale);
    double maxval = (exp_max_val/exp_prescale);

#if (CV_SIMD_64F || CV_SIMD_SCALABLE_64F)
    const int VECSZ = VTraits<v_float64>::vlanes();
    const v_float64 vprescale = vx_setall_f64(exp_prescale);
    const v_float64 vpostscale = vx_setall_f64(exp_postscale);
    const v_float64 vminval = vx_setall_f64(minval);
    const v_float64 vmaxval = vx_setall_f64(maxval);

    const v_float64 vA1 = vx_setall_f64(A1);
    const v_float64 vA2 = vx_setall_f64(A2);
    const v_float64 vA3 = vx_setall_f64(A3);
    const v_float64 vA4 = vx_setall_f64(A4);
    const v_float64 vA5 = vx_setall_f64(A5);

    const v_int32 vidxmask = vx_setall_s32(EXPTAB_MASK);
    bool y_aligned = (size_t)(void*)y % 32 == 0;

    for( ; i < n; i += VECSZ*2 )
    {
        if( i + VECSZ*2 > n )
        {
            if( i == 0 || _x == y )
                break;
            i = n - VECSZ*2;
            y_aligned = false;
        }

        v_float64 xf0 = vx_load(&x[i].f), xf1 = vx_load(&x[i + VECSZ].f);

        xf0 = v_min(v_max(xf0, vminval), vmaxval);
        xf1 = v_min(v_max(xf1, vminval), vmaxval);

        xf0 = v_mul(xf0, vprescale);
        xf1 = v_mul(xf1, vprescale);

        v_int32 xi0 = v_round(xf0);
        v_int32 xi1 = v_round(xf1);
        xf0 = v_mul(v_sub(xf0, v_cvt_f64(xi0)), vpostscale);
        xf1 = v_mul(v_sub(xf1, v_cvt_f64(xi1)), vpostscale);

        v_float64 yf0 = v_lut(expTab, v_and(xi0, vidxmask));
        v_float64 yf1 = v_lut(expTab, v_and(xi1, vidxmask));

        v_int32 v0 = vx_setzero_s32(), v1023 = vx_setall_s32(1023), v2047 = vx_setall_s32(2047);
        xi0 = v_min(v_max(v_add(v_shr<6>(xi0), v1023), v0), v2047);
        xi1 = v_min(v_max(v_add(v_shr<6>(xi1), v1023), v0), v2047);

        v_int64 xq0, xq1, dummy;
        v_expand(xi0, xq0, dummy);
        v_expand(xi1, xq1, dummy);

        yf0 = v_mul(yf0, v_reinterpret_as_f64(v_shl<52>(xq0)));
        yf1 = v_mul(yf1, v_reinterpret_as_f64(v_shl<52>(xq1)));

        v_float64 zf0 = v_add(xf0, vA1);
        v_float64 zf1 = v_add(xf1, vA1);

        zf0 = v_fma(zf0, xf0, vA2);
        zf1 = v_fma(zf1, xf1, vA2);

        zf0 = v_fma(zf0, xf0, vA3);
        zf1 = v_fma(zf1, xf1, vA3);

        zf0 = v_fma(zf0, xf0, vA4);
        zf1 = v_fma(zf1, xf1, vA4);

        zf0 = v_fma(zf0, xf0, vA5);
        zf1 = v_fma(zf1, xf1, vA5);

        zf0 = v_mul(zf0, yf0);
        zf1 = v_mul(zf1, yf1);

        if( y_aligned )
        {
            v_store_aligned(y + i, zf0);
            v_store_aligned(y + i + VECSZ, zf1);
        }
        else
        {
            v_store(y + i, zf0);
            v_store(y + i + VECSZ, zf1);
        }
    }
    vx_cleanup();
#endif

    for( ; i < n; i++ )
    {
        double x0 = x[i].f;
        x0 = std::min(std::max(x0, minval), maxval);
        x0 *= exp_prescale;
        Cv64suf buf;

        int xi = saturate_cast<int>(x0);
        x0 = (x0 - xi)*exp_postscale;

        int t = (xi >> EXPTAB_SCALE) + 1023;
        t = !(t & ~2047) ? t : t < 0 ? 0 : 2047;
        buf.i = (int64)t << 52;

        y[i] = buf.f * expTab[xi & EXPTAB_MASK] * (((((A0*x0 + A1)*x0 + A2)*x0 + A3)*x0 + A4)*x0 + A5);
    }
}

#undef EXPTAB_SCALE
#undef EXPTAB_MASK
#undef EXPPOLY_32F_A0

/////////////////////////////////////////// LOG ///////////////////////////////////////

#define LOGTAB_SCALE        8
#define LOGTAB_MASK         ((1 << LOGTAB_SCALE) - 1)

#define LOGTAB_TRANSLATE(tab, x, h) (((x) - 1.f)*tab[(h)+1])
static const double ln_2 = 0.69314718055994530941723212145818;

void log32f( const float *_x, float *y, int n )
{
    CV_INSTRUMENT_REGION();

    const float* const logTab_f = cv::details::getLogTab32f();

    const int LOGTAB_MASK2_32F = (1 << (23 - LOGTAB_SCALE)) - 1;
    const float
    A0 = 0.3333333333333333333333333f,
    A1 = -0.5f,
    A2 = 1.f;

    int i = 0;
    const int* x = (const int*)_x;

#if (CV_SIMD || CV_SIMD_SCALABLE)
    const int VECSZ = VTraits<v_float32>::vlanes();
    const v_float32 vln2 = vx_setall_f32((float)ln_2);
    const v_float32 v1 = vx_setall_f32(1.f);
    const v_float32 vshift = vx_setall_f32(-1.f/512);

    const v_float32 vA0 = vx_setall_f32(A0);
    const v_float32 vA1 = vx_setall_f32(A1);
    const v_float32 vA2 = vx_setall_f32(A2);

    for( ; i < n; i += VECSZ )
    {
        if( i + VECSZ > n )
        {
            if( i == 0 || _x == y )
                break;
            i = n - VECSZ;
        }

        v_int32 h0 = vx_load(x + i);
        v_int32 yi0 = v_sub(v_and(v_shr<23>(h0), vx_setall_s32(255)), vx_setall_s32(127));
        v_int32 xi0 = v_or(v_and(h0, vx_setall_s32(LOGTAB_MASK2_32F)), vx_setall_s32(127 << 23));

        h0 = v_and(v_shr<23 - 8 - 1>(h0), vx_setall_s32(((1 << 8) - 1) * 2));
        v_float32 yf0, xf0;

        v_lut_deinterleave(logTab_f, h0, yf0, xf0);

        yf0 = v_fma(v_cvt_f32(yi0), vln2, yf0);

        v_float32 delta = v_select(v_reinterpret_as_f32(v_eq(h0, vx_setall_s32(510))), vshift, vx_setall<float>(0));
        xf0 = v_fma((v_sub(v_reinterpret_as_f32(xi0), v1)), xf0, delta);

        v_float32 zf0 = v_fma(xf0, vA0, vA1);
        zf0 = v_fma(zf0, xf0, vA2);
        zf0 = v_fma(zf0, xf0, yf0);

        v_store(y + i, zf0);
    }
    vx_cleanup();
#endif

    for( ; i < n; i++ )
    {
        Cv32suf buf;
        int i0 = x[i];

        buf.i = (i0 & LOGTAB_MASK2_32F) | (127 << 23);
        int idx = (i0 >> (23 - LOGTAB_SCALE - 1)) & (LOGTAB_MASK*2);

        float y0 = (((i0 >> 23) & 0xff) - 127) * (float)ln_2 + logTab_f[idx];
        float x0 = (buf.f - 1.f)*logTab_f[idx + 1] + (idx == 510 ? -1.f/512 : 0.f);
        y[i] = ((A0*x0 + A1)*x0 + A2)*x0 + y0;
    }
}

void log64f( const double *x, double *y, int n )
{
    CV_INSTRUMENT_REGION();

    const double* const logTab = cv::details::getLogTab64f();

    const int64 LOGTAB_MASK2_64F = ((int64)1 << (52 - LOGTAB_SCALE)) - 1;
    const double
    A7 = 1.0,
    A6 = -0.5,
    A5 = 0.333333333333333314829616256247390992939472198486328125,
    A4 = -0.25,
    A3 = 0.2,
    A2 = -0.1666666666666666574148081281236954964697360992431640625,
    A1 = 0.1428571428571428769682682968777953647077083587646484375,
    A0 = -0.125;

    int i = 0;

#if (CV_SIMD_64F || CV_SIMD_SCALABLE_64F)
    const int VECSZ = VTraits<v_float64>::vlanes();
    const v_float64 vln2 = vx_setall_f64(ln_2);

    const v_float64
        vA0 = vx_setall_f64(A0), vA1 = vx_setall_f64(A1),
        vA2 = vx_setall_f64(A2), vA3 = vx_setall_f64(A3),
        vA4 = vx_setall_f64(A4), vA5 = vx_setall_f64(A5),
        vA6 = vx_setall_f64(A6), vA7 = vx_setall_f64(A7);

    for( ; i < n; i += VECSZ )
    {
        if( i + VECSZ > n )
        {
            if( i == 0 || x == y )
                break;
            i = n - VECSZ;
        }

        v_int64 h0 = vx_load((const int64*)x + i);
        v_int32 yi0 = v_pack(v_shr<52>(h0), vx_setzero_s64());
        yi0 = v_sub(v_and(yi0, vx_setall_s32(2047)), vx_setall_s32(1023));

        v_int64 xi0 = v_or(v_and(h0, vx_setall_s64(LOGTAB_MASK2_64F)), vx_setall_s64((int64)1023 << 52));
        h0 = v_shr<52 - LOGTAB_SCALE - 1>(h0);
        v_int32 idx = v_and(v_pack(h0, h0), vx_setall_s32(((1 << 8) - 1) * 2));

        v_float64 xf0, yf0;
        v_lut_deinterleave(logTab, idx, yf0, xf0);

        yf0 = v_fma(v_cvt_f64(yi0), vln2, yf0);
        v_float64 delta = v_mul(v_cvt_f64(v_eq(idx, vx_setall_s32(510))), vx_setall_f64(1. / 512));
        xf0 = v_fma(v_sub(v_reinterpret_as_f64(xi0), vx_setall_f64(1.)), xf0, delta);

        v_float64 xq = v_mul(xf0, xf0);
        v_float64 zf0 = v_fma(xq, vA0, vA2);
        v_float64 zf1 = v_fma(xq, vA1, vA3);
        zf0 = v_fma(zf0, xq, vA4);
        zf1 = v_fma(zf1, xq, vA5);
        zf0 = v_fma(zf0, xq, vA6);
        zf1 = v_fma(zf1, xq, vA7);
        zf1 = v_fma(zf1, xf0, yf0);
        zf0 = v_fma(zf0, xq, zf1);

        v_store(y + i, zf0);
    }
#endif

    for( ; i < n; i++ )
    {
        Cv64suf buf;
        int64 i0 = ((const int64*)x)[i];

        buf.i = (i0 & LOGTAB_MASK2_64F) | ((int64)1023 << 52);
        int idx = (int)(i0 >> (52 - LOGTAB_SCALE - 1)) & (LOGTAB_MASK*2);

        double y0 = (((int)(i0 >> 52) & 0x7ff) - 1023) * ln_2 + logTab[idx];
        double x0 = (buf.f - 1.)*logTab[idx + 1] + (idx == 510 ? -1./512 : 0.);

        double xq = x0*x0;
        y[i] = (((A0*xq + A2)*xq + A4)*xq + A6)*xq + (((A1*xq + A3)*xq + A5)*xq + A7)*x0 + y0;
    }
}

#endif // issue 7795

float fastAtan2( float y, float x )
{
    return atan_f32(y, x);
}

#endif // CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

CV_CPU_OPTIMIZATION_NAMESPACE_END

}} // namespace cv::hal
