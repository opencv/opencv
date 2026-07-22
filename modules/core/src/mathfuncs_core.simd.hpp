// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "mathfuncs.hpp"

namespace cv { namespace hal {

CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

// forward declarations
void cartToPolar32f(const float *X, const float *Y, float* mag, float *angle, int len, bool angleInDegrees);
void cartToPolar64f(const double *X, const double *Y, double* mag, double *angle, int len, bool angleInDegrees);
void polarToCart32f(const float *mag, const float *angle, float *X, float *Y, int len, bool angleInDegrees);
void polarToCart64f(const double *mag, const double *angle, double *X, double *Y, int len, bool angleInDegrees);
void fastAtan32f(const float *Y, const float *X, float *angle, int len, bool angleInDegrees);
void fastAtan64f(const double *Y, const double *X, double *angle, int len, bool angleInDegrees);
void fastAtan2(const float *Y, const float *X, float *angle, int len, bool angleInDegrees);
void magnitude32f(const float* x, const float* y, float* mag, int len);
void magnitude64f(const double* x, const double* y, double* mag, int len);
void invSqrt32f(const float* src, float* dst, int len);
void invSqrt64f(const double* src, double* dst, int len);
void sqrt32f(const float* src, float* dst, int len);
void sqrt64f(const double* src, double* dst, int len);
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

#if (CV_SIMD || CV_SIMD_SCALABLE)

v_float32 v_atan_f32(const v_float32& y, const v_float32& x)
{
    v_float32 eps = vx_setall_f32((float)DBL_EPSILON);
    v_float32 z = vx_setzero_f32();
    v_float32 p7 = vx_setall_f32(atan2_p7);
    v_float32 p5 = vx_setall_f32(atan2_p5);
    v_float32 p3 = vx_setall_f32(atan2_p3);
    v_float32 p1 = vx_setall_f32(atan2_p1);
    v_float32 val90 = vx_setall_f32(90.f);
    v_float32 val180 = vx_setall_f32(180.f);
    v_float32 val360 = vx_setall_f32(360.f);

    v_float32 ax = v_abs(x);
    v_float32 ay = v_abs(y);
    v_float32 c = v_div(v_min(ax, ay), v_add(v_max(ax, ay), eps));
    v_float32 cc = v_mul(c, c);
    v_float32 a = v_mul(v_fma(v_fma(v_fma(cc, p7, p5), cc, p3), cc, p1), c);
    a = v_select(v_ge(ax, ay), a, v_sub(val90, a));
    a = v_select(v_lt(x, z), v_sub(val180, a), a);
    a = v_select(v_lt(y, z), v_sub(val360, a), a);
    return a;
}

#endif

} // anonymous::

static void cartToPolar32f_(const float *X, const float *Y, float *mag, float *angle, int len, bool angleInDegrees )
{
    float scale = angleInDegrees ? 1.f : (float)(CV_PI/180);
    int i = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    const int VECSZ = VTraits<v_float32>::vlanes();
    v_float32 s = vx_setall_f32(scale);

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

        v_float32 r0 = v_mul(v_atan_f32(y0, x0), s);
        v_float32 r1 = v_mul(v_atan_f32(y1, x1), s);

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

namespace {

static inline void SinCos_32f(const float* mag, const float* angle, float* cosval, float* sinval, int len, int angle_in_degrees)
{
    const int N = 64;

    static const double sin_table[] =
    {
     0.00000000000000000000,     0.09801714032956060400,
     0.19509032201612825000,     0.29028467725446233000,
     0.38268343236508978000,     0.47139673682599764000,
     0.55557023301960218000,     0.63439328416364549000,
     0.70710678118654746000,     0.77301045336273699000,
     0.83146961230254524000,     0.88192126434835494000,
     0.92387953251128674000,     0.95694033573220894000,
     0.98078528040323043000,     0.99518472667219682000,
     1.00000000000000000000,     0.99518472667219693000,
     0.98078528040323043000,     0.95694033573220894000,
     0.92387953251128674000,     0.88192126434835505000,
     0.83146961230254546000,     0.77301045336273710000,
     0.70710678118654757000,     0.63439328416364549000,
     0.55557023301960218000,     0.47139673682599786000,
     0.38268343236508989000,     0.29028467725446239000,
     0.19509032201612861000,     0.09801714032956082600,
     0.00000000000000012246,    -0.09801714032956059000,
    -0.19509032201612836000,    -0.29028467725446211000,
    -0.38268343236508967000,    -0.47139673682599764000,
    -0.55557023301960196000,    -0.63439328416364527000,
    -0.70710678118654746000,    -0.77301045336273666000,
    -0.83146961230254524000,    -0.88192126434835494000,
    -0.92387953251128652000,    -0.95694033573220882000,
    -0.98078528040323032000,    -0.99518472667219693000,
    -1.00000000000000000000,    -0.99518472667219693000,
    -0.98078528040323043000,    -0.95694033573220894000,
    -0.92387953251128663000,    -0.88192126434835505000,
    -0.83146961230254546000,    -0.77301045336273688000,
    -0.70710678118654768000,    -0.63439328416364593000,
    -0.55557023301960218000,    -0.47139673682599792000,
    -0.38268343236509039000,    -0.29028467725446250000,
    -0.19509032201612872000,    -0.09801714032956050600,
    };

    static const double k2 = (2*CV_PI)/N;

    static const double sin_a0 = -0.166630293345647*k2*k2*k2;
    static const double sin_a2 = k2;

    static const double cos_a0 = -0.499818138450326*k2*k2;
    /*static const double cos_a2 =  1;*/

    double k1;
    int i = 0;

    if( !angle_in_degrees )
        k1 = N/(2*CV_PI);
    else
        k1 = N/360.;

#if (CV_SIMD || CV_SIMD_SCALABLE)
    const int VECSZ = VTraits<v_float32>::vlanes();
    const v_float32 scale = vx_setall_f32(angle_in_degrees ? (float)CV_PI / 180.f : 1.f);

    for( ; i < len; i += VECSZ*2 )
    {
        if( i + VECSZ*2 > len )
        {
            // if it's inplace operation, we cannot repeatedly process
            // the tail for the second time, so we have to use the
            // scalar code
            if( i == 0 || angle == cosval || angle == sinval || mag == cosval || mag == sinval )
                break;
            i = len - VECSZ*2;
        }

        v_float32 r0 = v_mul(vx_load(angle + i), scale);
        v_float32 r1 = v_mul(vx_load(angle + i + VECSZ), scale);

        v_float32 c0, c1, s0, s1;
        v_sincos(r0, s0, c0);
        v_sincos(r1, s1, c1);

        if( mag )
        {
            v_float32 m0 = vx_load(mag + i);
            v_float32 m1 = vx_load(mag + i + VECSZ);
            c0 = v_mul(c0, m0);
            c1 = v_mul(c1, m1);
            s0 = v_mul(s0, m0);
            s1 = v_mul(s1, m1);
        }

        v_store(cosval + i, c0);
        v_store(cosval + i + VECSZ, c1);

        v_store(sinval + i, s0);
        v_store(sinval + i + VECSZ, s1);
    }
    vx_cleanup();
#endif

    for( ; i < len; i++ )
    {
        double t = angle[i]*k1;
        int it = cvRound(t);
        t -= it;
        int sin_idx = it & (N - 1);
        int cos_idx = (N/4 - sin_idx) & (N - 1);

        double sin_b = (sin_a0*t*t + sin_a2)*t;
        double cos_b = cos_a0*t*t + 1;

        double sin_a = sin_table[sin_idx];
        double cos_a = sin_table[cos_idx];

        double sin_val = sin_a*cos_b + cos_a*sin_b;
        double cos_val = cos_a*cos_b - sin_a*sin_b;

        if (mag)
        {
            double mag_val = mag[i];
            sin_val *= mag_val;
            cos_val *= mag_val;
        }

        sinval[i] = (float)sin_val;
        cosval[i] = (float)cos_val;
    }
}

} // anonymous::

void polarToCart32f(const float *mag, const float *angle, float *X, float *Y, int len, bool angleInDegrees)
{
    CV_INSTRUMENT_REGION();
    SinCos_32f(mag, angle, X, Y, len, angleInDegrees);
}

void polarToCart64f(const double *mag, const double *angle, double *X, double *Y, int len, bool angleInDegrees)
{
    CV_INSTRUMENT_REGION();

    const int BLKSZ = 128;
    float ybuf[BLKSZ], xbuf[BLKSZ], _mbuf[BLKSZ], abuf[BLKSZ];
    float* mbuf = mag ? _mbuf : nullptr;
    for( int i = 0; i < len; i += BLKSZ )
    {
        int j, blksz = std::min(BLKSZ, len - i);
        for( j = 0; j < blksz; j++ )
        {
            if (mbuf)
                mbuf[j] = (float)mag[i + j];
            abuf[j] = (float)angle[i + j];
        }
        SinCos_32f(mbuf, abuf, xbuf, ybuf, blksz, angleInDegrees);
        for( j = 0; j < blksz; j++ )
            X[i + j] = xbuf[j];
        for( j = 0; j < blksz; j++ )
            Y[i + j] = ybuf[j];
    }
}

static void fastAtan32f_(const float *Y, const float *X, float *angle, int len, bool angleInDegrees )
{
    float scale = angleInDegrees ? 1.f : (float)(CV_PI/180);
    int i = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    const int VECSZ = VTraits<v_float32>::vlanes();
    v_float32 s = vx_setall_f32(scale);

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

        v_float32 r0 = v_mul(v_atan_f32(y0, x0), s);
        v_float32 r1 = v_mul(v_atan_f32(y1, x1), s);

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
#if (defined _MSC_VER && _MSC_VER >= 1900 && (defined(_M_IX86) || defined(_M_X64))) || defined(__EMSCRIPTEN__)
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


#endif // issue 7795

float fastAtan2( float y, float x )
{
    return atan_f32(y, x);
}

#endif // CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

CV_CPU_OPTIMIZATION_NAMESPACE_END

}} // namespace cv::hal
